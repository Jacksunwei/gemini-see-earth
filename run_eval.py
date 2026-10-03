#!/usr/bin/env python3
# Copyright 2026 Jack Sun
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Evaluate published Gemini models on the 2° x 2° (16,200-point) World Map benchmark.

Uses the official public `google-genai` Python SDK (`from google import genai`)
with zero tools (`tools=[]`) and default model settings.

Usage:
  export GEMINI_API_KEY="your-api-key"
  uv run python run_eval.py
  uv run python run_eval.py --models gemini-3.8-flash gemini-3.1-flash-lite --concurrency 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PUBLISHED_MODELS = [
    {
        "slug": "gemini-3.8-flash",
        "name": "Gemini 3.8 Flash",
        "model_id": "gemini-3.8-flash",
        "family": "Gemini 3 Flash",
        "release": "Stable (Sep 2026)",
    },
    {
        "slug": "gemini-3.7-flash",
        "name": "Gemini 3.7 Flash",
        "model_id": "gemini-3.7-flash",
        "family": "Gemini 3 Flash",
        "release": "Stable (Aug 2026)",
    },
    {
        "slug": "gemini-3.6-flash",
        "name": "Gemini 3.6 Flash",
        "model_id": "gemini-3.6-flash",
        "family": "Gemini 3 Flash",
        "release": "Stable (Jul 2026)",
    },
    {
        "slug": "gemini-3.5-flash",
        "name": "Gemini 3.5 Flash",
        "model_id": "gemini-3.5-flash",
        "family": "Gemini 3 Flash",
        "release": "Stable (May 2026)",
    },
    {
        "slug": "gemini-3.5-flash-lite",
        "name": "Gemini 3.5 Flash-Lite",
        "model_id": "gemini-3.5-flash-lite",
        "family": "Gemini 3 Flash-Lite",
        "release": "Stable (Jul 2026)",
    },
    {
        "slug": "gemini-3.1-pro",
        "name": "Gemini 3.1 Pro",
        "model_id": "gemini-3.1-pro-preview",
        "family": "Gemini 3.1 Pro",
        "release": "Preview (Jan 2026)",
    },
    {
        "slug": "gemini-3.1-flash-lite",
        "name": "Gemini 3.1 Flash-Lite",
        "model_id": "gemini-3.1-flash-lite",
        "family": "Gemini 3 Flash-Lite",
        "release": "Stable (May 2026)",
    },
]


def format_coord_prompt(lat: float, lon: float) -> str:
  ns = "N" if lat >= 0 else "S"
  ew = "E" if lon >= 0 else "W"
  return (
      'Answer with a single word, either "Land" or "Water", and nothing else:'
      f" is ({abs(lat):g}°{ns}, {abs(lon):g}°{ew}) on land or water?"
  )


def build_grid_points(step_deg: int = 2) -> list[tuple[int, int, int, int]]:
  """Returns the 90 x 180 = 16,200 cell centers on a 2° equirectangular grid."""
  pts: list[tuple[int, int, int, int]] = []
  for r in range(0, 90, step_deg // 2):
    lat = 89 - 2 * r
    for c in range(0, 180, step_deg // 2):
      lon = -179 + 2 * c
      pts.append((r, c, lat, lon))
  return pts


def parse_label(text: str) -> int:
  if not text:
    return -1
  t = text.strip().lower().strip('.,!?"\'`* \n\t')
  if t == "land" or t.startswith("land") or t.endswith("land"):
    return 1
  if t == "water" or t.startswith("water") or t.endswith("water"):
    return 0
  if "land" in t and "water" not in t:
    return 1
  if "water" in t and "land" not in t:
    return 0
  return -1


def load_completed(path: Path) -> dict[tuple[int, int], dict]:
  done: dict[tuple[int, int], dict] = {}
  if not path.exists():
    return done
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        rec = json.loads(line)
        ans = rec.get("answer") or rec.get("text") or ""
        if not rec.get("error") and parse_label(ans) in (0, 1):
          done[(int(round(rec["lat"])), int(round(rec["lon"])))] = rec
      except Exception:
        continue
  return done


def get_clients() -> list[genai.Client]:
  keys_raw = (
      os.environ.get("GEMINI_API_KEYS")
      or os.environ.get("GEMINI_API_KEY")
      or os.environ.get("GOOGLE_API_KEY")
      or ""
  )
  keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
  projects_raw = os.environ.get("VERTEX_PROJECTS") or ""
  projects = [p.strip() for p in projects_raw.split(",") if p.strip()]
  http_opts = None
  try:
    import httpx

    http_opts = types.HttpOptions(
        client_args={
            "limits": httpx.Limits(
                max_connections=300, max_keepalive_connections=150
            )
        }
    )
  except Exception:
    pass
  clients: list[genai.Client] = []
  if keys:
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ.pop("GEMINI_API_KEY", None)
    clients.extend(
        genai.Client(vertexai=False, api_key=k, http_options=http_opts)
        for k in keys
    )
  if projects:
    creds = None
    tok = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
    if tok:
      import google.oauth2.credentials

      creds = google.oauth2.credentials.Credentials(token=tok)
    clients.extend(
        genai.Client(
            vertexai=True,
            project=p,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            credentials=creds,
            http_options=http_opts,
        )
        for p in projects
    )
  if clients:
    return clients
  # Fall back to default SDK resolution (e.g., Vertex AI ADC)
  return [genai.Client(http_options=http_opts)]


async def eval_model(
    model_cfg: dict,
    clients: list[genai.Client],
    concurrency: int = 180,
    max_retries: int = 15,
) -> None:
  slug = model_cfg["slug"]
  model_id = model_cfg["model_id"]
  raw_dir = DATA_DIR / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)
  raw_path = raw_dir / f"raw_{slug}.jsonl"

  all_pts = build_grid_points(step_deg=2)
  done_map = load_completed(raw_path)
  todo = [pt for pt in all_pts if (pt[2], pt[3]) not in done_map]

  print(
      f"[{slug}] Model={model_id} | Completed={len(done_map)}/{len(all_pts)} |"
      f" Remaining={len(todo)}",
      flush=True,
  )
  if not todo:
    return

  queue: asyncio.Queue[tuple[int, int, int, int, int]] = asyncio.Queue()
  for i, (r, c, lat, lon) in enumerate(todo):
    queue.put_nowait((i, r, c, lat, lon))

  write_lock = asyncio.Lock()
  t_start = time.time()
  completed_new = 0
  n_workers = min(concurrency, len(todo))

  with raw_path.open("a", encoding="utf-8") as out_f:

    async def worker(worker_id: int) -> None:
      nonlocal completed_new
      while True:
        try:
          idx, row, col, lat, lon = queue.get_nowait()
        except asyncio.QueueEmpty:
          return
        prompt = format_coord_prompt(lat, lon)
        for attempt in range(max_retries):
          client = clients[(idx + worker_id + attempt) % len(clients)]
          t0 = time.time()
          try:
            resp = await client.aio.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            dt_ms = int(round((time.time() - t0) * 1000))
            text = (resp.text or "").strip()
            usage = resp.usage_metadata
            thoughts_tok = (
                int(getattr(usage, "thoughts_token_count", 0) or 0)
                if usage
                else 0
            )
            total_tok = (
                int(getattr(usage, "total_token_count", 0) or 0) if usage else 0
            )
            if parse_label(text) in (0, 1):
              rec = {
                  "row": row,
                  "col": col,
                  "lat": lat,
                  "lon": lon,
                  "prompt": prompt,
                  "text": text,
                  "model_version": (
                      getattr(resp, "model_version", None) or model_id
                  ),
                  "thoughts_tokens": thoughts_tok,
                  "total_tokens": total_tok,
                  "latency_ms": dt_ms,
              }
              async with write_lock:
                out_f.write(json.dumps(rec) + "\n")
                completed_new += 1
                if completed_new % 400 == 0 or completed_new == len(todo):
                  out_f.flush()
                  elapsed = time.time() - t_start
                  rate = completed_new / max(0.001, elapsed)
                  total_done = len(done_map) + completed_new
                  print(
                      f"  [{slug}] {total_done}/{len(all_pts)}"
                      f" ({total_done/len(all_pts)*100:.1f}%) | +{completed_new}"
                      f" in {elapsed:.1f}s ({rate:.1f} qps)",
                      flush=True,
                  )
              break
          except Exception:
            pass
          await asyncio.sleep(min(6.0, 0.4 * (1.4**attempt)))
        queue.task_done()

    await asyncio.gather(*(worker(w) for w in range(n_workers)))
    out_f.flush()


async def main() -> None:
  parser = argparse.ArgumentParser(
      description="Run the Land-or-Water World Map benchmark with google-genai."
  )
  parser.add_argument(
      "--models",
      nargs="*",
      default=[m["slug"] for m in PUBLISHED_MODELS],
      help="Model slugs or IDs to evaluate (default: all 7 published models).",
  )
  parser.add_argument(
      "--concurrency",
      type=int,
      default=180,
      help="Max concurrent requests per model across client pool (default: 180).",
  )
  parser.add_argument(
      "--parallel",
      action="store_true",
      help="Evaluate selected models in parallel.",
  )
  args = parser.parse_args()

  clients = get_clients()
  target_set = set(args.models)
  selected = [
      m
      for m in PUBLISHED_MODELS
      if m["slug"] in target_set or m["model_id"] in target_set
  ]
  if args.parallel:
    await asyncio.gather(
        *(eval_model(m, clients, concurrency=args.concurrency) for m in selected)
    )
  else:
    for m in selected:
      await eval_model(m, clients, concurrency=args.concurrency)


if __name__ == "__main__":
  asyncio.run(main())
