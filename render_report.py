#!/usr/bin/env python3
# Copyright 2026 Jack Sun
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Generate the Markdown report (README.md), standalone SVG/PNG visualizations, and enriched CSV/JSONL datasets."""

from __future__ import annotations

import html
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ASSETS_DIR = BASE_DIR / "assets"
MAPS_DIR = ASSETS_DIR / "maps"

# Publicly published Gemini models on ai.google.dev, ordered by recency descending
MODELS = [
    {
        "slug": "gemini-3.8-flash",
        "name": "Gemini 3.8 Flash",
        "model_id": "gemini-3.8-flash",
        "release": "Stable · Sep 2026",
        "family": "Gemini 3 Flash",
    },
    {
        "slug": "gemini-3.7-flash",
        "name": "Gemini 3.7 Flash",
        "model_id": "gemini-3.7-flash",
        "release": "Stable · Aug 2026",
        "family": "Gemini 3 Flash",
    },
    {
        "slug": "gemini-3.6-flash",
        "name": "Gemini 3.6 Flash",
        "model_id": "gemini-3.6-flash",
        "release": "Stable · Jul 2026",
        "family": "Gemini 3 Flash",
    },
    {
        "slug": "gemini-3.5-flash",
        "name": "Gemini 3.5 Flash",
        "model_id": "gemini-3.5-flash",
        "release": "Stable · May 2026",
        "family": "Gemini 3 Flash",
    },
    {
        "slug": "gemini-3.5-flash-lite",
        "name": "Gemini 3.5 Flash-Lite",
        "model_id": "gemini-3.5-flash-lite",
        "release": "Stable · Jul 2026",
        "family": "Gemini 3 Flash-Lite",
    },
    {
        "slug": "gemini-3.1-pro",
        "name": "Gemini 3.1 Pro",
        "model_id": "gemini-3.1-pro-preview",
        "release": "Preview · Jan 2026",
        "family": "Gemini 3.1 Pro",
    },
    {
        "slug": "gemini-3.1-flash-lite",
        "name": "Gemini 3.1 Flash-Lite",
        "model_id": "gemini-3.1-flash-lite",
        "release": "Stable · May 2026",
        "family": "Gemini 3 Flash-Lite",
    },
]

CLAUDE_REFS = [
    {"name": "Claude Opus 5.5 (thinks)", "acc": 99.0, "family": "Claude Ref"},
    {"name": "Claude Fable 5.1 (thinks)", "acc": 98.2, "family": "Claude Ref"},
    {"name": "Claude Fable 5 (thinks)", "acc": 97.8, "family": "Claude Ref"},
    {
        "name": "Claude Fable 5.1 (no thinking)",
        "acc": 96.5,
        "family": "Claude Ref",
    },
    {
        "name": "Claude Opus 5.5 (no thinking)",
        "acc": 96.4,
        "family": "Claude Ref",
    },
    {"name": "Claude Opus 5", "acc": 92.5, "family": "Claude Ref"},
    {"name": "Claude Opus 4.8", "acc": 91.7, "family": "Claude Ref"},
    {"name": "Claude Sonnet 5", "acc": 91.5, "family": "Claude Ref"},
    {"name": "Claude Opus 4.6 / 4.7", "acc": 91.2, "family": "Claude Ref"},
    {"name": "Claude Sonnet 4.6", "acc": 89.1, "family": "Claude Ref"},
    {"name": "Claude Opus 4.5", "acc": 82.6, "family": "Claude Ref"},
    {"name": "Claude Haiku 4.5", "acc": 81.3, "family": "Claude Ref"},
    {"name": "Claude Sonnet 4.5", "acc": 60.5, "family": "Claude Ref"},
]


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


def hex_to_rgb(h: str) -> tuple[int, int, int]:
  h = h.lstrip("#")
  return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def val_to_viridis_hex(t: float) -> str:
  t = max(0.0, min(1.0, float(t)))
  stops = [
      (0.0, (11, 16, 33)),
      (0.33, (124, 58, 237)),
      (0.66, (249, 115, 22)),
      (1.0, (254, 240, 138)),
  ]
  for i in range(len(stops) - 1):
    t0, c0 = stops[i]
    t1, c1 = stops[i + 1]
    if t <= t1:
      u = (t - t0) / (t1 - t0 + 1e-9)
      r_c = int(round(c0[0] + u * (c1[0] - c0[0])))
      g_c = int(round(c0[1] + u * (c1[1] - c0[1])))
      b_c = int(round(c0[2] + u * (c1[2] - c0[2])))
      return f"#{r_c:02x}{g_c:02x}{b_c:02x}"
  return "#fef08a"


def build_graticule_svg(mode: str = "bw", show: bool = True) -> str:
  if not show:
    return ""
  stroke = (
      "#f59e0b"
      if mode == "bw"
      else ("#facc15" if mode == "diff" else "#38bdf8")
  )
  return (
      '<g shape-rendering="geometricPrecision">'
      f'<line x1="0" y1="45" x2="180" y2="45" stroke="{stroke}"'
      ' stroke-width="1.2" stroke-dasharray="4,3"'
      ' vector-effect="non-scaling-stroke" opacity="0.65"/>'
      f'<line x1="90" y1="0" x2="90" y2="90" stroke="{stroke}"'
      ' stroke-width="1.2" stroke-dasharray="4,3"'
      ' vector-effect="non-scaling-stroke" opacity="0.65"/>'
      "</g>"
  )


def build_map_rects(
    grid: np.ndarray,
    mode: str = "bw",
    gt: np.ndarray | None = None,
    show_graticule: bool = True,
) -> str:
  rects = []
  bg_color = "#050505" if mode == "bw" else "#090d16"
  rects.append(f'<rect width="180" height="90" fill="{bg_color}"/>')

  for r in range(90):
    c = 0
    while c < 180:
      val = int(grid[r, c])
      gval = int(gt[r, c]) if gt is not None else 0
      if mode == "bw":
        if val == 0:
          color = bg_color
        elif val == 1:
          color = "#ffffff"
        elif val == -1:
          color = "#f97316"
        else:
          color = "#1e293b"
      else:
        if val == -2:
          color = "#1e293b"
        elif val == -1:
          color = "#f59e0b"
        elif val == 1 and gval == 1:
          color = "#f1f5f9"
        elif val == 0 and gval == 0:
          color = bg_color
        elif val == 1 and gval == 0:
          color = "#ef4444"
        elif val == 0 and gval == 1:
          color = "#38bdf8"
        else:
          color = bg_color
      c_end = c + 1
      while c_end < 180:
        v2 = int(grid[r, c_end])
        g2 = int(gt[r, c_end]) if gt is not None else 0
        if mode == "bw":
          same = v2 == val
        else:
          same = (v2 == val) and (
              g2 == gval or (val == 0 and gval == 0 and v2 == 0 and g2 == 0)
          )
        if not same:
          break
        c_end += 1
      if color != bg_color:
        w = c_end - c
        rects.append(
            f'<rect x="{c}" y="{r}" width="{w}" height="1" fill="{color}"/>'
        )
      c = c_end

  rects.append(build_graticule_svg(mode, show=show_graticule))
  return "".join(rects)


def build_heatmap_rects(
    values_grid: np.ndarray,
    mask_grid: np.ndarray,
    max_val: float = 800.0,
    fallback_grid: np.ndarray | None = None,
    show_graticule: bool = True,
) -> str:
  rects = ['<rect width="180" height="90" fill="#090d16"/>']

  if float(np.max(values_grid)) <= 0.0:
    if fallback_grid is not None:
      for r in range(90):
        for c in range(180):
          if int(fallback_grid[r, c]) == 1:
            rects.append(
                f'<rect x="{c}" y="{r}" width="1" height="1" fill="#1e293b"/>'
            )
    rects.append(build_graticule_svg("tt", show=show_graticule))
    rects.append(
        '<rect x="32" y="34" width="116" height="22" rx="4" fill="#0f172a"'
        ' fill-opacity="0.88" stroke="#475569" stroke-width="0.5"/>'
    )
    rects.append(
        '<text x="90" y="44" text-anchor="middle" fill="#e2e8f0"'
        ' font-family="system-ui, -apple-system, sans-serif" font-size="5.2"'
        ' font-weight="700">0 Thinking Tokens</text>'
    )
    rects.append(
        '<text x="90" y="51" text-anchor="middle" fill="#94a3b8"'
        ' font-family="system-ui, -apple-system, sans-serif"'
        ' font-size="3.8">(Non-thinking model by default)</text>'
    )
    return "".join(rects)

  for r in range(90):
    for c in range(180):
      if not mask_grid[r, c]:
        continue
      v = float(values_grid[r, c])
      norm = v / max_val if max_val > 0 else 0.0
      col = val_to_viridis_hex(norm)
      rects.append(
          f'<rect x="{c}" y="{r}" width="1" height="1" fill="{col}"/>'
      )
  rects.append(build_graticule_svg("tt", show=show_graticule))
  return "".join(rects)


def wrap_single_map_svg(inner_rects: str) -> str:
  return (
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 90"'
      ' width="1080" height="540" shape-rendering="crispEdges">'
      + inner_rects
      + "</svg>\n"
  )


def save_map_png(
    path: Path,
    grid: np.ndarray,
    mode: str = "bw",
    gt: np.ndarray | None = None,
    values_grid: np.ndarray | None = None,
    max_val: float = 800.0,
    scale: int = 6,
) -> None:
  img = np.zeros((90, 180, 3), dtype=np.uint8)
  bg = (5, 5, 5) if mode == "bw" else (9, 13, 22)
  img[:, :] = bg

  if mode in ("bw", "diff"):
    for r in range(90):
      for c in range(180):
        val = int(grid[r, c])
        gval = int(gt[r, c]) if gt is not None else 0
        if mode == "bw":
          if val == 1:
            img[r, c] = (255, 255, 255)
          elif val == -1:
            img[r, c] = (249, 115, 22)
        else:
          if val == 1 and gval == 1:
            img[r, c] = (241, 245, 249)
          elif val == 1 and gval == 0:
            img[r, c] = (239, 68, 68)
          elif val == 0 and gval == 1:
            img[r, c] = (56, 189, 248)
  elif mode == "tt" and values_grid is not None:
    if float(np.max(values_grid)) <= 0.0:
      for r in range(90):
        for c in range(180):
          if int(grid[r, c]) == 1:
            img[r, c] = (30, 41, 59)
    else:
      for r in range(90):
        for c in range(180):
          v = float(values_grid[r, c])
          norm = v / max_val if max_val > 0 else 0.0
          img[r, c] = hex_to_rgb(val_to_viridis_hex(norm))

  pil_img = Image.fromarray(img, mode="RGB").resize(
      (180 * scale, 90 * scale), resample=Image.Resampling.NEAREST
  )
  pil_img.save(path)


def build_composite_grid_svg(
    mode: str,
    gt_rects: str,
    model_stats: list[dict],
) -> str:
  """Builds a standalone, publication-ready 2x4 composite SVG figure (1640 x 696)."""
  width, height = 1640, 696
  font = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
  mono = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"

  if mode == "bw":
    title = "How Blind Geminis See the Earth — Classic Land / Water Map"
    legend_items = [
        ("#ffffff", "#cbd5e1", "Land"),
        ("#050505", "#050505", "Water"),
        ("#f59e0b", "#f59e0b", "0° Equator / Prime Meridian"),
    ]
  elif mode == "diff":
    title = (
        "How Blind Geminis See the Earth — Error Diagnostic Map (FP vs. FN)"
    )
    legend_items = [
        ("#f1f5f9", "#cbd5e1", "True Land (TP)"),
        ("#090d16", "#090d16", "True Water (TN)"),
        ("#ef4444", "#ef4444", "False Land / Hallucinated (FP)"),
        ("#38bdf8", "#38bdf8", "Missed Land / Submerged (FN)"),
    ]
  else:
    title = (
        "How Blind Geminis See the Earth — Dynamic Thinking Effort Heatmap"
    )
    legend_items = [
        ("#0b1021", "#0b1021", "Low (~20 tok)"),
        ("#7c3aed", "#7c3aed", "Medium (~250 tok)"),
        ("#f97316", "#f97316", "High (~500 tok)"),
        ("#fef08a", "#ca8a04", "Coastline Reasoning (>700 tok)"),
    ]

  parts = [
      f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"'
      f' width="{width}" height="{height}">',
      f'<rect width="{width}" height="{height}" rx="14" fill="#f8fafc"'
      ' stroke="#e2e8f0" stroke-width="2"/>',
      # Header Title & Subtitle
      f'<text x="28" y="38" font-family="{font}" font-size="22"'
      f' font-weight="800" fill="#0f172a">{html.escape(title)}</text>',
      f'<text x="28" y="62" font-family="{font}" font-size="13"'
      ' fill="#475569">By Jack Sun (@Jacksunwei) · 2° × 2° global grid (16,200'
      " coordinates/model, 113,400 queries) via public google-genai SDK"
      " (tools=[], default settings)</text>",
  ]

  # Top-right Legend Box
  lx = width - 28
  legend_svg = []
  cur_x = lx
  for fill_c, stroke_c, lbl in reversed(legend_items):
    text_w = len(lbl) * 7.2 + 28
    cur_x -= text_w
    legend_svg.append(
        f'<rect x="{cur_x:.1f}" y="29" width="12" height="12" rx="3"'
        f' fill="{fill_c}" stroke="{stroke_c}" stroke-width="1"/>'
        f'<text x="{cur_x + 17:.1f}" y="39.5" font-family="{font}"'
        f' font-size="12.5" font-weight="600" fill="#334155">'
        f"{html.escape(lbl)}</text>"
    )
    cur_x -= 14
  box_x = cur_x + 2
  box_w = lx - box_x + 12
  parts.append(
      f'<rect x="{box_x:.1f}" y="19" width="{box_w:.1f}" height="32" rx="8"'
      ' fill="#ffffff" stroke="#e2e8f0" stroke-width="1"/>'
  )
  parts.extend(reversed(legend_svg))

  # 8 Cards in a 2x4 Grid
  cards = [
      {
          "is_gt": True,
          "name": "The real Earth",
          "sub": "1-km Land Mask Ground Truth (29.0% Land · 71.0% Water)",
          "badge": "100.0%",
          "badge_color": "#16a34a",
          "border_color": "#0f172a",
          "border_width": "2",
          "rects": gt_rects,
          "foot_left": "Resolution: 2° × 2° (90 × 180 = 16,200 pts)",
          "foot_right": "Equirectangular",
      }
  ]
  for s in model_stats:
    rects = (
        s["rects_bw"]
        if mode == "bw"
        else (s["rects_diff"] if mode == "diff" else s["rects_tt"])
    )
    cards.append({
        "is_gt": False,
        "name": s["name"],
        "sub": f"{s['model_id']} · {s['release']}",
        "badge": f"{s['area_acc']:.1f}%",
        "badge_color": "#c2410c",
        "border_color": "#cbd5e1",
        "border_width": "1.2",
        "rects": rects,
        "foot_left": (
            f"Land F1: {s['land_f1']:.1f}% · Think: {s['avg_thoughts']:.0f}"
            f" tok · p50: {s['med_ms']/1000:.1f}s"
        ),
        "foot_right": f"+{s['fp']} FP / -{s['fn']} FN",
    })

  card_w, card_h = 381, 278
  gap_x, gap_y = 20, 20
  start_x, start_y = 28, 86
  map_w, map_h = 353, 176.5

  for idx, c in enumerate(cards):
    row = idx // 4
    col = idx % 4
    cx = start_x + col * (card_w + gap_x)
    cy = start_y + row * (card_h + gap_y)

    parts.append(
        f'<g transform="translate({cx},{cy})">'
        f'<rect width="{card_w}" height="{card_h}" rx="12" fill="#ffffff"'
        f' stroke="{c["border_color"]}" stroke-width="{c["border_width"]}"/>'
        f'<text x="14" y="25" font-family="{font}" font-size="16"'
        f' font-weight="700" fill="#0f172a">{html.escape(c["name"])}</text>'
        f'<text x="14" y="43" font-family="{mono}" font-size="11"'
        f' fill="#64748b">{html.escape(c["sub"])}</text>'
        f'<text x="{card_w - 14}" y="28" text-anchor="end"'
        f' font-family="{font}" font-size="20" font-weight="800"'
        f' fill="{c["badge_color"]}">{html.escape(c["badge"])}</text>'
        # Nested SVG for the 180x90 map
        f'<svg x="14" y="54" width="{map_w}" height="{map_h}"'
        ' viewBox="0 0 180 90" preserveAspectRatio="none"'
        ' shape-rendering="crispEdges">'
        + c["rects"]
        + "</svg>"
        f'<rect x="14" y="54" width="{map_w}" height="{map_h}" fill="none"'
        ' stroke="#334155" stroke-width="1"/>'
        f'<text x="14" y="254" font-family="{font}" font-size="11.5"'
        f' font-weight="500" fill="#475569">{html.escape(c["foot_left"])}</text>'
        f'<text x="{card_w - 14}" y="254" text-anchor="end"'
        f' font-family="{mono}" font-size="11" font-weight="600"'
        f' fill="#64748b">{html.escape(c["foot_right"])}</text>'
        "</g>"
    )

  parts.append("</svg>\n")
  return "".join(parts)


def build_comparison_chart_svg(model_stats: list[dict]) -> str:
  """Builds a standalone SVG horizontal bar chart comparing Gemini vs Claude Reference."""
  combined = []
  for s in model_stats:
    combined.append({
        "name": s["name"],
        "acc": s["area_acc"],
        "is_gemini": True,
        "is_lite": "Lite" in s["name"],
        "sub": f"avg {s['avg_thoughts']:.0f} thought tok",
    })
  for c in CLAUDE_REFS:
    combined.append({
        "name": c["name"],
        "acc": c["acc"],
        "is_gemini": False,
        "is_lite": False,
        "sub": "@celestepoasts benchmark",
    })
  combined.sort(key=lambda x: x["acc"], reverse=True)

  width = 1060
  row_h = 34
  top_y = 84
  height = top_y + len(combined) * row_h + 24
  font = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
  mono = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"

  parts = [
      f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"'
      f' width="{width}" height="{height}">',
      f'<rect width="{width}" height="{height}" rx="14" fill="#ffffff"'
      ' stroke="#e2e8f0" stroke-width="2"/>',
      f'<text x="28" y="36" font-family="{font}" font-size="19"'
      ' font-weight="800" fill="#0f172a">Cross-Family Comparison: Published'
      " Gemini Models vs. Claude Reference (@celestepoasts)</text>",
      f'<text x="28" y="58" font-family="{font}" font-size="12.5"'
      ' fill="#64748b">Area-weighted accuracy (cos(lat)) across 16,200 global'
      " coordinates (2° × 2°) with zero tools and default settings</text>",
  ]

  sub_x = 242
  bar_x = 408
  bar_max_w = 550

  for i, item in enumerate(combined):
    y = top_y + i * row_h
    w = max(4.0, (item["acc"] - 50.0) / 50.0 * bar_max_w)
    if item["is_gemini"]:
      bar_color = "#60a5fa" if item["is_lite"] else "#2563eb"
      name_weight = "700"
      name_color = "#0f172a"
      val_color = "#1d4ed8"
    else:
      bar_color = "#cbd5e1"
      name_weight = "500"
      name_color = "#475569"
      val_color = "#475569"

    parts.append(
        f'<text x="28" y="{y + 15}" font-family="{font}" font-size="13"'
        f' font-weight="{name_weight}" fill="{name_color}">'
        f'{html.escape(item["name"])}</text>'
        f'<text x="{sub_x}" y="{y + 15}" font-family="{mono}" font-size="11"'
        f' fill="#94a3b8">{html.escape(item["sub"])}</text>'
        f'<rect x="{bar_x}" y="{y + 3}" width="{bar_max_w}" height="16"'
        ' rx="4" fill="#f1f5f9"/>'
        f'<rect x="{bar_x}" y="{y + 3}" width="{w:.1f}" height="16" rx="4"'
        f' fill="{bar_color}"/>'
        f'<text x="{bar_x + bar_max_w + 68}" y="{y + 15.5}"'
        f' text-anchor="end" font-family="{mono}" font-size="13"'
        f' font-weight="700" fill="{val_color}">{item["acc"]:.1f}%</text>'
    )

  parts.append("</svg>\n")
  return "".join(parts)


def render_svg_to_png_via_browser(
    svg_path: Path, png_path: Path, width: int, height: int
) -> bool:
  gbrowser = shutil.which("gbrowser")
  if not gbrowser:
    return False
  import tempfile

  with tempfile.NamedTemporaryFile(
      mode="w", suffix=".html", encoding="utf-8", delete=False
  ) as tf:
    tf.write(
        '<!doctype html><html style="margin:0;padding:0"><body'
        ' style="margin:0;padding:0;overflow:hidden;background:#f8fafc">'
        + svg_path.read_text(encoding="utf-8")
        + "</body></html>"
    )
    tmp_html = Path(tf.name)
  batch_spec = json.dumps([
      {"action": "navigate", "url": f"file://{tmp_html.resolve()}"},
      {"action": "screenshot", "file": str(png_path.resolve())},
  ])
  try:
    subprocess.run(
        [
            gbrowser,
            "batch",
            f"--width={width}",
            f"--height={height}",
            batch_spec,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True
  except Exception:
    return False
  finally:
    tmp_html.unlink(missing_ok=True)


def compute_coastal_mask(gt: np.ndarray) -> np.ndarray:
  coastal = np.zeros_like(gt, dtype=bool)
  for r in range(90):
    for c in range(180):
      v = gt[r, c]
      is_b = False
      for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
          if dr == 0 and dc == 0:
            continue
          nr = min(89, max(0, r + dr))
          nc = (c + dc) % 180
          if gt[nr, nc] != v:
            is_b = True
            break
        if is_b:
          break
      coastal[r, c] = is_b
  return coastal


def render_all(export_data: bool = True) -> None:
  ASSETS_DIR.mkdir(parents=True, exist_ok=True)
  MAPS_DIR.mkdir(parents=True, exist_ok=True)

  gt = np.load(DATA_DIR / "ground_truth_90x180.npy")
  coastal_mask = compute_coastal_mask(gt)
  lats = 89.0 - 2.0 * np.arange(90)
  weights = np.cos(np.deg2rad(lats))[:, None] * np.ones((1, 180))

  gt_rects = build_map_rects(gt, mode="bw", gt=gt, show_graticule=True)
  (MAPS_DIR / "ground_truth.svg").write_text(
      wrap_single_map_svg(gt_rects), encoding="utf-8"
  )
  save_map_png(MAPS_DIR / "ground_truth.png", gt, mode="bw", gt=gt)

  model_stats = []
  for m in MODELS:
    slug = m["slug"]
    raw_path = RAW_DIR / f"raw_{slug}.jsonl"
    pred_grid = np.full((90, 180), -2, dtype=int)
    thoughts_grid = np.zeros((90, 180), dtype=float)
    lat_ms_list = []
    thoughts_list = []

    if raw_path.exists():
      with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
          line = line.strip()
          if not line:
            continue
          try:
            rec = json.loads(line)
          except Exception:
            continue
          lat = float(rec["lat"])
          lon = float(rec["lon"])
          r = int(round((89.0 - lat) / 2.0))
          c = int(round((lon + 179.0) / 2.0))
          if not (0 <= r < 90 and 0 <= c < 180):
            continue
          lbl = parse_label(rec.get("text", "") or rec.get("answer", ""))
          if lbl in (0, 1):
            pred_grid[r, c] = lbl
            thk = float(rec.get("thoughts_tokens", 0) or 0)
            thoughts_grid[r, c] = thk
            thoughts_list.append(thk)
            ms = float(rec.get("latency_ms", 0) or 0)
            if ms > 0:
              lat_ms_list.append(ms)

    queried_mask = pred_grid != -2
    done_pts = int(np.sum(queried_mask))
    valid_mask = (pred_grid == 0) | (pred_grid == 1)
    if np.any(valid_mask):
      correct_mask = (pred_grid == gt) & valid_mask
      area_acc = float(
          np.sum(weights[correct_mask]) / np.sum(weights[queried_mask]) * 100.0
      )
      raw_acc = float(np.sum(correct_mask) / np.sum(queried_mask) * 100.0)
      tp = int(np.sum((pred_grid == 1) & (gt == 1)))
      tn = int(np.sum((pred_grid == 0) & (gt == 0)))
      fp = int(np.sum((pred_grid == 1) & (gt == 0)))
      fn = int(np.sum((pred_grid == 0) & (gt == 1)))
      land_prec = tp / max(1, tp + fp) * 100.0
      land_rec = tp / max(1, tp + fn) * 100.0
      land_f1 = (
          2 * land_prec * land_rec / max(1e-9, land_prec + land_rec)
          if (land_prec + land_rec) > 0
          else 0.0
      )
      pred_land_area = float(
          np.sum(weights[pred_grid == 1])
          / np.sum(weights[queried_mask])
          * 100.0
      )
      c_mask = coastal_mask & valid_mask
      i_mask = (~coastal_mask) & valid_mask
      coast_tok = (
          float(np.mean(thoughts_grid[c_mask])) if np.any(c_mask) else 0.0
      )
      int_tok = float(np.mean(thoughts_grid[i_mask])) if np.any(i_mask) else 0.0
      coast_acc = (
          float(
              np.sum(weights[correct_mask & c_mask])
              / np.sum(weights[c_mask])
              * 100.0
          )
          if np.any(c_mask)
          else 0.0
      )
      int_acc = (
          float(
              np.sum(weights[correct_mask & i_mask])
              / np.sum(weights[i_mask])
              * 100.0
          )
          if np.any(i_mask)
          else 0.0
      )
    else:
      area_acc = raw_acc = land_prec = land_rec = land_f1 = pred_land_area = 0.0
      tp = tn = fp = fn = 0
      coast_tok = int_tok = coast_acc = int_acc = 0.0

    avg_thoughts = float(np.mean(thoughts_list)) if thoughts_list else 0.0
    p95_thoughts = (
        float(np.percentile(thoughts_list, 95)) if thoughts_list else 0.0
    )
    med_ms = float(np.median(lat_ms_list)) if lat_ms_list else 0.0
    max_tt = max(600.0, p95_thoughts)

    rects_bw = build_map_rects(pred_grid, mode="bw", gt=gt, show_graticule=True)
    rects_diff = build_map_rects(
        pred_grid, mode="diff", gt=gt, show_graticule=True
    )
    rects_tt = build_heatmap_rects(
        thoughts_grid,
        queried_mask,
        max_val=max_tt,
        fallback_grid=pred_grid,
        show_graticule=True,
    )

    # Save individual per-model SVGs & PNGs
    (MAPS_DIR / f"{slug}_bw.svg").write_text(
        wrap_single_map_svg(rects_bw), encoding="utf-8"
    )
    (MAPS_DIR / f"{slug}_diff.svg").write_text(
        wrap_single_map_svg(rects_diff), encoding="utf-8"
    )
    (MAPS_DIR / f"{slug}_thinking.svg").write_text(
        wrap_single_map_svg(rects_tt), encoding="utf-8"
    )
    save_map_png(MAPS_DIR / f"{slug}_bw.png", pred_grid, mode="bw", gt=gt)
    save_map_png(MAPS_DIR / f"{slug}_diff.png", pred_grid, mode="diff", gt=gt)
    save_map_png(
        MAPS_DIR / f"{slug}_thinking.png",
        pred_grid,
        mode="tt",
        gt=gt,
        values_grid=thoughts_grid,
        max_val=max_tt,
    )

    model_stats.append({
        **m,
        "done_pts": done_pts,
        "area_acc": area_acc,
        "raw_acc": raw_acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "land_prec": land_prec,
        "land_rec": land_rec,
        "land_f1": land_f1,
        "pred_land_area": pred_land_area,
        "avg_thoughts": avg_thoughts,
        "p95_thoughts": p95_thoughts,
        "med_ms": med_ms,
        "coast_tok": coast_tok,
        "int_tok": int_tok,
        "coast_acc": coast_acc,
        "int_acc": int_acc,
        "rects_bw": rects_bw,
        "rects_diff": rects_diff,
        "rects_tt": rects_tt,
    })

  # Write composite 2x4 SVGs and render PNGs
  for mode, fname in [
      ("bw", "world_maps_bw"),
      ("diff", "world_maps_diff"),
      ("tt", "world_maps_thinking"),
  ]:
    svg_path = ASSETS_DIR / f"{fname}.svg"
    png_path = ASSETS_DIR / f"{fname}.png"
    svg_content = build_composite_grid_svg(mode, gt_rects, model_stats)
    svg_path.write_text(svg_content, encoding="utf-8")
    render_svg_to_png_via_browser(svg_path, png_path, width=1640, height=696)

  comp_svg_path = ASSETS_DIR / "cross_family_comparison.svg"
  comp_png_path = ASSETS_DIR / "cross_family_comparison.png"
  comp_svg_content = build_comparison_chart_svg(model_stats)
  comp_svg_path.write_text(comp_svg_content, encoding="utf-8")
  render_svg_to_png_via_browser(
      comp_svg_path, comp_png_path, width=1060, height=788
  )

  if export_data:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gt_export = {
        "description": (
            "Ground truth 2-degree equirectangular land/water grid (90 rows x"
            " 180 cols). Row r corresponds to lat = 89 - 2*r, Col c corresponds"
            " to lon = -179 + 2*c. 1 = Land, 0 = Water."
        ),
        "lats": [float(89 - 2 * r) for r in range(90)],
        "lons": [float(-179 + 2 * c) for c in range(180)],
        "grid": gt.tolist(),
    }
    with (DATA_DIR / "ground_truth_90x180.json").open(
        "w", encoding="utf-8"
    ) as f:
      json.dump(gt_export, f)

    all_csv_path = DATA_DIR / "all_models.csv"
    all_jsonl_path = DATA_DIR / "all_models.jsonl"
    with (
        all_csv_path.open("w", encoding="utf-8") as csv_f,
        all_jsonl_path.open("w", encoding="utf-8") as jsonl_f,
    ):
      csv_f.write(
          "model,model_id,row,col,lat,lon,gt_label,pred_label,raw_answer,correct,area_weight,thoughts_tokens,total_tokens,latency_ms\n"
      )
      for m in MODELS:
        slug = m["slug"]
        raw_path = RAW_DIR / f"raw_{slug}.jsonl"
        dst_path = DATA_DIR / f"{slug}.jsonl"
        if not raw_path.exists():
          continue
        dedup = {}
        with raw_path.open("r", encoding="utf-8") as rf:
          for line in rf:
            line = line.strip()
            if not line:
              continue
            try:
              rec = json.loads(line)
              r = int(round((89.0 - float(rec["lat"])) / 2.0))
              c = int(round((float(rec["lon"]) + 179.0) / 2.0))
              lbl = parse_label(rec.get("text", ""))
              if (r, c) not in dedup or lbl in (0, 1):
                dedup[(r, c)] = rec
            except Exception:
              continue
        with dst_path.open("w", encoding="utf-8") as mf:
          for r, c in sorted(dedup.keys()):
            rec = dedup[(r, c)]
            gt_val = int(gt[r, c])
            gt_str = "Land" if gt_val == 1 else "Water"
            lbl = parse_label(rec.get("text", ""))
            pred_str = (
                "Land" if lbl == 1 else ("Water" if lbl == 0 else "Invalid")
            )
            correct = 1 if lbl == gt_val else 0
            w = float(np.cos(np.deg2rad(float(rec["lat"]))))
            enriched = {
                "model": slug,
                "model_id": m["model_id"],
                "row": r,
                "col": c,
                "lat": rec["lat"],
                "lon": rec["lon"],
                "gt_label": gt_str,
                "pred_label": pred_str,
                "raw_answer": rec.get("text", ""),
                "correct": bool(correct),
                "area_weight": round(w, 6),
                "thoughts_tokens": rec.get("thoughts_tokens", 0),
                "total_tokens": rec.get("total_tokens", 0),
                "latency_ms": rec.get("latency_ms", 0),
            }
            line_out = json.dumps(enriched) + "\n"
            mf.write(line_out)
            jsonl_f.write(line_out)
            ans_esc = (
                '"' + str(rec.get("text", "")).replace('"', '""') + '"'
            )
            csv_f.write(
                f"{slug},{m['model_id']},{r},{c},{rec['lat']},{rec['lon']},"
                f"{gt_str},{pred_str},{ans_esc},{correct},{w:.6f},"
                f"{rec.get('thoughts_tokens', 0)},{rec.get('total_tokens', 0)},"
                f"{rec.get('latency_ms', 0)}\n"
            )

  # Build README.md report
  sorted_stats = sorted(model_stats, key=lambda x: x["area_acc"], reverse=True)
  compact_rows = []
  detailed_rows = []
  for idx, s in enumerate(sorted_stats, 1):
    thk_str = (
        f"{s['avg_thoughts']:.0f} *(p95: {s['p95_thoughts']:.0f})*"
        if s["avg_thoughts"] > 0
        else "0 *(non-thinking)*"
    )
    compact_rows.append(
        f"| **#{idx}** | **{s['name']}** | `{s['model_id']}` |"
        f" **{s['area_acc']:.2f}%** | {s['raw_acc']:.2f}% |"
        f" {s['land_f1']:.2f}% | {thk_str} |"
    )
    detailed_rows.append(
        f"| **#{idx}** | **{s['name']}** (`{s['model_id']}`) | {s['release']} |"
        f" {s['land_prec']:.1f}% / {s['land_rec']:.1f}% |"
        f" {s['pred_land_area']:.1f}% | +{s['fp']} / -{s['fn']} |"
        f" {s['med_ms']/1000:.2f}s | [{s['slug']}.jsonl](data/{s['slug']}.jsonl) |"
    )

  coast_rows = []
  for s in sorted_stats:
    if s["avg_thoughts"] <= 0:
      continue
    ratio = s["coast_tok"] / max(1.0, s["int_tok"])
    coast_rows.append(
        f"| **{s['name']}** | {s['coast_tok']:.0f} tok |"
        f" {s['int_tok']:.0f} tok | **{ratio:.1f}×** | {s['coast_acc']:.2f}% |"
        f" {s['int_acc']:.2f}% |"
    )

  per_model_gallery_rows = [
      "| **The real Earth** *(1-km Ground Truth)* |"
      " [SVG](assets/maps/ground_truth.svg) ·"
      " [PNG](assets/maps/ground_truth.png) | — | — |"
      " [ground_truth_90x180.json](data/ground_truth_90x180.json) |"
  ]
  for s in model_stats:
    slug = s["slug"]
    per_model_gallery_rows.append(
        f"| **{s['name']}** (`{s['model_id']}`) |"
        f" [SVG](assets/maps/{slug}_bw.svg) ·"
        f" [PNG](assets/maps/{slug}_bw.png) |"
        f" [SVG](assets/maps/{slug}_diff.svg) ·"
        f" [PNG](assets/maps/{slug}_diff.png) |"
        f" [SVG](assets/maps/{slug}_thinking.svg) ·"
        f" [PNG](assets/maps/{slug}_thinking.png) |"
        f" [{slug}.jsonl](data/{slug}.jsonl) |"
    )

  readme_md = f"""# How Blind Geminis See the Earth 🌍

[![Website](https://img.shields.io/badge/Daily_AI_Digest-jacksunwei.me-EA580C?logo=rss&logoColor=white)](https://jacksunwei.me/)
[![Author](https://img.shields.io/badge/X-@Jacksunwei-000000?logo=x&logoColor=white)](https://x.com/Jacksunwei)
[![GitHub](https://img.shields.io/badge/GitHub-Jacksunwei-181717?logo=github&logoColor=white)](https://github.com/Jacksunwei)
[![SDK](https://img.shields.io/badge/SDK-google--genai-4285F4?logo=google&logoColor=white)](https://github.com/googleapis/python-genai)
[![Dataset](https://img.shields.io/badge/Dataset-113,400_queries-059669)](data/all_models.jsonl)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Can an LLM reconstruct the entire world map purely from its parametric weights—one coordinate at a time, with zero tools?**

Built by **[Wei (Jack) Sun](https://jacksunwei.me/)** ([@Jacksunwei](https://x.com/Jacksunwei)) · Inspired by [@karpathy](https://x.com/karpathy) & [@celestepoasts](https://x.com/celestepoasts/status/2039451120042442950).

We query **7 published Gemini 3.x models** independently across all **16,200 cell centers** of a global $2^\\circ \\times 2^\\circ$ grid ($90 \\times 180$, **113,400 API calls**) using the official [`google-genai`](https://github.com/googleapis/python-genai) SDK (`tools=[]`, no web search, default settings):

```text
Answer with a single word, either "Land" or "Water", and nothing else: is ({{lat}}°{{N/S}}, {{lon}}°{{E/W}}) on land or water?
```

### Key Takeaways

- **99.1% World Map Accuracy**: Every thinking Gemini 3.x model reaches **98.8%–99.1%** area-weighted accuracy from weights alone—led by **`gemini-3.8-flash` (99.09%)** and **`gemini-3.1-pro-preview` (99.02%)**, matching **`Claude Opus 5.5 (thinks)` (99.0%)**.
- **Emergent Coastline Detector**: Plotting internal reasoning tokens (`thoughts_token_count`) per coordinate reveals that Gemini automatically spends **1.8×–2.6× more compute along coastlines and islands** than in deep oceans or continental interiors—even though the output is a single word.
- **31% More Token-Efficient**: `gemini-3.8-flash` edges out `gemini-3.1-pro-preview` (`99.09%` vs. `99.02%`) while using **31% fewer thinking tokens** (`408` vs. `588` avg).

---

## World Maps from Pure Weights

![Classic Land/Water Map](assets/world_maps_bw.svg)

## Emergent Coastline Reasoning (`thoughts_token_count`)

Without being told where coastlines are, every thinking Gemini model dynamically scales test-time compute at land/water boundaries (`3,145` coastal cells vs. `13,055` interior cells). Non-thinking `Flash-Lite` models (`0` thinking tokens by default) answer in a single forward pass.

![Thinking Effort Heatmap](assets/world_maps_thinking.svg)

| Model | Coastal Thinking | Interior Thinking | Coastal / Interior Ratio | Coastal Acc | Interior Acc |
| :--- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(coast_rows)}

## Where Models Hallucinate (False Positives vs. False Negatives)

Red pixels mark **hallucinated land** (`False Positive`: mostly coastal shelves and small island chains); blue pixels mark **submerged land** (`False Negative`).

![Error Diagnostic Map](assets/world_maps_diff.svg)

---

## Leaderboard & Cross-Family Comparison

![Cross-Family Comparison](assets/cross_family_comparison.svg)

Primary accuracy is weighted by spherical surface area $w(\\phi) = \\cos(\\phi)$ against a 1-km global ground-truth mask (29.0% Land, 71.0% Water).

| Rank | Model | Public API ID | Area Acc | Raw Grid Acc | Land F1 | Avg Thinking Tokens |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
{chr(10).join(compact_rows)}

<details>
<summary><strong>📊 Full Breakdown (Precision/Recall, FP/FN, Latency) & Individual Map Downloads</strong></summary>

<br>

| Rank | Model | Release | Land Prec / Rec | Pred Land % | Errors (FP / FN) | Median Latency | Enriched Dataset |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
{chr(10).join(detailed_rows)}

### Individual Per-Model Maps & Datasets

- **All 7 Models Combined (`113,400` rows)**: [`data/all_models.jsonl`](data/all_models.jsonl) · [`data/all_models.csv`](data/all_models.csv)
- **Composite High-Res Figures**: [Classic B&W (PNG)](assets/world_maps_bw.png) · [Thinking Heatmap (PNG)](assets/world_maps_thinking.png) · [Error Map (PNG)](assets/world_maps_diff.png) · [Comparison Chart (PNG)](assets/cross_family_comparison.png)

| Model | Classic Map | Error Map | Thinking Heatmap | Raw Dataset |
| :--- | :--- | :--- | :--- | :--- |
{chr(10).join(per_model_gallery_rows)}

</details>

---

## Quick Start

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync

# 1. Run or resume the 16,200-coordinate benchmark via public google-genai SDK
export GEMINI_API_KEY="your-api-key"
uv run python run_eval.py --models gemini-3.8-flash --concurrency 150

# 2. Regenerate all SVG/PNG figures, datasets, and README.md
uv run python render_report.py
```

---

## About the Author

**[Wei (Jack) Sun](https://jacksunwei.me/)** — Hands-on with agentic AI all day: building frameworks at Google, reading what the industry ships, and writing it down.

- **X / Twitter**: [@Jacksunwei](https://x.com/Jacksunwei)
- **GitHub**: [@Jacksunwei](https://github.com/Jacksunwei)
- **Blog**: [jacksunwei.me](https://jacksunwei.me/) — daily AI digest (Tech, Research, News) and essays ([RSS](https://jacksunwei.me/feed.xml))

| Project | Description |
| :--- | :--- |
| **[`google/adk-python`](https://github.com/google/adk-python)** | Open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents. |
| **[`gemini-web-mcp`](https://github.com/Jacksunwei/gemini-web-mcp)** | Gemini-powered MCP server for real Google Search grounding, multi-URL synthesis, and Nano Banana image generation. |

## License

[Apache-2.0](LICENSE)
"""
  (BASE_DIR / "README.md").write_text(readme_md, encoding="utf-8")
  print("Rendered README.md, SVG/PNG assets, and datasets in", BASE_DIR)
  print(
      f"{'Model':<24} {'Done':>7} {'Area Acc':>9} {'Grid Acc':>9} {'Land F1':>8} {'Think Tok':>10}"
  )
  for s in model_stats:
    print(
        f"{s['name']:<24} {s['done_pts']:>7} {s['area_acc']:>8.2f}% {s['raw_acc']:>8.2f}% {s['land_f1']:>7.2f}% {s['avg_thoughts']:>10.1f}"
    )


if __name__ == "__main__":
  import sys

  do_export = "--no-export" not in sys.argv
  render_all(export_data=do_export)
