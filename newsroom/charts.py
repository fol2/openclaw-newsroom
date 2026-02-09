from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Any


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s or s.upper() in {"N/D", "NA", "N/A", "NULL"}:
            return None
        return float(s.replace(",", ""))
    except Exception:
        return None


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return length + chunk_type + data + struct.pack(">I", crc)


def _render_with_pillow(
    *,
    closes: list[float],
    labels: list[str],
    symbol: str,
    out_path: Path,
    title: str | None,
    subtitle: str | None,
    width: int,
    height: int,
) -> Path | None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except Exception:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Canvas
    img = Image.new("RGB", (int(width), int(height)), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    pad_l, pad_r, pad_t, pad_b = 54, 16, 34, 36
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b

    # Header
    main_title = title or f"{symbol} 60D"
    draw.text((pad_l, 10), main_title, fill=(25, 25, 25), font=font)
    if subtitle:
        draw.text((pad_l, 22), subtitle, fill=(80, 80, 80), font=font)

    lo = min(closes)
    hi = max(closes)
    if math.isclose(lo, hi):
        lo *= 0.99
        hi *= 1.01

    # Grid + axes
    grid_color = (220, 225, 232)
    axis_color = (120, 120, 120)
    for i in range(6):
        y = pad_t + int((chart_h * i) / 5)
        draw.line([(pad_l, y), (pad_l + chart_w, y)], fill=grid_color, width=1)
    draw.line([(pad_l, pad_t), (pad_l, pad_t + chart_h)], fill=axis_color, width=1)
    draw.line([(pad_l, pad_t + chart_h), (pad_l + chart_w, pad_t + chart_h)], fill=axis_color, width=1)

    # Price labels (left)
    for i in range(6):
        v = hi - ((hi - lo) * i / 5)
        y = pad_t + int((chart_h * i) / 5)
        draw.text((6, y - 6), f"{v:.2f}", fill=(90, 90, 90), font=font)

    # X tick labels (bottom) - 5 labels
    tick_count = 5
    if labels:
        for i in range(tick_count):
            idx = int(round((len(labels) - 1) * i / (tick_count - 1)))
            x = pad_l + int((chart_w * i) / (tick_count - 1))
            draw.text((x - 18, pad_t + chart_h + 6), labels[idx][5:], fill=(90, 90, 90), font=font)

    # Line path
    pts = []
    n = len(closes)
    for i, c in enumerate(closes):
        x = pad_l + int(chart_w * i / (n - 1))
        y = pad_t + int(chart_h * (1.0 - (c - lo) / (hi - lo)))
        pts.append((x, y))

    line_color = (26, 115, 232)
    for i in range(1, len(pts)):
        draw.line([pts[i - 1], pts[i]], fill=line_color, width=2)

    # Last point marker
    lx, ly = pts[-1]
    draw.ellipse((lx - 3, ly - 3, lx + 3, ly + 3), fill=line_color)

    img.save(out_path, format="PNG", optimize=True)
    return out_path


def _draw_line(rows: list[bytearray], *, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], width: int = 1) -> None:
    """Integer Bresenham line with a simple square 'brush' for thickness."""
    h = len(rows)
    if h <= 0:
        return
    w = len(rows[0]) // 3

    def set_px(x: int, y: int) -> None:
        if 0 <= x < w and 0 <= y < h:
            i = x * 3
            r, g, b = color
            rows[y][i] = r
            rows[y][i + 1] = g
            rows[y][i + 2] = b

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    half = max(0, int(width) // 2)

    while True:
        for oy in range(-half, half + 1):
            for ox in range(-half, half + 1):
                set_px(x0 + ox, y0 + oy)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _draw_filled_circle(
    rows: list[bytearray], *, cx: int, cy: int, radius: int, color: tuple[int, int, int]
) -> None:
    h = len(rows)
    if h <= 0:
        return
    w = len(rows[0]) // 3
    r2 = int(radius) * int(radius)

    for y in range(cy - radius, cy + radius + 1):
        if y < 0 or y >= h:
            continue
        for x in range(cx - radius, cx + radius + 1):
            if x < 0 or x >= w:
                continue
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) > r2:
                continue
            i = x * 3
            rows[y][i : i + 3] = bytes(color)


def _render_with_stdlib(
    *,
    closes: list[float],
    symbol: str,
    out_path: Path,
    width: int,
    height: int,
) -> Path:
    """Render a simple price line chart as a PNG with *no external dependencies*."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width = int(width)
    height = int(height)
    if width < 200 or height < 120:
        raise ValueError("invalid_canvas")

    # Image rows (RGB), pre-filled background.
    bg = (245, 247, 250)
    bg_row = bytes(bg) * width
    rows = [bytearray(bg_row) for _ in range(height)]

    pad_l, pad_r, pad_t, pad_b = 54, 18, 22, 30
    chart_w = max(10, width - pad_l - pad_r)
    chart_h = max(10, height - pad_t - pad_b)

    lo = min(closes)
    hi = max(closes)
    if math.isclose(lo, hi):
        lo *= 0.99
        hi *= 1.01

    # Grid + axes.
    grid = (220, 225, 232)
    axis = (120, 120, 120)
    for i in range(6):
        y = pad_t + int((chart_h * i) / 5)
        _draw_line(rows, x0=pad_l, y0=y, x1=pad_l + chart_w, y1=y, color=grid, width=1)
    _draw_line(rows, x0=pad_l, y0=pad_t, x1=pad_l, y1=pad_t + chart_h, color=axis, width=1)
    _draw_line(rows, x0=pad_l, y0=pad_t + chart_h, x1=pad_l + chart_w, y1=pad_t + chart_h, color=axis, width=1)

    # Line path.
    pts: list[tuple[int, int]] = []
    n = len(closes)
    for i, c in enumerate(closes):
        x = pad_l + int(chart_w * i / (n - 1))
        y = pad_t + int(chart_h * (1.0 - (c - lo) / (hi - lo)))
        pts.append((x, y))

    line = (26, 115, 232)
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        _draw_line(rows, x0=x0, y0=y0, x1=x1, y1=y1, color=line, width=2)

    # Last point marker.
    lx, ly = pts[-1]
    _draw_filled_circle(rows, cx=lx, cy=ly, radius=3, color=line)

    # Minimal "header" accent bar (text-free; avoid fonts).
    header_h = 4
    accent = (26, 115, 232)
    for y in range(0, min(header_h, height)):
        for x in range(0, width):
            i = x * 3
            rows[y][i : i + 3] = bytes(accent)

    # Encode PNG.
    raw = bytearray()
    for row in rows:
        raw.append(0)  # filter type 0
        raw.extend(row)

    compressed = zlib.compress(bytes(raw), level=6)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    png = bytearray()
    png.extend(signature)
    png.extend(_png_chunk(b"IHDR", ihdr))
    png.extend(_png_chunk(b"IDAT", compressed))
    png.extend(_png_chunk(b"IEND", b""))

    out_path.write_bytes(bytes(png))
    return out_path


def render_line_chart_png(
    *,
    points: list[dict[str, Any]],
    symbol: str,
    out_path: Path,
    title: str | None = None,
    subtitle: str | None = None,
    width: int = 900,
    height: int = 450,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    closes: list[float] = []
    labels: list[str] = []
    for p in points or []:
        if not isinstance(p, dict):
            continue
        c = _safe_float(p.get("close"))
        d = str(p.get("date") or "").strip()
        if c is None or not d:
            continue
        closes.append(float(c))
        labels.append(d)

    if len(closes) < 2:
        raise ValueError("not_enough_points")

    width = int(width)
    height = int(height)

    # Prefer Pillow when available (nicer output), but keep a zero-deps fallback.
    rendered = _render_with_pillow(
        closes=closes,
        labels=labels,
        symbol=str(symbol or "").strip() or "ASSET",
        out_path=out_path,
        title=title,
        subtitle=subtitle,
        width=width,
        height=height,
    )
    if rendered is not None:
        return rendered

    return _render_with_stdlib(closes=closes, symbol=str(symbol or "").strip() or "ASSET", out_path=out_path, width=width, height=height)
