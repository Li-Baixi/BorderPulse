# -*- coding: utf-8 -*-
"""Generate PWA icons (rounded square, pulse-line mark) into docs/icons/."""
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "docs", "icons")
os.makedirs(OUT, exist_ok=True)

BG = (28, 92, 171)        # blue step 550
BG_DARK = (13, 54, 107)   # blue step 700
WHITE = (255, 255, 255)


def pulse_line(w, y0, stroke):
    """Crossing-gate pulse: flat, spike up, dip down, flat."""
    p = stroke // 2
    pts = [
        (0.08 * w, y0), (0.30 * w, y0),
        (0.38 * w, y0 - 0.24 * w), (0.46 * w, y0 + 0.15 * w),
        (0.54 * w, y0), (0.92 * w, y0),
    ]
    lines = []
    for a, b in zip(pts, pts[1:]):
        lines.append([a, b])
    return lines


def draw_icon(size, maskable=False):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = size * 0.22
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=BG)
    # subtle bottom band for depth
    d.rounded_rectangle([0, size * 0.72, size - 1, size - 1], radius=r,
                        fill=BG_DARK)
    d.rectangle([0, size * 0.72, size - 1, size * 0.72 + r], fill=BG_DARK)
    # mark: keep inside 80% safe zone when maskable
    pad = 0.24 * size if maskable else 0.16 * size
    w = size - 2 * pad
    y0 = pad + 0.5 * w
    stroke = max(4, int(size * 0.055))
    for a, b in pulse_line(w, y0, stroke):
        d.line([a, b], fill=WHITE, width=stroke)
        for pt in (a, b):
            d.ellipse([pt[0] - stroke / 2, pt[1] - stroke / 2,
                       pt[0] + stroke / 2, pt[1] + stroke / 2], fill=WHITE)
    # gate posts flanking the pulse
    gh = 0.16 * w
    for x in (pad + 0.20 * w, pad + 0.80 * w):
        d.rounded_rectangle([x - stroke / 2, y0 - gh, x + stroke / 2, y0 + gh],
                            radius=stroke / 2, fill=WHITE)
    return img


draw_icon(192).save(os.path.join(OUT, "icon-192.png"))
draw_icon(512).save(os.path.join(OUT, "icon-512.png"))
draw_icon(512, maskable=True).save(os.path.join(OUT, "icon-maskable-512.png"))
draw_icon(180).save(os.path.join(OUT, "apple-touch-icon.png"))
draw_icon(32).save(os.path.join(OUT, "favicon-32.png"))
print("icons written to", OUT)
