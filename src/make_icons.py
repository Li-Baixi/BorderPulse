# -*- coding: utf-8 -*-
"""Generate PWA icons (rounded square, pulse-line mark) into docs/icons/.

Primary author: Lu Zengyi (product icon set).

【中文说明 —— PWA 图标生成器】
作用：用 Pillow 画出全套应用图标（192/512/maskable-512/180/32 px），
不需要任何图片素材文件。图形 = 蓝底圆角方块 + 白色"闸机脉冲线"
（平→尖峰→回落）+ 两侧闸机柱——呼应产品名 BorderPulse。
运行：python src/make_icons.py（图标已生成并提交，通常无需重跑）。
"""

import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "docs", "icons")
os.makedirs(OUT, exist_ok=True)

BG = (28, 92, 171)        # blue step 550
BG_DARK = (13, 54, 107)   # blue step 700
WHITE = (255, 255, 255)
# 中文：两个蓝取自产品配色（550 主色 / 700 深色），底部深色带做出层次感


def pulse_line(w, y0):
    """Crossing-gate pulse: flat, spike up, dip down, flat."""
    # 中文：脉冲折线的 6 个顶点（比例坐标）——水平的两段夹一个
    # 上尖峰和下回落，像过关人流的瞬时脉冲。
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
    # 中文：画一个尺寸的图标。maskable=True 时图形收缩到中心 80%
    # 安全区内（Android 圆形遮罩不会裁掉内容），其余留纯色底。
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
    for a, b in pulse_line(w, y0):
        d.line([a, b], fill=WHITE, width=stroke)
        # 中文：每个端点补一个小圆——折线交角处不会出现缺口
        for pt in (a, b):
            d.ellipse([pt[0] - stroke / 2, pt[1] - stroke / 2,
                       pt[0] + stroke / 2, pt[1] + stroke / 2], fill=WHITE)
    # gate posts flanking the pulse
    gh = 0.16 * w
    for x in (pad + 0.20 * w, pad + 0.80 * w):
        d.rounded_rectangle([x - stroke / 2, y0 - gh, x + stroke / 2, y0 + gh],
                            radius=stroke / 2, fill=WHITE)
    return img


# 中文：一次生成全套尺寸（manifest、iOS 触摸图标、favicon 各取所需）
draw_icon(192).save(os.path.join(OUT, "icon-192.png"))
draw_icon(512).save(os.path.join(OUT, "icon-512.png"))
draw_icon(512, maskable=True).save(os.path.join(OUT, "icon-maskable-512.png"))
draw_icon(180).save(os.path.join(OUT, "apple-touch-icon.png"))
draw_icon(32).save(os.path.join(OUT, "favicon-32.png"))
print("icons written to", OUT)
