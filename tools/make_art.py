#!/usr/bin/env python3
"""make_art.py: render the README banner and charts as black-and-white PNGs (PIL + Menlo + pyfiglet ansi_shadow).

  PYTHONPATH=<dir with pyfiglet> python3 tools/make_art.py assets/numbers.json assets/

numbers.json holds the measured numbers (see assets/numbers.json); every chart row carries its own method label.
"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont
import pyfiglet

FONT = "/System/Library/Fonts/Menlo.ttc"
S = 2  # render scale
W = 1280 * S
WHITE, GRAY, DIM, FRAME = (255, 255, 255), (150, 150, 150), (110, 110, 110), (55, 55, 55)

def f(size, bold=False):
    return ImageFont.truetype(FONT, size * S, index=1 if bold else 0)

def center(d, y, text, font, fill):
    w = d.textlength(text, font=font); d.text(((W - w) / 2, y), text, font=font, fill=fill)

def banner(nums, out):
    H = 440 * S
    im = Image.new("RGB", (W, H), (0, 0, 0)); d = ImageDraw.Draw(im)
    m = 24 * S
    center(d, 40 * S, "~/pico-dgx-ds4", f(13), GRAY)
    art = pyfiglet.figlet_format("PICO-DGX-DS4", font="ansi_shadow", width=400).rstrip("\n").split("\n")
    art = [l for l in art if l.strip()]
    af = f(15); lh = 17 * S
    aw = max(d.textlength(l, font=af) for l in art)
    y = 70 * S
    for l in art:
        d.text(((W - aw) / 2, y), l, font=af, fill=WHITE); y += lh
    center(d, y + 16 * S, nums["subtitle"], f(16), GRAY)
    stats = nums["stats"]; n = len(stats); colw = (W - 2 * 90 * S) / n
    ys = y + 62 * S
    for i, (big, small) in enumerate(stats):
        cx = 90 * S + colw * (i + 0.5)
        bf = f(40, True); sf = f(13)
        bw = d.textlength(big, font=bf); d.text((cx - bw / 2, ys), big, font=bf, fill=WHITE)
        sw = d.textlength(small, font=sf); d.text((cx - sw / 2, ys + 56 * S), small, font=sf, fill=GRAY)
        if i:
            x = 90 * S + colw * i; d.line([x, ys, x, ys + 76 * S], fill=FRAME, width=2 * S)
    bottom = ys + 76 * S + 30 * S
    im = im.crop((0, 0, W, bottom + m))
    d = ImageDraw.Draw(im); d.rectangle([m, m, W - m, bottom], outline=FRAME, width=2 * S)
    im.save(out)

def chart(title, sub, rows, out, unit="tok/s", note=None):
    # rows: [label, value, method_label, ours(bool)]
    lf = f(15); nf = f(15, True); mf = f(12)
    lh = 44 * S
    H = (120 + 44 * len(rows) + (40 if note else 10) + 30) * S
    im = Image.new("RGB", (W, H), (0, 0, 0)); d = ImageDraw.Draw(im)
    m = 24 * S
    d.text((60 * S, 44 * S), title, font=f(20, True), fill=WHITE)
    d.text((60 * S, 76 * S), sub, font=f(12), fill=GRAY)
    vmax = max(r[1] for r in rows)
    cols = 52
    y = 120 * S
    cw = d.textlength("█", font=lf)
    for lab, v, meth, ours in rows:
        col = WHITE if ours else GRAY
        d.text((60 * S, y), f"{lab:<24}", font=lf, fill=col)
        x0 = 60 * S + d.textlength(" " * 25, font=lf)
        fill = max(1, round(cols * v / vmax))
        d.text((x0, y), "█" * fill, font=lf, fill=col)
        d.text((x0 + cw * fill, y), "░" * (cols - fill), font=lf, fill=FRAME)
        xv = x0 + cw * cols + 14 * S
        d.text((xv, y), f"{v:,.0f}" if v >= 100 else f"{v:.1f}", font=nf, fill=col)
        d.text((xv + 80 * S, y + 3 * S), meth, font=mf, fill=DIM)
        y += lh
    if note:
        d.text((60 * S, y + 6 * S), note, font=f(11), fill=DIM)
        y += 30 * S
    im = im.crop((0, 0, W, y + 30 * S))
    d = ImageDraw.Draw(im); d.rectangle([m, m, W - m, im.size[1] - m], outline=FRAME, width=2 * S)
    im.save(out)

if __name__ == "__main__":
    nums = json.load(open(sys.argv[1])); od = sys.argv[2]; os.makedirs(od, exist_ok=True)
    banner(nums["banner"], os.path.join(od, "banner.png"))
    for c in nums["charts"]:
        chart(c["title"], c["sub"], c["rows"], os.path.join(od, c["file"]), note=c.get("note"))
    print("ok")
