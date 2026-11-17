"""Shared drawing helpers for the Moon animation."""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
# Inter font files (Inter-Regular/Medium/SemiBold/Bold.ttf); override with FONT_DIR
FD = os.environ.get("FONT_DIR", str(Path(__file__).resolve().parents[1] / "infografik" / "fonts")) + os.sep
def font(w, size):
    return ImageFont.truetype(FD + {"r": "Inter-Regular.ttf", "m": "Inter-Medium.ttf", "s": "Inter-SemiBold.ttf", "b": "Inter-Bold.ttf"}[w], size)
DARK_BG = (11, 13, 18)
CREAM = (245, 242, 234)
MUTED = (154, 149, 138)
GOLD = (245, 196, 0)
GOLD_DIM = (150, 120, 20)
BLUE = (27, 79, 138)
BLUE_LIGHT = (122, 168, 214)
TEAL_LIGHT = (86, 201, 176)

def tracked(dr, xy, text, f, fill, tracking=0):
    x, y = xy
    for ch in text:
        dr.text((x, y), ch, font=f, fill=fill)
        x += dr.textlength(ch, font=f) + tracking
    return x

def header(dr, W, S, headline, sub):
    # thin blue bar + kicker, like the reference style
    dr.rectangle([0, 0, W, 8 * S], fill=BLUE)
    tracked(dr, (56 * S, 34 * S), "KI-EXPERIMENTE", font("s", 19 * S), BLUE_LIGHT, tracking=4 * S)
    dr.text((56 * S, 66 * S), headline, font=font("b", 56 * S), fill=CREAM)
    if sub:
        dr.text((56 * S, 140 * S), sub, font=font("m", 28 * S), fill=MUTED)

def footer(dr, W, H, S, lines):
    y = H - (34 + 28 * (len(lines) - 1)) * S - 26 * S
    dr.line([56 * S, y - 16 * S, W - 56 * S, y - 16 * S], fill=(40, 43, 52), width=S)
    for i, l in enumerate(lines):
        dr.text((56 * S, y + i * 28 * S), l, font=font("r", 19 * S), fill=MUTED)
