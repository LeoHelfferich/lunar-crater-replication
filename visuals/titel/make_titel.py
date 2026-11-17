#!/usr/bin/env python3
"""Title images for the Moon crater visuals (4:5 and 1200×627 link preview).

Moon disc: NASA SVS "Moon Phase and Libration, 2026", plain frame 7864,
2026-11-24 15:00 UTC, phase 99.81 %. Public domain.
  https://svs.gsfc.nasa.gov/5587/
  https://svs.gsfc.nasa.gov/vis/a000000/a005500/a005587/frames/5760x3240_16x9_30p/plain/moon.7864.tif
Credit: NASA's Scientific Visualization Studio (NASA/GSFC).
Datasets: LROC WAC Color Mosaic (NASA/GSFC / Arizona State University), LOLA DEM.

Insets: USGS LROC WAC Global Morphology Mosaic 100 m, June 2013, public domain.
Credit: NASA/GSFC/Arizona State University (LROC WAC), USGS.
Detections: our seed-46 checkpoint, threshold 0.7, visualization only.
No NASA logo, no Sombench tiles, no AI-generated moon imagery.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "infografik" / "fonts"
SRC = ROOT / "titel" / "src"
OUT = ROOT / "titel"
OUT.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

MOON_TIF = SRC / "moon.7864.tif"
HOCHLAND_SRC = ROOT / "krater-palus-somni.png"
TYCHO_SRC = ROOT / "krater-tycho-nordostteil.png"

# SVS Dial-A-Moon, frame 7864 (API 2026-11-24T15:00)
SUB_LAT = -6.466
SUB_LON = -2.461
POSANGLE = 347.892

HOCHLAND = dict(name="Palus Somni", lat=12.0, lon=44.0, caption="Palus Somni  ·  51 Krater")
TYCHO = dict(name="Tycho, Nordostteil", lat=-42.5, lon=-10.5, caption="Tycho, Nordostteil  ·  12 Krater")

# Brand colours (STIL.md)
CREME = (245, 242, 234)
INK = (22, 20, 16)
MUTED = (92, 88, 80)
RULE = (213, 208, 196)
BLUE = (27, 79, 138)
BLUE_INK = (22, 63, 110)
TEAL = (24, 114, 101)
TEAL_INK = (15, 83, 73)
FOOTER_BG = (232, 227, 216)
PAPER = (252, 250, 246)
GOLD = (245, 196, 0)
GOLD_DIM = (201, 158, 20)

DARK_BG = (11, 13, 18)
DARK_BG2 = (8, 10, 15)
CREAM_TEXT = (245, 242, 234)
MUTED_DARK = (154, 149, 138)
TEAL_LIGHT = (86, 201, 176)
BLUE_LIGHT = (122, 168, 214)


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    names = {
        "regular": "Inter-Regular.ttf",
        "medium": "Inter-Medium.ttf",
        "semibold": "Inter-SemiBold.ttf",
        "bold": "Inter-Bold.ttf",
    }
    return ImageFont.truetype(
        str(FONTS / names[weight]),
        size=size,
        layout_engine=ImageFont.Layout.RAQM,
    )


def text_size(draw, text, fnt, features=("tnum",)):
    bbox = draw.textbbox((0, 0), text, font=fnt, features=list(features))
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def glyph_width(draw, ch, fnt, features):
    return draw.textlength(ch, font=fnt, features=list(features))


def tracked_width(draw, text, fnt, tracking, features):
    if not text:
        return 0
    return sum(glyph_width(draw, ch, fnt, features) for ch in text) + tracking * (len(text) - 1)


def ink_bbox(fnt, text, features, anchor="lt"):
    """Ink box of text for a given Pillow anchor. Always pass the anchor explicitly:
    Font.getbbox defaults to a different origin than ImageDraw.textbbox.
    """
    return fnt.getbbox(text, features=list(features), anchor=anchor)


def baseline_from_anchor(y, anchor, fnt, text, features):
    """Convert a Pillow text-anchor y into a baseline y."""
    bb = ink_bbox(fnt, text, features, anchor="ls")  # (l, t, r, b) relative to baseline
    kind = anchor[1] if len(anchor) > 1 else "t"
    if kind == "s":
        return y
    if kind == "t":
        return y - bb[1]
    if kind == "b":
        return y - bb[3]
    if kind == "m":
        return y - (bb[1] + bb[3]) / 2
    return y


def draw_text(draw, xy, text, fnt, fill, anchor="lt", tracking=0, features=("tnum",), stroke_width=0, stroke_fill=None):
    feats = list(features) if features is not None else []
    kwargs = dict(font=fnt, fill=fill, features=feats)
    if stroke_width:
        kwargs["stroke_width"] = stroke_width
        kwargs["stroke_fill"] = stroke_fill
    if not tracking:
        draw.text(xy, text, anchor=anchor, **kwargs)
        return
    # Glyphs share one baseline so commas and hyphens stay on the line (not at the cap height).
    x, y = xy
    total = tracked_width(draw, text, fnt, tracking, feats)
    if anchor.startswith("r"):
        x = x - total
    elif anchor.startswith("m"):
        x = x - total / 2
    baseline = baseline_from_anchor(y, anchor, fnt, text, feats)
    for i, ch in enumerate(text):
        draw.text((x, baseline), ch, anchor="ls", **kwargs)
        x += glyph_width(draw, ch, fnt, feats) + tracking


def load_moon_geometry():
    im = Image.open(MOON_TIF)
    arr = np.array(im)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 128)
    cx = float(xs.min() + xs.max()) / 2.0
    cy = float(ys.min() + ys.max()) / 2.0
    R = ((xs.max() - xs.min()) + (ys.max() - ys.min())) / 4.0
    return im, cx, cy, R


def project(lat, lon, cx, cy, R, sub_lat=SUB_LAT, sub_lon=SUB_LON, pa_deg=POSANGLE):
    lat, lon = math.radians(lat), math.radians(lon)
    lat0, lon0 = math.radians(sub_lat), math.radians(sub_lon)
    dlon = lon - lon0
    x_east = math.cos(lat) * math.sin(dlon)
    y_north = math.cos(lat0) * math.sin(lat) - math.sin(lat0) * math.cos(lat) * math.cos(dlon)
    z = math.sin(lat0) * math.sin(lat) + math.cos(lat0) * math.cos(lat) * math.cos(dlon)
    if z <= 0:
        return None
    # PA is measured from celestial north toward east (screen left); rotate accordingly.
    # Fixed 16 Nov 2026: old sign put south points ~0.25 R too far right (checked against
    # Copernicus, Kepler, Aristarchus, Tycho, Plato, Proclus on frame 7864).
    pa = -math.radians(pa_deg)
    x = x_east * math.cos(pa) + y_north * math.sin(pa)
    y_up = -x_east * math.sin(pa) + y_north * math.cos(pa)
    return cx + R * x, cy - R * y_up, z


def moon_disc_rgba() -> Image.Image:
    """Tight RGBA crop of the disc, with a few pixels of padding."""
    cache = SRC / "moon_disc.png"
    im, cx, cy, R = load_moon_geometry()
    pad = 8
    box = (int(cx - R - pad), int(cy - R - pad), int(math.ceil(cx + R + pad)), int(math.ceil(cy + R + pad)))
    crop = im.crop(box)
    # store offset so we can map projections: proj_xy in original -> crop
    meta = dict(cx=cx, cy=cy, R=R, box=box)
    crop.save(cache)
    np.savez(SRC / "moon_geom.npz", **{k: np.array(v) for k, v in meta.items()})
    return crop, meta


def to_disc(lat, lon, meta, disc_size):
    """Project a lat/lon to pixel coords in a square disc image of disc_size."""
    p = project(lat, lon, meta["cx"], meta["cy"], meta["R"])
    if p is None:
        return None
    x, y, z = p
    bx0, by0 = meta["box"][0], meta["box"][1]
    w = meta["box"][2] - meta["box"][0]
    s = disc_size / w
    return (x - bx0) * s, (y - by0) * s, z


def add_circle_glow(im: Image.Image, radius=10, gain=1.0) -> Image.Image:
    """Soft gold glow around the model's exact #F5C400 circles."""
    arr = np.array(im.convert("RGB"))
    mask = (arr[:, :, 0] == 245) & (arr[:, :, 1] == 196) & (arr[:, :, 2] == 0)
    glow = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    glow[mask] = (245, 196, 0, 255)
    gimg = Image.fromarray(glow, "RGBA")
    g1 = gimg.filter(ImageFilter.GaussianBlur(radius=radius))
    g2 = gimg.filter(ImageFilter.GaussianBlur(radius=radius * 2.2))
    # boost alpha a bit
    def boost(img, factor):
        a = np.array(img)
        a[:, :, 3] = np.clip(a[:, :, 3].astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return Image.fromarray(a, "RGBA")
    g1 = boost(g1, 1.35 * gain)
    g2 = boost(g2, 0.85 * gain)
    base = im.convert("RGBA")
    out = Image.new("RGBA", im.size, (0, 0, 0, 0))
    out = Image.alpha_composite(out, g2)
    out = Image.alpha_composite(out, g1)
    out = Image.alpha_composite(out, base)
    return out


def make_grid_overlay(size, meta, color=(245, 196, 0), max_alpha=48):
    """Faint orthographic lat/lon grid and limb ticks, HUD-style."""
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    # draw at 2x for smoother lines then downscale
    S = size * 2
    hi = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(hi)

    def pts(lats, lons):
        out = []
        for la, lo in zip(lats, lons):
            p = to_disc(la, lo, meta, S)
            if p and p[2] > 0.04:
                out.append((p[0], p[1], p[2]))
            else:
                if out:
                    yield out
                    out = []
        if out:
            yield out

    def draw_path(seq, alpha, width=2):
        if len(seq) < 2:
            return
        col = color + (alpha,)
        xy = [(p[0], p[1]) for p in seq]
        dr.line(xy, fill=col, width=width, joint="curve")

    # meridians every 30°
    for lon in range(-90, 91, 30):
        lats = np.linspace(-89, 89, 240)
        lons = np.full_like(lats, lon)
        for seq in pts(lats, lons):
            a = int(max_alpha * (0.95 if lon == 0 else 0.58))
            draw_path(seq, a, width=3 if lon == 0 else 2)
    # parallels every 30°
    for lat in range(-60, 61, 30):
        lons = np.linspace(-180, 180, 400)
        lats = np.full_like(lons, lat)
        for seq in pts(lats, lons):
            a = int(max_alpha * (1.0 if lat == 0 else 0.58))
            draw_path(seq, a, width=3 if lat == 0 else 2)

    # limb ring
    # disc center in 2x image
    # use geometry
    w = meta["box"][2] - meta["box"][0]
    s = S / w
    # center in disc image
    cx = (meta["cx"] - meta["box"][0]) * s
    cy = (meta["cy"] - meta["box"][1]) * s
    R = meta["R"] * s
    # slightly inside the limb so it sits on the moon, not in the halo
    rr = R - 3
    dr.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=color + (int(min(120, max_alpha*1.6)),), width=3)

    # tick marks around the limb, every 15°, longer every 30°
    for deg in range(0, 360, 15):
        rad = math.radians(deg)
        # 0° is right, 90° is down (image). For a compass, 0 is up.
        # ticks at true-image angles, purely decorative around the limb
        a = -math.radians(deg)  # 0 = up, clockwise visually if we...
        # use standard: deg 0 = up (north-ish, celestial)
        ang = math.radians(deg - 90)  # 0 at up? wait: cos,sin with 0 at right. For 0=up: angle = deg-90
        # celestial 0 at up: x = cx + R*sin(deg), y = cy - R*cos(deg)
        x1 = cx + rr * math.sin(math.radians(deg))
        y1 = cy - rr * math.cos(math.radians(deg))
        length = 22 if deg % 30 == 0 else 11
        x2 = cx + (rr - length) * math.sin(math.radians(deg))
        y2 = cy - (rr - length) * math.cos(math.radians(deg))
        dr.line((x1, y1, x2, y2), fill=color + (int(max_alpha*1.7) if deg % 30 == 0 else int(max_alpha),), width=3 if deg % 30 == 0 else 2)

    overlay = hi.resize((size, size), Image.Resampling.LANCZOS)
    return overlay


def make_scanlines(size, color=(220, 230, 245), period=7, alpha=11):
    """Very faint horizontal scan lines over a square (the moon)."""
    layer = Image.new("RGBA", size if isinstance(size, tuple) else (size, size), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    w, h = layer.size
    for y in range(0, h, period):
        dr.line((0, y, w, y), fill=color + (alpha,), width=1)
    return layer


def reticle(draw, x, y, scale=1.0, color=GOLD, width=3):
    r = 18 * scale
    inner = 6 * scale
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color + (255,), width=max(2, int(width)))
    draw.ellipse((x - inner, y - inner, x + inner, y + inner), fill=color + (255,))
    arm = 28 * scale
    gap = 10 * scale
    w = max(2, int(width))
    # four arms, leaving a gap at the centre
    draw.line((x - arm, y, x - gap, y), fill=color + (255,), width=w)
    draw.line((x + gap, y, x + arm, y), fill=color + (255,), width=w)
    draw.line((x, y - arm, x, y - gap), fill=color + (255,), width=w)
    draw.line((x, y + gap, x, y + arm), fill=color + (255,), width=w)


def rounded_paste(canvas, img, xy, radius, frame_color, frame_w=6, shadow=True, shadow_color=(0, 0, 0, 110)):
    """Paste a square image with a rounded gold frame and optional shadow. Returns the outer box."""
    x, y = xy
    w, h = img.size
    # shadow
    if shadow:
        sh = Image.new("RGBA", (w + 40, h + 40), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        sd.rounded_rectangle((8, 12, 8 + w, 12 + h), radius=radius, fill=shadow_color)
        sh = sh.filter(ImageFilter.GaussianBlur(14))
        canvas.alpha_composite(sh, (x - 8, y - 8))
    # mask
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    canvas.paste(img, (x, y), mask)
    dr = ImageDraw.Draw(canvas)
    # frame, half inside
    hw = frame_w / 2
    dr.rounded_rectangle(
        (x + hw, y + hw, x + w - hw, y + h - hw),
        radius=radius,
        outline=frame_color,
        width=frame_w,
    )
    return (x, y, x + w, y + h)


def leader(draw, start, end, color=GOLD, width=3, elbow="auto"):
    """Straight or single-elbow leader from start (on moon) to a point on the inset frame."""
    x1, y1 = start
    x2, y2 = end
    col = color if len(color) == 4 else color + (255,)
    # prefer a simple two-segment path that approaches the inset from its near side
    # elbow at (x2, y1) if that stays reasonable, else (x1, y2)
    if elbow == "h-then-v":
        mid = (x2, y1)
    elif elbow == "v-then-h":
        mid = (x1, y2)
    else:
        # auto: if mostly horizontal, h-then-v, else v-then-h
        if abs(x2 - x1) >= abs(y2 - y1):
            mid = (x2, y1)
        else:
            mid = (x1, y2)
    draw.line((x1, y1, mid[0], mid[1], x2, y2), fill=col, width=width, joint="curve")


def vignette(size, inner=0.45, outer=1.05, color=(0, 0, 0)):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    r = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((cy - yy) / (h / 2)) ** 2)
    t = np.clip((r - inner) / (outer - inner), 0, 1)
    t = t * t * (3 - 2 * t)
    a = (t * 180).astype(np.uint8)
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[:, :, 0] = color[0]
    layer[:, :, 1] = color[1]
    layer[:, :, 2] = color[2]
    layer[:, :, 3] = a
    return Image.fromarray(layer, "RGBA")


def moon_glow(disc: Image.Image, radius=28, alpha=90) -> Image.Image:
    """Soft bloom behind the disc."""
    a = disc.split()[-1].filter(ImageFilter.GaussianBlur(radius))
    a = Image.eval(a, lambda v: min(255, int(v * alpha / 140)))
    glow = Image.new("RGBA", disc.size, GOLD + (0,))
    # warm pale glow, not gold-heavy
    glow = Image.new("RGBA", disc.size, (230, 220, 190, 0))
    glow.putalpha(a)
    return glow


def prepare():
    disc, meta = moon_disc_rgba()
    hoch = add_circle_glow(Image.open(HOCHLAND_SRC), radius=11, gain=1.15)
    tych = add_circle_glow(Image.open(TYCHO_SRC), radius=11, gain=1.15)
    hoch.save(SRC / "palus_somni_glow.png")
    tych.save(SRC / "tycho_glow.png")
    return disc, meta, hoch, tych


def compose_45(theme: str, disc, meta, hoch, tych) -> Image.Image:
    """4:5 title, 2160×2700, rendered at 2× and downsampled."""
    W, H = 2160, 2700
    scale = 2
    w, h = W * scale, H * scale
    dark = theme == "dunkel"

    if dark:
        canvas = Image.new("RGBA", (w, h), DARK_BG + (255,))
        # subtle radial lift behind the moon
        lift = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lift)
        # we'll add a radial later via vignette inverse
        yy, xx = np.mgrid[0:h, 0:w]
        # warm-cool gradient: slightly deeper at bottom
        grad = np.zeros((h, w, 4), dtype=np.uint8)
        t = (yy / h).astype(np.float32)
        grad[:, :, 0] = (11 + t * 4).astype(np.uint8)
        grad[:, :, 1] = (13 + (1 - t) * 3).astype(np.uint8)
        grad[:, :, 2] = (20 + (1 - t) * 8).astype(np.uint8)
        grad[:, :, 3] = 255
        canvas = Image.fromarray(grad, "RGBA")
        kicker_c = BLUE_LIGHT
        hero_c = CREAM_TEXT
        sub_c = MUTED_DARK
        nasa_c = TEAL_LIGHT
        frame_c = GOLD
        footer_bg = (16, 18, 24, 255)
        footer_c = (140, 136, 126)
        footer_strong = (180, 176, 166)
        topbar = None
    else:
        canvas = Image.new("RGBA", (w, h), CREME + (255,))
        kicker_c = BLUE
        hero_c = INK
        sub_c = MUTED
        nasa_c = TEAL
        frame_c = GOLD
        footer_bg = FOOTER_BG + (255,)
        footer_c = MUTED
        footer_strong = MUTED
        topbar = BLUE

    dr = ImageDraw.Draw(canvas)

    # top bar (creme only)
    if topbar:
        dr.rectangle((0, 0, w, 16 * scale), fill=topbar)

    # --- layout in 1x, multiply ---
    s = scale
    # kicker
    kicker = "KI-EXPERIMENTE"
    fk = font("semibold", 22 * s)
    draw_text(dr, (80 * s, 56 * s), kicker, fk, kicker_c, anchor="lt", tracking=int(4.2 * s), features=())

    # hero numbers
    f_hero = font("bold", 168 * s)
    f_sub = font("medium", 34 * s)
    f_nasa = font("bold", 96 * s)
    f_nasa_sub = font("medium", 32 * s)

    hero_y = 108 * s
    draw_text(dr, (80 * s, hero_y), "0,2546", f_hero, hero_c, anchor="lt", tracking=int(-3 * s))
    hb = ink_bbox(f_hero, "0,2546", ["tnum"], anchor="lt")
    sub_y = hero_y + hb[3] + 10 * s
    draw_text(dr, (80 * s, sub_y), "im Mittel", f_sub, sub_c, anchor="lt")

    # NASA number on the right of the top band, aligned with the hero number
    nx = (W - 80) * s
    nasa_y = hero_y + 8 * s
    draw_text(dr, (nx, nasa_y), "0,2581", f_nasa, nasa_c, anchor="rt", tracking=int(-2 * s))
    nb = ink_bbox(f_nasa, "0,2581", ["tnum"], anchor="lt")
    draw_text(dr, (nx, nasa_y + nb[3] + 10 * s), "laut NASA und IBM", f_nasa_sub, sub_c, anchor="rt", features=())

    # --- moon ---
    D = 1480 * s
    moon_cx, moon_cy = 810 * s, 1300 * s
    # resize disc
    disc_r = disc.resize((D, D), Image.Resampling.LANCZOS)
    # glow behind (dark theme) or soft shadow (creme)
    mx, my = int(moon_cx - D / 2), int(moon_cy - D / 2)
    if dark:
        glow = moon_glow(disc_r, radius=48, alpha=110)
        # expand canvas for glow
        pad = 60
        g2 = Image.new("RGBA", (D + pad * 2, D + pad * 2), (0, 0, 0, 0))
        g2.paste(glow, (pad, pad), glow)
        g2 = g2.filter(ImageFilter.GaussianBlur(8))
        canvas.alpha_composite(g2, (mx - pad, my - pad))
    else:
        sh = Image.new("RGBA", (D + 80, D + 80), (0, 0, 0, 0))
        # disc-shaped shadow
        sm = disc_r.split()[-1].filter(ImageFilter.GaussianBlur(28))
        sh_col = Image.new("RGBA", (D, D), (40, 35, 25, 0))
        sh_col.putalpha(Image.eval(sm, lambda v: min(255, int(v * 0.35))))
        canvas.alpha_composite(sh_col, (mx + 10, my + 22))

    canvas.alpha_composite(disc_r, (mx, my))

    # HUD grid + limb ticks + scanlines, clipped to the disc
    grid = make_grid_overlay(D, meta, color=GOLD if dark else BLUE, max_alpha=72 if dark else 52)
    # mask with disc alpha so lines don't sit outside the limb
    g = np.array(grid)
    da = np.array(disc_r.split()[-1]).astype(np.float32) / 255.0
    g[:, :, 3] = (g[:, :, 3].astype(np.float32) * da).astype(np.uint8)
    grid = Image.fromarray(g, "RGBA")
    # slightly stronger on dark
    canvas.alpha_composite(grid, (mx, my))

    if dark:
        sl = make_scanlines((D, D), color=(210, 220, 240), period=8, alpha=10)
        sa = np.array(sl)
        sa[:, :, 3] = (sa[:, :, 3].astype(np.float32) * da).astype(np.uint8)
        sl = Image.fromarray(sa, "RGBA")
        canvas.alpha_composite(sl, (mx, my))

    # feature positions on the canvas
    def feat(spec):
        p = to_disc(spec["lat"], spec["lon"], meta, D)
        return mx + p[0], my + p[1]

    hx, hy = feat(HOCHLAND)
    tx, ty = feat(TYCHO)

    # --- insets (right column). Footer height is a hard ceiling for the caption.
    fh = 108 * s
    fy = h - fh
    inset = 520 * s
    hoch_i = hoch.resize((inset, inset), Image.Resampling.LANCZOS)
    tych_i = tych.resize((inset, inset), Image.Resampling.LANCZOS)

    cap_space = 60 * s  # caption + padding, must sit fully above the footer
    hx0 = 1588 * s
    hy0 = 660 * s
    # Tycho: same right column, lower, aligned near the feature and above the footer
    tx0 = hx0
    ty0 = fy - cap_space - inset

    box_h = rounded_paste(
        canvas, hoch_i, (int(hx0), int(hy0)),
        radius=10 * s, frame_color=frame_c, frame_w=6 * s if dark else 5 * s,
        shadow=True, shadow_color=(0, 0, 0, 160) if dark else (40, 35, 25, 70),
    )
    box_t = rounded_paste(
        canvas, tych_i, (int(tx0), int(ty0)),
        radius=10 * s, frame_color=frame_c, frame_w=6 * s if dark else 5 * s,
        shadow=True, shadow_color=(0, 0, 0, 160) if dark else (40, 35, 25, 70),
    )

    f_cap = font("medium", 28 * s)
    cap_col = CREAM_TEXT if dark else INK
    # caption with a small gold swatch (circle)
    def caption(box, text, side="below"):
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2
        y = y1 + 14 * s
        # swatch
        sw = 11 * s
        # measure text + swatch
        tw, th = text_size(dr, text, f_cap, features=())
        total = tw + 18 * s + sw * 2
        sx = cx - total / 2
        cy = y + th * 0.35
        dr.ellipse((sx, cy - sw, sx + 2 * sw, cy + sw), outline=GOLD, width=max(3, int(3.2 * s)))
        draw_text(dr, (sx + 2 * sw + 14 * s, y), text, f_cap, cap_col, anchor="lt", features=())

    caption(box_h, HOCHLAND["caption"])
    caption(box_t, TYCHO["caption"])

    # reticles + leaders
    # draw on a transparent layer so we can control compositing
    lines = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lines)
    # dark halo under gold lines for contrast on the bright moon
    def fat_leader(start, end, elbow, width):
        # outline
        leader(ld, start, end, color=(20, 16, 4, 200), width=width + 2 * s, elbow=elbow)
        leader(ld, start, end, color=GOLD + (255,), width=width, elbow=elbow)

    # Hochland: from reticle right to the left-middle of the inset
    h_end = (box_h[0], (box_h[1] + box_h[3]) / 2)
    fat_leader((hx, hy), h_end, elbow="h-then-v", width=max(3, int(2.6 * s)))
    # Tycho: down past the limb, then right to the left side of the inset
    y_elbow = (box_t[1] + box_t[3]) / 2
    t_end = (box_t[0], y_elbow)
    fat_leader((tx, ty), t_end, elbow="v-then-h", width=max(3, int(2.6 * s)))

    # reticles on top of the lines (after a dark under-stroke)
    def fat_reticle(x, y):
        # dark under
        reticle(ld, x, y, scale=s * 1.05, color=(20, 16, 4), width=max(3, int(4 * s)))
        reticle(ld, x, y, scale=s, color=GOLD, width=max(2, int(3 * s)))

    fat_reticle(hx, hy)
    fat_reticle(tx, ty)
    canvas.alpha_composite(lines)

    # --- footer (fy already set so captions clear it) ---
    dr.rectangle((0, fy, w, h), fill=footer_bg)
    f_foot = font("regular", 22 * s)
    f_foot_b = font("semibold", 22 * s)
    footer_1 = "Quellen.  Mondscheibe: NASA Scientific Visualization Studio, 24. Nov. 2026, 15:00 UTC (NASA/GSFC)."
    footer_2 = "Ausschnitte: LROC WAC-Mosaik 100 m (NASA/GSFC/ASU, USGS), gemeinfrei.  Kreise: offenes NASA-IBM-Modell, eigene Nachstellung, Schwelle 0,7."
    # draw "Quellen." in semibold, rest regular — simplest: two lines, first starts with Quellen.
    # manual split
    pad_x = 80 * s
    yf = fy + 28 * s
    # line 1
    q = "Quellen.  "
    qw = dr.textbbox((0, 0), q, font=f_foot_b, features=[])[2]
    draw_text(dr, (pad_x, yf), q, f_foot_b, footer_strong, anchor="lt", features=())
    rest1 = "Mondscheibe: NASA Scientific Visualization Studio, 24. Nov. 2026, 15:00 UTC (NASA/GSFC)."
    draw_text(dr, (pad_x + qw, yf), rest1, f_foot, footer_c, anchor="lt", features=())
    draw_text(dr, (pad_x, yf + 36 * s), footer_2, f_foot, footer_c, anchor="lt", features=())

    # thin rule above footer
    rule_c = (50, 52, 60) if dark else RULE
    dr.line((0, fy, w, fy), fill=rule_c, width=2)

    # downsample
    out = canvas.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    return out


def compose_link(theme: str, disc, meta, hoch, tych) -> Image.Image:
    """1200×627 link-preview cover, rendered at 2×."""
    W, H = 1200, 627
    scale = 2
    w, h = W * scale, H * scale
    dark = theme == "dunkel"
    s = scale

    if dark:
        yy, xx = np.mgrid[0:h, 0:w]
        t = (yy / h).astype(np.float32)
        grad = np.zeros((h, w, 4), dtype=np.uint8)
        grad[:, :, 0] = (11 + t * 4).astype(np.uint8)
        grad[:, :, 1] = (13 + (1 - t) * 3).astype(np.uint8)
        grad[:, :, 2] = (20 + (1 - t) * 8).astype(np.uint8)
        grad[:, :, 3] = 255
        canvas = Image.fromarray(grad, "RGBA")
        kicker_c = BLUE_LIGHT
        hero_c = CREAM_TEXT
        sub_c = MUTED_DARK
        nasa_c = TEAL_LIGHT
        frame_c = GOLD
        footer_c = (140, 136, 126)
        topbar = None
    else:
        canvas = Image.new("RGBA", (w, h), CREME + (255,))
        kicker_c = BLUE
        hero_c = INK
        sub_c = MUTED
        nasa_c = TEAL
        frame_c = GOLD
        footer_c = MUTED
        topbar = BLUE

    dr = ImageDraw.Draw(canvas)
    if topbar:
        dr.rectangle((0, 0, w, 10 * s), fill=topbar)

    # Geometry at 1x (1200×627), then *s.
    # Moon on the left, full disc, leaving a gutter so the leader never crosses type.
    footer_h = 36 * s
    fy = h - footer_h
    topbar_h = 10 * s if topbar else 0
    D = int(fy - topbar_h - 18 * s)
    # Leave room for the right column: moon + gutter + 520 of column + margin <= 1200
    # At 1x: D max ≈ 555 so right column has ~600 px.
    D = int(min(D, 552 * s))
    mx = int(16 * s)
    my = int(topbar_h + ((fy - topbar_h) - D) / 2)
    moon_cx, moon_cy = mx + D / 2, my + D / 2

    disc_r = disc.resize((D, D), Image.Resampling.LANCZOS)
    if dark:
        glow = moon_glow(disc_r, radius=36, alpha=110)
        pad = 40
        g2 = Image.new("RGBA", (D + pad * 2, D + pad * 2), (0, 0, 0, 0))
        g2.paste(glow, (pad, pad), glow)
        canvas.alpha_composite(g2, (mx - pad, my - pad))
    else:
        sm = disc_r.split()[-1].filter(ImageFilter.GaussianBlur(20))
        sh_col = Image.new("RGBA", (D, D), (40, 35, 25, 0))
        sh_col.putalpha(Image.eval(sm, lambda v: min(255, int(v * 0.32))))
        canvas.alpha_composite(sh_col, (mx + 6, my + 12))
    canvas.alpha_composite(disc_r, (mx, my))

    grid = make_grid_overlay(D, meta, color=GOLD if dark else BLUE, max_alpha=68 if dark else 48)
    g = np.array(grid)
    da = np.array(disc_r.split()[-1]).astype(np.float32) / 255.0
    g[:, :, 3] = (g[:, :, 3].astype(np.float32) * da).astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(g, "RGBA"), (mx, my))
    if dark:
        sl = make_scanlines((D, D), color=(210, 220, 240), period=8, alpha=10)
        sa = np.array(sl)
        sa[:, :, 3] = (sa[:, :, 3].astype(np.float32) * da).astype(np.uint8)
        canvas.alpha_composite(Image.fromarray(sa, "RGBA"), (mx, my))

    # Right column, after a clear gutter
    gutter = 40 * s
    col_x = mx + D + gutter
    col_r = (W - 24) * s

    fk = font("semibold", 15 * s)
    kicker_y = (28 * s) if not topbar else (36 * s)
    draw_text(dr, (col_x, kicker_y), "KI-EXPERIMENTE", fk, kicker_c, anchor="lt", tracking=int(3.0 * s), features=())

    f_hero = font("bold", 86 * s)
    f_sub = font("medium", 20 * s)
    f_nasa = font("bold", 52 * s)
    f_nasa_sub = font("medium", 18 * s)
    y0 = kicker_y + 32 * s
    draw_text(dr, (col_x, y0), "0,2546", f_hero, hero_c, anchor="lt", tracking=int(-2 * s))
    hb = ink_bbox(f_hero, "0,2546", ["tnum"], anchor="lt")
    sub_y = y0 + hb[3] + 6 * s
    draw_text(dr, (col_x, sub_y), "im Mittel", f_sub, sub_c, anchor="lt")
    y1 = sub_y + 30 * s
    draw_text(dr, (col_x, y1), "0,2581", f_nasa, nasa_c, anchor="lt", tracking=int(-1.2 * s))
    nb = ink_bbox(f_nasa, "0,2581", ["tnum"], anchor="lt")
    nasa_sub_y = y1 + nb[3] + 5 * s
    draw_text(dr, (col_x, nasa_sub_y), "laut NASA und IBM", f_nasa_sub, sub_c, anchor="lt", features=())

    # Inset under the type, caption fully above the footer
    cap_space = 32 * s
    available = fy - cap_space - (nasa_sub_y + 26 * s)
    inset = int(min(230 * s, available))
    hoch_i = hoch.resize((inset, inset), Image.Resampling.LANCZOS)
    ix = int(col_x)
    iy = int(fy - cap_space - inset)
    # If the inset would overlap the NASA sublabel, push it down only if room (shrink already applied)
    min_iy = nasa_sub_y + 22 * s
    if iy < min_iy:
        iy = int(min_iy)
        inset2 = int(fy - cap_space - iy)
        if inset2 < inset:
            inset = max(160 * s, inset2)
            hoch_i = hoch.resize((inset, inset), Image.Resampling.LANCZOS)
    box_h = rounded_paste(
        canvas, hoch_i, (ix, int(iy)),
        radius=7 * s, frame_color=frame_c, frame_w=4 * s,
        shadow=True, shadow_color=(0, 0, 0, 150) if dark else (40, 35, 25, 60),
    )
    f_cap = font("medium", 15 * s)
    cap = HOCHLAND["caption"]
    cap_col = CREAM_TEXT if dark else INK
    below = box_h[3] + 8 * s
    tw, th = text_size(dr, cap, f_cap, features=())
    sw = 6 * s
    cxp = (box_h[0] + box_h[2]) / 2
    sx = cxp - (tw + 14 * s + 2 * sw) / 2
    cyy = below + th * 0.35
    if below + th < fy - 2 * s:
        dr.ellipse((sx, cyy - sw, sx + 2 * sw, cyy + sw), outline=GOLD, width=max(2, int(2.2 * s)))
        draw_text(dr, (sx + 2 * sw + 8 * s, below), cap, f_cap, cap_col, anchor="lt", features=())

    # Reticle + leader routed through the gutter (never through the type)
    def feat(spec):
        p = to_disc(spec["lat"], spec["lon"], meta, D)
        return mx + p[0], my + p[1]

    hx, hy = feat(HOCHLAND)
    lines = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lines)
    gutter_x = mx + D + (gutter * 0.45)
    # path: reticle -> (gutter_x, hy) -> (gutter_x, mid_inset) -> left of inset
    mid_y = (box_h[1] + box_h[3]) / 2
    path = [(hx, hy), (gutter_x, hy), (gutter_x, mid_y), (box_h[0], mid_y)]
    ld.line(path, fill=(20, 16, 4, 200), width=int(4 * s), joint="curve")
    ld.line(path, fill=GOLD + (255,), width=int(2.4 * s), joint="curve")
    reticle(ld, hx, hy, scale=s * 0.7, color=(20, 16, 4), width=int(3.4 * s))
    reticle(ld, hx, hy, scale=s * 0.65, color=GOLD, width=int(2.4 * s))
    canvas.alpha_composite(lines)

    # compact footer
    if dark:
        dr.rectangle((0, fy, w, h), fill=(16, 18, 24, 255))
    else:
        dr.rectangle((0, fy, w, h), fill=FOOTER_BG + (255,))
    f_foot = font("regular", 12 * s)
    foot = "Quellen. Mondscheibe: NASA SVS, 24. Nov. 2026 (NASA/GSFC).  Ausschnitte: LROC WAC-Mosaik (NASA/GSFC/ASU, USGS).  Kreise: offenes NASA-IBM-Modell, eigene Nachstellung, Schwelle 0,7."
    draw_text(dr, (20 * s, fy + 10 * s), foot, f_foot, footer_c, anchor="lt", features=())

    out = canvas.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    return out


def main():
    print("preparing moon disc and glowing insets…")
    disc, meta, hoch, tych = prepare()
    print(f"  disc {disc.size}  meta R={meta['R']:.1f}")
    for theme in ("creme", "dunkel"):
        print(f"composing 4:5 {theme}…")
        im = compose_45(theme, disc, meta, hoch, tych)
        path = OUT / f"titel-{theme}-4x5.png"
        im.save(path, "PNG", optimize=True)
        print(f"  wrote {path} {im.size}")
        print(f"composing 1200x627 {theme}…")
        im2 = compose_link(theme, disc, meta, hoch, tych)
        path2 = OUT / f"titel-{theme}-1200x627.png"
        im2.save(path2, "PNG", optimize=True)
        print(f"  wrote {path2} {im2.size}")
        # previews for QA
        prev = im.copy()
        prev.thumbnail((1080, 1350), Image.Resampling.LANCZOS)
        prev.save(OUT / f"_preview-{theme}-4x5.png")
        prev2 = im2.copy()
        prev2.thumbnail((1200, 627), Image.Resampling.LANCZOS)
        prev2.save(OUT / f"_preview-{theme}-link.png")
    print("done")


if __name__ == "__main__":
    main()
