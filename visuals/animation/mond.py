"""Moon animation (mond.gif): seven LROC WAC mosaic crops on a rotating Moon, with crater
detections of the seed-46 checkpoint (score >= 0.7, circle fully inside the 512 px crop).

Inputs (not in this repository): visuals/crops/*.npy (mosaic crops, see crops/*.json for
coordinates), visuals/detections/*_preds.npz (archived predictions from run_detect.py) and the
NASA SVS CGI Moon Kit texture lroc_color_poles_4k.tif (https://svs.gsfc.nasa.gov/4720/).

Usage:
  python mond.py --texture path/to/lroc_color_poles_4k.tif --fps 15 --out frames_15
  ffmpeg -framerate 15 -i frames_15/%04d.png ... mond.gif   # palettegen/paletteuse
"""
import argparse, json, math, sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

V = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep   # visuals/
S = 2
W = H = 1080 * S
SLOT = 2.8
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--texture', default=os.environ.get('MOON_TEXTURE'), required=os.environ.get('MOON_TEXTURE') is None,
                help='path to lroc_color_poles_4k.tif (or set MOON_TEXTURE)')
ap.add_argument('--fps', type=int, default=10)
ap.add_argument('--out', default=None, help='output directory for PNG frames (default: mond_frames_<fps>)')
ap.add_argument('--frames', default=None, help='comma-separated frame indices to render (default: all)')
ARGS = ap.parse_args()
FPS = ARGS.fps
OUT = os.path.join(ARGS.out or f'mond_frames_{FPS}', '')
os.makedirs(OUT, exist_ok=True)
DEG_PX = 100 / 1737400 * 180 / math.pi          # mosaic pixel in degrees (both axes, simple cylindrical)
HALF = 256 * DEG_PX                              # crop half-size in degrees of lat AND lon

tex = np.asarray(Image.open(ARGS.texture).convert('L'), dtype=np.float32) / 255.0
TH, TW = tex.shape

# path order around the near side; loops back to the first
REG = [
    ('mare-scherben', 'Mare Imbrium'),
    ('copernicus-rand', 'Copernicus, Nordostteil'),
    ('tycho-umfeld', 'Tycho, Nordostteil'),
    ('clavius-zentriert', 'Clavius, Kraterboden'),
    ('theophilus-umfeld', 'Südliches Hochland'),
    ('petavius-umfeld', 'Petavius, Kratermitte'),
    ('hochland-ost', 'Palus Somni'),
]
THRESH = 0.7

def inside(cx, cy, r, Wc=512):
    # counting rule: circle (centre of box, mean half-side) lies fully inside the 512 px crop
    return cx - r >= 0 and cy - r >= 0 and cx + r <= Wc and cy + r <= Wc

def stretch(g):
    a = g.astype(np.float32); v = a[a > 0]
    lo, hi = np.percentile(v, [1, 99])
    return np.clip((a - lo) / (hi - lo), 0, 1)

regions = []
for key, label in REG:
    meta = json.load(open(V + f'crops/{key}.json'))
    z = np.load(V + f'detections/{key}_preds.npz')
    circ = []; n07 = 0
    for b, s in zip(z['boxes'], z['scores']):
        if s < THRESH: continue
        n07 += 1
        x1, y1, x2, y2 = map(float, b)
        cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, 0.25 * ((x2 - x1) + (y2 - y1))
        if inside(cx, cy, r): circ.append((cx, cy, r))
    circ.sort(key=lambda c: c[1])
    regions.append(dict(key=key, label=label, lat=meta['center_lat'], lon=meta['center_lon'], n07=n07,
                        img=stretch(np.load(V + f'crops/{key}.npy')), circ=circ))
NS = len(regions)
TOTAL = sum(len(r['circ']) for r in regions)
T = SLOT * NS
NF = int(round(T * FPS))
EXPECT = {'mare-scherben': 9, 'copernicus-rand': 9, 'tycho-umfeld': 12, 'clavius-zentriert': 17,
          'theophilus-umfeld': 105, 'petavius-umfeld': 30, 'hochland-ost': 51}
assert {r['key']: len(r['circ']) for r in regions} == EXPECT and TOTAL == 233, [(r['key'], len(r['circ'])) for r in regions]

def view_of(reg):            # camera centre for a region: exact lon, latitude a bit damped (keeps limb in view)
    return 0.8 * reg['lat'], reg['lon']

def rot(lat_c, lonc):
    a, b = math.radians(lonc), math.radians(lat_c)
    Ry = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return Rx @ Ry

def P(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([math.cos(la) * math.sin(lo), math.sin(la), math.cos(la) * math.cos(lo)])

CX, CY, R = 352 * S, 590 * S, 290 * S
yy, xx = np.mgrid[0:2 * R + 2, 0:2 * R + 2].astype(np.float32)
nx = (xx - R - 0.5) / R; ny = -(yy - R - 0.5) / R
rr = nx * nx + ny * ny
nz = np.sqrt(np.clip(1 - rr, 0, 1))
edge_alpha = np.clip((1.0 - np.sqrt(rr)) * R, 0, 1)
light = np.array([-0.45, 0.35, 0.82]); light /= np.linalg.norm(light)
shade = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0, 1)
shade = 0.42 + 0.58 * shade ** 0.8
V3 = np.stack([nx, ny, nz], -1)

def render_sphere(latc, lonc):
    w = V3 @ rot(latc, lonc)
    lat = np.degrees(np.arcsin(np.clip(w[..., 1], -1, 1)))
    lon = np.degrees(np.arctan2(w[..., 0], w[..., 2]))
    fx = (lon + 180) / 360 * TW - 0.5; fy = (90 - lat) / 180 * TH - 0.5
    x0 = np.floor(fx).astype(int); y0 = np.floor(fy).astype(int)
    ax = fx - x0; ay = fy - y0
    x0m, x1m = x0 % TW, (x0 + 1) % TW
    y0c, y1c = np.clip(y0, 0, TH - 1), np.clip(y0 + 1, 0, TH - 1)
    t = (tex[y0c, x0m] * (1 - ax) + tex[y0c, x1m] * ax) * (1 - ay) + (tex[y1c, x0m] * (1 - ax) + tex[y1c, x1m] * ax) * ay
    v = np.clip((t - 0.12) / 0.66, 0, 1) * shade * edge_alpha
    return (v * 255).astype(np.uint8), edge_alpha

def project(lat, lon, latc, lonc):
    v = rot(latc, lonc) @ P(lat, lon)
    return CX + v[0] * R, CY - v[1] * R, v[2]

def footprint(reg, latc, lonc):
    # true crop outline: +-256 mosaic pixels = +-HALF degrees in lat and in lon (simple cylindrical grid)
    pts = []
    for la, lo in [(HALF, -HALF), (HALF, HALF), (-HALF, HALF), (-HALF, -HALF)]:
        x, y, _ = project(reg['lat'] + la, reg['lon'] + lo, latc, lonc)
        pts.append((x, y))
    return pts

def ease(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)

def orientation(k, p):
    cur = view_of(regions[k]); prev = view_of(regions[(k - 1) % NS])
    e = ease(p / 0.30)
    return prev[0] + (cur[0] - prev[0]) * e, prev[1] + (cur[1] - prev[1]) * e

IX, IY, IS = 700 * S, 318 * S, 340 * S
inset_cache = {}
def inset_img(reg):
    if reg['key'] not in inset_cache:
        inset_cache[reg['key']] = Image.fromarray((reg['img'] * 255).astype(np.uint8)).resize((IS, IS), Image.Resampling.LANCZOS).convert('RGB')
    return inset_cache[reg['key']]

F_H = font('b', 50 * S); F_SUB = font('m', 26 * S)
F_LAB = font('b', 30 * S); F_CO = font('m', 23 * S); F_NUM = font('b', 64 * S); F_NUMS = font('m', 26 * S)

def fmt_coord(lat, lon):
    a = f"{abs(lat):.2f}".replace('.', ',') + ('° N' if lat >= 0 else '° S')
    b = f"{abs(lon):.2f}".replace('.', ',') + ('° O' if lon >= 0 else '° W')
    return a + ' · ' + b

HEAD1, HEAD2 = 'Mondkrater, erkannt mit dem', 'offenen NASA-IBM-Modell'
SUB = f'Eigene Nachstellung · {NS} Mosaik-Ausschnitte · {TOTAL} Krater markiert'
FOOT = ['Modell: NASA-IBM Lunar Foundation Model, eigenes Feintuning (Lauf 46), Schwelle 0,7.',
        'Keine Prüfung durch NASA oder IBM. Ausschnitte: 512 × 512 Pixel, 100 m/Pixel.',
        'Bilder: LROC WAC (NASA/GSFC/Arizona State University), USGS-Mosaik, NASA SVS']

def draw_header(dr):
    dr.rectangle([0, 0, W, 8 * S], fill=BLUE)
    tracked(dr, (56 * S, 32 * S), 'KI-EXPERIMENTE', font('s', 19 * S), BLUE_LIGHT, tracking=4 * S)
    dr.text((56 * S, 62 * S), HEAD1, font=F_H, fill=CREAM)
    dr.text((56 * S, 122 * S), HEAD2, font=F_H, fill=CREAM)
    dr.text((56 * S, 194 * S), SUB, font=F_SUB, fill=MUTED)

def frame(i):
    u = i / NF
    k = int(u * NS) % NS; p = u * NS - int(u * NS)
    cur = regions[k]
    latc, lonc = orientation(k, p)
    img = Image.new('RGB', (W, H), DARK_BG)
    g, a = render_sphere(latc, lonc)
    sph = Image.fromarray(g).filter(ImageFilter.GaussianBlur(1.3)).convert('RGB')
    img.paste(sph, (CX - R - 1, CY - R - 1), Image.fromarray((a * 255).astype(np.uint8)))
    dr = ImageDraw.Draw(img, 'RGBA')
    draw_header(dr)
    env = ease((p - 0.28) / 0.10) * (1 - ease((p - 0.965) / 0.035))
    for reg in regions:
        if reg is cur: continue
        x, y, z = project(reg['lat'], reg['lon'], latc, lonc)
        if z < 0.15: continue
        dr.polygon(footprint(reg, latc, lonc), outline=GOLD_DIM + (255,), width=2 * S)
        dr.ellipse([x - 11 * S, y - 11 * S, x + 11 * S, y + 11 * S], outline=GOLD_DIM + (200,), width=2 * S)
    x, y, z = project(cur['lat'], cur['lon'], latc, lonc)
    al = max(env, 0.35 if z > 0.15 else 0)
    rg = 18 * S
    dr.ellipse([x - rg, y - rg, x + rg, y + rg], outline=GOLD + (int(255 * al),), width=3 * S)
    dr.polygon(footprint(cur, latc, lonc), fill=GOLD + (int(200 * al),), outline=GOLD + (int(255 * al),))
    if env > 0.01:
        dr.line([(x + rg, y), (IX - 2 * S, IY + IS // 2)], fill=GOLD + (int(200 * env),), width=2 * S)
    # inset with scan
    ins = inset_img(cur).copy()
    di = ImageDraw.Draw(ins, 'RGBA')
    sc = IS / 512
    scan = ease((p - 0.32) / 0.48) * 512 if p > 0.32 else -1
    n_on = 0
    for (cx, cy, r) in cur['circ']:
        if cy <= scan:
            n_on += 1
            age = (scan - cy) / 512 / 0.12
            rr_ = max(r * sc * (1 + 0.6 * (1 - ease(age))), 3 * S)
            box = [cx * sc - rr_, cy * sc - rr_, cx * sc + rr_, cy * sc + rr_]
            di.ellipse(box, outline=(40, 32, 0, 255), width=4 * S)
            di.ellipse(box, outline=GOLD + (255,), width=2 * S + 1)
    if 0 <= scan < 512 and p < 0.80:
        di.line([0, scan * sc, IS, scan * sc], fill=GOLD + (150,), width=2 * S)
    bg = img.crop((IX, IY, IX + IS, IY + IS))
    img.paste(Image.composite(ins, bg, Image.new('L', (IS, IS), int(255 * env))), (IX, IY))
    dr = ImageDraw.Draw(img, 'RGBA')
    dr.rectangle([IX - 2 * S, IY - 2 * S, IX + IS + 2 * S, IY + IS + 2 * S], outline=GOLD + (int(255 * env),), width=3 * S)
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); dl = ImageDraw.Draw(lay)
    ty = IY + IS + 22 * S
    dl.text((IX, ty), cur['label'], font=F_LAB, fill=CREAM)
    dl.text((IX, ty + 44 * S), fmt_coord(cur['lat'], cur['lon']), font=F_CO, fill=MUTED)
    dl.text((IX, ty + 86 * S), str(n_on), font=F_NUM, fill=GOLD)
    nw = dl.textlength(str(n_on), font=F_NUM)
    dl.text((IX + nw + 14 * S, ty + 122 * S), 'Krater', font=F_NUMS, fill=CREAM)
    lay.putalpha(lay.getchannel('A').point(lambda v: int(v * env)))
    img = Image.alpha_composite(img.convert('RGBA'), lay).convert('RGB')
    dr = ImageDraw.Draw(img, 'RGBA')
    fy = H - (26 + 26 * len(FOOT)) * S
    dr.line([56 * S, fy - 14 * S, W - 56 * S, fy - 14 * S], fill=(40, 43, 52), width=S)
    for j, l in enumerate(FOOT):
        dr.text((56 * S, fy + j * 26 * S), l, font=font('r', 18 * S), fill=MUTED)
    out = np.asarray(img.resize((W // S, H // S), Image.Resampling.LANCZOS)).astype(np.int16)
    yy1, xx1 = np.mgrid[0:H // S, 0:W // S]
    m = (((xx1 - CX / S) ** 2 + (yy1 - CY / S) ** 2) <= (R / S + 1) ** 2) & ((out.max(-1) - out.min(-1)) < 10)
    L = 26
    gq = np.round(out[..., 1] / 255 * (L - 1)) / (L - 1) * 255
    for c in range(3):
        ch = out[..., c]; ch[m] = gq[m]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))

if __name__ == '__main__':
    print({r['key']: (r['n07'], len(r['circ'])) for r in regions}, TOTAL, 'frames', NF)
    only = [int(x) for x in ARGS.frames.split(',')] if ARGS.frames else range(NF)
    for i in only:
        frame(i).save(OUT + f'{i:04d}.png')
