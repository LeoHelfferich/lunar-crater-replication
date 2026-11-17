"""Clean publication PNGs: moon + gold circles, no titles, 2160px."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
CROPS = HERE / 'crops'
DETS = HERE / 'detections'
FINAL = HERE / 'final'  # output dir for clean renders
FINAL.mkdir(exist_ok=True)

GOLD = (245, 196, 0, 255)
SIZE = 2160  # 4x the 512 native
SCALE = SIZE / 512
LW = 7
THRESH = {
    'clavius-zentriert': 0.7,
    'hochland-ost': 0.7,
    'tycho-umfeld': 0.7,
    'copernicus-rand': 0.7,
}
# published file names (crop ids keep their original names; labels corrected 16 Nov 2026)
OUTNAME = {
    'clavius-zentriert': 'krater-clavius.png',
    'hochland-ost': 'krater-palus-somni.png',
    'tycho-umfeld': 'krater-tycho-nordostteil.png',
    'copernicus-rand': 'krater-copernicus-nordostteil.png',
}


def stretch(gray, out=SIZE):
    a = gray.astype(np.float32)
    valid = a[a > 0]
    lo, hi = np.percentile(valid, [1, 99]) if valid.size else (0, 255)
    if hi <= lo:
        hi = lo + 1
    x = np.clip((a - lo) / (hi - lo), 0, 1)
    im = Image.fromarray((x * 255).astype(np.uint8), mode='L')
    return im.resize((out, out), Image.Resampling.LANCZOS).convert('RGB')


def keep_circle(cx, cy, r, W=512, max_outside=0.0):
    """Keep only circles that lie fully inside the crop (strict edge rule)."""
    # fraction of the axis-aligned bbox that lies inside
    x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
    ix1, iy1, ix2, iy2 = max(0, x1), max(0, y1), min(W, x2), min(W, y2)
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    iarea = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return (iarea / area) >= (1 - max_outside)


def render(name, thresh):
    gray = np.load(CROPS / f'{name}.npy')
    z = np.load(DETS / f'{name}_preds.npz')
    boxes, scores = z['boxes'], z['scores']
    im = stretch(gray)
    draw = ImageDraw.Draw(im)
    kept = 0
    dropped = 0
    for b, sc in zip(boxes, scores):
        if sc < thresh:
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r = 0.25 * ((x2 - x1) + (y2 - y1))
        if not keep_circle(cx, cy, r):
            dropped += 1
            continue
        # scale to output pixels
        R = r * SCALE
        CX, CY = cx * SCALE, cy * SCALE
        box = [CX - R, CY - R, CX + R, CY + R]
        # slight dark under-stroke for contrast on bright terrain
        draw.ellipse([v + 0 for v in box], outline=(40, 32, 0, 255), width=LW + 3)
        draw.ellipse(box, outline=GOLD, width=LW)
        kept += 1
    path = FINAL / OUTNAME[name]
    im.save(path, 'PNG')
    print(f'{path.name}: kept={kept} dropped_clipped={dropped} size={im.size}')
    return path, kept, dropped


# crops shown in the Moon animation (animation/mond.py), in path order, with display labels
ANIMATION = [
    ('mare-scherben', 'Mare Imbrium'),
    ('copernicus-rand', 'Copernicus, Nordostteil'),
    ('tycho-umfeld', 'Tycho, Nordostteil'),
    ('clavius-zentriert', 'Clavius, Kraterboden'),
    ('theophilus-umfeld', 'Südliches Hochland, 26° S 27,5° O'),
    ('petavius-umfeld', 'Petavius, Kratermitte'),
    ('hochland-ost', 'Palus Somni'),
]
LABEL = dict(ANIMATION)


def count(name, thresh=0.7):
    """Detections >= thresh, and how many of them lie fully inside the crop (strict edge rule)."""
    z = np.load(DETS / f'{name}_preds.npz')
    n = shown = 0
    for b, sc in zip(z['boxes'], z['scores']):
        if sc < thresh:
            continue
        n += 1
        x1, y1, x2, y2 = [float(v) for v in b]
        if keep_circle((x1 + x2) / 2, (y1 + y2) / 2, 0.25 * ((x2 - x1) + (y2 - y1))):
            shown += 1
    return n, shown


def main():
    out = {}
    for name, t in THRESH.items():
        if not (CROPS / f'{name}.npy').exists():
            print('skip missing', name)
            continue
        p, k, d = render(name, t)
        out[name] = {'label': LABEL[name], 'path': str(Path('final') / p.name), 'kept': k, 'dropped_clipped': d, 'thresh': t}
    anim = {}
    for name, label in ANIMATION:
        if (DETS / f'{name}_preds.npz').exists():
            n, shown = count(name)
            anim[name] = {'label': label, 'detections_ge_0.7': n, 'shown': shown}
    out['mond_animation'] = {
        'rule': 'score >= 0.7; circle (box centre, mean half side) fully inside the 512 px crop',
        'crops': anim,
        'total_shown': sum(v['shown'] for v in anim.values()),
        'total_detections_ge_0.7': sum(v['detections_ge_0.7'] for v in anim.values()),
    }
    (FINAL / 'clean_manifest.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
