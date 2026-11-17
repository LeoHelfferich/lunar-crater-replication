"""Run our seed-46 WAC crater checkpoint on public-domain USGS mosaic crops.

Input is a 1-band morphology mosaic, standardized and replicated to the
5 vis channels the model expects. This is a visualization, not a benchmark
measurement. No Sombench tiles are used.
"""
import os, sys, json, time, copy, gc
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Keep Hugging Face from trying the network for tokenizer etc.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as P
from PIL import Image

# Register NASA task/backbone
import terratorch_integration  # noqa: F401
from terratorch_integration import LunarObjectDetectionTask

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "out1/lunar_real/runs/seed46/ckpt/best.ckpt"
BACKBONE = ROOT / "lunar/hf/backbone"
CROPS = ROOT / "visuals/crops"
OUT = ROOT / "visuals/detections"
OUT.mkdir(parents=True, exist_ok=True)

MEANS = np.array([0.018106, 0.025262, 0.027113, 0.028616, 0.030757], np.float32)
STDS = np.array([0.010051, 0.013646, 0.014491, 0.015267, 0.016555], np.float32)

THRESHOLDS = [0.3, 0.5, 0.7]


def prepare_tensor(gray_u8: np.ndarray) -> torch.Tensor:
    gray = gray_u8.astype(np.float32)
    valid = gray > 0
    mu = gray[valid].mean() if valid.any() else gray.mean()
    sd = gray[valid].std() if valid.any() else gray.std()
    sd = max(float(sd), 1e-6)
    z = (gray - mu) / sd
    # Replicated 5-band: after official per-channel norm this equals z on every band.
    vis = np.stack([z * STDS[c] + MEANS[c] for c in range(5)], axis=0)
    vis_n = (vis - MEANS[:, None, None]) / STDS[:, None, None]
    return torch.from_numpy(vis_n.astype(np.float32))


def stretch(gray_u8: np.ndarray) -> np.ndarray:
    a = gray_u8.astype(np.float32)
    valid = a[a > 0]
    if valid.size == 0:
        lo, hi = 0, 255
    else:
        lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        hi = lo + 1
    x = np.clip((a - lo) / (hi - lo), 0, 1)
    return x


def strip_unknown(hp):
    def strip(d):
        if isinstance(d, dict):
            d.pop("use_resolution_conditioning", None)
            for v in d.values():
                strip(v)
        elif isinstance(d, list):
            for v in d:
                strip(v)
        elif hasattr(d, "__dict__"):
            for k in list(vars(d)):
                if k == "use_resolution_conditioning":
                    delattr(d, k)
                else:
                    strip(getattr(d, k))
    strip(hp)


def load_model():
    print("loading checkpoint…", CKPT, flush=True)
    t0 = time.time()
    # Lightning stores the checkpoint as a pickle, so weights_only=True cannot read it.
    # Only the self-produced seed-46 checkpoint is loaded.
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    strip_unknown(hp)
    model_args = copy.deepcopy(hp["model_args"])
    model_args["backbone_cfg"] = str(BACKBONE / "config.yaml")
    # Weights live in the lightning ckpt; don't reload the raw backbone from a Kaggle path.
    model_args["backbone_checkpoint_path"] = None
    model_args.pop("backbone_modality_info_path", None)
    tmp = OUT / "_patched_seed46.ckpt"
    if not tmp.exists():
        torch.save(ck, tmp)
    del ck
    gc.collect()
    model = LunarObjectDetectionTask.load_from_checkpoint(
        str(tmp), map_location="cpu", model_args=model_args
    ).eval()
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)
    return model


def predict(model, x: torch.Tensor):
    with torch.no_grad():
        y = model.model(x[None])
        y = y.output[0] if hasattr(y, "output") else y[0]
    boxes = y["boxes"].cpu().numpy()
    scores = y["scores"].cpu().numpy()
    labels = y["labels"].cpu().numpy() if "labels" in y else np.ones(len(scores))
    return boxes, scores, labels


def draw(gray, boxes, scores, title, path, thresh, style="box"):
    img = stretch(gray)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0b0d12")
    ax.set_facecolor("#0b0d12")
    ax.imshow(img, cmap="gray", origin="upper", vmin=0, vmax=1)
    ax.set_xlim(0, 511)
    ax.set_ylim(511, 0)
    ax.axis("off")
    k = 0
    for b, sc in zip(boxes, scores):
        if sc < thresh:
            continue
        k += 1
        x1, y1, x2, y2 = b
        color = "#F5C400"
        if style == "box":
            ax.add_patch(P.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, ec=color, lw=1.15, alpha=0.95))
        else:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            r = 0.5 * max(x2 - x1, y2 - y1) / 2 + 0.5 * min(x2 - x1, y2 - y1) / 2
            # mean of half-width and half-height
            r = 0.25 * ((x2 - x1) + (y2 - y1))
            ax.add_patch(P.Circle((cx, cy), r, fill=False, ec=color, lw=1.15, alpha=0.95))
    ax.set_title(title + f"  ·  {k} bei Schwelle {thresh:g}",
                 color="white", fontsize=11, pad=8, fontweight="medium")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return k


def main():
    model = load_model()
    crops = sorted(CROPS.glob("*.npy"))
    summary = []
    for npy in crops:
        name = npy.stem
        gray = np.load(npy)
        print(f"infer {name} {gray.shape}…", flush=True)
        t0 = time.time()
        x = prepare_tensor(gray)
        boxes, scores, labels = predict(model, x)
        dt = time.time() - t0
        print(f"  {len(scores)} raw preds in {dt:.1f}s; "
              f"n>=0.3={(scores>=0.3).sum()} n>=0.5={(scores>=0.5).sum()} n>=0.7={(scores>=0.7).sum()} "
              f"max={scores.max() if len(scores) else 'na'}", flush=True)
        rec = {
            "name": name,
            "n_raw": int(len(scores)),
            "seconds": round(dt, 2),
            "max_score": float(scores.max()) if len(scores) else None,
            "counts": {str(t): int((scores >= t).sum()) for t in THRESHOLDS},
        }
        np.savez(OUT / f"{name}_preds.npz", boxes=boxes, scores=scores, labels=labels)
        for t in THRESHOLDS:
            for style in ("box", "circle"):
                k = draw(gray, boxes, scores,
                         name, OUT / f"{name}_t{t}_{style}.png", t, style)
                rec.setdefault("drawn", {})[f"{style}@{t}"] = k
        summary.append(rec)
        # free
        del x, boxes, scores, labels
        gc.collect()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("done")


if __name__ == "__main__":
    main()
