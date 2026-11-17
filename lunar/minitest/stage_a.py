"""Stage A: NASA's released WAC crater checkpoint, inference only, on N test tiles.
Uses the upstream data module + test_step, so the metric path is identical to the paper's.
Usage: python stage_a.py CKPT BACKBONE_DIR DATA_DIR ANN_JSON [--plot out.png]
The optional plot writes one tile with reference craters and model boxes."""
import os
os.environ["MPLBACKEND"] = "Agg"  # Colab sets an inline backend that only exists in its kernel
import sys, time, copy, os, json, argparse
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch, lightning.pytorch as pl
import terratorch_integration  # registers backbone/necks/task
from terratorch_integration import LunarObjectDetectionTask
from terratorch_integration.data_adapter import LunarWACCraterDataModule

p = argparse.ArgumentParser(); p.add_argument("ckpt"); p.add_argument("backbone_dir"); p.add_argument("data_dir"); p.add_argument("ann"); p.add_argument("--plot")
a = p.parse_args()
CKPT = a.ckpt; CFG = f"{a.backbone_dir}/config.yaml"
MEANS = [0.018106, 0.025262, 0.027113, 0.028616, 0.030757]   # from NASA WAC_config.yaml (train split)
STDS = [0.010051, 0.013646, 0.014491, 0.015267, 0.016555]

# Lightning stores the checkpoint as a pickle, so weights_only=True cannot read it.
# Only the pinned NASA crater checkpoint, or a patched copy this script wrote, is loaded.
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model_args = copy.deepcopy(ck["hyper_parameters"]["model_args"])
# The released ckpt stores one kwarg ('use_resolution_conditioning') that public terratorch 1.2.x
# does not accept. Strip it once into a patched copy; weights are untouched.
hp = ck["hyper_parameters"]
if "use_resolution_conditioning" in hp or "use_resolution_conditioning" in hp.get("_instantiator_kwargs", {}) or \
   "use_resolution_conditioning" in str(hp):
    def strip(d):
        if isinstance(d, dict):
            d.pop("use_resolution_conditioning", None)
            for v in d.values(): strip(v)
        elif isinstance(d, list):
            for v in d: strip(v)
        elif hasattr(d, "__dict__"):
            for k in list(vars(d)):
                if k == "use_resolution_conditioning": delattr(d, k)
                else: strip(getattr(d, k))
    strip(hp)
    PATCHED = CKPT.replace(".ckpt", "_patched.ckpt")
    if not os.path.exists(PATCHED): torch.save(ck, PATCHED)
    CKPT = PATCHED
del ck
model_args["backbone_cfg"] = CFG; model_args["backbone_checkpoint_path"] = None  # ckpt carries weights
model_args.pop("backbone_modality_info_path", None)  # absolute path on NASA cluster; not needed
t0 = time.time()
model = LunarObjectDetectionTask.load_from_checkpoint(CKPT, map_location="cpu", model_args=model_args).eval()
print(f"model loaded in {time.time()-t0:.1f}s")

dm = LunarWACCraterDataModule(data_dir=a.data_dir, train_annotations_file=a.ann, val_annotations_file=a.ann,
                              test_annotations_file=a.ann, batch_size=1, num_workers=0, image_size=512,
                              min_diameter_px=10.0, drop_no_crater_images=True,
                              wac_norm_means=MEANS, wac_norm_stds=STDS)
acc = "gpu" if (torch.cuda.is_available() or torch.backends.mps.is_available()) else "cpu"
trainer = pl.Trainer(accelerator=acc, devices=1, logger=False, enable_checkpointing=False, precision="32")
if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else acc)
t0 = time.time(); res = trainer.test(model, datamodule=dm, verbose=False); dt = time.time() - t0
n = len(dm.test_dataset) if hasattr(dm, "test_dataset") else None
out = {"accelerator": acc, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else acc,
       "gpu_total_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None, "tiles": n, "seconds_total": round(dt, 1),
       "seconds_per_tile": round(dt / n, 1) if n else None,
       "test_mAP": float(res[0].get("test_map", float("nan"))),
       "test_mAP_50": float(res[0].get("test_map_50", float("nan"))),
       "ref_craters": int(sum(len(dm.test_dataset[i]["boxes"]) for i in range(n))) if n else None,
       "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None}
print("RAW", {k: float(v) for k, v in res[0].items()})
print(json.dumps(out, indent=1))
json.dump(out, open(os.environ.get("STAGE_A_OUT", "stage_a_result.json"), "w"), indent=1)

if a.plot:  # one tile: reference craters (grey) vs model boxes >= 0.5 (yellow)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.patches as P
    s = dm.test_dataset[0]; x = s["vis"] if "vis" in s else s["image"]
    dev = next(model.parameters()).device
    with torch.no_grad():
        y = model.model(x[None].to(dev)); y = y.output[0] if hasattr(y, "output") else y[0]
    img = x[2].numpy(); fig, ax = plt.subplots(figsize=(7, 7)); ax.imshow(img, cmap="gray"); ax.axis("off")
    for b in s["boxes"].tolist():
        ax.add_patch(P.Circle(((b[0]+b[2])/2, (b[1]+b[3])/2), (b[2]-b[0])/2, fill=False, ec="#9a9a9a", lw=0.8))
    k = 0
    for b, sc in zip(y["boxes"].cpu().tolist(), y["scores"].cpu().tolist()):
        if sc >= 0.5:
            k += 1; ax.add_patch(P.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec="#F5C400", lw=1.1))
    ax.set_title(f"Referenz (grau): {len(s['boxes'])}  ·  Modell >=0.5 (gelb): {k}")
    fig.savefig(a.plot, dpi=110, bbox_inches="tight"); print("plot", a.plot, "ref", len(s["boxes"]), "pred", k)
