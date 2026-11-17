"""Stage B: short LoRA fine-tune from NASA's pretrained backbone, official WAC config values.
Measures seconds per epoch and peak GPU memory. Mini-test only: 2 epochs on 12 tiles says
nothing about final accuracy; it tells us whether and how fast the real run fits the GPU.
Usage: python stage_b.py CONFIG_YAML HF_BACKBONE_DIR MINI_DIR [--epochs 2 --batch 4 --accum 1
       --precision 32|amp|bf16-mixed|16-mixed (amp = bf16 if the GPU supports it, else fp16) --patch 8 --deterministic 1 --limit-batches 0]"""
import os
os.environ["MPLBACKEND"] = "Agg"  # Colab sets an inline backend that only exists in its kernel
import os, sys, time, json, argparse, yaml, torch, lightning.pytorch as pl
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import terratorch_integration
from terratorch_integration import LunarObjectDetectionTask, DeterministicAugmentation
from terratorch_integration.data_adapter import LunarWACCraterDataModule
from terratorch_integration.data_utils import D4DetectionTransform

p = argparse.ArgumentParser()
p.add_argument("config"); p.add_argument("backbone_dir"); p.add_argument("mini_dir")
p.add_argument("--epochs", type=int, default=2); p.add_argument("--batch", type=int, default=4)
p.add_argument("--accum", type=int, default=1); p.add_argument("--precision", default="32")
p.add_argument("--patch", type=int, default=8); p.add_argument("--deterministic", type=int, default=1)
p.add_argument("--limit-batches", type=int, default=0); p.add_argument("--out", default="stage_b_result.json")
a = p.parse_args()

cfg = yaml.safe_load(open(a.config)); pl.seed_everything(cfg.get("seed_everything", 46))
m = cfg["model"]["init_args"]; d = cfg["data"]["init_args"]
m["model_args"]["backbone_checkpoint_path"] = f"{a.backbone_dir}/checkpoint.pt"
m["model_args"]["backbone_cfg"] = f"{a.backbone_dir}/config.yaml"
if a.patch != 8:  # patch 16 = 4x fewer tokens, the low-VRAM fallback; ReshapeTokensToImage h must match
    m["model_args"]["backbone_patch_size"] = a.patch
    for n in m["model_args"]["necks"]:
        if n["name"] == "ReshapeTokensToImage": n["h"] = d["image_size"] // a.patch
model = LunarObjectDetectionTask(**m)

dm = LunarWACCraterDataModule(
    data_dir=a.mini_dir, train_annotations_file=f"{a.mini_dir}/mini_train.json",
    val_annotations_file=f"{a.mini_dir}/mini_val.json", test_annotations_file=f"{a.mini_dir}/mini_test.json",
    batch_size=a.batch, num_workers=2, image_size=d["image_size"], min_diameter_px=d["min_diameter_px"],
    drop_no_crater_images=d["drop_no_crater_images"], wac_norm_means=d["wac_norm_means"],
    wac_norm_stds=d["wac_norm_stds"], train_transforms=D4DetectionTransform(image_keys=["image", "vis"], p=1.0))

class Clock(pl.Callback):
    def __init__(s): s.epochs = []
    def on_train_epoch_start(s, *_):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        s.t = time.time()
    def on_train_epoch_end(s, *_):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        s.epochs.append(round(time.time() - s.t, 1))

clock = Clock()
acc = "gpu" if (torch.cuda.is_available() or torch.backends.mps.is_available()) else "cpu"
lim = a.limit_batches or None
if a.precision == "amp":  # T4 (Turing) has no bf16 -> fp16 mixed
    # is_bf16_supported() can be True on a T4 (sm_75) via emulation, which is slow; native bf16 needs Ampere (sm_80)+.
    a.precision = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else "16-mixed"
    if acc == "cpu": a.precision = "bf16-mixed"
trainer = pl.Trainer(accelerator=acc, devices=1, precision=a.precision, max_epochs=a.epochs,
                     accumulate_grad_batches=a.accum, deterministic=bool(a.deterministic), benchmark=False,
                     logger=False, enable_checkpointing=False, log_every_n_steps=1, num_sanity_val_steps=0,
                     callbacks=[DeterministicAugmentation(), clock],
                     limit_train_batches=lim, limit_val_batches=lim, limit_test_batches=lim)
if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
t0 = time.time(); trainer.fit(model, datamodule=dm); fit_s = time.time() - t0
res = trainer.test(model, datamodule=dm, verbose=False)[0]
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else acc
out = {"device": gpu,
       "gpu_total_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
       "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
       "peak_vram_reserved_GB": round(torch.cuda.max_memory_reserved() / 1e9, 2) if torch.cuda.is_available() else None,
       "precision": a.precision, "deterministic": bool(a.deterministic),
       "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None, "batch": a.batch, "accum": a.accum, "patch": a.patch,
       "train_tiles": len(dm.train_dataset), "seconds_per_epoch": clock.epochs, "fit_seconds_total": round(fit_s, 1),
       "test_mAP_after_mini_finetune": float(res.get("test_map", float("nan")))}
print(json.dumps(out, indent=1)); json.dump(out, open(a.out, "w"), indent=1)
