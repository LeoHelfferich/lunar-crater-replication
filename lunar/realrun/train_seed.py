"""Train + test one seed of the WAC crater LoRA fine-tune, with checkpoint every epoch and auto-resume.
Writes into RUN_DIR: ckpt/last.ckpt, ckpt/best.ckpt, best.json, metrics.jsonl (one line per epoch),
status.json (progress), result.json (final test metrics; its presence means 'seed done').

Usage: python train_seed.py --seed 46 --config WAC_config.yaml --backbone-dir DIR --data-dir DIR --ann-dir DIR
       --run-dir DIR [--epochs 30 --batch 1 --accum 4 --precision 16-mixed --patch 8 --deterministic 0
       --limit-train 0 --limit-eval 0 --deadline UNIX_TS --num-workers 2]"""
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, glob, json, time, shutil, argparse, yaml, torch, lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import terratorch_integration
from terratorch_integration import LunarObjectDetectionTask, DeterministicAugmentation
from terratorch_integration.data_adapter import LunarWACCraterDataModule
from terratorch_integration.data_utils import D4DetectionTransform

p = argparse.ArgumentParser()
for k, v in [("--seed", int), ("--config", str), ("--backbone-dir", str), ("--data-dir", str), ("--ann-dir", str), ("--run-dir", str)]:
    p.add_argument(k, type=v, required=True)
p.add_argument("--epochs", type=int, default=30); p.add_argument("--batch", type=int, default=1)
p.add_argument("--accum", type=int, default=4); p.add_argument("--precision", default="16-mixed")
p.add_argument("--patch", type=int, default=8); p.add_argument("--deterministic", type=int, default=0)
p.add_argument("--limit-train", type=int, default=0); p.add_argument("--limit-eval", type=int, default=0)
p.add_argument("--deadline", type=float, default=0); p.add_argument("--num-workers", type=int, default=2)
a = p.parse_args()
R = a.run_dir; os.makedirs(f"{R}/ckpt", exist_ok=True)
now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def write_json(path, obj):  # atomic: a kill never leaves a half-written file
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, path)
if os.path.exists(f"{R}/result.json"):
    print(f"seed {a.seed}: already done ->", json.load(open(f"{R}/result.json"))["test_map"]); sys.exit(0)

cuda = torch.cuda.is_available()
if a.precision == "amp":
    a.precision = "bf16-mixed" if cuda and torch.cuda.get_device_capability(0)[0] >= 8 else ("16-mixed" if cuda else "bf16-mixed")
pl.seed_everything(a.seed, workers=True)
cfg = yaml.safe_load(open(a.config)); m = cfg["model"]["init_args"]; d = cfg["data"]["init_args"]
m["model_args"]["backbone_checkpoint_path"] = f"{a.backbone_dir}/checkpoint.pt"
m["model_args"]["backbone_cfg"] = f"{a.backbone_dir}/config.yaml"
if a.patch != 8:
    m["model_args"]["backbone_patch_size"] = a.patch
    for n in m["model_args"]["necks"]:
        if n["name"] == "ReshapeTokensToImage": n["h"] = d["image_size"] // a.patch
model = LunarObjectDetectionTask(**m)
dm = LunarWACCraterDataModule(
    data_dir=a.data_dir, train_annotations_file=f"{a.ann_dir}/train.json", val_annotations_file=f"{a.ann_dir}/val.json",
    test_annotations_file=f"{a.ann_dir}/test.json", batch_size=a.batch, num_workers=a.num_workers,
    image_size=d["image_size"], min_diameter_px=d["min_diameter_px"], drop_no_crater_images=d["drop_no_crater_images"],
    wac_norm_means=d["wac_norm_means"], wac_norm_stds=d["wac_norm_stds"],
    train_transforms=D4DetectionTransform(image_keys=["image", "vis"], p=1.0))

class EpochLog(Callback):
    """One JSON line per finished epoch; also stops cleanly before an epoch that would not fit the deadline."""
    def __init__(s): s.durations = []
    def on_train_epoch_start(s, trainer, pl_module):
        s.t0 = time.time(); s.losses = []; s.nan_losses = 0; s.skipped = 0
        if cuda: torch.cuda.reset_peak_memory_stats()
    def on_train_batch_end(s, trainer, pl_module, outputs, batch, batch_idx):
        v = trainer.callback_metrics.get("train_loss")
        if v is None or not hasattr(s, "losses"): return
        if torch.isfinite(v): s.losses.append(float(v))
        else: s.nan_losses += 1
    def on_before_optimizer_step(s, trainer, pl_module, optimizer):
        # Guard: a single non-finite gradient (possible with mixed precision) must not poison the weights.
        # With fp16 the GradScaler already skips such steps; this also covers bf16/fp32.
        grads = [q.grad for g in optimizer.param_groups for q in g["params"] if q.grad is not None]
        if grads and not all(torch.isfinite(g).all() for g in grads):
            for g in grads: g.zero_()
            s.skipped += 1
    def on_train_epoch_end(s, trainer, pl_module):  # runs after this epoch's validation
        dur = time.time() - getattr(s, "t0", time.time()); s.durations.append(dur); cm = trainer.callback_metrics
        rec = {"seed": a.seed, "epoch": trainer.current_epoch + 1, "utc": now(),
               "train_loss": round(sum(s.losses) / len(s.losses), 5) if s.losses else None,
               "val_map": round(float(cm["val_map"]), 5) if "val_map" in cm else None,
               "val_map_50": round(float(cm["val_map_50"]), 5) if "val_map_50" in cm else None,
               "seconds": round(dur, 1),
               "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None,
               "lr": trainer.optimizers[0].param_groups[0]["lr"],
               "nan_loss_batches": s.nan_losses, "skipped_nonfinite_steps": s.skipped}
        with open(f"{R}/metrics.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
        print("EPOCH", json.dumps(rec), flush=True)
        if not all(torch.isfinite(q).all() for q in pl_module.parameters() if q.requires_grad):
            raise RuntimeError("trainable weights became NaN/inf; stopping so no session time is wasted")
        if s.losses == [] and s.nan_losses:
            raise RuntimeError("every training loss in this epoch was NaN/inf")
        if a.deadline and rec["epoch"] < a.epochs:
            need = 1.25 * max(s.durations[-2:]) + 120
            if time.time() + need > a.deadline:
                print(f"seed {a.seed}: stopping after epoch {rec['epoch']} (next epoch would pass the session deadline)", flush=True)
                trainer.should_stop = True

class Checkpoints(Callback):
    """Own checkpointing, written atomically (temp file + rename), so a kill can never leave a broken file
    and the names stay fixed across sessions: ckpt/last.ckpt after every epoch, ckpt/best.ckpt for the best val mAP."""
    def __init__(s):
        s.best = json.load(open(f"{R}/best.json")) if os.path.exists(f"{R}/best.json") else {"epoch": 0, "val_map": -1.0}
    def on_train_epoch_end(s, trainer, pl_module):  # after this epoch's validation and after EpochLog
        if trainer.sanity_checking: return
        tmp = f"{R}/ckpt/last.ckpt.tmp"; trainer.save_checkpoint(tmp); os.replace(tmp, f"{R}/ckpt/last.ckpt")
        v = float(trainer.callback_metrics.get("val_map", -1.0))
        if v > s.best["val_map"]:
            shutil.copyfile(f"{R}/ckpt/last.ckpt", f"{R}/ckpt/best.ckpt.tmp"); os.replace(f"{R}/ckpt/best.ckpt.tmp", f"{R}/ckpt/best.ckpt")
            s.best = {"epoch": trainer.current_epoch + 1, "val_map": v}; write_json(f"{R}/best.json", s.best)

log = EpochLog(); ckpt = Checkpoints()
trainer = pl.Trainer(accelerator="gpu" if cuda else "cpu", devices=1, precision=a.precision, max_epochs=a.epochs,
                     accumulate_grad_batches=a.accum, deterministic=bool(a.deterministic), benchmark=False,
                     logger=False, enable_checkpointing=False, log_every_n_steps=10, num_sanity_val_steps=0, enable_progress_bar=False,
                     callbacks=[DeterministicAugmentation(), log, ckpt],
                     limit_train_batches=a.limit_train or None, limit_val_batches=a.limit_eval or None,
                     limit_test_batches=a.limit_eval or None)

def ckpt_epochs(path):  # completed epochs stored in a Lightning checkpoint, or None if unreadable
    # The epoch field is an int, but a Lightning .ckpt is a pickle of the whole trainer state,
    # so weights_only=True cannot load it. Only a checkpoint this run wrote is passed here.
    try: return int(torch.load(path, map_location="cpu", weights_only=False, mmap=True)["epoch"]) + 1
    except Exception as e: print("checkpoint unreadable, ignoring:", path, e); return None
last = f"{R}/ckpt/last.ckpt"
done_in_ckpt = ckpt_epochs(last) if os.path.exists(last) else None
resume = last if done_in_ckpt else None
if os.path.exists(f"{R}/metrics.jsonl"):  # drop log lines of epochs the checkpoint does not contain (killed in between)
    keep = done_in_ckpt or 0
    lines = [l for l in open(f"{R}/metrics.jsonl") if l.strip() and json.loads(l)["epoch"] <= keep]
    open(f"{R}/metrics.jsonl", "w").writelines(lines)
print(f"seed {a.seed}: {'resume from ' + resume if resume else 'fresh start'} | precision {a.precision} | "
      f"device {torch.cuda.get_device_name(0) if cuda else 'cpu'}", flush=True)
t0 = time.time()
trainer.fit(model, datamodule=dm, ckpt_path=resume)
done_epochs = trainer.current_epoch  # number of completed epochs after fit
write_json(f"{R}/status.json", {"seed": a.seed, "epochs_done": done_epochs, "epochs_target": a.epochs, "utc": now(),
                                "fit_seconds_this_session": round(time.time() - t0, 1)})
if done_epochs < a.epochs:
    print(f"seed {a.seed}: paused at epoch {done_epochs}/{a.epochs}; resumes in the next session", flush=True); sys.exit(3)

best = f"{R}/ckpt/best.ckpt"
if cuda: torch.cuda.reset_peak_memory_stats()
res = trainer.test(model, datamodule=dm, ckpt_path=best, verbose=False)[0]
recs = [json.loads(l) for l in open(f"{R}/metrics.jsonl")]
write_json(f"{R}/result.json", {
    "seed": a.seed, "test_map": float(res["test_map"]), "test_map_50": float(res["test_map_50"]),
    "test_map_75": float(res.get("test_map_75", float("nan"))), "best_epoch": ckpt.best["epoch"],
    "best_val_map": max((r["val_map"] or 0) for r in recs), "epochs": a.epochs,
    "mean_epoch_seconds": round(sum(r["seconds"] for r in recs) / len(recs), 1),
    "peak_vram_GB": max((r["peak_vram_GB"] or 0) for r in recs) if cuda else None,
    "precision": a.precision, "batch": a.batch, "accum": a.accum, "patch": a.patch, "deterministic": bool(a.deterministic),
    "limit_train": a.limit_train, "limit_eval": a.limit_eval, "utc": now()})
if os.path.exists(last): os.remove(last)  # done: keep only the best checkpoint (saves 1.1 GB of output space)
print(f"seed {a.seed}: DONE test mAP {res['test_map']:.4f} (mAP50 {res['test_map_50']:.4f})", flush=True)
