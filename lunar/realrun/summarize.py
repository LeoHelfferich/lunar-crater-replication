"""Summary over all seeds: per-seed test mAP, mean, spread, comparison with NASA's published 0.2581 +- 0.0017.
Usage: python summarize.py WORK_DIR [--epochs 30 --seeds 46,47,48]"""
import os, sys, json, glob, statistics as st, argparse
p = argparse.ArgumentParser(); p.add_argument("work"); p.add_argument("--epochs", type=int, default=30); p.add_argument("--seeds", default="46,47,48")
a = p.parse_args()
NASA = {"test_map": 0.2581, "seed_spread": 0.0017, "test_map_50": 0.6183, "seeds": 5, "source":
        "Model card nasa-ibm-ai4science/Crater-Detection-NASA-IBM-Lunar-Foundation-Model, WAC, 100% data, ps8 LoRA"}
TOL = 0.01
per = {}
for s in [int(x) for x in a.seeds.split(",")]:
    R = f"{a.work}/runs/seed{s}"
    recs = list({r["epoch"]: r for r in (json.loads(l) for l in open(f"{R}/metrics.jsonl") if l.strip())}.values()) if os.path.exists(f"{R}/metrics.jsonl") else []
    entry = {"epochs_done": len(recs), "epochs_target": a.epochs,
             "last_epoch": recs[-1] if recs else None,
             "best_val_map_so_far": max((r["val_map"] or 0) for r in recs) if recs else None}
    if os.path.exists(f"{R}/result.json"):
        entry.update(json.load(open(f"{R}/result.json"))); entry["status"] = "done"
    else:
        entry["status"] = "running/paused" if recs else "not started"
    per[f"seed{s}"] = entry
done = [v for v in per.values() if v["status"] == "done"]
out = {"status": "complete" if len(done) == len(per) else f"{len(done)} of {len(per)} seeds done", "per_seed": per}
if done:
    maps = [v["test_map"] for v in done]; m50 = [v["test_map_50"] for v in done]
    mean = st.mean(maps)
    out["test_map_mean"] = round(mean, 4)
    out["test_map_std"] = round(st.stdev(maps), 4) if len(maps) > 1 else None
    out["test_map_min_max"] = [round(min(maps), 4), round(max(maps), 4)]
    out["test_map_50_mean"] = round(st.mean(m50), 4)
    out["nasa_reference"] = NASA
    out["difference_to_nasa"] = round(mean - NASA["test_map"], 4)
    out["reproduced_within_0.01"] = bool(abs(mean - NASA["test_map"]) <= TOL) if len(done) == len(per) else "pending"
out["setup_differences_to_disclose"] = [
    f"{a.epochs} epochs fixed (model card: up to 100 with early stopping, patience 15)",
    "hyperparameters from released WAC_config.yaml: lr 1e-4, layer decay 1.0, weight decay 0 (model card text: 5e-5, 0.65, 0.05-0.1)",
    "batch 1 x gradient accumulation 4 (NASA: batch 4)",
    "fp16 mixed precision (16-mixed) on Tesla T4; NASA config: fp32",
    "deterministic=False (torchvision roi_align fallback needs too much memory); runs not bit-identical",
    f"{len(per)} seeds (NASA: 5); test set 100 tiles, so differences below the seed spread are not a ranking",
]
json.dump(out, open(f"{a.work}/summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
