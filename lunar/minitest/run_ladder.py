"""Stage B memory ladder: each rung runs in its own process, so an out-of-memory error only stops that rung.
Usage: python run_ladder.py CONFIG BACKBONE_DIR MINI_DIR [--epochs 2] [--limit-batches 0] [--only name,...]"""
import os
os.environ["MPLBACKEND"] = "Agg"  # Colab sets an inline backend that only exists in its kernel
import subprocess, json, sys, argparse
p = argparse.ArgumentParser(); p.add_argument("config"); p.add_argument("backbone"); p.add_argument("mini")
p.add_argument("--epochs", default="2"); p.add_argument("--limit-batches", default="0"); p.add_argument("--only", default="")
a = p.parse_args()
LADDER = [  # name, precision, batch, accumulate, patch, deterministic
    ("offiziell_fp32_b4", "32", 4, 1, 8, 1),
    ("amp_b4", "amp", 4, 1, 8, 1),
    ("amp_b2_acc2", "amp", 2, 2, 8, 1),
    ("amp_b1_acc4", "amp", 1, 4, 8, 1),
    # deterministic=1 makes torchvision use a pure-PyTorch roi_align on CUDA, which needs much more memory.
    ("amp_b1_acc4_nondet", "amp", 1, 4, 8, 0),
    ("amp_ps16_b2_acc2", "amp", 2, 2, 16, 1),
    ("amp_ps16_b4_nondet", "amp", 4, 1, 16, 0),
]
results = {}
for name, prec, b, acc, ps, det in LADDER:
    if a.only and name not in a.only.split(","): continue
    cmd = [sys.executable, "stage_b.py", a.config, a.backbone, a.mini, "--epochs", a.epochs, "--batch", str(b),
           "--accum", str(acc), "--precision", prec, "--patch", str(ps), "--deterministic", str(det), "--limit-batches", a.limit_batches,
           "--out", f"b_{name}.json"]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg"})
    if r.returncode == 0:
        results[name] = json.load(open(f"b_{name}.json"))
        print(name, "OK |", results[name]["precision"], "|", results[name]["peak_vram_GB"], "GB |", results[name]["seconds_per_epoch"], "s/Epoche", flush=True)
    else:
        err = r.stderr + r.stdout
        if "OutOfMemory" in err or "out of memory" in err:
            msg = "OOM (GPU-Speicher voll)"
        elif r.returncode in (-9, 137):
            msg = "Prozess beendet, vermutlich Arbeitsspeicher (RAM) voll"
        else:
            lines = [l for l in err.splitlines() if "Error" in l or "error:" in l] or err.strip().splitlines() or ["?"]
            msg = lines[-1][:300]
        results[name] = {"error": msg, "returncode": r.returncode}
        print(name, "FEHLER", results[name]["error"], flush=True)
json.dump(results, open("stage_b_ladder.json", "w"), indent=1)
