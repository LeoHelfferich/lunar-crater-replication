"""Run all seeds: one worker per GPU, each pulls the next unfinished seed (2 in parallel, the 3rd after).
Restores earlier progress from /kaggle/input (output of the previous notebook version) before starting.
Stops cleanly before the session deadline so Kaggle can save the output.

Usage: python orchestrate.py --work DIR --data DIR [--seeds 46,47,48 --epochs 30 --deadline UNIX_TS
       --devices auto|0,1|cpu --input-root /kaggle/input --min-start-hours 1.0 -- <extra args for train_seed.py>]"""
import os
os.environ["MPLBACKEND"] = "Agg"
import sys, json, glob, time, shutil, argparse, subprocess, threading, queue
p = argparse.ArgumentParser()
p.add_argument("--work", required=True); p.add_argument("--data", required=True)
p.add_argument("--seeds", default="46,47,48"); p.add_argument("--epochs", type=int, default=30)
p.add_argument("--deadline", type=float, default=0); p.add_argument("--devices", default="auto")
p.add_argument("--input-root", default="/kaggle/input"); p.add_argument("--min-start-hours", type=float, default=1.0)
a, extra = p.parse_known_args()
extra = [x for x in extra if x != "--"]
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = f"{a.work}/runs"; os.makedirs(RUNS, exist_ok=True)
seeds = [int(s) for s in a.seeds.split(",")]
tag = os.path.basename(os.path.normpath(a.work))

def progress(run_dir):
    if os.path.exists(f"{run_dir}/result.json"): return 10**6
    try: return len({json.loads(l)["epoch"] for l in open(f"{run_dir}/metrics.jsonl") if l.strip()})
    except FileNotFoundError: return -1

# 1) restore from the previous version's output, if it is attached as input and further along
for s in seeds:
    mine = f"{RUNS}/seed{s}"
    cands = [c for c in glob.glob(f"{a.input_root}/**/{tag}/runs/seed{s}", recursive=True) if os.path.isdir(c)]
    if not cands: continue
    best = max(cands, key=progress)
    if progress(best) > progress(mine):
        if os.path.exists(mine): shutil.rmtree(mine)
        shutil.copytree(best, mine)
        print(f"restored seed {s} from {best} (progress {progress(best) if progress(best) < 10**6 else 'done'})", flush=True)
for f in ["manifest.json"]:  # keep the earliest manifest so all sessions document the same selection
    old = glob.glob(f"{a.input_root}/**/{tag}/{f}", recursive=True)
    if old and not os.path.exists(f"{a.work}/{f}"): shutil.copy(old[0], f"{a.work}/{f}")

# 2) devices
if a.devices == "auto":
    try:
        n = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU")])
    except FileNotFoundError:
        n = 0
    devices = [str(i) for i in range(n)] or ["cpu"]
else:
    devices = a.devices.split(",")
todo = queue.Queue()
for s in seeds:
    if not os.path.exists(f"{RUNS}/seed{s}/result.json"): todo.put(s)
print(f"devices {devices} | open seeds {list(todo.queue)} | deadline in "
      f"{(a.deadline - time.time())/3600:.2f} h" if a.deadline else "no deadline", flush=True)
D = a.data
base = [sys.executable, f"{HERE}/train_seed.py", "--config", f"{D}/model/WAC_config.yaml", "--backbone-dir", f"{D}/model/backbone",
        "--data-dir", f"{D}/hf", "--ann-dir", f"{D}/ann", "--epochs", str(a.epochs)] + (["--deadline", str(a.deadline)] if a.deadline else []) + extra

def worker(dev):
    while True:
        if a.deadline and time.time() + a.min_start_hours * 3600 > a.deadline:
            print(f"[dev {dev}] too little session time left to start another seed", flush=True); return
        try: s = todo.get_nowait()
        except queue.Empty: return
        run = f"{RUNS}/seed{s}"; os.makedirs(run, exist_ok=True)
        env = {**os.environ, "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": "" if dev == "cpu" else dev}
        logf = open(f"{run}/log.txt", "a"); logf.write(f"\n===== session start {time.strftime('%Y-%m-%d %H:%M:%S')} on dev {dev}\n"); logf.flush()
        print(f"[dev {dev}] seed {s} started", flush=True)
        proc = subprocess.Popen(base + ["--seed", str(s), "--run-dir", run], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            logf.write(line)
            if line.startswith(("EPOCH", "seed ")) or "Error" in line or "out of memory" in line: print(f"[dev {dev}] {line.rstrip()}", flush=True)
        rc = proc.wait(); logf.close()
        print(f"[dev {dev}] seed {s} finished with code {rc} ({'done' if rc == 0 else 'paused' if rc == 3 else 'ERROR, see ' + run + '/log.txt'})", flush=True)
        if rc not in (0, 3):
            with open(f"{run}/log.txt") as f: print("".join(f.readlines()[-25:]), flush=True)
        if rc == 3: return  # paused because of the deadline: this device is done for this session

threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
for t in threads: t.start()
for t in threads: t.join()
subprocess.run([sys.executable, f"{HERE}/summarize.py", a.work, "--epochs", str(a.epochs), "--seeds", a.seeds])
