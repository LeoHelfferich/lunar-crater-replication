import json
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
wf = lambda name: code(f"%%writefile /tmp/realrun/{name}\n" + open(name).read())
cells = [
md("""# Mondkrater: echter Lauf auf Kaggle (NASA-IBM Lunar Foundation Model, WAC)

Ziel: NASAs veröffentlichten Wert **0,2581 mAP** (WAC-Krater, Patch 8, LoRA) mit 3 Wiederholungen nachmessen.
Einstellung: Patch 8, fp16 (`16-mixed`), Batch 1 × Gradient-Akkumulation 4, `deterministic=False`, 30 Epochen, Seeds 46/47/48,
offizielle Aufteilung 800 Training / 100 Validierung / 100 Test, Lernwerte aus der veröffentlichten `WAC_config.yaml`.

**Vor dem Start (einmalig):** Konto mit bestätigter Telefonnummer. Rechts unter *Settings*: **Accelerator: GPU T4 x2**, **Internet: an**.

**Ablauf**
1. **Probelauf:** In Zelle 1 steht `MODE = "smoke"`. Oben *Run All* im Editor (ca. 15 Minuten). Die letzte Zelle muss „SMOKE OK“ zeigen.
2. **Echter Lauf:** In Zelle 1 `MODE = "full"` setzen. Dann *Save Version* → **Save & Run All (Commit)** → *Save*. Der Lauf arbeitet im Hintergrund, der Browser darf zu. Nach höchstens etwa 11,5 Stunden ist die Version fertig.
3. **Fortsetzen:** Steht in der letzten Ausgabe „noch nicht fertig“: Notebook öffnen → rechts *Input* → *Add Input* → *Your Work* → **dieses Notebook** (neueste Version) hinzufügen. Dann wieder *Save & Run All*. Das Notebook findet den alten Stand selbst und macht weiter. Vor jedem weiteren Durchgang die Eingabe auf die **neueste** Version umstellen.

Daten: Modell und Grundmodell (Apache-2.0) und der Benchmark `Sombench-WAC-Crater-Detection` (CC BY 4.0) werden zur Laufzeit von Hugging Face geladen, mit festen Versionen. Die Bilder liegen nur im Zwischenspeicher und landen **nicht** in der Ausgabe. Heruntergeladene Kacheln nicht festschreiben."""),
code("""# 1) Einstellungen
import os, time, json, subprocess, sys
MODE = "full"   # "smoke" = Probelauf (~15 min) | "full" = echter Lauf (Save & Run All)
START = time.time()
os.environ["MPLBACKEND"] = "Agg"   # Kaggle/Colab setzen ein Backend, das nur im Kernel existiert
SESSION_HOURS = 11.0               # Training stoppt sauber vorher; Kaggle bricht eine Version nach 12 h ab
DEADLINE = START + SESSION_HOURS * 3600
PY = "/tmp/py312/bin/python"; CODE = "/tmp/realrun"; DATA = "/tmp/lunar_data"
WORK = "/kaggle/working/lunar_real" if MODE == "full" else "/kaggle/working/lunar_smoke"
SEEDS = "46,47,48" if MODE == "full" else "46,47"
os.makedirs(CODE, exist_ok=True); os.makedirs(WORK, exist_ok=True)
ENV = {**os.environ, "MPLBACKEND": "Agg"}
print("Modus:", MODE, "| Ausgabe:", WORK, "| Kernel-Python:", sys.version.split()[0])"""),
md("### 2) Skripte schreiben"),
wf("setup_env.sh"), wf("fetch_data.py"), wf("train_seed.py"), wf("orchestrate.py"), wf("summarize.py"),
code("""# 3) GPUs prüfen und Python-3.12-Umgebung bauen (ca. 3-5 min)
def sh(cmd): return subprocess.run(cmd, capture_output=True, text=True, env=ENV)
gpu = sh(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"])
assert gpu.returncode == 0, "Keine GPU: rechts Settings -> Accelerator -> GPU T4 x2"
print(gpu.stdout)
r = sh(["bash", f"{CODE}/setup_env.sh"]); print(r.stdout[-1500:]); assert r.returncode == 0, r.stderr[-3000:]
CHECK = "import torch; print('GPUs:', torch.cuda.device_count(), '| torch', torch.__version__, '| CUDA-Build', torch.version.cuda)"
def ngpu():
    r = sh([PY, "-c", CHECK]); print((r.stdout + r.stderr).strip()[-800:])
    try: return int(r.stdout.split("GPUs:")[1].split("|")[0])
    except Exception: return 0
n = ngpu()
if n == 0:
    print("PyTorch sieht keine GPU -> CUDA-Räder neu installieren ...")
    r = sh(["bash", f"{CODE}/setup_env.sh", "--force-torch"]); print(r.stdout[-1500:]); n = ngpu()
assert n > 0, "PyTorch sieht keine GPU. Bitte die Ausgabe dieser Zelle prüfen."
if n < 2: print("Hinweis: nur 1 GPU sichtbar. Läuft, aber die Seeds laufen dann nacheinander (langsamer). Für 2 parallel: GPU T4 x2 wählen.")
print("GPU bereit.")"""),
code("""# 4) Daten laden (fest versioniert). Smoke: 40/10/10 Kacheln. Full: alle 800/100/100 (~4 GB, ca. 5-10 min)
subset = [] if MODE == "full" else ["--subset", "40,10,10"]
r = subprocess.run([PY, f"{CODE}/fetch_data.py", DATA, f"{WORK}/manifest.json", "--seeds", SEEDS] + subset, env=ENV,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print("\\n".join(l for l in r.stdout.splitlines() if "Fetching" not in l and "Warning" not in l)[-2000:])
if r.returncode != 0:
    out = r.stdout
    if "HTTP 429" in out or "Too Many Requests" in out or any(f"HTTP {c}" in out for c in (500, 502, 503, 504)):
        raise RuntimeError("Download fehlgeschlagen: Hugging Face war auch nach 10 Versuchen überlastet (HTTP 429/5xx). "
                           "Internet ist an. Einfach später erneut 'Save & Run All' starten.")
    if "NameResolution" in out or "Failed to resolve" in out or "Temporary failure in name resolution" in out:
        raise RuntimeError("Download fehlgeschlagen: kein Internet. Rechts unter Settings 'Internet' einschalten.")
    raise RuntimeError("Download fehlgeschlagen (Details oben). Bitte die Ausgabe dieser Zelle prüfen.")"""),
code("""# 5) Training. Full: Seeds 46 und 47 parallel (je eine GPU), 48 danach. Fortschritt erscheint hier pro Epoche.
EPOCHS = 30 if MODE == "full" else 1
train_args = ["--precision", "16-mixed", "--batch", "1", "--accum", "4", "--patch", "8", "--deterministic", "0", "--num-workers", "2"]
if MODE != "full": train_args += ["--limit-train", "40", "--limit-eval", "10"]
cmd = [PY, f"{CODE}/orchestrate.py", "--work", WORK, "--data", DATA, "--seeds", SEEDS, "--epochs", str(EPOCHS),
       "--deadline", str(DEADLINE), "--input-root", "/kaggle/input", "--min-start-hours", "1.0", "--"] + train_args
p = subprocess.Popen(cmd, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if not line.startswith((" ", "{", "}")): print(line.rstrip(), flush=True)   # summary JSON comes in the next cell
print("Orchestrator-Exitcode:", p.wait())"""),
code("""# 6) Zusammenfassung
s = json.load(open(f"{WORK}/summary.json"))
print(json.dumps(s, indent=1))
if MODE != "full":
    ok = all(v["status"] == "done" and v.get("last_epoch") and v["last_epoch"]["train_loss"] is not None
             and v["last_epoch"]["nan_loss_batches"] == 0 for v in s["per_seed"].values())
    ep = [v["last_epoch"] for v in s["per_seed"].values() if v.get("last_epoch")]
    if ep:
        sec = max(e["seconds"] for e in ep)   # 40 Trainings- + 10 Validierungs-Kacheln, beide GPUs parallel
        print(f"Grobe Hochrechnung: ~{sec / 50 * 900 / 60:.0f} min pro Epoche im echten Lauf "
              f"(~{sec / 50 * 900 * 30 / 3600:.1f} h für 30 Epochen pro Seed)")
    print("SMOKE OK: jetzt MODE = \\"full\\" setzen und 'Save & Run All' starten." if ok else "SMOKE FEHLER: bitte die Ausgabe dieser Zelle prüfen.")
elif s["status"] == "complete":
    print(f"FERTIG: Mittelwert {s['test_map_mean']} (NASA 0.2581), Abweichung {s['difference_to_nasa']}, "
          f"innerhalb ±0,01: {s['reproduced_within_0.01']}")
else:
    print("NOCH NICHT FERTIG (" + s["status"] + "). Fortsetzen: Input -> Add Input -> dieses Notebook (neueste Version), dann wieder 'Save & Run All'.")"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open("../kaggle_real_run.ipynb", "w"), indent=1, ensure_ascii=False)
print("ok", len(cells))
