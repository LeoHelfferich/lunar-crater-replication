import json
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
wf = lambda name: code(f"%%writefile {name}\n" + open(name).read())
cells = [
md("""# Mondkrater Mini-Test (NASA-IBM Lunar Foundation Model)

Ziel: Bevor ein vollständiger Trainingslauf startet, messen wir, **wie viel Grafikspeicher (VRAM)** und **wie viel Zeit pro Durchgang (Epoche)** das Training braucht.

* **Stufe A, Erkennen:** NASAs fertiges Krater-Modell läuft über 4 Testkacheln. Nur ein Ladetest: Die Zahl soll grob in der Gegend von NASAs 0,258 liegen, mit 4 Kacheln schwankt sie stark.
* **Stufe B, kurz selbst trainieren:** 2 Epochen auf 12 Kacheln mit der offiziellen Konfiguration, in 5 Speicher-Stufen. Die Genauigkeit danach ist bedeutungslos, es geht nur um VRAM und Zeit.

Technik: Das NASA-Repo braucht Python 3.11 oder 3.12, Colab läuft auf 3.13. Zelle 2 baut deshalb eine eigene Python-3.12-Umgebung (`/content/py312`) mit GPU-PyTorch. Alle Rechenschritte laufen darin, ein Neustart ist nicht nötig.

Daten: Modell und Grundmodell von Hugging Face (Apache-2.0). Die 20 Kacheln kommen aus `Sombench-WAC-Crater-Detection` (CC BY 4.0) und werden **nur zur Laufzeit** geladen. Bilder daraus nicht speichern oder teilen.

**So geht's:** *Laufzeit → Laufzeittyp ändern → T4 GPU → Speichern*, dann *Laufzeit → Alle ausführen*. Dauer ca. 25 bis 40 Minuten. Am Ende die Ausgabe der letzten Zelle prüfen."""),
code("""# 0) Grafik-Backend festlegen: Colab setzt eines, das nur im Notebook-Kernel existiert und Hilfsprozesse abstürzen ließe
import os
os.environ["MPLBACKEND"] = "Agg"
print("MPLBACKEND =", os.environ["MPLBACKEND"])"""),
md("### 1) Skripte schreiben"),
wf("setup_env.sh"), wf("download.py"), wf("make_mini_split.py"), wf("stage_a.py"), wf("stage_b.py"), wf("run_ladder.py"),
code("""# 2) GPU prüfen und Python-3.12-Umgebung bauen (einmalig ca. 3-5 min, beim zweiten Mal sofort fertig)
import subprocess, sys, os
PY = "/content/py312/bin/python"
ENV = {**os.environ, "MPLBACKEND": "Agg"}
def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, env=ENV)
print("Kernel-Python:", sys.version.split()[0])
gpu = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])
assert gpu.returncode == 0, "Keine GPU gefunden: Laufzeit -> Laufzeittyp ändern -> T4 GPU, dann erneut 'Alle ausführen'."
print(gpu.stdout)

def setup(*args):
    r = sh(["bash", "setup_env.sh", *args]); print(r.stdout[-2000:])
    assert r.returncode == 0, r.stderr[-3000:]

CHECK = "import torch; print('GPU sichtbar:', torch.cuda.is_available(), '| torch', torch.__version__, '| CUDA-Build', torch.version.cuda)"
def gpu_ok():  # eigener Test, unabhängig von der Ausgabe von setup_env.sh
    r = sh([PY, "-c", CHECK]); print((r.stdout + r.stderr).strip()[-1500:])
    return r.returncode == 0 and "GPU sichtbar: True" in r.stdout

setup()
if not gpu_ok():
    print("PyTorch sieht die GPU nicht -> installiere die CUDA-Räder (cu126) neu und prüfe erneut ...")
    setup("--force-torch")
    if not gpu_ok():
        print("--- Diagnose ---")
        for cmd in (["nvidia-smi"], [PY, "-c", "import torch; torch.cuda.init()"],
                    ["bash", "-c", "echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH; uv pip list --python " + PY + " 2>/dev/null | grep -iE 'torch|nvidia' || true"]):
            r = sh(cmd); print("$", " ".join(cmd)[:120]); print((r.stdout + r.stderr)[-2500:])
        raise AssertionError("PyTorch sieht die GPU auch nach Neuinstallation nicht. Bitte die komplette Ausgabe dieser Zelle zurückschicken.")
print("GPU bereit.")"""),
code("""# 3) Downloads: Krater-Modell (1,1 GB), Grundmodell (2,4 GB), 20 Kacheln (~80 MB) + Annotationen
!MPLBACKEND=Agg {PY} download.py hf
!MPLBACKEND=Agg {PY} make_mini_split.py wac_mini 12 4 4"""),
code("""# 4) Stufe A: NASAs Krater-Modell auf 4 Testkacheln
!MPLBACKEND=Agg STAGE_A_OUT=stage_a_result.json {PY} stage_a.py hf/WAC_ni_lfm_ps8_lora_s46.ckpt hf/backbone wac_mini wac_mini/mini_test.json 2>&1 | grep -vi warn | tail -16"""),
code("""# 5) Stufe B: Speicher-Leiter (je 2 Epochen). Ein Speicherüberlauf (OOM) bricht nur die jeweilige Stufe ab.
LIMIT = 0  # 0 = alle Kacheln
!MPLBACKEND=Agg {PY} run_ladder.py hf/WAC_config.yaml hf/backbone wac_mini --epochs 2 --limit-batches {LIMIT}"""),
code("""# 6) Zusammenfassung zum Zurückschicken (bitte diese Ausgabe komplett kopieren)
import json
a = json.load(open("stage_a_result.json")); b = json.load(open("stage_b_ladder.json"))
for v in b.values():
    if "seconds_per_epoch" in v:
        s = v["seconds_per_epoch"][-1] / v["train_tiles"]   # letzte Epoche, ohne Aufwärmen
        v["hochrechnung_stunden_800_kacheln_100_epochen"] = round(s * 800 * 100 / 3600, 1)  # grob, ohne Validierung
print(json.dumps({"gpu": a.get("device"), "stage_A": a, "stage_B": b}, indent=1))"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("../lunar_minitest_colab.ipynb", "w"), indent=1, ensure_ascii=False)
print("ok", len(cells))
