"""Download everything at runtime from Hugging Face, pinned to fixed revisions.
Nothing from the Sombench dataset (CC BY 4.0) is written to the output folder except the
list of tile names in manifest.json. Data goes to a scratch folder that Kaggle does not save.

Usage: python fetch_data.py DATA_DIR MANIFEST_JSON [--subset N_TRAIN,N_VAL,N_TEST] [--seeds 46,47,48]
  full run: all official splits (800 / 100 / 100 tiles)
  --subset: smoke test; per split the first N tiles sorted by image id (deterministic)"""
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")   # default 10 s is too short when the Hub is busy
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
import sys, json, hashlib, argparse, time, random
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download as _hf_hub_download

# Hugging Face sometimes answers 429 "Too Many Requests / maximum queue size reached" or 5xx when busy.
# Every file is fetched on its own (no repo tree listing) and retried with exponential backoff.
RETRY_TRIES, RETRY_BASE, RETRY_MAX = 10, 30, 300
def _status(e):
    r = getattr(e, "response", None)
    return getattr(r, "status_code", None)
def hf_hub_download(*args, **kw):
    for attempt in range(1, RETRY_TRIES + 1):
        try:
            return _hf_hub_download(*args, **kw)
        except Exception as e:
            code = _status(e)
            transient = code in (408, 425, 429, 500, 502, 503, 504) or code is None and any(
                w in type(e).__name__ for w in ("Timeout", "Connection", "ChunkedEncoding", "Protocol"))
            if not transient or attempt == RETRY_TRIES:
                raise RuntimeError(f"Hugging-Face-Download endgueltig fehlgeschlagen nach {attempt} Versuchen "
                                   f"({args[0]} / {args[1] if len(args) > 1 else kw.get('filename')}): HTTP {code} {type(e).__name__}: {str(e)[:300]}") from e
            wait = min(RETRY_MAX, RETRY_BASE * 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
            print(f"  HF HTTP {code} ({type(e).__name__}) bei {args[1] if len(args) > 1 else kw.get('filename')}, "
                  f"Versuch {attempt}/{RETRY_TRIES}, warte {wait:.0f}s", flush=True)
            time.sleep(wait)

DATASET = ("nasa-ibm-ai4science/Sombench-WAC-Crater-Detection", "20f800becb64a0c7ad314652a683f97f0281fe67")
BACKBONE = ("nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model", "b657bdfe6fc4c0445d918f77a2d7b919805f3bce")
CRATER = ("nasa-ibm-ai4science/Crater-Detection-NASA-IBM-Lunar-Foundation-Model", "f09ebeae2099c28f09bf4060f56d3e327a6192de")
UPSTREAM_CODE = "d54c67aad513cb9daca444afa425cfb278e4fbf8"

p = argparse.ArgumentParser()
p.add_argument("data_dir"); p.add_argument("manifest")
p.add_argument("--subset", default=""); p.add_argument("--seeds", default="46,47,48")
a = p.parse_args()
d = a.data_dir; os.makedirs(d, exist_ok=True); t0 = time.time()
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()

cfg = hf_hub_download(CRATER[0], "WAC_config.yaml", revision=CRATER[1], local_dir=f"{d}/model")
for f in ["backbone/config.yaml", "backbone/checkpoint.pt"]:   # the only files in backbone/ at this revision
    hf_hub_download(BACKBONE[0], f, revision=BACKBONE[1], local_dir=f"{d}/model")
print("backbone ready", flush=True)
sub = dict(zip(["train", "val", "test"], map(int, a.subset.split(",")))) if a.subset else {}
splits = {}
os.makedirs(f"{d}/ann", exist_ok=True)
for split in ["train", "val", "test"]:
    src = hf_hub_download(DATASET[0], f"{split}.json", repo_type="dataset", revision=DATASET[1], local_dir=f"{d}/hf")
    full = json.load(open(src))
    imgs = sorted(full["images"], key=lambda i: i["id"])
    if split in sub: imgs = imgs[:sub[split]]
    ids = {i["id"] for i in imgs}
    out = {k: v for k, v in full.items() if k not in ("images", "annotations")}
    out["images"] = imgs; out["annotations"] = [x for x in full["annotations"] if x["image_id"] in ids]
    json.dump(out, open(f"{d}/ann/{split}.json", "w"))
    stems = [os.path.splitext(os.path.basename(i["file_name"]))[0] for i in imgs]
    splits[split] = {"source_json_sha256": sha(src), "n_tiles": len(stems), "n_annotations": len(out["annotations"]), "tiles": stems}
    print(split, len(stems), "tiles", len(out["annotations"]), "annotations", flush=True)

all_stems = [s for v in splits.values() for s in v["tiles"]]
files = [f"images_tiff/{s}.tif" for s in all_stems] + ["metadata.parquet"]
fetch = lambda f: hf_hub_download(DATASET[0], f, repo_type="dataset", revision=DATASET[1], local_dir=f"{d}/hf")
with ThreadPoolExecutor(4) as ex:   # 4 parallel downloads: fast enough, gentle on the Hub
    for n, _ in enumerate(ex.map(fetch, files), 1):
        if n % 100 == 0 or n == len(files): print(f"  tiles {n}/{len(files)}", flush=True)
os.makedirs(f"{d}/hf/images_tiff", exist_ok=True)
missing = [s for s in all_stems if not os.path.exists(f"{d}/hf/images_tiff/{s}.tif")]
assert not missing, f"{len(missing)} tiles missing, e.g. {missing[:3]}"

manifest = {
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "mode": "subset " + a.subset if a.subset else "full official splits",
    "seeds": [int(s) for s in a.seeds.split(",")],
    "revisions": {"dataset": DATASET, "backbone": BACKBONE, "crater_model_config": CRATER, "upstream_code_commit": UPSTREAM_CODE},
    "config_sha256": sha(cfg),
    "license_note": "Sombench-WAC-Crater-Detection is CC BY 4.0 and is downloaded at runtime only; this file lists tile names, no image data.",
    "splits": splits,
}
os.makedirs(os.path.dirname(os.path.abspath(a.manifest)), exist_ok=True)
json.dump(manifest, open(a.manifest, "w"), indent=1)
print(f"data ready in {time.time()-t0:.0f}s:", d, "| manifest:", a.manifest)
