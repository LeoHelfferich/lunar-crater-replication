"""Download a 20-tile mini split of Sombench-WAC-Crater-Detection (CC BY 4.0) at runtime.
Nothing from this dataset is stored in our repo. Default: 12 train / 4 val / 4 test, fixed seed.
Usage: python make_mini_split.py OUT_DIR [n_train n_val n_test]"""
import os
os.environ["MPLBACKEND"] = "Agg"  # Colab sets an inline backend that only exists in its kernel
import sys, json, random, os
from huggingface_hub import hf_hub_download
REPO = "nasa-ibm-ai4science/Sombench-WAC-Crater-Detection"
out = sys.argv[1]; n = dict(zip(["train", "val", "test"], map(int, sys.argv[2:5] or [12, 4, 4])))
os.makedirs(out, exist_ok=True)
hf_hub_download(REPO, "metadata.parquet", repo_type="dataset", local_dir=out)
for split, k in n.items():
    full = json.load(open(hf_hub_download(REPO, f"{split}.json", repo_type="dataset", local_dir=out)))
    per = {}
    for a in full["annotations"]:
        per[a["image_id"]] = per.get(a["image_id"], 0) + 1
    imgs = sorted([i for i in full["images"] if per.get(i["id"], 0) >= 20], key=lambda i: i["id"])
    pick = random.Random(46).sample(imgs, k); ids = {i["id"] for i in pick}
    mini = {kk: v for kk, v in full.items() if kk not in ("images", "annotations")}
    mini["images"] = pick; mini["annotations"] = [a for a in full["annotations"] if a["image_id"] in ids]
    json.dump(mini, open(f"{out}/mini_{split}.json", "w"))
    for i in pick:
        stem = os.path.splitext(os.path.basename(i["file_name"]))[0]
        hf_hub_download(REPO, f"images_tiff/{stem}.tif", repo_type="dataset", local_dir=out)
    print(split, len(pick), "tiles,", len(mini["annotations"]), "annotations")
