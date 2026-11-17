"""Downloads at runtime: crater model (Apache-2.0, 1.1 GB), backbone (Apache-2.0, 2.4 GB). Usage: python download.py HF_DIR"""
import os
os.environ["MPLBACKEND"] = "Agg"  # Colab sets an inline backend that only exists in its kernel
import sys
from huggingface_hub import hf_hub_download, snapshot_download
d = sys.argv[1]; R = "nasa-ibm-ai4science/Crater-Detection-NASA-IBM-Lunar-Foundation-Model"
for f in ["WAC_ni_lfm_ps8_lora_s46.ckpt", "WAC_config.yaml"]:
    print(hf_hub_download(R, f, local_dir=d))
snapshot_download("nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model", allow_patterns="backbone/*", local_dir=d)
print("fertig")
