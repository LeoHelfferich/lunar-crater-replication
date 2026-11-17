# Lunar crater detection: an independent fine-tune

An independent replication of the NASA–IBM Lunar Foundation Model crater-detection fine-tune (LRO WAC, LoRA, patch size 8) on Kaggle’s free Tesla T4 GPUs.

This is **not** a NASA or IBM project, and it is **not** endorsed, reviewed, or supported by NASA or IBM. The numbers below are from our own three training runs. We use the publicly released model, config, and Sombench WAC dataset; both are downloaded at runtime and are not stored in this repository.

Upstream code (pinned): [NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model) at commit `d54c67aad513cb9daca444afa425cfb278e4fbf8`.

## Results

Three seeds (46, 47, 48), official split 800 / 100 / 100 tiles (11,384 test annotations), 30 epochs each. Test metrics are taken from `out1/lunar_real/runs/seed*/result.json`. Means and spread match `out1/lunar_real/summary.json`.

| Seed | mAP50-95 | mAP50 | mAP75 | Best val. epoch | Peak VRAM | Mean s/epoch |
|------|----------|-------|-------|-----------------|-----------|--------------|
| 46 | 0.2550536692 (0.2551) | 0.6125 | 0.2217 | 29 | 7.19 GB | 593.6 |
| 47 | 0.2547720373 (0.2548) | 0.6101 | 0.2225 | 29 | 7.19 GB | 592.0 |
| 48 | 0.2538950145 (0.2539) | 0.6106 | 0.2223 | 27 | 7.19 GB | 588.6 |
| **Mean** | **0.2546** | **0.6111** | **0.2221** | | 7.19 GB | ~10 min |
| Sample std. | 0.0006 | | | | | |
| Min–max | 0.2539–0.2551 | | | | | |

According to NASA and IBM (model card and Fraccaro et al., 2026, [arXiv:2609.13283](https://arxiv.org/abs/2609.13283), Table 4; 100% training data, ps8 LoRA, 5 seeds): mAP50-95 **0.2581 ± 0.0017**, mAP50 **0.6183 ± 0.0034**, mAP75 **0.2263 ± 0.0027**.

Difference of means (ours − theirs): **−0.0035**. Training ran on two Tesla T4 GPUs (15,360 MiB each) at no cost, mixed precision `16-mixed`. The last seed finished at 2026-11-12 06:53 UTC. The Kaggle notebook log is `out1/kaggle-run.log`.

This is a close result under the deviations listed below, not an exact reproduction. Hugging Face revisions used for data, backbone, and config are recorded in `out1/lunar_real/manifest.json`.

## Deviations from the NASA/IBM setup

Taken from `out1/lunar_real/summary.json` (`setup_differences_to_disclose`):

1. **30 fixed epochs**, instead of up to 100 with early stopping (patience 15) as described in the model-card text. Best checkpoints were at epoch 27 or 29.
2. **Learning settings taken from the released `WAC_config.yaml`** (learning rate 1e-4, layer decay 1.0, weight decay 0). The model-card text (and the paper’s fine-tuning section) states 5e-5, layer decay 0.65, and weight decay 0.05–0.1 for the pretrained model. A copy of the YAML we used is in `configs/WAC_config.yaml`.
3. **Batch size 1 with gradient accumulation 4** (effective step size 4), versus batch 4 in the NASA config.
4. **fp16 mixed precision on 2× Tesla T4**. The released config specifies fp32.
5. **`deterministic=False`**, because the deterministic `roi_align` fallback needed too much memory. Runs are not bit-identical.
6. **3 seeds** rather than 5.

## Setup

Python 3.12 is required (`>=3.11,<3.13`). See `requirements.txt` for the install order and pins (`torch==2.12.1+cu126`, `terratorch[peft]==1.2.13`, upstream commit above). On Kaggle or Colab, run `bash lunar/realrun/setup_env.sh` instead of installing into the 3.13 kernel.

Weights and the Sombench WAC crater dataset (CC BY 4.0) are **not** in this repo. `lunar/realrun/fetch_data.py` downloads them to a scratch directory at runtime, pinned to fixed Hugging Face revisions. Only tile names, not images, are written to `out1/lunar_real/manifest.json`.

## How to run on Kaggle

1. New notebook, GPU: 2× Tesla T4, internet on. Upload this repository (or copy `lunar/kaggle_real_run.ipynb` and the `lunar/realrun/` scripts).
2. Open `lunar/kaggle_real_run.ipynb` and set the mode to `"full"` (or `"smoke"` for a short trial).
3. Run all. The notebook writes the scripts to `/tmp`, creates a Python 3.12 venv, fetches data and the backbone, then trains seeds 46 and 47 in parallel and seed 48 afterwards. Checkpoints are written every epoch and the run can resume from a previous Kaggle output attached as input.
4. Outputs land in `lunar_real/`: `summary.json`, `manifest.json`, and `runs/seed*/result.json` plus `metrics.jsonl`. Checkpoints (`*.ckpt`) are large and are gitignored; do not commit them.

The same scripts can be run outside the notebook:

```bash
bash lunar/realrun/setup_env.sh
# DATA_DIR should be a scratch path, not the git working tree
python lunar/realrun/fetch_data.py "$DATA_DIR" out1/lunar_real/manifest.json
python lunar/realrun/orchestrate.py --work out1/lunar_real --data "$DATA_DIR" --seeds 46,47,48 --epochs 30
```

A smaller CPU/Colab ladder (mini split, short fine-tune) lives in `lunar/minitest/` and `lunar/lunar_minitest_colab.ipynb`.

## Visuals

`visuals/` contains scripts to overlay detections from one of our checkpoints on public-domain USGS LROC WAC mosaic crops (credit: NASA/GSFC/Arizona State University, USGS mosaic, June 2013). Those images are a visualization, not a measurement run, and they are not Sombench test tiles. Crop coordinates are in `visuals/crops/*.json`. `visuals/animation/mond.py` renders the frames of a Moon animation over seven crops; a detection is drawn and counted only if its score is at least 0.7 and its circle lies fully inside the 512 px crop (per-crop counts in `visuals/final/clean_manifest.json`). Rendered images and mosaic rasters are not included here.

## Credits and sources

- Model card: [nasa-ibm-ai4science/Crater-Detection-NASA-IBM-Lunar-Foundation-Model](https://huggingface.co/nasa-ibm-ai4science/Crater-Detection-NASA-IBM-Lunar-Foundation-Model)
- Paper: Fraccaro et al., 2026, *Multimodal-Multiresolution Foundation Model for Lunar Remote Sensing*, [arXiv:2609.13283](https://arxiv.org/abs/2609.13283), Table 4
- Foundation model: [nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model)
- Dataset: [Sombench-WAC-Crater-Detection](https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-WAC-Crater-Detection) (CC BY 4.0), official split, downloaded at runtime only
- Upstream training code: [NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model) (Apache-2.0), commit `d54c67aad513cb9daca444afa425cfb278e4fbf8`
- `configs/WAC_config.yaml` is the released config from the crater-detection model repo (revision `f09ebeae2099c28f09bf4060f56d3e327a6192de`)

NASA and IBM state that these models are not validated for operational decisions. This repository makes the same limitation explicit.
