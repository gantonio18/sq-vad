# SQ-VAD — Selectively Queried Video Anomaly Detection

Closing the blind spot of pose-based video anomaly detection with a selectively
queried frozen vision–language model.

A pose-based anomaly detector can only score what the pose estimator detects. On
ShanghaiTech, frames with no detected person are scored as *maximally normal*, and
anomalies caused by non-human objects (a van, a cart, a bicycle on a walkway)
produce no signal at all. SQ-VAD keeps the pose model **frozen**, adds a **frozen**
vision–language model (VLM) as a second stream queried on only the 3.7% of frames
where the pose stream is unreliable, and fuses the two with a category-aware,
positive-only rule, so the VLM can raise suspicion but never vouch for normality.

On ShanghaiTech this raises the cross-fitted global ROC-AUC from **0.8594** to
**0.8857** (Δ = +0.0263, video-level bootstrap 95% CI [+0.0085, +0.0483]), at
roughly five euros of API inference. Nothing is retrained.

This repository contains the code, the per-frame score tables, and the **raw cached
VLM prompt/response pairs**, so every number, figure and table can be regenerated
offline, with no API key and no re-billing, after the specific model backend is
retired.

> **Relationship to STG-NF.** This repository is built on
> [orhir/STG-NF](https://github.com/orhir/STG-NF) (Hirschorn & Avidan, ICCV 2023),
> whose pose stream, checkpoints and scoring path are used **unmodified**. Upstream
> files (`train_eval.py`, `models/`, `utils/`, `dataset.py`, `data/`) are theirs;
> everything under `scripts/` is this work. The original README is preserved as
> [README_STG-NF_upstream.md](README_STG-NF_upstream.md). See [License](#license).

## Contents

| Path | What it is |
|---|---|
| `scripts/` | All SQ-VAD code: router, VLM client, fusion, cross-fitting, ablations, figures |
| `results/*.parquet` | Per-frame pose, VLM and fused score tables |
| `results/*.json` | Headline numbers behind every claim (see [Results map](#results-map)) |
| `results/figs/` | Generated figures |
| `cache/vlm/` | Raw cached VLM responses, keyed by backend, video, frame, encoding and prompt version |
| `checkpoints/` | Upstream STG-NF checkpoints |
| `SCORING.md`, `EVAL.md` | Scoring path and evaluation protocol notes |

## Install

Python 3.8 (the version the upstream checkpoint and this pipeline were run with).

```bash
conda create -n sqvad python=3.8 && conda activate sqvad
pip install -r requirements.txt
```

`requirements.txt` pins the versions actually used. PyTorch is only needed to
re-run the pose stream; every offline analysis works without a GPU.

**Windows/GPU note.** On some GPUs the STG-NF convolutions raise
`Unable to find a valid cuDNN algorithm`. The model is tiny (~10³ parameters), so
pass `--device cpu`; evaluation still finishes in a few minutes.

## Data

Nothing large is committed. You need:

| Asset | Needed for | Where to get it |
|---|---|---|
| ShanghaiTech pose graphs + checkpoint | re-running the pose stream | [STG-NF](https://github.com/orhir/STG-NF) → `data/ShanghaiTech/pose/` |
| UBnormal pose graphs | UBnormal transfer | same, → `data/UBnormal/pose/` |
| ShanghaiTech RGB frames | re-querying the VLM, qualitative figures | the [ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html) dataset |
| UBnormal videos | UBnormal VLM stream | the [UBnormal](https://github.com/lilygeorgescu/UBnormal) dataset |
| YOLOv8 weights | detector ablation only | downloaded automatically by `ultralytics` on first use |

Scripts that read raw imagery take a `--frames_root` argument (see
[Known rough edges](#known-rough-edges)).

**You do not need any of this to reproduce the published numbers** — the committed
parquet tables and the cached VLM responses are sufficient.

## Reproduce the paper offline (no API key, no GPU, no datasets)

Run everything from the repository root.

```bash
python scripts/make_figs.py
```

That regenerates every figure and table from `results/` alone. The individual
steps, in the order the argument is made:

```bash
# 1. Frozen baseline + the structural blind spot
python scripts/dump_baseline_scores.py          # -> results/baseline_auc.json (global 0.8594, HR 0.8738)
python scripts/error_analysis.py                # recall vs pose coverage; 25.2% of misses are pose-blind

# 2. Main result: cross-fitted fusion (the paper's primary, label-free router)
python scripts/ablation_labelfree_router.py     # -> results/cv_fusion_labelfree.json   <-- HEADLINE
python scripts/cv_fusion.py                     # -> results/cv_fusion.json (label-informed variant)

# 3. What drives the gain, and what does not
python scripts/fuse.py                          # F1 / F2 / F3 fusion rules, lambda and w sweeps
python scripts/plot_lambda_sensitivity.py       # lambda plateau
python scripts/ablation_blind_only.py           # blind tier only: +0.0099, CI touches zero
python scripts/oracle_ceiling.py                # oracle 0.967: headroom is VLM recall, not coverage

# 4. Frozen object detector instead of the VLM
python scripts/compare_detector_vlm.py          # YOLOv8n / YOLOv8x vs VLM, same pipeline

# 5. Cross-dataset transfer to UBnormal
python scripts/blindspot_ubnormal.py            # blind spot transfers, more strongly
python scripts/cv_fusion_ubnormal.py            # fixed trusted set does not transfer
python scripts/adaptive_trust_probe.py \
    --pose results/stgnf_frame_scores_ubnormal.parquet \
    --vlm  results/vlm_frame_scores_ubnormal_haiku_v2.parquet --tag ub-haiku
# adaptive trusted set: +0.0244 (Haiku), +0.0186 (Sonnet), ShanghaiTech unchanged
```

### Results map

Which file backs which claim. Getting this wrong silently swaps the primary system
for an ablation.

| Claim in the paper | File |
|---|---|
| **Primary**: label-free router, Δ = +0.0263, CI [+0.0085, +0.0483], 25 wins / 15 losses | `results/cv_fusion_labelfree.json` |
| Label-informed (Youden-band) variant, 25 / 16 | `results/cv_fusion.json` |
| Baseline, blind-spot statistics | `results/baseline_auc.json` |
| Fusion rules F1/F2/F3, λ and w sweeps | `results/fusion_metrics.json` |
| Oracle ceiling 0.967 | `results/oracle_ceiling.json` |
| Detector ablation | `results/cv_fusion_yolov8{n,x}.json` |
| UBnormal baseline and blind spot | `results/baseline_auc_ubnormal.json`, `results/ubnormal_blindspot.json` |
| UBnormal fixed trusted set | `results/cv_fusion_ubnormal_{haiku,sonnet}.json` |

`cv_fusion.py` writes to `results/cv_fusion.json` by default — always pass `--out`
and `--fig` when running it on an alternative evidence stream, or it overwrites the
headline file.

## Re-running the pose stream

```bash
python train_eval.py --dataset ShanghaiTech    --checkpoint checkpoints/ShanghaiTech_85_9.tar --device cpu
python train_eval.py --dataset ShanghaiTech-HR --checkpoint checkpoints/ShanghaiTech_85_9.tar --device cpu
python train_eval.py --dataset UBnormal --seg_len 16 --checkpoint checkpoints/UBnormal_unsupervised_71_8.tar --device cpu
```

Passing `--checkpoint` makes it evaluation-only. Run from the repository root: the
ground-truth path is relative to the working directory. On Windows add
`--num_workers 0`.

## Re-running the VLM stream (costs money)

**Only needed if you change the prompt, the encoding or the backend.** Reproducing
the published numbers does not require this: every response is already cached.

The API key is read from the `ANTHROPIC_API_KEY` environment variable. **Never put
a key in a file inside the repository.**

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."

python scripts/score_vlm.py \
    --frames_root /path/to/shanghaitech/testing/frames \
    --backend claude_sonnet --model claude-sonnet-4-5 \
    --motion_overlay none --prompt_version v2 \
    --blind_stride 12 --other_stride 60 \
    --out_frame results/vlm_frame_scores_sonnet.parquet \
    --out_query results/vlm_query_results_sonnet.parquet
```

Those flags are the published configuration; **the script defaults are not**
(they are the first Haiku run: `--motion_overlay stack --prompt_version v1
--other_stride 36`). The full Sonnet run is 1,520 calls, about five euros at
June 2026 prices.

Every call is cached under `cache/vlm/<backend>/<video>/<frame>_<encoding>_<hash>.json`,
so a re-run with an unchanged configuration makes no API calls at all. To rebuild a
parquet from the cache without touching the API:

```bash
python scripts/harvest_cache_ubnormal.py --backend claude_haiku --prompt_version v2 \
    --out results/vlm_frame_scores_ubnormal_haiku_v2.parquet \
    --out_query results/vlm_query_results_ubnormal_haiku_v2.parquet
```

## Known rough edges

Honest notes for anyone trying to run this.

- **Hard-coded dataset paths.** `contamination_probe.py`, `qualitative_montage.py`,
  `score_detector.py`, `devset_ubnormal.py`, `scene_profiles_ubnormal.py` and
  `score_vlm_ubnormal.py` still contain an absolute path to the author's machine at
  the top of the file. Edit that constant before running them. The offline
  reproduction path above is unaffected.
- **Two routers.** `router.py` centres its uncertainty band on Youden's *J*, which
  uses test labels; the paper's primary system is the **label-free** median-centred
  band in `ablation_labelfree_router.py`. The two select 1,520 and 1,509 queries and
  agree to the fourth decimal, but only the label-free one is the headline.
- **Script docstrings refer to internal "Phase" numbers** from the project plan
  rather than to sections of the paper.
- `adaptive_trust_probe.py` prints its results and does not write a JSON file.

## License

Upstream STG-NF is released under **Creative Commons Attribution-NonCommercial 4.0**
([LICENSE](LICENSE)), so this derivative work is distributed under the same terms:
free for research and other non-commercial use, with attribution.

The datasets (ShanghaiTech, UBnormal) and the libraries used here carry their own
licenses, which you must also follow. `ultralytics` (used only for the optional
detector ablation) is AGPL-3.0; its weights are not redistributed here.

## Citation

```bibtex
@article{antonio2026sqvad,
  author  = {Ant\'onio, Guilherme and Cardoso, Pedro J. S. and Rodrigues, Jo\~ao M. F.},
  title   = {Closing the Blind Spot of Pose-Based Video Anomaly Detection
             with a Selectively Queried Frozen Vision--Language Model},
  year    = {2026},
  note    = {Submitted}
}
```

Please also cite the baseline this work builds on:

```bibtex
@InProceedings{Hirschorn_2023_ICCV,
  author    = {Hirschorn, Or and Avidan, Shai},
  title     = {Normalizing Flows for Human Pose Anomaly Detection},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023},
  pages     = {13545--13554}
}
```
