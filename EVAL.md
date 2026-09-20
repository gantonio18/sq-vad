# Evaluation protocol

What is measured, how, and which file holds each number. This matches the paper;
where an earlier draft of this file disagreed, the paper is right.

## Metric

- **Global ROC-AUC (micro).** Per-frame scores are concatenated across all 107
  ShanghaiTech test clips into one vector and scored with a single
  `roc_auc_score` (`utils/scoring_utils.py::score_auc`). We reproduce 0.85937 and
  use the identical procedure for every fused system, so all numbers are
  comparable.
- **HR ROC-AUC.** The same, after dropping the six non-human-pose clips
  `[(1,130),(1,135),(1,136),(6,144),(6,145),(12,152)]`.
- **Micro vs macro.** The headline is micro, as in the STG-NF repository.
  `evaluate.py` also reports per-video AUC, used for the win/loss counts.

## Score path (see SCORING.md)

1. Per-frame pose score = min log-likelihood over detected people, canonicalised
   to `s_pose = -normality`, so higher means more anomalous.
2. Person-free frames keep the repository's `+inf` → max-normal fill.
3. The fused score is `s_pose_smoothed + lambda * trusted_evidence`
   (Equation 1 in the paper); `lambda = 0` reproduces the baseline exactly, which
   `fuse.py` asserts.

## Router

The **primary** router is **label-free**: its uncertain band is centred on the
median of the smoothed pose scores, and it issues **1,509** queries (3.7% of
frames). A label-informed variant centred on Youden's *J* of the global ROC
selects 97.8% of the same queries (1,520) and reproduces the result to the fourth
decimal; it is reported only to show that labels buy nothing here.

## Significance

**Video-level bootstrap**, 2,000 resamples of the 107 videos with replacement, of
the fused − baseline AUC delta. We report the 95% percentile interval, the
fraction of positive resamples, and the per-video win/loss count. Video-level
rather than frame-level resampling respects the temporal correlation within a
clip.

Hyperparameters (trusted set, `lambda`, confidence threshold) are chosen by
**5-fold cross-fitting** over the videos: each fold is scored with parameters
selected on the other four, and the cross-fitted per-frame scores are pooled into
one AUC.

## Results and the file behind each one

| Claim | Value | File |
|---|---|---|
| Frozen baseline | 0.8594 global / 0.8738 HR | `results/baseline_auc.json` |
| **Primary: label-free router, cross-fitted** | **+0.0263**, CI [+0.0085, +0.0483], P=0.9995, 25 W / 15 L | `results/cv_fusion_labelfree.json` |
| Label-informed variant | +0.0263, CI [+0.0084, +0.0483], 25 W / 16 L | `results/cv_fusion.json` |
| Blind tier only | +0.0099, CI [−0.0007, +0.0269], not significant | `results/cv_fusion_blindonly.json` |
| In-sample F1 / F2 / F3 | 0.8807 / 0.8871 / 0.8851 | `results/fusion_metrics.json` |
| VLM alone | 0.7021 | `results/fusion_metrics.json` |
| Oracle ceiling | 0.9670 (+0.1076) | `results/oracle_ceiling.json` |
| Haiku, motion overlay + base prompt | +0.0075, not significant | `results/cv_fusion_v1.json` |
| Haiku, clean RGB + object prompt | +0.0160, significant | `results/cv_fusion_v1.json` |
| YOLOv8n / YOLOv8x | +0.0123 / +0.0129, neither significant | `results/cv_fusion_yolov8{n,x}.json` |
| UBnormal baseline and blind spot | 0.7178; 45.9% of misses pose-blind | `results/baseline_auc_ubnormal.json`, `results/ubnormal_blindspot.json` |
| UBnormal, fixed trusted set | +0.0171 Haiku / −0.0046 Sonnet, neither significant | `results/cv_fusion_ubnormal_{haiku,sonnet}.json` |
| UBnormal, adaptive trusted set | +0.0244 Haiku / +0.0186 Sonnet, both significant | `scripts/adaptive_trust_probe.py` (prints; no JSON) |

## Ablations

Each is a row or a figure, all reading cached artefacts only: no-pose rule on/off
(`fuse.py --nopose_rule`), the fusion-weight sweep, F1 vs F2 vs F3, frame encoding
(clean RGB vs motion overlay), region-crop tiles, router coverage, and the
label-free vs label-informed router.

## Reproducibility

`python reproduce.py all` regenerates every figure and table from
`results/*.parquet` and `cache/vlm/*.jsonl`. No VLM is re-queried, the flow is not
re-run, and no API key is needed.
