# error_analysis.md — Where the frozen STG-NF fails (Phase 3)

Operating point: Youden's J on the global ROC, threshold `s_official >= 1.1061`.

Buckets approximate the brief's A/B/C/D by **detected-pose coverage**, the structural axis that makes the flow blind (ShanghaiTech ships no per-type labels). The VLM's Phase-4 `category` output will let us cross-tabulate by true type later.

## Anomaly frames & STG-NF recall by pose coverage

| Bucket | # anomaly frames | STG-NF recall |
|---|---|---|
| C: no-pose (0) | 598 | 0.184 |
| B: low (1) | 1365 | 0.508 |
| mid (2-3) | 5060 | 0.697 |
| A: high (4+) | 10303 | 0.814 |

## The spine sentence

> Of **4606** missed anomalous frames (false negatives at the Youden point), **1159 (25.2%)** occur where STG-NF is structurally blind (≤1 detected pose) — exactly the frames a VLM appearance/context stream can address.

Recall is **0.81** on high-pose-coverage anomalies (4+ people) but **0.18** on no-pose anomalies — the latter are silently scored as *max-normal* by the `+inf`→max fill (see SCORING.md).

## Artifacts

- `results/figs/error_analysis/failure_buckets.png`
- `results/figs/error_analysis/score_vs_npose.png`
- `results/figs/error_analysis/fn_gallery/montage.png` — montage of worst pose-present false negatives