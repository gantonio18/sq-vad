# SCORING.md — Canonical anomaly score & the STG-NF scoring path (verified from source)

> All facts below were read directly from `utils/scoring_utils.py` and
> `dataset.py` of this exact clone (commit recorded in `SETUP_NOTES.md`),
> **not** from the design brief. Where the brief and the code disagree, the code wins.

## 1. What the model emits

`Trainer.test()` (`models/training.py`) returns `normality_scores`: one scalar
per pose-segment (a sliding window of `seg_len=24` frames for one tracked person).
The flow is trained so that **normal poses get HIGH log-likelihood**. Therefore:

- `normality` (raw model output): **higher = more normal**.
- An anomaly is a *low-likelihood* segment.

## 2. From segments to per-frame scores (`get_clip_score`)

For each test clip:

1. A buffer `scores_zeros = np.ones(n_frames) * np.inf` is created **per person**.
2. Each person's segment normality is written at frame index
   `pid_frame_inds + seg_len/2` (the segment's center).
   → Frames a person was never scored at stay `+inf`.
3. Across people: `clip_score = np.amin(clip_ppl_score_arr, axis=0)`
   → the **least-normal person sets the frame score** (min normality = max anomaly).
4. GT is loaded from `data/ShanghaiTech/gt/test_frame_mask/<clip>.npy` and **flipped**:
   `clip_gt = 1 - clip_gt`, so in the code's convention **1 = normal, 0 = abnormal**.

## 3. The no-pose blind spot (the central lever — `get_dataset_scores`)

After concatenating all clips:

```python
scores_np[scores_np ==  np.inf] = scores_np[scores_np !=  np.inf].max()   # line ~55
scores_np[scores_np == -np.inf] = scores_np[scores_np != -np.inf].min()
```

**Every frame with zero detected poses is `+inf` and is replaced by the global
maximum finite normality — i.e. scored as the MOST NORMAL frame in the dataset.**
This is verified, not assumed. These are pure false-negative generators that only
exist in the Global set, and they are the primary target of the VLM stream.

`n_poses_detected` per frame = number of persons with a non-`inf` entry at that
frame (recoverable from the per-person buffers before the `amin`).

## 4. Normalization & smoothing

- **No z-score / min-max normalization is applied anywhere.** The only
  post-processing is iterated Gaussian temporal smoothing per clip:
  `smooth_scores` runs `gaussian_filter1d` for `sigma = 1..6` (default arg `sigma=7`
  → `range(1, 7)`), applied AFTER the inf-fill, BEFORE concatenation/AUC.
- Order of operations in `score_dataset`:
  per-clip `amin` → concat → global inf-fill → redistribute to per-clip
  → per-clip iterated smoothing → concat → `roc_auc_score`.

## 5. AUC convention (`score_auc`)

```python
roc_auc_score(gt, scores_np)   # gt: 1=normal,0=abnormal ; scores: higher=more normal
```

Because both gt and score increase with "normality", a higher normality score
ranks correctly against the (flipped) gt. The reported number matches the paper's
**Global ROC-AUC**.

## 6. Canonical anomaly score for THIS project

To make STG-NF commensurable with a bounded VLM signal we define **one** canonical
score, used everywhere downstream:

> **`s_pose(t) = − normality(t)`  →  higher = MORE anomalous.**

- We assert this sign in code (`assert` on a known anomalous clip having higher
  mean `s_pose` than a normal clip).
- We also define a project-canonical **gt**: `y(t) = 1 - clip_gt = 1` for anomaly,
  `0` for normal (the un-flipped original mask), so "higher score ↔ label 1".
- For AUC we therefore call `roc_auc_score(y_anom, s_pose)`, which is numerically
  identical to the repo's `roc_auc_score(clip_gt, normality)` (both flips cancel).
  This identity is unit-tested in `scripts/dump_baseline_scores.py`.

## 7. Where fusion must intervene (preview of Phase 6)

- **No-pose frames** must NOT take the `max-normal` fill; under gated/max fusion
  they take the VLM score instead. Quantifying the Global-AUC delta from this one
  rule is a clean, defensible contribution.
- Both streams are z-scored (per the brief's §6.1) BEFORE fusion, since STG-NF
  emits raw unnormalized log-likelihoods while the VLM emits a bounded [0,1] score.
