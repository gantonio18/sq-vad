# EVAL.md — evaluation protocol (Phase 7)

## Metric (matches the STG-NF repo exactly)
- **Global ROC-AUC = micro AUC**: per-frame scores are concatenated across all
  107 test clips into one vector and a single `roc_auc_score(gt, score)` is
  computed (`utils/scoring_utils.py::score_auc`). We reproduce this number
  (0.85937) and use the identical procedure for every fused system, so all
  numbers are directly comparable.
- **HR AUC**: same, after dropping the 6 non-human-pose clips
  `[(1,130),(1,135),(1,136),(6,144),(6,145),(12,152)]` (the `ShanghaiTech-HR`
  protocol). Reported for every system alongside Global.
- **micro vs macro**: the headline is micro (the repo's choice). `evaluate.py`
  additionally reports per-video AUC (a macro view) for the per-scene
  wins/losses count, per Noghre et al.'s critique that a single micro-AUC can
  hide per-scene behaviour.

## Score processing (identical pose path; see SCORING.md)
1. Per-frame pose score = min log-likelihood across people (`np.amin`),
   canonicalised to `s_pose = -normality` (higher = more anomalous).
2. No-pose frames are the `+inf`→max-normal fill in the baseline.
3. **Fusion order:** z-normalise each stream **globally** (a monotonic transform,
   so each stream's own AUC is unchanged and F1 at w=0 reproduces 0.85937 — this
   is asserted in `fuse.py`), fuse the **unsmoothed** streams, then apply STG-NF's
   iterated Gaussian smoothing (`sigma=1..6`) **once** to the fused score before
   AUC. This mirrors the repo (smoothing is the last step before scoring).

## Systems compared (`results/comparison.csv`, `comparison_table.png`)
| System | Global | HR | Notes |
|---|---|---|---|
| STG-NF (frozen baseline) | reproduced 0.8594 | 0.8738 | anchor |
| VLM only | z(s_vlm) smoothed | — | standalone ranker |
| F1 late fusion (best w) | sweep w∈[0,1] | | |
| F2 gated fusion | g=1 at no-pose, g↑ near boundary | | **primary** |
| F3 max fusion | max(zp, zv) | | |

## Ablations (each a row/figure)
- **No-pose rule on vs off** (`fuse.py --nopose_rule`): isolates the single
  cleanest contribution (`gate_effect.png`).
- **Fusion weight w** sweep (`auc_vs_w.png`).
- **Strategy** F1 vs F2 vs F3 (`roc_overlay.png`).
- **Motion overlay** none vs stack (`score_vlm.py --motion_overlay`): rerun the
  VLM stream with `none` and re-fuse (uses cache; overlay change re-keys cache).
- **Router coverage** full vs selective (cost vs AUC) — vary `--other_stride`.
- (Pose aggregation max vs mean is noted as future work: the dumped per-frame
  score is the repo's `min`-normality = `max`-anomaly; `mean` would require
  re-dumping per-person scores.)

## Significance (`auc_bootstrap.png`, `bootstrap.json`)
- **Video-level bootstrap**, 1000 resamples *of the 107 videos with replacement*;
  for each resample compute Global AUC for baseline and for the best fused system
  and record the delta. Report the 95% percentile CI of the delta and whether it
  **excludes 0**. Video-level (not frame-level) resampling respects the temporal
  correlation within a clip.
- Also report **per-scene wins/losses**: how many of the 107 videos improved.

## Results (headline)

VLM backend = **Claude Sonnet 4.5** (clean RGB + recall/object prompt v2).

| System | Global AUC | HR AUC |
|---|---|---|
| STG-NF (frozen baseline) | 0.8594 | 0.8738 |
| VLM only | 0.702 | — |
| F1 late fusion (best w) | 0.8807 | 0.8828 |
| **F2 category-aware fusion (primary)** | **0.8871** (+0.0277) | 0.8848 |
| F3 max fusion | 0.8851 | 0.8823 |

**Overfitting-free (5-fold cross-fitted, `cv_fusion.py`):** Global **0.8857 (+0.0263)**;
video-bootstrap (2000×) 95% CI **[+0.0084, +0.0483] — excludes 0 (significant)**;
**P(Δ>0)=0.9995**; per-scene 25 wins / 16 losses; clean 50/50 held-out split +0.0352.
HR also improves (0.8738→0.8848), so the VLM helps beyond the non-pose subset.

### Decisive ablations (cross-fitted Δ Global)
| VLM config | Δ Global | significant? |
|---|---|---|
| Haiku, motion-stack overlay + base prompt (v1) | +0.0075 | no (CI incl. 0) |
| Haiku, clean RGB + recall/object prompt (v2) | +0.0160 | yes (P=0.993) |
| Haiku v2 + region-crop tiles (naive) | +0.0121 | no (false distant-bikes) |
| **Sonnet, clean RGB + prompt v2** | **+0.0263** | **yes (P=0.9995)** |

### Router-leakage check (`ablation_blind_only.py`, `ablation_labelfree_router.py`)

The deployed router's UNCERTAIN band is centred on Youden's J (computed from
test GT). Two cache-only ablations bound the impact:

| Router variant | Δ Global (cross-fit) | 95% CI | significant? |
|---|---|---|---|
| Blind tier only (strictest, 970 queries) | +0.0099 | [−0.0007, +0.0269] | no (borderline) |
| **Fully label-free** (median-centred band, 1509/1520 queries) | **+0.0264** | [+0.0085, +0.0483] | **yes (P=0.9995)** |

Replacing Youden's J with the label-free score **median** keeps 97.8% of the
uncertain queries and reproduces the headline delta — the result does not
depend on label-informed routing. The blind-only variant shows the uncertain
tier does contribute (~60% of trusted evidence lives outside blind windows).

Three levers mattered, in order: (1) **frame encoding** — the motion-stack overlay
*ghosted* moving vehicles so the VLM called them "empty"; clean RGB + a
recall-leaning, object-scanning prompt doubled the gain and crossed significance.
(2) **model capability** — Sonnet lifts rescue-target recall 29%→40% at equal/
better precision, again ~doubling the gain. (3) Region-crop and "more frames" did
NOT help (oracle shows coverage already sufficient; zoom adds false distant-bike
detections). Throughout, the binding constraint was VLM **recall** on the rescue
anomalies (Haiku labeled 64% "none"), not coverage or fusion math.

## Reproducibility
- `make_figs.py` regenerates every figure/table from `results/*.parquet` alone —
  no VLM re-query, no flow re-run (brief §8). VLM calls are cached under
  `cache/vlm/` keyed by `(backend, video_id, frame_idx, overlay, image-hash)`.
