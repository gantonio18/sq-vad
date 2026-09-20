# SETUP_NOTES.md — Phase 1 (environment + repo read)

## Repo
- Fresh clone of the public baseline: **https://github.com/orhir/STG-NF.git**
- Commit: `edb5f3220332e160e4d20ce258787d5e2d7e0200` (2023-10-13), depth-1.
- Location: `ThesisProject/STG-NF/` (per the project decision: literal fresh start,
  code lives under `ThesisProject/`).

## The REAL entrypoints (verified, not from the brief)
- **Eval-only** (no training) is the same script as training; passing
  `--checkpoint` makes it eval-only (`train_eval.py:34-44`, loads test set only
  when a checkpoint is given):
  ```
  python train_eval.py --dataset ShanghaiTech    --checkpoint checkpoints/ShanghaiTech_85_9.tar
  python train_eval.py --dataset ShanghaiTech-HR --checkpoint checkpoints/ShanghaiTech_85_9.tar
  ```
- Default `--seg_len 24`, `--seg_stride 6`, `--num_workers 8`. On Windows use
  `--num_workers 0` (DataLoader spawn issues otherwise).
- Scoring entrypoint: `utils/scoring_utils.py::score_dataset` → returns
  `(auc, scores_np)`. **GT path is HARDCODED** to
  `data/ShanghaiTech/gt/test_frame_mask/` relative to cwd — must run from repo root.
- HR subset = ShanghaiTech minus 6 clips, list verified in `dataset.py:12`:
  `SHANGHAITECH_HR_SKIP = [(1,130),(1,135),(1,136),(6,144),(6,145),(12,152)]`.

## Baseline reproduction (Phase 2 headline, confirmed)
- **Global ROC-AUC = 85.937%** (`python train_eval.py --dataset ShanghaiTech ...`),
  vs paper 85.93% / checkpoint name `ShanghaiTech_85_9.tar`. **Within 0.5 pt → DoD met.**
- 40,791 scored frames over 107 test clips.
- HR AUC + per-frame parquet produced by `scripts/dump_baseline_scores.py`
  (see `results/baseline_auc.json`).

## Environment (deviation from the brief — documented)
The brief says create a new `stgnf_vlm` conda env. The `environment.yml` is
Linux-pinned (cudatoolkit 10.2 / pytorch 1.10.1, `prefix: /home/orhir/...`) and
will not build cleanly on this Windows host. A working CUDA env already exists and
is **reused**:
- Interpreter: `C:\ProgramData\anaconda3\envs\stgnf\python.exe`
- **python 3.8.20, torch 1.12.1, CUDA available, GPU = NVIDIA GeForce RTX 2060.**
- Core stack already present: numpy 1.23.5, scipy 1.10.1, scikit-learn 1.3.2,
  opencv 4.13, matplotlib 3.7.5, tqdm, pyyaml.
- Installed for this project: **pandas 2.0.3, pyarrow 17.0.0 (parquet), seaborn
  0.13.2, anthropic 0.72.0**.
- Full snapshot: `env_report.txt`.
- This is the same kind of pragmatic reuse as data/checkpoint below; it does NOT
  touch the model weights or the design — those are built fresh per the rule.

## Reused external assets (NOT prior pipeline — these are the repo's own downloads)
The fresh-start rule forbids reusing the *prior pipeline/scripts/design*. It does
not require re-downloading ~2.6 GB of poses or the checkpoint, which are the
baseline repo's own external artifacts. Reused:
- **Checkpoint**: `checkpoints/ShanghaiTech_85_9.tar` (shipped inside the clone;
  byte-identical to the previously downloaded copy).
- **Pose + GT**: `data/ShanghaiTech/gt/test_frame_mask/` (107 masks, shipped in
  clone) and `data/ShanghaiTech/pose/{test,train}` wired via a **directory
  junction** → `..\..\STG-NF-baseline\data\ShanghaiTech\pose` (213 test files).
- **RGB frames** (for the Phase 4 VLM stream): present at
  `..\..\Datasets\shanghaitech\testing\frames` (one folder per `scene_clip`).

## What is built fresh (per the fresh-start rule)
Everything downstream is written from scratch in `scripts/`, not reused from the
older `STG-NF-baseline` work and not using any cached VLM verdicts:
`dump_baseline_scores.py` (Phase 2), and the upcoming `router.py`,
`vlm_client.py`, `score_vlm.py`, `fuse.py`, `evaluate.py`, `make_figs.py`.

## Canonical conventions
- See `SCORING.md` for the verified scoring path, the `+inf→max-normal` no-pose
  fill (the central lever), and the project-canonical anomaly score
  `s_pose = -normality` (higher = more anomalous).

## Phase 1 Definition of Done — status
- [x] Env activates and repo imports; baseline runs end-to-end.
- [x] `SETUP_NOTES.md` and `env_report.txt` exist.
- [x] Baseline AUC reproduced within 0.5 pt (85.937% vs 85.93%).
