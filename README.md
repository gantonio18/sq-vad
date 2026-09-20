# SQ-VAD — Selectively Queried Video Anomaly Detection

Code and data for *"Closing the Blind Spot of Pose-Based Video Anomaly Detection
with a Selectively Queried Frozen Vision–Language Model"*.

A pose-based detector can only score what the pose estimator sees: person-free
frames are scored as maximally normal, and a van or a cart on a walkway produces no
signal at all. SQ-VAD keeps the pose model (STG-NF) **frozen**, queries a **frozen**
VLM on only the 3.7% of frames where the pose stream is unreliable, and fuses the
two so the VLM can raise suspicion but never vouch for normality.

On ShanghaiTech this raises the cross-fitted global ROC-AUC from **0.8594** to
**0.8857** (Δ = +0.0263, 95% CI [+0.0085, +0.0483]), for about five euros of API
inference. Nothing is retrained.

Built on [orhir/STG-NF](https://github.com/orhir/STG-NF) (Hirschorn & Avidan, ICCV
2023), used unmodified; their README is kept as
[README_STG-NF_upstream.md](README_STG-NF_upstream.md).

## Install

```bash
conda create -n sqvad python=3.8 && conda activate sqvad
pip install -r requirements.txt
```

## Reproduce the results

No API key, GPU or dataset needed: the per-frame score tables (`results/`) and the
cached VLM responses (`cache/vlm/*.jsonl`) are committed. Run from the repository
root.

```bash
python reproduce.py all
```

`python reproduce.py --list` shows the eleven steps; each can be run on its own,
for example `python reproduce.py main figures`. The ones worth knowing:

```bash
python reproduce.py main       # the headline: +0.0263, CI [+0.0085,+0.0483]
python reproduce.py oracle     # oracle 0.967 -> the headroom is VLM recall
python reproduce.py detector   # frozen YOLOv8 instead of the VLM
python reproduce.py ubnormal   # the blind spot transfers, more strongly
python reproduce.py figures    # every figure and table
```

`results/cv_fusion_labelfree.json` backs the paper's primary system (label-free
router). `results/cv_fusion.json` is the label-informed variant — the two agree to
the fourth decimal, but only the first is the headline.

## Re-running the pose or VLM stream

Optional, and only needed to change the method. The pose stream needs the STG-NF
pose graphs; the VLM stream needs the RGB frames and costs money.

```bash
python train_eval.py --dataset ShanghaiTech --checkpoint checkpoints/ShanghaiTech_85_9.tar --device cpu

export ANTHROPIC_API_KEY=sk-ant-...
python scripts/score_vlm.py --frames_root /path/to/shanghaitech/testing/frames \
    --backend claude_sonnet --model claude-sonnet-4-5 \
    --motion_overlay none --prompt_version v2 --blind_stride 12 --other_stride 60 \
    --out_frame results/vlm_frame_scores_sonnet.parquet \
    --out_query results/vlm_query_results_sonnet.parquet
```

Those flags are the published configuration; the script defaults are the earlier
Haiku run. Responses are cached, so re-running an unchanged configuration costs
nothing.

Scripts that read raw imagery take their dataset locations from the environment,
`SHANGHAITECH_FRAMES` and `UBNORMAL_ROOT`; copy [.env.example](.env.example) and
fill it in. The API key is read from `ANTHROPIC_API_KEY` and nowhere else — never
commit one.

**One caveat:** `cv_fusion.py` overwrites `results/cv_fusion.json` unless you pass
`--out`, so always pass it when running an alternative evidence stream.

## License

Upstream STG-NF is **CC BY-NC 4.0** ([LICENSE](LICENSE)), so this derivative is too:
research and other non-commercial use, with attribution. The datasets and
`ultralytics` (AGPL-3.0, optional) carry their own terms.

## Citation

```bibtex
@article{antonio2026sqvad,
  author = {Ant\'onio, Guilherme and Cardoso, Pedro J. S. and Rodrigues, Jo\~ao M. F.},
  title  = {Closing the Blind Spot of Pose-Based Video Anomaly Detection
            with a Selectively Queried Frozen Vision--Language Model},
  year   = {2026},
  note   = {Submitted}
}
```

Please also cite [STG-NF](https://github.com/orhir/STG-NF) (Hirschorn & Avidan, ICCV 2023).
