"""One entry point for everything in the paper that runs offline.

Every step below reads only `results/*.parquet` and `cache/vlm/*.jsonl`, so none
of them needs an API key, a GPU, or the raw datasets.

    python reproduce.py all            # every step, in the order the paper argues
    python reproduce.py main           # just the headline result
    python reproduce.py --list         # what the steps are

Steps that DO need something extra (the pose stream, the VLM stream, the
qualitative montages) are deliberately not here; see the README.
"""
from __future__ import print_function

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")

STEPS = [
    ("baseline",   "scripts/dump_baseline_scores.py",
     "frozen STG-NF baseline: global 0.8594 / HR 0.8738"),
    ("blindspot",  "scripts/error_analysis.py",
     "the structural blind spot: recall vs pose coverage"),
    ("main",       "scripts/ablation_labelfree_router.py",
     "HEADLINE cross-fitted fusion, label-free router: +0.0263"),
    ("fusion",     "scripts/fuse.py",
     "F1/F2/F3 fusion rules and the lambda / w sweeps"),
    ("lambda",     "scripts/plot_lambda_sensitivity.py",
     "sensitivity of the gain to lambda"),
    ("blindonly",  "scripts/ablation_blind_only.py",
     "blind tier only: +0.0099, interval touches zero"),
    ("oracle",     "scripts/oracle_ceiling.py",
     "oracle ceiling 0.967: the headroom is VLM recall"),
    ("detector",   "scripts/compare_detector_vlm.py",
     "frozen YOLOv8 instead of the VLM, same pipeline"),
    ("ubnormal",   "scripts/blindspot_ubnormal.py",
     "the blind spot transfers to UBnormal, more strongly"),
    ("ubfusion",   "scripts/cv_fusion_ubnormal.py",
     "UBnormal fusion with the fixed trusted set (does not transfer)"),
    ("figures",    "scripts/make_figs.py",
     "regenerate every figure and table"),
]


def run(step, script):
    print("\n" + "=" * 72)
    print("[{}] {}".format(step, script))
    print("=" * 72)
    env = dict(os.environ)
    env["PYTHONPATH"] = SCRIPTS + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    rc = subprocess.call([sys.executable, script], cwd=HERE, env=env)
    print("[{}] {} in {:.0f}s".format(step, "OK" if rc == 0 else "FAILED", time.time() - t0))
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", default=["all"],
                    help="step names, or 'all' (default)")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    a = ap.parse_args()

    if a.list:
        for name, script, desc in STEPS:
            print("  {:11s} {:42s} {}".format(name, script, desc))
        return 0

    wanted = [s for s in STEPS] if "all" in a.steps else \
             [s for s in STEPS if s[0] in a.steps]
    unknown = set(a.steps) - {s[0] for s in STEPS} - {"all"}
    if unknown:
        raise SystemExit("unknown step(s): {}. Try --list.".format(", ".join(sorted(unknown))))

    failed = [name for name, script, _ in wanted if run(name, script) != 0]
    print("\n" + "=" * 72)
    if failed:
        print("FAILED: {}".format(", ".join(failed)))
        return 1
    print("all {} step(s) completed".format(len(wanted)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
