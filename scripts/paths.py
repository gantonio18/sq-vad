"""Where the raw datasets live, resolved from the environment.

Only the scripts that read raw imagery need these; every offline analysis runs
from `results/` and `cache/` alone and ignores this module.

Set the variables before running (see .env.example):

    export SHANGHAITECH_FRAMES=/data/shanghaitech/testing/frames
    export UBNORMAL_ROOT=/data/UBnormal

On Windows PowerShell:

    $env:SHANGHAITECH_FRAMES="D:/data/shanghaitech/testing/frames"
    $env:UBNORMAL_ROOT="D:/data/UBnormal"

Expected contents:
    $SHANGHAITECH_FRAMES/<video_id>/<frame:03d>.jpg      e.g. 01_0014/000.jpg
    $UBNORMAL_ROOT/Scene<N>/<clip>.mp4
"""
import os

SHANGHAITECH_FRAMES = os.environ.get(
    "SHANGHAITECH_FRAMES", os.path.join("datasets", "shanghaitech", "testing", "frames"))

UBNORMAL_ROOT = os.environ.get("UBNORMAL_ROOT", os.path.join("datasets", "UBnormal"))


def require(path, env_var):
    """Fail with an actionable message instead of an obscure file-not-found."""
    if not os.path.exists(path):
        raise SystemExit(
            "{} not found: {}\n"
            "Set {} to the dataset location, or pass the matching command-line\n"
            "argument. The offline reproduction in the README needs neither."
            .format(env_var, path, env_var))
    return path
