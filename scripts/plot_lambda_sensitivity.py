"""Reviewer item #4 — dedicated Delta-AUC vs lambda sensitivity plot.

Reads results/fusion_metrics.json (the cached F2 lambda sweep, in-sample global
ROC-AUC) and draws Delta AUC = AUC(lambda) - baseline vs lambda, highlighting the
broad plateau so the gain is visibly not a knife-edge on one hyperparameter.
Writes ../Article/figures/lambda_sensitivity.png. No API calls, no re-query.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = "results/fusion_metrics.json"
OUT = "../Article/figures/lambda_sensitivity.png"

# cross-fitting selected lambda in {0.8, 2.0} across the 5 folds (cv_fusion.json)
CF_LAMBDAS = [0.8, 2.0]


def main():
    met = json.load(open(METRICS))
    base = met["baseline_global"]
    sweep = met["f2_lambda_sweep"]
    lams = sorted(float(k) for k in sweep)
    delta = [sweep[str(l) if str(l) in sweep else "{:g}".format(l)] - base for l in lams]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    # plateau band: lambda in [0.5, 2.5]
    ax.axvspan(0.5, 2.5, color="tab:green", alpha=0.10, label="broad plateau (0.5-2.5)")
    ax.axhline(0.0, color="k", ls="--", lw=1, alpha=0.6)
    ax.plot(lams, delta, "o-", color="tab:green", lw=2, ms=6, label=r"$\Delta$ Global AUC (in-sample)")
    for cl in CF_LAMBDAS:
        ax.axvline(cl, color="tab:red", ls=":", lw=1.2, alpha=0.8)
    ax.text(1.4, 0.0135, "cross-fitting picks\n" + r"$\lambda\in\{0.8, 2.0\}$",
            color="tab:red", fontsize=10, ha="center", va="center")

    ax.set_xlabel(r"fusion weight $\lambda$", fontsize=12)
    ax.set_ylabel(r"$\Delta$ Global ROC-AUC over baseline", fontsize=12)
    ax.set_title(r"F2 fusion is robust to $\lambda$: a broad plateau, not a knife-edge",
                 fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("wrote", OUT)
    print("delta range over lambda>=0.5: {:.4f}..{:.4f}".format(
        min(d for l, d in zip(lams, delta) if l >= 0.5),
        max(d for l, d in zip(lams, delta) if l >= 0.5)))


if __name__ == "__main__":
    main()
