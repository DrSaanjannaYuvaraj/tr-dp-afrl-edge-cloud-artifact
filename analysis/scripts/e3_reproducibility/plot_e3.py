import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "tables/E3_publication_table.csv"

OUTDIR = ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "E3_final_reward_sensitivity_2x2.png"
PDF = OUTDIR / "E3_final_reward_sensitivity_2x2.pdf"

with INPUT.open(newline="") as f:
    rows = list(csv.DictReader(f))

order = [
    "reference",
    "accuracy_low",
    "accuracy_high",
    "latency_low",
    "latency_high",
    "offload_low",
    "offload_high",
    "stability_low",
    "stability_high",
]

lookup = {r["configuration"]: r for r in rows}
rows = [lookup[x] for x in order]

labels = [
    "Ref.",
    r"$\alpha$ Low",
    r"$\alpha$ High",
    r"$\beta$ Low",
    r"$\beta$ High",
    r"$\gamma$ Low",
    r"$\gamma$ High",
    r"$\lambda$ Low",
    r"$\lambda$ High",
]

metrics = [
    (
        "acc_mean",
        "acc_sd",
        "(a) Accuracy",
        "Accuracy",
    ),
    (
        "latency_s_mean",
        "latency_s_sd",
        "(b) Latency",
        "Latency (s)",
    ),
    (
        "offload_intensity_mean",
        "offload_intensity_sd",
        "(c) Offload Intensity",
        "Offload intensity",
    ),
    (
        "churn_ratio_mean",
        "churn_ratio_sd",
        "(d) Churn Ratio",
        "Churn ratio",
    ),
]

x = np.arange(len(rows))

fig, axes = plt.subplots(
    2, 2,
    figsize=(12.5, 8.8),
    constrained_layout=True
)

axes = axes.flatten()

for ax, (mean_key, sd_key, title, ylabel) in zip(axes, metrics):

    means = np.array(
        [float(r[mean_key]) for r in rows]
    )

    sds = np.array(
        [float(r[sd_key]) for r in rows]
    )

    ax.errorbar(
        x,
        means,
        yerr=sds,
        fmt="o-",
        linewidth=1.8,
        markersize=5,
        capsize=4,
    )

    # Reference configuration
    ref = means[0]

    ax.axhline(
        ref,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="Reference"
    )

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(ylabel)

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right"
    )

    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.45
    )

    ax.legend(
        loc="best",
        fontsize=8,
        frameon=False
    )

fig.suptitle(
    "E3: Reward-Weight Sensitivity of TR-DP-AFRL",
    fontsize=15,
    fontweight="bold"
)

fig.text(
    0.5,
    -0.01,
    "Mean ± SD over 5 seeds; steady-state rounds 251–300",
    ha="center",
    fontsize=10
)

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    PDF,
    bbox_inches="tight"
)

plt.close(fig)

print("=" * 68)
print(" E3 FINAL PUBLICATION FIGURE CREATED")
print("=" * 68)
print("Input :", INPUT)
print("PNG   :", PNG)
print("PDF   :", PDF)
print("PNG bytes:", PNG.stat().st_size)
print("PDF bytes:", PDF.stat().st_size)
print("=" * 68)
