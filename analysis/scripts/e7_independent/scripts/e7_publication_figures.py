from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TABLE = Path(
    "/home/ubuntu/afrl_runs/E7_PUBLICATION/tables/"
    "E7_5seed_summary.csv"
)

FIGDIR = Path(
    '/home/ubuntu/afrl_runs/E7_FIGURE_INDEPENDENT_AUDIT/figures'
)

FIGDIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load already-verified publication statistics
# ------------------------------------------------------------

with TABLE.open(newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 9:
    raise RuntimeError(
        f"Expected 9 strategy-condition rows, found {len(rows)}"
    )

lookup = {
    (r["strategy"], r["condition"]): r
    for r in rows
}

strategies = [
    ("fedavg", "FedAvg"),
    ("tl_ppo", "TL-PPO"),
    ("tr_dp_afrl", "TR-DP-AFRL"),
]

conditions = ["none", "cpu", "network"]

phases = ["PRE", "IMPAIRMENT", "RECOVERY"]
x = [0, 1, 2]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def values(row, metric):
    """
    Read phase-wise mean and sample-SD values directly from the
    verified E7_5seed_summary.csv.

    No statistics are recomputed here.
    """

    means = [
        float(row[f"{metric}_pre_mean"]),
        float(row[f"{metric}_impair_mean"]),
        float(row[f"{metric}_recovery_mean"]),
    ]

    sds = [
        float(row[f"{metric}_pre_sd"]),
        float(row[f"{metric}_impair_sd"]),
        float(row[f"{metric}_recovery_sd"]),
    ]

    return means, sds


def make_phase_figure(
    condition,
    metric,
    ylabel,
    filename,
):
    """
    Generate one PRE -> IMPAIRMENT -> RECOVERY figure for a
    specified E7 condition.
    """

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for strategy, label in strategies:

        row = lookup[(strategy, condition)]
        means, sds = values(row, metric)

        ax.errorbar(
            x,
            means,
            yerr=sds,
            marker="o",
            linewidth=2,
            capsize=4,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(phases)

    ax.set_xlabel("Experiment phase")
    ax.set_ylabel(ylabel)

    title_condition = {
        "none": "No impairment",
        "cpu": "CPU impairment",
        "network": "Network impairment",
    }[condition]

    ax.set_title(title_condition)

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()

    png = FIGDIR / f"{filename}.png"
    pdf = FIGDIR / f"{filename}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("WROTE:", png)
    print("WROTE:", pdf)


# ------------------------------------------------------------
# Main publication figures
# ------------------------------------------------------------

for condition in conditions:

    make_phase_figure(
        condition,
        "acc",
        "Mean test accuracy",
        f"E7_accuracy_{condition}",
    )

    make_phase_figure(
        condition,
        "lat",
        "Mean round latency (s)",
        f"E7_latency_{condition}",
    )

    make_phase_figure(
        condition,
        "train",
        "Mean training time (s)",
        f"E7_training_time_{condition}",
    )


print()
print(
    "PASS: E7 publication figures generated "
    "from verified 5-seed summary CSV"
)
