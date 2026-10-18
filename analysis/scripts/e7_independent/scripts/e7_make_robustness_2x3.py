from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# VERIFIED INPUT / OUTPUT
# ============================================================

TABLE = Path(
    "/home/ubuntu/afrl_runs/E7_PUBLICATION/tables/"
    "E7_5seed_summary.csv"
)

FIGDIR = Path(
    '/home/ubuntu/afrl_runs/E7_FIGURE_INDEPENDENT_AUDIT/figures'
)

FIGDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD VERIFIED FIVE-SEED SUMMARY
# ============================================================

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

conditions = [
    ("cpu", "CPU impairment"),
    ("network", "Network impairment"),
]

metrics = [
    ("acc", "Accuracy"),
    ("lat", "Latency (s)"),
    ("train", "Training time (s)"),
]

phases = ["PRE", "IMPAIR", "RECOVERY"]
x = [0, 1, 2]


# ============================================================
# VERIFIED CSV ACCESS
# No statistics are recomputed.
# ============================================================

def values(row, metric):
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


# ============================================================
# CREATE 2 x 3 COMPOSITE
# Rows    = CPU / NETWORK
# Columns = Accuracy / Latency / Training time
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(13.5, 7.2),
)

panel_labels = [
    "(a)", "(b)", "(c)",
    "(d)", "(e)", "(f)",
]

panel_index = 0

for row_index, (condition, condition_label) in enumerate(conditions):

    for col_index, (metric, ylabel) in enumerate(metrics):

        ax = axes[row_index][col_index]

        for strategy, strategy_label in strategies:

            source = lookup[(strategy, condition)]
            means, sds = values(source, metric)

            ax.errorbar(
                x,
                means,
                yerr=sds,
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=strategy_label,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(phases)

        ax.set_ylabel(ylabel)

        ax.set_title(
            f"{panel_labels[panel_index]} "
            f"{condition_label} — {ylabel}"
        )

        ax.grid(True, alpha=0.25)

        panel_index += 1


# One shared legend instead of six duplicate legends.
handles, labels = axes[0][0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.01),
)

fig.tight_layout(
    rect=[0, 0, 1, 0.95]
)


# ============================================================
# SAVE PUBLICATION OUTPUT
# ============================================================

png = FIGDIR / "E7_robustness_2x3.png"
pdf = FIGDIR / "E7_robustness_2x3.pdf"

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
print()
print(
    "PASS: E7 2x3 robustness figure generated "
    "from verified five-seed summary CSV"
)
