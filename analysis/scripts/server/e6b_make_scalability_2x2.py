from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ubuntu/afrl_runs/E6B_FINAL")
CSV = ROOT / "E6B_5seed_summary.csv"

FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

PNG = FIGDIR / "E6B_controller_scalability_2x2.png"
PDF = FIGDIR / "E6B_controller_scalability_2x2.pdf"

if not CSV.exists():
    raise FileNotFoundError(f"Missing E6-B summary: {CSV}")

with CSV.open(newline="") as f:
    rows = list(csv.DictReader(f))

rows = sorted(rows, key=lambda r: int(r["N"]))

N = [int(r["N"]) for r in rows]

controller_med = [
    float(r["controller_ms_median_mean"]) for r in rows
]
controller_med_sd = [
    float(r["controller_ms_median_sd"]) for r in rows
]

controller_p95 = [
    float(r["controller_ms_p95_mean"]) for r in rows
]
controller_p95_sd = [
    float(r["controller_ms_p95_sd"]) for r in rows
]

selection_med = [
    float(r["selection_ms_median_mean"]) for r in rows
]
selection_sd = [
    float(r["selection_ms_median_sd"]) for r in rows
]

offload_med = [
    float(r["offload_ms_median_mean"]) for r in rows
]
offload_sd = [
    float(r["offload_ms_median_sd"]) for r in rows
]

selection_params = [
    int(r["selection_policy_parameters"]) for r in rows
]
offload_params = [
    int(r["offload_policy_parameters"]) for r in rows
]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    2, 2,
    figsize=(9.2, 6.7),
    constrained_layout=True,
)

axes = axes.ravel()

# ------------------------------------------------------------
# (a) Median controller latency
# ------------------------------------------------------------

axes[0].errorbar(
    N,
    controller_med,
    yerr=controller_med_sd,
    marker="o",
    linewidth=1.8,
    markersize=6,
    capsize=4,
)

axes[0].set_title(
    "(a) Median controller latency",
    loc="left",
    fontweight="bold",
)
axes[0].set_xlabel("Logical number of clients (N)")
axes[0].set_ylabel("Latency (ms)")
axes[0].set_xticks(N)


# ------------------------------------------------------------
# (b) p95 controller latency
# ------------------------------------------------------------

axes[1].errorbar(
    N,
    controller_p95,
    yerr=controller_p95_sd,
    marker="o",
    linewidth=1.8,
    markersize=6,
    capsize=4,
)

axes[1].set_title(
    "(b) p95 controller latency",
    loc="left",
    fontweight="bold",
)
axes[1].set_xlabel("Logical number of clients (N)")
axes[1].set_ylabel("Latency (ms)")
axes[1].set_xticks(N)


# ------------------------------------------------------------
# (c) Controller latency decomposition
# ------------------------------------------------------------

axes[2].errorbar(
    N,
    selection_med,
    yerr=selection_sd,
    marker="o",
    linewidth=1.8,
    markersize=6,
    capsize=4,
    label="Selection",
)

axes[2].errorbar(
    N,
    offload_med,
    yerr=offload_sd,
    marker="s",
    linewidth=1.8,
    markersize=6,
    capsize=4,
    label="Offloading",
)

axes[2].set_title(
    "(c) Median latency decomposition",
    loc="left",
    fontweight="bold",
)
axes[2].set_xlabel("Logical number of clients (N)")
axes[2].set_ylabel("Latency (ms)")
axes[2].set_xticks(N)
axes[2].legend(frameon=False)


# ------------------------------------------------------------
# (d) Policy parameter growth
# ------------------------------------------------------------

axes[3].plot(
    N,
    selection_params,
    marker="o",
    linewidth=1.8,
    markersize=6,
    label="Selection policy",
)

axes[3].plot(
    N,
    offload_params,
    marker="s",
    linewidth=1.8,
    markersize=6,
    label="Offloading policy",
)

axes[3].set_title(
    "(d) Policy parameter growth",
    loc="left",
    fontweight="bold",
)
axes[3].set_xlabel("Logical number of clients (N)")
axes[3].set_ylabel("Number of parameters")
axes[3].set_xticks(N)
axes[3].legend(frameon=False)


# ------------------------------------------------------------
# Common formatting
# ------------------------------------------------------------

for ax in axes:
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.4,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "AFRL Controller-Only Logical Scalability",
    fontsize=13,
    fontweight="bold",
)

fig.text(
    0.5,
    0.002,
    "Logical N only; no physical FL clients are executed. "
    "Latency points show 5-seed mean ± SD.",
    ha="center",
    fontsize=8.5,
)

fig.savefig(PNG, bbox_inches="tight")
fig.savefig(PDF, bbox_inches="tight")

plt.close(fig)

print("=== E6-B PUBLICATION FIGURE CREATED ===")
print("Input :", CSV)
print("N     :", N)
print("PNG   :", PNG)
print("PDF   :", PDF)
print("PNG exists:", PNG.exists())
print("PDF exists:", PDF.exists())
print("PNG bytes :", PNG.stat().st_size if PNG.exists() else 0)
print("PDF bytes :", PDF.stat().st_size if PDF.exists() else 0)
