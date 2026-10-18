#!/usr/bin/env python3
"""Independently reproduce the E5 four-panel publication PNG."""

from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summary(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    return rows[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = load_summary(args.summary)

    assert summary["n_seeds"] == "5"
    assert summary["steady_round_start"] == "251"
    assert summary["steady_round_end"] == "300"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output = (
        args.output_dir /
        "E5_final_physical_deployment_2x2.png"
    )

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing figure: {output}"
        )

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
    })

    fig, axes = plt.subplots(
        2, 2, figsize=(11.27, 2492 / 300)
    )

    def draw(ax, labels, metrics, factor, title,
             ylabel, fmt, percent=False):

        means = [
            float(summary[f"{m}_mean"]) * factor
            for m in metrics
        ]

        sds = [
            float(summary[f"{m}_sd"]) * factor
            for m in metrics
        ]

        x = np.arange(len(labels))

        bars = ax.bar(
            x,
            means,
            yerr=sds,
            capsize=4,
            color="tab:blue",
            edgecolor="black",
            linewidth=0.6,
            error_kw={"elinewidth": 1.2},
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

        top = max(
            (mean + sd for mean, sd in zip(means, sds)),
            default=1,
        )

        ax.set_ylim(0, top * 1.10 if top else 1)

        for bar, mean, sd in zip(bars, means, sds):
            label = format(mean, fmt)
            if percent:
                label += "%"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + sd + top * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    draw(
        axes[0, 0],
        [
            "Physical\nphase",
            "Max physical\nrequest",
            "Max client\ncompute",
            "Cloud\ncompute",
        ],
        [
            "physical_phase_wall_s",
            "physical_request_max_s",
            "client_compute_max_s",
            "cloud_compute_s",
        ],
        1,
        "(a) Physical execution time",
        "Time (s)",
        ".2f",
    )

    draw(
        axes[0, 1],
        ["Total", "Download", "Upload"],
        [
            "communication_bytes_total",
            "physical_download_bytes",
            "physical_upload_bytes",
        ],
        1e-6,
        "(b) Communication cost",
        "Communication (MB/round)",
        ".1f",
    )

    draw(
        axes[1, 0],
        ["Controller", "Selection", "Offload", "Telemetry"],
        [
            "controller_decision_time_s",
            "selection_decision_time_s",
            "offload_decision_time_s",
            "telemetry_overhead_ms",
        ],
        1,
        "(c) Decision and telemetry overhead",
        "Overhead (ms)",
        ".2f",
    )

    # The first three controller metrics are in seconds,
    # while telemetry is already recorded in milliseconds.
    ax = axes[1, 0]
    ax.clear()

    metrics = [
        "controller_decision_time_s",
        "selection_decision_time_s",
        "offload_decision_time_s",
        "telemetry_overhead_ms",
    ]

    factors = [1000, 1000, 1000, 1]

    means = [
        float(summary[f"{m}_mean"]) * factor
        for m, factor in zip(metrics, factors)
    ]

    sds = [
        float(summary[f"{m}_sd"]) * factor
        for m, factor in zip(metrics, factors)
    ]

    x = np.arange(4)

    bars = ax.bar(
        x, means, yerr=sds,
        capsize=4,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Controller", "Selection", "Offload", "Telemetry"]
    )
    ax.set_title("(c) Decision and telemetry overhead")
    ax.set_ylabel("Overhead (ms)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 44.5)

    for i, bar in enumerate(bars):
        value = means[i]
        label = (
            f"{value:.3f}" if i == 2
            else f"{value:.2f}"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + sds[i] + 1.0,
            label,
            ha="center",
            fontsize=9,
        )

    draw(
        axes[1, 1],
        ["10 s", "12 s", "15 s"],
        [
            "sla_violation_10s",
            "sla_violation_12s",
            "sla_violation_15s",
        ],
        100,
        "(d) SLA violation rate",
        "Violation rate (%)",
        ".1f",
        percent=True,
    )

    fig.suptitle(
        "E5: Physical Deployment Performance (N=10, 5 Seeds)"
    )

    fig.text(
        0.5, 0.025,
        "Bars show mean ± SD across five seeds; "
        "steady-state rounds 251–300.",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    fig.savefig(output, dpi=300)
    plt.close(fig)

    print("=" * 70)
    print("E5 INDEPENDENT FIGURE REPRODUCTION")
    print("=" * 70)
    print("Source:", args.summary)
    print("Output:", output)
    print("Panels: 4")
    print("Seeds: 5")
    print("Steady-state rounds: 251–300")
    print("STATUS: PNG GENERATED")
    print("Original publication figure unchanged")


if __name__ == "__main__":
    main()
