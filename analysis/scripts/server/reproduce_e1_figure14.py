#!/usr/bin/env python3
"""Reconstruct FGCS E1 Figure 14 from final per-round JSON results.

This is a newly reconstructed reproducibility script, NOT the original
figure-generation source. It never changes experiment inputs or existing
publication figures. Its four panels show mean ± sample SD across five
seed-level averages over communication rounds 251–300.

Example:
  python reproduce_e1_figure14.py \
    --input-dir /home/ubuntu/afrl_runs/E1_FINAL_RERUN_300R_5SEED \
    --output-dir /home/ubuntu/afrl_runs/E1_REPRODUCED_FIGURE14
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STRATEGIES = ("tl_ppo", "tr_dp_afrl")
LABELS = ("TL-PPO", "TR-DP-AFRL")
SEEDS = (42, 43, 44, 45, 46)
METRICS = (
    ("acc", "Accuracy", "Accuracy", ".4f"),
    ("latency_s", "Latency", "Latency (s)", ".2f"),
    ("train_time_s", "Training Time", "Time (s)", ".2f"),
    ("offload_intensity", "Offload Intensity", "Offload intensity", ".3f"),
)


def load_seed_means(input_dir: Path, first_round: int, last_round: int):
    records = []
    for strategy in STRATEGIES:
        for seed in SEEDS:
            path = input_dir / f"results_{strategy}_E1_MAIN_COMPARISON_seed{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing required final-run JSON: {path}")
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            rounds = data.get("round")
            if not isinstance(rounds, list) or rounds != list(range(1, 301)):
                raise ValueError(f"Expected round sequence 1..300: {path}")
            rec = {"strategy": strategy, "seed": seed}
            for key, _, _, _ in METRICS:
                values = data.get(key)
                if not isinstance(values, list) or len(values) != 300:
                    raise ValueError(f"Expected 300 entries for {key}: {path}")
                subset = np.asarray(values[first_round - 1:last_round], dtype=float)
                if not np.isfinite(subset).all():
                    raise ValueError(f"Nonfinite values for {key}: {path}")
                rec[f"{key}_steady_mean"] = float(np.mean(subset))
            records.append(rec)
    return records


def aggregate(records):
    summary = {}
    for strategy in STRATEGIES:
        group = [r for r in records if r["strategy"] == strategy]
        summary[strategy] = {}
        for key, _, _, _ in METRICS:
            vals = [r[f"{key}_steady_mean"] for r in group]
            summary[strategy][key] = (statistics.mean(vals), statistics.stdev(vals))
    return summary


def write_csvs(records, summary, output_dir: Path):
    with (output_dir / "E1_figure14_per_seed_reproduced.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "E1_figure14_summary_reproduced.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["strategy"]
        for key, _, _, _ in METRICS:
            fields += [f"{key}_mean", f"{key}_sd"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for strategy in STRATEGIES:
            row = {"strategy": strategy}
            for key, _, _, _ in METRICS:
                row[f"{key}_mean"], row[f"{key}_sd"] = summary[strategy][key]
            writer.writerow(row)


def make_figure(summary, output_dir: Path, first_round: int, last_round: int):
    fig, axs = plt.subplots(2, 2, figsize=(14.6, 10.6))
    fig.suptitle(
        f"E1: TL-PPO vs TR-DP-AFRL (Steady-State Rounds {first_round}–{last_round}, Five Seeds)",
        fontsize=17, fontweight="bold", y=0.985
    )
    for ax, (key, title, ylabel, fmt) in zip(axs.flat, METRICS):
        means = [summary[s][key][0] for s in STRATEGIES]
        sds = [summary[s][key][1] for s in STRATEGIES]
        xs = np.arange(len(STRATEGIES))
        ax.bar(xs, means, yerr=sds, width=0.62, capsize=6,
               color="tab:blue", edgecolor="black", linewidth=1.0)
        ax.set_xticks(xs, LABELS)
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        # Use a nonzero baseline for accuracy, as in the existing Figure 14.
        if key == "acc":
            ax.set_ylim(0.900, max(0.9185, max(m + sd for m, sd in zip(means, sds)) + 0.001))
        else:
            ax.set_ylim(bottom=0)
        # Place values above error bars with enough room for the labels.
        bottom, top = ax.get_ylim()
        padding = (top - bottom) * 0.025
        needed_top = max(m + sd for m, sd in zip(means, sds)) + padding * 3
        if needed_top > top:
            ax.set_ylim(bottom, needed_top)
        for x, mean, sd in zip(xs, means, sds):
            ax.text(x, mean + sd + padding, format(mean, fmt),
                    ha="center", va="bottom", fontsize=11)
    fig.text(0.5, 0.018,
             "Bars show five-seed means; error bars show sample standard deviation.",
             ha="center", fontsize=11)
    fig.tight_layout(rect=(0.02, 0.045, 0.98, 0.94), h_pad=2.5, w_pad=2.0)
    basename = "E1_final_main_comparison_2x2_reproduced"
    fig.savefig(output_dir / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{basename}.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing ten FINAL E1 results JSON files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Separate output directory; must not be the input directory")
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir:
        parser.error("Output directory must differ from input directory")
    if output_dir == Path('/home/ubuntu/afrl_runs/E1_FINAL_PUBLICATION').resolve() or \
       Path('/home/ubuntu/afrl_runs/E1_FINAL_PUBLICATION').resolve() in output_dir.parents:
        parser.error("Refusing to write inside the original E1_FINAL_PUBLICATION directory")
    records = load_seed_means(input_dir, 251, 300)
    summary = aggregate(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csvs(records, summary, output_dir)
    make_figure(summary, output_dir, 251, 300)
    for strategy in STRATEGIES:
        print(f"{strategy}:")
        for key, _, _, _ in METRICS:
            mean, sd = summary[strategy][key]
            print(f"  {key}: {mean:.12f} ± {sd:.12f}")
    print(f"PASS: reconstructed Figure 14 and CSVs in {output_dir}")


if __name__ == "__main__":
    main()
