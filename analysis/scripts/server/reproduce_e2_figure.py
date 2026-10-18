#!/usr/bin/env python3
"""Reconstructed E2 component-ablation publication artifacts.

This is a reconstruction, NOT the original plotting source. It consumes the
30 final E2 JSON logs, does not rerun experiments, and never edits input files.

Example:
  python reproduce_e2_figure.py \
    --input-dir /home/ubuntu/afrl_runs/E2_FINAL_RERUN_300R_5SEED \
    --output-dir /home/ubuntu/afrl_runs/E2_REPRODUCED_FIGURE

Dependencies: numpy, scipy, matplotlib (Python standard library otherwise).
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VARIANTS = [
    'tr_dp_afrl', 'selection_only', 'execution_only',
    'no_kl', 'no_selection_kl', 'no_offload_kl',
]
LABELS = [
    'TR-DP-AFRL', 'Selection\nonly', 'Execution\nonly',
    'No KL', 'No selection\nKL', 'No offload\nKL',
]
METRICS = [
    'acc', 'reward', 'latency_s', 'train_time_s',
    'telemetry_overhead_ms', 'telemetry_overhead_bytes',
    'offload_intensity', 'churn_ratio', 'selected_count',
]
PAIRED_METRICS = [
    'acc', 'reward', 'latency_s', 'train_time_s',
    'offload_intensity', 'churn_ratio', 'selected_count',
    'telemetry_overhead_ms',
]
METRIC_LABELS = {
    'acc': 'Accuracy', 'reward': 'Reward', 'latency_s': 'Latency (s)',
    'train_time_s': 'Training time (s)',
    'offload_intensity': 'Offload intensity',
    'churn_ratio': 'Client-selection churn',
    'selected_count': 'Selected clients',
    'telemetry_overhead_ms': 'Telemetry overhead (ms)',
}
SEEDS = tuple(range(42, 47))


def write_csv(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_runs(folder):
    found = list(folder.glob('results_*.json'))
    if len(found) != 30:
        raise ValueError(f'Expected exactly 30 results_*.json files; found {len(found)} in {folder}')
    runs = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            p = folder / f'results_{variant}_E2_COMPONENT_ABLATION_seed{seed}.json'
            if not p.is_file():
                raise FileNotFoundError(f'Missing E2 input: {p}')
            data = json.loads(p.read_text(encoding='utf-8'))
            for metric in METRICS:
                if metric not in data or len(data[metric]) != 300:
                    raise ValueError(f'{p.name}: {metric} must have 300 rounds')
                a = np.asarray(data[metric], dtype=float)
                if not np.isfinite(a).all():
                    raise ValueError(f'{p.name}: nonfinite {metric}')
            if len(data.get('round', [])) != 300:
                raise ValueError(f'{p.name}: invalid round count')
            runs[(variant, seed)] = data
    return runs


def calc_tables(runs, tables_dir):
    per_seed = []
    per_seed_by_key = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            data = runs[(variant, seed)]
            row = dict(strategy=variant, seed=seed,
                       steady_round_start=251, steady_round_end=300,
                       steady_round_count=50)
            for metric in METRICS:
                arr = np.asarray(data[metric][250:300], dtype=float)
                row[f'{metric}_steady_mean'] = float(np.mean(arr))
                row[f'{metric}_steady_sd'] = float(np.std(arr, ddof=1))
            per_seed.append(row)
            per_seed_by_key[(variant, seed)] = row
    per_columns = ['strategy', 'seed', 'steady_round_start',
                   'steady_round_end', 'steady_round_count']
    per_columns += [f'{m}_steady_{s}' for m in METRICS for s in ('mean', 'sd')]
    write_csv(tables_dir / 'E2_per_seed_steady_summary.csv', per_columns, per_seed)

    summary = []
    for variant in VARIANTS:
        row = dict(strategy=variant, n_seeds=5,
                   steady_round_start=251, steady_round_end=300)
        for metric in METRICS:
            values = np.asarray([
                per_seed_by_key[(variant, seed)][f'{metric}_steady_mean']
                for seed in SEEDS
            ])
            row[f'{metric}_steady_mean_mean'] = float(np.mean(values))
            row[f'{metric}_steady_mean_sd'] = float(np.std(values, ddof=1))
        summary.append(row)
    summary_cols = ['strategy', 'n_seeds', 'steady_round_start', 'steady_round_end']
    summary_cols += [f'{m}_steady_mean_{s}' for m in METRICS for s in ('mean', 'sd')]
    write_csv(tables_dir / 'E2_5seed_summary.csv', summary_cols, summary)

    pub = []
    pub_metrics = ['acc', 'reward', 'latency_s', 'train_time_s',
                   'offload_intensity', 'churn_ratio', 'selected_count',
                   'telemetry_overhead_ms']
    for src in summary:
        row = dict(configuration=src['strategy'], n_seeds=src['n_seeds'])
        for metric in pub_metrics:
            row[f'{metric}_mean'] = src[f'{metric}_steady_mean_mean']
            row[f'{metric}_sd'] = src[f'{metric}_steady_mean_sd']
        pub.append(row)
    pub_cols = ['configuration', 'n_seeds']
    pub_cols += [f'{m}_{s}' for m in pub_metrics for s in ('mean', 'sd')]
    write_csv(tables_dir / 'E2_publication_table.csv', pub_cols, pub)

    paired = []
    for metric in PAIRED_METRICS:
        ref = np.asarray([per_seed_by_key[('tr_dp_afrl', s)][f'{metric}_steady_mean']
                          for s in SEEDS], dtype=float)
        for variant in VARIANTS[1:]:
            cur = np.asarray([per_seed_by_key[(variant, s)][f'{metric}_steady_mean']
                              for s in SEEDS], dtype=float)
            diff = cur - ref
            dmean = float(np.mean(diff))
            sd = float(np.std(diff, ddof=1))
            ref_mean = float(np.mean(ref))
            if sd == 0:
                low = high = dmean
                t_p = math.nan
                dz = math.nan
            else:
                margin = float(stats.t.ppf(.975, len(diff)-1) * sd / np.sqrt(len(diff)))
                low, high = dmean-margin, dmean+margin
                t_p = float(stats.ttest_rel(cur, ref).pvalue)
                dz = dmean / sd
            # All-zero differences have no signed-rank statistic.
            if np.all(diff == 0):
                w_p = math.nan
            else:
                w_p = float(stats.wilcoxon(diff).pvalue)
            paired.append(dict(
                metric=metric, label=METRIC_LABELS[metric], configuration=variant,
                reference='tr_dp_afrl', n_pairs=5,
                reference_mean=ref_mean, configuration_mean=float(np.mean(cur)),
                paired_difference_ablation_minus_reference=dmean,
                percent_change_vs_reference=(100*dmean/ref_mean if ref_mean != 0 else math.nan),
                ci95_low=low, ci95_high=high,
                paired_t_pvalue=t_p, wilcoxon_pvalue=w_p, cohens_dz=dz,
            ))
    paired_cols = ['metric', 'label', 'configuration', 'reference', 'n_pairs',
                   'reference_mean', 'configuration_mean',
                   'paired_difference_ablation_minus_reference',
                   'percent_change_vs_reference', 'ci95_low', 'ci95_high',
                   'paired_t_pvalue', 'wilcoxon_pvalue', 'cohens_dz']
    write_csv(tables_dir / 'E2_paired_effects.csv', paired_cols, paired)
    return {row['strategy']: row for row in summary}


def make_figure(summary, figures_dir, dpi):
    panels = [
        ('acc', '(a) Steady-state accuracy', 'Accuracy'),
        ('latency_s', '(b) Steady-state latency', 'Latency (s)'),
        ('offload_intensity', '(c) Offload intensity', 'Offload intensity'),
        ('churn_ratio', '(d) Client-selection churn', 'Churn ratio'),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(13.4, 9.49))
    for ax, (metric, title, ylabel) in zip(axs.flat, panels):
        means = [summary[v][f'{metric}_steady_mean_mean'] for v in VARIANTS]
        sds = [summary[v][f'{metric}_steady_mean_sd'] for v in VARIANTS]
        bars = ax.bar(range(6), means, yerr=sds, capsize=4,
                      color='#438fbe', edgecolor='#333333', linewidth=.8,
                      error_kw={'elinewidth': 1.2, 'capthick': 1.2})
        bars[0].set_hatch('///')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(6), LABELS, fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=.25)
        ax.set_axisbelow(True)
        if metric == 'acc':
            ax.set_ylim(.9005, .916)
        else:
            ax.set_ylim(bottom=0)
    fig.suptitle('E2 Component Ablation: Steady-State Performance (Rounds 251–300)',
                 fontsize=14, fontweight='bold', y=.995)
    fig.text(.5, .018,
             'Bars show mean ± SD across five paired seeds (42–46). '
             'Hatched bar denotes the full TR-DP-AFRL configuration.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .045, 1, .965], h_pad=2.0, w_pad=2.0)
    stem = 'E2_final_component_ablation_2x2'
    fig.savefig(figures_dir / f'{stem}.png', dpi=dpi, facecolor='white')
    fig.savefig(figures_dir / f'{stem}.pdf', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    source = args.input_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Input and output directories must be separate and non-nested')
    if not source.is_dir():
        raise FileNotFoundError(source)
    runs = load_runs(source)  # Validate all inputs before writing outputs.
    tables_dir = output / 'tables'
    figures_dir = output / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = calc_tables(runs, tables_dir)
    make_figure(summary, figures_dir, args.dpi)
    print('PASS: reconstructed E2 2x2 figure and four publication CSVs')
    print('Input JSON files:', len(runs))
    print('Output directory:', output)
    for v in VARIANTS:
        r = summary[v]
        print(f"{v:20s} accuracy={r['acc_steady_mean_mean']:.9f} ± "
              f"{r['acc_steady_mean_sd']:.9f}")
    print('NOTE: reconstructed figure need not be pixel-identical to the original.')


if __name__ == '__main__':
    main()
