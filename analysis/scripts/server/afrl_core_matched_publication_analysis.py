#!/usr/bin/env python3
"""Publication analysis of CORE_MATCHED_300R_5SEED experiment.

Uses five independent seeds per strategy. Original JSON files are never modified.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGIES = ['fedavg', 'tl_ppo', 'tr_dp_afrl']
LABELS = {'fedavg': 'FedAvg', 'tl_ppo': 'TL-PPO', 'tr_dp_afrl': 'TR-DP-AFRL'}
METRICS = {
    'acc': ('Accuracy', 'Accuracy'),
    'reward': ('Reward', 'Reward'),
    'latency_s': ('Round latency', 'Seconds'),
    'train_time_s': ('Training time', 'Seconds'),
    'offload_intensity': ('Offloading intensity', 'Intensity'),
    'churn_ratio': ('Client-selection churn', 'Churn ratio'),
    'telemetry_overhead_ms': ('Telemetry overhead (time)', 'Milliseconds'),
    'telemetry_overhead_bytes': ('Telemetry overhead (data)', 'Bytes'),
    'weight_delta': ('Model weight change', 'Weight delta'),
}
SEEDS = list(range(42, 47))
N_ROUNDS = 300
K = 5


def mean_sd(vals):
    a = np.asarray(vals, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='/home/ubuntu/afrl_runs/CORE_MATCHED_300R_5SEED')
    p.add_argument('--output', default='/home/ubuntu/afrl_runs/CORE_MATCHED_300R_5SEED_ANALYSIS')
    args = p.parse_args()
    src, out = Path(args.input), Path(args.output)
    csvdir, pngdir, pdfdir = out / 'csv', out / 'figures_png', out / 'figures_pdf'
    for directory in (csvdir, pngdir, pdfdir):
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    per_seed = []
    for strategy in STRATEGIES:
        for seed in SEEDS:
            path = src / f'results_{strategy}_CORE_MATCHED_300R_5SEED_seed{seed}.json'
            if not path.is_file():
                raise FileNotFoundError(path)
            d = json.loads(path.read_text())
            rounds = d['round']
            if rounds != list(range(1, N_ROUNDS + 1)):
                raise ValueError(f'Unexpected round sequence: {path}')
            for field in METRICS:
                if field not in d or len(d[field]) != N_ROUNDS:
                    raise ValueError(f'Missing/invalid metric {field}: {path}')
            for field in ('strategy', 'seed', 'selected_clients_per_round', 'selected_mask'):
                if field not in d or len(d[field]) != N_ROUNDS:
                    raise ValueError(f'Missing/invalid field {field}: {path}')
            if any(s != strategy for s in d['strategy']) or any(int(s) != seed for s in d['seed']):
                raise ValueError(f'Strategy/seed metadata mismatch: {path}')
            for r, (clients, mask) in enumerate(zip(d['selected_clients_per_round'], d['selected_mask']), 1):
                if (not isinstance(clients, list) or len(clients) != K
                        or len(set(clients)) != K or len(mask) != 10
                        or any(v not in (0, 1) for v in mask) or sum(mask) != K):
                    raise ValueError(f'K=5 selection validation failed {path} round {r}')
                expected = {f'client{i+1}' for i, v in enumerate(mask) if v == 1}
                if set(clients) != expected:
                    raise ValueError(f'Selected IDs do not match mask: {path} round {r}')
            for metric in METRICS:
                a = np.asarray(d[metric], dtype=float)
                if not np.isfinite(a).all():
                    raise ValueError(f'Non-finite values: {path}, {metric}')
            row = {'strategy': strategy, 'seed': seed, 'rounds': N_ROUNDS, 'clients_per_round': K}
            for metric in METRICS:
                a = np.asarray(d[metric], dtype=float)
                row[f'{metric}_mean_300r'] = float(a.mean())
                row[f'{metric}_final_r300'] = float(a[-1])
                row[f'{metric}_last50_mean'] = float(a[-50:].mean())
            per_seed.append(row)
            for idx, round_no in enumerate(rounds):
                rec = {'strategy': strategy, 'seed': seed, 'round': round_no,
                       'selected_client_count': K}
                rec.update({metric: float(d[metric][idx]) for metric in METRICS})
                records.append(rec)
            print(f'PASS: {strategy:12s} seed={seed} rounds=300 K=5')

    raw = pd.DataFrame(records)
    seed_df = pd.DataFrame(per_seed)
    raw.to_csv(csvdir / 'per_round_per_seed.csv', index=False)
    seed_df.to_csv(csvdir / 'per_seed_summary.csv', index=False)

    round_rows = []
    for (strategy, round_no), group in raw.groupby(['strategy', 'round'], sort=True):
        row = {'strategy': strategy, 'round': int(round_no), 'n_seeds': 5}
        for metric in METRICS:
            row[f'{metric}_mean'], row[f'{metric}_sd'] = mean_sd(group[metric])
        round_rows.append(row)
    round_df = pd.DataFrame(round_rows)
    round_df.to_csv(csvdir / 'roundwise_five_seed_mean_sd.csv', index=False)

    summary_rows = []
    for strategy in STRATEGIES:
        group = seed_df[seed_df['strategy'] == strategy]
        row = {'strategy': strategy, 'n_seeds': 5, 'rounds_per_seed': 300, 'K': 5, 'N': 10}
        for metric in METRICS:
            for stat in ('mean_300r', 'final_r300', 'last50_mean'):
                key = f'{metric}_{stat}'
                row[f'{key}_mean'], row[f'{key}_sd'] = mean_sd(group[key])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(csvdir / 'five_seed_summary.csv', index=False)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300})
    colors = {'fedavg': '#3366aa', 'tl_ppo': '#d9822b', 'tr_dp_afrl': '#28876c'}
    for metric, (title, ylabel) in METRICS.items():
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for strategy in STRATEGIES:
            subset = round_df[round_df['strategy'] == strategy].sort_values('round')
            x = subset['round'].to_numpy()
            mu = subset[f'{metric}_mean'].to_numpy()
            sd = subset[f'{metric}_sd'].to_numpy()
            ax.plot(x, mu, label=LABELS[strategy], lw=1.7, color=colors[strategy])
            ax.fill_between(x, mu - sd, mu + sd, color=colors[strategy], alpha=0.13, linewidth=0)
        ax.set(xlabel='Communication round', ylabel=ylabel, title=f'{title}: five-seed mean ± SD')
        ax.set_xlim(1, N_ROUNDS)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=3, fontsize=9)
        fig.savefig(pngdir / f'{metric}_roundwise.png', dpi=300)
        fig.savefig(pdfdir / f'{metric}_roundwise.pdf')
        plt.close(fig)

    # Independent-seed variability of final-round accuracy.
    fig, ax = plt.subplots(figsize=(6.5, 4.4), constrained_layout=True)
    x = np.arange(len(STRATEGIES))
    means = [float(summary.loc[summary.strategy == s, 'acc_final_r300_mean'].iloc[0]) for s in STRATEGIES]
    sds = [float(summary.loc[summary.strategy == s, 'acc_final_r300_sd'].iloc[0]) for s in STRATEGIES]
    ax.bar(x, means, yerr=sds, capsize=5, color=[colors[s] for s in STRATEGIES], alpha=0.85)
    ax.set_xticks(x, [LABELS[s] for s in STRATEGIES])
    ax.set_ylabel('Final-round accuracy')
    ax.set_title('Round 300 accuracy: mean ± SD across five seeds')
    ax.grid(axis='y', alpha=0.2)
    fig.savefig(pngdir / 'final_round_accuracy.png', dpi=300)
    fig.savefig(pdfdir / 'final_round_accuracy.pdf')
    plt.close(fig)

    with (out / 'README.txt').open('w') as f:
        f.write('CORE_MATCHED_300R_5SEED publication analysis\n')
        f.write('15 runs: 3 strategies x 5 seeds x 300 rounds; N=10, K=5.\n')
        f.write('All selected client ID lists and masks validated per round.\n')
        f.write('Five-seed summary: first compute one metric per seed, then mean and sample SD (ddof=1).\n')
        f.write('Roundwise figures: mean and sample SD across five independent seeds at each round.\n')
        f.write('Metrics are as recorded in source JSON; no extra normalization applied.\n')
        f.write('Participation budget is matched; compute and offloading budgets are not necessarily matched.\n')
        f.write('Source JSON files are not changed.\n')
    print('\nCOMPLETE:', out)
    print('CSV files:', len(list(csvdir.glob('*.csv'))))
    print('PNG figures:', len(list(pngdir.glob('*.png'))))
    print('PDF figures:', len(list(pdfdir.glob('*.pdf'))))
    print('\nFive-seed accuracy summary:')
    print(summary[['strategy', 'acc_mean_300r_mean', 'acc_mean_300r_sd',
                   'acc_final_r300_mean', 'acc_final_r300_sd']].to_string(index=False))


if __name__ == '__main__':
    main()
