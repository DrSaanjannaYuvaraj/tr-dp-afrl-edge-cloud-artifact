#!/usr/bin/env python3
"""Reproduce E3 reward-sensitivity tables from recorded 300-round JSON runs.

No experiments are executed. Outputs are written to a distinct directory.
Dependencies: numpy, scipy (standard library otherwise).
"""
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

CONFIGS = ('reference', 'accuracy_low', 'accuracy_high', 'latency_low',
           'latency_high', 'offload_low', 'offload_high', 'stability_low',
           'stability_high')
SEEDS = tuple(range(42, 47))
METRICS = ('acc', 'reward', 'latency_s', 'train_time_s',
           'telemetry_overhead_ms', 'telemetry_overhead_bytes',
           'offload_intensity', 'churn_ratio', 'selected_count')
FIG_METRICS = ('acc', 'latency_s', 'offload_intensity', 'churn_ratio')
LABELS = {'acc': 'Accuracy', 'latency_s': 'Latency (s)',
          'offload_intensity': 'Offload intensity', 'churn_ratio': 'Churn ratio'}
PATTERN = re.compile(r'results_tr_dp_afrl_E3_REWARD_SENSITIVITY_(.+)_seed(\d+)\.json')


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_means(data_dir):
    files = sorted(data_dir.glob('results_*.json'))
    if len(files) != 45:
        raise ValueError(f'Expected exactly 45 JSON files, found {len(files)}')
    means = {}
    for path in files:
        match = PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f'Unexpected filename: {path.name}')
        config, seed_str = match.groups()
        seed = int(seed_str)
        key = (config, seed)
        if config not in CONFIGS or seed not in SEEDS or key in means:
            raise ValueError(f'Unexpected or duplicated configuration/seed: {key}')
        data = json.loads(path.read_text())
        if data['round'] != list(range(1, 301)):
            raise ValueError(f'Expected rounds 1–300: {path}')
        for metric in METRICS:
            if len(data[metric]) != 300:
                raise ValueError(f'Unexpected {metric} length: {path}')
        means[key] = {metric: statistics.mean(float(x) for x in data[metric][250:300])
                      for metric in METRICS}
    expected = {(c, s) for c in CONFIGS for s in SEEDS}
    if set(means) != expected:
        raise ValueError(f'Missing runs: {expected - set(means)}')
    return means


def generate(means):
    per_seed = []
    for config in CONFIGS:
        for seed in SEEDS:
            row = dict(configuration=config, seed=seed,
                       steady_round_start=251, steady_round_end=300)
            row.update({f'{metric}_steady_mean': means[(config, seed)][metric]
                        for metric in METRICS})
            per_seed.append(row)

    aggregate = []
    aggs = {}
    for config in CONFIGS:
        row = dict(configuration=config, n_seeds=5,
                   steady_round_start=251, steady_round_end=300)
        aggs[config] = {}
        for metric in METRICS:
            vals = [means[(config, seed)][metric] for seed in SEEDS]
            avg, sd = statistics.mean(vals), statistics.stdev(vals)
            aggs[config][metric] = (avg, sd)
            row[f'{metric}_steady_mean_mean'] = avg
            row[f'{metric}_steady_mean_sd'] = sd
        aggregate.append(row)

    publication = []
    for config in CONFIGS:
        row = dict(configuration=config, n_seeds=5)
        for metric in FIG_METRICS:
            row[f'{metric}_mean'], row[f'{metric}_sd'] = aggs[config][metric]
        publication.append(row)

    paired = []
    for config in CONFIGS[1:]:
        for metric in FIG_METRICS:
            ref = np.array([means[('reference', seed)][metric] for seed in SEEDS], dtype=float)
            alt = np.array([means[(config, seed)][metric] for seed in SEEDS], dtype=float)
            diff = alt - ref
            mean_ref = float(np.mean(ref))
            mean_alt = float(np.mean(alt))
            mean_diff = float(np.mean(diff))
            sd_diff = float(np.std(diff, ddof=1))
            if sd_diff > 1e-12:
                margin = float(stats.t.ppf(.975, df=4) * sd_diff / math.sqrt(5))
                low, high = mean_diff - margin, mean_diff + margin
                t_p = float(stats.ttest_rel(alt, ref).pvalue)
                dz = mean_diff / sd_diff
            else:
                low = high = mean_diff
                t_p = dz = float('nan')
            w_p = (float('nan') if np.all(np.abs(diff) < 1e-12)
                   else float(stats.wilcoxon(diff).pvalue))
            paired.append(dict(
                configuration=config, metric=metric, label=LABELS[metric], n_pairs=5,
                reference_mean=mean_ref, configuration_mean=mean_alt,
                paired_difference_config_minus_reference=mean_diff,
                percent_change_vs_reference=100 * mean_diff / mean_ref,
                ci95_low=low, ci95_high=high, paired_t_pvalue=t_p,
                wilcoxon_pvalue=w_p, cohens_dz=dz))

    return {
        'E3_per_seed_steady_summary.csv': per_seed,
        'E3_5seed_summary.csv': aggregate,
        'E3_publication_table.csv': publication,
        'E3_paired_effects.csv': paired,
    }


def verify(generated, originals):
    checks, failures = 0, []
    for name, rows in generated.items():
        path = originals / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing original CSV: {path}')
        with path.open(newline='') as f:
            original = list(csv.DictReader(f))
        if len(original) != len(rows):
            failures.append(f'{name}: row count {len(rows)} vs {len(original)}')
            continue
        if list(original[0]) != list(rows[0]):
            failures.append(f'{name}: column order differs')
        for i, (expected, actual) in enumerate(zip(original, rows), start=1):
            for col, value in actual.items():
                if col not in expected:
                    failures.append(f'{name} row {i}: missing column {col}')
                    continue
                if isinstance(value, str):
                    if value != expected[col]:
                        failures.append(f'{name} row {i} {col}: label differs')
                else:
                    if name == 'E3_paired_effects.csv' and col == 'n_pairs':
                        # The prior independent audit included this numeric column.
                        pass
                    if name == 'E3_per_seed_steady_summary.csv' and col in (
                        'seed', 'steady_round_start', 'steady_round_end'):
                        continue
                    if name == 'E3_5seed_summary.csv' and col in (
                        'n_seeds', 'steady_round_start', 'steady_round_end'):
                        continue
                    if name == 'E3_publication_table.csv' and col == 'n_seeds':
                        continue
                    checks += 1
                    a, b = float(expected[col]), float(value)
                    if not ((math.isnan(a) and math.isnan(b)) or
                            math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-8)):
                        failures.append(f'{name} row {i} {col}: original={a}, reproduced={b}')
    return checks, failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Directory of 45 E3 JSON files')
    p.add_argument('--output', type=Path, required=True, help='New output directory for CSV files')
    p.add_argument('--verify-against', type=Path, help='Original publication CSV directory')
    args = p.parse_args()
    if args.output.resolve() == args.input.resolve() or (
        args.verify_against is not None and args.output.resolve() == args.verify_against.resolve()
    ):
        p.error('Output directory must differ from input and verification directories')
    if args.output.exists() and any(args.output.iterdir()):
        p.error(f'Output directory is not empty: {args.output}')

    means = load_means(args.input)
    generated = generate(means)
    if args.verify_against:
        checks, failures = verify(generated, args.verify_against)
        print(f'Original CSV numerical comparisons: {checks}')
        print(f'Mismatches: {len(failures)}')
        for issue in failures[:30]:
            print('MISMATCH:', issue)
        if checks != 959 or failures:
            raise SystemExit('VERIFICATION FAILED — no output files written')
        print('PASS — all 959 publication numerical comparisons match')

    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in generated.items():
        write_csv(args.output / name, list(rows[0]), rows)
        print(f'WROTE {name}: {len(rows)} rows')
    print('SUCCESS — source JSON and original CSVs unchanged')


if __name__ == '__main__':
    main()
