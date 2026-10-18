#!/usr/bin/env python3
"""Reproduce E4 telemetry-ablation publication CSVs from 10 recorded JSON runs.

No experiments are executed. Original data is read-only. Requires scipy.
"""
import argparse
import csv
import json
import math
import re
import statistics as st
from pathlib import Path
from scipy import stats

CONDITIONS = ('WITH_TELEMETRY', 'NO_TELEMETRY')
SEEDS = range(42, 47)
METRICS = ('acc', 'reward', 'latency_s', 'train_time_s',
           'telemetry_overhead_ms', 'telemetry_overhead_bytes',
           'offload_intensity', 'churn_ratio', 'selected_count')
PAIRED_METRICS = ('acc', 'reward', 'latency_s', 'train_time_s',
                  'telemetry_overhead_ms', 'offload_intensity',
                  'churn_ratio', 'selected_count')
LABELS = dict(zip(PAIRED_METRICS, (
    'Accuracy', 'Reward', 'Latency (s)', 'Train time (s)',
    'Telemetry overhead (ms)', 'Offload intensity', 'Churn ratio',
    'Selected clients')))
PATTERN = re.compile(r'results_tr_dp_afrl_E4_TELEMETRY_ABLATION_(WITH_TELEMETRY|NO_TELEMETRY)_seed(\d+)\.json')


def load(input_dir):
    files = sorted(input_dir.glob('*.json'))
    if len(files) != 10:
        raise ValueError(f'Expected 10 JSON files, found {len(files)}')
    runs = {}
    for path in files:
        m = PATTERN.fullmatch(path.name)
        if not m:
            raise ValueError(f'Unexpected JSON filename: {path.name}')
        condition, seed = m.group(1), int(m.group(2))
        if seed not in SEEDS or (condition, seed) in runs:
            raise ValueError(f'Unexpected or duplicate run: {path.name}')
        data = json.loads(path.read_text())
        if data['round'] != list(range(1, 301)):
            raise ValueError(f'Invalid rounds in {path.name}')
        if len(data['ablate_telemetry']) != 300 or set(data['ablate_telemetry']) != {condition == 'NO_TELEMETRY'}:
            raise ValueError(f'Incorrect ablation flags in {path.name}')
        if len(data['seed']) != 300 or set(data['seed']) != {seed}:
            raise ValueError(f'Incorrect seeds in {path.name}')
        for metric in METRICS:
            if len(data[metric]) != 300:
                raise ValueError(f'Invalid metric length {metric}: {path.name}')
        runs[(condition, seed)] = data
    expected = {(c, s) for c in CONDITIONS for s in SEEDS}
    if set(runs) != expected:
        raise ValueError(f'Missing runs: {expected - set(runs)}')
    return runs


def make_tables(runs):
    per_seed, summary, paired, publication, statistical = [], [], [], [], []
    seed_means = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            data = runs[(condition, seed)]
            row = dict(condition=condition, seed=seed, steady_round_start=251,
                       steady_round_end=300, n_steady_rounds=50)
            seed_means[(condition, seed)] = {}
            for metric in METRICS:
                vals = [float(v) for v in data[metric][250:300]]
                avg, sd = st.mean(vals), st.stdev(vals)
                seed_means[(condition, seed)][metric] = avg
                row[f'{metric}_steady_mean'] = avg
                row[f'{metric}_steady_sd'] = sd
            per_seed.append(row)

    aggs = {}
    for condition in CONDITIONS:
        row = dict(condition=condition, n_seeds=5,
                   steady_round_start=251, steady_round_end=300)
        aggs[condition] = {}
        for metric in METRICS:
            vals = [seed_means[(condition, seed)][metric] for seed in SEEDS]
            avg, sd = st.mean(vals), st.stdev(vals)
            aggs[condition][metric] = (avg, sd)
            row[f'{metric}_steady_mean_mean'] = avg
            row[f'{metric}_steady_mean_sd'] = sd
        summary.append(row)

    for metric in PAIRED_METRICS:
        w = [seed_means[('WITH_TELEMETRY', seed)][metric] for seed in SEEDS]
        n = [seed_means[('NO_TELEMETRY', seed)][metric] for seed in SEEDS]
        d = [a - b for a, b in zip(w, n)]
        mean_w, mean_n = st.mean(w), st.mean(n)
        mean_d, sd_d = st.mean(d), st.stdev(d)
        pct = 100 * mean_d / mean_n if mean_n != 0 else float('nan')
        margin = float(stats.t.ppf(.975, df=4)) * sd_d / math.sqrt(5)
        if sd_d == 0:
            t_stat = t_p = wil_stat = wil_p = dz = float('nan')
        else:
            t = stats.ttest_rel(w, n)
            wil = stats.wilcoxon(d)
            t_stat, t_p = float(t.statistic), float(t.pvalue)
            wil_stat, wil_p = float(wil.statistic), float(wil.pvalue)
            dz = mean_d / sd_d
        row = dict(metric=metric, label=LABELS[metric], n_pairs=5,
                   with_mean=mean_w, no_mean=mean_n,
                   paired_difference_with_minus_no=mean_d,
                   percent_change_vs_no=pct, paired_difference_sd=sd_d,
                   ci95_low=mean_d - margin, ci95_high=mean_d + margin,
                   paired_t_statistic=t_stat, paired_t_pvalue=t_p,
                   wilcoxon_statistic=wil_stat, wilcoxon_pvalue=wil_p,
                   cohens_dz=dz)
        paired.append(row)
        publication.append(dict(metric=metric, label=LABELS[metric],
            with_telemetry_mean=aggs['WITH_TELEMETRY'][metric][0],
            with_telemetry_sd=aggs['WITH_TELEMETRY'][metric][1],
            no_telemetry_mean=aggs['NO_TELEMETRY'][metric][0],
            no_telemetry_sd=aggs['NO_TELEMETRY'][metric][1],
            difference_with_minus_no=mean_d, percent_change_vs_no=pct))
        statistical.append({k: row[k] for k in (
            'metric', 'label', 'n_pairs', 'paired_difference_with_minus_no',
            'ci95_low', 'ci95_high', 'paired_t_pvalue', 'wilcoxon_pvalue', 'cohens_dz')})

    return {
        'E4_per_seed_steady_summary.csv': per_seed,
        'E4_5seed_summary.csv': summary,
        'E4_paired_effects.csv': paired,
        'E4_publication_table.csv': publication,
        'E4_statistical_table.csv': statistical,
    }


def compare(tables, original_dir):
    comparisons, mismatches = 0, []
    for name, rows in tables.items():
        with (original_dir / name).open(newline='') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            originals = list(reader)
        if fields != list(rows[0]):
            mismatches.append(f'{name}: column order differs')
        if len(originals) != len(rows):
            mismatches.append(f'{name}: row count differs ({len(originals)} vs {len(rows)})')
            continue
        for index, (old, new) in enumerate(zip(originals, rows), 1):
            for col, value in new.items():
                if col not in old:
                    mismatches.append(f'{name} row {index}: missing {col}')
                    continue
                if isinstance(value, str):
                    if old[col] != value:
                        mismatches.append(f'{name} row {index} {col}: label/order mismatch')
                else:
                    # Match the 424 independently audited numerical cells;
                    # metadata is separately checked for exact equality.
                    metadata = col in ('seed', 'steady_round_start',
                                       'steady_round_end', 'n_steady_rounds', 'n_seeds')
                    if metadata:
                        if float(old[col]) != float(value):
                            mismatches.append(f'{name} row {index} {col}: metadata mismatch')
                        continue
                    comparisons += 1
                    actual, expected = float(value), float(old[col])
                    equal = ((math.isnan(actual) and math.isnan(expected)) or
                             (math.isinf(actual) and actual == expected) or
                             (math.isfinite(actual) and math.isfinite(expected) and
                              math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8)))
                    if not equal:
                        mismatches.append(f'{name} row {index} {col}: original={expected}, reproduced={actual}')
    return comparisons, mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--verify-against', type=Path)
    args = parser.parse_args()
    if args.output.resolve() in {args.input.resolve(),
                                 args.verify_against.resolve() if args.verify_against else None}:
        parser.error('Output must differ from input and original publication directory')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error(f'Output directory must be empty or absent: {args.output}')
    tables = make_tables(load(args.input))
    if args.verify_against:
        checks, issues = compare(tables, args.verify_against)
        print(f'Publication CSV numerical comparisons: {checks}')
        print(f'Mismatches: {len(issues)}')
        for issue in issues[:30]:
            print('MISMATCH:', issue)
        if issues or checks != 424:
            raise SystemExit('VERIFICATION FAILED — no CSVs written')
        print('PASS — all 424 numerical comparisons match')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        with (args.output / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'WROTE {name}: {len(rows)} rows')
    print('SUCCESS — original JSON and publication CSVs unchanged')

if __name__ == '__main__':
    main()
