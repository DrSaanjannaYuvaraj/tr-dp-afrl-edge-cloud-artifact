#!/usr/bin/env python3
"""Reproduce the E4 publication figure without changing original outputs."""
from pathlib import Path
import argparse
import csv
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Reproduced E4_5seed_summary.csv')
    parser.add_argument('--output', type=Path, required=True, help='Separate figure output directory')
    parser.add_argument('--compare-png', type=Path, help='Original publication PNG to compare')
    args = parser.parse_args()
    csv_path = args.input.resolve()
    out_dir = args.output.resolve()
    png = out_dir / 'E4_final_telemetry_ablation_2x2.png'
    pdf = out_dir / 'E4_final_telemetry_ablation_2x2.pdf'
    if not csv_path.is_file():
        parser.error(f'Input CSV not found: {csv_path}')
    if args.compare_png and (png == args.compare_png.resolve() or pdf == args.compare_png.resolve()):
        parser.error('Output would overwrite the comparison file')
    if args.compare_png and not args.compare_png.is_file():
        parser.error(f'Comparison PNG not found: {args.compare_png}')
    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    data = {r['condition']: r for r in rows}
    conditions = ['WITH_TELEMETRY', 'NO_TELEMETRY']
    if len(rows) != 2 or set(data) != set(conditions):
        parser.error('Expected exactly WITH_TELEMETRY and NO_TELEMETRY rows')
    labels = ['With telemetry', 'No telemetry']
    panels = [
        ('Accuracy', 'acc_steady_mean_mean', 'acc_steady_mean_sd', 'Accuracy'),
        ('End-to-end latency', 'latency_s_steady_mean_mean', 'latency_s_steady_mean_sd', 'Latency (s)'),
        ('Training time', 'train_time_s_steady_mean_mean', 'train_time_s_steady_mean_sd', 'Training time (s)'),
        ('Telemetry overhead', 'telemetry_overhead_ms_steady_mean_mean', 'telemetry_overhead_ms_steady_mean_sd', 'Overhead (ms)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.ravel()
    for ax, (title, mean_key, sd_key, ylabel) in zip(axes, panels):
        means = [float(data[c][mean_key]) for c in conditions]
        sds = [float(data[c][sd_key]) for c in conditions]
        x = range(len(labels))
        ax.bar(x, means, yerr=sds, capsize=5, edgecolor='black', linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.25)
        if title == 'Accuracy':
            low = min(m - s for m, s in zip(means, sds))
            high = max(m + s for m, s in zip(means, sds))
            margin = max((high - low) * 0.45, 0.002)
            ax.set_ylim(low - margin, high + margin)
        for i, mean in enumerate(means):
            if title == 'Accuracy':
                label = f'{mean:.4f}'
            elif title == 'Telemetry overhead':
                label = f'{mean:.2f}'
            else:
                label = f'{mean:.2f}'
            ax.text(i, mean, label, ha='center', va='bottom', fontsize=9)
    fig.suptitle('E4 Telemetry Ablation — TR-DP-AFRL\nSteady-state rounds 251–300, five paired seeds', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print('Input:', csv_path)
    print('PNG:', png)
    print('PDF:', pdf)
    print('PNG bytes:', png.stat().st_size)
    print('PDF bytes:', pdf.stat().st_size)
    print('Reproduced PNG SHA256:', sha256(png))
    if args.compare_png:
        original = args.compare_png.resolve()
        print('Original PNG SHA256:  ', sha256(original))
        if sha256(png) == sha256(original):
            print('PASS — PNG is byte-identical')
        else:
            print('NOT BYTE-IDENTICAL — inspect rendering environment and visual differences')
    print('Original publication files unchanged')

if __name__ == '__main__':
    main()
