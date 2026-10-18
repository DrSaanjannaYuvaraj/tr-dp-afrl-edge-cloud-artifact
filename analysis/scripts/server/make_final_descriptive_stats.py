#!/usr/bin/env python3

import os
from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
INDIR = str(REPO_ROOT / "results" / "frozen" / "core")
OUTDIR = str(
    REPO_ROOT / "outputs" / "regenerated_analysis" / "core" / "tables"
)

W = 20

METRICS = [
    "acc",
    "reward",
    "latency_s",
    "churn_ratio",
    "offload_intensity",
    "telemetry_overhead_ms",
]

os.makedirs(OUTDIR, exist_ok=True)

rows = []

files = sorted(glob.glob(os.path.join(INDIR, "results_*.json")))

for path in files:

    with open(path) as f:
        d = json.load(f)

    label = d.get("variant_label", "")
    if isinstance(label, list):
        label = label[0] if label else ""

    label = str(label).upper()

    # Exclude telemetry WITH/NO ablation runs
    if "WITH" in label or "NO" in label:
        continue

    strategy = d.get("strategy", "")
    if isinstance(strategy, list):
        strategy = strategy[0] if strategy else ""

    seed = d.get("seed", "")
    if isinstance(seed, list):
        seed = seed[0] if seed else ""

    row = {
        "strategy": str(strategy),
        "seed": int(seed),
    }

    for metric in METRICS:
        vals = np.asarray(d[metric], dtype=float)
        tail = vals[-W:]
        row[metric] = float(np.mean(tail))

    rows.append(row)


seed_df = pd.DataFrame(rows).sort_values(["strategy", "seed"])

seed_file = os.path.join(
    OUTDIR,
    "Table_Final_seedlevel_W20.csv"
)

seed_df.to_csv(seed_file, index=False)


summary_rows = []

for strategy, g in seed_df.groupby("strategy"):

    for metric in METRICS:

        x = g[metric].to_numpy(dtype=float)

        n = len(x)
        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1))

        if n > 1:
            sem = stats.sem(x)
            tcrit = stats.t.ppf(0.975, df=n-1)
            ci_low = mean - tcrit * sem
            ci_high = mean + tcrit * sem
        else:
            ci_low = np.nan
            ci_high = np.nan

        summary_rows.append({
            "strategy": strategy,
            "metric": metric,
            "n_seeds": n,
            "mean": mean,
            "sd": sd,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        })


summary_df = pd.DataFrame(summary_rows)

summary_file = os.path.join(
    OUTDIR,
    "Table_Final_descriptive_95CI_W20.csv"
)

summary_df.to_csv(summary_file, index=False)

print("SEED-LEVEL FILE:")
print(seed_file)

print()
print(seed_df.to_string(index=False))

print()
print("DESCRIPTIVE + 95% CI FILE:")
print(summary_file)

print()
print(summary_df.to_string(index=False))
