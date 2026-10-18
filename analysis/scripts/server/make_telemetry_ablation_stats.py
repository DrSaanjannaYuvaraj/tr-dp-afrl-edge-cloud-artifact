#!/usr/bin/env python3

import os
from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
INDIR = str(REPO_ROOT / "results" / "frozen" / "e4_final_rerun")
OUTDIR = REPO_ROOT / "outputs" / "regenerated_analysis" / "e4" / "tables"
OUTFILE = str(OUTDIR / "Table_Final_telemetry_ablation_stats_W20.csv")

W = 20

STRATEGIES = ["tr_dp_afrl"]

METRICS = [
    "acc",
    "reward",
    "latency_s",
    "offload_intensity",
    "churn_ratio",
]

records = []

for path in glob.glob(os.path.join(INDIR, "results_*.json")):

    with open(path) as f:
        d = json.load(f)

    strategy = d.get("strategy", "")
    if isinstance(strategy, list):
        strategy = strategy[0] if strategy else ""

    if strategy not in STRATEGIES:
        continue

    label = d.get("variant_label", "")
    if isinstance(label, list):
        label = label[0] if label else ""

    label = str(label).upper()

    if "WITH" in label:
        condition = "WITH"
    elif "NO" in label:
        condition = "NO"
    else:
        continue

    seed = d.get("seed", "")
    if isinstance(seed, list):
        seed = seed[0]

    row = {
        "strategy": strategy,
        "condition": condition,
        "seed": int(seed),
    }

    for metric in METRICS:
        vals = np.asarray(d[metric], dtype=float)
        row[metric] = float(np.mean(vals[-W:]))

    records.append(row)

df = pd.DataFrame(records)

rows = []

for strategy in STRATEGIES:

    for metric in METRICS:

        w = df[
            (df.strategy == strategy) &
            (df.condition == "WITH")
        ][["seed", metric]]

        n = df[
            (df.strategy == strategy) &
            (df.condition == "NO")
        ][["seed", metric]]

        m = w.merge(
            n,
            on="seed",
            suffixes=("_WITH", "_NO")
        ).sort_values("seed")

        x = m[f"{metric}_WITH"].to_numpy(float)
        y = m[f"{metric}_NO"].to_numpy(float)

        diff = x - y

        mean_with = np.mean(x)
        sd_with = np.std(x, ddof=1)

        mean_no = np.mean(y)
        sd_no = np.std(y, ddof=1)

        mean_diff = np.mean(diff)
        sd_diff = np.std(diff, ddof=1)

        t_stat, p_t = stats.ttest_rel(x, y)

        sem = stats.sem(diff)
        tcrit = stats.t.ppf(0.975, df=len(diff)-1)

        ci_low = mean_diff - tcrit * sem
        ci_high = mean_diff + tcrit * sem

        dz = mean_diff / sd_diff if sd_diff > 0 else np.nan

        if np.allclose(diff, 0):
            p_w = 1.0
        else:
            try:
                _, p_w = stats.wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto"
                )
            except TypeError:
                _, p_w = stats.wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox",
                    mode="auto"
                )

        rows.append({
            "strategy": strategy,
            "metric": metric,
            "n_pairs": len(diff),

            "WITH_mean": mean_with,
            "WITH_sd": sd_with,

            "NO_mean": mean_no,
            "NO_sd": sd_no,

            "mean_diff_WITH_minus_NO": mean_diff,

            "ci95_diff_low": ci_low,
            "ci95_diff_high": ci_high,

            "paired_t_p": p_t,
            "wilcoxon_p": p_w,
            "cohen_dz": dz,
        })

out = pd.DataFrame(rows)

OUTDIR.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTFILE, index=False)

print("OUTPUT:")
print(OUTFILE)
print()
print(out.to_string(index=False))
