#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = str(
    Path(__file__).resolve().parents[3]
    / "outputs" / "regenerated_analysis" / "core" / "tables"
)

INFILE = os.path.join(BASE, "Table_Final_seedlevel_W20.csv")
OUTFILE = os.path.join(BASE, "Table_Final_inferential_Holm_W20.csv")

PRIMARY = "tr_dp_afrl"

METRICS = [
    "acc",
    "reward",
    "latency_s",
    "churn_ratio",
    "offload_intensity",
    "telemetry_overhead_ms",
]

BASELINES = [
    "ddqn",
    "dqn",
    "fedavg",
    "heuristic",
    "tl_ppo",
]

df = pd.read_csv(INFILE)

rows = []

for metric in METRICS:

    metric_rows = []

    for baseline in BASELINES:

        a = df[df["strategy"] == PRIMARY][["seed", metric]].copy()
        b = df[df["strategy"] == baseline][["seed", metric]].copy()

        m = a.merge(b, on="seed", suffixes=("_trdp", "_base")).sort_values("seed")

        x = m[f"{metric}_trdp"].to_numpy(dtype=float)
        y = m[f"{metric}_base"].to_numpy(dtype=float)

        diff = x - y
        n = len(diff)

        mean_diff = float(np.mean(diff))
        sd_diff = float(np.std(diff, ddof=1))

        # Paired t-test
        if n > 1:
            t_stat, p_t = stats.ttest_rel(x, y)

            sem_diff = stats.sem(diff)
            tcrit = stats.t.ppf(0.975, df=n-1)
            ci_low = mean_diff - tcrit * sem_diff
            ci_high = mean_diff + tcrit * sem_diff
        else:
            t_stat = np.nan
            p_t = np.nan
            ci_low = np.nan
            ci_high = np.nan

        # Cohen's dz
        if sd_diff > 0:
            dz = mean_diff / sd_diff
        else:
            if mean_diff == 0:
                dz = 0.0
            else:
                dz = np.inf if mean_diff > 0 else -np.inf

        # Wilcoxon sensitivity analysis
        try:
            if np.allclose(diff, 0):
                p_w = 1.0
            else:
                _, p_w = stats.wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto"
                )
        except TypeError:
            if np.allclose(diff, 0):
                p_w = 1.0
            else:
                _, p_w = stats.wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox",
                    mode="auto"
                )

        metric_rows.append({
            "metric": metric,
            "baseline": baseline,
            "n_pairs": n,
            "mean_diff_trdp_minus_baseline": mean_diff,
            "ci95_diff_low": ci_low,
            "ci95_diff_high": ci_high,
            "paired_t_stat": t_stat,
            "paired_t_p_raw": p_t,
            "wilcoxon_p": p_w,
            "cohen_dz": dz,
        })

    # Holm correction across the 5 baseline comparisons for this metric
    raw_p = np.array([r["paired_t_p_raw"] for r in metric_rows], dtype=float)

    order = np.argsort(raw_p)
    adjusted = np.empty_like(raw_p)

    running_max = 0.0

    for rank, idx in enumerate(order):
        factor = len(raw_p) - rank
        adj = min(1.0, factor * raw_p[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max

    for i, r in enumerate(metric_rows):
        r["paired_t_p_holm"] = adjusted[i]
        r["significant_holm_0_05"] = bool(adjusted[i] < 0.05)

    rows.extend(metric_rows)

out = pd.DataFrame(rows)

out = out[
    [
        "metric",
        "baseline",
        "n_pairs",
        "mean_diff_trdp_minus_baseline",
        "ci95_diff_low",
        "ci95_diff_high",
        "paired_t_stat",
        "paired_t_p_raw",
        "paired_t_p_holm",
        "significant_holm_0_05",
        "wilcoxon_p",
        "cohen_dz",
    ]
]

out.to_csv(OUTFILE, index=False)

print("OUTPUT:")
print(OUTFILE)
print()
print(out.to_string(index=False))
