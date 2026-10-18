import json
import csv
import math
import statistics
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "results" / "frozen" / "e7_final"
OUT = REPO_ROOT / "outputs" / "regenerated_analysis" / "e7"
TABLES = OUT / "tables"
AUDIT = OUT / "audit"

TABLES.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["fedavg", "tl_ppo", "tr_dp_afrl"]
CONDITIONS = ["none", "cpu", "network"]
SEEDS = [42, 43, 44, 45, 46]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def mean(x):
    return statistics.mean(x)

def sd(x):
    return statistics.stdev(x) if len(x) > 1 else 0.0

def mean_sd(x):
    return mean(x), sd(x)

def window(x, start_round, end_round):
    # Python indices: round 1 -> index 0
    return x[start_round - 1:end_round]

def numeric_list(d, key):
    x = d.get(key)

    if not isinstance(x, list):
        raise RuntimeError(f"{key} is missing or not a list")

    if len(x) != 300:
        raise RuntimeError(
            f"{key}: expected 300 entries, got {len(x)}"
        )

    vals = [float(v) for v in x]

    if not all(math.isfinite(v) for v in vals):
        raise RuntimeError(f"{key}: NaN/Inf detected")

    return vals


# ------------------------------------------------------------------
# Load and identify all 45 runs
# ------------------------------------------------------------------

runs = {}

files = sorted(ROOT.glob("results_*.json"))

if len(files) != 45:
    raise RuntimeError(
        f"Expected 45 frozen JSON files, found {len(files)}"
    )

for p in files:
    d = json.loads(p.read_text())

    strategy_values = d.get("strategy", [])
    seed_values = d.get("seed", [])
    variant_values = d.get("variant_label", [])

    if not strategy_values or not seed_values or not variant_values:
        raise RuntimeError(f"Missing identity fields in {p.name}")

    strategy = str(strategy_values[0])
    seed = int(seed_values[0])
    variant = str(variant_values[0])

    if "_none_" in variant:
        condition = "none"
    elif "_cpu_" in variant:
        condition = "cpu"
    elif "_network_" in variant:
        condition = "network"
    else:
        raise RuntimeError(
            f"Cannot determine condition from {p.name}: {variant}"
        )

    key = (strategy, condition, seed)

    if key in runs:
        raise RuntimeError(f"Duplicate run: {key}")

    rounds = d.get("round", [])

    if rounds != list(range(1, 301)):
        raise RuntimeError(f"Invalid rounds in {p.name}")

    runs[key] = {
        "path": p,
        "data": d,
    }

expected = {
    (s, c, seed)
    for s in STRATEGIES
    for c in CONDITIONS
    for seed in SEEDS
}

if set(runs) != expected:
    missing = expected - set(runs)
    extra = set(runs) - expected

    raise RuntimeError(
        f"Experiment matrix mismatch\n"
        f"Missing={missing}\nExtra={extra}"
    )

print("PASS: loaded exact 3 x 3 x 5 E7 matrix")


# ------------------------------------------------------------------
# Per-run statistics
# ------------------------------------------------------------------

per_run = []

for strategy in STRATEGIES:
    for condition in CONDITIONS:
        for seed in SEEDS:

            d = runs[(strategy, condition, seed)]["data"]

            acc = numeric_list(d, "acc")
            reward = numeric_list(d, "reward")
            latency = numeric_list(d, "latency_s")
            train = numeric_list(d, "train_time_s")
            comm = numeric_list(d, "communication_bytes_total")

            sla10 = numeric_list(d, "sla_violation_10s")
            sla12 = numeric_list(d, "sla_violation_12s")
            sla15 = numeric_list(d, "sla_violation_15s")

            offload = numeric_list(d, "offload_intensity")
            churn = numeric_list(d, "churn_ratio")
            selected = numeric_list(d, "selected_count")
            cloud = numeric_list(d, "cloud_samples")

            row = {
                "strategy": strategy,
                "condition": condition,
                "seed": seed,

                "final_acc": acc[-1],
                "mean_acc": mean(acc),
                "mean_reward": mean(reward),
                "mean_latency_s": mean(latency),
                "mean_train_time_s": mean(train),

                # Sum round-level communication, convert bytes -> MiB
                "total_comm_mib": sum(comm) / (1024 ** 2),

                "sla10_rate": mean(sla10),
                "sla12_rate": mean(sla12),
                "sla15_rate": mean(sla15),

                "mean_offload": mean(offload),
                "mean_churn": mean(churn),
                "mean_selected": mean(selected),
                "total_cloud_samples": sum(cloud),

                # Accuracy windows
                "acc_pre": mean(window(acc, 1, 100)),
                "acc_impair": mean(window(acc, 101, 200)),
                "acc_recovery": mean(window(acc, 201, 300)),

                # Latency windows
                "lat_pre": mean(window(latency, 1, 100)),
                "lat_impair": mean(window(latency, 101, 200)),
                "lat_recovery": mean(window(latency, 201, 300)),

                # Training-time windows
                "train_pre": mean(window(train, 1, 100)),
                "train_impair": mean(window(train, 101, 200)),
                "train_recovery": mean(window(train, 201, 300)),

                # SLA during actual impairment window
                "sla10_impair": mean(window(sla10, 101, 200)),
                "sla12_impair": mean(window(sla12, 101, 200)),
                "sla15_impair": mean(window(sla15, 101, 200)),
            }

            per_run.append(row)


# ------------------------------------------------------------------
# Write per-run table
# ------------------------------------------------------------------

per_run_path = TABLES / "E7_per_run_metrics.csv"

with per_run_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(per_run[0].keys())
    )
    writer.writeheader()
    writer.writerows(per_run)

print(f"WROTE: {per_run_path}")


# ------------------------------------------------------------------
# Aggregate across five seeds
# ------------------------------------------------------------------

groups = defaultdict(list)

for row in per_run:
    groups[(row["strategy"], row["condition"])].append(row)

aggregate_rows = []

metric_names = [
    k for k in per_run[0]
    if k not in {"strategy", "condition", "seed"}
]

for strategy in STRATEGIES:
    for condition in CONDITIONS:

        rows = groups[(strategy, condition)]

        if len(rows) != 5:
            raise RuntimeError(
                f"{strategy}/{condition}: expected 5 seeds"
            )

        out = {
            "strategy": strategy,
            "condition": condition,
            "n_seeds": 5,
        }

        for metric in metric_names:
            values = [float(r[metric]) for r in rows]

            m, s = mean_sd(values)

            out[f"{metric}_mean"] = m
            out[f"{metric}_sd"] = s

        aggregate_rows.append(out)

aggregate_path = TABLES / "E7_5seed_summary.csv"

with aggregate_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(aggregate_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(aggregate_rows)

print(f"WROTE: {aggregate_path}")


# ------------------------------------------------------------------
# Robustness degradation relative to matching NONE baseline
# ------------------------------------------------------------------

robustness_rows = []

for strategy in STRATEGIES:

    baseline_rows = {
        r["seed"]: r
        for r in per_run
        if r["strategy"] == strategy
        and r["condition"] == "none"
    }

    for condition in ["cpu", "network"]:

        condition_rows = {
            r["seed"]: r
            for r in per_run
            if r["strategy"] == strategy
            and r["condition"] == condition
        }

        seed_results = []

        for seed in SEEDS:
            base = baseline_rows[seed]
            impaired = condition_rows[seed]

            # Compare same-seed impairment window with
            # same-seed NONE condition's R101-R200 window.
            lat_pct = (
                (impaired["lat_impair"] - base["lat_impair"])
                / base["lat_impair"]
                * 100.0
            )

            train_pct = (
                (impaired["train_impair"] - base["train_impair"])
                / base["train_impair"]
                * 100.0
            )

            # Positive number = accuracy loss.
            acc_loss_pp = (
                base["acc_impair"] -
                impaired["acc_impair"]
            ) * 100.0

            # Recovery gap relative to matching NONE run.
            recovery_acc_gap_pp = (
                base["acc_recovery"] -
                impaired["acc_recovery"]
            ) * 100.0

            seed_results.append({
                "latency_degradation_pct": lat_pct,
                "train_time_degradation_pct": train_pct,
                "accuracy_loss_pp": acc_loss_pp,
                "recovery_accuracy_gap_pp": recovery_acc_gap_pp,
                "sla10_impair": impaired["sla10_impair"],
                "sla12_impair": impaired["sla12_impair"],
                "sla15_impair": impaired["sla15_impair"],
            })

        out = {
            "strategy": strategy,
            "condition": condition,
            "n_seeds": 5,
        }

        for metric in seed_results[0]:
            values = [r[metric] for r in seed_results]
            m, s = mean_sd(values)

            out[f"{metric}_mean"] = m
            out[f"{metric}_sd"] = s

        robustness_rows.append(out)

robustness_path = TABLES / "E7_robustness_summary.csv"

with robustness_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(robustness_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(robustness_rows)

print(f"WROTE: {robustness_path}")


# ------------------------------------------------------------------
# Human-readable audit report
# ------------------------------------------------------------------

audit_path = AUDIT / "E7_publication_analysis_audit.txt"

with audit_path.open("w") as f:

    f.write("E7 PUBLICATION ANALYSIS AUDIT\n")
    f.write("=" * 80 + "\n\n")

    f.write("Source:\n")
    f.write(str(ROOT) + "\n\n")

    f.write("Frozen JSON files: 45\n")
    f.write("Strategies: fedavg, tl_ppo, tr_dp_afrl\n")
    f.write("Conditions: none, cpu, network\n")
    f.write("Seeds: 42,43,44,45,46\n")
    f.write("Rounds/run: 300\n\n")

    f.write("Windows:\n")
    f.write("PRE      = R1-R100\n")
    f.write("IMPAIR   = R101-R200\n")
    f.write("RECOVERY = R201-R300\n\n")

    f.write("ROBUSTNESS BASELINE:\n")
    f.write(
        "Each CPU/network run is compared against the matching "
        "strategy + seed NONE run over the same R101-R200 window.\n\n"
    )

    f.write("OUTPUTS:\n")
    f.write(str(per_run_path) + "\n")
    f.write(str(aggregate_path) + "\n")
    f.write(str(robustness_path) + "\n\n")

    f.write("STATUS: PASS\n")

print(f"WROTE: {audit_path}")


# ------------------------------------------------------------------
# Console summary
# ------------------------------------------------------------------

print()
print("=" * 110)
print("E7 FIVE-SEED PUBLICATION SUMMARY")
print("=" * 110)

print(
    f"{'STRATEGY':14s} {'COND':9s} "
    f"{'FINAL ACC':>17s} "
    f"{'MEAN ACC':>17s} "
    f"{'LATENCY(s)':>17s} "
    f"{'TRAIN(s)':>17s}"
)

print("-" * 110)

for r in aggregate_rows:

    print(
        f"{r['strategy']:14s} "
        f"{r['condition']:9s} "
        f"{r['final_acc_mean']:.4f}±{r['final_acc_sd']:.4f} "
        f"{r['mean_acc_mean']:.4f}±{r['mean_acc_sd']:.4f} "
        f"{r['mean_latency_s_mean']:.3f}±{r['mean_latency_s_sd']:.3f} "
        f"{r['mean_train_time_s_mean']:.3f}±{r['mean_train_time_s_sd']:.3f}"
    )

print()
print("=" * 110)
print("E7 ROBUSTNESS SUMMARY — MATCHED-SEED NONE BASELINE")
print("=" * 110)

for r in robustness_rows:

    print(
        f"{r['strategy']:14s} {r['condition']:8s} "
        f"lat_deg={r['latency_degradation_pct_mean']:.2f}"
        f"±{r['latency_degradation_pct_sd']:.2f}%  "
        f"train_deg={r['train_time_degradation_pct_mean']:.2f}"
        f"±{r['train_time_degradation_pct_sd']:.2f}%  "
        f"acc_loss={r['accuracy_loss_pp_mean']:.3f}"
        f"±{r['accuracy_loss_pp_sd']:.3f}pp  "
        f"recovery_gap={r['recovery_accuracy_gap_pp_mean']:.3f}"
        f"±{r['recovery_accuracy_gap_pp_sd']:.3f}pp"
    )

print()
print("PASS: E7 publication statistics generated from frozen 45-run dataset")
