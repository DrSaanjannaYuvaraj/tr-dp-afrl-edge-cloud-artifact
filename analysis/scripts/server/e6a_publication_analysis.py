from pathlib import Path
import csv
import json
import re
import statistics as st


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

SRC = REPO_ROOT / "results" / "frozen" / "e6a_physical"

PUB = REPO_ROOT / "outputs" / "regenerated_analysis" / "e6a"

TABLEDIR = PUB / "tables"
AUDITDIR = PUB / "audit"

TABLEDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Frozen experimental design
# ------------------------------------------------------------

EXPECTED_N = [5, 8, 10]
EXPECTED_SEEDS = [42, 43, 44, 45, 46]

STEADY_START = 251
STEADY_END = 300

# Python slice corresponding to rounds 251..300.
STEADY_SLICE = slice(250, 300)


# ------------------------------------------------------------
# Publication metrics
# ------------------------------------------------------------

STEADY_METRICS = [
    "acc",
    "reward",
    "latency_s",
    "train_time_s",
    "physical_phase_wall_s",
    "client_compute_max_s",
    "controller_decision_time_s",
    "telemetry_overhead_ms",
    "churn_ratio",
    "sla_violation_10s",
    "communication_bytes_total",
]

TOTAL_METRICS = [
    "cumulative_wall_time_s",
    "cumulative_communication_bytes",
]


# ------------------------------------------------------------
# Discover and validate exactly 15 source files
# ------------------------------------------------------------

pattern = re.compile(
    r"results_tr_dp_afrl_E6A_N(5|8|10)_seed(42|43|44|45|46)\.json$"
)

records = {}

for path in sorted(SRC.glob("results_tr_dp_afrl_E6A_N*_seed*.json")):

    m = pattern.fullmatch(path.name)

    if not m:
        continue

    n = int(m.group(1))
    seed = int(m.group(2))

    key = (n, seed)

    if key in records:
        raise RuntimeError(
            f"Duplicate result for N={n}, seed={seed}"
        )

    with path.open() as f:
        data = json.load(f)

    records[key] = (path, data)


expected = {
    (n, seed)
    for n in EXPECTED_N
    for seed in EXPECTED_SEEDS
}

if set(records) != expected:
    missing = sorted(expected - set(records))
    extra = sorted(set(records) - expected)

    raise RuntimeError(
        f"E6A source-set mismatch. "
        f"Missing={missing}, Extra={extra}"
    )


# ------------------------------------------------------------
# Validate round-series structure
# ------------------------------------------------------------

for (n, seed), (path, data) in sorted(records.items()):

    if "round" not in data:
        raise RuntimeError(
            f"{path.name}: missing round series"
        )

    if len(data["round"]) != 300:
        raise RuntimeError(
            f"{path.name}: expected 300 rounds, "
            f"found {len(data['round'])}"
        )

    expected_rounds = list(range(1, 301))

    if [int(x) for x in data["round"]] != expected_rounds:
        raise RuntimeError(
            f"{path.name}: round sequence is not 1..300"
        )

    for metric in STEADY_METRICS + TOTAL_METRICS:

        if metric not in data:
            raise RuntimeError(
                f"{path.name}: missing metric {metric}"
            )

        if not isinstance(data[metric], list):
            raise RuntimeError(
                f"{path.name}: {metric} is not a list"
            )

        if len(data[metric]) != 300:
            raise RuntimeError(
                f"{path.name}: {metric} length "
                f"{len(data[metric])}, expected 300"
            )


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def mean_window(data, metric):
    values = [
        float(x)
        for x in data[metric][STEADY_SLICE]
    ]

    if len(values) != 50:
        raise RuntimeError(
            f"{metric}: steady-state window "
            f"contains {len(values)} values, expected 50"
        )

    return st.mean(values)


def final_value(data, metric):
    return float(data[metric][-1])


def mean_sd(values):
    values = [float(x) for x in values]

    if len(values) != 5:
        raise RuntimeError(
            f"Expected five seed values, found {len(values)}"
        )

    return st.mean(values), st.stdev(values)


# ------------------------------------------------------------
# Per-seed publication table
# ------------------------------------------------------------

per_seed_rows = []

for n in EXPECTED_N:

    for seed in EXPECTED_SEEDS:

        path, data = records[(n, seed)]

        row = {
            "N": n,
            "seed": seed,
            "source_file": path.name,
            "steady_round_start": STEADY_START,
            "steady_round_end": STEADY_END,
        }

        for metric in STEADY_METRICS:
            row[f"{metric}_steady_mean"] = mean_window(
                data,
                metric,
            )

        # Convenience publication units.
        row["controller_decision_ms_steady_mean"] = (
            row["controller_decision_time_s_steady_mean"]
            * 1000.0
        )

        row["communication_mb_per_round_steady_mean"] = (
            row["communication_bytes_total_steady_mean"]
            / 1_000_000.0
        )

        # Full-run totals are the final cumulative values.
        row["total_wall_time_s"] = final_value(
            data,
            "cumulative_wall_time_s",
        )

        row["total_communication_gb"] = (
            final_value(
                data,
                "cumulative_communication_bytes",
            )
            / 1_000_000_000.0
        )

        per_seed_rows.append(row)


PER_SEED_CSV = TABLEDIR / "E6A_per_seed_summary.csv"

with PER_SEED_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(per_seed_rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(per_seed_rows)


# ------------------------------------------------------------
# Five-seed publication summary
# mean +/- sample SD across the five per-seed values
# ------------------------------------------------------------

SUMMARY_FIELDS = [
    "acc_steady_mean",
    "reward_steady_mean",
    "latency_s_steady_mean",
    "train_time_s_steady_mean",
    "physical_phase_wall_s_steady_mean",
    "client_compute_max_s_steady_mean",
    "controller_decision_ms_steady_mean",
    "telemetry_overhead_ms_steady_mean",
    "churn_ratio_steady_mean",
    "sla_violation_10s_steady_mean",
    "communication_mb_per_round_steady_mean",
    "total_wall_time_s",
    "total_communication_gb",
]

summary_rows = []

for n in EXPECTED_N:

    seed_rows = [
        r
        for r in per_seed_rows
        if int(r["N"]) == n
    ]

    if len(seed_rows) != 5:
        raise RuntimeError(
            f"N={n}: expected five seed rows"
        )

    out = {
        "N": n,
        "n_seeds": 5,
        "steady_round_start": STEADY_START,
        "steady_round_end": STEADY_END,
    }

    for field in SUMMARY_FIELDS:

        vals = [
            float(r[field])
            for r in seed_rows
        ]

        mean, sd = mean_sd(vals)

        out[f"{field}_mean"] = mean
        out[f"{field}_sd"] = sd

    summary_rows.append(out)


SUMMARY_CSV = TABLEDIR / "E6A_5seed_summary.csv"

with SUMMARY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(summary_rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# ------------------------------------------------------------
# Human-readable audit
# ------------------------------------------------------------

AUDIT = AUDITDIR / "E6A_publication_analysis_audit.txt"

with AUDIT.open("w") as f:

    f.write("E6A PUBLICATION ANALYSIS AUDIT\n")
    f.write("=" * 72 + "\n\n")

    f.write(
        "Source directory: "
        f"{SRC}\n"
    )

    f.write(
        "Source result files: 15\n"
    )

    f.write(
        "Physical client populations: "
        "N={5,8,10}\n"
    )

    f.write(
        "Seeds: {42,43,44,45,46}\n"
    )

    f.write(
        "Rounds per run: 300\n"
    )

    f.write(
        "Steady-state window: "
        "rounds 251-300 inclusive\n"
    )

    f.write(
        "Within-seed statistic: "
        "arithmetic mean over rounds 251-300\n"
    )

    f.write(
        "Across-seed statistic: "
        "arithmetic mean +/- sample SD "
        "(statistics.stdev, n=5)\n"
    )

    f.write(
        "Communication conversion: "
        "decimal MB=1e6 bytes, GB=1e9 bytes\n"
    )

    f.write(
        "Full-run totals: final values of "
        "cumulative_wall_time_s and "
        "cumulative_communication_bytes\n"
    )

    f.write("\nVALIDATION\n")
    f.write("-" * 72 + "\n")

    for n in EXPECTED_N:
        for seed in EXPECTED_SEEDS:
            path, _ = records[(n, seed)]
            f.write(
                f"PASS N={n} seed={seed} "
                f"{path.name}\n"
            )

    f.write("\nPASS: exact 15-run source set verified\n")
    f.write("PASS: all required series contain 300 values\n")
    f.write("PASS: round sequence verified as 1..300\n")
    f.write(
        "PASS: E6A publication statistics generated "
        "without modifying source results\n"
    )


# ------------------------------------------------------------
# Console summary
# ------------------------------------------------------------

print("=== E6A PUBLICATION ANALYSIS READY ===")
print()
print("Source runs:", len(records))
print(
    "Steady-state window:",
    f"{STEADY_START}-{STEADY_END}",
)
print(
    "Across-seed statistic:",
    "mean +/- sample SD (n=5)",
)

print()

for row in summary_rows:

    n = row["N"]

    print("=" * 72)
    print(f"N={n}")
    print("=" * 72)

    print(
        "Accuracy:",
        f"{row['acc_steady_mean_mean']:.4f} "
        f"+/- {row['acc_steady_mean_sd']:.4f}"
    )

    print(
        "Reward:",
        f"{row['reward_steady_mean_mean']:.3f} "
        f"+/- {row['reward_steady_mean_sd']:.3f}"
    )

    print(
        "Latency (s):",
        f"{row['latency_s_steady_mean_mean']:.3f} "
        f"+/- {row['latency_s_steady_mean_sd']:.3f}"
    )

    print(
        "Train time (s):",
        f"{row['train_time_s_steady_mean_mean']:.3f} "
        f"+/- {row['train_time_s_steady_mean_sd']:.3f}"
    )

    print(
        "Physical wall (s):",
        f"{row['physical_phase_wall_s_steady_mean_mean']:.3f} "
        f"+/- {row['physical_phase_wall_s_steady_mean_sd']:.3f}"
    )

    print(
        "Controller decision (ms):",
        f"{row['controller_decision_ms_steady_mean_mean']:.3f} "
        f"+/- {row['controller_decision_ms_steady_mean_sd']:.3f}"
    )

    print(
        "Telemetry overhead (ms):",
        f"{row['telemetry_overhead_ms_steady_mean_mean']:.3f} "
        f"+/- {row['telemetry_overhead_ms_steady_mean_sd']:.3f}"
    )

    print(
        "Comm / round (MB):",
        f"{row['communication_mb_per_round_steady_mean_mean']:.3f} "
        f"+/- {row['communication_mb_per_round_steady_mean_sd']:.3f}"
    )

    print(
        "Total wall (s):",
        f"{row['total_wall_time_s_mean']:.1f} "
        f"+/- {row['total_wall_time_s_sd']:.1f}"
    )

    print(
        "Total communication (GB):",
        f"{row['total_communication_gb_mean']:.3f} "
        f"+/- {row['total_communication_gb_sd']:.3f}"
    )

    print()

print("WROTE:", PER_SEED_CSV)
print("WROTE:", SUMMARY_CSV)
print("WROTE:", AUDIT)
print()
print("PASS: E6A publication analysis complete")
