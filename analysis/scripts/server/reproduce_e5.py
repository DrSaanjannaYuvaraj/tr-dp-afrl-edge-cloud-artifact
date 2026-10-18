#!/usr/bin/env python3
"""Reproduce E5 publication CSVs from five original experiment JSON files."""

from pathlib import Path
import argparse
import csv
import json
import statistics

METRICS = [
    "acc", "reward", "latency_s", "train_time_s",
    "offload_intensity", "churn_ratio", "selected_count",
    "physical_phase_wall_s", "physical_request_max_s",
    "physical_request_sum_s", "client_compute_max_s",
    "client_compute_sum_s", "cloud_compute_s",
    "communication_bytes_total", "physical_model_comm_bytes",
    "physical_download_bytes", "physical_upload_bytes",
    "model_payload_bytes", "controller_decision_time_s",
    "selection_decision_time_s", "offload_decision_time_s",
    "aggregation_time_s", "evaluation_time_s",
    "telemetry_overhead_ms", "telemetry_overhead_bytes",
    "sla_violation_10s", "sla_violation_12s",
    "sla_violation_15s",
]

CUMULATIVE = [
    "cumulative_communication_bytes",
    "cumulative_wall_time_s",
    "cumulative_sla_violation_rate_10s",
    "cumulative_sla_violation_rate_12s",
    "cumulative_sla_violation_rate_15s",
]

PUBLICATION = [
    ("Learning", "Test accuracy", "acc", 1, 4, ""),
    ("Learning", "Reward", "reward", 1, 4, ""),
    ("Physical execution", "Physical phase wall time",
     "physical_phase_wall_s", 1, 3, "s"),
    ("Physical execution", "Maximum physical request time",
     "physical_request_max_s", 1, 3, "s"),
    ("Physical execution", "Maximum client compute time",
     "client_compute_max_s", 1, 3, "s"),
    ("Physical execution", "Cloud compute time",
     "cloud_compute_s", 1, 3, "s"),
    ("Communication", "Total communication per round",
     "communication_bytes_total", 1e-6, 2, "MB/round"),
    ("Communication", "Download traffic",
     "physical_download_bytes", 1e-6, 2, "MB/round"),
    ("Communication", "Upload traffic",
     "physical_upload_bytes", 1e-6, 2, "MB/round"),
    ("Controller", "Controller decision latency",
     "controller_decision_time_s", 1000, 3, "ms"),
    ("Controller", "Selection decision latency",
     "selection_decision_time_s", 1000, 3, "ms"),
    ("Controller", "Offload decision latency",
     "offload_decision_time_s", 1000, 3, "ms"),
    ("Telemetry", "Telemetry overhead",
     "telemetry_overhead_ms", 1, 2, "ms"),
    ("SLA", "10-second violation rate",
     "sla_violation_10s", 100, 1, "%"),
    ("SLA", "12-second violation rate",
     "sla_violation_12s", 100, 1, "%"),
    ("SLA", "15-second violation rate",
     "sla_violation_15s", 100, 1, "%"),
]

def write_csv(path, columns, rows):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []

    for seed in range(42, 47):
        path = args.raw_dir / (
            f"results_tr_dp_afrl_N10_seed{seed}.json"
        )
        data = json.loads(path.read_text())

        rounds = data["round"]
        assert len(rounds) == 300

        indices = [
            i for i, r in enumerate(rounds)
            if 251 <= r <= 300
        ]
        assert len(indices) == 50

        row = {
            "configuration": "tr_dp_afrl",
            "variant": "N10",
            "seed": seed,
            "steady_round_start": 251,
            "steady_round_end": 300,
            "n_steady_rounds": 50,
        }

        for metric in METRICS:
            values = [
                float(data[metric][i])
                for i in indices
            ]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_round_sd"] = statistics.stdev(values)

        for metric in CUMULATIVE:
            row[f"{metric}_final"] = float(data[metric][-1])

        per_seed_rows.append(row)

    per_seed_columns = [
        "configuration", "variant", "seed",
        "steady_round_start", "steady_round_end",
        "n_steady_rounds",
    ]

    for metric in METRICS:
        per_seed_columns.extend([
            f"{metric}_mean",
            f"{metric}_round_sd",
        ])

    per_seed_columns.extend(
        f"{metric}_final" for metric in CUMULATIVE
    )

    write_csv(
        args.output_dir / "E5_per_seed_steady_summary.csv",
        per_seed_columns,
        per_seed_rows,
    )

    summary = {
        "configuration": "tr_dp_afrl",
        "variant": "N10",
        "n_seeds": 5,
        "steady_round_start": 251,
        "steady_round_end": 300,
    }

    summary_columns = list(summary)

    for metric in METRICS + [
        f"{name}_final" for name in CUMULATIVE
    ]:
        source = (
            metric if metric.endswith("_final")
            else f"{metric}_mean"
        )

        values = [
            float(row[source])
            for row in per_seed_rows
        ]

        summary[f"{metric}_mean"] = statistics.mean(values)
        summary[f"{metric}_sd"] = statistics.stdev(values)

        summary_columns.extend([
            f"{metric}_mean",
            f"{metric}_sd",
        ])

    write_csv(
        args.output_dir / "E5_5seed_summary.csv",
        summary_columns,
        [summary],
    )

    publication_rows = []

    for category, label, metric, factor, decimals, unit in PUBLICATION:
        mean = float(summary[f"{metric}_mean"]) * factor
        sd = float(summary[f"{metric}_sd"]) * factor

        mean_text = f"{mean:.{decimals}f}"
        sd_text = f"{sd:.{decimals}f}"

        publication_rows.append({
            "Category": category,
            "Metric": label,
            "Mean": mean_text,
            "SD": sd_text,
            "Mean ± SD": f"{mean_text} ± {sd_text}",
            "Unit": unit,
            "Seeds": 5,
            "Steady-state rounds": "251-300",
        })

    publication_columns = [
        "Category", "Metric", "Mean", "SD",
        "Mean ± SD", "Unit", "Seeds",
        "Steady-state rounds",
    ]

    write_csv(
        args.output_dir / "E5_publication_table.csv",
        publication_columns,
        publication_rows,
    )

    print("=" * 70)
    print("E5 PORTABLE CSV REPRODUCTION")
    print("=" * 70)
    print("Seeds: 42–46")
    print("Steady-state rounds: 251–300")
    print("Per-seed metrics:", len(METRICS))
    print("Cumulative metrics:", len(CUMULATIVE))
    print("Publication metrics:", len(PUBLICATION))
    print("Output:", args.output_dir)
    print("STATUS: THREE CSV FILES GENERATED")
    print("Original publication files unchanged")

if __name__ == "__main__":
    main()
