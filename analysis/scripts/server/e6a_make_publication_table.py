from pathlib import Path
import csv

PUB = Path("/home/ubuntu/afrl_runs/E6A_PUBLICATION")
SRC = PUB / "tables" / "E6A_5seed_summary.csv"
OUT = PUB / "tables" / "E6A_scalability_publication_table.tex"

if not SRC.exists():
    raise FileNotFoundError(f"Missing verified summary: {SRC}")

with SRC.open(newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 3:
    raise RuntimeError(f"Expected exactly 3 N rows, found {len(rows)}")

expected_N = [5, 8, 10]
actual_N = [int(r["N"]) for r in rows]

if actual_N != expected_N:
    raise RuntimeError(
        f"Expected N={expected_N}, found N={actual_N}"
    )

def pm(mean_key, sd_key, decimals):
    mean = float(row[mean_key])
    sd = float(row[sd_key])
    return f"{mean:.{decimals}f} $\\pm$ {sd:.{decimals}f}"

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Physical scalability of TR-DP-AFRL across increasing numbers of physical clients. "
    r"Values report mean $\pm$ sample standard deviation across five seeds. "
    r"Steady-state metrics are averaged over rounds 251--300.}",
    r"\label{tab:e6a_physical_scalability}",
    r"\resizebox{\columnwidth}{!}{%",
    r"\begin{tabular}{c c c c c c}",
    r"\hline",
    r"$N$ & Accuracy & Latency (s) & Train time (s) & "
    r"Telemetry (ms) & Comm./round (MB) \\",
    r"\hline",
]

for row in rows:
    n = int(row["N"])

    acc = pm(
        "acc_steady_mean_mean",
        "acc_steady_mean_sd",
        4,
    )

    latency = pm(
        "latency_s_steady_mean_mean",
        "latency_s_steady_mean_sd",
        3,
    )

    train = pm(
        "train_time_s_steady_mean_mean",
        "train_time_s_steady_mean_sd",
        3,
    )

    telemetry = pm(
        "telemetry_overhead_ms_steady_mean_mean",
        "telemetry_overhead_ms_steady_mean_sd",
        3,
    )

    comm = pm(
        "communication_mb_per_round_steady_mean_mean",
        "communication_mb_per_round_steady_mean_sd",
        3,
    )

    lines.append(
        f"{n} & {acc} & {latency} & {train} & "
        f"{telemetry} & {comm} \\\\"
    )

lines += [
    r"\hline",
    r"\end{tabular}%",
    r"}",
    r"\end{table}",
    "",
]

OUT.write_text("\n".join(lines))

print(f"WROTE: {OUT}")
print()
print(OUT.read_text())
print("PASS: E6A publication table generated from verified five-seed summary")
