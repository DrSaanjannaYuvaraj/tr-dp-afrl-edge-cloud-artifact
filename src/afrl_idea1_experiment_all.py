# afrl_idea1_experiment_all.py (SERVER) — SINGLE REFERENCE (FINAL)
# Purpose: orchestrate multi-seed, multi-strategy runs + ablation labels, and enable logging needed for Fig8/9/11.
#
# Label scheme (IMPORTANT for make_journal_figs_and_table.py):
#   main_label = N10            (feeds Fig3–Fig6 + Fig8–Fig11 + Fig14 + TableX main)
#   with_label = N10_WITH       (ablation telemetry ON for TL/TRDP)
#   no_label   = N10_NO         (ablation telemetry OFF for TL/TRDP)

from __future__ import annotations

import argparse
import os
import subprocess
from typing import List


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_base_cmd(args: argparse.Namespace, seed: int, strat: str) -> List[str]:
    cmd = [
        "python3", "-u", "afrl_idea1_run_one_strategy.py",
        "--strategy", strat,
        "--csv", args.csv,
        "--clients", args.clients,
        "--rounds", str(args.rounds),
        "--seed", str(seed),
        "--outdir", args.outdir,
        "--K", str(args.K),
        "--min_participants", str(args.min_participants),
        "--win", str(args.win),
        "--local_epochs", str(args.local_epochs),
        "--lr_local", str(args.lr_local),
        "--weight_decay", str(args.weight_decay),
        "--batch_size", str(args.batch_size),
        "--max_local_batches", str(args.max_local_batches),
        "--max_local_samples", str(args.max_local_samples),
        "--warm_start_epochs", str(args.warm_start_epochs),
        "--warm_start_lr", str(args.warm_start_lr),
        "--warm_start_max_batches", str(args.warm_start_max_batches),
        "--val_frac", str(args.val_frac),
        "--eval_max_batches", str(args.eval_max_batches),
        "--cloud_lr", str(args.cloud_lr),
        "--cloud_max_batches", str(args.cloud_max_batches),
        "--cloud_samples_per_offload", str(args.cloud_samples_per_offload),
        "--delta_off", str(args.delta_off),
        "--delta_sel", str(args.delta_sel),
        "--http_timeout", str(args.http_timeout),
        "--alpha_acc", str(args.alpha_acc),
        "--beta_lat", str(args.beta_lat),
        "--gamma_off", str(args.gamma_off),
        "--lambda_stab", str(args.lambda_stab),
        "--latency_norm", str(args.latency_norm),
        "--n_classes", str(args.n_classes),
        "--class_map", args.class_map,
        "--scaler_path", args.scaler_path,
        "--label_col", args.label_col,
        # TR-DP knobs (used only by TRDP inside runner)
        "--trdp_eta", str(args.trdp_eta),
        "--trdp_wmin", str(args.trdp_wmin),
        "--trdp_wmax", str(args.trdp_wmax),
        "--trdp_starve_pen", str(args.trdp_starve_pen),
    ]

    if args.client_ids:
        cmd += ["--client_ids", args.client_ids]

    if args.impair_start_round is not None:
        cmd += ["--impair_start_round", str(args.impair_start_round)]
    if args.impair_end_round is not None:
        cmd += ["--impair_end_round", str(args.impair_end_round)]
    if args.cpu_impair_url:
        cmd += ["--cpu_impair_url", args.cpu_impair_url]
    if args.net_impair_url:
        cmd += ["--net_impair_url", args.net_impair_url]

    cmd += [
        "--net_delay_ms", str(args.net_delay_ms),
        "--net_loss_pct", str(args.net_loss_pct),
        "--cpu_workers", str(args.cpu_workers),
    ]

    # Optional toggles (safe even if runner ignores; your runner already logs these anyway)
    if args.log_selection:
        cmd += ["--log_selection"]
    if args.log_overhead:
        cmd += ["--log_overhead"]
    if args.log_churn:
        cmd += ["--log_churn"]
    if args.log_kl:
        cmd += ["--log_kl"]

    return cmd


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--label_col", type=str, default="scenario")
    ap.add_argument("--clients", type=str, required=True)
    ap.add_argument("--client_ids", type=str, default=None)

    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--strategies", type=str, default="fedavg,heuristic,dqn,ddqn,tl_ppo,tr_dp_afrl")
    ap.add_argument("--outdir", type=str, default="/home/ubuntu/afrl_runs")

    # ✅ FIX (1): base label used by plot script
    ap.add_argument(
        "--variant_label",
        type=str,
        default="N10",
        help="Base label for plots (e.g., N10, N4, SEV_A). Produces: N10 (main), N10_WITH, N10_NO (ablation)."
    )

    # Journal defaults
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--min_participants", type=int, default=3)

    # old option retained (but we won't use it for naming; ablation handled by --run_ablation)
    ap.add_argument("--run_ablation", action="store_true",
                    help="Also run TL-PPO/TR-DP-AFRL telemetry ablation: N*_WITH and N*_NO (Fig7).")

    ap.add_argument("--win", type=int, default=10)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--lr_local", type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--max_local_batches", type=int, default=60)
    ap.add_argument("--max_local_samples", type=int, default=2500)

    ap.add_argument("--warm_start_epochs", type=int, default=5)
    ap.add_argument("--warm_start_lr", type=float, default=1e-3)
    ap.add_argument("--warm_start_max_batches", type=int, default=250)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--eval_max_batches", type=int, default=250)

    ap.add_argument("--cloud_lr", type=float, default=1e-3)
    ap.add_argument("--cloud_max_batches", type=int, default=2)
    ap.add_argument("--cloud_samples_per_offload", type=int, default=512)

    ap.add_argument("--delta_off", type=float, default=0.02)
    ap.add_argument("--delta_sel", type=float, default=0.02)
    ap.add_argument("--http_timeout", type=float, default=90.0)

    ap.add_argument("--alpha_acc", type=float, default=1.0)
    ap.add_argument("--beta_lat", type=float, default=0.25)
    ap.add_argument("--gamma_off", type=float, default=0.10)
    ap.add_argument("--lambda_stab", type=float, default=0.25)

    ap.add_argument("--latency_norm", type=float, default=1.0)

    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--class_map", type=str, default="config/class_map.json")
    ap.add_argument("--scaler_path", type=str, default="config/scaler.json")

    # impairments (optional)
    ap.add_argument("--impair_start_round", type=int, default=4)
    ap.add_argument("--impair_end_round", type=int, default=7)
    ap.add_argument("--cpu_impair_url", type=str, default=None)
    ap.add_argument("--net_impair_url", type=str, default=None)
    ap.add_argument("--net_delay_ms", type=float, default=300.0)
    ap.add_argument("--net_loss_pct", type=float, default=10.0)
    ap.add_argument("--cpu_workers", type=int, default=2)

    ap.add_argument("--trdp_eta", type=float, default=0.6)
    ap.add_argument("--trdp_wmin", type=float, default=0.02)
    ap.add_argument("--trdp_wmax", type=float, default=0.70)
    ap.add_argument("--trdp_starve_pen", type=float, default=0.10)

    # logging toggles (no algorithm change)
    ap.add_argument("--log_selection", action="store_true", help="Log selected clients per round into results JSON (Fig8).")
    ap.add_argument("--log_overhead", action="store_true", help="Log telemetry overhead time/bytes into results JSON (Fig9).")
    ap.add_argument("--log_churn", action="store_true", help="Log churn per round into results JSON (Fig11).")
    ap.add_argument("--log_kl", action="store_true", help="Log KL(off/sel) per round into results JSON (audit δ_off/δ_sel).")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    seeds = parse_csv_ints(args.seeds)
    strategies = parse_csv_strs(args.strategies)

    # ✅ FIX (3): correct total count for progress display
    total = len(seeds) * len(strategies)
    if args.run_ablation:
        total += len(seeds) * 4  # TL/TRDP x (WITH + NO)

    run_idx = 0

    # ✅ FIX (2): consistent labels for plot script
    base = (args.variant_label or "").strip()
    if not base:
        base = "N10"
    main_label = base
    with_label = f"{base}_WITH"
    no_label = f"{base}_NO"

    for seed in seeds:
        # -------------------------
        # MAIN RUNS (telemetry ON, label = N10)
        # -------------------------
        for strat in strategies:
            run_idx += 1
            print(f"\n[RUN {run_idx}/{total}] strategy={strat} seed={seed} label={main_label} (telemetry ON)")

            cmd = build_base_cmd(args, seed, strat)
            cmd += ["--variant_label", main_label]
            # telemetry ON => do NOT add --ablate_telemetry
            subprocess.run(cmd, check=True)

        # -------------------------
        # ABLATION RUNS (only TL + TRDP)
        #   N10_WITH (telemetry ON)
        #   N10_NO   (telemetry OFF)
        # -------------------------
        if args.run_ablation:
            for strat in ["tl_ppo", "tr_dp_afrl"]:
                run_idx += 1
                print(f"\n[RUN {run_idx}/{total}] strategy={strat} seed={seed} label={with_label} (telemetry ON)")
                cmd = build_base_cmd(args, seed, strat)
                cmd += ["--variant_label", with_label]
                subprocess.run(cmd, check=True)

            for strat in ["tl_ppo", "tr_dp_afrl"]:
                run_idx += 1
                print(f"\n[RUN {run_idx}/{total}] strategy={strat} seed={seed} label={no_label} (telemetry OFF)")
                cmd = build_base_cmd(args, seed, strat)
                cmd += ["--variant_label", no_label, "--ablate_telemetry"]
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
