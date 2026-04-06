#!/usr/bin/env bash
set -euo pipefail

echo "Update CLIENT1_IP to CLIENT10_IP placeholders before running."
echo "This script mirrors the optional ablation run from the execution workflow."

mkdir -p outputs/N10_300r_main

nohup python3 src/afrl_idea1_experiment_all.py \
  --csv data/telemetry_merged.csv \
  --class_map config/class_map.json \
  --scaler_path config/scaler.json \
  --clients "http://CLIENT1_IP:8000,http://CLIENT2_IP:8000,http://CLIENT3_IP:8000,http://CLIENT4_IP:8000,http://CLIENT5_IP:8000,http://CLIENT6_IP:8000,http://CLIENT7_IP:8000,http://CLIENT8_IP:8000,http://CLIENT9_IP:8000,http://CLIENT10_IP:8000" \
  --rounds 300 \
  --seeds 46 \
  --run_ablation \
  --strategies tl_ppo,tr_dp_afrl \
  --K 8 \
  --min_participants 5 \
  --outdir outputs/N10_300r_main \
  --variant_label N10 \
  --cloud_max_batches 4 \
  --cloud_samples_per_offload 1024 \
  --alpha_acc 1.2 \
  --beta_lat 0.12 \
  --gamma_off 0.04 \
  --latency_norm 10.0 \
  --delta_off 0.035 \
  --delta_sel 0.06 \
  --lambda_stab 0.06 \
  --trdp_eta 0.9 \
  --trdp_wmin 0.01 \
  --trdp_wmax 0.60 \
  --trdp_starve_pen 0.10 \
  --http_timeout 90 \
  --log_selection --log_overhead --log_churn --log_kl \
  > outputs/N10_300r_main/run_N10_300r_ablation.log 2>&1 &

echo "Ablation job launched. Tail outputs/N10_300r_main/run_N10_300r_ablation.log"
