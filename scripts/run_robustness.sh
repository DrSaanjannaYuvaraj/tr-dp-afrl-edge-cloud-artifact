#!/usr/bin/env bash
set -euo pipefail

echo "Update CLIENT1_IP to CLIENT4_IP placeholders before running."
echo "This script mirrors the original robustness severity sweep A-E."
echo "Each run is launched with nohup and a separate output folder."

mkdir -p \
  outputs/SEV_A_N1CPU_300r \
  outputs/SEV_B_N2CPU_300r \
  outputs/SEV_C_N3CPU_300r \
  outputs/SEV_D_N4CPU_300r \
  outputs/SEV_E_N4CPU_300r

COMMON_ARGS=(
  --csv data/telemetry_merged.csv
  --class_map config/class_map.json
  --scaler_path config/scaler.json
  --clients "http://CLIENT1_IP:8000,http://CLIENT2_IP:8000,http://CLIENT3_IP:8000,http://CLIENT4_IP:8000"
  --rounds 300
  --seeds 42,43,44,45,46
  --strategies fedavg,tl_ppo,tr_dp_afrl
  --K 3
  --min_participants 2
  --impair_start_round 100
  --impair_end_round 200
  --cpu_impair_url http://CLIENT1_IP:8000
  --net_impair_url http://CLIENT4_IP:8000
  --cloud_max_batches 4
  --cloud_samples_per_offload 1024
  --alpha_acc 1.2
  --beta_lat 0.12
  --gamma_off 0.04
  --latency_norm 10.0
  --delta_off 0.035
  --delta_sel 0.06
  --lambda_stab 0.06
  --trdp_eta 0.9
  --trdp_wmin 0.01
  --trdp_wmax 0.60
  --trdp_starve_pen 0.10
  --log_selection --log_overhead --log_churn --log_kl
)

# Severity A
nohup python3 src/afrl_idea1_experiment_all.py \
  "${COMMON_ARGS[@]}" \
  --outdir outputs/SEV_A_N1CPU_300r \
  --variant_label SEV_A_N1CPU \
  --net_delay_ms 90 \
  --net_loss_pct 1 \
  --cpu_workers 1 \
  --http_timeout 120 \
  > outputs/SEV_A_N1CPU_300r/run_SEV_A_N1CPU_300r.log 2>&1 &

# Severity B
nohup python3 src/afrl_idea1_experiment_all.py \
  "${COMMON_ARGS[@]}" \
  --outdir outputs/SEV_B_N2CPU_300r \
  --variant_label SEV_B_N2CPU \
  --net_delay_ms 90 \
  --net_loss_pct 3 \
  --cpu_workers 2 \
  --http_timeout 120 \
  > outputs/SEV_B_N2CPU_300r/run_SEV_B_N2CPU_300r.log 2>&1 &

# Severity C
nohup python3 src/afrl_idea1_experiment_all.py \
  "${COMMON_ARGS[@]}" \
  --outdir outputs/SEV_C_N3CPU_300r \
  --variant_label SEV_C_N3CPU \
  --net_delay_ms 180 \
  --net_loss_pct 5 \
  --cpu_workers 3 \
  --http_timeout 180 \
  > outputs/SEV_C_N3CPU_300r/run_SEV_C_N3CPU_300r.log 2>&1 &

# Severity D
nohup python3 src/afrl_idea1_experiment_all.py \
  "${COMMON_ARGS[@]}" \
  --outdir outputs/SEV_D_N4CPU_300r \
  --variant_label SEV_D_N4CPU \
  --net_delay_ms 180 \
  --net_loss_pct 8 \
  --cpu_workers 4 \
  --http_timeout 180 \
  > outputs/SEV_D_N4CPU_300r/run_SEV_D_N4CPU_300r.log 2>&1 &

# Severity E
nohup python3 src/afrl_idea1_experiment_all.py \
  "${COMMON_ARGS[@]}" \
  --outdir outputs/SEV_E_N4CPU_300r \
  --variant_label SEV_E_N4CPU \
  --net_delay_ms 240 \
  --net_loss_pct 12 \
  --cpu_workers 4 \
  --http_timeout 240 \
  > outputs/SEV_E_N4CPU_300r/run_SEV_E_N4CPU_300r.log 2>&1 &

echo "Robustness jobs launched. Use ps/pgrep and tail the logs under outputs/."
