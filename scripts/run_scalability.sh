#!/usr/bin/env bash
set -euo pipefail

echo "Update all CLIENT*_IP placeholders before running."
echo "This script mirrors the original scalability workflow: N=4, N=6, N=8, N=10."
echo "All runs use nohup and write to separate output folders."

mkdir -p outputs/N4_300r_trial outputs/N6_300r_trial outputs/N8_300r_trial outputs/N10_300r_trial

# N=4 : clients 1,2,7,8
nohup python3 src/afrl_idea1_experiment_all.py \
  --csv data/telemetry_merged.csv \
  --class_map config/class_map.json \
  --scaler_path config/scaler.json \
  --clients "http://CLIENT1_IP:8000,http://CLIENT2_IP:8000,http://CLIENT7_IP:8000,http://CLIENT8_IP:8000" \
  --rounds 300 \
  --seeds 42,43,44,45,46 \
  --strategies tl_ppo,tr_dp_afrl \
  --K 3 \
  --min_participants 2 \
  --outdir outputs/N4_300r_trial \
  --variant_label N4 \
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
  > outputs/N4_300r_trial/run_N4_300r.log 2>&1 &

# N=6 : clients 1,2,5,6,7,8
nohup python3 src/afrl_idea1_experiment_all.py \
  --csv data/telemetry_merged.csv \
  --class_map config/class_map.json \
  --scaler_path config/scaler.json \
  --clients "http://CLIENT1_IP:8000,http://CLIENT2_IP:8000,http://CLIENT5_IP:8000,http://CLIENT6_IP:8000,http://CLIENT7_IP:8000,http://CLIENT8_IP:8000" \
  --rounds 300 \
  --seeds 42,43,44,45,46 \
  --strategies tl_ppo,tr_dp_afrl \
  --K 5 \
  --min_participants 3 \
  --outdir outputs/N6_300r_trial \
  --variant_label N6 \
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
  > outputs/N6_300r_trial/run_N6_300r_5seeds.log 2>&1 &

# N=8 : clients 1..8
nohup python3 src/afrl_idea1_experiment_all.py \
  --csv data/telemetry_merged.csv \
  --class_map config/class_map.json \
  --scaler_path config/scaler.json \
  --clients "http://CLIENT1_IP:8000,http://CLIENT2_IP:8000,http://CLIENT3_IP:8000,http://CLIENT4_IP:8000,http://CLIENT5_IP:8000,http://CLIENT6_IP:8000,http://CLIENT7_IP:8000,http://CLIENT8_IP:8000" \
  --rounds 300 \
  --seeds 42,43,44,45,46 \
  --strategies tl_ppo,tr_dp_afrl \
  --K 6 \
  --min_participants 4 \
  --outdir outputs/N8_300r_trial \
  --variant_label N8 \
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
  > outputs/N8_300r_trial/run_N8_300r_5seeds.log 2>&1 &

# N=10 : clients 1..10
nohup python3 src/afrl_idea1_experiment_all.py \
  --csv data/telemetry_merged.csv \
  --class_map config/class_map.json \
  --scaler_path config/scaler.json \
  --clients "http://CLIENT1_IP:8000,http://CLIENT2_IP:8000,http://CLIENT3_IP:8000,http://CLIENT4_IP:8000,http://CLIENT5_IP:8000,http://CLIENT6_IP:8000,http://CLIENT7_IP:8000,http://CLIENT8_IP:8000,http://CLIENT9_IP:8000,http://CLIENT10_IP:8000" \
  --rounds 300 \
  --seeds 42,43,44,45,46 \
  --strategies tl_ppo,tr_dp_afrl \
  --K 8 \
  --min_participants 5 \
  --outdir outputs/N10_300r_trial \
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
  > outputs/N10_300r_trial/run_N10_300r_5seeds.log 2>&1 &

echo "Scalability jobs launched. Use ps/pgrep and tail the logs under outputs/."
