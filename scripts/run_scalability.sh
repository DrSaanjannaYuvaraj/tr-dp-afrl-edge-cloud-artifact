#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User-editable placeholders
# -----------------------------
CLIENT_ID1="<CLIENT_ID1>"
CLIENT_ID2="<CLIENT_ID2>"
CLIENT_ID3="<CLIENT_ID3>"
CLIENT_ID4="<CLIENT_ID4>"
CLIENT_ID5="<CLIENT_ID5>"
CLIENT_ID6="<CLIENT_ID6>"
CLIENT_ID7="<CLIENT_ID7>"
CLIENT_ID8="<CLIENT_ID8>"
CLIENT_ID9="<CLIENT_ID9>"
CLIENT_ID10="<CLIENT_ID10>"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"

echo "Repository root: $REPO_ROOT"
echo "Reminder: replace all <CLIENT_ID*> placeholders with real client IP addresses before running."

if [[ -n "$VENV_ACTIVATE" ]]; then
  if [[ -f "$VENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
    echo "Activated environment from: $VENV_ACTIVATE"
  else
    echo "ERROR: VENV_ACTIVATE is set but file not found: $VENV_ACTIVATE"
    exit 1
  fi
else
  echo "No VENV_ACTIVATE provided. Assuming the Python environment is already active."
fi

ALL_CLIENT_IDS=("$CLIENT_ID1" "$CLIENT_ID2" "$CLIENT_ID3" "$CLIENT_ID4" "$CLIENT_ID5" "$CLIENT_ID6" "$CLIENT_ID7" "$CLIENT_ID8" "$CLIENT_ID9" "$CLIENT_ID10")
for client_ip in "${ALL_CLIENT_IDS[@]}"; do
  if [[ "$client_ip" == "<CLIENT_ID"* ]]; then
    echo "ERROR: One or more client placeholders are still not updated."
    echo "Please replace <CLIENT_ID1> ... <CLIENT_ID10> with real client IP addresses."
    exit 1
  fi
done

OUTROOT="${OUTROOT:-$REPO_ROOT/outputs}"

mkdir -p \
  "$OUTROOT/N4_300r_trial" \
  "$OUTROOT/N6_300r_trial" \
  "$OUTROOT/N8_300r_trial" \
  "$OUTROOT/N10_300r_trial"

cd "$REPO_ROOT"

nohup python3 -u src/afrl_idea1_experiment_all.py \
  --csv "$REPO_ROOT/data/telemetry_merged.csv" \
  --class_map "$REPO_ROOT/config/class_map.json" \
  --scaler_path "$REPO_ROOT/config/scaler.json" \
  --clients "http://${CLIENT_ID1}:8000,http://${CLIENT_ID2}:8000,http://${CLIENT_ID7}:8000,http://${CLIENT_ID8}:8000" \
  --rounds 300 \
  --seeds 42,43,44,45,46 \
  --strategies tl_ppo,tr_dp_afrl \
  --K 3 \
  --min_participants 2 \
  --outdir "$OUTROOT/N4_300r_trial" \
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
  > "$OUTROOT/N4_300r_trial/run_N4_300r.log" 2>&1 &

nohup python3 -u src/afrl_idea1_experiment_all.py \
  --csv "$REPO_ROOT/data/telemetry_merged.csv" \
  --class_map "$REPO_ROOT/config/class_map.json" \
  --scaler_path "$REPO_ROOT/config/scaler.json" \
  --clients "http://${CLIENT_ID1}:8000,http://${CLIENT_ID2}:8000,http://${CLIENT_ID5}:8000,http://${CLIENT_ID6}:8000,http://${CLIENT_ID7}:8000,http://${CLIENT_ID8}:8000" \
  --rounds 300 \
  --seeds 42,43,44,45,46 \
  --strategies tl_ppo,tr_dp_afrl \
  --K 5 \
  --min_participants 3 \
  --outdir "$OUTROOT/N6_300r_trial" \
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
  > "$OUTROOT/N6_300r_trial/run_N6_300r_5seeds.log" 2>&1 &

nohup python3 -u src/afrl_idea1_experiment_all.py \
  --csv "$REPO_ROOT/data/telemetry_merged.csv" \
  --class_map "$REPO_ROOT/config/class_map.json" \
  --scaler_path "$REPO_ROOT/config/scaler.json" \
  --clients "http://${CLIENT_ID1}:8000,http://${CLIENT_ID2}:8000,http://${CLIENT_ID3}:8000,http://${CLIENT_ID4}:8000,http://${CLIENT_ID5}:8000,http://${CLIENT_ID6}:8000,http://${CLIENT_ID7}:8000,http://${CLIENT_ID8}:8000" \
  --rounds 300 \
  --seeds 42,43,44,45,46 \
  --strategies tl_ppo,tr_dp_afrl \
  --K 6 \
  --min_participants 4 \
  --outdir "$OUTROOT/N8_300r_trial" \
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
  > "$OUTROOT/N8_300r_trial/run_N8_300r_5seeds.log" 2>&1 &

nohup python3 -u src/afrl_idea1_experiment_all.py \
  --csv "$REPO_ROOT/data/telemetry_merged.csv" \
  --class_map "$REPO_ROOT/config/class_map.json" \
  --scaler_path "$REPO_ROOT/config/scaler.json" \
  --clients "http://${CLIENT_ID1}:8000,http://${CLIENT_ID2}:8000,http://${CLIENT_ID3}:8000,http://${CLIENT_ID4}:8000,http://${CLIENT_ID5}:8000,http://${CLIENT_ID6}:8000,http://${CLIENT_ID7}:8000,http://${CLIENT_ID8}:8000,http://${CLIENT_ID9}:8000,http://${CLIENT_ID10}:8000" \
  --rounds 300 \
  --seeds 42,43,44,45,46 \
  --strategies tl_ppo,tr_dp_afrl \
  --K 8 \
  --min_participants 5 \
  --outdir "$OUTROOT/N10_300r_trial" \
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
  > "$OUTROOT/N10_300r_trial/run_N10_300r_5seeds.log" 2>&1 &

echo "Scalability jobs started."
echo "Update placeholders first, then monitor logs under: $OUTROOT"
