#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User-editable placeholders
# -----------------------------
CLIENT_ID1="<CLIENT_ID1>"
CLIENT_ID2="<CLIENT_ID2>"
CLIENT_ID3="<CLIENT_ID3>"
CLIENT_ID4="<CLIENT_ID4>"

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

if [[ "$CLIENT_ID1" == "<CLIENT_ID1>" || "$CLIENT_ID2" == "<CLIENT_ID2>" || "$CLIENT_ID3" == "<CLIENT_ID3>" || "$CLIENT_ID4" == "<CLIENT_ID4>" ]]; then
  echo "ERROR: One or more client placeholders are still not updated."
  echo "Please replace <CLIENT_ID1> ... <CLIENT_ID4> with real client IP addresses."
  exit 1
fi

OUTROOT="${OUTROOT:-$REPO_ROOT/outputs}"

mkdir -p \
  "$OUTROOT/SEV_A_N1CPU_300r" \
  "$OUTROOT/SEV_B_N2CPU_300r" \
  "$OUTROOT/SEV_C_N3CPU_300r" \
  "$OUTROOT/SEV_D_N4CPU_300r" \
  "$OUTROOT/SEV_E_N4CPU_300r"

cd "$REPO_ROOT"

COMMON_ARGS=(
  --csv "$REPO_ROOT/data/telemetry_merged.csv"
  --class_map "$REPO_ROOT/config/class_map.json"
  --scaler_path "$REPO_ROOT/config/scaler.json"
  --clients "http://${CLIENT_ID1}:8000,http://${CLIENT_ID2}:8000,http://${CLIENT_ID3}:8000,http://${CLIENT_ID4}:8000"
  --rounds 300
  --seeds 42,43,44,45,46
  --strategies fedavg,tl_ppo,tr_dp_afrl
  --K 3
  --min_participants 2
  --impair_start_round 100
  --impair_end_round 200
  --cpu_impair_url "http://${CLIENT_ID1}:8000"
  --net_impair_url "http://${CLIENT_ID4}:8000"
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

nohup python3 -u src/afrl_idea1_experiment_all.py \
  "${COMMON_ARGS[@]}" \
  --outdir "$OUTROOT/SEV_A_N1CPU_300r" \
  --variant_label SEV_A_N1CPU \
  --net_delay_ms 90 \
  --net_loss_pct 1 \
  --cpu_workers 1 \
  --http_timeout 120 \
  > "$OUTROOT/SEV_A_N1CPU_300r/run_SEV_A_N1CPU_300r.log" 2>&1 &

nohup python3 -u src/afrl_idea1_experiment_all.py \
  "${COMMON_ARGS[@]}" \
  --outdir "$OUTROOT/SEV_B_N2CPU_300r" \
  --variant_label SEV_B_N2CPU \
  --net_delay_ms 90 \
  --net_loss_pct 3 \
  --cpu_workers 2 \
  --http_timeout 120 \
  > "$OUTROOT/SEV_B_N2CPU_300r/run_SEV_B_N2CPU_300r.log" 2>&1 &

nohup python3 -u src/afrl_idea1_experiment_all.py \
  "${COMMON_ARGS[@]}" \
  --outdir "$OUTROOT/SEV_C_N3CPU_300r" \
  --variant_label SEV_C_N3CPU \
  --net_delay_ms 180 \
  --net_loss_pct 5 \
  --cpu_workers 3 \
  --http_timeout 180 \
  > "$OUTROOT/SEV_C_N3CPU_300r/run_SEV_C_N3CPU_300r.log" 2>&1 &

nohup python3 -u src/afrl_idea1_experiment_all.py \
  "${COMMON_ARGS[@]}" \
  --outdir "$OUTROOT/SEV_D_N4CPU_300r" \
  --variant_label SEV_D_N4CPU \
  --net_delay_ms 180 \
  --net_loss_pct 8 \
  --cpu_workers 4 \
  --http_timeout 180 \
  > "$OUTROOT/SEV_D_N4CPU_300r/run_SEV_D_N4CPU_300r.log" 2>&1 &

nohup python3 -u src/afrl_idea1_experiment_all.py \
  "${COMMON_ARGS[@]}" \
  --outdir "$OUTROOT/SEV_E_N4CPU_300r" \
  --variant_label SEV_E_N4CPU \
  --net_delay_ms 240 \
  --net_loss_pct 12 \
  --cpu_workers 4 \
  --http_timeout 240 \
  > "$OUTROOT/SEV_E_N4CPU_300r/run_SEV_E_N4CPU_300r.log" 2>&1 &

echo "Robustness jobs started."
echo "Update placeholders first, then monitor logs under: $OUTROOT"
