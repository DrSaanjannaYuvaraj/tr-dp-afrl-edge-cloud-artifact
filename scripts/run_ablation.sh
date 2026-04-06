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

OUTDIR="${OUTDIR:-$REPO_ROOT/outputs/N10_300r_main}"
LOGFILE="$OUTDIR/run_N10_300r_ablation.log"

echo "Repository root: $REPO_ROOT"
echo "Output directory: $OUTDIR"
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

CLIENTS="http://${CLIENT_ID1}:8000,http://${CLIENT_ID2}:8000,http://${CLIENT_ID3}:8000,http://${CLIENT_ID4}:8000,http://${CLIENT_ID5}:8000,http://${CLIENT_ID6}:8000,http://${CLIENT_ID7}:8000,http://${CLIENT_ID8}:8000,http://${CLIENT_ID9}:8000,http://${CLIENT_ID10}:8000"

if [[ "$CLIENTS" == *"<CLIENT_ID"* ]]; then
  echo "ERROR: One or more client placeholders are still not updated."
  echo "Please replace <CLIENT_ID1> ... <CLIENT_ID10> with real client IP addresses."
  exit 1
fi

mkdir -p "$OUTDIR"
cd "$REPO_ROOT"

nohup python3 -u src/afrl_idea1_experiment_all.py \
  --csv "$REPO_ROOT/data/telemetry_merged.csv" \
  --class_map "$REPO_ROOT/config/class_map.json" \
  --scaler_path "$REPO_ROOT/config/scaler.json" \
  --clients "$CLIENTS" \
  --rounds 300 \
  --seeds 46 \
  --run_ablation \
  --strategies tl_ppo,tr_dp_afrl \
  --K 8 \
  --min_participants 5 \
  --outdir "$OUTDIR" \
  --variant_label N10 \
  --run_ablation \
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
  > "$LOGFILE" 2>&1 &

echo "Started ablation run."
echo "Monitor with:"
echo "  tail -f $LOGFILE"
echo '  pgrep -af afrl_idea1_experiment_all.py || echo "RUN COMPLETED"'
