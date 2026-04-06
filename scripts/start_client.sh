#!/usr/bin/env bash
set -euo pipefail

CLIENT_NUM="${1:-1}"
PORT="${2:-8000}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"

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

CLIENT_ID="client${CLIENT_NUM}"
CSV_PATH="${CSV_PATH:-$REPO_ROOT/data/raw/telemetry_client${CLIENT_NUM}.csv}"
LOGFILE="${LOGFILE:-$REPO_ROOT/outputs/client${CLIENT_NUM}_api.log}"

mkdir -p "$(dirname "$LOGFILE")"
cd "$REPO_ROOT"

nohup python3 -u src/live_client_api_idea1.py \
  --client_id "$CLIENT_ID" \
  --csv "$CSV_PATH" \
  --port "$PORT" \
  --win 10 \
  --class_map "$REPO_ROOT/config/class_map.json" \
  --scaler_path "$REPO_ROOT/config/scaler.json" \
  > "$LOGFILE" 2>&1 &

echo "Started $CLIENT_ID on port $PORT"
echo "Log file: $LOGFILE"
