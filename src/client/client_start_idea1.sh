#!/usr/bin/env bash
set -euo pipefail

# ---------------- USER SETTINGS (edit per client) ----------------
CLIENT_ID="${CLIENT_ID:-client1}"            # export CLIENT_ID=client4 etc.
PORT="${PORT:-8000}"
CSV="${CSV:-telemetry_${CLIENT_ID}.csv}"     # your run_scenarios.sh outputs telemetry_clientX.csv by default
# If your CSV is telemetry_client1.csv etc, set CSV like:
# CSV="telemetry_${CLIENT_ID}.csv"  (recommended) OR CSV="telemetry_client1.csv" (your older naming)

FEATURE_COLS_JSON="${FEATURE_COLS_JSON:-/home/ubuntu/feature_cols.json}"
CLASS_MAP_JSON="${CLASS_MAP_JSON:-/home/ubuntu/class_map.json}"
SCALER_JSON="${SCALER_JSON:-/home/ubuntu/scaler.json}"

API_PY="${API_PY:-/home/ubuntu/telemetry_client/live_client_api_idea1.py}"
LOG="${LOG:-/home/ubuntu/telemetry_client/api_${CLIENT_ID}.log}"
PIDFILE="${PIDFILE:-/home/ubuntu/telemetry_client/api_${CLIENT_ID}.pid}"

# ---------------- INTERNALS ----------------
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "[INFO] Starting Idea-1 client API"
echo "       CLIENT_ID : $CLIENT_ID"
echo "       PORT      : $PORT"
echo "       CSV       : $CSV"
echo "       LOG       : $LOG"
echo "       API_PY    : $API_PY"

# Activate venv if exists (optional)
if [[ -f "$HERE/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HERE/venv/bin/activate"
  echo "[OK] Activated venv: $HERE/venv"
else
  echo "[WARN] venv not found at $HERE/venv. Using system python."
fi

# --- sanity: python file exists ---
if [[ ! -f "$API_PY" ]]; then
  echo "[ERROR] Missing $API_PY"
  echo "       Put live_client_api_idea1.py at: $API_PY"
  exit 1
fi

# --- sanity: CSV exists ---
if [[ ! -f "$CSV" ]]; then
  echo "[ERROR] Missing telemetry CSV: $CSV"
  echo "       Run telemetry first, e.g.:"
  echo "       CLIENT_ID=$CLIENT_ID ./run_scenarios.sh"
  exit 1
fi

# --- stop any previous instance ---
echo "[INFO] Stopping old API (if running)..."
if [[ -f "$PIDFILE" ]]; then
  OLD_PID="$(cat "$PIDFILE" || true)"
  if [[ -n "${OLD_PID:-}" ]] && ps -p "$OLD_PID" >/dev/null 2>&1; then
    kill "$OLD_PID" 2>/dev/null || true
    sleep 0.5
  fi
  rm -f "$PIDFILE"
fi

# also kill any uvicorn/python that matches this script (best-effort)
pkill -f "live_client_api_idea1.py" 2>/dev/null || true
sleep 0.5

# --- start ---
echo "[INFO] Launching API with nohup..."
nohup python3 -u "$API_PY" \
  --client_id "$CLIENT_ID" \
  --csv "$CSV" \
  --feature_cols_json "$FEATURE_COLS_JSON" \
  --class_map_json "$CLASS_MAP_JSON" \
  --scaler_json "$SCALER_JSON" \
  --port "$PORT" \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"
sleep 1

# --- verify port is listening ---
if ss -ltnp | grep -q ":${PORT}"; then
  echo "[OK] Port $PORT is listening (PID=$PID)"
else
  echo "[ERROR] Port $PORT not listening. Log tail:"
  tail -n 60 "$LOG" || true
  exit 1
fi

# --- verify endpoints ---
echo "[INFO] Checking /health..."
if ! curl -s --max-time 3 "http://127.0.0.1:${PORT}/health" | head -c 200; then
  echo
  echo "[ERROR] /health failed. Log tail:"
  tail -n 60 "$LOG" || true
  exit 1
fi
echo
echo "[INFO] Checking /telemetry_round..."
if ! curl -s --max-time 3 "http://127.0.0.1:${PORT}/telemetry_round?round_id=1&win=10" | head -c 250; then
  echo
  echo "[ERROR] /telemetry_round failed. Log tail:"
  tail -n 60 "$LOG" || true
  exit 1
fi
echo
echo "[DONE] Idea-1 client API READY ✅"
echo "       PID: $(cat "$PIDFILE")"
