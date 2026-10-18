#!/usr/bin/env bash
set -euo pipefail

CLIENT_ID="${CLIENT_ID:-client1}"
PORT="${PORT:-8000}"
PIDFILE="${PIDFILE:-/home/ubuntu/telemetry_client/api_${CLIENT_ID}.pid}"

echo "[INFO] Stopping Idea-1 client API"
echo "       CLIENT_ID : $CLIENT_ID"
echo "       PORT      : $PORT"

# --- stop via PID file ---
if [[ -f "$PIDFILE" ]]; then
  PID="$(cat "$PIDFILE" || true)"
  if [[ -n "${PID:-}" ]] && ps -p "$PID" >/dev/null 2>&1; then
    echo "[INFO] Killing PID=$PID"
    kill "$PID" 2>/dev/null || true
    sleep 0.5
  else
    echo "[WARN] PID file exists but process not running"
  fi
  rm -f "$PIDFILE"
else
  echo "[WARN] No PID file found"
fi

# --- best-effort cleanup ---
pkill -f "live_client_api_idea1.py" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
sleep 0.5

# --- verify port is free ---
if ss -ltnp | grep -q ":${PORT}"; then
  echo "[WARN] Port ${PORT} still in use. Forcing kill..."
  sudo fuser -k ${PORT}/tcp 2>/dev/null || true
else
  echo "[OK] Port ${PORT} is free"
fi

echo "[DONE] Idea-1 client API stopped ✅"
