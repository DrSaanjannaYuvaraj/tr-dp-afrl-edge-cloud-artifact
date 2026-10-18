#!/usr/bin/env bash
set -euo pipefail

PIDFILE="/home/ubuntu/telemetry_client/stress_cpu.pid"
LOGFILE="/home/ubuntu/telemetry_client/stress_cpu.log"

usage() {
  echo "Usage: $0 start <seconds> | stop | status"
}

start_cpu() {
  local dur="${1:-600}"
  echo "[INFO] Starting CPU stress for ${dur}s" | tee -a "$LOGFILE"

  # stop any previous
  "$0" stop >/dev/null 2>&1 || true

  # start stress-ng
  nohup stress-ng --cpu 2 --cpu-method matrixprod --timeout "${dur}s" \
    >> "$LOGFILE" 2>&1 &

  echo $! > "$PIDFILE"
  echo "[OK] CPU stress started PID=$(cat "$PIDFILE")" | tee -a "$LOGFILE"
}

stop_cpu() {
  echo "[INFO] Stopping CPU stress" | tee -a "$LOGFILE"
  if [[ -f "$PIDFILE" ]]; then
    PID="$(cat "$PIDFILE" || true)"
    if [[ -n "${PID:-}" ]] && ps -p "$PID" >/dev/null 2>&1; then
      kill "$PID" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
  fi
  # best-effort cleanup
  pkill -f "stress-ng --cpu" 2>/dev/null || true
  echo "[OK] CPU stress stopped" | tee -a "$LOGFILE"
}

status_cpu() {
  if [[ -f "$PIDFILE" ]]; then
    PID="$(cat "$PIDFILE" || true)"
    if [[ -n "${PID:-}" ]] && ps -p "$PID" >/dev/null 2>&1; then
      echo "RUNNING PID=$PID"
      exit 0
    fi
  fi
  echo "NOT_RUNNING"
}

cmd="${1:-}"
case "$cmd" in
  start) start_cpu "${2:-600}" ;;
  stop)  stop_cpu ;;
  status) status_cpu ;;
  *) usage; exit 1 ;;
esac

