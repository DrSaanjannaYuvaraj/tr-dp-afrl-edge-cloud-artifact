#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${FGCS_CLIENT_RUNTIME_CONFIG:-$HERE/client_runtime.env}"

[[ -f "$CONFIG" ]] || {
    echo "ERROR: Missing $CONFIG" >&2
    exit 1
}

source "$CONFIG"

: "${CLIENT_ID:?CLIENT_ID is required}"
: "${EXPERIMENT:?EXPERIMENT is required}"

PORT="${CLIENT_PORT:-8000}"
RUNTIME="${FGCS_CLIENT_RUN_DIR:-$HERE/runtime}"
PIDFILE="$RUNTIME/${CLIENT_ID}_${EXPERIMENT}_${PORT}.pid"

[[ -f "$PIDFILE" ]] || {
    echo "No managed PID file found: $PIDFILE"
    exit 0
}

PID="$(cat "$PIDFILE")"

[[ "$PID" =~ ^[0-9]+$ ]] || {
    echo "ERROR: Invalid PID file. No process stopped." >&2
    exit 1
}

if ! kill -0 "$PID" 2>/dev/null; then
    echo "Process PID=$PID is no longer running."
    rm -f "$PIDFILE"
    exit 0
fi

# Support both the original AWS layout and the published repository.
if [[ -n "${FGCS_CLIENT_SOURCE_ROOT:-}" ]]; then
    SOURCE_ROOT="$FGCS_CLIENT_SOURCE_ROOT"
elif [[ -d "$HERE/../../src/client" ]]; then
    SOURCE_ROOT="$(cd "$HERE/../../src/client" && pwd)"
else
    SOURCE_ROOT="$(cd "$HERE/.." && pwd)"
fi

case "$EXPERIMENT" in
    core|core_matched) API="live_client_api_core.py" ;;
    e1|e2|e3|e4|e6a|e7) API="live_client_api_${EXPERIMENT}.py" ;;
    e5) API="live_client_api_e1.py" ;;
    *) echo "ERROR: Unsupported experiment." >&2; exit 1 ;;
esac

CMDLINE="$(ps -p "$PID" -o args= || true)"

if [[ "$CMDLINE" != *"$SOURCE_ROOT/$API"* ]] ||
   [[ "$CMDLINE" != *"--client_id $CLIENT_ID"* ]]; then
    echo "ERROR: PID does not match the expected client API." >&2
    echo "No process stopped." >&2
    exit 1
fi

kill "$PID"
echo "Stop signal sent to managed client PID=$PID"
rm -f "$PIDFILE"
