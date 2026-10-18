#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Support both the original AWS layout and the published repository.
if [[ -n "${FGCS_CLIENT_SOURCE_ROOT:-}" ]]; then
    SOURCE_ROOT="$FGCS_CLIENT_SOURCE_ROOT"
elif [[ -d "$HERE/../../src/client" ]]; then
    SOURCE_ROOT="$(cd "$HERE/../../src/client" && pwd)"
else
    SOURCE_ROOT="$(cd "$HERE/.." && pwd)"
fi
CONFIG="${FGCS_CLIENT_RUNTIME_CONFIG:-$HERE/client_runtime.env}"

[[ -f "$CONFIG" ]] || {
    echo "ERROR: Missing $CONFIG" >&2
    echo "Copy client_runtime.env.example to client_runtime.env." >&2
    exit 1
}

source "$CONFIG"

: "${CLIENT_ID:?CLIENT_ID is required}"
: "${EXPERIMENT:?EXPERIMENT is required}"
: "${CLIENT_CSV:?CLIENT_CSV is required}"
: "${CLASS_MAP_PATH:?CLASS_MAP_PATH is required}"
: "${SCALER_PATH:?SCALER_PATH is required}"

[[ "$CLIENT_ID" =~ ^client([1-9]|10)$ ]] || {
    echo "ERROR: CLIENT_ID must be client1 through client10." >&2
    exit 1
}

case "$EXPERIMENT" in
    core|core_matched)
        API="live_client_api_core.py"
        NEED_INDEX=0
        ;;
    e1|e2|e3|e4|e6a|e7)
        API="live_client_api_${EXPERIMENT}.py"
        NEED_INDEX=1
        ;;
    e5)
        API="live_client_api_e1.py"
        NEED_INDEX=1
        ;;
    *)
        echo "ERROR: Unsupported experiment: $EXPERIMENT" >&2
        exit 1
        ;;
esac

CLIENT_INDEX=$(( ${CLIENT_ID#client} - 1 ))
PORT="${CLIENT_PORT:-8000}"
HOST="${CLIENT_HOST:-0.0.0.0}"
WIN="${CLIENT_WIN:-10}"
PYTHON="${PYTHON_BIN:-python3}"

[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || {
    echo "ERROR: Invalid port." >&2
    exit 1
}

for file in "$SOURCE_ROOT/$API" "$CLIENT_CSV" "$CLASS_MAP_PATH" "$SCALER_PATH"; do
    [[ -f "$file" ]] || {
        echo "ERROR: Missing file: $file" >&2
        exit 1
    }
done

command -v "$PYTHON" >/dev/null 2>&1 || {
    echo "ERROR: Python interpreter not found: $PYTHON" >&2
    exit 1
}

RUNTIME="${FGCS_CLIENT_RUN_DIR:-$HERE/runtime}"
mkdir -p "$RUNTIME"

KEY="${CLIENT_ID}_${EXPERIMENT}_${PORT}"
PIDFILE="$RUNTIME/${KEY}.pid"
LOG="$RUNTIME/${KEY}.log"

if [[ -f "$PIDFILE" ]]; then
    OLD_PID="$(cat "$PIDFILE")"
    if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: Existing process PID=$OLD_PID; stop it first." >&2
        exit 1
    fi
    echo "ERROR: Stale PID file: $PIDFILE" >&2
    echo "Inspect and remove it manually before restarting." >&2
    exit 1
fi

COMMAND=(
    "$PYTHON" -u "$SOURCE_ROOT/$API"
    --client_id "$CLIENT_ID"
    --csv "$CLIENT_CSV"
    --win "$WIN"
    --class_map "$CLASS_MAP_PATH"
    --scaler_path "$SCALER_PATH"
    --host "$HOST"
    --port "$PORT"
)

if (( NEED_INDEX )); then
    COMMAND+=(--client_index "$CLIENT_INDEX")
fi

if [[ "${FGCS_CONFIRM_RUN:-}" != "YES" ]]; then
    echo "DRY RUN — client API not started."
    printf 'Command: '
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    echo "To launch intentionally, set FGCS_CONFIRM_RUN=YES."
    exit 0
fi

nohup "${COMMAND[@]}" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDFILE"

echo "Client: $CLIENT_ID"
echo "Experiment: $EXPERIMENT"
echo "API: $API"
echo "PID: $PID"
echo "Log: $LOG"
