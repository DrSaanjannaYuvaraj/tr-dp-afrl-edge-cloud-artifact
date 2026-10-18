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

if [[ ! -f "$PIDFILE" ]]; then
    echo "NOT MANAGED: No PID file for $CLIENT_ID / $EXPERIMENT"
    exit 0
fi

PID="$(cat "$PIDFILE")"

if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
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

    if [[ "$CMDLINE" == *"$SOURCE_ROOT/$API"* ]] &&
       [[ "$CMDLINE" == *"--client_id $CLIENT_ID"* ]]; then
        echo "RUNNING: Verified managed client PID=$PID"
        ps -p "$PID" -o pid=,stat=,args=
    else
        echo "PID MISMATCH: PID=$PID is not the expected client API."
        echo "No process modified."
        exit 1
    fi
else
    echo "NOT RUNNING: Stale PID file: $PIDFILE"
fi
