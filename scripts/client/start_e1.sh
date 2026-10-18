#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${FGCS_CLIENT_RUNTIME_CONFIG:-$HERE/client_runtime.env}"

[[ -f "$CONFIG" ]] || {
    echo "ERROR: Missing client runtime configuration: $CONFIG" >&2
    exit 1
}

source "$CONFIG"

if [[ "${EXPERIMENT:-}" != "e1" ]]; then
    echo "ERROR: Set EXPERIMENT=e1 in $CONFIG" >&2
    exit 1
fi

exec bash "$HERE/start_client.sh"
