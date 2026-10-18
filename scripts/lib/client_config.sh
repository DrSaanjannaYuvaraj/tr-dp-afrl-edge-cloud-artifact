#!/usr/bin/env bash
# FGCS TR-DP-AFRL shared client configuration
# Source this helper from server experiment launchers.

FGCS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FGCS_CLIENT_CONFIG="${FGCS_CLIENT_CONFIG:-$FGCS_ROOT/config/clients.env}"

fgcs_config_error() {
    echo "ERROR: $*" >&2
    return 1
}

fgcs_load_clients() {
    if [[ ! -f "$FGCS_CLIENT_CONFIG" ]]; then
        fgcs_config_error "Missing $FGCS_CLIENT_CONFIG" || return 1
        echo "Copy config/clients.env.example to config/clients.env and update IPs." >&2
        return 1
    fi

    # shellcheck disable=SC1090
    source "$FGCS_CLIENT_CONFIG" || return 1

    local count="${1:-${NUM_CLIENTS:-10}}"
    local port="${CLIENT_PORT:-8000}"
    local i variable address

    if ! [[ "$count" =~ ^[0-9]+$ ]] ||
       (( count < 1 || count > 10 )); then
        fgcs_config_error "Client count must be between 1 and 10." || return 1
    fi

    if ! [[ "$port" =~ ^[0-9]+$ ]] ||
       (( port < 1 || port > 65535 )); then
        fgcs_config_error "Invalid CLIENT_PORT." || return 1
    fi

    FGCS_CLIENT_URL_ARRAY=()

    for ((i=1; i<=count; i++)); do
        variable="CLIENT${i}_IP"
        address="${!variable:-}"

        if [[ -z "$address" ]]; then
            fgcs_config_error "$variable is missing." || return 1
        fi

        case "$address" in
            192.0.2.*|198.51.100.*|203.0.113.*|YOUR_*)
                fgcs_config_error "$variable contains a placeholder address." || return 1
                ;;
        esac

        FGCS_CLIENT_URL_ARRAY+=("http://${address}:${port}")
    done

    CLIENT_URLS="$(IFS=,; echo "${FGCS_CLIENT_URL_ARRAY[*]}")"

    FGCS_CLIENT_ID_ARRAY=()
    for ((i=1; i<=count; i++)); do
        FGCS_CLIENT_ID_ARRAY+=("client${i}")
    done

    CLIENT_IDS="$(IFS=,; echo "${FGCS_CLIENT_ID_ARRAY[*]}")"
    NUM_CLIENTS="$count"
    CLIENT_PORT="$port"

    export CLIENT_URLS CLIENT_IDS NUM_CLIENTS CLIENT_PORT

    echo "Client configuration loaded: ${count} clients."
}
