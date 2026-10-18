#!/usr/bin/env bash
# FGCS TR-DP-AFRL shared experiment configuration.
# Source this helper, then call fgcs_load_experiment_config.

FGCS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FGCS_EXPERIMENT_CONFIG="${FGCS_EXPERIMENT_CONFIG:-$FGCS_ROOT/config/experiments.env}"

fgcs_load_experiment_config() {
    if [[ ! -f "$FGCS_EXPERIMENT_CONFIG" ]]; then
        echo "ERROR: Missing $FGCS_EXPERIMENT_CONFIG" >&2
        echo "Copy config/experiments.env.example to config/experiments.env." >&2
        return 1
    fi

    # shellcheck disable=SC1090
    source "$FGCS_EXPERIMENT_CONFIG" || return 1

    local variable value

    for variable in ROUNDS SEED K MIN_PARTICIPANTS WIN \
                    LOCAL_EPOCHS BATCH_SIZE; do
        value="${!variable:-}"

        if ! [[ "$value" =~ ^[0-9]+$ ]]; then
            echo "ERROR: $variable must be a non-negative integer." >&2
            return 1
        fi
    done

    if (( ROUNDS < 1 || K < 1 || MIN_PARTICIPANTS < 1 ||
          WIN < 1 || LOCAL_EPOCHS < 1 || BATCH_SIZE < 1 )); then
        echo "ERROR: Invalid experiment parameter range." >&2
        return 1
    fi

    if (( MIN_PARTICIPANTS > K )); then
        echo "ERROR: MIN_PARTICIPANTS cannot exceed K." >&2
        return 1
    fi

    if [[ -z "${DATASET_CSV:-}" || ! -f "$DATASET_CSV" ]]; then
        echo "ERROR: Configure DATASET_CSV with an existing CSV file." >&2
        return 1
    fi

    if [[ -z "${CLASS_MAP_PATH:-}" || ! -f "$CLASS_MAP_PATH" ]]; then
        echo "ERROR: Configure CLASS_MAP_PATH with an existing JSON file." >&2
        return 1
    fi

    if [[ -z "${SCALER_PATH:-}" || ! -f "$SCALER_PATH" ]]; then
        echo "ERROR: Configure SCALER_PATH with an existing JSON file." >&2
        return 1
    fi

    PYTHON_BIN="${PYTHON_BIN:-python3}"

    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "ERROR: Python interpreter not found: $PYTHON_BIN" >&2
        return 1
    fi

    export ROUNDS SEED K MIN_PARTICIPANTS WIN
    export LOCAL_EPOCHS LR_LOCAL BATCH_SIZE
    export ALPHA_ACC BETA_LAT GAMMA_OFF LAMBDA_STAB
    export TRDP_ETA TRDP_WMIN TRDP_WMAX TRDP_STARVE_PEN
    export DIRICHLET_ALPHA MAX_PARALLEL_CLIENTS
    export IMPAIR_CONDITION IMPAIR_START_ROUND IMPAIR_STOP_ROUND
    export NET_DELAY_MS NET_LOSS_PCT
    export E6B_SIZES E6B_WARMUP E6B_REPETITIONS
    export DATASET_CSV CLASS_MAP_PATH SCALER_PATH PYTHON_BIN

    echo "Experiment configuration loaded successfully."
    echo "Rounds: $ROUNDS | Seed: $SEED | K: $K"
    echo "Stability weight: ${LAMBDA_STAB:-unset}"
}

fgcs_load_e6b_config() {
    if [[ ! -f "$FGCS_EXPERIMENT_CONFIG" ]]; then
        echo "ERROR: Missing $FGCS_EXPERIMENT_CONFIG" >&2
        return 1
    fi

    # E6-B does not require a dataset or physical clients.
    # shellcheck disable=SC1090
    source "$FGCS_EXPERIMENT_CONFIG" || return 1

    PYTHON_BIN="${PYTHON_BIN:-python3}"

    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "ERROR: Python interpreter not found: $PYTHON_BIN" >&2
        return 1
    fi

    export PYTHON_BIN E6B_SIZES E6B_WARMUP E6B_REPETITIONS SEED

    echo "E6-B controller configuration loaded."
}
