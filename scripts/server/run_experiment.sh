#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER="$ROOT/src/server"

usage() {
    echo "Usage: bash run_experiment.sh EXPERIMENT STRATEGY [additional Python arguments]"
    echo "Experiments: core, core_matched, e1, e2, e3, e4, e5, e6a, e6b, e7"
    echo "Example: bash run_experiment.sh core tr_dp_afrl"
    echo "E6-B example: bash run_experiment.sh e6b controller"
}

if (( $# < 2 )); then
    usage
    exit 2
fi

EXPERIMENT="$1"
STRATEGY="$2"
shift 2

case "$EXPERIMENT" in
    core)         RUNNER="afrl_core_run_one_strategy.py" ;;
    core_matched) RUNNER="afrl_core_run_one_strategy_matched.py" ;;
    e1)           RUNNER="afrl_e1_run_one_strategy.py" ;;
    e2)           RUNNER="afrl_e2_run_one_strategy.py" ;;
    e3)           RUNNER="afrl_e3_run_one_strategy.py" ;;
    e4)           RUNNER="afrl_e4_run_one_strategy.py" ;;
    e5)           RUNNER="afrl_e5_run_one_strategy.py" ;;
    e6a)          RUNNER="afrl_e6a_run_one_strategy.py" ;;
    e6b)          RUNNER="afrl_e6b_controller_scalability.py" ;;
    e7)           RUNNER="afrl_e7_run_one_strategy.py" ;;
    *) echo "ERROR: Unknown experiment: $EXPERIMENT" >&2; exit 2 ;;
esac

[[ -f "$SERVER/$RUNNER" ]] || {
    echo "ERROR: Missing runner: $SERVER/$RUNNER" >&2
    exit 1
}

source "$ROOT/scripts/lib/experiment_config.sh"

if [[ "$EXPERIMENT" == "e6b" ]]; then
    [[ "$STRATEGY" == "controller" ]] || {
        echo "ERROR: E6-B requires strategy 'controller'." >&2
        exit 2
    }
    fgcs_load_e6b_config
else
    fgcs_load_experiment_config

    source "$ROOT/scripts/lib/client_config.sh"

    COUNT="${FGCS_NUM_CLIENTS:-${NUM_CLIENTS:-10}}"

    if [[ "$EXPERIMENT" == "e7" && "$COUNT" != "10" ]]; then
        echo "ERROR: E7 requires exactly 10 physical clients." >&2
        exit 2
    fi

    fgcs_load_clients "$COUNT"
fi

# Refuse arguments that could redirect outputs into frozen results
# or override centrally validated data/client paths.
for arg in "$@"; do
    case "$arg" in
        --outdir|--outdir=*|--out|--out=*|\
        --csv|--csv=*|--clients|--clients=*|\
        --client_ids|--client_ids=*|\
        --class_map|--class_map=*|\
        --scaler_path|--scaler_path=*)
            echo "ERROR: Restricted argument: $arg" >&2
            exit 2
            ;;
    esac
done

RUN_BASE="${FGCS_RUN_BASE:-$ROOT/outputs/manual_runs}"
mkdir -p "$RUN_BASE"

RUN_DIR="$(mktemp -d "$RUN_BASE/${EXPERIMENT}_${STRATEGY}_XXXXXXXX")"
LOG="$RUN_DIR/nohup.log"
PID_FILE="$RUN_DIR/pid"

if [[ "$EXPERIMENT" == "e6b" ]]; then
    read -r -a SIZES <<< "${E6B_SIZES:-10 20 50 100}"

    COMMAND=(
        "$PYTHON_BIN" "$SERVER/$RUNNER"
        --sizes "${SIZES[@]}"
        --warmup "${E6B_WARMUP:-100}"
        --repetitions "${E6B_REPETITIONS:-1000}"
        --seed "${SEED:-42}"
        --out "$RUN_DIR/controller_results.json"
    )
else
    COMMAND=(
        "$PYTHON_BIN" "$SERVER/$RUNNER"
        --strategy "$STRATEGY"
        --csv "$DATASET_CSV"
        --clients "$CLIENT_URLS"
        --client_ids "$CLIENT_IDS"
        --rounds "$ROUNDS"
        --seed "$SEED"
        --K "$K"
        --min_participants "$MIN_PARTICIPANTS"
        --win "$WIN"
        --local_epochs "$LOCAL_EPOCHS"
        --lr_local "$LR_LOCAL"
        --batch_size "$BATCH_SIZE"
        --alpha_acc "$ALPHA_ACC"
        --beta_lat "$BETA_LAT"
        --gamma_off "$GAMMA_OFF"
        --lambda_stab "$LAMBDA_STAB"
        --trdp_eta "$TRDP_ETA"
        --trdp_wmin "$TRDP_WMIN"
        --trdp_wmax "$TRDP_WMAX"
        --trdp_starve_pen "$TRDP_STARVE_PEN"
        --class_map "$CLASS_MAP_PATH"
        --scaler_path "$SCALER_PATH"
        --outdir "$RUN_DIR"
    )

    if [[ "$EXPERIMENT" != "core" && "$EXPERIMENT" != "core_matched" ]]; then
        COMMAND+=(
            --num_clients "$COUNT"
            --dirichlet_alpha "$DIRICHLET_ALPHA"
            --max_parallel_clients "$MAX_PARALLEL_CLIENTS"
        )
    fi

    if [[ "$EXPERIMENT" == "e7" ]]; then
        COMMAND+=(
            --impair_condition "$IMPAIR_CONDITION"
            --impair_start_round "$IMPAIR_START_ROUND"
            --impair_stop_round "$IMPAIR_STOP_ROUND"
            --net_delay_ms "$NET_DELAY_MS"
            --net_loss_pct "$NET_LOSS_PCT"
        )
    fi
fi

COMMAND+=("$@")

# Record exact invocation in the new run directory.
printf '%q ' "${COMMAND[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"

# Starting an experiment requires explicit user confirmation.
if [[ "${FGCS_CONFIRM_RUN:-}" != "YES" ]]; then
    echo "DRY RUN — experiment not started."
    echo "Prepared command: $RUN_DIR/command.sh"
    echo "To launch intentionally, set FGCS_CONFIRM_RUN=YES."
    exit 0
fi

nohup "${COMMAND[@]}" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"

echo "Experiment: $EXPERIMENT"
echo "Strategy: $STRATEGY"
echo "PID: $PID"
echo "Output directory: $RUN_DIR"
echo "Log: $LOG"
