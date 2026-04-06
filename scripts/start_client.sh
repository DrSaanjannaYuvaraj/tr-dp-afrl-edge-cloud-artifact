#!/usr/bin/env bash
set -euo pipefail

CLIENT_ID="${1:-client1}"
CSV_PATH="${2:-data/telemetry_client1.csv}"
PORT="${3:-8000}"

python3 src/live_client_api_idea1.py   --client_id "$CLIENT_ID"   --csv "$CSV_PATH"   --port "$PORT"   --win 10   --class_map config/class_map.json   --scaler_path config/scaler.json
