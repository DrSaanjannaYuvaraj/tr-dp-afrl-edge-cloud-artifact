#!/usr/bin/env bash
set -euo pipefail

# ====== YOUR SETUP ======
SERVER_IP="${SERVER_IP:?Set SERVER_IP to your server address}"
CLIENT_ID="${CLIENT_ID:-client1}"

# Auto-detect default interface
IFACE="$(ip route | awk '/default/ {print $5; exit}')"
if [[ -z "${IFACE}" ]]; then
  echo "[ERROR] Could not detect default network interface."
  exit 1
fi

# ====== DATA COLLECTION CONFIG (13k total) ======
INTERVAL=3
NORMAL=4000
HIGHCPU=3000
DISKLOAD=3000
NETFLUCT=3000
OUT="telemetry_${CLIENT_ID}.csv"

# HTTP traffic target (server must run http.server 9999)
HTTP_PORT=9999
HTTP_URL="http://${SERVER_IP}:${HTTP_PORT}/"

# PID files (avoid /tmp issues)
PID_STRESS="/home/ubuntu/stressng_${CLIENT_ID}.pid"
PID_DISK="/home/ubuntu/diskload_${CLIENT_ID}.pid"
PID_NET="/home/ubuntu/netload_${CLIENT_ID}.pid"

cd "$(dirname "$0")"
source venv/bin/activate

echo "[INFO] Client: $CLIENT_ID"
echo "[INFO] Server: $SERVER_IP"
echo "[INFO] Interface: $IFACE"
echo "[INFO] Output: $OUT"
echo "[INFO] Plan: normal=$NORMAL, high_cpu=$HIGHCPU, disk_load=$DISKLOAD, net_fluct=$NETFLUCT (interval=${INTERVAL}s)"
echo "[INFO] Total samples: $((NORMAL + HIGHCPU + DISKLOAD + NETFLUCT))"

cleanup_netem() {
  sudo -n tc qdisc del dev "$IFACE" root 2>/dev/null || true
}

start_highcpu() {
  stress-ng --cpu 2 --cpu-method matrixprod --timeout 999999s >/dev/null 2>&1 &
  echo $! > "$PID_STRESS"
  echo "[INFO] stress-ng PID=$(cat "$PID_STRESS")"
}

stop_highcpu() {
  if [[ -f "$PID_STRESS" ]]; then
    kill "$(cat "$PID_STRESS")" 2>/dev/null || true
    rm -f "$PID_STRESS"
  fi
  pkill -f "stress-ng --cpu" 2>/dev/null || true
}

start_diskload() {
  # Disk load runs until killed (so it lasts for full DISKLOAD window)
  nohup bash -c '
    while true; do
      dd if=/dev/zero of=/tmp/io_test.bin bs=4M count=16 oflag=direct conv=fsync status=none
      rm -f /tmp/io_test.bin
      sleep 0.2
    done
  ' > disk_load.log 2>&1 &
  echo $! > "$PID_DISK"
  echo "[INFO] disk load PID=$(cat "$PID_DISK")"
}

stop_diskload() {
  if [[ -f "$PID_DISK" ]]; then
    kill "$(cat "$PID_DISK")" 2>/dev/null || true
    rm -f "$PID_DISK"
  fi
  pkill -f "dd if=/dev/zero of=/tmp/io_test.bin" 2>/dev/null || true
  rm -f /tmp/io_test.bin 2>/dev/null || true
}

apply_net_fluct() {
  cleanup_netem
  sudo -n tc qdisc add dev "$IFACE" root handle 1: netem delay 60ms 20ms distribution normal loss 2%
  sudo -n tc qdisc add dev "$IFACE" parent 1: handle 2: tbf rate 10mbit burst 32kbit latency 400ms
}

start_netload() {
  # Timeouts prevent hangs under loss/jitter
  nohup bash -c "
    while true; do
      curl -s --connect-timeout 1 --max-time 2 '${HTTP_URL}' >/dev/null || true
      sleep 0.5
    done
  " > net_load.log 2>&1 &
  echo $! > "$PID_NET"
  echo "[INFO] net load PID=$(cat "$PID_NET")"
}

stop_netload() {
  if [[ -f "$PID_NET" ]]; then
    kill "$(cat "$PID_NET")" 2>/dev/null || true
    rm -f "$PID_NET"
  fi
  pkill -f "curl .*${SERVER_IP}:${HTTP_PORT}" 2>/dev/null || true
}

cleanup_all() {
  echo "[CLEANUP] stopping loads + netem"
  stop_netload || true
  stop_diskload || true
  stop_highcpu || true
  cleanup_netem || true
}
trap cleanup_all INT TERM EXIT

# Fresh output each run
rm -f "$OUT"

echo "[STEP 1/4] normal"
python3 telemetry_client.py \
  --client_id "$CLIENT_ID" --server_ip "$SERVER_IP" --scenario normal \
  --interval "$INTERVAL" --samples "$NORMAL" --out "$OUT"

echo "[STEP 2/4] high_cpu (stress-ng only)"
start_highcpu
python3 telemetry_client.py \
  --client_id "$CLIENT_ID" --server_ip "$SERVER_IP" --scenario high_cpu \
  --interval "$INTERVAL" --samples "$HIGHCPU" --out "$OUT"
stop_highcpu

echo "[STEP 3/4] disk_load (dd only)"
start_diskload
python3 telemetry_client.py \
  --client_id "$CLIENT_ID" --server_ip "$SERVER_IP" --scenario disk_load \
  --interval "$INTERVAL" --samples "$DISKLOAD" --out "$OUT"
stop_diskload

echo "[STEP 4/4] net_fluct (tc netem + traffic)"
apply_net_fluct
start_netload
python3 telemetry_client.py \
  --client_id "$CLIENT_ID" --server_ip "$SERVER_IP" --scenario net_fluct \
  --interval "$INTERVAL" --samples "$NETFLUCT" --out "$OUT"
stop_netload
cleanup_netem

trap - INT TERM EXIT

echo "[DONE] Completed."
echo "[DONE] Row count (including header):"
wc -l "$OUT"
echo "[DONE] Scenario counts:"
cut -d, -f5 "$OUT" | tail -n +2 | sort | uniq -c
