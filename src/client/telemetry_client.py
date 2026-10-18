#!/usr/bin/env python3
import argparse
import csv
import os
import socket
import time
import statistics
from datetime import datetime, timezone
from typing import Optional, Tuple

import psutil


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_primary_private_ip() -> str:
    addrs = psutil.net_if_addrs()
    for _, addr_list in addrs.items():
        for a in addr_list:
            if getattr(a, "family", None) == socket.AF_INET:
                if a.address and not a.address.startswith("127."):
                    return a.address
    return ""


def read_net_bytes():
    c = psutil.net_io_counters(pernic=False)
    return c.bytes_sent, c.bytes_recv


def read_disk_bytes():
    c = psutil.disk_io_counters()
    if not c:
        return 0, 0
    return c.read_bytes, c.write_bytes


def cpu_freq_mhz():
    f = psutil.cpu_freq()
    if not f:
        return None
    return f.current


def ping_once(target_ip: str, timeout_s: int = 1) -> Tuple[Optional[float], bool]:
    rc = os.system(f"ping -c 1 -W {timeout_s} {target_ip} >/dev/null 2>&1")
    if rc != 0:
        return None, True

    out = os.popen(f"ping -c 1 -W {timeout_s} {target_ip} 2>/dev/null").read()
    rtt = None
    for token in out.split():
        if token.startswith("time="):
            try:
                rtt = float(token.split("=")[1])
            except Exception:
                rtt = None
            break
    return rtt, False


def compute_ping_stats(target_ip: str, k: int = 3) -> Tuple[float, float, float]:
    rtts = []
    losses = 0

    for _ in range(k):
        rtt, lost = ping_once(target_ip, timeout_s=1)
        if lost or rtt is None:
            losses += 1
        else:
            rtts.append(float(rtt))

    loss_pct = (losses / k) * 100.0

    if len(rtts) == 0:
        return 999.0, 0.0, float(loss_pct)

    avg_rtt = float(sum(rtts) / len(rtts))
    jitter = float(statistics.pstdev(rtts) if len(rtts) >= 2 else 0.0)
    return avg_rtt, jitter, float(loss_pct)


def safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if v != v:
        return float(default)
    if v == float("inf") or v == float("-inf"):
        return float(default)
    return float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client_id", required=True, help="client1..client10")
    ap.add_argument("--server_ip", required=True, help="server private IP, e.g., 192.0.2.10")
    ap.add_argument(
        "--scenario",
        required=True,
        choices=["normal", "high_cpu", "disk_load", "net_fluct"],
        help="telemetry label for the current capture window",
    )
    ap.add_argument("--interval", type=float, default=3.0, help="sampling interval in seconds")
    ap.add_argument("--samples", type=int, required=True, help="number of samples to write")
    ap.add_argument("--out", required=True, help="output CSV (append mode; header written if file absent)")
    args = ap.parse_args()

    out_path = os.path.abspath(args.out)
    write_header = not os.path.exists(out_path)

    prev_tx, prev_rx = read_net_bytes()
    prev_dr, prev_dw = read_disk_bytes()
    prev_t = time.time()

    psutil.cpu_percent(interval=None)

    hostname = socket.gethostname()
    private_ip = get_primary_private_ip()

    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp_utc",
                "hostname",
                "client_id",
                "client_private_ip",
                "scenario",
                "cpu_percent",
                "cpu_freq_mhz",
                "ram_percent",
                "loadavg_1m",
                "disk_read_Bps",
                "disk_write_Bps",
                "net_tx_Bps",
                "net_rx_Bps",
                "disk_read_bytes_total",
                "disk_write_bytes_total",
                "net_tx_bytes_total",
                "net_rx_bytes_total",
                "ping_rtt_ms",
                "ping_jitter_ms",
                "ping_loss_pct",
            ])

        for _ in range(args.samples):
            t0 = time.time()

            cpu = safe_float(psutil.cpu_percent(interval=None), 0.0)
            freq = cpu_freq_mhz()
            freq = "" if freq is None else f"{safe_float(freq, 0.0):.2f}"

            ram = safe_float(psutil.virtual_memory().percent, 0.0)
            load1 = safe_float(os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0, 0.0)

            tx, rx = read_net_bytes()
            dr, dw = read_disk_bytes()

            t1 = time.time()
            dt = max(t1 - prev_t, 1e-6)

            net_tx_bps = safe_float((tx - prev_tx) / dt, 0.0)
            net_rx_bps = safe_float((rx - prev_rx) / dt, 0.0)
            disk_r_bps = safe_float((dr - prev_dr) / dt, 0.0)
            disk_w_bps = safe_float((dw - prev_dw) / dt, 0.0)

            prev_tx, prev_rx = tx, rx
            prev_dr, prev_dw = dr, dw
            prev_t = t1

            rtt_ms, jitter_ms, loss_pct = compute_ping_stats(args.server_ip, k=3)

            w.writerow([
                iso_utc_now(),
                hostname,
                args.client_id,
                private_ip,
                args.scenario,
                f"{cpu:.2f}",
                freq,
                f"{ram:.2f}",
                f"{load1:.3f}",
                f"{disk_r_bps:.2f}",
                f"{disk_w_bps:.2f}",
                f"{net_tx_bps:.2f}",
                f"{net_rx_bps:.2f}",
                int(dr),
                int(dw),
                int(tx),
                int(rx),
                f"{safe_float(rtt_ms, 999.0):.3f}",
                f"{safe_float(jitter_ms, 0.0):.3f}",
                f"{safe_float(loss_pct, 100.0):.2f}",
            ])
            f.flush()

            elapsed = time.time() - t0
            sleep_s = args.interval - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    print(f"[DONE] Wrote {args.samples} samples to {out_path}")


if __name__ == "__main__":
    main()
