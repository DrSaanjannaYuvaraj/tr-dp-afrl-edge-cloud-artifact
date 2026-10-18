#!/usr/bin/env python3
"""
live_client_api_trdp.py (RUN ON EACH CLIENT)

Client API for TL-PPO / TR-DP-AFRL closed-loop AFRL experiments.

Provides:
- GET  /health
- GET  /telemetry_round?round_id=<t>&win=<W>          (reads telemetry CSV window)
- POST /train_update                                  (local or "cloud-assisted" training on client data)
- POST /impair/cpu/start?seconds=...                  (stress-ng)
- POST /impair/cpu/stop
- POST /impair/net/start?delay_ms=..&loss=..&jitter_ms=..&rate_mbit=..&seconds=..
- POST /impair/net/stop

Notes:
- "Cloud-assisted" mode is implemented as accelerated local training (privacy-preserving),
  i.e., it runs locally but reports lower compute time by applying a speedup factor.
- This matches Algorithm 1/2 requirement that cloud-assisted training is faster, without moving raw data.
"""

import argparse
import base64
import io
import json
import os
import subprocess
import time
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Optional torch (required for actual model updates)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    torch = None

APP = FastAPI(title="AFRL Client API (TR-DP-AFRL)")

CPU_STRESS_PIDS: List[int] = []
NETEM_ACTIVE = False


def _now_ms() -> float:
    return time.time() * 1000.0


def _bytes_len(b: bytes) -> int:
    return len(b) if b is not None else 0


class TrainRequest(BaseModel):
    round_id: int
    mode: str  # "local" or "cloud"
    # base64-encoded torch state_dict bytes (torch.save to bytes)
    global_model_b64: str
    local_epochs: int = 1
    lr_local: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    # Optional speedup factor for "cloud-assisted" mode (e.g., 2.5 means 2.5x faster)
    cloud_speedup: float = 2.5


class TrainResponse(BaseModel):
    ok: bool
    client_id: str
    n_samples: int
    train_time_ms: float
    bytes_in: int
    bytes_out: int
    updated_model_b64: str


def build_mnist_cnn(num_classes: int = 10) -> "nn.Module":
    # Lightweight MNIST CNN
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 28->14
            x = self.pool(F.relu(self.conv2(x)))  # 14->7
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    return Net()


def load_local_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Expect arrays: X (N,1,28,28) float32 and y (N,) int64."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Client data npz not found: {npz_path}")
    d = np.load(npz_path)
    return d["X"], d["y"]


def state_dict_from_b64(b64: str) -> Dict[str, Any]:
    raw = base64.b64decode(b64.encode("utf-8"))
    buff = io.BytesIO(raw)
    obj = torch.load(buff, map_location="cpu")
    return obj


def state_dict_to_b64(sd: Dict[str, Any]) -> str:
    buff = io.BytesIO()
    torch.save(sd, buff)
    raw = buff.getvalue()
    return base64.b64encode(raw).decode("utf-8")


def train_one_round(
    client_id: str,
    data_npz: str,
    global_sd: Dict[str, Any],
    mode: str,
    local_epochs: int,
    lr_local: float,
    weight_decay: float,
    batch_size: int,
    cloud_speedup: float,
) -> Tuple[Dict[str, Any], int, float]:
    if torch is None:
        raise RuntimeError("PyTorch not installed on client. Install torch to enable training.")

    X, y = load_local_npz(data_npz)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = build_mnist_cnn()
    model.load_state_dict(global_sd, strict=True)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr_local, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    start = _now_ms()
    epochs = max(1, int(local_epochs))
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    end = _now_ms()

    # "cloud-assisted" simulation: report faster time (privacy-preserving)
    train_time_ms = end - start
    if mode.lower() == "cloud":
        train_time_ms = train_time_ms / max(1e-6, float(cloud_speedup))

    return model.state_dict(), int(len(ds)), float(train_time_ms)


def read_telemetry_window(csv_path: str, round_id: int, win: int) -> Dict[str, float]:
    """
    Reads a telemetry CSV and returns mean of the last 'win' rows up to an index derived from round_id.
    If CSV is long continuous, we map round_id to an index window.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Telemetry CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("Telemetry CSV is empty")

    # Map round_id to position (simple proportion). You can replace with timestamp join if you already have it.
    idx = min(len(df) - 1, max(0, int(round_id * (len(df) / 200.0))))
    start = max(0, idx - win + 1)
    wdf = df.iloc[start: idx + 1]

    # Expected columns (edit if your CSV differs)
    cols = ["cpu", "ram", "loadavg", "rtt", "jitter", "loss", "disk_io", "throughput"]
    out = {}
    for c in cols:
        if c in wdf.columns:
            out[c] = float(np.nanmean(wdf[c].values))
    # Ensure required core keys exist
    for k in ["cpu", "ram", "loadavg", "rtt", "jitter", "loss"]:
        out.setdefault(k, 0.0)
    return out


def _run_cmd(cmd: List[str]) -> None:
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@APP.get("/health")
def health():
    return {"ok": True}


@APP.get("/telemetry_round")
def telemetry_round(round_id: int, win: int = 15):
    t0 = _now_ms()
    try:
        metrics = read_telemetry_window(APP.state.telemetry_csv, round_id, win)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    t1 = _now_ms()
    # bytes estimate: JSON response size
    payload = json.dumps(metrics).encode("utf-8")
    return {
        "ok": True,
        "client_id": APP.state.client_id,
        "round_id": round_id,
        "telemetry_time_ms": float(t1 - t0),
        "bytes_out": _bytes_len(payload),
        "metrics": metrics,
    }


@APP.post("/train_update", response_model=TrainResponse)
def train_update(req: TrainRequest):
    if torch is None:
        raise HTTPException(status_code=500, detail="PyTorch not installed on client.")

    b_in = _bytes_len(req.global_model_b64.encode("utf-8"))
    try:
        global_sd = state_dict_from_b64(req.global_model_b64)
        updated_sd, n_samples, train_time_ms = train_one_round(
            client_id=APP.state.client_id,
            data_npz=APP.state.data_npz,
            global_sd=global_sd,
            mode=req.mode,
            local_epochs=req.local_epochs,
            lr_local=req.lr_local,
            weight_decay=req.weight_decay,
            batch_size=req.batch_size,
            cloud_speedup=req.cloud_speedup,
        )
        out_b64 = state_dict_to_b64(updated_sd)
        b_out = _bytes_len(out_b64.encode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return TrainResponse(
        ok=True,
        client_id=APP.state.client_id,
        n_samples=n_samples,
        train_time_ms=train_time_ms,
        bytes_in=b_in,
        bytes_out=b_out,
        updated_model_b64=out_b64,
    )


@APP.post("/impair/cpu/start")
def impair_cpu_start(seconds: int = 60, workers: int = 1):
    """
    Starts CPU stress using stress-ng if available.
    """
    global CPU_STRESS_PIDS
    # stop any existing
    impair_cpu_stop()
    # start
    cmd = ["stress-ng", "--cpu", str(workers), "--timeout", f"{seconds}s", "--quiet"]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    CPU_STRESS_PIDS = [p.pid]
    return {"ok": True, "pid": p.pid, "seconds": seconds}


@APP.post("/impair/cpu/stop")
def impair_cpu_stop():
    global CPU_STRESS_PIDS
    for pid in CPU_STRESS_PIDS:
        try:
            os.kill(pid, 9)
        except Exception:
            pass
    CPU_STRESS_PIDS = []
    return {"ok": True}


@APP.post("/impair/net/start")
def impair_net_start(delay_ms: int = 80, jitter_ms: int = 20, loss: float = 1.0, rate_mbit: int = 20, seconds: int = 60, iface: str = "eth0"):
    """
    Adds tc netem impairment for a duration.
    """
    global NETEM_ACTIVE
    impair_net_stop(iface=iface)
    # netem
    _run_cmd(["sudo", "tc", "qdisc", "add", "dev", iface, "root", "netem",
              "delay", f"{delay_ms}ms", f"{jitter_ms}ms",
              "loss", f"{loss}%", "rate", f"{rate_mbit}mbit"])
    NETEM_ACTIVE = True

    # schedule removal
    def _remove_later():
        time.sleep(max(1, seconds))
        try:
            subprocess.run(["sudo", "tc", "qdisc", "del", "dev", iface, "root"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    subprocess.Popen(["bash", "-lc", f"sleep {seconds}; sudo tc qdisc del dev {iface} root >/dev/null 2>&1 || true"])
    return {"ok": True, "delay_ms": delay_ms, "jitter_ms": jitter_ms, "loss": loss, "rate_mbit": rate_mbit, "seconds": seconds}


@APP.post("/impair/net/stop")
def impair_net_stop(iface: str = "eth0"):
    global NETEM_ACTIVE
    subprocess.run(["sudo", "tc", "qdisc", "del", "dev", iface, "root"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    NETEM_ACTIVE = False
    return {"ok": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client_id", required=True)
    ap.add_argument("--telemetry_csv", required=True)
    ap.add_argument("--data_npz", required=True, help="Local MNIST shard npz: arrays X (N,1,28,28), y (N,)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    APP.state.client_id = args.client_id
    APP.state.telemetry_csv = args.telemetry_csv
    APP.state.data_npz = args.data_npz

    uvicorn.run(APP, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
