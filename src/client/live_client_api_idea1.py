from __future__ import annotations
"""
live_client_api_idea1.py (CLIENT)

Endpoints:
- GET  /health
- GET  /telemetry          -> returns x (15) in F15 order (scaled if scaler_path provided)
- POST /train_local        -> runs budgeted local training if offload=0; returns updated weights + timing
- POST /impair/cpu_start|cpu_stop
- POST /impair/net_start|net_stop
"""

import argparse
import subprocess
import time
from typing import Any, Dict

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from telemetry_dataset import TelemetryWindowDataset
from model_utils import (
    CNN1DClassifier,
    get_device,
    train_one_epoch_budget,
    state_dict_to_jsonable,
    jsonable_to_state_dict,
)

app = FastAPI()

STATE = {
    "client_id": "clientX",
    "csv": "",
    "win": 10,
    "label_col": "scenario",
    "class_map": None,
    "scaler_path": None,
    "device": get_device(force_cpu=True),
    "ds": None,
    "cpu_impair_on": False,
    "net_impair_on": False,
    "cpu_workers": 0,
    "net_delay_ms": 0.0,
    "net_loss_pct": 0.0,
}


class TrainReq(BaseModel):
    round: int
    local_epochs: int = 2
    lr: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    offload: int = 0
    global_state_dict: Dict[str, Any]
    max_local_batches: int = 60
    max_local_samples: int = 2500


def _ensure_dataset():
    if STATE["ds"] is None:
        STATE["ds"] = TelemetryWindowDataset(
            STATE["csv"],
            win=int(STATE["win"]),
            label_col=str(STATE["label_col"]),
            class_map_path=STATE["class_map"],
            scaler_path=STATE["scaler_path"],
        )


@app.get("/health")
def health():
    return {"ok": True, "client_id": STATE["client_id"]}


@app.get("/telemetry")
def telemetry():
    """
    Returns a 15-dim vector in F15 order.
    Pick a random window and mean over time -> (15,)
    """
    t0 = time.time()
    _ensure_dataset()
    ds: TelemetryWindowDataset = STATE["ds"]

    idx = np.random.randint(0, len(ds))
    xw, _ = ds[idx]  # (15, win)
    x = xw.float().mean(dim=-1).cpu().numpy().astype(np.float32)

    return {
        "x": x.tolist(),
        "telemetry_gen_s": float(time.time() - t0),
        "impair": {"cpu": STATE["cpu_impair_on"], "net": STATE["net_impair_on"]},
    }


def _apply_tc_netem(delay_ms: float, loss_pct: float):
    cmd = f"sudo tc qdisc replace dev eth0 root netem delay {int(delay_ms)}ms loss {float(loss_pct)}%"
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _clear_tc_netem():
    cmd = "sudo tc qdisc del dev eth0 root netem"
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _start_cpu_stress(workers: int):
    if workers <= 0:
        return
    cmd = f"nohup stress-ng --cpu {int(workers)} --cpu-method all --timeout 600s > /tmp/stressng.log 2>&1 &"
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _stop_cpu_stress():
    subprocess.run("pkill -f 'stress-ng --cpu' || true", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@app.post("/impair/cpu_start")
def impair_cpu_start(payload: Dict[str, Any]):
    workers = int(payload.get("cpu_workers", 1))
    STATE["cpu_workers"] = workers
    STATE["cpu_impair_on"] = True
    _start_cpu_stress(workers)
    return {"ok": True, "cpu_workers": workers}


@app.post("/impair/cpu_stop")
def impair_cpu_stop(payload: Dict[str, Any]):
    STATE["cpu_impair_on"] = False
    STATE["cpu_workers"] = 0
    _stop_cpu_stress()
    return {"ok": True}


@app.post("/impair/net_start")
def impair_net_start(payload: Dict[str, Any]):
    delay_ms = float(payload.get("net_delay_ms", 120.0))
    loss_pct = float(payload.get("net_loss_pct", 5.0))
    STATE["net_delay_ms"] = delay_ms
    STATE["net_loss_pct"] = loss_pct
    STATE["net_impair_on"] = True
    _apply_tc_netem(delay_ms, loss_pct)
    return {"ok": True, "delay_ms": delay_ms, "loss_pct": loss_pct}


@app.post("/impair/net_stop")
def impair_net_stop(payload: Dict[str, Any]):
    STATE["net_impair_on"] = False
    STATE["net_delay_ms"] = 0.0
    STATE["net_loss_pct"] = 0.0
    _clear_tc_netem()
    return {"ok": True}


@app.post("/train_local")
def train_local(req: TrainReq):
    _ensure_dataset()
    ds: TelemetryWindowDataset = STATE["ds"]

    device = STATE["device"]
    model = CNN1DClassifier(in_channels=15, n_classes=int(ds.num_classes), win=int(STATE["win"])).to(device)
    gsd = jsonable_to_state_dict(req.global_state_dict, device=device)
    model.load_state_dict(gsd, strict=False)

    t0 = time.time()

    if int(req.offload) == 1:
        time.sleep(0.02)
        return {
            "state_dict": state_dict_to_jsonable(model.state_dict()),
            "n_samples": 0,
            "train_time_s": float(time.time() - t0),
            "batches": 0,
            "offload": 1,
        }

    n_samples = min(int(req.max_local_samples), len(ds))
    idx = np.random.choice(len(ds), size=n_samples, replace=False)
    xb = torch.stack([ds[i][0] for i in idx], dim=0)
    yb = torch.tensor([int(ds[i][1]) for i in idx], dtype=torch.long)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xb, yb),
        batch_size=int(req.batch_size),
        shuffle=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=float(req.lr), weight_decay=float(req.weight_decay))

    batches_total = 0
    for _ in range(int(req.local_epochs)):
        stats = train_one_epoch_budget(model, loader, opt, device, max_batches=int(req.max_local_batches))
        batches_total += int(stats.batches)

    return {
        "state_dict": state_dict_to_jsonable(model.state_dict()),
        "n_samples": int(n_samples),
        "train_time_s": float(time.time() - t0),
        "batches": int(batches_total),
        "offload": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client_id", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--win", type=int, default=10)
    ap.add_argument("--label_col", type=str, default="scenario")
    ap.add_argument("--class_map", type=str, default=None)
    ap.add_argument("--scaler_path", type=str, default=None)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    STATE["client_id"] = args.client_id
    STATE["csv"] = args.csv
    STATE["win"] = args.win
    STATE["label_col"] = args.label_col
    STATE["class_map"] = args.class_map
    STATE["scaler_path"] = args.scaler_path

    uvicorn.run(app, host=args.host, port=int(args.port))


if __name__ == "__main__":
    main()
