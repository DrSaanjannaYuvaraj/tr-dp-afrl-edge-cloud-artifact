#!/usr/bin/env python3
"""
live_client_api_v4.py (RUN ON EACH CLIENT)

Endpoints:
- GET  /health
- GET  /telemetry_latest
- POST /train_local
- GET  /get_batch

Matches:
- feature_cols.json (15 features)
- class_map.json (4 scenarios)
- scaler.json (mean/scale)
"""

import argparse, io, json, time, base64
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI()
STATE: Dict[str, Any] = {}

MODEL_ARCH = "Telemetry1DCNN_conv16_32_v4"

# -------------------- CNN model (MUST MATCH SERVER) --------------------
class Telemetry1DCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, n_classes),
        )
    def forward(self, x):  # (B,F)
        return self.net(x.unsqueeze(1))

# -------------------- Base64 state_dict helpers --------------------
def sd_to_b64(sd) -> str:
    buf = io.BytesIO()
    torch.save(sd, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def b64_to_sd(s: str):
    raw = base64.b64decode(s.encode("ascii"))
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location=DEVICE)

# -------------------- Scaler helpers --------------------
def load_scaler(path: str) -> dict:
    sc = json.load(open(path))
    if "scale" not in sc and "std" in sc:
        sc["scale"] = sc["std"]
    if "mean" not in sc or "scale" not in sc:
        raise RuntimeError("scaler.json must contain mean + scale")
    sc["mean"] = np.array(sc["mean"], dtype=np.float32)
    sc["scale"] = np.array(sc["scale"], dtype=np.float32)
    sc["scale"] = np.where(sc["scale"] == 0, 1.0, sc["scale"])
    return sc

def apply_scaler(X: np.ndarray, scaler: Optional[dict]) -> np.ndarray:
    X = X.astype(np.float32)
    if scaler is None:
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean = scaler["mean"].reshape(1, -1)
    scale = scaler["scale"].reshape(1, -1)
    Xs = (X - mean) / scale
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# -------------------- Data loaders --------------------
def build_loaders(df: pd.DataFrame,
                  feature_cols: List[str],
                  label_col: str,
                  class_map: dict,
                  scaler: Optional[dict],
                  seed: int = 42):

    # features
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    X = apply_scaler(X, scaler)

    # labels
    y = df[label_col].astype(str).map(class_map)
    if y.isna().any():
        bad = df.loc[y.isna(), label_col].astype(str).unique().tolist()
        raise RuntimeError(f"Unknown labels not in class_map.json: {bad}")
    y = y.astype(int).to_numpy(dtype=np.int64)

    n = len(df)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    cut = int(0.8 * n)
    tr_idx, va_idx = idx[:cut], idx[cut:]

    train_ds = TensorDataset(torch.tensor(X[tr_idx]), torch.tensor(y[tr_idx]))
    val_ds   = TensorDataset(torch.tensor(X[va_idx]), torch.tensor(y[va_idx]))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False)
    batch_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    return train_loader, val_loader, batch_loader

@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += len(yb)
    return correct / max(total, 1)

def train_epochs(model, loader, epochs=1, lr=1e-3):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(max(1, int(epochs))):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

# -------------------- JSON sanitize --------------------
def sanitize_json_value(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float) and (not np.isfinite(v)):
        return 0.0
    if v is None:
        return 0.0
    return v

def sanitize_row_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: sanitize_json_value(v) for k, v in d.items()}

# -------------------- API schemas --------------------
class TrainReq(BaseModel):
    state_dict_b64: str
    local_epochs: int = 1
    lr: float = 1e-3

# -------------------- Endpoints --------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ok": True,
        "client_id": STATE["client_id"],
        "rows": int(len(STATE["df"])),
        "features": int(STATE["n_features"]),
        "classes": STATE["classes"],
        "scaled": bool(STATE["scaled"]),
        "model_arch": MODEL_ARCH,
        "feature_cols": STATE["feature_cols"],
    }

@app.get("/telemetry_latest")
def telemetry_latest():
    i = STATE["tel_idx"]
    df = STATE["df"]
    if i >= len(df):
        i = 0

    row = df.iloc[i].to_dict()
    STATE["tel_idx"] = i + 1

    row["client_id"] = STATE["client_id"]
    row["row_index"] = int(i)
    row["ts_unix"] = float(time.time())
    return sanitize_row_dict(row)

@app.post("/train_local")
def train_local(req: TrainReq):
    try:
        model = Telemetry1DCNN(STATE["n_features"], STATE["n_classes"]).to(DEVICE)
        sd = b64_to_sd(req.state_dict_b64)
        model.load_state_dict(sd, strict=True)

        t0 = time.time()
        train_epochs(model, STATE["train_loader"], epochs=req.local_epochs, lr=req.lr)
        train_time = time.time() - t0
        val = eval_acc(model, STATE["val_loader"])

        out_b64 = sd_to_b64(model.state_dict())
        out_bytes = len(base64.b64decode(out_b64.encode("ascii")))

        return {
            "ok": True,
            "client_id": STATE["client_id"],
            "state_dict_b64": out_b64,
            "n_samples": int(len(STATE["train_loader"].dataset)),
            "train_time_s": float(train_time),
            "val_acc": float(val),
            "bytes_out": int(out_bytes),
        }
    except Exception as e:
        return {"ok": False, "client_id": STATE.get("client_id","unknown"), "error": f"{type(e).__name__}: {e}"}

@app.get("/get_batch")
def get_batch():
    xb, yb = next(iter(STATE["batch_loader"]))
    # xb is ALREADY SCALED (we scaled in build_loaders)
    return {"xb": xb.numpy().tolist(), "yb": yb.numpy().tolist()}

# -------------------- main --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--client_id", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--feature_cols_json", required=True)
    p.add_argument("--class_map_json", required=True)
    p.add_argument("--scaler_json", required=False, default=None)
    p.add_argument("--label_col", default="scenario")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    STATE["client_id"] = args.client_id

    df = pd.read_csv(args.csv).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    feature_cols = json.load(open(args.feature_cols_json))
    class_map = json.load(open(args.class_map_json))

    scaler = None
    scaled = False
    if args.scaler_json:
        scaler = load_scaler(args.scaler_json)
        if "feature_cols" in scaler and list(scaler["feature_cols"]) != list(feature_cols):
            raise RuntimeError("feature_cols.json != scaler.json feature_cols")
        scaled = True

    # Validate required columns exist
    missing = [c for c in feature_cols + [args.label_col] if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing columns: {missing}")

    classes = [c for c,_ in sorted(class_map.items(), key=lambda x: x[1])]

    STATE["df"] = df
    STATE["tel_idx"] = 0
    STATE["feature_cols"] = feature_cols
    STATE["classes"] = classes
    STATE["n_features"] = len(feature_cols)
    STATE["n_classes"] = len(class_map)
    STATE["scaled"] = scaled

    tr, va, bl = build_loaders(df, feature_cols, args.label_col, class_map, scaler)
    STATE["train_loader"], STATE["val_loader"], STATE["batch_loader"] = tr, va, bl

    print(f"[{args.client_id}] READY rows={len(df)} features={STATE['n_features']} classes={STATE['n_classes']} scaled={scaled} arch={MODEL_ARCH} port={args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
