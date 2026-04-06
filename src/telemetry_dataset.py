from __future__ import annotations
"""
telemetry_dataset.py (COMMON: identical on server + all clients)

Provides:
- F15: exact 15 telemetry features in fixed order
- Window builders: make_windows(), load_client_windows()
- Server helper: make_windows_per_client()
- PyTorch Dataset: TelemetryWindowDataset (client API compatible)

Key fixes for your current issues:
1) Robust scaler loading: supports scaler.json saved as dict OR list.
2) Scaler is applied consistently to BOTH:
   - windowed CNN inputs (Xw)
   - live telemetry vectors (client /telemetry should already be scaled by client using same scaler, but server can scale too)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

# -----------------------------
# EXACT 15 telemetry features (order matters)
# -----------------------------
F15: List[str] = [
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
]


@dataclass
class WindowConfig:
    win: int = 10
    stride: int = 1
    label_col: str = "scenario"
    client_id_col: str = "client_id"
    sort_col: str = "timestamp_utc"
    label_mode: str = "majority"  # "majority" or "last"


# -----------------------------
# Scaler (robust loader)
# -----------------------------
class Scaler:
    """Simple per-feature standard scaler for F15 order."""
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        if mean.size != 15 or scale.size != 15:
            raise ValueError(f"Scaler expects 15-dim mean/scale. Got mean={mean.size} scale={scale.size}")
        scale = np.where(scale == 0, 1.0, scale)
        self.mean = mean
        self.scale = scale

    def transform_2d(self, X: np.ndarray) -> np.ndarray:
        # X: (T,15)
        return (X - self.mean[None, :]) / self.scale[None, :]

    def transform_3d(self, Xw: np.ndarray) -> np.ndarray:
        # Xw: (Nw,15,win)
        return (Xw - self.mean[None, :, None]) / self.scale[None, :, None]


def load_scaler_json(scaler_path: Optional[str], feature_cols: List[str]) -> Optional[Scaler]:
    """
    Supports scaler.json formats:
      A) dict with keys like {"mean": {...} or [...], "scale": {...} or [...]} OR {"mean":..., "std":...}
      B) list:
         - [ {"mean":..., "scale":...} ]  (single element dict)
         - [ {"feature": "cpu_percent", "mean":..., "scale":...}, ... ]  (per-feature list)
    """
    if not scaler_path:
        return None
    try:
        with open(scaler_path, "r") as f:
            obj = json.load(f)

        def _get_ms(d: dict, key_a: str, key_b: str) -> Optional[Any]:
            return d.get(key_a, None) if isinstance(d, dict) else None

        # Case B: list
        if isinstance(obj, list):
            if len(obj) == 0:
                return None
            if isinstance(obj[0], dict) and ("mean" in obj[0] or "scale" in obj[0] or "std" in obj[0]):
                obj = obj[0]  # unwrap single-element list dict
            else:
                # list of per-feature dicts
                mean = np.zeros((15,), dtype=np.float32)
                scale = np.ones((15,), dtype=np.float32)
                by_feat = {str(d.get("feature")): d for d in obj if isinstance(d, dict) and "feature" in d}
                for i, feat in enumerate(feature_cols):
                    d = by_feat.get(feat, {})
                    m = d.get("mean", d.get("mu", 0.0))
                    s = d.get("scale", d.get("std", d.get("sigma", 1.0)))
                    mean[i] = float(m)
                    scale[i] = float(s) if float(s) != 0 else 1.0
                return Scaler(mean=mean, scale=scale)

        # Case A: dict
        if isinstance(obj, dict):
            mean_obj = obj.get("mean", obj.get("mu", None))
            scale_obj = obj.get("scale", obj.get("std", obj.get("sigma", None)))

            def _vector_from(maybe):
                if maybe is None:
                    return None
                if isinstance(maybe, list):
                    return np.asarray(maybe, dtype=np.float32)
                if isinstance(maybe, dict):
                    return np.asarray([maybe.get(f, 0.0) for f in feature_cols], dtype=np.float32)
                return None

            mean = _vector_from(mean_obj)
            scale = _vector_from(scale_obj)

            if mean is None or scale is None:
                raise ValueError("scaler.json missing mean/scale (or std).")

            return Scaler(mean=mean, scale=scale)

        return None
    except Exception as e:
        print(f"[WARN] load_scaler_json failed: {e} (scaler_path={scaler_path})")
        return None


# -----------------------------
# Cleaning + label mapping
# -----------------------------
def _coerce_numeric_and_fill(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    for c in feature_cols:
        med = X[c].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        X[c] = X[c].fillna(med)

    X = X.replace([np.inf, -np.inf], 0.0)
    return X


def _load_class_map(class_map_path: Optional[str]) -> Optional[dict]:
    if not class_map_path:
        return None
    try:
        with open(class_map_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _map_labels(series: pd.Series, class_map: Optional[dict]) -> np.ndarray:
    def map_one(v):
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, float) and float(v).is_integer():
            return int(v)
        sv = str(v)
        if class_map is not None and sv in class_map:
            return int(class_map[sv])
        try:
            return int(sv)
        except Exception:
            return int(abs(hash(sv)) % 4)

    return series.apply(map_one).to_numpy(dtype=np.int64)


# -----------------------------
# Windowing
# -----------------------------
def make_windows(
    X: np.ndarray,  # (T, 15)
    y: np.ndarray,  # (T,)
    win: int,
    stride: int = 1,
    label_mode: str = "majority",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      Xw: (num_windows, 15, win)
      yw: (num_windows,)
    """
    if X.ndim != 2 or X.shape[1] != 15:
        raise ValueError(f"make_windows expects X shape (T,15). Got {X.shape}")
    if len(X) != len(y):
        raise ValueError("X and y must have same length")
    if len(X) < win:
        return np.zeros((0, 15, win), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    Xw_list = []
    yw_list = []

    for i in range(0, len(X) - win + 1, stride):
        chunk = X[i:i + win, :]     # (win, 15)
        Xw_list.append(chunk.T)     # -> (15, win)

        if label_mode == "last":
            lbl = int(y[i + win - 1])
        else:
            lbl = int(np.bincount(y[i:i + win]).argmax())
        yw_list.append(lbl)

    return np.stack(Xw_list, axis=0).astype(np.float32), np.asarray(yw_list, dtype=np.int64)


def load_client_windows(
    csv_path: str,
    win: int,
    feature_cols: List[str] = F15,
    label_col: str = "scenario",
    stride: int = 1,
    label_mode: str = "majority",
    class_map_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    sort_col: str = "timestamp_utc",
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    required = [label_col] + list(feature_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if sort_col in df.columns:
        df = df.sort_values(sort_col)

    class_map = _load_class_map(class_map_path)
    scaler = load_scaler_json(scaler_path, feature_cols)

    Xclean = _coerce_numeric_and_fill(df, list(feature_cols))
    yint = _map_labels(df[label_col], class_map)

    X = Xclean.to_numpy(dtype=np.float32)  # (T,15)
    if scaler is not None:
        X = scaler.transform_2d(X)

    return make_windows(X, yint, win=win, stride=stride, label_mode=label_mode)


def make_windows_per_client(
    csv_path: str,
    win: int,
    feature_cols: List[str] = F15,
    label_col: str = "scenario",
    client_id_col: str = "client_id",
    sort_col: str = "timestamp_utc",
    stride: int = 1,
    label_mode: str = "majority",
    class_map_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    df = pd.read_csv(csv_path)

    required = [client_id_col, label_col] + list(feature_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if sort_col in df.columns:
        df = df.sort_values(sort_col)

    class_map = _load_class_map(class_map_path)
    scaler = load_scaler_json(scaler_path, feature_cols)

    Xclean = _coerce_numeric_and_fill(df, list(feature_cols))
    yint = _map_labels(df[label_col], class_map)
    cid = df[client_id_col].astype(str).to_numpy()

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for c in np.unique(cid):
        mask = (cid == c)
        Xc = Xclean.loc[mask, feature_cols].to_numpy(dtype=np.float32)
        if scaler is not None:
            Xc = scaler.transform_2d(Xc)
        yc = yint[mask]
        Xw, yw = make_windows(Xc, yc, win=win, stride=stride, label_mode=label_mode)
        if len(Xw) == 0:
            continue
        out[str(c)] = (Xw, yw)

    if not out:
        raise RuntimeError("make_windows_per_client produced no windows. Check win/client_id_col/CSV content.")
    return out


# -----------------------------
# PyTorch Dataset (client API compatible)
# -----------------------------
class TelemetryWindowDataset(Dataset):
    """
    (A) CSV-loader mode (used by live_client_api_idea1.py):
        TelemetryWindowDataset(csv_path, win=..., label_col=..., class_map_path=..., scaler_path=...)

    (B) Tensor mode:
        TelemetryWindowDataset(Xw, y)

    Exposes:
      - num_classes
      - input_dim (=15)
      - win
    """
    def __init__(
        self,
        data,
        y=None,
        *,
        win: int = 10,
        feature_cols: List[str] = F15,
        label_col: str = "scenario",
        stride: int = 1,
        label_mode: str = "majority",
        class_map_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        sort_col: str = "timestamp_utc",
        num_classes: int = 4,
    ):
        self.input_dim = 15
        self.win = int(win)

        if isinstance(data, str) and y is None:
            Xw, yw = load_client_windows(
                csv_path=data,
                win=win,
                feature_cols=feature_cols,
                label_col=label_col,
                stride=stride,
                label_mode=label_mode,
                class_map_path=class_map_path,
                scaler_path=scaler_path,
                sort_col=sort_col,
            )
            if Xw.shape[0] == 0:
                raise RuntimeError(f"No windows produced from {data}. Check win={win} and CSV content.")

            self.X = torch.tensor(Xw, dtype=torch.float32)
            self.y = torch.tensor(yw, dtype=torch.long)
            self.num_classes = int(np.max(yw) + 1) if len(yw) > 0 else int(num_classes)
            return

        if y is None:
            raise TypeError("TelemetryWindowDataset: pass CSV path OR (Xw,y) arrays.")

        Xw = np.asarray(data)
        y_arr = np.asarray(y)

        if Xw.ndim != 3 or Xw.shape[1] != 15:
            raise ValueError(f"TelemetryWindowDataset expects Xw shape (N,15,win). Got {Xw.shape}")
        if len(Xw) != len(y_arr):
            raise ValueError("Xw and y must have same length")

        self.X = torch.tensor(Xw, dtype=torch.float32)
        self.y = torch.tensor(y_arr, dtype=torch.long)
        self.num_classes = int(np.max(y_arr) + 1) if len(y_arr) > 0 else int(num_classes)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    print("[telemetry_dataset] F15 length:", len(F15))
