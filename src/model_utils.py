from __future__ import annotations
"""
model_utils.py (COMMON: identical on server + all clients)

Fixes/guarantees:
- CNN1DClassifier supports both (input_channels,num_classes) and (in_channels,n_classes)
- JSON-safe state_dict serialization (Tensor -> list) to avoid "Tensor not JSON serializable"
- Budgeted training: train_one_epoch_budget(...)
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(force_cpu: bool = False) -> torch.device:
    if (not force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CNN1DClassifier(nn.Module):
    """
    Input:  (B, 15, win)
    Output: (B, num_classes)

    Accepts either naming convention:
      - input_channels / num_classes
      - in_channels / n_classes
    """
    def __init__(
        self,
        input_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        win: int = 10,
        dropout: float = 0.2,
        in_channels: Optional[int] = None,
        n_classes: Optional[int] = None,
    ):
        super().__init__()
        if input_channels is None:
            input_channels = in_channels
        if num_classes is None:
            num_classes = n_classes
        if input_channels is None:
            input_channels = 15
        if num_classes is None:
            num_classes = 4

        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.win = int(win)

        self.conv1 = nn.Conv1d(self.input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.drop = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


TelemetryCNN1D = CNN1DClassifier


@dataclass
class TrainStats:
    loss: float
    acc: float
    batches: int = 0


@torch.no_grad()
def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())


def train_one_epoch_budget(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_batches: int = 200,
    max_grad_norm: float = 1.0,
) -> TrainStats:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for step, (xb, yb) in enumerate(loader):
        if step >= int(max_batches):
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()

        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += float(loss.item())
        total_acc += _accuracy_from_logits(logits.detach(), yb)
        n += 1

    if n == 0:
        return TrainStats(loss=0.0, acc=0.0, batches=0)
    return TrainStats(loss=total_loss / n, acc=total_acc / n, batches=n)


@torch.no_grad()
def eval_model(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    *,
    max_batches: int = 200,
) -> TrainStats:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for step, (xb, yb) in enumerate(loader):
        if step >= int(max_batches):
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        total_loss += float(loss.item())
        total_acc += _accuracy_from_logits(logits, yb)
        n += 1
    if n == 0:
        return TrainStats(loss=0.0, acc=0.0, batches=0)
    return TrainStats(loss=total_loss / n, acc=total_acc / n, batches=n)


# -----------------------------
# JSON-safe state_dict
# -----------------------------
def state_dict_to_jsonable(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = {
                "dtype": str(v.dtype).replace("torch.", ""),
                "shape": list(v.shape),
                "data": v.detach().cpu().contiguous().view(-1).tolist(),
            }
        else:
            out[k] = v
    return out


def jsonable_to_state_dict(obj: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if isinstance(v, dict) and "data" in v and "shape" in v:
            data = np.asarray(v["data"], dtype=np.float32)

            # -----------------------------
            # PATCH: robust shape handling (supports scalar / empty shape)
            # -----------------------------
            shape_raw = v.get("shape", None)
            t = torch.tensor(data, dtype=torch.float32)

            if shape_raw is None:
                # leave tensor as-is (already matches data)
                pass
            else:
                # normalize to tuple of ints
                if isinstance(shape_raw, (list, tuple)):
                    shape = tuple(int(x) for x in shape_raw)
                else:
                    shape = (int(shape_raw),)

                # scalar parameter: shape == () or []
                if len(shape) == 0:
                    t = t.reshape(())
                else:
                    t = t.reshape(shape)

            out[k] = t.to(device)
    return out


@torch.no_grad()
def l2_norm_of_state_dict(sd: Dict[str, torch.Tensor]) -> float:
    s = 0.0
    for v in sd.values():
        if torch.is_tensor(v):
            s += float((v.detach().float().cpu() ** 2).sum().item())
    return float(np.sqrt(max(s, 0.0)))


@torch.no_grad()
def delta_norm(sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor]) -> float:
    s = 0.0
    for k in sd_a.keys():
        if k not in sd_b:
            continue
        va = sd_a[k]
        vb = sd_b[k]
        if torch.is_tensor(va) and torch.is_tensor(vb):
            d = (va.detach().float().cpu() - vb.detach().float().cpu())
            s += float((d ** 2).sum().item())
    return float(np.sqrt(max(s, 0.0)))
