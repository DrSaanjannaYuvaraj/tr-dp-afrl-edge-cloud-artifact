# afrl_idea1_run_one_strategy.py (SERVER) — SINGLE REFERENCE (FINAL)
# Purpose: produce JSON logs that can generate Fig3–Fig14 + TableX + significance using make_journal_figs_and_table.py
#
# IMPORTANT:
# - Algorithm 1/2 logic is unchanged (selection, K budget, offloading, reward, PPO update, aggregation).
# - We only add/standardize logging keys so plotting never misses Fig8/Fig9/Fig11 again.
#
# Adds/ensures keys:
#   - acc / reward / latency_s  -> Fig3/4/5
#   - tl_ppo vs tr_dp_afrl      -> Fig6
#   - WITH/NO telemetry runs    -> Fig7a/Fig7b (requires running both variants)
#   - selected_clients (alias)  -> Fig8 heatmap (your plotter warned missing selected_clients)
#   - telemetry_overhead_ms/bytes -> Fig9 overhead
#   - offload_intensity         -> Fig10
#   - churn (alias) + churn_ratio -> Fig11
#
# Fig12 (scalability) + Fig13 (robustness) are produced by plotting script when you pass folders via
# --scal_dirs and --robust_dirs (not a single-run logging issue).

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, TensorDataset

from telemetry_dataset import F15, make_windows_per_client
from model_utils import (
    CNN1DClassifier,
    get_device,
    train_one_epoch_budget,
    eval_model,
    state_dict_to_jsonable,
    jsonable_to_state_dict,
    delta_norm,
)

from ppo_core import PPOConfig, OffloadPPO, SelectionPPO
from dqn_core import DQNConfig, MultiHeadDQN


# -----------------------------
# Networking helpers
# -----------------------------
def safe_get(url: str, path: str, timeout: float, retries: int = 3, backoff: float = 1.6):
    last = None
    for k in range(retries):
        try:
            return requests.get(url + path, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last = e
            time.sleep(backoff ** k)
    raise last


def safe_post(url: str, path: str, payload: dict, timeout: float, retries: int = 3, backoff: float = 1.6):
    last = None
    for k in range(retries):
        try:
            return requests.post(url + path, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last = e
            time.sleep(backoff ** k)
    raise last


def preflight_clients(clients: List[str], http_timeout: float) -> List[str]:
    alive = []
    for c in clients:
        try:
            r = requests.get(c + "/health", timeout=min(5.0, http_timeout))
            if r.status_code == 200:
                alive.append(c)
            else:
                print(f"[WARN] not healthy status={r.status_code} -> drop {c}")
        except Exception as e:
            print(f"[WARN] unreachable -> drop {c} ({e})")
    if not alive:
        raise RuntimeError("No clients reachable after /health preflight.")
    if len(alive) != len(clients):
        print(f"[INFO] Preflight alive {len(alive)}/{len(clients)}")
    return alive


# -----------------------------
# Running normalizer (min-max) + CLAMP
# -----------------------------
class RunningMinMax:
    """
    Always returns a value in [0,1] (clamped).
    """
    def __init__(self):
        self.min_v = None
        self.max_v = None

    def update(self, x: float):
        x = float(x)
        if self.min_v is None:
            self.min_v = x
            self.max_v = x
        else:
            self.min_v = min(self.min_v, x)
            self.max_v = max(self.max_v, x)

    @staticmethod
    def _clamp01(v: float) -> float:
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return float(v)

    def norm(self, x: float, fallback_scale: float = 1.0) -> float:
        x = float(x)
        if self.min_v is None or self.max_v is None or abs(self.max_v - self.min_v) < 1e-9:
            v = x / max(1e-6, float(fallback_scale))
            return self._clamp01(v)
        v = (x - self.min_v) / (self.max_v - self.min_v + 1e-9)
        return self._clamp01(v)


# -----------------------------
# Reward (normalized)
# -----------------------------
@dataclass
class RewardWeights:
    alpha_acc: float = 1.0
    beta_lat: float = 0.25
    gamma_off: float = 0.10
    lambda_stab: float = 0.15


def compute_reward(
    acc: float,
    lat_normed: float,
    off_normed: float,
    churn_ratio: float,
    rw: RewardWeights,
) -> float:
    # maximize accuracy, penalize latency/offload/stability
    return (
        (rw.alpha_acc * float(acc))
        - (rw.beta_lat * float(lat_normed))
        - (rw.gamma_off * float(off_normed))
        - (rw.lambda_stab * float(churn_ratio))
    )


# -----------------------------
# Impairment triggers
# -----------------------------
def maybe_apply_impairments(t_round: int, args: argparse.Namespace, http_timeout: float):
    if args.impair_start_round is None or args.impair_end_round is None:
        return

    payload_common = {
        "round": int(t_round),
        "net_delay_ms": float(args.net_delay_ms),
        "net_loss_pct": float(args.net_loss_pct),
        "cpu_workers": int(args.cpu_workers),
    }

    if t_round == args.impair_start_round:
        if args.cpu_impair_url:
            print(f"[IMPAIR] cpu_start round={t_round} url={args.cpu_impair_url} workers={args.cpu_workers}")
            safe_post(args.cpu_impair_url, "/impair/cpu_start", payload_common, timeout=http_timeout)
        if args.net_impair_url:
            print(
                f"[IMPAIR] net_start round={t_round} url={args.net_impair_url} delay={args.net_delay_ms}ms loss={args.net_loss_pct}%"
            )
            safe_post(args.net_impair_url, "/impair/net_start", payload_common, timeout=http_timeout)

    if t_round == args.impair_end_round:
        if args.cpu_impair_url:
            print(f"[IMPAIR] cpu_stop round={t_round} url={args.cpu_impair_url}")
            safe_post(args.cpu_impair_url, "/impair/cpu_stop", {"round": int(t_round)}, timeout=http_timeout)
        if args.net_impair_url:
            print(f"[IMPAIR] net_stop round={t_round} url={args.net_impair_url}")
            safe_post(args.net_impair_url, "/impair/net_stop", {"round": int(t_round)}, timeout=http_timeout)


# -----------------------------
# FedAvg aggregation
# -----------------------------
def weighted_average_state_dict(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("No state_dicts to average.")
    keys = state_dicts[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = None
        for sd, w in zip(state_dicts, weights):
            v = sd[k].detach().cpu().float()
            acc = v * w if acc is None else acc + v * w
        out[k] = acc
    return out


# -----------------------------
# LIVE telemetry -> aggregate stats + overhead bytes/time
# -----------------------------
def get_live_telemetry_matrix(clients: List[str], http_timeout: float) -> Tuple[np.ndarray, float, int]:
    """
    Returns:
      Xlive: (N,15)
      telemetry_time_s: wall time for collecting telemetry
      telemetry_bytes: bytes read from telemetry HTTP responses (best-effort)
    """
    t0 = time.time()
    total_bytes = 0
    X = []
    for c in clients:
        try:
            r = safe_get(c, "/telemetry", timeout=min(http_timeout, 10.0))
            try:
                total_bytes += int(len(r.content))
            except Exception:
                pass

            if r.status_code == 200:
                x = r.json().get("x", [0.0] * 15)
                if not isinstance(x, list) or len(x) != 15:
                    x = [0.0] * 15
            else:
                x = [0.0] * 15
        except Exception:
            x = [0.0] * 15
        X.append(np.array(x, dtype=np.float32))
    return np.stack(X, axis=0), float(time.time() - t0), int(total_bytes)


def agg_telemetry_stats(Xlive: np.ndarray, ablate_telemetry: bool) -> Tuple[np.ndarray, np.ndarray]:
    if ablate_telemetry:
        mu = np.zeros((15,), dtype=np.float32)
        sd = np.zeros((15,), dtype=np.float32)
    else:
        mu = Xlive.mean(axis=0).astype(np.float32)
        sd = Xlive.std(axis=0).astype(np.float32)
    return mu, sd


def make_state_vec(
    mu: np.ndarray,
    sd: np.ndarray,
    acc_prev: float,
    lat_prev: float,
    churn_prev: float,
    ablate_telemetry: bool,
) -> np.ndarray:
    # State = telemetry aggregate + learning feedback
    if ablate_telemetry:
        extra = np.array([acc_prev, 0.0, 0.0], dtype=np.float32)
    else:
        extra = np.array([acc_prev, lat_prev, churn_prev], dtype=np.float32)
    return np.concatenate([mu, sd, extra], axis=0).astype(np.float32)


# -----------------------------
# Warm-start pretraining on SERVER
# -----------------------------
def warm_start_pretrain(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, args: argparse.Namespace, device: torch.device):
    if args.warm_start_epochs <= 0:
        return
    print(f"[WARM_START] epochs={args.warm_start_epochs} max_batches={args.warm_start_max_batches}")
    opt = torch.optim.Adam(model.parameters(), lr=args.warm_start_lr, weight_decay=args.weight_decay)

    best_val = -1.0
    best_sd = None

    for ep in range(1, args.warm_start_epochs + 1):
        tr = train_one_epoch_budget(model, train_loader, opt, device, max_batches=args.warm_start_max_batches)
        va = eval_model(model, val_loader, device, max_batches=args.eval_max_batches)
        print(f"[WARM_START][ep={ep}] train_acc={tr.acc:.4f} val_acc={va.acc:.4f} train_loss={tr.loss:.4f} val_loss={va.loss:.4f}")
        if va.acc > best_val:
            best_val = va.acc
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_sd is not None:
        model.load_state_dict(best_sd, strict=False)
        print(f"[WARM_START] loaded best checkpoint val_acc={best_val:.4f}")


# -----------------------------
# Baseline offload for non-RL strategies
# -----------------------------
def choose_baseline_offload(strategy: str, Xlive: np.ndarray) -> np.ndarray:
    n = Xlive.shape[0]
    if strategy in ("fedavg", "contextaware", "contextaware_cnn", "heuristic"):
        rtt = Xlive[:, 12]
        loss = Xlive[:, 14]
        a = ((rtt > np.percentile(rtt, 60)) | (loss > np.percentile(loss, 60))).astype(np.int64)
        if a.shape[0] != n:
            a = np.zeros((n,), dtype=np.int64)
        return a
    raise ValueError(strategy)


# -----------------------------
# Enforce min local participants
# -----------------------------
def enforce_min_local_training(
    a: np.ndarray,
    selected: np.ndarray,
    Xlive: np.ndarray,
    min_participants: int,
) -> np.ndarray:
    if min_participants <= 0:
        return a

    sel_idx = np.where(selected == 1)[0]
    if sel_idx.size == 0:
        return a

    local_mask = (selected == 1) & (a == 0)
    n_local = int(local_mask.sum())
    if n_local >= min_participants:
        return a

    need = min_participants - n_local
    cand = sel_idx[a[sel_idx] == 1]
    if cand.size == 0:
        return a

    cpu = Xlive[cand, 0]
    freq = Xlive[cand, 1]
    load1 = Xlive[cand, 3]

    score = (-cpu) + (-0.5 * load1) + (0.001 * freq)
    order = np.argsort(-score)
    flip = cand[order[: min(need, cand.size)]]
    a2 = a.copy()
    a2[flip] = 0
    return a2


# -----------------------------
# Cloud pseudo-client update (returns w_i, n_i)
# -----------------------------
def cloud_pseudo_client_update(
    global_model: torch.nn.Module,
    windows_by_client: Dict[str, Tuple[np.ndarray, np.ndarray]],
    client_id: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], int]:
    if client_id not in windows_by_client:
        return {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}, 0

    Xw, yw = windows_by_client[client_id]
    if len(yw) == 0:
        return {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}, 0

    model = CNN1DClassifier(in_channels=15, n_classes=args.n_classes, win=args.win).to(device)
    model.load_state_dict({k: v.detach().clone() for k, v in global_model.state_dict().items()}, strict=False)

    rng = np.random.default_rng(seed=args.seed + 1000 * int(args._round_seed_offset))
    idx = rng.choice(len(yw), size=min(args.cloud_samples_per_offload, len(yw)), replace=False)

    xb = torch.tensor(Xw[idx], dtype=torch.float32)
    yb = torch.tensor(yw[idx], dtype=torch.long)

    loader = DataLoader(TensorDataset(xb, yb), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.cloud_lr, weight_decay=args.weight_decay)
    train_one_epoch_budget(model, loader, opt, device, max_batches=args.cloud_max_batches)

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, int(len(idx))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.outdir, exist_ok=True)
    http_timeout = float(args.http_timeout)

    variant_label = getattr(args, "variant_label", None) or "default"
    print(f"[VARIANT] variant_label={variant_label}")

    # Console validation (you asked to verify these)
    print(f"[KL_CFG] delta_off={args.delta_off:.6f} delta_sel={args.delta_sel:.6f}")
    print(f"[STAB_CFG] lambda_stab={args.lambda_stab:.6f}")
    print(f"[ABLATION] ablate_telemetry={bool(args.ablate_telemetry)}")
    print(f"[MIN_LOCAL] min_participants={int(args.min_participants)}")

    clients = [c.strip() for c in args.clients.split(",") if c.strip()]
    clients = preflight_clients(clients, http_timeout=http_timeout)
    n_clients = len(clients)

    if args.client_ids:
        client_ids = [x.strip() for x in args.client_ids.split(",") if x.strip()]
        if len(client_ids) != n_clients:
            raise ValueError("--client_ids must have same length as --clients")
    else:
        client_ids = [f"client{i+1}" for i in range(n_clients)]

    windows_by_client = make_windows_per_client(
        csv_path=args.csv,
        win=args.win,
        feature_cols=F15,
        label_col=args.label_col,
        class_map_path=args.class_map,
        scaler_path=args.scaler_path,
    )

    Xall = np.concatenate([v[0] for v in windows_by_client.values()], axis=0)
    yall = np.concatenate([v[1] for v in windows_by_client.values()], axis=0)

    n_all = len(yall)
    split = int(n_all * (1.0 - args.val_frac))
    Xtr, ytr = Xall[:split], yall[:split]
    Xva, yva = Xall[split:], yall[split:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(Xva), torch.tensor(yva)),
        batch_size=args.batch_size,
        shuffle=False
    )

    device = get_device(force_cpu=True)
    model = CNN1DClassifier(in_channels=15, n_classes=args.n_classes, win=args.win).to(device)

    warm_start_pretrain(model, train_loader, val_loader, args, device=device)

    state_dim = 15 + 15 + 3
    off_cfg = PPOConfig(delta_kl=args.delta_off)
    sel_cfg = PPOConfig(delta_kl=args.delta_sel)

    off_ppo = None
    sel_ppo = None
    dqn = None

    if args.strategy in ("tl_ppo", "tr_dp_afrl"):
        off_ppo = OffloadPPO(state_dim=state_dim, n_clients=n_clients, cfg=off_cfg)
    if args.strategy == "tr_dp_afrl":
        sel_ppo = SelectionPPO(state_dim=state_dim, n_clients=n_clients, cfg=sel_cfg)
    if args.strategy in ("dqn", "ddqn"):
        dqn = MultiHeadDQN(state_dim=state_dim, n_clients=n_clients, cfg=DQNConfig(), ddqn=(args.strategy == "ddqn"))

    rw = RewardWeights(args.alpha_acc, args.beta_lat, args.gamma_off, args.lambda_stab)

    # Keep old keys + add aliases your plotter expects
    logs = {k: [] for k in [
        "round", "acc", "reward",
        "latency_s", "telemetry_time_s", "train_time_s",
        "offload_intensity", "offload_norm",
        "churn_l1", "churn_ratio",
        "selected_mask", "weight_delta",
        "variant_label", "strategy", "seed",
        # REQUIRED for Fig8/Fig9/Fig11:
        "selected_clients",                 # <-- key your plot script complained is missing
        "selected_clients_per_round",       # helpful extra
        "telemetry_overhead_ms",
        "telemetry_overhead_bytes",
        "churn",                            # alias for churn_ratio (Fig11)
        # audit:
        "delta_off", "delta_sel", "lambda_stab",
    ]}

    # churn baseline
    if args.strategy == "tr_dp_afrl":
        prev_selected = np.zeros(n_clients, dtype=np.int64)
    else:
        prev_selected = np.ones(n_clients, dtype=np.int64)

    prev_acc, prev_lat, prev_churn = 0.0, 0.0, 0.0

    X0, _t0, _b0 = get_live_telemetry_matrix(clients, http_timeout=http_timeout)
    mu0, sd0 = agg_telemetry_stats(X0, ablate_telemetry=bool(args.ablate_telemetry))
    st_prev_mu, st_prev_sd = mu0, sd0

    lat_norm = RunningMinMax()
    off_norm = RunningMinMax()

    args._round_seed_offset = 0

    for t in range(1, int(args.rounds) + 1):
        args._round_seed_offset = t
        maybe_apply_impairments(t, args, http_timeout=http_timeout)

        t_round_start = time.time()

        Xlive, tel_time_s, tel_bytes = get_live_telemetry_matrix(clients, http_timeout=http_timeout)
        mu_t, sd_t = agg_telemetry_stats(Xlive, ablate_telemetry=bool(args.ablate_telemetry))

        s = make_state_vec(
            st_prev_mu, st_prev_sd,
            prev_acc, prev_lat, prev_churn,
            ablate_telemetry=bool(args.ablate_telemetry),
        )

        # -----------------------------
        # Selection (Algorithm 2)  (UNCHANGED)
        # -----------------------------
        if args.strategy == "tr_dp_afrl":
            g, lam_raw, logp_sel = sel_ppo.act(s)  # type: ignore

            idx_sorted = np.argsort(-lam_raw)
            selected = np.zeros(n_clients, dtype=np.int64)
            for j in idx_sorted:
                if selected.sum() >= args.K:
                    break
                if g[j] == 1:
                    selected[j] = 1

            if selected.sum() < args.K:
                selected[:] = 0
                selected[idx_sorted[: args.K]] = 1

            w = lam_raw * selected
            if w.sum() <= 0:
                w = selected.astype(np.float32)
            w = w / (w.sum() + 1e-12)

            w_min = float(args.trdp_wmin)
            w_max = float(args.trdp_wmax)
            sel_mask = (selected == 1)
            if sel_mask.any():
                w_sel = w[sel_mask]
                w_sel = np.clip(w_sel, w_min, w_max)
                w_sel = w_sel / (w_sel.sum() + 1e-12)
                w = np.zeros_like(w, dtype=np.float32)
                w[sel_mask] = w_sel.astype(np.float32)
            else:
                w = np.ones(n_clients, dtype=np.float32) / float(n_clients)

        else:
            selected = np.ones(n_clients, dtype=np.int64)
            w = np.ones(n_clients, dtype=np.float32) / float(n_clients)
            logp_sel = 0.0
            g = selected.copy()
            lam_raw = w.copy()

        churn_l1 = float(np.sum(np.abs(selected - prev_selected)))
        churn_ratio = float(churn_l1) / float(n_clients)
        prev_selected = selected.copy()

        selected_client_ids = [client_ids[i] for i in np.where(selected == 1)[0].tolist()]

        # -----------------------------
        # Offloading action a_t (Algorithm 1/2) (UNCHANGED)
        # -----------------------------
        if args.strategy in ("tl_ppo", "tr_dp_afrl"):
            a, logp_off, _ = off_ppo.act(s)  # type: ignore
        elif args.strategy in ("dqn", "ddqn"):
            a = dqn.act(s)  # type: ignore
            logp_off = 0.0
        else:
            a = choose_baseline_offload(args.strategy, Xlive)
            logp_off = 0.0

        if not isinstance(a, np.ndarray):
            a = np.array(a, dtype=np.int64)
        a = a.astype(np.int64).reshape(-1)
        if a.shape[0] != n_clients:
            a = np.zeros((n_clients,), dtype=np.int64)

        a = enforce_min_local_training(a=a, selected=selected, Xlive=Xlive, min_participants=int(args.min_participants))

        if args.strategy == "tr_dp_afrl":
            sel_idx = np.where(selected == 1)[0]
            offload_intensity = float(np.mean(a[sel_idx])) if sel_idx.size > 0 else float(np.mean(a))
        else:
            offload_intensity = float(np.mean(a))

        sd_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        local_sds: List[Dict[str, torch.Tensor]] = []
        local_wts: List[float] = []
        contrib_idx: List[int] = []

        train_time_s = 0.0
        cloud_samples = 0

        for i, url in enumerate(clients):
            if selected[i] == 0:
                continue

            # Cloud pseudo-client update
            if int(a[i]) == 1 and args.strategy in ("tl_ppo", "tr_dp_afrl"):
                t_cloud0 = time.time()
                sd_cloud, ns_cloud = cloud_pseudo_client_update(model, windows_by_client, client_ids[i], args, device=device)
                train_time_s += float(time.time() - t_cloud0)
                cloud_samples += int(ns_cloud)

                local_sds.append(sd_cloud)
                local_wts.append(float(max(1, ns_cloud)))
                contrib_idx.append(i)
                continue

            payload = {
                "round": t,
                "local_epochs": int(args.local_epochs),
                "lr": float(args.lr_local),
                "weight_decay": float(args.weight_decay),
                "batch_size": int(args.batch_size),
                "offload": int(a[i]),
                "global_state_dict": state_dict_to_jsonable(model.state_dict()),
                "max_local_batches": int(args.max_local_batches),
                "max_local_samples": int(args.max_local_samples),
            }

            t_loc0 = time.time()
            resp = safe_post(url, "/train_local", payload, timeout=http_timeout)
            train_time_s += float(time.time() - t_loc0)

            if resp.status_code != 200:
                continue

            data = resp.json()
            if "state_dict" not in data:
                continue

            sd = jsonable_to_state_dict(data["state_dict"], device=device)
            local_sds.append(sd)

            ns = float(data.get("n_samples", 1.0))
            local_wts.append(float(max(1.0, ns)))
            contrib_idx.append(i)

        # -----------------------------
        # Aggregate (UNCHANGED)
        # -----------------------------
        if local_sds:
            if args.strategy == "tr_dp_afrl":
                agg_w = np.array([float(w[i]) for i in contrib_idx], dtype=np.float64)
                if (not np.all(np.isfinite(agg_w))) or (agg_w.sum() <= 1e-12):
                    agg_w = np.array(local_wts, dtype=np.float64)
                agg_w = agg_w / (agg_w.sum() + 1e-12)

                sw = np.array(local_wts, dtype=np.float64)
                sw = sw / (sw.sum() + 1e-12)
                eta = float(args.trdp_eta)
                agg_w = eta * agg_w + (1.0 - eta) * sw
                agg_w = agg_w / (agg_w.sum() + 1e-12)
                agg_w = agg_w.tolist()
            else:
                ssum = sum(local_wts) if sum(local_wts) > 0 else 1.0
                agg_w = [x / ssum for x in local_wts]

            new_sd = weighted_average_state_dict(local_sds, agg_w)
            model.load_state_dict(new_sd, strict=False)

        # -----------------------------
        # Evaluate + reward (UNCHANGED)
        # -----------------------------
        va = eval_model(model, val_loader, device=device, max_batches=args.eval_max_batches)
        acc = float(va.acc)

        latency_s = float(time.time() - t_round_start)

        lat_norm.update(latency_s)
        lat_n = lat_norm.norm(latency_s, fallback_scale=args.latency_norm)

        off_n = float(offload_intensity)

        if not (0.0 <= lat_n <= 1.0):
            raise RuntimeError(
                f"lat_n out of [0,1]: lat_n={lat_n} latency_s={latency_s} min={lat_norm.min_v} max={lat_norm.max_v}"
            )

        reward = compute_reward(acc, lat_n, off_n, churn_ratio, rw)

        reward_sel = reward
        if args.strategy == "tr_dp_afrl":
            sel_cnt = int(selected.sum())
            if sel_cnt < int(args.K):
                reward_sel -= float(args.trdp_starve_pen) * float(int(args.K) - sel_cnt)

        s2 = make_state_vec(mu_t, sd_t, acc, latency_s, churn_ratio, ablate_telemetry=bool(args.ablate_telemetry))

        if off_ppo is not None and args.strategy in ("tl_ppo", "tr_dp_afrl"):
            off_ppo.store(s, a, reward, s2, logp_off)
            off_ppo.update(tag="OFF")

        if sel_ppo is not None and args.strategy == "tr_dp_afrl":
            sel_ppo.store(s, g, lam_raw, reward_sel, s2, logp_sel)
            sel_ppo.update(tag="SEL")

        if dqn is not None and args.strategy in ("dqn", "ddqn"):
            dqn.store(s, a, reward, s2, done=False)
            dqn.update()

        st_prev_mu, st_prev_sd = mu_t, sd_t

        wdelta = delta_norm(sd_before, model.state_dict())

        # -----------------------------
        # Logs (standardized for your plotting)
        # -----------------------------
        logs["round"].append(t)
        logs["acc"].append(acc)
        logs["reward"].append(reward)
        logs["latency_s"].append(latency_s)
        logs["telemetry_time_s"].append(float(tel_time_s))
        logs["train_time_s"].append(float(train_time_s))
        logs["offload_intensity"].append(offload_intensity)
        logs["offload_norm"].append(off_n)
        logs["churn_l1"].append(churn_l1)
        logs["churn_ratio"].append(churn_ratio)
        logs["selected_mask"].append(selected.tolist())
        logs["weight_delta"].append(wdelta)
        logs["variant_label"].append(variant_label)
        logs["strategy"].append(args.strategy)
        logs["seed"].append(int(args.seed))

        # Fig8 heatmap (KEY your plotter expects) + extra alias
        logs["selected_clients"].append(selected_client_ids)
        logs["selected_clients_per_round"].append(selected_client_ids)

        # Fig9 overhead
        logs["telemetry_overhead_ms"].append(float(tel_time_s) * 1000.0)
        logs["telemetry_overhead_bytes"].append(int(tel_bytes))

        # Fig11 churn (alias)
        logs["churn"].append(float(churn_ratio))

        # audit validation
        logs["delta_off"].append(float(args.delta_off))
        logs["delta_sel"].append(float(args.delta_sel))
        logs["lambda_stab"].append(float(args.lambda_stab))

        # Console line
        if args.strategy == "tr_dp_afrl":
            g_mean = float(np.mean(g))
            sel_ids_idx = [int(i) for i in np.where(selected == 1)[0].tolist()]

            lam = np.asarray(lam_raw, dtype=np.float64)
            lam = np.clip(lam, 1e-12, None)
            lam = lam / lam.sum()
            H_lam = float(-np.sum(lam * np.log(lam)))

            w_max = float(w.max()) if w.size else 0.0
            eps = 1e-12
            H_w = float(-np.sum(w * np.log(w + eps)))

            print(
                f"[R{t:03d}] acc={acc:.4f} reward={reward:.4f} lat={latency_s:.2f}s(lat_n={lat_n:.3f}) "
                f"off={offload_intensity:.2f}(off_n={off_n:.3f}) churn={churn_ratio:.2f}(L1={churn_l1:.0f}) "
                f"sel={int(selected.sum())}/{n_clients} sel_ids={sel_ids_idx} "
                f"gate_mean={g_mean:.2f} lam[min/mean/max]={lam.min():.3f}/{lam.mean():.3f}/{lam.max():.3f} H(lam)={H_lam:.2f} "
                f"w_max={w_max:.2f} H(w)={H_w:.2f} Δw={wdelta:.3e} cloudN={cloud_samples} "
                f"tel_s={tel_time_s:.3f} tel_B={tel_bytes} train_s={train_time_s:.3f}"
            )
        else:
            print(
                f"[R{t:03d}] acc={acc:.4f} reward={reward:.4f} lat={latency_s:.2f}s(lat_n={lat_n:.3f}) "
                f"off={offload_intensity:.2f}(off_n={off_n:.3f}) churn={churn_ratio:.2f}(L1={churn_l1:.0f}) "
                f"sel_all={int(selected.sum())}/{n_clients} Δw={wdelta:.3e} cloudN={cloud_samples} "
                f"tel_s={tel_time_s:.3f} tel_B={tel_bytes} train_s={train_time_s:.3f}"
            )

        prev_acc, prev_lat, prev_churn = acc, latency_s, churn_ratio

    out_json = os.path.join(args.outdir, f"results_{args.strategy}_{variant_label}_seed{args.seed}.json")
    with open(out_json, "w") as f:
        json.dump(logs, f, indent=2)
    return {"out_json": out_json}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", type=str, default="fedavg",
                   choices=["fedavg", "heuristic", "contextaware", "contextaware_cnn", "dqn", "ddqn", "tl_ppo", "tr_dp_afrl"])
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--label_col", type=str, default="scenario")
    p.add_argument("--clients", type=str, required=True)
    p.add_argument("--client_ids", type=str, default=None)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="/home/ubuntu/afrl_runs")

    p.add_argument("--win", type=int, default=10)
    p.add_argument("--K", type=int, default=5)

    p.add_argument("--variant_label", type=str, default="with_telemetry",
                   help="Label for grouping results (e.g., N10_WITH / N10_NO / SEV_A_WITH).")
    p.add_argument("--min_participants", type=int, default=3)
    p.add_argument("--ablate_telemetry", action="store_true")

    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--lr_local", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=128)

    p.add_argument("--max_local_batches", type=int, default=60)
    p.add_argument("--max_local_samples", type=int, default=2500)

    p.add_argument("--warm_start_epochs", type=int, default=5)
    p.add_argument("--warm_start_lr", type=float, default=1e-3)
    p.add_argument("--warm_start_max_batches", type=int, default=250)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--eval_max_batches", type=int, default=250)

    p.add_argument("--cloud_lr", type=float, default=1e-3)
    p.add_argument("--cloud_max_batches", type=int, default=2)
    p.add_argument("--cloud_samples_per_offload", type=int, default=512)

    p.add_argument("--delta_off", type=float, default=0.02)
    p.add_argument("--delta_sel", type=float, default=0.02)
    p.add_argument("--http_timeout", type=float, default=90.0)

    p.add_argument("--alpha_acc", type=float, default=1.0)
    p.add_argument("--beta_lat", type=float, default=0.25)
    p.add_argument("--gamma_off", type=float, default=0.10)
    p.add_argument("--lambda_stab", type=float, default=0.15)

    p.add_argument("--latency_norm", type=float, default=1.0)

    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--class_map", type=str, default="config/class_map.json")
    p.add_argument("--scaler_path", type=str, default="config/scaler.json")

    p.add_argument("--impair_start_round", type=int, default=None)
    p.add_argument("--impair_end_round", type=int, default=None)
    p.add_argument("--cpu_impair_url", type=str, default=None)
    p.add_argument("--net_impair_url", type=str, default=None)
    p.add_argument("--net_delay_ms", type=float, default=120.0)
    p.add_argument("--net_loss_pct", type=float, default=5.0)
    p.add_argument("--cpu_workers", type=int, default=1)

    p.add_argument("--trdp_eta", type=float, default=0.6)
    p.add_argument("--trdp_wmin", type=float, default=0.02)
    p.add_argument("--trdp_wmax", type=float, default=0.70)
    p.add_argument("--trdp_starve_pen", type=float, default=0.10)

    # in afrl_idea1_run_one_strategy.py -> build_parser()
    p.add_argument("--log_selection", action="store_true")
    p.add_argument("--log_overhead", action="store_true")
    p.add_argument("--log_churn", action="store_true")
    p.add_argument("--log_kl", action="store_true")

    return p


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
