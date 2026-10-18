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
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Subset

from afrl_e1_fashion_dataset import (
    get_fashion_mnist_datasets,
    make_dirichlet_partitions,
    validate_partitions,
)

from model_utils_e1 import FashionMNISTCNN

from model_utils import (
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
def get_live_telemetry_matrix(
    clients: List[str],
    http_timeout: float,
    seed: int = None,
    round_id: int = 0,
) -> Tuple[np.ndarray, float, int]:
    """
    Returns:
      Xlive: (N,15)
      telemetry_time_s: wall time for collecting telemetry
      telemetry_bytes: bytes read from telemetry HTTP responses (best-effort)
    """
    t0 = time.time()
    total_bytes = 0
    X = []
    for i, c in enumerate(clients):
        try:
            if seed is None:
                telemetry_path = "/telemetry"
            else:
                telemetry_seed = (
                    int(seed) * 1_000_000
                    + int(round_id) * 1_000
                    + int(i)
                )
                telemetry_path = f"/telemetry?seed={telemetry_seed}"

            r = safe_get(
                c,
                telemetry_path,
                timeout=min(http_timeout, 10.0),
            )
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
    # State = telemetry aggregate + learning feedback.
    # Telemetry ablation removes only telemetry features; learning
    # feedback (accuracy, latency, churn) remains available.
    extra = np.array(
        [acc_prev, lat_prev, churn_prev],
        dtype=np.float32,
    )
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
    # FedAvg baseline: all selected clients train locally.
    # No cloud offloading.
    if strategy == "fedavg":
        return np.zeros((n,), dtype=np.int64)

    # Heuristic/context-aware baselines:
    # offload clients with relatively high RTT or packet loss.
    if strategy in ("contextaware", "contextaware_cnn", "heuristic"):
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
    train_dataset,
    partitions,
    client_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], int]:

    indices = partitions.get(int(client_index), [])

    if len(indices) == 0:
        return {
            k: v.detach().cpu().clone()
            for k, v in global_model.state_dict().items()
        }, 0

    model = FashionMNISTCNN().to(device)

    model.load_state_dict(
        {
            k: v.detach().clone()
            for k, v in global_model.state_dict().items()
        },
        strict=True,
    )

    round_id = int(getattr(args, "_round_seed_offset", 0))

    rng = np.random.default_rng(
        int(args.seed) * 1_000_000
        + round_id * 1_000
        + int(client_index)
    )

    chosen = rng.choice(
        np.asarray(indices, dtype=np.int64),
        size=min(
            int(args.cloud_samples_per_offload),
            len(indices)
        ),
        replace=False,
    )

    subset = Subset(
        train_dataset,
        [int(x) for x in chosen]
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(
        int(args.seed) * 1_000_000
        + round_id * 1_000
        + int(client_index)
    )

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.cloud_lr,
        weight_decay=args.weight_decay,
    )

    train_stats = train_one_epoch_budget(
        model,
        loader,
        opt,
        device,
        max_batches=args.cloud_max_batches,
    )

    # Aggregation weight must reflect the amount of data actually
    # processed under the cloud batch budget, not merely the number
    # of samples placed in the candidate subset.
    n_trained = min(
        int(len(chosen)),
        int(train_stats.batches) * int(args.batch_size),
    )

    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }, int(n_trained)


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

    # =====================================================
    # E1 independent FL task: Fashion-MNIST
    # =====================================================
    # Official TRAIN split is partitioned among FL clients.
    # Official TEST split is reserved for global evaluation.
    # Controller telemetry remains separate and unchanged.

    fashion_train, fashion_test = get_fashion_mnist_datasets()

    partitions = make_dirichlet_partitions(
        train_dataset=fashion_train,
        num_clients=n_clients,
        alpha=float(args.dirichlet_alpha),
        seed=int(args.seed),
    )

    validate_partitions(
        train_dataset=fashion_train,
        test_dataset=fashion_test,
        partitions=partitions,
    )

    if len(partitions) != n_clients:
        raise RuntimeError(
            f"Partition/client mismatch: "
            f"{len(partitions)} partitions vs {n_clients} clients"
        )

    test_loader = DataLoader(
        fashion_test,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(
        f"[E1_DATA] FashionMNIST "
        f"train={len(fashion_train)} "
        f"test={len(fashion_test)} "
        f"clients={n_clients} "
        f"alpha={args.dirichlet_alpha} "
        f"seed={args.seed}"
    )

    print(
        "[E1_PARTITIONS] "
        + ", ".join(
            f"client{i+1}={len(partitions[i])}"
            for i in range(n_clients)
        )
    )

    device = get_device(force_cpu=True)
    model = FashionMNISTCNN().to(device)

    # No server-side warm start in E1.
    # Federated learning begins from the initial global model.
    print("[E1_WARM_START] disabled")

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
        "kl_off", "kl_sel", "kl_gate", "kl_dir",
        "kl_off_max", "kl_sel_max",
        "ppo_off_updated", "ppo_sel_updated",
        "ppo_off_early_stop", "ppo_sel_early_stop",
        "cloud_samples", "selected_count",

        # E5 independent systems measurements
        "selection_decision_time_s",
        "offload_decision_time_s",
        "controller_decision_time_s",

        # E5 predeclared round-deadline sensitivity metrics.
        "sla_violation_10s",
        "sla_violation_12s",
        "sla_violation_15s",
        "cumulative_sla_violation_rate_10s",
        "cumulative_sla_violation_rate_12s",
        "cumulative_sla_violation_rate_15s",

        "physical_phase_wall_s",
        "physical_request_sum_s",
        "physical_request_max_s",
        "client_compute_sum_s",
        "client_compute_max_s",
        "client_compute_time_by_id",
        "cloud_compute_s",
        "aggregation_time_s",
        "evaluation_time_s",
        "model_payload_bytes",
        "physical_download_bytes",
        "physical_upload_bytes",
        "physical_model_comm_bytes",
        "communication_bytes_total",
        "cumulative_communication_bytes",
        "cumulative_wall_time_s",
    ]}

    # E5 cumulative external systems counters.
    e5_cumulative_wall_s = 0.0
    e5_cumulative_comm_bytes = 0

    # E5 evaluation-only round-deadline counters.
    # These thresholds are fixed before the final E5 experiment.
    e5_sla_violations_10s = 0
    e5_sla_violations_12s = 0
    e5_sla_violations_15s = 0

    # churn baseline
    if args.strategy == "tr_dp_afrl":
        prev_selected = np.zeros(n_clients, dtype=np.int64)
    else:
        prev_selected = np.ones(n_clients, dtype=np.int64)

    prev_acc, prev_lat, prev_churn = 0.0, 0.0, 0.0

    if bool(args.ablate_telemetry):
        X0 = np.zeros((n_clients, 15), dtype=np.float32)
        _t0, _b0 = 0.0, 0
    else:
        X0, _t0, _b0 = get_live_telemetry_matrix(
            clients,
            http_timeout=http_timeout,
            seed=int(args.seed),
            round_id=0,
        )
    mu0, sd0 = agg_telemetry_stats(X0, ablate_telemetry=bool(args.ablate_telemetry))
    st_prev_mu, st_prev_sd = mu0, sd0

    lat_norm = RunningMinMax()
    off_norm = RunningMinMax()

    args._round_seed_offset = 0

    for t in range(1, int(args.rounds) + 1):
        args._round_seed_offset = t
        maybe_apply_impairments(t, args, http_timeout=http_timeout)

        t_round_start = time.time()

        if bool(args.ablate_telemetry):
            Xlive = np.zeros((n_clients, 15), dtype=np.float32)
            tel_time_s, tel_bytes = 0.0, 0
        else:
            Xlive, tel_time_s, tel_bytes = get_live_telemetry_matrix(
                clients,
                http_timeout=http_timeout,
                seed=int(args.seed),
                round_id=int(t),
            )
        mu_t, sd_t = agg_telemetry_stats(Xlive, ablate_telemetry=bool(args.ablate_telemetry))

        s = make_state_vec(
            st_prev_mu, st_prev_sd,
            prev_acc, prev_lat, prev_churn,
            ablate_telemetry=bool(args.ablate_telemetry),
        )

        # -----------------------------
        # Selection (Algorithm 2)  (UNCHANGED)
        # -----------------------------
        # E5 measurement only: complete selection-decision latency,
        # including policy inference and deterministic post-processing.
        t_selection_decision0 = time.perf_counter()

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

        selection_decision_time_s = float(
            time.perf_counter() - t_selection_decision0
        )

        churn_l1 = float(np.sum(np.abs(selected - prev_selected)))
        churn_ratio = float(churn_l1) / float(n_clients)
        prev_selected = selected.copy()

        selected_client_ids = [client_ids[i] for i in np.where(selected == 1)[0].tolist()]

        # -----------------------------
        # Offloading action a_t (Algorithm 1/2) (UNCHANGED)
        # -----------------------------
        # E5 measurement only: complete offload-decision latency,
        # including policy/baseline inference, action conversion,
        # and the minimum-local-training execution projection.
        t_offload_decision0 = time.perf_counter()

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

        # Preserve the raw PPO action associated with logp_off.
        # The projected action below is used for actual execution.
        if args.strategy in ("tl_ppo", "tr_dp_afrl"):
            a_sampled = a.copy()
        else:
            a_sampled = None

        a = enforce_min_local_training(
            a=a,
            selected=selected,
            Xlive=Xlive,
            min_participants=int(args.min_participants),
        )

        offload_decision_time_s = float(
            time.perf_counter() - t_offload_decision0
        )

        controller_decision_time_s = float(
            selection_decision_time_s
            + offload_decision_time_s
        )

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

        # E5 measurement-only systems counters.
        physical_request_times = []
        client_compute_times = []

        # E5 measurement only:
        # physical-client compute duration by client ID.
        client_compute_time_by_id = {}

        physical_request_body_bytes = []
        physical_response_body_bytes = []
        cloud_compute_s = 0.0
        aggregation_time_s = 0.0
        evaluation_time_s = 0.0
        physical_phase_wall_s = 0.0

        # Freeze the round-start global model once.
        # Every selected physical client receives exactly the same model.
        global_state_json = state_dict_to_jsonable(model.state_dict())

        # -------------------------------------------------
        # E1 parallel physical-client execution
        #
        # Server-side cloud pseudo-clients remain sequential.
        # Physical /train_local requests run concurrently.
        # Results are stored by client index and later inserted
        # in ascending client order so aggregation semantics
        # remain identical to the original implementation.
        # -------------------------------------------------
        physical_jobs = []
        physical_results = {}

        for i, url in enumerate(clients):
            if selected[i] == 0:
                continue

            # TL-PPO / TR-DP-AFRL true cloud pseudo-client.
            if int(a[i]) == 1 and args.strategy in ("tl_ppo", "tr_dp_afrl"):
                t_cloud0 = time.time()

                sd_cloud, ns_cloud = cloud_pseudo_client_update(
                    model,
                    fashion_train,
                    partitions,
                    i,
                    args,
                    device=device,
                )

                cloud_elapsed = float(time.time() - t_cloud0)
                train_time_s += cloud_elapsed
                cloud_compute_s += cloud_elapsed
                cloud_samples += int(ns_cloud)

                physical_results[i] = (
                    sd_cloud,
                    float(max(1, ns_cloud)),
                    0.0,
                )
                continue

            payload = {
                "round": t,
                "local_epochs": int(args.local_epochs),
                "lr": float(args.lr_local),
                "weight_decay": float(args.weight_decay),
                "batch_size": int(args.batch_size),
                "offload": int(a[i]),
                "global_state_dict": global_state_json,
                "max_local_batches": int(args.max_local_batches),
                "max_local_samples": int(args.max_local_samples),

                # E1 Fashion-MNIST partition configuration.
                "data_seed": int(args.seed),
                "dirichlet_alpha": float(args.dirichlet_alpha),
                "num_clients": int(args.num_clients),
            }

            physical_jobs.append((i, url, payload))

        def _train_physical_client(job):
            i, url, payload = job
            t0 = time.time()

            try:
                resp = safe_post(
                    url,
                    "/train_local",
                    payload,
                    timeout=http_timeout,
                )

                elapsed = float(time.time() - t0)

                if resp.status_code != 200:
                    return i, None, None, elapsed, None, 0, 0

                data = resp.json()

                if "state_dict" not in data:
                    return i, None, None, elapsed, None, 0, 0

                sd = jsonable_to_state_dict(
                    data["state_dict"],
                    device=device,
                )

                ns = float(data.get("n_samples", 1.0))
                client_compute_s = float(
                    data.get("train_time_s", 0.0)
                )

                # E5 application-layer HTTP body measurement.
                # The request has already been prepared and sent.
                request_body = resp.request.body

                if request_body is None:
                    request_body_bytes = 0
                elif isinstance(request_body, bytes):
                    request_body_bytes = len(request_body)
                else:
                    request_body_bytes = len(
                        str(request_body).encode("utf-8")
                    )

                # requests has already materialized the response body
                # before resp.json() above, so this records its actual
                # application-layer body size.
                response_body_bytes = len(resp.content)

                return (
                    i,
                    sd,
                    float(max(1.0, ns)),
                    elapsed,
                    client_compute_s,
                    int(request_body_bytes),
                    int(response_body_bytes),
                )

            except Exception as exc:
                elapsed = float(time.time() - t0)
                print(
                    f"[CLIENT_TRAIN_ERROR] "
                    f"client={i+1} error={exc}"
                )
                return i, None, None, elapsed, None, 0, 0

        if physical_jobs:
            max_workers = max(
                1,
                min(
                    int(args.max_parallel_clients),
                    len(physical_jobs),
                ),
            )

            t_physical_phase0 = time.time()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_train_physical_client, job)
                    for job in physical_jobs
                ]

                for future in as_completed(futures):
                    (
                        i,
                        sd,
                        ns,
                        elapsed,
                        client_compute_s,
                        request_body_bytes,
                        response_body_bytes,
                    ) = future.result()

                    # Preserve previous train_time_s semantics:
                    # sum of individual client execution durations.
                    train_time_s += float(elapsed)

                    # E5 measurement only.
                    physical_request_times.append(float(elapsed))

                    physical_request_body_bytes.append(
                        int(request_body_bytes)
                    )
                    physical_response_body_bytes.append(
                        int(response_body_bytes)
                    )

                    if client_compute_s is not None:
                        client_compute_times.append(
                            float(client_compute_s)
                        )

                        # E5 measurement only. Preserve the
                        # heterogeneous physical-client identity.
                        client_compute_time_by_id[
                            str(client_ids[i])
                        ] = float(client_compute_s)

                    if sd is None:
                        continue

                    physical_results[i] = (
                        sd,
                        float(ns),
                        elapsed,
                    )

            physical_phase_wall_s = float(
                time.time() - t_physical_phase0
            )

        # Restore deterministic client-index ordering before aggregation.
        for i in sorted(physical_results):
            sd, ns, _ = physical_results[i]

            local_sds.append(sd)
            local_wts.append(float(ns))
            contrib_idx.append(i)

        # -----------------------------
        # Aggregate (UNCHANGED)
        # -----------------------------

        # E5 measurement only.
        t_aggregation0 = time.time()

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

        # E5 measurement only.
        aggregation_time_s = float(
            time.time() - t_aggregation0
        )

        # -----------------------------
        # Evaluate + reward (UNCHANGED)
        # -----------------------------
        # E1 global evaluation:
        # official Fashion-MNIST TEST split only.

        # E5 measurement only.
        t_evaluation0 = time.time()

        va = eval_model(
            model,
            test_loader,
            device=device,
            max_batches=args.eval_max_batches,
        )

        evaluation_time_s = float(
            time.time() - t_evaluation0
        )

        acc = float(va.acc)

        latency_s = float(time.time() - t_round_start)

        # -------------------------------------------------
        # E5 evaluation-only round-deadline sensitivity.
        #
        # These fixed thresholds are not part of the RL
        # reward, state, action selection, or policy update.
        # They measure whether the completed FL round
        # exceeded each predeclared wall-clock deadline.
        # -------------------------------------------------
        sla_violation_10s = int(latency_s > 10.0)
        sla_violation_12s = int(latency_s > 12.0)
        sla_violation_15s = int(latency_s > 15.0)

        e5_sla_violations_10s += sla_violation_10s
        e5_sla_violations_12s += sla_violation_12s
        e5_sla_violations_15s += sla_violation_15s

        cumulative_sla_violation_rate_10s = float(
            e5_sla_violations_10s / float(t)
        )
        cumulative_sla_violation_rate_12s = float(
            e5_sla_violations_12s / float(t)
        )
        cumulative_sla_violation_rate_15s = float(
            e5_sla_violations_15s / float(t)
        )

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
            if a_sampled is None:
                raise RuntimeError("Missing sampled PPO offload action")
            off_ppo.store(s, a_sampled, reward, s2, logp_off)
            off_ppo.update(tag="OFF")

        if sel_ppo is not None and args.strategy == "tr_dp_afrl":
            sel_ppo.store(s, g, lam_raw, reward_sel, s2, logp_sel)
            sel_ppo.update(tag="SEL")

        if dqn is not None and args.strategy in ("dqn", "ddqn"):
            dqn.store(s, a, reward, s2, done=False)
            dqn.update()

        st_prev_mu, st_prev_sd = mu_t, sd_t

        wdelta = delta_norm(sd_before, model.state_dict())

        # -------------------------------------------------
        # E5 independent systems accounting.
        #
        # These quantities are calculated only after the
        # original latency/reward/policy update path, so they
        # do not enter the RL reward or policy state.
        #
        # Communication values below are application-level
        # model-payload accounting, not packet-level traffic.
        # -------------------------------------------------
        physical_request_sum_s = float(
            sum(physical_request_times)
        )

        physical_request_max_s = float(
            max(physical_request_times)
            if physical_request_times
            else 0.0
        )

        client_compute_sum_s = float(
            sum(client_compute_times)
        )

        client_compute_max_s = float(
            max(client_compute_times)
            if client_compute_times
            else 0.0
        )

        # E5 measurement only.
        # Calculate payload size after the latency/reward/PPO-critical
        # path so instrumentation does not perturb the measured round
        # latency or the learning signal.
        model_payload_bytes = len(
            json.dumps(
                global_state_json,
                separators=(",", ":"),
            ).encode("utf-8")
        )

        # E5 measured application-layer HTTP body communication
        # for physical /train_local exchanges.
        #
        # physical_download_bytes:
        #   server -> physical clients (request bodies)
        #
        # physical_upload_bytes:
        #   physical clients -> server (response bodies)
        #
        # This excludes HTTP/TCP/IP headers and packet overhead.
        physical_download_bytes = int(
            sum(physical_request_body_bytes)
        )

        physical_upload_bytes = int(
            sum(physical_response_body_bytes)
        )

        physical_model_comm_bytes = int(
            physical_download_bytes
            + physical_upload_bytes
        )

        # Telemetry payload accounting already exists in
        # tel_bytes. Combine it with model payload accounting.
        communication_bytes_total = int(
            physical_model_comm_bytes
            + int(tel_bytes)
        )

        e5_cumulative_wall_s += float(latency_s)
        e5_cumulative_comm_bytes += int(
            communication_bytes_total
        )

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

        def finite_or_none(x):
            try:
                v = float(x)
                return v if np.isfinite(v) else None
            except (TypeError, ValueError):
                return None

        logs["kl_off"].append(
            finite_or_none(off_ppo.last_kl)
            if off_ppo is not None
            else None
        )

        logs["kl_sel"].append(
            finite_or_none(sel_ppo.last_kl_total)
            if sel_ppo is not None
            else None
        )

        logs["kl_gate"].append(
            finite_or_none(sel_ppo.last_kl_gate)
            if sel_ppo is not None
            else None
        )

        logs["kl_dir"].append(
            finite_or_none(sel_ppo.last_kl_dir)
            if sel_ppo is not None
            else None
        )

        logs["kl_off_max"].append(
            finite_or_none(off_ppo.last_kl_max)
            if off_ppo is not None
            else None
        )

        logs["kl_sel_max"].append(
            finite_or_none(sel_ppo.last_kl_max)
            if sel_ppo is not None
            else None
        )

        logs["ppo_off_updated"].append(
            bool(off_ppo.last_updated)
            if off_ppo is not None
            else False
        )

        logs["ppo_sel_updated"].append(
            bool(sel_ppo.last_updated)
            if sel_ppo is not None
            else False
        )

        logs["ppo_off_early_stop"].append(
            bool(off_ppo.last_early_stop)
            if off_ppo is not None
            else False
        )

        logs["ppo_sel_early_stop"].append(
            bool(sel_ppo.last_early_stop)
            if sel_ppo is not None
            else False
        )

        logs["cloud_samples"].append(int(cloud_samples))
        logs["selected_count"].append(int(selected.sum()))
        logs["lambda_stab"].append(float(args.lambda_stab))

        # E5 independent systems measurements.
        logs["selection_decision_time_s"].append(
            float(selection_decision_time_s)
        )
        logs["offload_decision_time_s"].append(
            float(offload_decision_time_s)
        )
        logs["controller_decision_time_s"].append(
            float(controller_decision_time_s)
        )
        logs["sla_violation_10s"].append(
            int(sla_violation_10s)
        )
        logs["sla_violation_12s"].append(
            int(sla_violation_12s)
        )
        logs["sla_violation_15s"].append(
            int(sla_violation_15s)
        )

        logs["cumulative_sla_violation_rate_10s"].append(
            float(cumulative_sla_violation_rate_10s)
        )
        logs["cumulative_sla_violation_rate_12s"].append(
            float(cumulative_sla_violation_rate_12s)
        )
        logs["cumulative_sla_violation_rate_15s"].append(
            float(cumulative_sla_violation_rate_15s)
        )

        logs["physical_phase_wall_s"].append(
            float(physical_phase_wall_s)
        )
        logs["physical_request_sum_s"].append(
            float(physical_request_sum_s)
        )
        logs["physical_request_max_s"].append(
            float(physical_request_max_s)
        )
        logs["client_compute_sum_s"].append(
            float(client_compute_sum_s)
        )
        logs["client_compute_max_s"].append(
            float(client_compute_max_s)
        )

        logs["client_compute_time_by_id"].append(
            dict(client_compute_time_by_id)
        )

        logs["cloud_compute_s"].append(
            float(cloud_compute_s)
        )
        logs["aggregation_time_s"].append(
            float(aggregation_time_s)
        )
        logs["evaluation_time_s"].append(
            float(evaluation_time_s)
        )
        logs["model_payload_bytes"].append(
            int(model_payload_bytes)
        )
        logs["physical_download_bytes"].append(
            int(physical_download_bytes)
        )
        logs["physical_upload_bytes"].append(
            int(physical_upload_bytes)
        )
        logs["physical_model_comm_bytes"].append(
            int(physical_model_comm_bytes)
        )
        logs["communication_bytes_total"].append(
            int(communication_bytes_total)
        )
        logs["cumulative_communication_bytes"].append(
            int(e5_cumulative_comm_bytes)
        )
        logs["cumulative_wall_time_s"].append(
            float(e5_cumulative_wall_s)
        )

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

        # Atomic per-round checkpoint
        out_json = os.path.join(
            args.outdir,
            f"results_{args.strategy}_{variant_label}_seed{args.seed}.json"
        )
        tmp_json = out_json + ".tmp"

        with open(tmp_json, "w") as f:
            json.dump(logs, f, indent=2)

        os.replace(tmp_json, out_json)

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

    # E1 independent Fashion-MNIST benchmark
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--dirichlet_alpha", type=float, default=0.5)
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
    p.add_argument(
        "--max_parallel_clients",
        type=int,
        default=10,
        help="Maximum concurrent physical client /train_local requests.",
    )

    p.add_argument("--alpha_acc", type=float, default=1.0)
    p.add_argument("--beta_lat", type=float, default=0.25)
    p.add_argument("--gamma_off", type=float, default=0.10)
    p.add_argument("--lambda_stab", type=float, default=0.15)

    p.add_argument("--latency_norm", type=float, default=1.0)

    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--class_map", type=str, default="/home/ubuntu/class_map.json")
    p.add_argument("--scaler_path", type=str, default="/home/ubuntu/scaler.json")

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
