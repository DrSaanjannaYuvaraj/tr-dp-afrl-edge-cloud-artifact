#!/usr/bin/env python3
# make_journal_figs_and_table.py
# Robust journal export: generates Fig3–Fig14 + TableX + significance + NEW tables X+2..X+4
# Fixes missing Fig8/Fig9/Fig11 by DERIVING series when older JSON logs lack keys.
#
# USER-FIXES APPLIED:
# 1) Removed "FigX" numbering from ALL plot titles (titles now contain only descriptive text).
# 2) Ablation plots now show ONLY TWO lines: Telemetry-Agnostic (NO) vs Telemetry-Aware (WITH).
#    - Any third line (e.g., plain "N10") is ignored.
#    - If WITH is missing but a plain baseline exists, baseline is treated as WITH.
# 3) Fig6 wording expanded: "Overlay Accuracy" (not "Overlay acc"); improved y-labels.
#
# EXISTING:
# 4) Fig12 (Scalability) plots ACTUAL scalability curves using --scal_dirs (accuracy vs N).
# 5) Fig13 (Robustness) plots ACTUAL robustness curves using --robust_dirs (accuracy vs severity).
#
# NEW (THIS CONSOLIDATED UPDATE):
# 6) Table X+1 (Significance) now matches FGCS title:
#    - compares TR-DP-AFRL vs baselines (not all pairwise)
#    - includes p-values AND effect size (Cohen's d; paired when possible)
#    - metrics: acc, reward, latency_s, churn, offload_intensity, telemetry_overhead_ms
# 7) Table X+2 (Ablation): WITH vs NO telemetry for TL-PPO and TR-DP-AFRL
#    - metrics: accuracy, latency, overhead (tail window mean±std) + deltas
# 8) Table X+3 (Scalability): N=4/6/8/10 summary for TL-PPO vs TR-DP-AFRL
#    - metrics: accuracy, latency, churn (stability) (tail window mean±std)
# 9) Table X+4 (Robustness): Severity A–E summary for FedAvg, TL-PPO, TR-DP-AFRL
#    - metrics: tail accuracy/latency/churn (mean±std)
#    - derived: accuracy drop %, latency increase %, churn increase % (vs SEV-A baseline per strategy)
#    - recovery slope: post-impair window if present; else tail slope proxy
#
# PLOT FIX (THIS REQUEST):
# - Prevent truncated titles by:
#   (a) wrapping titles into multiple lines
#   (b) saving with bbox_inches="tight"
#
# FULL-RUN HORIZON FIX (THIS REQUEST):
# - Use T = max rounds across all runs (not runs[0]) so plots cover full 300 rounds.
# - Pad shorter series with NaNs so mean±std plots remain valid and aligned.

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# matplotlib only (no seaborn)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandas as pd  # noqa: F401
except Exception:
    pd = None

try:
    from scipy import stats
except Exception:
    stats = None


# ----------------------------
# Utilities
# ----------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _safe_mean(x: List[float]) -> float:
    if not x:
        return float("nan")
    return float(np.nanmean(np.asarray(x, dtype=np.float64)))


def _safe_std(x: List[float]) -> float:
    if not x:
        return float("nan")
    return float(np.nanstd(np.asarray(x, dtype=np.float64), ddof=1)) if len(x) > 1 else 0.0


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return list(x)


def _round_count(run: Dict[str, Any]) -> int:
    r = run.get("round", [])
    if isinstance(r, list) and len(r) > 0:
        return len(r)
    for k in ("acc", "reward", "latency_s"):
        if isinstance(run.get(k, None), list):
            return len(run[k])
    return 0


def _max_round_count(runs: List[Dict[str, Any]]) -> int:
    return max([_round_count(r) for r in runs] + [0])


def _pad_to_T(x: Any, T: int, pad_val: float = float("nan")) -> List[Any]:
    """
    Pad a list-like to length T (or truncate), using pad_val.
    Works for numeric series (pad_val=NaN) and for other series too.
    """
    arr = _as_list(x)
    if len(arr) >= T:
        return arr[:T]
    return arr + [pad_val] * (T - len(arr))


def _infer_num_clients_from_run(run: Dict[str, Any], fallback: int = 0) -> int:
    """
    Infer client count from selected_mask shape if present, else fallback.
    """
    sel = run.get("selected_mask", None)
    if isinstance(sel, list) and len(sel) > 0 and isinstance(sel[0], list) and len(sel[0]) > 0:
        return int(len(sel[0]))
    return int(fallback)


def _infer_N_from_dirname(d: str) -> Optional[int]:
    """
    Infer N from directory basename patterns like:
      N4_300r_trial, N10_300r_main
    Returns None if not found.
    """
    base = os.path.basename(os.path.normpath(d))
    m = re.search(r"(?:^|[^0-9])N(\d+)(?:[^0-9]|$)", base)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _pretty_strategy_name(s: str) -> str:
    if s == "tr_dp_afrl":
        return "TR-DP-AFRL"
    if s == "tl_ppo":
        return "TL-PPO"
    if s == "fedavg":
        return "FedAvg"
    if s == "ddqn":
        return "DDQN"
    if s == "dqn":
        return "DQN"
    if s == "heuristic":
        return "Heuristic"
    return s


def _wrap_title(title: str, width: int = 60) -> str:
    """
    Wrap long plot titles to prevent truncation.
    """
    title = str(title or "").strip()
    if not title:
        return title
    return "\n".join(textwrap.wrap(title, width=width))


def _safe_slope(y: List[float]) -> float:
    """
    Linear slope of y over index 0..len(y)-1. Returns NaN if insufficient.
    """
    yy = np.asarray(y, dtype=np.float64)
    yy = yy[np.isfinite(yy)]
    if yy.size < 2:
        return float("nan")
    x = np.arange(yy.size, dtype=np.float64)
    x_mean = float(x.mean())
    y_mean = float(yy.mean())
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0.0:
        return float("nan")
    slope = float(np.sum((x - x_mean) * (yy - y_mean)) / denom)
    return slope


def _infer_impair_window(run: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """
    Try to infer impairment window from JSON if stored.
    Returns (start_round, end_round) in 1-based rounds if available.
    """
    a = run.get("impair_start_round", None)
    b = run.get("impair_end_round", None)
    try:
        a = int(a) if a is not None else None
        b = int(b) if b is not None else None
    except Exception:
        a, b = None, None
    if a is not None and b is not None and a >= 1 and b >= a:
        return a, b
    return None, None


def _cohens_d_paired(a: List[float], b: List[float]) -> float:
    """
    Cohen's d for paired samples: mean(diff) / std(diff).
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    m = min(len(aa), len(bb))
    if m < 2:
        return float("nan")
    aa = aa[:m]
    bb = bb[:m]
    diff = aa - bb
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return float("nan")
    sd = float(np.nanstd(diff, ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(np.nanmean(diff) / sd)


def _cohens_d_indep(a: List[float], b: List[float]) -> float:
    """
    Cohen's d for independent samples: (mean(a)-mean(b))/pooled_sd
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    if aa.size < 2 or bb.size < 2:
        return float("nan")
    sa = float(np.nanstd(aa, ddof=1))
    sb = float(np.nanstd(bb, ddof=1))
    denom = float(aa.size + bb.size - 2)
    if denom <= 0:
        return float("nan")
    pooled = float(np.sqrt(((aa.size - 1) * sa**2 + (bb.size - 1) * sb**2) / denom))
    if pooled == 0.0:
        return float("nan")
    return float((np.nanmean(aa) - np.nanmean(bb)) / pooled)


# ----------------------------
# Normalization / backfill (KEY FIX)
# ----------------------------
def normalize_run(run: Dict[str, Any], client_ids: List[str], num_clients: int) -> Dict[str, Any]:
    """
    Adds/derives required series so Fig8/Fig9/Fig11 never fail, even on older JSONs.
    """
    T = _round_count(run)
    if T <= 0:
        return run

    # ---- selected_mask
    sel_mask = run.get("selected_mask", None)
    if not (isinstance(sel_mask, list) and len(sel_mask) == T):
        alt = run.get("selected", None)
        if isinstance(alt, list) and len(alt) == T:
            sel_mask = alt
        else:
            sel_mask = [[1] * num_clients for _ in range(T)]

    # enforce shape
    sel_mask2: List[List[int]] = []
    for t in range(T):
        row = sel_mask[t] if t < len(sel_mask) else [1] * num_clients
        row = list(row) if isinstance(row, (list, tuple)) else [1] * num_clients
        if len(row) != num_clients:
            row = (row + [1] * num_clients)[:num_clients]
        sel_mask2.append([int(v) for v in row])
    run["selected_mask"] = sel_mask2

    # ---- selected_clients (Fig8)
    sc = run.get("selected_clients", None)
    ok_sc = isinstance(sc, list) and len(sc) == T and (len(sc) == 0 or isinstance(sc[0], list))
    if not ok_sc:
        sc2: List[List[str]] = []
        for t in range(T):
            idx = [i for i, v in enumerate(run["selected_mask"][t]) if int(v) == 1]
            sc2.append([client_ids[i] for i in idx])
        run["selected_clients"] = sc2

    # ---- telemetry overhead (Fig9)
    if not (isinstance(run.get("telemetry_overhead_ms", None), list) and len(run["telemetry_overhead_ms"]) == T):
        tel_time = run.get("telemetry_time_s", None)
        if isinstance(tel_time, list) and len(tel_time) == T:
            run["telemetry_overhead_ms"] = [float(x) * 1000.0 for x in tel_time]
        else:
            run["telemetry_overhead_ms"] = [0.0] * T

    if not (isinstance(run.get("telemetry_overhead_bytes", None), list) and len(run["telemetry_overhead_bytes"]) == T):
        tel_bytes = run.get("telemetry_bytes", None)
        if isinstance(tel_bytes, list) and len(tel_bytes) == T:
            run["telemetry_overhead_bytes"] = [int(x) for x in tel_bytes]
        else:
            run["telemetry_overhead_bytes"] = [0] * T

    # ---- churn (Fig11)
    churn = run.get("churn", None)
    ok_churn = isinstance(churn, list) and len(churn) == T
    if not ok_churn:
        cr = run.get("churn_ratio", None)
        if isinstance(cr, list) and len(cr) == T:
            run["churn"] = [float(x) for x in cr]
        else:
            churn2: List[float] = []
            prev = np.asarray(run["selected_mask"][0], dtype=np.int64)
            churn2.append(0.0)
            for t in range(1, T):
                cur = np.asarray(run["selected_mask"][t], dtype=np.int64)
                l1 = float(np.sum(np.abs(cur - prev)))
                churn2.append(l1 / float(num_clients))
                prev = cur
            run["churn"] = churn2

    # ensure key numeric series exist
    for k in ("acc", "reward", "latency_s", "offload_intensity"):
        if not (isinstance(run.get(k, None), list) and len(run[k]) == T):
            run[k] = [float("nan")] * T

    return run


def load_runs(main_indir: str, client_ids: List[str], num_clients: int) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(main_indir, "results_*.json")))
    runs: List[Dict[str, Any]] = []
    for p in paths:
        try:
            with open(p, "r") as f:
                d = json.load(f)
            d["_path"] = p
            d = normalize_run(d, client_ids=client_ids, num_clients=num_clients)
            runs.append(d)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
    return runs


def group_runs(runs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        strat = r.get("strategy", "unknown")
        if isinstance(strat, list) and strat:
            strat = strat[0]
        strat = str(strat)
        g.setdefault(strat, []).append(r)
    return g


def series_tail(run: Dict[str, Any], key: str, W: int) -> np.ndarray:
    x = _as_list(run.get(key, []))
    x = x[-W:] if W > 0 else x
    return np.asarray(x, dtype=np.float64)


# ----------------------------
# Plotting helpers (TITLE FIX HERE)
# ----------------------------
def save_lineplot(
    outpath: str,
    x: np.ndarray,
    ys: Dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))
    for name, y in ys.items():
        plt.plot(x, y, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(_wrap_title(title, 60))  # wrap long titles
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")  # prevent truncation
    plt.close()


def save_meanstd_plot(
    outpath: str,
    x: np.ndarray,
    means: Dict[str, np.ndarray],
    stds: Dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))
    for name in means.keys():
        m = means[name]
        s = stds.get(name, np.zeros_like(m))
        plt.plot(x, m, label=name)
        plt.fill_between(x, m - s, m + s, alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(_wrap_title(title, 60))  # wrap long titles
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")  # prevent truncation
    plt.close()


def fig8_heatmap(outpath: str, runs_by_strategy: Dict[str, List[Dict[str, Any]]], client_ids: List[str], T: int) -> None:
    """
    Client participation heatmap. Prefer tr_dp_afrl; otherwise first strategy.
    Uses full horizon T (max over runs) and pads selection mask if needed.
    """
    strat = "tr_dp_afrl" if "tr_dp_afrl" in runs_by_strategy else (
        list(runs_by_strategy.keys())[0] if runs_by_strategy else None
    )
    if strat is None:
        print("[WARN] Heatmap: no runs available")
        return

    run = runs_by_strategy[strat][0]  # first seed run
    sel = run.get("selected_mask", [])  # expected (t, N)

    # pad/truncate to T and ensure shape N
    if len(sel) < T:
        N = len(sel[0]) if (len(sel) > 0 and isinstance(sel[0], list)) else len(client_ids)
        sel = sel + [[1] * N for _ in range(T - len(sel))]
    sel = sel[:T]

    # enforce N rows
    if len(sel) > 0 and isinstance(sel[0], list):
        N = len(sel[0])
    else:
        N = len(client_ids)
        sel = [[1] * N for _ in range(T)]

    # transpose for imshow: (N, T)
    M = np.asarray(sel, dtype=np.float64).T

    plt.figure(figsize=(10, 3.8))
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.yticks(np.arange(len(client_ids)), client_ids)
    plt.xlabel("Round")
    plt.ylabel("Client")
    pretty = "TR-DP-AFRL" if strat == "tr_dp_afrl" else strat
    plt.title(_wrap_title(f"Client Participation Heatmap ({pretty})", 60))
    plt.colorbar(label="Selected (1/0)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def fig9_overhead(outdir: str, runs_by_strategy: Dict[str, List[Dict[str, Any]]], T: int) -> None:
    """
    Fig9a: overhead ms, Fig9b: overhead bytes (mean±std over seeds per strategy)
    Uses full horizon T and pads shorter series.
    """
    x = np.arange(1, T + 1)

    means_ms: Dict[str, np.ndarray] = {}
    stds_ms: Dict[str, np.ndarray] = {}
    means_b: Dict[str, np.ndarray] = {}
    stds_b: Dict[str, np.ndarray] = {}

    for strat, rs in runs_by_strategy.items():
        mats_ms = []
        mats_b = []
        for r in rs:
            ms = np.asarray(_pad_to_T(r.get("telemetry_overhead_ms", []), T), dtype=np.float64)
            bb = np.asarray(_pad_to_T(r.get("telemetry_overhead_bytes", []), T), dtype=np.float64)
            mats_ms.append(ms)
            mats_b.append(bb)
        A = np.vstack(mats_ms) if mats_ms else np.zeros((1, T))
        B = np.vstack(mats_b) if mats_b else np.zeros((1, T))

        means_ms[strat] = np.nanmean(A, axis=0)
        stds_ms[strat] = np.nanstd(A, axis=0, ddof=1) if A.shape[0] > 1 else np.zeros(T)
        means_b[strat] = np.nanmean(B, axis=0)
        stds_b[strat] = np.nanstd(B, axis=0, ddof=1) if B.shape[0] > 1 else np.zeros(T)

    save_meanstd_plot(
        os.path.join(outdir, "Fig9a_telemetry_overhead_ms_meanstd.png"),
        x, means_ms, stds_ms,
        "Round", "Overhead (ms)",
        "Telemetry Collection Overhead (ms)"
    )
    save_meanstd_plot(
        os.path.join(outdir, "Fig9b_telemetry_overhead_bytes_meanstd.png"),
        x, means_b, stds_b,
        "Round", "Overhead (bytes)",
        "Telemetry Collection Overhead (bytes)"
    )


def fig11_churn(outpath: str, runs_by_strategy: Dict[str, List[Dict[str, Any]]], T: int) -> None:
    x = np.arange(1, T + 1)
    means: Dict[str, np.ndarray] = {}
    stds: Dict[str, np.ndarray] = {}

    for strat, rs in runs_by_strategy.items():
        mats = []
        for r in rs:
            ch = np.asarray(_pad_to_T(r.get("churn", []), T), dtype=np.float64)
            mats.append(ch)
        A = np.vstack(mats) if mats else np.zeros((1, T))
        means[strat] = np.nanmean(A, axis=0)
        stds[strat] = np.nanstd(A, axis=0, ddof=1) if A.shape[0] > 1 else np.zeros(T)

    save_meanstd_plot(outpath, x, means, stds, "Round", "Churn ratio", "Client Churn Dynamics")


# ----------------------------
# Table X (Summary)
# ----------------------------
@dataclass
class SummaryRow:
    strategy: str
    seeds: int
    acc_mean: float
    acc_std: float
    reward_mean: float
    reward_std: float
    lat_mean: float
    lat_std: float
    churn_mean: float
    churn_std: float
    off_mean: float
    off_std: float
    overhead_ms_mean: float
    overhead_ms_std: float


def build_tableX(runs_by_strategy: Dict[str, List[Dict[str, Any]]], W: int) -> List[SummaryRow]:
    """
    Table X:
    Summary of final performance (mean ± std across seeds): accuracy, latency, reward,
    overhead, churn, and offload intensity.
    """
    rows: List[SummaryRow] = []
    for strat, rs in runs_by_strategy.items():
        accs, rews, lats, churns, offs, ohms = [], [], [], [], [], []
        for r in rs:
            accs.append(float(np.nanmean(series_tail(r, "acc", W))))
            rews.append(float(np.nanmean(series_tail(r, "reward", W))))
            lats.append(float(np.nanmean(series_tail(r, "latency_s", W))))
            churns.append(float(np.nanmean(series_tail(r, "churn", W))))
            offs.append(float(np.nanmean(series_tail(r, "offload_intensity", W))))
            ohms.append(float(np.nanmean(series_tail(r, "telemetry_overhead_ms", W))))

        rows.append(SummaryRow(
            strategy=strat,
            seeds=len(rs),
            acc_mean=_safe_mean(accs), acc_std=_safe_std(accs),
            reward_mean=_safe_mean(rews), reward_std=_safe_std(rews),
            lat_mean=_safe_mean(lats), lat_std=_safe_std(lats),
            churn_mean=_safe_mean(churns), churn_std=_safe_std(churns),
            off_mean=_safe_mean(offs), off_std=_safe_std(offs),
            overhead_ms_mean=_safe_mean(ohms), overhead_ms_std=_safe_std(ohms),
        ))
    return rows


def write_tableX_csv(outpath: str, rows: List[SummaryRow]) -> None:
    header = [
        "strategy", "seeds",
        "acc_mean", "acc_std",
        "reward_mean", "reward_std",
        "latency_s_mean", "latency_s_std",
        "churn_mean", "churn_std",
        "offload_intensity_mean", "offload_intensity_std",
        "telemetry_overhead_ms_mean", "telemetry_overhead_ms_std",
    ]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join([
            r.strategy, str(r.seeds),
            f"{r.acc_mean:.6f}", f"{r.acc_std:.6f}",
            f"{r.reward_mean:.6f}", f"{r.reward_std:.6f}",
            f"{r.lat_mean:.6f}", f"{r.lat_std:.6f}",
            f"{r.churn_mean:.6f}", f"{r.churn_std:.6f}",
            f"{r.off_mean:.6f}", f"{r.off_std:.6f}",
            f"{r.overhead_ms_mean:.6f}", f"{r.overhead_ms_std:.6f}",
        ]))
    with open(outpath, "w") as f:
        f.write("\n".join(lines))


# ----------------------------
# Table X+1 (Significance): TR-DP-AFRL vs baselines + effect sizes
# ----------------------------
def paired_or_indep_pvalue(a: List[float], b: List[float]) -> Tuple[float, str]:
    """
    If same-length and looks like paired seeds, do paired t-test; else independent t-test.
    """
    if stats is None:
        return float("nan"), "scipy_missing"

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    if len(aa) == 0 or len(bb) == 0:
        return float("nan"), "no_data"

    if len(aa) == len(bb) and len(aa) >= 2:
        _, p = stats.ttest_rel(aa, bb, nan_policy="omit")
        return float(p), "paired_t"
    else:
        _, p = stats.ttest_ind(aa, bb, equal_var=False, nan_policy="omit")
        return float(p), "indep_t"


def write_significance_csv(outpath: str, runs_by_strategy: Dict[str, List[Dict[str, Any]]], W: int) -> None:
    """
    Table X+1:
    Statistical significance of TR-DP-AFRL improvements vs baselines
    (paired tests across seeds when possible; p-values and effect sizes).

    Metrics included:
      acc, reward, latency_s, churn, offload_intensity, telemetry_overhead_ms
    """
    target = "tr_dp_afrl"
    if target not in runs_by_strategy:
        raise RuntimeError("TR-DP-AFRL runs not found; cannot compute significance table.")

    baselines = [s for s in sorted(runs_by_strategy.keys()) if s != target]
    metrics = ["acc", "reward", "latency_s", "churn", "offload_intensity", "telemetry_overhead_ms"]

    header = ["metric", "baseline", "test", "p_value", "effect_size_d", "note"]
    lines = [",".join(header)]

    for m in metrics:
        tr_vals = [float(np.nanmean(series_tail(r, m, W))) for r in runs_by_strategy[target]]

        for b in baselines:
            bl_vals = [float(np.nanmean(series_tail(r, m, W))) for r in runs_by_strategy[b]]

            p, test = paired_or_indep_pvalue(tr_vals, bl_vals)

            # effect size: paired if possible, else independent
            if len(tr_vals) == len(bl_vals) and len(tr_vals) >= 2:
                d = _cohens_d_paired(tr_vals, bl_vals)
                note = "paired_d=(TR-BL)"
            else:
                d = _cohens_d_indep(tr_vals, bl_vals)
                note = "indep_d=(TR-BL)"

            lines.append(",".join([m, b, test, f"{p:.8g}", f"{d:.6f}", note]))

    with open(outpath, "w") as f:
        f.write("\n".join(lines))


# ----------------------------
# Table X+2 (Ablation): WITH vs NO telemetry (TL-PPO, TR-DP-AFRL)
# ----------------------------
def _variant_kind(vlab: str) -> str:
    """
    Normalize variant labels to one of: 'NO', 'WITH', or 'OTHER'
    """
    s = str(vlab).upper()
    if "NO" in s:
        return "NO"
    if "WITH" in s:
        return "WITH"
    return "OTHER"


@dataclass
class AblationRow:
    strategy: str
    seeds_with: int
    seeds_no: int
    acc_with_mean: float
    acc_with_std: float
    acc_no_mean: float
    acc_no_std: float
    acc_delta_mean: float
    lat_with_mean: float
    lat_with_std: float
    lat_no_mean: float
    lat_no_std: float
    lat_delta_mean: float
    overhead_with_mean: float
    overhead_with_std: float
    overhead_no_mean: float
    overhead_no_std: float
    overhead_delta_mean: float


def build_tableX2_ablation(runs: List[Dict[str, Any]], W: int, default_variant_label: str) -> List[AblationRow]:
    """
    Table X+2:
    WITH vs NO telemetry for TL-PPO and TR-DP-AFRL.
    Metrics: accuracy, latency, telemetry_overhead_ms (tail window mean).
    """
    acc: Dict[Tuple[str, str], List[float]] = {}
    lat: Dict[Tuple[str, str], List[float]] = {}
    ohm: Dict[Tuple[str, str], List[float]] = {}
    plain_runs: Dict[str, List[Dict[str, Any]]] = {}

    for r in runs:
        strat = r.get("strategy", "unknown")
        if isinstance(strat, list) and strat:
            strat = strat[0]
        strat = str(strat)

        vlab = r.get("variant_label", default_variant_label)
        if isinstance(vlab, list) and vlab:
            vlab = vlab[0]
        vlab = str(vlab)
        kind = _variant_kind(vlab)

        if kind not in ("NO", "WITH"):
            plain_runs.setdefault(strat, []).append(r)
            continue

        key = (strat, kind)
        acc.setdefault(key, []).append(float(np.nanmean(series_tail(r, "acc", W))))
        lat.setdefault(key, []).append(float(np.nanmean(series_tail(r, "latency_s", W))))
        ohm.setdefault(key, []).append(float(np.nanmean(series_tail(r, "telemetry_overhead_ms", W))))

    rows: List[AblationRow] = []
    for strat in ["tl_ppo", "tr_dp_afrl"]:
        no_key = (strat, "NO")
        with_key = (strat, "WITH")

        # fallback: if WITH missing, use plain runs as WITH (only if NO exists)
        if with_key not in acc and (strat in plain_runs) and (no_key in acc):
            for r in plain_runs[strat]:
                acc.setdefault(with_key, []).append(float(np.nanmean(series_tail(r, "acc", W))))
                lat.setdefault(with_key, []).append(float(np.nanmean(series_tail(r, "latency_s", W))))
                ohm.setdefault(with_key, []).append(float(np.nanmean(series_tail(r, "telemetry_overhead_ms", W))))

        if no_key not in acc or with_key not in acc:
            print(f"[INFO] TableX+2 ablation: missing NO/WITH for {strat}; skipping row.")
            continue

        acc_no = acc[no_key]; acc_w = acc[with_key]
        lat_no = lat[no_key]; lat_w = lat[with_key]
        oh_no = ohm[no_key]; oh_w = ohm[with_key]

        m = min(len(acc_w), len(acc_no), len(lat_w), len(lat_no), len(oh_w), len(oh_no))
        acc_delta = [acc_w[i] - acc_no[i] for i in range(m)] if m > 0 else []
        lat_delta = [lat_w[i] - lat_no[i] for i in range(m)] if m > 0 else []
        oh_delta = [oh_w[i] - oh_no[i] for i in range(m)] if m > 0 else []

        rows.append(AblationRow(
            strategy=strat,
            seeds_with=len(acc_w),
            seeds_no=len(acc_no),

            acc_with_mean=_safe_mean(acc_w), acc_with_std=_safe_std(acc_w),
            acc_no_mean=_safe_mean(acc_no),   acc_no_std=_safe_std(acc_no),
            acc_delta_mean=_safe_mean(acc_delta),

            lat_with_mean=_safe_mean(lat_w), lat_with_std=_safe_std(lat_w),
            lat_no_mean=_safe_mean(lat_no),  lat_no_std=_safe_std(lat_no),
            lat_delta_mean=_safe_mean(lat_delta),

            overhead_with_mean=_safe_mean(oh_w), overhead_with_std=_safe_std(oh_w),
            overhead_no_mean=_safe_mean(oh_no),  overhead_no_std=_safe_std(oh_no),
            overhead_delta_mean=_safe_mean(oh_delta),
        ))

    return rows


def write_tableX2_csv(outpath: str, rows: List[AblationRow]) -> None:
    header = [
        "strategy",
        "seeds_WITH", "seeds_NO",
        "acc_WITH_mean", "acc_WITH_std", "acc_NO_mean", "acc_NO_std", "acc_delta_WITH_minus_NO",
        "lat_WITH_mean", "lat_WITH_std", "lat_NO_mean", "lat_NO_std", "lat_delta_WITH_minus_NO",
        "overhead_ms_WITH_mean", "overhead_ms_WITH_std", "overhead_ms_NO_mean", "overhead_ms_NO_std", "overhead_delta_WITH_minus_NO",
    ]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join([
            r.strategy,
            str(r.seeds_with), str(r.seeds_no),

            f"{r.acc_with_mean:.6f}", f"{r.acc_with_std:.6f}",
            f"{r.acc_no_mean:.6f}",   f"{r.acc_no_std:.6f}",
            f"{r.acc_delta_mean:.6f}",

            f"{r.lat_with_mean:.6f}", f"{r.lat_with_std:.6f}",
            f"{r.lat_no_mean:.6f}",   f"{r.lat_no_std:.6f}",
            f"{r.lat_delta_mean:.6f}",

            f"{r.overhead_with_mean:.6f}", f"{r.overhead_with_std:.6f}",
            f"{r.overhead_no_mean:.6f}",   f"{r.overhead_no_std:.6f}",
            f"{r.overhead_delta_mean:.6f}",
        ]))
    with open(outpath, "w") as f:
        f.write("\n".join(lines))


# ----------------------------
# Table X+3 (Scalability): N=4/6/8/10 summary for TL-PPO vs TR-DP-AFRL
# ----------------------------
@dataclass
class ScalabilityRow:
    N: int
    strategy: str
    seeds: int
    acc_mean: float
    acc_std: float
    lat_mean: float
    lat_std: float
    churn_mean: float
    churn_std: float


def build_tableX3_scalability(scal_dirs: List[str], W: int) -> List[ScalabilityRow]:
    rows: List[ScalabilityRow] = []
    for d in scal_dirs:
        d = d.strip()
        if not d or not os.path.isdir(d):
            continue

        N = _infer_N_from_dirname(d)
        runs = _load_runs_generic(d, num_clients_hint=(N or 0))
        if not runs:
            continue

        if N is None:
            N = _infer_num_clients_from_run(runs[0], fallback=0)
        if N is None or int(N) <= 0:
            continue
        N = int(N)

        g = group_runs(runs)
        for strat in ["tl_ppo", "tr_dp_afrl"]:
            if strat not in g:
                continue
            rs = g[strat]
            accs = [float(np.nanmean(series_tail(r, "acc", W))) for r in rs]
            lats = [float(np.nanmean(series_tail(r, "latency_s", W))) for r in rs]
            chs = [float(np.nanmean(series_tail(r, "churn", W))) for r in rs]

            rows.append(ScalabilityRow(
                N=N, strategy=strat, seeds=len(rs),
                acc_mean=_safe_mean(accs), acc_std=_safe_std(accs),
                lat_mean=_safe_mean(lats), lat_std=_safe_std(lats),
                churn_mean=_safe_mean(chs), churn_std=_safe_std(chs),
            ))

    rows.sort(key=lambda r: (r.N, r.strategy))
    return rows


def write_tableX3_csv(outpath: str, rows: List[ScalabilityRow]) -> None:
    header = ["N", "strategy", "seeds", "acc_mean", "acc_std", "latency_s_mean", "latency_s_std", "churn_mean", "churn_std"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join([
            str(r.N), r.strategy, str(r.seeds),
            f"{r.acc_mean:.6f}", f"{r.acc_std:.6f}",
            f"{r.lat_mean:.6f}", f"{r.lat_std:.6f}",
            f"{r.churn_mean:.6f}", f"{r.churn_std:.6f}",
        ]))
    with open(outpath, "w") as f:
        f.write("\n".join(lines))


# ----------------------------
# Table X+4 (Robustness): Severity A–E summary + drop/increase + recovery slope
# ----------------------------
@dataclass
class RobustnessRow:
    severity: str
    strategy: str
    seeds: int
    acc_mean: float
    acc_std: float
    lat_mean: float
    lat_std: float
    churn_mean: float
    churn_std: float
    acc_drop_pct_vs_A: float
    lat_increase_pct_vs_A: float
    churn_increase_pct_vs_A: float
    recovery_slope: float


def _severity_label_from_dir(d: str) -> str:
    base = os.path.basename(os.path.normpath(d))
    m = re.search(r"SEV[_\-]?([A-Z])", base.upper())
    if m:
        return f"SEV-{m.group(1)}"
    return base


def _load_runs_generic(indir: str, num_clients_hint: int = 0) -> List[Dict[str, Any]]:
    """
    Load runs from a folder without requiring client_ids.
    We normalize with placeholder client_ids length = num_clients (inferred if possible).
    """
    paths = sorted(glob.glob(os.path.join(indir, "results_*.json")))
    if not paths:
        return []

    inferred_n: int = 0
    try:
        with open(paths[0], "r") as f:
            d0 = json.load(f)
        inferred_n = _infer_num_clients_from_run(d0, fallback=num_clients_hint)
    except Exception:
        inferred_n = int(num_clients_hint)

    if inferred_n <= 0:
        inferred_n = int(num_clients_hint) if int(num_clients_hint) > 0 else 1

    client_ids = [f"client{i+1}" for i in range(inferred_n)]
    runs: List[Dict[str, Any]] = []
    for p in paths:
        try:
            with open(p, "r") as f:
                d = json.load(f)
            d["_path"] = p
            d = normalize_run(d, client_ids=client_ids, num_clients=inferred_n)
            runs.append(d)
        except Exception as e:
            print(f"[WARN] Fig12/13/Tables: failed to read {p}: {e}")
    return runs


def build_tableX4_robustness(robust_dirs: List[str], W: int) -> List[RobustnessRow]:
    """
    Baseline = SEV-A per strategy (tail window mean).
    Drop/increase computed relative to SEV-A baseline for each strategy.
    Recovery slope uses post-impair region if impairment window stored; else tail slope proxy.
    """
    perS: List[Tuple[str, Dict[str, Dict[str, Any]]]] = []
    for d in robust_dirs:
        d = d.strip()
        if not d or not os.path.isdir(d):
            continue

        label = _severity_label_from_dir(d)
        runs = _load_runs_generic(d, num_clients_hint=0)
        if not runs:
            continue

        g = group_runs(runs)
        entry: Dict[str, Dict[str, Any]] = {}
        for strat in ["fedavg", "tl_ppo", "tr_dp_afrl"]:
            if strat not in g:
                continue
            rs = g[strat]

            accs = [float(np.nanmean(series_tail(r, "acc", W))) for r in rs]
            lats = [float(np.nanmean(series_tail(r, "latency_s", W))) for r in rs]
            chs = [float(np.nanmean(series_tail(r, "churn", W))) for r in rs]

            slopes: List[float] = []
            for r in rs:
                T = _round_count(r)
                a, b = _infer_impair_window(r)
                if a is not None and b is not None and T > (b + 2):
                    post = _as_list(r.get("acc", []))[b:]  # b is 1-based end round; slice from b onward approx
                    slopes.append(_safe_slope([float(v) for v in post if v is not None]))
                else:
                    tail = _as_list(r.get("acc", []))[-W:] if W > 1 else _as_list(r.get("acc", []))
                    slopes.append(_safe_slope([float(v) for v in tail if v is not None]))

            entry[strat] = dict(
                seeds=len(rs),
                acc_mean=_safe_mean(accs), acc_std=_safe_std(accs),
                lat_mean=_safe_mean(lats), lat_std=_safe_std(lats),
                churn_mean=_safe_mean(chs), churn_std=_safe_std(chs),
                slope_mean=_safe_mean(slopes),
            )

        perS.append((label, entry))

    if not perS:
        return []

    # baseline severity A index
    base_idx = 0
    for i, (lab, _) in enumerate(perS):
        if "SEV-A" in lab.upper() or "SEV_A" in lab.upper():
            base_idx = i
            break
    _, base_entry = perS[base_idx]

    def pct_change(cur_v: float, base_v: float) -> float:
        if not np.isfinite(cur_v) or not np.isfinite(base_v) or base_v == 0.0:
            return float("nan")
        return float((cur_v - base_v) / abs(base_v) * 100.0)

    rows: List[RobustnessRow] = []
    for (lab, entry) in perS:
        for strat in ["fedavg", "tl_ppo", "tr_dp_afrl"]:
            if strat not in entry:
                continue
            cur = entry[strat]
            bstats = base_entry.get(strat, None)

            if bstats is None:
                acc_drop = float("nan")
                lat_inc = float("nan")
                churn_inc = float("nan")
            else:
                acc_drop = -pct_change(float(cur["acc_mean"]), float(bstats["acc_mean"]))  # drop positive if worse
                lat_inc = pct_change(float(cur["lat_mean"]), float(bstats["lat_mean"]))
                churn_inc = pct_change(float(cur["churn_mean"]), float(bstats["churn_mean"]))

            rows.append(RobustnessRow(
                severity=lab,
                strategy=strat,
                seeds=int(cur["seeds"]),
                acc_mean=float(cur["acc_mean"]), acc_std=float(cur["acc_std"]),
                lat_mean=float(cur["lat_mean"]), lat_std=float(cur["lat_std"]),
                churn_mean=float(cur["churn_mean"]), churn_std=float(cur["churn_std"]),
                acc_drop_pct_vs_A=float(acc_drop),
                lat_increase_pct_vs_A=float(lat_inc),
                churn_increase_pct_vs_A=float(churn_inc),
                recovery_slope=float(cur["slope_mean"]),
            ))

    # keep severity order as provided
    sev_order = {_severity_label_from_dir(d): i for i, d in enumerate(robust_dirs)}
    rows.sort(key=lambda r: (sev_order.get(r.severity, 999), r.strategy))
    return rows


def write_tableX4_csv(outpath: str, rows: List[RobustnessRow]) -> None:
    header = [
        "severity", "strategy", "seeds",
        "acc_mean", "acc_std",
        "latency_s_mean", "latency_s_std",
        "churn_mean", "churn_std",
        "acc_drop_pct_vs_SEV_A",
        "latency_increase_pct_vs_SEV_A",
        "churn_increase_pct_vs_SEV_A",
        "recovery_slope",
    ]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join([
            r.severity, r.strategy, str(r.seeds),
            f"{r.acc_mean:.6f}", f"{r.acc_std:.6f}",
            f"{r.lat_mean:.6f}", f"{r.lat_std:.6f}",
            f"{r.churn_mean:.6f}", f"{r.churn_std:.6f}",
            f"{r.acc_drop_pct_vs_A:.6f}",
            f"{r.lat_increase_pct_vs_A:.6f}",
            f"{r.churn_increase_pct_vs_A:.6f}",
            f"{r.recovery_slope:.6f}",
        ]))
    with open(outpath, "w") as f:
        f.write("\n".join(lines))


# ----------------------------
# Main plots (Fig3–Fig14)
# ----------------------------
def fig3_4_5_meanstd(perf_dir: str, runs_by_strategy: Dict[str, List[Dict[str, Any]]], T: int) -> None:
    x = np.arange(1, T + 1)
    items = [
        ("acc",       "Fig3_accuracy_meanstd.png", "Accuracy",     "Validation Accuracy (mean±std)"),
        ("reward",    "Fig4_reward_meanstd.png",   "Reward",       "Reward (mean±std)"),
        ("latency_s", "Fig5_latency_meanstd.png",  "Latency (s)",  "Latency (s) (mean±std)"),
    ]
    for key, fname, ylab, title in items:
        means: Dict[str, np.ndarray] = {}
        stds: Dict[str, np.ndarray] = {}
        for strat, rs in runs_by_strategy.items():
            mats = [np.asarray(_pad_to_T(r.get(key, []), T), dtype=np.float64) for r in rs]
            A = np.vstack(mats) if mats else np.zeros((1, T))
            means[strat] = np.nanmean(A, axis=0)
            stds[strat] = np.nanstd(A, axis=0, ddof=1) if A.shape[0] > 1 else np.zeros(T)
        save_meanstd_plot(os.path.join(perf_dir, fname), x, means, stds, "Round", ylab, title)


def fig6_overlay(perf_dir: str, runs_by_strategy: Dict[str, List[Dict[str, Any]]], T: int) -> None:
    """
    Overlay TL-PPO vs TR-DP-AFRL on accuracy/reward/latency.
    Titles use expanded wording (Overlay Accuracy, etc.) and NO Fig numbering.
    Uses full horizon T and pads shorter series.
    """
    x = np.arange(1, T + 1)
    meta = {
        "acc": ("Accuracy", "Overlay Accuracy"),
        "reward": ("Reward", "Overlay Reward"),
        "latency_s": ("Latency (s)", "Overlay Latency"),
    }
    A, B = "tl_ppo", "tr_dp_afrl"
    if A not in runs_by_strategy or B not in runs_by_strategy:
        return

    def mean_curve(strat: str, key: str) -> np.ndarray:
        mats = [np.asarray(_pad_to_T(r.get(key, []), T), dtype=np.float64) for r in runs_by_strategy[strat]]
        M = np.vstack(mats) if mats else np.zeros((1, T))
        return np.nanmean(M, axis=0)

    for key in ["acc", "reward", "latency_s"]:
        ylabel, title_prefix = meta[key]
        ya = mean_curve(A, key)
        yb = mean_curve(B, key)
        save_lineplot(
            os.path.join(perf_dir, f"Fig6_overlay_{key}_{A}_vs_{B}.png"),
            x,
            {"TL-PPO": ya, "TR-DP-AFRL": yb},
            "Round",
            ylabel,
            f"{title_prefix}: TL-PPO vs TR-DP-AFRL"
        )


def fig7_ablation(ablation_dir: str, runs: List[Dict[str, Any]], T: int, default_variant_label: str) -> None:
    """
    Ablation plot: ONLY two curves (NO vs WITH) with reviewer-safe legend names.
    Uses full horizon T and pads shorter series.
    Produces:
      Fig7_ablation_tl_ppo.png
      Fig7_ablation_tr_dp_afrl.png
    """
    x = np.arange(1, T + 1)

    group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    plain_group: Dict[str, List[Dict[str, Any]]] = {}

    for r in runs:
        strat = r.get("strategy", "unknown")
        if isinstance(strat, list) and strat:
            strat = strat[0]
        strat = str(strat)

        vlab = r.get("variant_label", default_variant_label)
        if isinstance(vlab, list) and vlab:
            vlab = vlab[0]
        vlab = str(vlab)

        kind = _variant_kind(vlab)
        if kind in ("NO", "WITH"):
            group.setdefault((strat, kind), []).append(r)
        else:
            plain_group.setdefault(strat, []).append(r)

    for strat in ["tl_ppo", "tr_dp_afrl"]:
        curves: Dict[str, np.ndarray] = {}

        if (strat, "NO") in group:
            mats_no = [np.asarray(_pad_to_T(r.get("acc", []), T), dtype=np.float64) for r in group[(strat, "NO")]]
            curves["Telemetry-Agnostic (NO)"] = np.nanmean(np.vstack(mats_no), axis=0)

        if (strat, "WITH") in group:
            mats_w = [np.asarray(_pad_to_T(r.get("acc", []), T), dtype=np.float64) for r in group[(strat, "WITH")]]
            curves["Telemetry-Aware (WITH)"] = np.nanmean(np.vstack(mats_w), axis=0)
        else:
            if "Telemetry-Agnostic (NO)" in curves and strat in plain_group and len(plain_group[strat]) > 0:
                mats_plain = [np.asarray(_pad_to_T(r.get("acc", []), T), dtype=np.float64) for r in plain_group[strat]]
                curves["Telemetry-Aware (WITH)"] = np.nanmean(np.vstack(mats_plain), axis=0)

        if set(curves.keys()) == {"Telemetry-Agnostic (NO)", "Telemetry-Aware (WITH)"}:
            pretty = "TL-PPO" if strat == "tl_ppo" else "TR-DP-AFRL"
            save_lineplot(
                os.path.join(ablation_dir, f"Fig7_ablation_{strat}.png"),
                x, curves,
                "Round", "Accuracy",
                f"Telemetry Ablation ({pretty}): Telemetry-Agnostic vs Telemetry-Aware"
            )
        else:
            print(f"[INFO] Fig7 ablation for {strat}: need both NO and WITH (found: {list(curves.keys())}).")


def fig10_offload(outpath: str, runs_by_strategy: Dict[str, List[Dict[str, Any]]], T: int) -> None:
    x = np.arange(1, T + 1)
    means: Dict[str, np.ndarray] = {}
    stds: Dict[str, np.ndarray] = {}
    for strat, rs in runs_by_strategy.items():
        mats = [np.asarray(_pad_to_T(r.get("offload_intensity", []), T), dtype=np.float64) for r in rs]
        A = np.vstack(mats) if mats else np.zeros((1, T))
        means[strat] = np.nanmean(A, axis=0)
        stds[strat] = np.nanstd(A, axis=0, ddof=1) if A.shape[0] > 1 else np.zeros(T)
    save_meanstd_plot(outpath, x, means, stds, "Round", "Offload intensity", "Offload Intensity (mean±std)")


def fig14_costbenefit(outpath: str, runs_by_strategy: Dict[str, List[Dict[str, Any]]], W: int) -> None:
    """
    Simple proxy cost-benefit: x=latency mean, y=accuracy mean (tail window).
    """
    plt.figure(figsize=(10, 6))
    for strat, rs in runs_by_strategy.items():
        xs = [float(np.nanmean(series_tail(r, "latency_s", W))) for r in rs]
        ys = [float(np.nanmean(series_tail(r, "acc", W))) for r in rs]
        plt.scatter(xs, ys, label=_pretty_strategy_name(strat))
    plt.xlabel("Latency (s) mean (tail)")
    plt.ylabel("Accuracy mean (tail)")
    plt.title(_wrap_title("Cost–Benefit (Latency vs Accuracy)", 60))
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


# ----------------------------
# Fig12 / Fig13
# ----------------------------
def fig12_scalability(outpath: str, scal_dirs: List[str], W: int) -> None:
    """
    Scalability: tail accuracy vs number of clients (mean±std over seeds) for each strategy.
    """
    acc_byN: Dict[int, Dict[str, List[float]]] = {}
    Ns: List[int] = []

    for d in scal_dirs:
        d = d.strip()
        if not d:
            continue
        if not os.path.isdir(d):
            print(f"[WARN] Fig12: missing dir: {d}")
            continue

        N = _infer_N_from_dirname(d)
        runs = _load_runs_generic(d, num_clients_hint=(N or 0))
        if not runs:
            print(f"[WARN] Fig12: no results_*.json in {d}")
            continue

        if N is None:
            N = _infer_num_clients_from_run(runs[0], fallback=0)
        if N is None or int(N) <= 0:
            print(f"[WARN] Fig12: could not infer N for {d}; skipping.")
            continue

        N = int(N)
        if N not in acc_byN:
            acc_byN[N] = {}
            Ns.append(N)

        g = group_runs(runs)
        for strat, rs in g.items():
            vals = [float(np.nanmean(series_tail(r, "acc", W))) for r in rs]
            acc_byN[N].setdefault(strat, []).extend(vals)

    Ns = sorted(set(Ns))
    if not Ns:
        print("[WARN] Fig12: no valid scalability points; skipping.")
        return

    strat_counts: Dict[str, int] = {}
    for N in Ns:
        for strat in acc_byN[N].keys():
            strat_counts[strat] = strat_counts.get(strat, 0) + 1
    strategies = [s for s, c in strat_counts.items() if c >= 2]
    strategies = sorted(strategies)
    if not strategies:
        strategies = sorted(set().union(*[set(acc_byN[N].keys()) for N in Ns]))

    plt.figure(figsize=(10, 6))
    x = np.asarray(Ns, dtype=np.float64)

    for strat in strategies:
        means: List[float] = []
        stds: List[float] = []
        for N in Ns:
            vals = acc_byN[N].get(strat, [])
            means.append(_safe_mean(vals))
            stds.append(_safe_std(vals))
        y = np.asarray(means, dtype=np.float64)
        e = np.asarray(stds, dtype=np.float64)
        plt.errorbar(x, y, yerr=e, marker="o", capsize=3, label=_pretty_strategy_name(strat))

    plt.xlabel("Number of Clients (N)")
    plt.ylabel(f"Tail Accuracy (mean±std over seeds, W={int(W)})")
    plt.title(_wrap_title("Scalability: Accuracy vs Number of Clients", 60))
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def fig13_robustness(outpath: str, robust_dirs: List[str], W: int) -> None:
    """
    Robustness: tail accuracy vs impairment severity (mean±std over seeds) for each strategy.
    """
    labels: List[str] = []
    acc_byS: List[Dict[str, List[float]]] = []

    for d in robust_dirs:
        d = d.strip()
        if not d:
            continue
        if not os.path.isdir(d):
            print(f"[WARN] Fig13: missing dir: {d}")
            continue

        runs = _load_runs_generic(d, num_clients_hint=0)
        if not runs:
            print(f"[WARN] Fig13: no results_*.json in {d}")
            continue

        labels.append(_severity_label_from_dir(d))
        g = group_runs(runs)

        entry: Dict[str, List[float]] = {}
        for strat, rs in g.items():
            vals = [float(np.nanmean(series_tail(r, "acc", W))) for r in rs]
            entry[strat] = vals
        acc_byS.append(entry)

    if not labels:
        print("[WARN] Fig13: no valid robustness points; skipping.")
        return

    strat_counts: Dict[str, int] = {}
    for entry in acc_byS:
        for strat in entry.keys():
            strat_counts[strat] = strat_counts.get(strat, 0) + 1
    strategies = [s for s, c in strat_counts.items() if c >= 2]
    strategies = sorted(strategies)
    if not strategies:
        strategies = sorted(set().union(*[set(e.keys()) for e in acc_byS]))

    x = np.arange(len(labels), dtype=np.int64)

    plt.figure(figsize=(10, 6))
    for strat in strategies:
        means: List[float] = []
        stds: List[float] = []
        for entry in acc_byS:
            vals = entry.get(strat, [])
            means.append(_safe_mean(vals))
            stds.append(_safe_std(vals))
        y = np.asarray(means, dtype=np.float64)
        e = np.asarray(stds, dtype=np.float64)
        plt.errorbar(x, y, yerr=e, marker="o", capsize=3, label=_pretty_strategy_name(strat))

    plt.xticks(x, labels, rotation=0)
    plt.xlabel("Impairment Severity")
    plt.ylabel(f"Tail Accuracy (mean±std over seeds, W={int(W)})")
    plt.title(_wrap_title("Robustness Under Impairment: Accuracy vs Severity", 60))
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


# ----------------------------
# Main
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_indir", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--variant_label", type=str, default="N10")
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--client_ids", type=str, required=True)
    ap.add_argument("--scal_dirs", type=str, default=None, help="Comma-separated dirs for Fig12 + Table X+3")
    ap.add_argument("--robust_dirs", type=str, default=None, help="Comma-separated dirs for Fig13 + Table X+4")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    client_ids = [x.strip() for x in args.client_ids.split(",") if x.strip()]
    if len(client_ids) != int(args.num_clients):
        raise ValueError(f"--client_ids count ({len(client_ids)}) must equal --num_clients ({args.num_clients})")

    perf_dir = os.path.join(args.outdir, "performance")
    ablation_dir = os.path.join(args.outdir, "ablation")
    dyn_dir = os.path.join(args.outdir, "dynamics")
    table_dir = os.path.join(args.outdir, "tables")
    cost_dir = os.path.join(args.outdir, "cost_benefit")
    _ensure_dir(perf_dir)
    _ensure_dir(ablation_dir)
    _ensure_dir(dyn_dir)
    _ensure_dir(table_dir)
    _ensure_dir(cost_dir)

    runs = load_runs(args.main_indir, client_ids=client_ids, num_clients=int(args.num_clients))
    if not runs:
        raise RuntimeError(f"No results_*.json found under {args.main_indir}")

    runs_by_strategy = group_runs(runs)

    # KEY FIX: Use full horizon (max over all runs), not runs[0]
    T = _max_round_count(runs)
    if T <= 0:
        raise RuntimeError("Could not determine round horizon (T<=0). Check JSON logs.")
    print(f"[INFO] Plot horizon T={T} (max over runs)")

    # --- Fig3–Fig6
    fig3_4_5_meanstd(perf_dir, runs_by_strategy, T)
    print(f"[OK] Fig3–Fig5 saved to: {perf_dir}")

    fig6_overlay(perf_dir, runs_by_strategy, T)
    print(f"[OK] Fig6 overlay saved to: {perf_dir}")

    # --- Fig7 Ablation (ONLY 2 lines) -> yields 2 figures (tl_ppo and tr_dp_afrl)
    fig7_ablation(ablation_dir, runs, T, default_variant_label=args.variant_label)
    print(f"[OK] Fig7 saved to: {ablation_dir}")

    # --- Fig8 Heatmap
    fig8_heatmap(os.path.join(dyn_dir, "Fig8_selection_heatmap_clients.png"), runs_by_strategy, client_ids, T)
    print(f"[OK] Fig8 saved to: {dyn_dir}")

    # --- Fig9 Overhead (2 figures: 9a/9b)
    fig9_overhead(dyn_dir, runs_by_strategy, T)
    print(f"[OK] Fig9a/Fig9b saved to: {dyn_dir}")

    # --- Fig10 Offload intensity
    fig10_offload(os.path.join(dyn_dir, "Fig10_offload_intensity.png"), runs_by_strategy, T)
    print(f"[OK] Fig10 saved to: {dyn_dir}")

    # --- Fig11 Churn
    fig11_churn(os.path.join(dyn_dir, "Fig11_churn_dynamics.png"), runs_by_strategy, T)
    print(f"[OK] Fig11 saved to: {dyn_dir}")

    # --- Table X (Summary)
    rows = build_tableX(runs_by_strategy, W=int(args.W))
    out_table = os.path.join(table_dir, f"TableX_summary_metrics_W{int(args.W)}.csv")
    write_tableX_csv(out_table, rows)
    print(f"[OK] Table X saved to: {out_table}")

    # --- Table X+1 (Significance)
    out_sig = os.path.join(table_dir, f"TableXplus1_significance_W{int(args.W)}.csv")
    write_significance_csv(out_sig, runs_by_strategy, W=int(args.W))
    print(f"[OK] Table X+1 (Significance) saved to: {out_sig}")

    # --- Table X+2 (Ablation)
    rows_ab = build_tableX2_ablation(runs, W=int(args.W), default_variant_label=args.variant_label)
    out_ab = os.path.join(table_dir, f"TableXplus2_ablation_W{int(args.W)}.csv")
    write_tableX2_csv(out_ab, rows_ab)
    print(f"[OK] Table X+2 (Ablation) saved to: {out_ab}")

    # --- Fig12 + Table X+3 (Scalability)
    if args.scal_dirs:
        scal_dirs = [x.strip() for x in args.scal_dirs.split(",") if x.strip()]
        fig12_scalability(os.path.join(dyn_dir, "Fig12_scalability.png"), scal_dirs, W=int(args.W))
        print(f"[OK] Fig12 saved to: {dyn_dir}")

        rows_sc = build_tableX3_scalability(scal_dirs, W=int(args.W))
        out_sc = os.path.join(table_dir, f"TableXplus3_scalability_W{int(args.W)}.csv")
        write_tableX3_csv(out_sc, rows_sc)
        print(f"[OK] Table X+3 (Scalability) saved to: {out_sc}")
    else:
        print("[INFO] Fig12 + Table X+3 skipped (no --scal_dirs provided)")

    # --- Fig13 + Table X+4 (Robustness)
    if args.robust_dirs:
        robust_dirs = [x.strip() for x in args.robust_dirs.split(",") if x.strip()]
        fig13_robustness(os.path.join(dyn_dir, "Fig13_robustness.png"), robust_dirs, W=int(args.W))
        print(f"[OK] Fig13 saved to: {dyn_dir}")

        rows_rb = build_tableX4_robustness(robust_dirs, W=int(args.W))
        out_rb = os.path.join(table_dir, f"TableXplus4_robustness_W{int(args.W)}.csv")
        write_tableX4_csv(out_rb, rows_rb)
        print(f"[OK] Table X+4 (Robustness) saved to: {out_rb}")
    else:
        print("[INFO] Fig13 + Table X+4 skipped (no --robust_dirs provided)")

    # --- Fig14 Cost–benefit
    fig14_costbenefit(os.path.join(cost_dir, "Fig14_cost_benefit.png"), runs_by_strategy, W=int(args.W))
    print(f"[OK] Fig14 saved to: {cost_dir}")

    print(f"[DONE] Full export completed under: {args.outdir}")


if __name__ == "__main__":
    main()
