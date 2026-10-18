#!/usr/bin/env python3

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ppo_core_e6b import PPOConfig, OffloadPPO, SelectionPPO


def enforce_min_local_training(
    a,
    selected,
    Xlive,
    min_participants,
):
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


def make_state_vec(mu, sd, acc_prev, lat_prev, churn_prev):
    extra = np.array(
        [acc_prev, lat_prev, churn_prev],
        dtype=np.float32,
    )

    return np.concatenate(
        [mu, sd, extra],
        axis=0,
    ).astype(np.float32)


def percentile(values, q):
    return float(np.percentile(
        np.asarray(values, dtype=np.float64),
        q,
    ))


def count_parameters(module):
    return int(sum(p.numel() for p in module.parameters()))


def benchmark_one(
    n_clients,
    warmup,
    repetitions,
    seed,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    K = int(math.ceil(0.8 * n_clients))
    min_local = int(math.ceil(0.5 * n_clients))

    state_dim = 33

    off_ppo = OffloadPPO(
        state_dim=state_dim,
        n_clients=n_clients,
        cfg=PPOConfig(delta_kl=0.02),
    )

    sel_ppo = SelectionPPO(
        state_dim=state_dim,
        n_clients=n_clients,
        cfg=PPOConfig(delta_kl=0.02),
    )

    # Fixed synthetic logical-client telemetry.
    # This is controller-only input and is NOT a physical FL run.
    rng = np.random.default_rng(seed + n_clients)

    Xlive = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_clients, 15),
    ).astype(np.float32)

    mu = Xlive.mean(axis=0).astype(np.float32)
    sd = Xlive.std(axis=0).astype(np.float32)

    s = make_state_vec(
        mu,
        sd,
        acc_prev=0.75,
        lat_prev=10.0,
        churn_prev=0.20,
    )

    selection_times = []
    offload_times = []
    controller_times = []

    selected_counts = []
    executed_offload_counts = []

    total_iterations = warmup + repetitions

    with torch.no_grad():

        for rep in range(total_iterations):

            t_controller0 = time.perf_counter_ns()

            # -----------------------------------------
            # Selection decision:
            # policy + top-K + aggregation processing
            # -----------------------------------------
            t_selection0 = time.perf_counter_ns()

            g, lam_raw, _ = sel_ppo.act(s)

            idx_sorted = np.argsort(-lam_raw)

            selected = np.zeros(
                n_clients,
                dtype=np.int64,
            )

            for j in idx_sorted:
                if selected.sum() >= K:
                    break

                if g[j] == 1:
                    selected[j] = 1

            if selected.sum() < K:
                selected[:] = 0
                selected[idx_sorted[:K]] = 1

            w = lam_raw * selected

            if w.sum() <= 0:
                w = selected.astype(np.float32)

            w = w / (w.sum() + 1e-12)

            w_min = 0.02
            w_max = 0.40

            sel_mask = selected == 1

            if sel_mask.any():
                w_sel = w[sel_mask]
                w_sel = np.clip(
                    w_sel,
                    w_min,
                    w_max,
                )
                w_sel = w_sel / (
                    w_sel.sum() + 1e-12
                )

                w = np.zeros_like(
                    w,
                    dtype=np.float32,
                )

                w[sel_mask] = (
                    w_sel.astype(np.float32)
                )

            else:
                w = (
                    np.ones(
                        n_clients,
                        dtype=np.float32,
                    )
                    / float(n_clients)
                )

            t_selection1 = time.perf_counter_ns()

            # -----------------------------------------
            # Offload decision:
            # policy + action conversion +
            # minimum-local execution projection
            # -----------------------------------------
            t_offload0 = time.perf_counter_ns()

            a, _, _ = off_ppo.act(s)

            if not isinstance(a, np.ndarray):
                a = np.array(
                    a,
                    dtype=np.int64,
                )

            a = a.astype(
                np.int64
            ).reshape(-1)

            if a.shape[0] != n_clients:
                a = np.zeros(
                    n_clients,
                    dtype=np.int64,
                )

            a = enforce_min_local_training(
                a=a,
                selected=selected,
                Xlive=Xlive,
                min_participants=min_local,
            )

            t_offload1 = time.perf_counter_ns()
            t_controller1 = time.perf_counter_ns()

            if rep >= warmup:
                selection_times.append(
                    (t_selection1 - t_selection0)
                    / 1_000_000.0
                )

                offload_times.append(
                    (t_offload1 - t_offload0)
                    / 1_000_000.0
                )

                controller_times.append(
                    (t_controller1 - t_controller0)
                    / 1_000_000.0
                )

                selected_counts.append(
                    int(selected.sum())
                )

                executed_offload_counts.append(
                    int(
                        np.sum(
                            (selected == 1)
                            & (a == 1)
                        )
                    )
                )

    result = {
        "benchmark": "E6B_CONTROLLER_LOGICAL_SCALABILITY",
        "physical_num_clients": 0,
        "logical_num_clients": int(n_clients),

        "state_dim": int(state_dim),

        "selection_gate_action_dim": int(n_clients),
        "selection_weight_action_dim": int(n_clients),
        "offload_action_dim": int(n_clients),

        "K": int(K),
        "min_local": int(min_local),

        "warmup_iterations": int(warmup),
        "measured_iterations": int(repetitions),

        "offload_policy_parameters": count_parameters(
            off_ppo.pi
        ),
        "offload_value_parameters": count_parameters(
            off_ppo.v
        ),
        "selection_policy_parameters": count_parameters(
            sel_ppo.pi
        ),
        "selection_value_parameters": count_parameters(
            sel_ppo.v
        ),

        "selection_ms_median": percentile(
            selection_times,
            50,
        ),
        "selection_ms_p95": percentile(
            selection_times,
            95,
        ),

        "offload_ms_median": percentile(
            offload_times,
            50,
        ),
        "offload_ms_p95": percentile(
            offload_times,
            95,
        ),

        "controller_ms_median": percentile(
            controller_times,
            50,
        ),
        "controller_ms_p95": percentile(
            controller_times,
            95,
        ),

        "selected_count_mean": float(
            np.mean(selected_counts)
        ),

        "executed_offload_count_mean": float(
            np.mean(executed_offload_counts)
        ),

        "claim_scope": (
            "Controller-only logical scalability. "
            "No physical FL execution, network communication, "
            "training accuracy, cloud cost, or end-to-end "
            "system scalability is measured."
        ),
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100],
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--out",
        type=str,
        default=(
            "/home/ubuntu/afrl_runs/"
            "E6B_CONTROLLER_SCALABILITY.json"
        ),
    )

    args = parser.parse_args()

    results = []

    print("=" * 72)
    print("E6-B CONTROLLER-ONLY LOGICAL SCALABILITY")
    print("=" * 72)
    print(
        "No physical FL clients are contacted by this benchmark."
    )
    print()

    for n in args.sizes:

        result = benchmark_one(
            n_clients=int(n),
            warmup=int(args.warmup),
            repetitions=int(args.repetitions),
            seed=int(args.seed),
        )

        results.append(result)

        print(
            f"N={n:3d} "
            f"K={result['K']:3d} "
            f"min_local={result['min_local']:3d} | "
            f"controller median="
            f"{result['controller_ms_median']:.4f} ms "
            f"p95="
            f"{result['controller_ms_p95']:.4f} ms | "
            f"selection params="
            f"{result['selection_policy_parameters']} "
            f"offload params="
            f"{result['offload_policy_parameters']}"
        )

    out = Path(args.out)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "experiment": (
            "E6B_CONTROLLER_LOGICAL_SCALABILITY"
        ),
        "physical_num_clients": 0,
        "sizes": [
            int(x) for x in args.sizes
        ],
        "warmup_iterations": int(args.warmup),
        "measured_iterations": int(
            args.repetitions
        ),
        "seed": int(args.seed),
        "results": results,
    }

    out.write_text(
        json.dumps(
            payload,
            indent=2,
        )
    )

    print()
    print(f"Saved: {out}")
    print(
        "FINAL: E6-B CONTROLLER BENCHMARK COMPLETE"
    )


if __name__ == "__main__":
    main()
