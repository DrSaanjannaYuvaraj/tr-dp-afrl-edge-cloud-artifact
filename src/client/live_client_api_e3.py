from __future__ import annotations

"""
live_client_api_e1.py

Independent Fashion-MNIST FL benchmark client.

Endpoints:
- GET  /health
- GET  /telemetry
      -> preserves the ORIGINAL telemetry pipeline used by the RL controller
- POST /train_local
      -> trains Fashion-MNIST rather than telemetry classification
- POST /impair/cpu_start|cpu_stop
- POST /impair/net_start|net_stop

Critical separation:
    RL/controller state  = AWS/system telemetry
    FL learning task     = Fashion-MNIST
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

# Original telemetry pipeline remains untouched for /telemetry
from telemetry_dataset_e3 import TelemetryWindowDataset

# New independent FL dataset/model
from afrl_e3_fashion_dataset import (
    get_fashion_mnist_datasets,
    make_dirichlet_partitions,
)

from model_utils_e3 import FashionMNISTCNN

# Reuse audited generic training / serialization utilities
from model_utils_e3_shared import (
    get_device,
    train_one_epoch_budget,
    state_dict_to_jsonable,
    jsonable_to_state_dict,
)


app = FastAPI()


STATE = {
    # Identity
    "client_id": "clientX",
    "client_index": 0,

    # Original telemetry data
    "csv": "",
    "win": 10,
    "label_col": "scenario",
    "class_map": None,
    "scaler_path": None,
    "telemetry_ds": None,

    # Fashion-MNIST caches.
    # Key = partition seed
    "fashion_train": None,
    "fashion_partitions": {},

    # Compute
    "device": get_device(force_cpu=True),

    # Impairments
    "cpu_impair_on": False,
    "net_impair_on": False,
    "cpu_workers": 0,
    "net_delay_ms": 0.0,
    "net_loss_pct": 0.0,
}


class TrainReq(BaseModel):
    round: int

    # Existing local-training hyperparameters
    local_epochs: int = 2
    lr: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 128

    # Existing execution action
    offload: int = 0

    global_state_dict: Dict[str, Any]

    max_local_batches: int = 60
    max_local_samples: int = 2500

    # E1 reproducibility parameters
    data_seed: int = 42
    dirichlet_alpha: float = 0.5
    num_clients: int = 10


# ------------------------------------------------------------------
# ORIGINAL TELEMETRY DATASET
# ------------------------------------------------------------------

def _ensure_telemetry_dataset():
    """
    Preserve the original telemetry dataset exclusively for
    controller-state generation.
    """

    if STATE["telemetry_ds"] is None:
        STATE["telemetry_ds"] = TelemetryWindowDataset(
            STATE["csv"],
            win=int(STATE["win"]),
            label_col=str(STATE["label_col"]),
            class_map_path=STATE["class_map"],
            scaler_path=STATE["scaler_path"],
        )


# ------------------------------------------------------------------
# E1 FASHION-MNIST DATA
# ------------------------------------------------------------------

def _ensure_fashion_train():
    """
    Load official Fashion-MNIST TRAIN split.

    The official TEST split is deliberately not used by clients.
    It will be reserved for global evaluation on the server.
    """

    if STATE["fashion_train"] is None:
        train_dataset, _ = get_fashion_mnist_datasets()
        STATE["fashion_train"] = train_dataset


def _get_client_partition(
    seed: int,
    alpha: float,
    num_clients: int,
):
    """
    Reconstruct the deterministic Dirichlet partition for the
    requested experiment seed and return this client's indices.
    """

    _ensure_fashion_train()

    cache_key = (
        int(seed),
        float(alpha),
        int(num_clients),
    )

    if cache_key not in STATE["fashion_partitions"]:

        partitions = make_dirichlet_partitions(
            train_dataset=STATE["fashion_train"],
            num_clients=int(num_clients),
            alpha=float(alpha),
            seed=int(seed),
            min_samples_per_client=100,
        )

        STATE["fashion_partitions"][cache_key] = partitions

    partitions = STATE["fashion_partitions"][cache_key]

    client_index = int(STATE["client_index"])

    if client_index not in partitions:
        raise RuntimeError(
            f"Client index {client_index} is not present "
            f"in partition with {num_clients} clients."
        )

    return partitions[client_index]


# ------------------------------------------------------------------
# BASIC ENDPOINTS
# ------------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "ok": True,
        "experiment": "E3_FashionMNIST",
        "client_id": STATE["client_id"],
        "client_index": int(STATE["client_index"]),
        "fl_task": "FashionMNIST",
        "controller_input": "telemetry",
    }


@app.get("/telemetry")
def telemetry(seed: int | None = None):
    """
    ORIGINAL controller telemetry behaviour.

    Returns a 15-dimensional vector in the original F15 order.

    No Fashion-MNIST information enters this endpoint.
    """

    t0 = time.time()

    _ensure_telemetry_dataset()

    ds: TelemetryWindowDataset = STATE["telemetry_ds"]

    if seed is None:
        idx = np.random.randint(0, len(ds))
    else:
        rng = np.random.default_rng(int(seed))
        idx = int(rng.integers(0, len(ds)))

    xw, _ = ds[idx]

    # Original:
    # telemetry window shape = (15, win)
    # average across time -> (15,)
    x = (
        xw.float()
        .mean(dim=-1)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return {
        "x": x.tolist(),
        "telemetry_gen_s": float(time.time() - t0),
        "impair": {
            "cpu": STATE["cpu_impair_on"],
            "net": STATE["net_impair_on"],
        },
    }


# ------------------------------------------------------------------
# IMPAIRMENT FUNCTIONS
# ------------------------------------------------------------------

def _apply_tc_netem(
    delay_ms: float,
    loss_pct: float,
):
    cmd = (
        "sudo tc qdisc replace dev eth0 root netem "
        f"delay {int(delay_ms)}ms "
        f"loss {float(loss_pct)}%"
    )

    subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _clear_tc_netem():

    cmd = "sudo tc qdisc del dev eth0 root netem"

    subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _start_cpu_stress(workers: int):

    if workers <= 0:
        return

    cmd = (
        f"nohup stress-ng --cpu {int(workers)} "
        "--cpu-method all --timeout 600s "
        "> /tmp/stressng.log 2>&1 &"
    )

    subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_cpu_stress():

    subprocess.run(
        "pkill -f 'stress-ng --cpu' || true",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@app.post("/impair/cpu_start")
def impair_cpu_start(payload: Dict[str, Any]):

    workers = int(
        payload.get("cpu_workers", 1)
    )

    STATE["cpu_workers"] = workers
    STATE["cpu_impair_on"] = True

    _start_cpu_stress(workers)

    return {
        "ok": True,
        "cpu_workers": workers,
    }


@app.post("/impair/cpu_stop")
def impair_cpu_stop(payload: Dict[str, Any]):

    STATE["cpu_impair_on"] = False
    STATE["cpu_workers"] = 0

    _stop_cpu_stress()

    return {
        "ok": True,
    }


@app.post("/impair/net_start")
def impair_net_start(payload: Dict[str, Any]):

    delay_ms = float(
        payload.get("net_delay_ms", 120.0)
    )

    loss_pct = float(
        payload.get("net_loss_pct", 5.0)
    )

    STATE["net_delay_ms"] = delay_ms
    STATE["net_loss_pct"] = loss_pct
    STATE["net_impair_on"] = True

    _apply_tc_netem(
        delay_ms,
        loss_pct,
    )

    return {
        "ok": True,
        "delay_ms": delay_ms,
        "loss_pct": loss_pct,
    }


@app.post("/impair/net_stop")
def impair_net_stop(payload: Dict[str, Any]):

    STATE["net_impair_on"] = False
    STATE["net_delay_ms"] = 0.0
    STATE["net_loss_pct"] = 0.0

    _clear_tc_netem()

    return {
        "ok": True,
    }


# ------------------------------------------------------------------
# E1 LOCAL TRAINING
# ------------------------------------------------------------------

@app.post("/train_local")
def train_local(req: TrainReq):
    """
    Train the independent Fashion-MNIST FL model.

    Important:
      offload = 0
          -> client performs local Fashion-MNIST training.

      offload = 1
          -> preserve audited client-side semantics:
             client performs no local training and returns the
             unchanged global model.

    Actual server-side cloud pseudo-client execution for TL-PPO and
    TR-DP-AFRL is implemented separately in the E1 server runner.
    """

    device = STATE["device"]

    model = FashionMNISTCNN(
        num_classes=10
    ).to(device)

    # Load global FL model
    gsd = jsonable_to_state_dict(
        req.global_state_dict,
        device=device,
    )

    model.load_state_dict(
        gsd,
        strict=True,
    )

    t0 = time.time()

    # --------------------------------------------------------------
    # Preserve original offload/skip semantics on physical clients
    # --------------------------------------------------------------

    if int(req.offload) == 1:

        time.sleep(0.02)

        return {
            "state_dict":
                state_dict_to_jsonable(
                    model.state_dict()
                ),

            "n_samples": 0,

            "train_time_s":
                float(time.time() - t0),

            "batches": 0,
            "offload": 1,

            "dataset":
                "FashionMNIST",
        }

    # --------------------------------------------------------------
    # Obtain this client's deterministic non-IID partition
    # --------------------------------------------------------------

    client_indices = _get_client_partition(
        seed=int(req.data_seed),
        alpha=float(req.dirichlet_alpha),
        num_clients=int(req.num_clients),
    )

    train_dataset = STATE["fashion_train"]

    # --------------------------------------------------------------
    # Budgeted sampling
    #
    # Deterministic per:
    #     experiment seed
    #     round
    #     client
    #
    # This improves reproducibility across matched experiments.
    # --------------------------------------------------------------

    n_samples = min(
        int(req.max_local_samples),
        len(client_indices),
    )

    sampling_seed = (
        int(req.data_seed) * 1_000_000
        + int(req.round) * 1_000
        + int(STATE["client_index"])
    )

    rng = np.random.default_rng(
        sampling_seed
    )

    chosen = rng.choice(
        np.asarray(client_indices),
        size=n_samples,
        replace=False,
    )

    # Create subset directly from official Fashion-MNIST TRAIN split
    subset = torch.utils.data.Subset(
        train_dataset,
        chosen.tolist(),
    )

    # Deterministic DataLoader shuffle
    loader_generator = torch.Generator()

    loader_generator.manual_seed(
        sampling_seed
    )

    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=int(req.batch_size),
        shuffle=True,
        generator=loader_generator,
    )

    # --------------------------------------------------------------
    # Preserve audited optimiser / local training settings
    # --------------------------------------------------------------

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(req.lr),
        weight_decay=float(req.weight_decay),
    )

    batches_total = 0

    for _ in range(
        int(req.local_epochs)
    ):

        stats = train_one_epoch_budget(
            model,
            loader,
            opt,
            device,
            max_batches=int(
                req.max_local_batches
            ),
        )

        batches_total += int(
            stats.batches
        )

    return {
        "state_dict":
            state_dict_to_jsonable(
                model.state_dict()
            ),

        "n_samples":
            int(n_samples),

        "train_time_s":
            float(time.time() - t0),

        "batches":
            int(batches_total),

        "offload":
            0,

        "dataset":
            "FashionMNIST",

        "partition_seed":
            int(req.data_seed),

        "client_index":
            int(STATE["client_index"]),

        "client_partition_size":
            int(len(client_indices)),
    }


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--client_id",
        type=str,
        required=True,
    )

    # Explicit numerical index removes ambiguity such as
    # client1 = partition 0 vs partition 1.
    ap.add_argument(
        "--client_index",
        type=int,
        required=True,
        choices=range(10),
    )

    # Original telemetry source
    ap.add_argument(
        "--csv",
        type=str,
        required=True,
    )

    ap.add_argument(
        "--win",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--label_col",
        type=str,
        default="scenario",
    )

    ap.add_argument(
        "--class_map",
        type=str,
        default=None,
    )

    ap.add_argument(
        "--scaler_path",
        type=str,
        default=None,
    )

    ap.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
    )

    ap.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    args = ap.parse_args()

    STATE["client_id"] = args.client_id
    STATE["client_index"] = int(
        args.client_index
    )

    STATE["csv"] = args.csv
    STATE["win"] = args.win
    STATE["label_col"] = args.label_col
    STATE["class_map"] = args.class_map
    STATE["scaler_path"] = args.scaler_path

    print()
    print("E3 Fashion-MNIST client")
    print("-----------------------")
    print(
        "Client ID      :",
        STATE["client_id"],
    )
    print(
        "Client index   :",
        STATE["client_index"],
    )
    print(
        "FL task        : Fashion-MNIST"
    )
    print(
        "Controller data: original telemetry"
    )
    print()

    uvicorn.run(
        app,
        host=args.host,
        port=int(args.port),
    )


if __name__ == "__main__":
    main()
