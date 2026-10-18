import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


DATA_ROOT = Path("/home/ubuntu/afrl/data/fashion_mnist")


def get_fashion_mnist_datasets():
    """
    Load the official Fashion-MNIST train and test splits.

    IMPORTANT:
    - Only the official TRAIN split is partitioned across FL clients.
    - The official TEST split is reserved for global evaluation only.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


def make_dirichlet_partitions(
    train_dataset: Dataset,
    num_clients: int = 10,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples_per_client: int = 100
) -> Dict[int, List[int]]:
    """
    Create deterministic non-IID label-skew partitions using a
    Dirichlet distribution.

    Returns:
        {
            client_id: [dataset indices]
        }
    """

    targets = np.asarray(train_dataset.targets)
    num_classes = int(targets.max()) + 1

    attempt = 0

    while True:
        rng = np.random.default_rng(seed + attempt)

        client_indices = {
            client_id: []
            for client_id in range(num_clients)
        }

        for class_id in range(num_classes):
            class_indices = np.where(targets == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(
                np.repeat(alpha, num_clients)
            )

            split_points = (
                np.cumsum(proportions)[:-1] * len(class_indices)
            ).astype(int)

            class_splits = np.split(
                class_indices,
                split_points
            )

            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(
                    split.tolist()
                )

        sizes = [
            len(client_indices[c])
            for c in range(num_clients)
        ]

        if min(sizes) >= min_samples_per_client:
            break

        attempt += 1

        if attempt > 1000:
            raise RuntimeError(
                "Could not create a valid Dirichlet partition "
                f"with at least {min_samples_per_client} samples "
                "per client."
            )

    final_rng = np.random.default_rng(seed)

    for client_id in range(num_clients):
        final_rng.shuffle(client_indices[client_id])

    return client_indices


def build_client_subsets(
    train_dataset: Dataset,
    partitions: Dict[int, List[int]]
) -> Dict[int, Subset]:
    """
    Build PyTorch Subset objects for each FL client.
    """

    return {
        client_id: Subset(train_dataset, indices)
        for client_id, indices in partitions.items()
    }


def make_partition_manifest(
    train_dataset: Dataset,
    partitions: Dict[int, List[int]],
    alpha: float,
    seed: int
) -> dict:
    """
    Create a reproducibility manifest describing the class
    distribution assigned to every client.
    """

    targets = np.asarray(train_dataset.targets)
    num_classes = int(targets.max()) + 1

    manifest = {
        "dataset": "FashionMNIST",
        "partition_type": "Dirichlet label skew",
        "alpha": alpha,
        "seed": seed,
        "num_clients": len(partitions),
        "official_train_size": len(train_dataset),
        "clients": {}
    }

    all_assigned = []

    for client_id, indices in partitions.items():
        indices_array = np.asarray(indices, dtype=int)

        class_counts = {
            str(class_id): int(
                np.sum(targets[indices_array] == class_id)
            )
            for class_id in range(num_classes)
        }

        manifest["clients"][str(client_id)] = {
            "num_samples": len(indices),
            "class_counts": class_counts
        }

        all_assigned.extend(indices)

    manifest["total_assigned_samples"] = len(all_assigned)
    manifest["unique_assigned_samples"] = len(set(all_assigned))

    return manifest


def save_partition_manifest(
    manifest: dict,
    output_path: str
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def validate_partitions(
    train_dataset: Dataset,
    test_dataset: Dataset,
    partitions: Dict[int, List[int]]
):
    """
    Validate that:

    1. every training sample is assigned exactly once;
    2. there is no overlap between client partitions;
    3. test samples are never used in client partitions.
    """

    all_indices = []

    for indices in partitions.values():
        all_indices.extend(indices)

    if len(all_indices) != len(train_dataset):
        raise AssertionError(
            "Not all training samples were assigned."
        )

    if len(set(all_indices)) != len(train_dataset):
        raise AssertionError(
            "Duplicate training samples found across clients."
        )

    expected = set(range(len(train_dataset)))

    if set(all_indices) != expected:
        raise AssertionError(
            "Partition indices do not exactly match the "
            "official training split."
        )

    # The torchvision train and test datasets are separate objects.
    # Client partitions contain only train-dataset indices.
    assert len(test_dataset) == 10000

    return True


if __name__ == "__main__":

    NUM_CLIENTS = 10
    ALPHA = 0.5
    SEED = 42

    train_dataset, test_dataset = (
        get_fashion_mnist_datasets()
    )

    partitions = make_dirichlet_partitions(
        train_dataset=train_dataset,
        num_clients=NUM_CLIENTS,
        alpha=ALPHA,
        seed=SEED
    )

    validate_partitions(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        partitions=partitions
    )

    manifest = make_partition_manifest(
        train_dataset=train_dataset,
        partitions=partitions,
        alpha=ALPHA,
        seed=SEED
    )

    manifest_path = (
        f"/home/ubuntu/afrl/e1_manifests/"
        f"fashion_partition_seed{SEED}.json"
    )

    save_partition_manifest(
        manifest,
        manifest_path
    )

    print()
    print("Fashion-MNIST E1 partition validation")
    print("------------------------------------")
    print("Train samples:", len(train_dataset))
    print("Test samples :", len(test_dataset))
    print("Clients      :", NUM_CLIENTS)
    print("Dirichlet α  :", ALPHA)
    print("Seed         :", SEED)
    print()

    for client_id in range(NUM_CLIENTS):
        info = manifest["clients"][str(client_id)]

        print(
            f"Client {client_id:02d}: "
            f"{info['num_samples']:5d} samples | "
            f"class counts = {info['class_counts']}"
        )

    print()
    print(
        "Total assigned :",
        manifest["total_assigned_samples"]
    )

    print(
        "Unique assigned:",
        manifest["unique_assigned_samples"]
    )

    print()
    print("Partition validation: PASS")
    print("Manifest:", manifest_path)
