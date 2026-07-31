from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _validate_partition_inputs(
    labels: Sequence[int] | np.ndarray,
    num_clients: int,
    alpha: float,
    min_samples: int,
) -> np.ndarray:
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1 or len(labels_array) == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if (
        isinstance(num_clients, bool)
        or not isinstance(num_clients, (int, np.integer))
        or num_clients <= 0
        or isinstance(min_samples, bool)
        or not isinstance(min_samples, (int, np.integer))
        or min_samples < 0
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("invalid partition parameters")
    if len(labels_array) < num_clients * min_samples:
        raise ValueError("not enough samples to satisfy min_samples")
    return labels_array


def _largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    """Allocate an integer total proportionally with deterministic tie breaks."""

    if total < 0 or weights.ndim != 1 or len(weights) == 0:
        raise ValueError("invalid integer allocation inputs")
    if total == 0:
        return np.zeros(len(weights), dtype=np.int64)
    normalized = weights / weights.sum()
    raw = normalized * total
    allocated = np.floor(raw).astype(np.int64)
    remainder = total - int(allocated.sum())
    if remainder:
        fractions = raw - allocated
        order = np.lexsort((np.arange(len(weights)), -fractions))
        allocated[order[:remainder]] += 1
    return allocated


def dirichlet_client_size_partition(
    num_samples: int,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples: int = 1,
) -> list[np.ndarray]:
    """Partition sample indices using only Dirichlet-distributed client sizes.

    Unlike a class-wise Dirichlet split, this primitive does not impose or draw
    client label proportions. It first draws client capacities from
    ``Dirichlet(alpha)`` and then assigns a shuffled global sample pool to those
    capacities. This is the sample-size-only rule used by SFedCA's ``CI``
    definition and the unconstrained component underlying ``Dir_N``; their
    distinct class constraints are handled by their callers.
    """

    if (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, (int, np.integer))
        or num_samples <= 0
        or isinstance(num_clients, bool)
        or not isinstance(num_clients, (int, np.integer))
        or num_clients <= 0
        or isinstance(min_samples, bool)
        or not isinstance(min_samples, (int, np.integer))
        or min_samples < 0
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("invalid client-size partition parameters")
    if num_samples < num_clients * min_samples:
        raise ValueError("not enough samples to satisfy min_samples")

    rng = np.random.default_rng(seed)
    lower_bounds = np.full(num_clients, min_samples, dtype=np.int64)
    weights = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
    sizes = lower_bounds + _largest_remainder(
        num_samples - int(lower_bounds.sum()), weights
    )
    permutation = rng.permutation(num_samples)
    cuts = np.cumsum(sizes)[:-1]
    partitions = [
        np.asarray(sorted(chunk.tolist()), dtype=np.int64)
        for chunk in np.split(permutation, cuts)
    ]
    merged = np.concatenate(partitions)
    if not np.array_equal(
        np.sort(merged), np.arange(num_samples, dtype=np.int64)
    ):
        raise AssertionError("client-size partition lost or duplicated samples")
    if min(map(len, partitions)) < min_samples:
        raise AssertionError("client-size partition violated min_samples")
    return partitions


def dirichlet_all_classes_partition(
    labels: Sequence[int] | np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples: int = 1,
) -> list[np.ndarray]:
    """Reconstruct SFedCA's ``Dir_N(alpha)`` partition.

    The final SFedCA paper says that every client contains every class while
    client sample totals follow a Dirichlet distribution, but it does not
    publish the allocation code.  This deterministic reconstruction maximizes
    per-class client coverage first, then draws the *residual* client sizes
    from ``Dirichlet(alpha)`` subject to ``min_samples`` and the coverage
    seeds.  If every class has at least ``num_clients`` examples, every client
    is guaranteed to contain every class.

    Every input index is assigned exactly once.  Sorting the returned indices
    makes equality and provenance checks independent of construction order.
    """

    labels_array = _validate_partition_inputs(labels, num_clients, alpha, min_samples)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels_array)
    clients: list[list[int]] = [[] for _ in range(num_clients)]

    # Maximize coverage without replacement.  Rotating a fresh client order
    # per class also balances unavoidable gaps when a class is scarce.
    remaining: list[int] = []
    for label in classes:
        indices = np.flatnonzero(labels_array == label)
        rng.shuffle(indices)
        covered_clients = min(len(indices), num_clients)
        client_order = rng.permutation(num_clients)
        for sample_index, client_index in zip(
            indices[:covered_clients], client_order[:covered_clients]
        ):
            clients[int(client_index)].append(int(sample_index))
        remaining.extend(indices[covered_clients:].tolist())

    coverage_sizes = np.asarray([len(client) for client in clients], dtype=np.int64)
    lower_bounds = np.maximum(coverage_sizes, min_samples)
    if int(lower_bounds.sum()) > len(labels_array):
        raise ValueError(
            "not enough samples to satisfy both min_samples and maximal class coverage"
        )

    # The lower bounds are necessary to preserve coverage.  Dirichlet controls
    # all remaining sample-count heterogeneity rather than silently violating
    # those bounds after rounding.
    weights = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
    target_sizes = lower_bounds + _largest_remainder(
        len(labels_array) - int(lower_bounds.sum()), weights
    )
    capacities = target_sizes - coverage_sizes

    remaining_array = np.asarray(remaining, dtype=np.int64)
    rng.shuffle(remaining_array)
    offset = 0
    for client, capacity in zip(clients, capacities):
        next_offset = offset + int(capacity)
        client.extend(remaining_array[offset:next_offset].tolist())
        offset = next_offset
    if offset != len(remaining_array):
        raise AssertionError("partition capacities did not consume all samples")

    arrays = [np.asarray(sorted(client), dtype=np.int64) for client in clients]
    merged = np.concatenate(arrays)
    if len(merged) != len(labels_array) or not np.array_equal(
        np.sort(merged), np.arange(len(labels_array), dtype=np.int64)
    ):
        raise AssertionError("partition lost or duplicated samples")
    if min(map(len, arrays)) < min_samples:
        raise AssertionError("partition violated min_samples")
    return arrays


def class_imbalance_partition(
    labels: Sequence[int] | np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples: int = 1,
    majority_to_minority: tuple[int, int] = (3, 1),
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    """Reconstruct SFedCA's ``CI(n1:n2; alpha)`` partition.

    Classes are sorted, with the first five treated as the majority group and
    the last five as the minority group.  The largest feasible equal per-class
    sample counts satisfying ``n1:n2`` are retained without replacement.  The
    retained samples are then shuffled into client capacities drawn from
    :func:`dirichlet_client_size_partition`. CI does not inherit ``Dir_N``'s
    all-classes-per-client constraint.

    Returned partition indices always refer to the original ``labels`` array.
    ``retained_indices`` and metadata make the intentional downsampling
    explicit, so discarded examples cannot be mistaken for training data.
    """

    labels_array = _validate_partition_inputs(labels, num_clients, alpha, min_samples)
    classes = np.unique(labels_array)
    if len(classes) != 10:
        raise ValueError(
            "SFedCA class-imbalance reconstruction requires exactly 10 classes"
        )
    if (
        len(majority_to_minority) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in majority_to_minority
        )
        or majority_to_minority[0] <= 0
        or majority_to_minority[1] <= 0
    ):
        raise ValueError("majority_to_minority must contain two positive integers")

    majority_factor, minority_factor = map(int, majority_to_minority)
    common = int(np.gcd(majority_factor, minority_factor))
    majority_factor //= common
    minority_factor //= common
    majority_classes = classes[:5]
    minority_classes = classes[5:]
    per_class_available = {
        label.item() if hasattr(label, "item") else label: int(
            np.count_nonzero(labels_array == label)
        )
        for label in classes
    }
    units = min(
        min(
            per_class_available[label.item() if hasattr(label, "item") else label]
            // majority_factor
            for label in majority_classes
        ),
        min(
            per_class_available[label.item() if hasattr(label, "item") else label]
            // minority_factor
            for label in minority_classes
        ),
    )
    if units <= 0:
        raise ValueError("not enough samples to construct the requested class ratio")

    rng = np.random.default_rng(seed)
    retained_chunks: list[np.ndarray] = []
    retained_per_class: dict[Any, int] = {}
    for label in classes:
        label_key = label.item() if hasattr(label, "item") else label
        target = units * (
            majority_factor if label in majority_classes else minority_factor
        )
        indices = np.flatnonzero(labels_array == label)
        rng.shuffle(indices)
        retained_chunks.append(indices[:target])
        retained_per_class[label_key] = int(target)

    retained_indices = np.sort(np.concatenate(retained_chunks)).astype(np.int64)
    if len(retained_indices) < num_clients * min_samples:
        raise ValueError("retained class-imbalanced subset cannot satisfy min_samples")

    partition_seed = int(rng.integers(0, np.iinfo(np.int64).max))
    relative_partitions = dirichlet_client_size_partition(
        len(retained_indices),
        num_clients,
        alpha,
        partition_seed,
        min_samples=min_samples,
    )
    partitions = [
        np.asarray(retained_indices[relative], dtype=np.int64)
        for relative in relative_partitions
    ]
    merged = np.concatenate(partitions)
    if not np.array_equal(np.sort(merged), retained_indices):
        raise AssertionError("class-imbalance partition does not match retained_indices")

    majority_count = int(
        sum(
            retained_per_class[label.item() if hasattr(label, "item") else label]
            for label in majority_classes
        )
    )
    minority_count = int(
        sum(
            retained_per_class[label.item() if hasattr(label, "item") else label]
            for label in minority_classes
        )
    )
    metadata: dict[str, Any] = {
        "fidelity": "paper_reconstruction_missing_official_partition_code",
        "seed": int(seed),
        "partition_seed": partition_seed,
        "alpha": float(alpha),
        "min_samples": int(min_samples),
        "majority_classes": [
            label.item() if hasattr(label, "item") else label for label in majority_classes
        ],
        "minority_classes": [
            label.item() if hasattr(label, "item") else label for label in minority_classes
        ],
        "majority_to_minority": [majority_factor, minority_factor],
        "original_num_samples": int(len(labels_array)),
        "retained_num_samples": int(len(retained_indices)),
        "discarded_num_samples": int(len(labels_array) - len(retained_indices)),
        "retained_majority_samples": majority_count,
        "retained_minority_samples": minority_count,
        "retained_per_class": retained_per_class,
        "client_size_distribution": "dirichlet",
        "all_classes_per_client_required": False,
        "client_sizes": [int(len(indices)) for indices in partitions],
    }
    return partitions, retained_indices, metadata


def iid_partition(num_samples: int, num_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.asarray(x, dtype=np.int64) for x in np.array_split(rng.permutation(num_samples), num_clients)]


def dirichlet_partition(
    labels: Sequence[int] | np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples: int = 1,
    max_attempts: int = 1000,
) -> list[np.ndarray]:
    """Class-wise Dirichlet split with deterministic retries and no sample loss."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if num_clients <= 0 or alpha <= 0 or min_samples < 0:
        raise ValueError("invalid partition parameters")
    if len(labels) < num_clients * min_samples:
        raise ValueError("not enough samples to satisfy min_samples")

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(max_attempts):
        clients: list[list[int]] = [[] for _ in range(num_clients)]
        for label in classes:
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
            for client, chunk in zip(clients, np.split(indices, cuts)):
                client.extend(chunk.tolist())
        if min(map(len, clients)) >= min_samples:
            arrays = [np.asarray(sorted(x), dtype=np.int64) for x in clients]
            merged = np.concatenate(arrays) if arrays else np.array([], dtype=np.int64)
            if len(merged) != len(labels) or len(np.unique(merged)) != len(labels):
                raise AssertionError("partition lost or duplicated samples")
            return arrays
    raise RuntimeError(f"failed to satisfy min_samples after {max_attempts} attempts")


def balanced_dirichlet_partition(
    labels: Sequence[int] | np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples: int = 10,
    max_attempts: int = 1000,
) -> list[np.ndarray]:
    """FedSNN upstream's capped class-wise Dirichlet partition.

    The cap suppresses allocations to clients that already reached the average
    dataset size. RandomState intentionally matches the legacy NumPy API used by
    the official implementation.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or num_clients <= 0 or alpha <= 0:
        raise ValueError("invalid partition parameters")
    rng = np.random.RandomState(seed)
    target_size = len(labels) / num_clients
    for _ in range(max_attempts):
        clients: list[list[int]] = [[] for _ in range(num_clients)]
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions *= np.asarray([len(client) < target_size for client in clients])
            if proportions.sum() == 0:
                proportions = np.ones(num_clients)
            proportions /= proportions.sum()
            cuts = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
            clients = [client + chunk.tolist() for client, chunk in zip(clients, np.split(indices, cuts))]
        if min(map(len, clients)) >= min_samples:
            for client in clients:
                rng.shuffle(client)
            arrays = [np.asarray(client, dtype=np.int64) for client in clients]
            merged = np.concatenate(arrays)
            if len(merged) != len(labels) or len(np.unique(merged)) != len(labels):
                raise AssertionError("partition lost or duplicated samples")
            return arrays
    raise RuntimeError(f"failed to satisfy min_samples after {max_attempts} attempts")
