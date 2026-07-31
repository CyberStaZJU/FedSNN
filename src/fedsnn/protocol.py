from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .data import (
    load_cifar10_unit_interval,
    load_cifar100_unit_interval,
    load_fashion_mnist_unit_interval,
    load_mnist_unit_interval,
)
from .models import (
    build_fashion_mnist_2conv2fc_bntt,
    build_fashion_mnist_vgg5_bntt,
    build_fedlec_vgg9,
    build_fedsnn_alexnet_bntt,
    build_fedsnn_mnist_bntt,
    build_fedsnn_resnet18_bntt,
    build_fedsnn_vgg5_bntt,
    build_fedsnn_vgg9_bntt,
    build_mnist_2conv2fc_bntt,
    build_snn_cifar10,
    build_sfedca_mnist,
    build_sfedca_vgg5,
    build_sfedca_vgg9,
)
from .partition import (
    class_imbalance_partition,
    dirichlet_all_classes_partition,
    dirichlet_partition,
)


@dataclass(frozen=True)
class ProtocolAssets:
    """Dataset, partition, and model assets shared by controlled baselines."""

    dataset_name: str
    train_set: Any
    test_set: Any
    partitions: tuple[np.ndarray, ...]
    model_builder: Callable[[], Any]
    partition_metadata: dict[str, Any]
    retained_indices: np.ndarray


@dataclass(frozen=True)
class ProtocolPartition:
    partitions: tuple[np.ndarray, ...]
    retained_indices: np.ndarray
    metadata: dict[str, Any]


def protocol_model_builder(
    config: dict[str, Any],
    dataset_name: str,
    timesteps: int,
    *,
    track_runtime_activity: bool = False,
) -> Callable[[], Any]:
    """Resolve an explicitly named model for the controlled protocol."""

    model_name = str(config["model"].get("name", "")).lower()
    if dataset_name == "mnist":
        if model_name == "mnist_2conv2fc_bntt":
            if str(config["model"].get("normalization", "bntt")).lower() != "bntt":
                raise ValueError("mnist_2conv2fc_bntt requires normalization: bntt")
            if bool(config["model"].get("bias", False)):
                raise ValueError("mnist_2conv2fc_bntt requires bias: false")
            if str(config["model"].get("pooling", "avg")).lower() != "avg":
                raise ValueError("mnist_2conv2fc_bntt requires pooling: avg")
            return lambda: build_mnist_2conv2fc_bntt(
                timesteps=timesteps,
                classes=10,
                surrogate_beta=float(config["model"].get("surrogate_beta", 2.0)),
                threshold=float(config["model"].get("threshold", 1.0)),
                membrane_decay=float(config["model"].get("membrane_decay", 0.95)),
                bntt_eps=float(config["model"].get("bntt_eps", 1e-4)),
                bntt_momentum=float(config["model"].get("bntt_momentum", 0.1)),
                track_runtime_activity=track_runtime_activity,
            )
        if model_name == "fedsnn_mnist_bntt":
            if str(config["model"].get("normalization", "bntt")).lower() != "bntt":
                raise ValueError("fedsnn_mnist_bntt requires normalization: bntt")
            if bool(config["model"].get("bias", False)):
                raise ValueError("fedsnn_mnist_bntt requires bias: false")
            if str(config["model"].get("pooling", "avg")).lower() != "avg":
                raise ValueError("fedsnn_mnist_bntt requires pooling: avg")
            return lambda: build_fedsnn_mnist_bntt(
                timesteps=timesteps,
                classes=10,
                surrogate_beta=float(config["model"].get("surrogate_beta", 2.0)),
                threshold=float(config["model"].get("threshold", 1.0)),
                membrane_decay=float(config["model"].get("membrane_decay", 0.95)),
                bntt_eps=float(config["model"].get("bntt_eps", 1e-4)),
                bntt_momentum=float(config["model"].get("bntt_momentum", 0.1)),
                track_runtime_activity=track_runtime_activity,
            )
        if model_name and model_name not in {"sfedca_mnist", "sfedca_mnist_2conv2fc"}:
            raise ValueError(f"unsupported MNIST protocol model: {model_name}")
        return lambda: build_sfedca_mnist(
            timesteps=timesteps,
            track_runtime_activity=track_runtime_activity,
        )
    if dataset_name == "fashion_mnist":
        if model_name == "fashion_mnist_2conv2fc_bntt":
            if str(config["model"].get("normalization", "bntt")).lower() != "bntt":
                raise ValueError("fashion_mnist_2conv2fc_bntt requires normalization: bntt")
            if bool(config["model"].get("bias", False)):
                raise ValueError("fashion_mnist_2conv2fc_bntt requires bias: false")
            if str(config["model"].get("pooling", "avg")).lower() != "avg":
                raise ValueError("fashion_mnist_2conv2fc_bntt requires pooling: avg")
            return lambda: build_fashion_mnist_2conv2fc_bntt(
                timesteps=timesteps,
                classes=10,
                surrogate_beta=float(config["model"].get("surrogate_beta", 2.0)),
                threshold=float(config["model"].get("threshold", 1.0)),
                membrane_decay=float(config["model"].get("membrane_decay", 0.95)),
                bntt_eps=float(config["model"].get("bntt_eps", 1e-4)),
                bntt_momentum=float(config["model"].get("bntt_momentum", 0.1)),
                track_runtime_activity=track_runtime_activity,
            )
        if model_name == "fashion_mnist_vgg5_bntt":
            if str(config["model"].get("normalization", "bntt")).lower() != "bntt":
                raise ValueError("fashion_mnist_vgg5_bntt requires normalization: bntt")
            if bool(config["model"].get("bias", False)):
                raise ValueError("fashion_mnist_vgg5_bntt requires bias: false")
            if str(config["model"].get("pooling", "avg")).lower() != "avg":
                raise ValueError("fashion_mnist_vgg5_bntt requires pooling: avg")
            return lambda: build_fashion_mnist_vgg5_bntt(
                timesteps=timesteps,
                classes=10,
                surrogate_beta=float(config["model"].get("surrogate_beta", 2.0)),
                threshold=float(config["model"].get("threshold", 1.0)),
                membrane_decay=float(config["model"].get("membrane_decay", 0.95)),
                bntt_eps=float(config["model"].get("bntt_eps", 1e-4)),
                bntt_momentum=float(config["model"].get("bntt_momentum", 0.1)),
                track_runtime_activity=track_runtime_activity,
            )
        if model_name and model_name != "sfedca_vgg5":
            raise ValueError(f"unsupported Fashion-MNIST protocol model: {model_name}")
        return lambda: build_sfedca_vgg5(
            timesteps=timesteps,
            track_runtime_activity=track_runtime_activity,
        )
    if dataset_name == "cifar100":
        if model_name != "fedsnn_resnet18_bntt":
            raise ValueError(f"unsupported CIFAR-100 protocol model: {model_name}")
        if str(config["model"].get("normalization", "bntt")).lower() != "bntt":
            raise ValueError("fedsnn_resnet18_bntt requires normalization: bntt")
        if bool(config["model"].get("bias", False)):
            raise ValueError("fedsnn_resnet18_bntt requires bias: false")
        if str(config["model"].get("pooling", "avg")).lower() != "avg":
            raise ValueError("fedsnn_resnet18_bntt requires pooling: avg")
        return lambda: build_fedsnn_resnet18_bntt(
            timesteps=timesteps,
            classes=100,
            surrogate_beta=float(config["model"].get("surrogate_beta", 2.0)),
            threshold=float(config["model"].get("threshold", 1.0)),
            membrane_decay=float(config["model"].get("membrane_decay", 0.95)),
            bntt_eps=float(config["model"].get("bntt_eps", 1e-4)),
            bntt_momentum=float(config["model"].get("bntt_momentum", 0.1)),
            track_runtime_activity=track_runtime_activity,
        )
    if dataset_name != "cifar10":
        raise ValueError(f"unsupported SFedCA-protocol dataset: {dataset_name}")
    if model_name == "snn_cifar10":
        retired_fields = {
            "hidden_features",
            "dropout",
            "batch_norm",
            "bntt_eps",
            "bntt_momentum",
        } & set(
            config["model"]
        )
        if retired_fields:
            names = ", ".join(sorted(retired_fields))
            raise ValueError(f"retired snn_cifar10 model fields: {names}")
        if str(config["model"].get("normalization", "none")).lower() != "none":
            raise ValueError("snn_cifar10 requires normalization: none")
        if not bool(config["model"].get("bias", True)):
            raise ValueError("snn_cifar10 requires bias: true")
        if str(config["model"].get("pooling", "max")).lower() != "max":
            raise ValueError("snn_cifar10 requires pooling: max")
        return lambda: build_snn_cifar10(
            timesteps=timesteps,
            classes=10,
            surrogate_beta=float(config["model"].get("surrogate_beta", 2.0)),
            track_runtime_activity=track_runtime_activity,
        )
    if model_name == "fedsnn_vgg5_bntt":
        if str(config["model"].get("normalization", "bntt")).lower() != "bntt":
            raise ValueError("fedsnn_vgg5_bntt requires normalization: bntt")
        if bool(config["model"].get("bias", False)):
            raise ValueError("fedsnn_vgg5_bntt requires bias: false")
        if str(config["model"].get("pooling", "avg")).lower() != "avg":
            raise ValueError("fedsnn_vgg5_bntt requires pooling: avg")
        return lambda: build_fedsnn_vgg5_bntt(
            timesteps=timesteps,
            classes=10,
            surrogate_beta=float(config["model"].get("surrogate_beta", 2.0)),
            threshold=float(config["model"].get("threshold", 1.0)),
            membrane_decay=float(config["model"].get("membrane_decay", 0.95)),
            bntt_eps=float(config["model"].get("bntt_eps", 1e-4)),
            bntt_momentum=float(config["model"].get("bntt_momentum", 0.1)),
            track_runtime_activity=track_runtime_activity,
        )
    if model_name == "fedsnn_vgg9_bntt":
        if str(config["model"].get("normalization", "bntt")).lower() != "bntt":
            raise ValueError("fedsnn_vgg9_bntt requires normalization: bntt")
        if bool(config["model"].get("bias", False)):
            raise ValueError("fedsnn_vgg9_bntt requires bias: false")
        if str(config["model"].get("pooling", "avg")).lower() != "avg":
            raise ValueError("fedsnn_vgg9_bntt requires pooling: avg")
        return lambda: build_fedsnn_vgg9_bntt(
            timesteps=timesteps,
            classes=10,
            surrogate_beta=float(config["model"].get("surrogate_beta", 2.0)),
            threshold=float(config["model"].get("threshold", 1.0)),
            membrane_decay=float(config["model"].get("membrane_decay", 0.95)),
            bntt_eps=float(config["model"].get("bntt_eps", 1e-4)),
            bntt_momentum=float(config["model"].get("bntt_momentum", 0.1)),
            track_runtime_activity=track_runtime_activity,
        )
    if model_name == "fedsnn_alexnet_bntt":
        if str(config["model"].get("normalization", "bntt")).lower() != "bntt":
            raise ValueError("fedsnn_alexnet_bntt requires normalization: bntt")
        if bool(config["model"].get("bias", False)):
            raise ValueError("fedsnn_alexnet_bntt requires bias: false")
        if str(config["model"].get("pooling", "avg")).lower() != "avg":
            raise ValueError("fedsnn_alexnet_bntt requires pooling: avg")
        return lambda: build_fedsnn_alexnet_bntt(
            timesteps=timesteps,
            classes=10,
            surrogate_beta=float(config["model"].get("surrogate_beta", 2.0)),
            threshold=float(config["model"].get("threshold", 1.0)),
            membrane_decay=float(config["model"].get("membrane_decay", 0.95)),
            bntt_eps=float(config["model"].get("bntt_eps", 1e-4)),
            bntt_momentum=float(config["model"].get("bntt_momentum", 0.1)),
            track_runtime_activity=track_runtime_activity,
            execution_backend=str(
                config["model"].get("execution_backend", "legacy_stepwise")
            ),
            execution_backend_strict=bool(
                config["model"].get("execution_backend_strict", False)
            ),
        )
    if model_name == "sfedca_vgg9":
        if str(config["model"].get("normalization", "none")).lower() != "none":
            raise ValueError("sfedca_vgg9 requires normalization: none")
        if not bool(config["model"].get("bias", True)):
            raise ValueError("sfedca_vgg9 requires bias: true")
        if str(config["model"].get("pooling", "max")).lower() != "max":
            raise ValueError("sfedca_vgg9 requires pooling: max")
        return lambda: build_sfedca_vgg9(
            timesteps=timesteps,
            classes=10,
            surrogate_beta=float(config["model"].get("surrogate_beta", 2.0)),
            track_runtime_activity=track_runtime_activity,
        )
    if model_name == "fedlec_vgg9_snn":
        return lambda: build_fedlec_vgg9(
            timesteps=timesteps,
            classes=10,
            tau=float(config["model"].get("tau", 2.0)),
            threshold=float(config["model"].get("threshold", 1.0)),
            surrogate_beta=float(config["model"].get("surrogate_beta", 2.0)),
            track_runtime_activity=track_runtime_activity,
        )
    raise ValueError(f"unsupported CIFAR-10 protocol model: {model_name}")


def load_protocol_dataset_and_model(
    config: dict[str, Any],
    data_root: str | Path,
    *,
    timesteps: int,
    download: bool = False,
    track_runtime_activity: bool = False,
) -> tuple[str, Any, Any, Callable[[], Any]]:
    """Load protocol data and resolve its configured model without partitioning."""

    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    dataset_name = str(config["dataset"]["name"]).lower()
    if dataset_name == "mnist":
        train_set = load_mnist_unit_interval(data_root, train=True, download=download)
        test_set = load_mnist_unit_interval(data_root, train=False, download=download)
    elif dataset_name == "fashion_mnist":
        train_set = load_fashion_mnist_unit_interval(
            data_root, train=True, download=download
        )
        test_set = load_fashion_mnist_unit_interval(
            data_root, train=False, download=download
        )
    elif dataset_name == "cifar10":
        train_set = load_cifar10_unit_interval(data_root, train=True, download=download)
        test_set = load_cifar10_unit_interval(data_root, train=False, download=download)
    elif dataset_name == "cifar100":
        train_set = load_cifar100_unit_interval(data_root, train=True, download=download)
        test_set = load_cifar100_unit_interval(data_root, train=False, download=download)
    else:
        raise ValueError(f"unsupported SFedCA-protocol dataset: {dataset_name}")
    builder = protocol_model_builder(
        config,
        dataset_name,
        timesteps,
        track_runtime_activity=track_runtime_activity,
    )
    return dataset_name, train_set, test_set, builder


def partition_protocol_labels(
    config: dict[str, Any], labels: Any, *, min_samples: int
) -> ProtocolPartition:
    """Resolve one of the three final-paper SFedCA partition reconstructions."""

    labels_array = np.asarray(labels)
    dataset = config["dataset"]
    name = str(dataset["partition"]).lower()
    clients = int(config["federation"]["clients"])
    alpha = float(dataset.get("alpha", 0.3))
    seed = int(config["training"]["seed"])
    if name in {"dirichlet", "dir", "dirichlet_classwise"}:
        partitions = dirichlet_partition(
            labels_array, clients, alpha, seed, min_samples=min_samples
        )
        retained = np.arange(len(labels_array), dtype=np.int64)
        metadata = {
            "name": "Dir(alpha)",
            "alpha": alpha,
            "fidelity": "paper_reconstruction_missing_official_partition_code",
            "retained_num_samples": int(len(retained)),
            "discarded_num_samples": 0,
        }
    elif name in {
        "dir_n",
        "dirn",
        "dirichlet_n",
        "dirichlet_100",
        "dir_100",
        "dirichlet_all_classes",
    }:
        partitions = dirichlet_all_classes_partition(
            labels_array, clients, alpha, seed, min_samples=min_samples
        )
        retained = np.arange(len(labels_array), dtype=np.int64)
        metadata = {
            "name": "Dir_N(alpha)",
            "alpha": alpha,
            "fidelity": "paper_reconstruction_missing_official_partition_code",
            "all_classes_per_client": True,
            "retained_num_samples": int(len(retained)),
            "discarded_num_samples": 0,
        }
    elif name in {"class_imbalance", "ci"}:
        ratio = dataset.get("majority_to_minority", [3, 1])
        partitions, retained, metadata = class_imbalance_partition(
            labels_array,
            clients,
            alpha,
            seed,
            min_samples=min_samples,
            majority_to_minority=tuple(ratio),
        )
        metadata = {"name": "CI(n1:n2; alpha)", **metadata}
    else:
        raise ValueError(f"unsupported SFedCA-protocol partition: {name}")
    return ProtocolPartition(tuple(partitions), retained, metadata)


def load_protocol_assets(
    config: dict[str, Any],
    data_root: str | Path,
    *,
    timesteps: int | None = None,
    download: bool = False,
) -> ProtocolAssets:
    """Resolve the common SFedCA-controlled dataset, model, and split.

    Paper-specific reproduction remains in the individual paper trainers. This
    helper is deliberately limited to the controlled three-dataset protocol so
    that every compared method receives the same samples and architecture.
    """

    if timesteps is None:
        timesteps = int(config["model"]["timesteps"])
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")

    dataset_name, train_set, test_set, model_builder = load_protocol_dataset_and_model(
        config,
        data_root,
        timesteps=timesteps,
        download=download,
        track_runtime_activity=False,
    )

    partition = partition_protocol_labels(
        config,
        train_set.targets,
        # Dir(alpha=0.1) intentionally creates very small clients. A batch size
        # is an upper bound, not a requirement to discard those clients.
        min_samples=1,
    )
    return ProtocolAssets(
        dataset_name=dataset_name,
        train_set=train_set,
        test_set=test_set,
        partitions=partition.partitions,
        model_builder=model_builder,
        partition_metadata=partition.metadata,
        retained_indices=partition.retained_indices,
    )


def clone_state(state: dict[str, Any], device: Any = "cpu") -> dict[str, Any]:
    """Detach and clone a state dict directly onto the requested device."""

    return {
        key: value.detach().to(device=device, copy=True)
        for key, value in state.items()
    }


def cpu_tensor_tree(value: Any) -> Any:
    """Create a detached CPU checkpoint snapshot of a nested tensor payload."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("PyTorch is required for checkpoint state handling") from exc
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tensor_tree(item) for item in value)
    return value


def floating_layout(state: dict[str, Any]) -> tuple[tuple[str, tuple[int, ...], int], ...]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("PyTorch is required for protocol state handling") from exc
    return tuple(
        (key, tuple(value.shape), value.numel())
        for key, value in state.items()
        if torch.is_floating_point(value) or torch.is_complex(value)
    )


def flatten_state_difference(
    local_state: dict[str, Any],
    base_state: dict[str, Any],
    layout: tuple[tuple[str, tuple[int, ...], int], ...],
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("PyTorch is required for protocol state handling") from exc
    if set(local_state) != set(base_state):
        raise ValueError("local and base states must have identical keys")
    return torch.cat(
        [(local_state[key] - base_state[key]).reshape(-1) for key, _, _ in layout]
    )


def apply_flat_update(
    global_state: dict[str, Any],
    flat_update: Any,
    layout: tuple[tuple[str, tuple[int, ...], int], ...],
    *,
    step_size: float = 1.0,
) -> dict[str, Any]:
    if not global_state:
        raise ValueError("global_state must not be empty")
    state_device = next(iter(global_state.values())).device
    result = clone_state(global_state, state_device)
    offset = 0
    for key, shape, count in layout:
        result[key].add_(
            flat_update[offset : offset + count].reshape(shape), alpha=float(step_size)
        )
        offset += count
    if offset != flat_update.numel():
        raise ValueError("flat update does not match the state layout")
    return result


def model_wire_bytes(state: dict[str, Any]) -> int:
    """Count a dense state dict exactly as represented by its tensor dtypes."""

    return sum(value.numel() * value.element_size() for value in state.values())


def bits_to_bytes(bits: int) -> int:
    if isinstance(bits, bool) or not isinstance(bits, int) or bits < 0:
        raise ValueError("bits must be a non-negative integer")
    return (bits + 7) // 8


def reconcile_metrics_for_resume(metrics_path: Path, checkpoint_round: int) -> None:
    """Retain exactly the contiguous metric prefix committed by a checkpoint."""

    if checkpoint_round < 0:
        raise ValueError("checkpoint_round must be non-negative")
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"cannot resume without the matching metrics file: {metrics_path}"
        )
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid metrics JSON on line {line_number}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("round"), int):
            raise ValueError(f"metrics line {line_number} has no integer round")
        records.append(record)

    expected = list(range(checkpoint_round + 1))
    observed = [record["round"] for record in records]
    if observed[: len(expected)] != expected:
        raise ValueError(
            "metrics do not contain the contiguous prefix committed by the checkpoint"
        )
    retained = records[: len(expected)]
    temporary = metrics_path.with_suffix(metrics_path.suffix + ".resume.tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in retained),
        encoding="utf-8",
    )
    temporary.replace(metrics_path)
