from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import yaml

from .aggregation import allocate_topk_budget, equal_topk_budget, harmonic_cache_weights
from .config import load_config, result_dir
from .device import activate_device, resolve_device, seed_everything
from .protocol import (
    load_protocol_dataset_and_model,
    partition_protocol_labels,
    reconcile_metrics_for_resume,
)
from .runtime import (
    StaticBatchCudaGraph,
    model_forward_runner,
    model_forward_runtime_metadata,
)
from .train import _append_jsonl, _atomic_torch_save, _code_commit, _evaluate
from .train_sfedca import _class_firing_rates


SAW_METHODS = {
    "dense_saw_snn",
    "topk_saw_snn",
    "credit_topk_saw_snn",
    "probe_credit_topk_saw_snn",
    "training_integrated_credit_topk_saw_snn",
    "probe_credit_topk_saw_ef_snn",
    "training_integrated_credit_topk_saw_ef_snn",
}
PURE_TOPK_METHODS = {
    "topk_fedavg_snn",
    "probe_credit_topk_snn",
    "training_integrated_credit_topk_snn",
    # Dual equal Top-K both ways, NO error feedback (uplink or downlink).
    "dual_topk_fedavg_snn",
    # Bidirectional global Top-K: uplink |Δ| top-k, downlink |model| top-k.
    # No residual conservation; sparse coords outside the selection lag forever.
    "dual_global_topk_snn",
    # D1: per-client catch-up gap residual downlink (no periodic full refresh).
    "gap_residual_dual_topk_snn",
    # D2 (archived exploratory): dual-channel quota on per-client gap.
    # Code retained for reproducibility; configs live under configs/archive/.
    "dual_channel_quota_dual_topk_snn",
}
TOPK_EF_METHODS = {
    "probe_credit_topk_ef_snn",
    "training_integrated_credit_topk_ef_snn",
    "anchor_credit_topk_ef_snn",
    "anchor_credit_topk_ef_ema_snn",
    "double_credit_topk_ef_snn",
    # Dual equal Top-K+EF both ways (no temporal credit on downlink).
    # Control for double_credit_topk_ef_snn: isolates downlink credit scoring.
    "dual_topk_fedavg_ef_snn",
    # Uplink anchor-credit Top-K+EF + downlink temporal-credit Top-K+server-EF.
    "double_anchor_credit_topk_ef_snn",
    "double_anchor_neuron_topk_ef_snn",
    # dual_global uplink (|Δ| top-k, no client EF) + server downlink EF on
    # shared global increment residual (credit_mode=none). Contrasts pure
    # |model| absolute-write dual_global_topk_snn.
    "dual_global_topk_ef_snn",
    # Scheme A: bidirectional architecture-aligned *block* Top-K + server
    # whole-block downlink EF (shared base). UL is block-RMS of local |Δ|
    # without client EF.
    "symmetric_block_dual_topk_ef_snn",
    # Scheme A follow-up: block-RMS shortlist, block-local coordinate support,
    # real signed INT8 values, and quantization-aware server EF.
    "symmetric_block_local_int8_dual_topk_ef_snn",
}
CURRENT_ROUND_TOPK_METHODS = PURE_TOPK_METHODS | TOPK_EF_METHODS
NO_ERROR_FEEDBACK_METHODS = PURE_TOPK_METHODS | {
    "probe_credit_topk_saw_snn",
    "training_integrated_credit_topk_saw_snn",
    # Uplink is pure top-k / block top-k; only the server downlink residual uses EF.
    "dual_global_topk_ef_snn",
    "symmetric_block_dual_topk_ef_snn",
    "symmetric_block_local_int8_dual_topk_ef_snn",
}
# Authorized dual-budget matrix methods (CIFAR-10 UL/DL ratio experiments).
# D2 remains listed so archived configs still validate if re-run intentionally.
DUAL_BUDGET_METHODS = {
    "dual_global_topk_snn",
    "dual_global_topk_ef_snn",
    "gap_residual_dual_topk_snn",
    "dual_channel_quota_dual_topk_snn",
    "symmetric_block_dual_topk_ef_snn",
    "symmetric_block_local_int8_dual_topk_ef_snn",
}
# Methods that keep a distinct sparse base per client and catch up before train.
PER_CLIENT_GAP_DOWNLINK_METHODS = {
    "gap_residual_dual_topk_snn",
    "dual_channel_quota_dual_topk_snn",
}
# Shared-base |model| absolute-write downlink (no server EF residual).
DUAL_GLOBAL_MODEL_DOWNLINK_METHODS = {
    "dual_global_topk_snn",
}
# Shared-base |Δ|+server-EF downlink (no uplink EF; credit_mode forced none).
DUAL_GLOBAL_SERVER_EF_METHODS = {
    "dual_global_topk_ef_snn",
    "symmetric_block_dual_topk_ef_snn",
    "symmetric_block_local_int8_dual_topk_ef_snn",
}
# Bidirectional whole-block encoded dual-budget (historical Scheme A).
BLOCK_DUAL_METHODS = {
    "symmetric_block_dual_topk_ef_snn",
}
# New Scheme A identity: whole architecture blocks only define the shortlist;
# the wire payload selects coordinates locally and transmits actual INT8 codes.
BLOCK_LOCAL_INT8_METHODS = {
    "symmetric_block_local_int8_dual_topk_ef_snn",
}
ARCHITECTURE_BLOCK_METHODS = BLOCK_DUAL_METHODS | BLOCK_LOCAL_INT8_METHODS
DEFAULT_BLOCK_LINEAR_SIZE = 16
METHODS = SAW_METHODS | CURRENT_ROUND_TOPK_METHODS
CREDIT_METHODS = {
    "credit_topk_saw_snn",
    "probe_credit_topk_saw_snn",
    "training_integrated_credit_topk_saw_snn",
    "probe_credit_topk_saw_ef_snn",
    "training_integrated_credit_topk_saw_ef_snn",
    "probe_credit_topk_snn",
    "training_integrated_credit_topk_snn",
    "probe_credit_topk_ef_snn",
    "training_integrated_credit_topk_ef_snn",
    "anchor_credit_topk_ef_snn",
    "anchor_credit_topk_ef_ema_snn",
    "double_anchor_credit_topk_ef_snn",
    "double_anchor_neuron_topk_ef_snn",
}
CREDIT_SCORE_BITS_PER_CLIENT = 64
NEURON_RATE_BITS = 32


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value, name: str) -> float:
    """Accept positive int/float (not bool); used for fractional upload budgets."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _optional_positive_int(value, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _uses_credit_scores(method: str) -> bool:
    if method not in METHODS:
        raise ValueError(f"unknown SAW comparison method: {method}")
    return method in CREDIT_METHODS


def _credit_mode_for_method(method: str, compression: dict) -> str:
    if method in {
        "probe_credit_topk_snn",
        "probe_credit_topk_ef_snn",
        "probe_credit_topk_saw_snn",
        "probe_credit_topk_saw_ef_snn",
    }:
        return "probe"
    if method in {
        "training_integrated_credit_topk_snn",
        "training_integrated_credit_topk_ef_snn",
        "training_integrated_credit_topk_saw_snn",
        "training_integrated_credit_topk_saw_ef_snn",
    }:
        return "training_integrated"
    if method in {
        "anchor_credit_topk_ef_snn",
        "anchor_credit_topk_ef_ema_snn",
        "double_anchor_credit_topk_ef_snn",
        "double_anchor_neuron_topk_ef_snn",
    }:
        return "anchor"
    return _credit_mode(compression)


def _credit_mode(compression: dict) -> str:
    mode = str(compression.get("credit_mode", "full_scan"))
    if mode not in {"full_scan", "probe", "training_integrated", "anchor"}:
        raise ValueError(
            "compression.credit_mode must be full_scan, probe, training_integrated, or anchor"
        )
    return mode


def _credit_uses_ema(method: str) -> bool:
    """Whether the method smooths raw credit observations with EMA."""
    return method not in {
        "anchor_credit_topk_ef_snn",
        "double_anchor_credit_topk_ef_snn",
        "double_anchor_neuron_topk_ef_snn",
    }


def _fixed_class_probe_indices(targets, client_indices, per_class: int, seed: int):
    """Choose one deterministic, class-balanced local probe for an entire run."""
    if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class <= 0:
        raise ValueError("compression.credit_probe_per_class must be a positive integer")
    local = np.asarray(client_indices, dtype=np.int64)
    labels = np.asarray(targets)[local]
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(np.unique(labels).tolist()):
        members = local[labels == label].copy()
        rng.shuffle(members)
        selected.extend(members[:per_class].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _update_credit_ema(history: dict[int, float], client_id: int, observed: float, beta: float):
    if not 0.0 <= beta < 1.0:
        raise ValueError("compression.credit_ema_beta must be in [0, 1)")
    previous = history.get(client_id)
    smoothed = float(observed) if previous is None else beta * previous + (1.0 - beta) * float(observed)
    history[client_id] = smoothed
    return smoothed


def _accumulate_class_activity(sums, counts, labels, sample_rates):
    rates = sample_rates.detach().to(dtype=sums.dtype)
    sums.scatter_add_(0, labels, rates)
    counts.scatter_add_(0, labels, rates.new_ones(rates.shape))


def _finish_class_activity(sums, counts):
    import torch

    return torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums))


def _accelerator_state_device(config: dict, device):
    state_storage = str(config["training"].get("state_storage", "accelerator"))
    if state_storage != "accelerator":
        raise ValueError("training.state_storage must be accelerator")
    return device


def _communication_plan(
    method: str,
    *,
    sparse_dimension: int,
    dense_affine_dimension: int,
    dense_buffer_dimension: int,
    candidates_per_round: int,
    requested_dense_upload_equivalents: int | float,
    value_bits: int,
    structured_credit_values_per_client: int = 0,
    upload_coordinates_per_round_override: int | None = None,
) -> dict[str, int | float | None]:
    """Resolve the requested sparse budget and the method's actual wire cost.

    ``requested_dense_upload_equivalents`` may be fractional (e.g. 0.36 for a
    2% per-client top-k target after accounting for index overhead). An explicit
    coordinate override is available for controls matched to an externally
    specified wire budget; the resulting plan must not exceed the configured
    dense-equivalent ceiling.
    """
    sparse_dimension = _positive_int(sparse_dimension, "sparse_dimension")
    if dense_affine_dimension < 0 or dense_buffer_dimension < 0:
        raise ValueError("dense dimensions must be non-negative")
    total_dimension = (
        sparse_dimension + dense_affine_dimension + dense_buffer_dimension
    )
    candidates_per_round = _positive_int(candidates_per_round, "candidates_per_round")
    requested_dense_upload_equivalents = _positive_number(
        requested_dense_upload_equivalents, "requested_dense_upload_equivalents"
    )
    value_bits = _positive_int(value_bits, "value_bits")
    upload_coordinates_per_round_override = _optional_positive_int(
        upload_coordinates_per_round_override,
        "compression.upload_coordinates_per_round",
    )
    if (
        isinstance(structured_credit_values_per_client, bool)
        or not isinstance(structured_credit_values_per_client, int)
        or structured_credit_values_per_client < 0
    ):
        raise ValueError("structured_credit_values_per_client must be a non-negative integer")
    uses_credit = _uses_credit_scores(method)
    index_bits = (
        int(math.ceil(math.log2(sparse_dimension))) if sparse_dimension > 1 else 0
    )
    dense_payload_bits = total_dimension * value_bits
    # The configured budget is an all-inclusive uplink wire budget: sparse
    # coordinates, mandatory dense normalization state, scalar client credit,
    # and structured neuron/channel telemetry all consume it.
    requested_total_budget_bits = int(
        requested_dense_upload_equivalents * dense_payload_bits
    )
    dense_affine_upload_bits = (
        candidates_per_round * dense_affine_dimension * value_bits
    )
    dense_buffer_upload_bits = (
        candidates_per_round * dense_buffer_dimension * value_bits
    )
    mandatory_dense_bits = dense_affine_upload_bits + dense_buffer_upload_bits
    credit_payload_bits = (
        candidates_per_round * CREDIT_SCORE_BITS_PER_CLIENT if uses_credit else 0
    )
    structured_credit_payload_bits = (
        candidates_per_round * structured_credit_values_per_client * NEURON_RATE_BITS
    )
    control_payload_bits = credit_payload_bits + structured_credit_payload_bits

    if method == "dense_saw_snn":
        if upload_coordinates_per_round_override is not None:
            raise ValueError("dense_saw_snn does not accept an upload coordinate override")
        upload_coordinates = candidates_per_round * sparse_dimension
        sparse_upload_bits = upload_coordinates * value_bits
    else:
        mandatory_bits = mandatory_dense_bits + control_payload_bits
        if mandatory_bits >= requested_total_budget_bits:
            raise ValueError("mandatory normalization/credit state exhausts the upload budget")
        bits_per_coordinate = value_bits + index_bits
        maximum_coordinates = min(
            (requested_total_budget_bits - mandatory_bits) // bits_per_coordinate,
            candidates_per_round * sparse_dimension,
        )
        upload_coordinates = (
            maximum_coordinates
            if upload_coordinates_per_round_override is None
            else upload_coordinates_per_round_override
        )
        if upload_coordinates > candidates_per_round * sparse_dimension:
            raise ValueError("upload coordinate override exceeds the dense sparse state")
        if upload_coordinates > maximum_coordinates:
            raise ValueError("upload coordinate override exceeds the configured wire budget")
        if upload_coordinates < candidates_per_round:
            raise ValueError("requested budget cannot send one coordinate per candidate")
        sparse_upload_bits = upload_coordinates * bits_per_coordinate

    data_upload_bits = sparse_upload_bits + mandatory_dense_bits
    total_upload_bits = (
        data_upload_bits + credit_payload_bits + structured_credit_payload_bits
    )
    return {
        "global_index_bits": index_bits,
        "requested_dense_upload_equivalents": requested_dense_upload_equivalents,
        "requested_total_budget_bits_per_round": requested_total_budget_bits,
        "upload_coordinates_per_round_override": upload_coordinates_per_round_override,
        "upload_coordinates_per_round": upload_coordinates,
        "planned_sparse_upload_bits_per_round": sparse_upload_bits,
        "planned_dense_affine_upload_bits_per_round": dense_affine_upload_bits,
        "planned_dense_buffer_upload_bits_per_round": dense_buffer_upload_bits,
        "planned_data_upload_bits_per_round": data_upload_bits,
        "planned_credit_payload_bits_per_round": credit_payload_bits,
        "planned_structured_credit_payload_bits_per_round": structured_credit_payload_bits,
        "planned_total_upload_bits_per_round": total_upload_bits,
        "planned_actual_data_dense_upload_equivalents_per_round": (
            data_upload_bits / dense_payload_bits
        ),
        "planned_actual_total_dense_upload_equivalents_per_round": (
            total_upload_bits / dense_payload_bits
        ),
    }


def _round_communication(
    method: str,
    budgets,
    *,
    sparse_dimension: int,
    dense_affine_dimension: int,
    dense_buffer_dimension: int,
    value_bits: int,
    index_bits: int,
    structured_credit_values_per_client: int = 0,
) -> dict[str, int | float]:
    """Measure a realized round, keeping data and credit traffic separate."""
    sparse_dimension = _positive_int(sparse_dimension, "sparse_dimension")
    if dense_affine_dimension < 0 or dense_buffer_dimension < 0:
        raise ValueError("dense dimensions must be non-negative")
    total_dimension = (
        sparse_dimension + dense_affine_dimension + dense_buffer_dimension
    )
    value_bits = _positive_int(value_bits, "value_bits")
    if isinstance(index_bits, bool) or not isinstance(index_bits, int) or index_bits < 0:
        raise ValueError("index_bits must be a non-negative integer")
    if (
        isinstance(structured_credit_values_per_client, bool)
        or not isinstance(structured_credit_values_per_client, int)
        or structured_credit_values_per_client < 0
    ):
        raise ValueError("structured_credit_values_per_client must be a non-negative integer")
    if not budgets:
        raise ValueError("at least one client upload budget is required")
    coordinates = 0
    for budget in budgets.values():
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("client upload budgets must be positive integers")
        coordinates += budget
    bits_per_coordinate = (
        value_bits if method == "dense_saw_snn" else value_bits + index_bits
    )
    sparse_upload_bits = coordinates * bits_per_coordinate
    dense_affine_upload_bits = len(budgets) * dense_affine_dimension * value_bits
    dense_buffer_upload_bits = len(budgets) * dense_buffer_dimension * value_bits
    data_upload_bits = (
        sparse_upload_bits + dense_affine_upload_bits + dense_buffer_upload_bits
    )
    credit_payload_bits = (
        len(budgets) * CREDIT_SCORE_BITS_PER_CLIENT if _uses_credit_scores(method) else 0
    )
    structured_credit_payload_bits = (
        len(budgets) * structured_credit_values_per_client * NEURON_RATE_BITS
    )
    total_upload_bits = (
        data_upload_bits + credit_payload_bits + structured_credit_payload_bits
    )
    dense_payload_bits = total_dimension * value_bits
    return {
        "sparse_upload_bits": sparse_upload_bits,
        "dense_affine_upload_bits": dense_affine_upload_bits,
        "dense_buffer_upload_bits": dense_buffer_upload_bits,
        "data_upload_bits": data_upload_bits,
        "credit_payload_bits": credit_payload_bits,
        "structured_credit_payload_bits": structured_credit_payload_bits,
        "total_upload_bits": total_upload_bits,
        "actual_data_dense_upload_equivalents": data_upload_bits / dense_payload_bits,
        "actual_total_dense_upload_equivalents": total_upload_bits / dense_payload_bits,
    }


def _clone_state(state, device=None):
    return {
        key: value.detach().to(device=device, copy=True)
        if device is not None
        else value.detach().clone()
        for key, value in state.items()
    }


def _move_tensor_tree(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device)
    if isinstance(value, dict):
        return {key: _move_tensor_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensor_tree(item, device) for item in value]
    return value


def _release_cpu_memory() -> None:
    """Return dead round-local CPU buffers to the OS when libc supports it."""
    gc.collect()
    if os.name != "posix":
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _layout_for_keys(state, keys):
    import torch

    return [
        (key, tuple(value.shape), value.numel())
        for key, value in state.items()
        if key in keys and (torch.is_floating_point(value) or torch.is_complex(value))
    ]


def _state_layouts(model, state):
    """Separate sparse trainable state from dense normalization state."""
    import torch

    trainable_keys = {name for name, _ in model.named_parameters()}
    dense_affine_keys = set()
    dense_buffer_keys = set()
    batch_norm_type = torch.nn.modules.batchnorm._BatchNorm
    for module_name, module in model.named_modules():
        if not isinstance(module, batch_norm_type):
            continue
        prefix = f"{module_name}." if module_name else ""
        dense_affine_keys.update(
            prefix + name for name, _ in module.named_parameters(recurse=False)
        )
        dense_buffer_keys.update(
            prefix + name
            for name, value in module.named_buffers(recurse=False)
            if torch.is_floating_point(value) or torch.is_complex(value)
        )

    sparse_keys = trainable_keys - dense_affine_keys
    floating_keys = {
        key
        for key, value in state.items()
        if torch.is_floating_point(value) or torch.is_complex(value)
    }
    classified = sparse_keys | dense_affine_keys | dense_buffer_keys
    unclassified = floating_keys - classified
    if unclassified:
        raise ValueError(
            "floating model state is neither trainable nor a normalization buffer: "
            + ", ".join(sorted(unclassified))
        )
    return (
        _layout_for_keys(state, sparse_keys),
        _layout_for_keys(state, dense_affine_keys),
        _layout_for_keys(state, dense_buffer_keys),
    )


def _flatten_difference(local_state, base_state, layout):
    import torch

    if not layout:
        reference = next(
            value
            for value in local_state.values()
            if torch.is_floating_point(value) or torch.is_complex(value)
        )
        return reference.new_empty(0)
    return torch.cat([(local_state[key] - base_state[key]).reshape(-1) for key, _, _ in layout])


def _apply_flat_update(global_state, flat_update, layout, step_size: float):
    result = _clone_state(global_state)
    offset = 0
    for key, shape, count in layout:
        result[key].add_(flat_update[offset : offset + count].reshape(shape), alpha=step_size)
        offset += count
    if offset != flat_update.numel():
        raise AssertionError("flat update does not match state layout")
    return result


RANKING_SCORES = {
    "raw_global_magnitude",
    "rms_normalized_global",
    "coordinate_relative_distortion",
}
RANKING_SCORE_EPSILON = 1e-12
_DEFAULT_RANKING_SCORE = "raw_global_magnitude"


def _ranking_score(compression: dict | None) -> str:
    ranking = str(
        (compression or {}).get("ranking_score", _DEFAULT_RANKING_SCORE)
    )
    if ranking not in RANKING_SCORES:
        raise ValueError(
            f"compression.ranking_score must be one of {sorted(RANKING_SCORES)}; "
            f"got {ranking!r}"
        )
    return ranking


def _ranking_scores(corrected, layout, ranking: str, epsilon: float):
    """Per-coordinate ranking scores aligned with the flat corrected vector."""
    import torch

    if ranking == "raw_global_magnitude":
        return corrected.abs()
    segments = []
    offset = 0
    for _, _, count in layout:
        segment = corrected[offset : offset + count]
        offset += count
        if ranking == "rms_normalized_global":
            scale = torch.sqrt(segment.pow(2).mean().clamp_min(epsilon))
            segments.append(segment.abs() / scale)
        elif ranking == "coordinate_relative_distortion":
            energy = segment.pow(2).sum()
            if float(energy) <= epsilon:
                segments.append(torch.zeros_like(segment))
            else:
                segments.append(segment.pow(2) / energy)
        else:  # pragma: no cover - guarded by _ranking_score
            raise ValueError(f"unsupported ranking score: {ranking}")
    if not segments:
        return corrected.abs()
    return torch.cat(segments)


def _stable_topk(scores, k: int):
    """Global Top-k with deterministic tie-break: score desc, flat index asc."""
    import torch

    if scores.numel() == 0:
        raise ValueError("Top-k cannot operate on an empty update")
    # 64-bit lexicographic key: score descending, index ascending.
    order = torch.argsort(scores, descending=True, stable=True)
    # Within equal scores, torch's stable argsort preserves ascending index order.
    return order[:k]


def _compress_with_error_feedback(
    update,
    residual,
    k: int,
    layout=None,
    ranking: str = _DEFAULT_RANKING_SCORE,
    epsilon: float = RANKING_SCORE_EPSILON,
    coordinate_weight=None,
):
    """Top-k sparse payload with optional EF and per-tensor score normalization.

    Always uploads the original corrected values; unselected coordinates are
    written back to the next residual. `layout` is only needed for the two
    normalized rankings; raw magnitude falls back to the legacy flat path.
    """
    import torch

    corrected = update if residual is None else update + residual
    if not 0 < k <= corrected.numel():
        raise ValueError("Top-k must be in [1, update dimension]")
    if not torch.isfinite(corrected).all():
        raise ValueError("Top-k corrected update contains non-finite values")
    if not epsilon > 0:
        raise ValueError("ranking epsilon must be positive")
    if ranking != "raw_global_magnitude":
        if layout is None:
            raise ValueError("normalized ranking scores require the sparse layout")
        scores = _ranking_scores(corrected, layout, ranking, epsilon)
    else:
        scores = corrected.abs()
    if coordinate_weight is not None:
        if coordinate_weight.shape != corrected.shape:
            raise ValueError("coordinate weight must match the flat update shape")
        scores = scores * coordinate_weight.to(device=scores.device, dtype=scores.dtype)
    if not torch.isfinite(scores).all():
        raise ValueError("Top-k ranking scores contain non-finite values")
    indices = _stable_topk(scores, k)
    values = corrected[indices].clone()
    next_residual = corrected.clone()
    next_residual[indices] = 0
    return indices.to(torch.int32), values, next_residual


def _spike_layer_order(model):
    """Return the firing-rate layer order the model reports activity for.

    The SNN builders append one entry to ``layer_activity`` per spiking layer in
    forward order (conv layers first, then the hidden FC), so the count equals
    ``len(model.spike_layer_sizes)``. We reconstruct the matching list of
    producing-module prefixes (``convs.0`` ... ``fc1``) so each sparse parameter
    can be attributed to the spike layer whose activity it drives.
    """

    num_layers = len(getattr(model, "spike_layer_sizes", ()))
    if num_layers == 0:
        raise ValueError("model exposes no spike_layer_sizes; downlink credit unavailable")
    # Collect conv-producing prefixes in forward order.
    conv_prefixes = []
    if hasattr(model, "convs"):
        conv_prefixes = [f"convs.{i}" for i in range(len(model.convs))]
    else:
        named = dict(model.named_modules())
        idx = 1
        while f"conv{idx}" in named:
            conv_prefixes.append(f"conv{idx}")
            idx += 1
    # Remaining spiking layers are the hidden FC(s); the readout (fc2 / last
    # linear) does not spike in these models, so it is excluded. With N convs
    # and ``num_layers`` spike entries, the FC spiking layers are the tail.
    fc_count = num_layers - len(conv_prefixes)
    fc_prefixes = [f"fc{i}" for i in range(1, fc_count + 1)]
    order = conv_prefixes + fc_prefixes
    if len(order) != num_layers:
        raise ValueError(
            f"cannot align {len(order)} producing prefixes with {num_layers} spike layers"
        )
    return order


def _build_layer_index_map(layout, spike_order):
    """Map each flat coordinate of the sparse layout to a spike-layer id.

    ``spike_order`` lists producing-module prefixes (e.g. ``convs.0`` ... ``fc1``)
    in firing-rate order. Each sparse state key is attributed to the spike layer
    whose prefix matches the key's module path; coordinates of unmatched keys get
    id ``-1`` and receive a neutral (1.0) credit weight. Returns a LongTensor of
    shape [sparse_dimension].
    """
    import torch

    if not layout:
        return torch.zeros(0, dtype=torch.long)
    ids = []
    for key, _, count in layout:
        layer_id = -1
        for idx, prefix in enumerate(spike_order):
            if key == prefix or key.startswith(prefix + "."):
                layer_id = idx
                break
        ids.append(torch.full((count,), layer_id, dtype=torch.long))
    return torch.cat(ids)


def _credit_coordinate_weight(layer_rates, layer_index_map, alpha: float):
    """Broadcast per-layer temporal credit to a per-coordinate weight vector.

    ``layer_rates`` is a 1-D tensor of per-layer firing rates (shape
    [num_layers], aligned with spike-layer order). The weight for a layer is
    ``exp(alpha * (rate - mean_rate) / (std_rate + eps))`` so that
    above-average-firing layers get weight > 1 and below-average < 1. With
    ``alpha == 0`` the weight is uniformly 1 (pure |delta| selection).
    Coordinates whose layer id is ``-1`` (non-spiking producers such as the
    readout layer) receive a neutral weight of 1.0. Returns a tensor aligned
    with the flat sparse coordinate space.
    """
    import torch

    if layer_rates.numel() == 0:
        return torch.ones(layer_index_map.numel(), device=layer_index_map.device)
    rates = layer_rates.to(torch.float32)
    mean = rates.mean()
    std = rates.std(unbiased=False)
    normalized = (rates - mean) / (std + RANKING_SCORE_EPSILON)
    layer_weight = torch.exp(alpha * normalized)
    # id -1 (unmapped / non-spiking) -> neutral weight 1.0
    safe_index = layer_index_map.clamp_min(0)
    weight = layer_weight[safe_index]
    return torch.where(
        layer_index_map < 0, torch.ones_like(weight), weight
    )


def _build_neuron_coordinate_map(layout, spike_order, channel_sizes):
    """Map sparse coordinates to (spike layer, output channel); unmapped is -1."""
    import torch

    if len(channel_sizes) != len(spike_order):
        raise ValueError("channel sizes must align with the spike-layer order")
    if not layout:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty.clone()
    layer_ids = []
    channel_ids = []
    for key, shape, count in layout:
        layer_id = next(
            (
                idx
                for idx, prefix in enumerate(spike_order)
                if key == prefix or key.startswith(prefix + ".")
            ),
            -1,
        )
        channels = int(shape[0]) if shape else 0
        if (
            layer_id < 0
            or channels <= 0
            or channels != int(channel_sizes[layer_id])
            or count % channels != 0
        ):
            layer_ids.append(torch.full((count,), -1, dtype=torch.long))
            channel_ids.append(torch.full((count,), -1, dtype=torch.long))
            continue
        per_channel = count // channels
        layer_ids.append(torch.full((count,), layer_id, dtype=torch.long))
        channel_ids.append(torch.arange(channels).repeat_interleave(per_channel))
    return torch.cat(layer_ids), torch.cat(channel_ids)


def _neuron_coordinate_weight(
    neuron_rates,
    layer_ids,
    channel_ids,
    temperature: float,
    credit_fn: str = "exp",
):
    """Layer-wise z-score channel rates and broadcast them to parameter coordinates.

    ``credit_fn`` selects the transform from per-channel z-score to weight:
    - ``exp``: ``exp(temp * z)`` (original, aggressive outlier amplification)
    - ``linear``: ``max(0, 1 + temp * z)`` (bounded, robust to outliers)
    """
    import torch

    if layer_ids.shape != channel_ids.shape:
        raise ValueError("neuron layer/channel maps must have identical shapes")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("neuron credit temperature must be finite and non-negative")
    if credit_fn not in {"exp", "linear"}:
        raise ValueError("neuron credit_fn must be 'exp' or 'linear'")
    weight = torch.ones(layer_ids.numel(), device=layer_ids.device, dtype=torch.float32)
    for layer_id, rates in enumerate(neuron_rates):
        rates = rates.to(device=weight.device, dtype=torch.float32)
        if rates.ndim != 1 or rates.numel() == 0:
            raise ValueError("each neuron-rate layer must be a non-empty vector")
        if not torch.isfinite(rates).all():
            raise ValueError("neuron firing rates must be finite")
        mask = layer_ids == layer_id
        if mask.any() and int(channel_ids[mask].max()) >= rates.numel():
            raise ValueError("neuron-rate vector does not match the coordinate map")
        normalized = (rates - rates.mean()) / (
            rates.std(unbiased=False) + RANKING_SCORE_EPSILON
        )
        if credit_fn == "linear":
            channel_weight = (1.0 + temperature * normalized).clamp(min=0.0)
        else:
            channel_weight = torch.exp((temperature * normalized).clamp(-20.0, 20.0))
        if mask.any():
            weight[mask] = channel_weight[channel_ids[mask]]
    return weight


def _downlink_topk_with_credit(
    delta,
    residual,
    credit_weight,
    k: int,
    use_error_feedback: bool,
):
    """Server-side downlink Top-k selection with optional EF and credit weighting.

    ``delta`` is the global increment (flat, sparse space) for this round. With
    EF, the outstanding residual is folded in (``corrected = delta + residual``)
    and unselected coordinates are written back to the next residual, mirroring
    the uplink ``_compress_with_error_feedback`` contract. ``credit_weight`` is a
    per-coordinate multiplier (from temporal credit); selection ranks
    ``corrected.abs() * credit_weight`` so high-credit layers are prioritized.
    Returns ``(indices[int32], values, next_residual)``.
    """
    import torch

    corrected = delta if (residual is None or not use_error_feedback) else delta + residual
    if not 0 < k <= corrected.numel():
        raise ValueError("downlink Top-k must be in [1, delta dimension]")
    if not torch.isfinite(corrected).all():
        raise ValueError("downlink corrected delta contains non-finite values")
    scores = corrected.abs() * credit_weight
    if not torch.isfinite(scores).all():
        raise ValueError("downlink ranking scores contain non-finite values")
    indices = _stable_topk(scores, k)
    values = corrected[indices].clone()
    if use_error_feedback:
        next_residual = corrected.clone()
        next_residual[indices] = 0
    else:
        next_residual = torch.zeros_like(corrected)
    return indices.to(torch.int32), values, next_residual


def _block_values(tensor, block):
    """Slice one architecture-aligned block out of a weight tensor."""
    output_index, input_index, start, stop = block
    if tensor.ndim == 4:
        return tensor[output_index, input_index]
    if tensor.ndim == 2:
        return tensor[output_index, start:stop]
    raise ValueError(f"unsupported block tensor rank: {tensor.ndim}")


def _architecture_block_layout(sparse_layout, linear_block_size: int = DEFAULT_BLOCK_LINEAR_SIZE):
    """Build architecture-aligned blocks over the full sparse layout.

    Conv: one block per ``(out, in)`` kernel. Linear: contiguous input segments
    of length ``linear_block_size`` within each output row. Unlike the Proposed
    spike-gated path, *all* sparse keys (including non-spiking readout) are
    block-encoded so UL/DL wire math stays consistent under dual-budget.
    """
    if linear_block_size <= 0:
        raise ValueError("linear_block_size must be positive")
    blocks = {}
    for key, shape, _ in sparse_layout:
        layer_blocks = []
        if len(shape) == 4:
            for output_index in range(shape[0]):
                for input_index in range(shape[1]):
                    layer_blocks.append(
                        (output_index, input_index, 0, shape[2] * shape[3])
                    )
        elif len(shape) == 2:
            for output_index in range(shape[0]):
                for start in range(0, shape[1], linear_block_size):
                    stop = min(start + linear_block_size, shape[1])
                    layer_blocks.append((output_index, -1, start, stop))
        else:
            raise ValueError(f"unsupported sparse weight shape for {key}: {shape}")
        blocks[key] = tuple(layer_blocks)
    return blocks


def _block_rms_vector(tensor, linear_block_size: int = DEFAULT_BLOCK_LINEAR_SIZE):
    """Per-block RMS in the same order as ``_architecture_block_layout``."""
    import torch

    if linear_block_size <= 0:
        raise ValueError("linear_block_size must be positive")
    values = tensor.detach().float()
    if values.ndim == 4:
        flat = values.reshape(values.shape[0] * values.shape[1], -1)
        return flat.square().mean(dim=1).sqrt()
    if values.ndim == 2:
        out_features, in_features = values.shape
        full_blocks = in_features // linear_block_size
        remainder = in_features % linear_block_size
        parts = []
        if full_blocks > 0:
            full = values[:, : full_blocks * linear_block_size].reshape(
                out_features, full_blocks, linear_block_size
            )
            parts.append(full.square().mean(dim=2).sqrt())
        if remainder:
            partial = (
                values[:, full_blocks * linear_block_size :]
                .square()
                .mean(dim=1)
                .sqrt()
            )
            parts.append(partial.unsqueeze(1))
        if not parts:
            raise ValueError("linear weight has empty input dimension")
        return torch.cat(parts, dim=1).reshape(-1)
    raise ValueError(f"unsupported block tensor rank: {values.ndim}")


def _block_index_width(n_blocks: int) -> int:
    return max(1, int(math.ceil(math.log2(max(int(n_blocks), 1)))))


def _count_width(max_count: int) -> int:
    """Bits needed to encode an inclusive integer count in ``[0, max_count]``."""
    if max_count < 0:
        raise ValueError("max_count must be non-negative")
    return max(1, int(math.ceil(math.log2(int(max_count) + 1))))


def _block_meta_table(block_layout, sparse_layout, state_example=None):
    """Flatten block layout into (key, block, n_values, index_width) rows."""
    layout_shapes = {key: shape for key, shape, _ in sparse_layout}
    meta = []
    for key, blocks in block_layout.items():
        index_width = _block_index_width(len(blocks))
        shape = layout_shapes[key]
        for block in blocks:
            if len(shape) == 4:
                n_values = int(shape[2] * shape[3])
            else:
                n_values = int(block[3] - block[2])
            if state_example is not None:
                n_values = int(_block_values(state_example[key], block).numel())
            meta.append((key, block, n_values, index_width))
    return meta


def _precompute_block_flat_table(block_layout, sparse_layout):
    """Map each architecture block to a contiguous flat range.

    Returns a dict with CPU int64 tensors aligned across all blocks in layout
    order (layer order, then block order within layer):
      starts, sizes, index_widths, and parallel Python lists keys/blocks.
    """
    import torch

    starts = []
    sizes = []
    index_widths = []
    keys = []
    blocks_out = []
    layer_ids = []
    layer_keys = []
    layer_block_counts = []
    layer_offsets = []
    offset = 0
    block_offset = 0
    for layer_id, (key, shape, count) in enumerate(sparse_layout):
        layer_blocks = block_layout[key]
        index_width = _block_index_width(len(layer_blocks))
        layer_keys.append(key)
        layer_block_counts.append(len(layer_blocks))
        layer_offsets.append(block_offset)
        if len(shape) == 4:
            n_per = int(shape[2] * shape[3])
            in_ch = int(shape[1])
            for block in layer_blocks:
                o, i, _, _ = block
                local = (int(o) * in_ch + int(i)) * n_per
                starts.append(offset + local)
                sizes.append(n_per)
                index_widths.append(index_width)
                keys.append(key)
                blocks_out.append(block)
                layer_ids.append(layer_id)
        elif len(shape) == 2:
            in_f = int(shape[1])
            for block in layer_blocks:
                o, _, start, stop = block
                local = int(o) * in_f + int(start)
                n_values = int(stop) - int(start)
                starts.append(offset + local)
                sizes.append(n_values)
                index_widths.append(index_width)
                keys.append(key)
                blocks_out.append(block)
                layer_ids.append(layer_id)
        else:
            raise ValueError(f"unsupported sparse weight shape for {key}: {shape}")
        offset += count
        block_offset += len(layer_blocks)
    if not starts:
        raise ValueError("block layout is empty")
    linear_sizes = [
        int(block[3]) - int(block[2])
        for blocks in block_layout.values()
        for block in blocks
        if int(block[1]) == -1
    ]
    return {
        "starts": torch.tensor(starts, dtype=torch.int64),
        "sizes": torch.tensor(sizes, dtype=torch.int64),
        "index_widths": torch.tensor(index_widths, dtype=torch.int64),
        "keys": keys,
        "blocks": blocks_out,
        "layer_ids": torch.tensor(layer_ids, dtype=torch.int64),
        "layer_keys": tuple(layer_keys),
        "layer_block_counts": torch.tensor(layer_block_counts, dtype=torch.int64),
        "layer_offsets": torch.tensor(layer_offsets, dtype=torch.int64),
        "layer_count_bits": torch.tensor(
            [_count_width(count) for count in layer_block_counts], dtype=torch.int64
        ),
        "num_blocks": len(starts),
        "sparse_dimension": offset,
        "linear_block_size": max(linear_sizes) if linear_sizes else DEFAULT_BLOCK_LINEAR_SIZE,
    }


def _block_rms_from_flat(flat, sparse_layout, linear_block_size: int = DEFAULT_BLOCK_LINEAR_SIZE):
    """Concatenate per-layer block RMS in layout order (matches block table)."""
    import torch

    parts = []
    offset = 0
    for key, shape, count in sparse_layout:
        layer = flat[offset : offset + count].reshape(shape)
        parts.append(_block_rms_vector(layer, linear_block_size))
        offset += count
    if offset != flat.numel():
        raise AssertionError("flat does not match sparse layout")
    return torch.cat(parts)


def _select_blocks_by_rms_budget(
    state_tensors,
    block_layout,
    sparse_layout,
    target_coordinates: int,
    linear_block_size: int = DEFAULT_BLOCK_LINEAR_SIZE,
    block_table=None,
):
    """Select whole blocks by global RMS until ``target_coordinates`` is met.

    Ranking is global across layers (not per-layer proportional) so a single
    coordinate budget maps cleanly onto block wire accounting. Returns
    ``selected_blocks`` as ``{key: set(block_tuple)}`` plus selection stats.
    When ``block_table`` is provided, selection uses vectorized RMS + greedy
    fill on flat ranges (training hot path).
    """
    import torch

    if target_coordinates <= 0:
        raise ValueError("target_coordinates must be positive")
    if block_table is None:
        # Slow path for unit tests that pass raw state tensors only.
        scores = []
        meta = []
        for key, blocks in block_layout.items():
            tensor = state_tensors[key]
            rms = _block_rms_vector(tensor, linear_block_size)
            if int(rms.numel()) != len(blocks):
                raise ValueError(
                    f"block RMS width mismatch for {key}: {int(rms.numel())} vs {len(blocks)}"
                )
            for block_id, (block, score) in enumerate(zip(blocks, rms.tolist())):
                n_values = int(_block_values(tensor, block).numel())
                scores.append(float(score))
                meta.append((key, block, n_values, block_id))
        if not scores:
            raise ValueError("block layout is empty")
        score_tensor = torch.tensor(scores, dtype=torch.float64)
        order = torch.argsort(score_tensor, descending=True, stable=True).tolist()
        selected = {}
        selected_coords = 0
        selected_blocks = 0
        value_bits = 0
        index_bits = 0
        for rank_pos in order:
            if selected_coords >= target_coordinates:
                break
            key, block, n_values, _ = meta[rank_pos]
            selected.setdefault(key, set()).add(block)
            selected_coords += n_values
            selected_blocks += 1
            value_bits += n_values * 32
            index_bits += _block_index_width(len(block_layout[key]))
        return selected, {
            "selected_blocks": selected_blocks,
            "selected_coordinates": selected_coords,
            "value_bits": value_bits,
            "index_bits": index_bits,
            "payload_bits": value_bits + index_bits,
            "target_coordinates": int(target_coordinates),
            "selected_block_ids": None,
        }

    # Fast path: RMS from flat-shaped tensors (or build flat).
    if isinstance(state_tensors, dict) and "flat" in state_tensors:
        flat = state_tensors["flat"]
    else:
        flat = torch.cat(
            [state_tensors[key].reshape(-1) for key, _, _ in sparse_layout]
        )
    rms = _block_rms_from_flat(flat, sparse_layout, linear_block_size)
    if int(rms.numel()) != int(block_table["num_blocks"]):
        raise ValueError(
            f"block RMS width mismatch: {int(rms.numel())} vs {block_table['num_blocks']}"
        )
    # Stable: score desc, block id asc.
    order = torch.argsort(rms.detach().to(dtype=torch.float64).cpu(), descending=True, stable=True)
    sizes = block_table["sizes"]
    widths = block_table["index_widths"]
    selected_coords = 0
    chosen = []
    for idx in order.tolist():
        if selected_coords >= target_coordinates:
            break
        chosen.append(idx)
        selected_coords += int(sizes[idx])
    if not chosen:
        raise ValueError("block selection produced an empty set")
    chosen_t = torch.tensor(chosen, dtype=torch.int64)
    selected_blocks = int(chosen_t.numel())
    selected_coords = int(sizes[chosen_t].sum())
    value_bits = selected_coords * 32
    index_bits = int(widths[chosen_t].sum())
    selected = {}
    keys = block_table["keys"]
    blocks = block_table["blocks"]
    for idx in chosen:
        selected.setdefault(keys[idx], set()).add(blocks[idx])
    return selected, {
        "selected_blocks": selected_blocks,
        "selected_coordinates": selected_coords,
        "value_bits": value_bits,
        "index_bits": index_bits,
        "payload_bits": value_bits + index_bits,
        "target_coordinates": int(target_coordinates),
        "selected_block_ids": chosen_t,
    }


def _expand_block_ids_to_flat_indices(block_table, selected_block_ids):
    """Expand selected block ids into a 1-D int64 index tensor (CPU)."""
    import torch

    starts = block_table["starts"]
    sizes = block_table["sizes"]
    if selected_block_ids.numel() == 0:
        return torch.empty(0, dtype=torch.int64)
    sel_starts = starts[selected_block_ids]
    sel_sizes = sizes[selected_block_ids]
    total = int(sel_sizes.sum().item())
    if total == 0:
        return torch.empty(0, dtype=torch.int64)
    # Fully vectorized range expand: for each block, starts[i] + 0..size[i]-1.
    out_start = torch.zeros(sel_sizes.numel(), dtype=torch.int64)
    if sel_sizes.numel() > 1:
        out_start[1:] = sel_sizes[:-1].cumsum(0)
    repeated_starts = torch.repeat_interleave(sel_starts, sel_sizes)
    within = torch.arange(total, dtype=torch.int64) - torch.repeat_interleave(
        out_start, sel_sizes
    )
    return repeated_starts + within


def _pack_blocks_from_flat(flat, sparse_layout, selected_blocks, block_table=None, selected_block_ids=None):
    """Pack selected block values from a flat sparse vector (layout order).

    Fast path: when ``block_table`` + ``selected_block_ids`` are given, pack is
    represented as ``{"__flat__": (indices, values)}`` for aggregation/scatter.
    Slow path keeps ``{key: {block: tensor}}`` for unit tests.
    """

    if block_table is not None and selected_block_ids is not None:
        indices = _expand_block_ids_to_flat_indices(block_table, selected_block_ids)
        # Move indices to flat device for gather.
        indices = indices.to(device=flat.device)
        values = flat[indices].detach().clone()
        return {"__flat__": (indices, values)}

    packed = {}
    offset = 0
    for key, shape, count in sparse_layout:
        layer = flat[offset : offset + count].reshape(shape)
        offset += count
        for block in selected_blocks.get(key, ()):
            packed.setdefault(key, {})[block] = _block_values(layer, block).detach().clone()
    if offset != flat.numel():
        raise AssertionError("flat does not match sparse layout")
    return packed


def _pack_blocks_from_state(state, selected_blocks):
    packed = {}
    for key, blocks in selected_blocks.items():
        if not blocks:
            continue
        packed[key] = {
            block: _block_values(state[key], block).detach().clone() for block in blocks
        }
    return packed


def _scatter_blocks_into_flat(flat, sparse_layout, packed_blocks, *, mode: str = "add"):
    """Apply packed blocks onto a flat sparse vector. mode: add|set|zero."""

    result = flat.clone()
    if "__flat__" in packed_blocks:
        indices, values = packed_blocks["__flat__"]
        indices = indices.to(device=result.device)
        values = values.to(device=result.device, dtype=result.dtype)
        if mode == "add":
            result.index_add_(0, indices, values)
        elif mode == "set":
            result[indices] = values
        elif mode == "zero":
            result[indices] = 0
        else:
            raise ValueError(f"unknown scatter mode: {mode}")
        return result
    offset = 0
    for key, shape, count in sparse_layout:
        layer = result[offset : offset + count].reshape(shape)
        for block, values in packed_blocks.get(key, {}).items():
            slot = _block_values(layer, block)
            if mode == "add":
                slot.add_(values)
            elif mode == "set":
                slot.copy_(values)
            elif mode == "zero":
                slot.zero_()
            else:
                raise ValueError(f"unknown scatter mode: {mode}")
        offset += count
    return result


def _zero_blocks_in_flat(flat, sparse_layout, selected_blocks, block_table=None, selected_block_ids=None):
    """Zero selected blocks in a flat residual (whole-block EF)."""

    if block_table is not None and selected_block_ids is not None:
        indices = _expand_block_ids_to_flat_indices(block_table, selected_block_ids).to(
            device=flat.device
        )
        result = flat.clone()
        result[indices] = 0
        return result
    result = flat.clone()
    offset = 0
    for key, shape, count in sparse_layout:
        layer = result[offset : offset + count].reshape(shape)
        for block in selected_blocks.get(key, ()):
            _block_values(layer, block).zero_()
        offset += count
    return result


def _block_payload_stats(selected_blocks, block_layout, sparse_layout, state=None):
    """Exact value/block-index accounting for a bidirectional block payload."""
    value_bits = 0
    index_bits = 0
    selected_coordinates = 0
    selected_block_count = 0
    total_block_count = 0
    total_coordinates = sum(count for _, _, count in sparse_layout)
    layout_shapes = {key: shape for key, shape, _ in sparse_layout}
    for key, blocks in block_layout.items():
        total_block_count += len(blocks)
        index_width = _block_index_width(len(blocks))
        shape = layout_shapes[key]
        for block in selected_blocks.get(key, set()):
            if state is not None:
                n_values = int(_block_values(state[key], block).numel())
            elif len(shape) == 4:
                n_values = int(shape[2] * shape[3])
            else:
                n_values = int(block[3] - block[2])
            selected_coordinates += n_values
            selected_block_count += 1
            value_bits += n_values * 32
            index_bits += index_width
    return {
        "selected_blocks": selected_block_count,
        "total_blocks": total_block_count,
        "selected_coordinates": selected_coordinates,
        "total_coordinates": total_coordinates,
        "value_bits": value_bits,
        "index_bits": index_bits,
        "payload_bits": value_bits + index_bits,
        "parameter_retention_ratio": selected_coordinates / max(total_coordinates, 1),
        "encoding": "block_id_per_layer",
    }


def _aggregate_block_packs(client_packs, client_weights, sparse_layout, like=None):
    """Sample-weight average of packed block updates (missing block = 0).

    Fast path: ``__flat__`` packs are index_add'd into a dense flat and scaled.
    Slow path: structured ``{key: {block: tensor}}`` dicts for unit tests.
    """
    import torch

    if not client_packs or len(client_packs) != len(client_weights):
        raise ValueError("client packs and weights must be non-empty and aligned")
    total_weight = float(sum(client_weights))
    if total_weight <= 0:
        raise ValueError("client weights must have positive mass")
    if any("__flat__" in pack for pack in client_packs):
        if like is None:
            raise ValueError("like flat required for __flat__ pack aggregation")
        acc = torch.zeros_like(like)
        for pack, weight in zip(client_packs, client_weights):
            if "__flat__" not in pack:
                continue
            indices, values = pack["__flat__"]
            indices = indices.to(device=acc.device)
            values = values.to(device=acc.device, dtype=acc.dtype) * float(weight)
            acc.index_add_(0, indices, values)
        acc.mul_(1.0 / total_weight)
        # Represent aggregate as a dense flat pack for apply.
        nonzero = acc.nonzero(as_tuple=False).view(-1)
        return {"__flat__": (nonzero, acc[nonzero])}
    # Collect union of blocks.
    union = {}
    for pack in client_packs:
        for key, blocks in pack.items():
            if key == "__flat__":
                continue
            union.setdefault(key, set()).update(blocks.keys())
    aggregated = {}
    for key, blocks in union.items():
        aggregated[key] = {}
        for block in blocks:
            weighted = None
            for pack, weight in zip(client_packs, client_weights):
                update = pack.get(key, {}).get(block)
                if update is None:
                    continue
                contrib = update * float(weight)
                weighted = contrib.clone() if weighted is None else weighted + contrib
            if weighted is not None:
                aggregated[key][block] = weighted / total_weight
    return aggregated


def _apply_block_pack_to_state(state, pack, sparse_layout, step_size: float = 1.0):
    """Add packed block updates into a state dict (sparse keys only)."""
    import torch

    if "__flat__" in pack:
        like = torch.cat([state[key].reshape(-1) for key, _, _ in sparse_layout])
        flat = _flat_from_block_pack(pack, sparse_layout, like)
        return _apply_flat_update(state, flat * float(step_size), sparse_layout, 1.0)
    updated = {key: value.clone() for key, value in state.items()}
    for key, blocks in pack.items():
        for block, update in blocks.items():
            _block_values(updated[key], block).add_(update, alpha=float(step_size))
    return updated


def _flat_from_block_pack(pack, sparse_layout, like):
    """Scatter a block pack into a zero flat of ``like`` shape."""
    import torch

    flat = torch.zeros_like(like)
    return _scatter_blocks_into_flat(flat, sparse_layout, pack, mode="add")


def _compress_blocks_no_ef(
    flat_update,
    sparse_layout,
    block_layout,
    target_coordinates: int,
    linear_block_size: int = DEFAULT_BLOCK_LINEAR_SIZE,
    block_table=None,
):
    """UL path: select blocks by |update| RMS and pack original values (no EF)."""

    if block_table is not None:
        selected, stats = _select_blocks_by_rms_budget(
            {"flat": flat_update},
            block_layout,
            sparse_layout,
            target_coordinates,
            linear_block_size=linear_block_size,
            block_table=block_table,
        )
        packed = _pack_blocks_from_flat(
            flat_update,
            sparse_layout,
            selected,
            block_table=block_table,
            selected_block_ids=stats["selected_block_ids"],
        )
        return packed, selected, stats
    # Slow path: materialize per-key tensors from flat for RMS scoring.
    state_tensors = {}
    offset = 0
    for key, shape, count in sparse_layout:
        state_tensors[key] = flat_update[offset : offset + count].reshape(shape)
        offset += count
    selected, stats = _select_blocks_by_rms_budget(
        state_tensors,
        block_layout,
        sparse_layout,
        target_coordinates,
        linear_block_size=linear_block_size,
    )
    packed = _pack_blocks_from_flat(flat_update, sparse_layout, selected)
    return packed, selected, stats


def _downlink_block_topk_with_ef(
    delta_flat,
    residual_flat,
    sparse_layout,
    block_layout,
    target_coordinates: int,
    *,
    use_error_feedback: bool = True,
    linear_block_size: int = DEFAULT_BLOCK_LINEAR_SIZE,
    block_table=None,
):
    """Server DL: whole-block Top-K on corrected residual with optional EF.

    ``corrected = delta + residual`` (when EF on). Selected blocks are packed
    out of ``corrected``; unselected whole blocks remain in ``next_residual``.
    Returns ``(packed, selected, next_residual, stats)``.
    """
    import torch

    if residual_flat is None or not use_error_feedback:
        corrected = delta_flat
    else:
        corrected = delta_flat + residual_flat
    if not torch.isfinite(corrected).all():
        raise ValueError("downlink corrected block residual contains non-finite values")
    if block_table is not None:
        selected, stats = _select_blocks_by_rms_budget(
            {"flat": corrected},
            block_layout,
            sparse_layout,
            target_coordinates,
            linear_block_size=linear_block_size,
            block_table=block_table,
        )
        packed = _pack_blocks_from_flat(
            corrected,
            sparse_layout,
            selected,
            block_table=block_table,
            selected_block_ids=stats["selected_block_ids"],
        )
        if use_error_feedback:
            next_residual = _zero_blocks_in_flat(
                corrected,
                sparse_layout,
                selected,
                block_table=block_table,
                selected_block_ids=stats["selected_block_ids"],
            )
        else:
            next_residual = torch.zeros_like(corrected)
        return packed, selected, next_residual, stats
    state_tensors = {}
    offset = 0
    for key, shape, count in sparse_layout:
        state_tensors[key] = corrected[offset : offset + count].reshape(shape)
        offset += count
    selected, stats = _select_blocks_by_rms_budget(
        state_tensors,
        block_layout,
        sparse_layout,
        target_coordinates,
        linear_block_size=linear_block_size,
    )
    packed = _pack_blocks_from_flat(corrected, sparse_layout, selected)
    if use_error_feedback:
        next_residual = _zero_blocks_in_flat(corrected, sparse_layout, selected)
    else:
        next_residual = torch.zeros_like(corrected)
    return packed, selected, next_residual, stats


def _block_local_support_bits(block_size: int, selected_count: int) -> tuple[str, int]:
    """Return the cheaper decodable per-block support representation."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0 < selected_count <= block_size:
        raise ValueError("selected_count must be in [1, block_size]")
    mode_bits = 1
    count_bits = _count_width(block_size)
    bitmap_bits = mode_bits + count_bits + block_size
    local_index_width = max(1, int(math.ceil(math.log2(block_size))))
    local_id_bits = mode_bits + count_bits + selected_count * local_index_width
    if bitmap_bits <= local_id_bits:
        return "bitmap", bitmap_bits
    return "local_ids", local_id_bits


def _validate_block_local_codec_args(
    *,
    sparse_bit_cap: int,
    local_keep_ratio: float,
    sparse_value_bits: int,
    scale_bits: int,
) -> None:
    if isinstance(sparse_bit_cap, bool) or not isinstance(sparse_bit_cap, int) or sparse_bit_cap <= 0:
        raise ValueError("sparse_bit_cap must be a positive integer")
    if not math.isfinite(local_keep_ratio) or not 0.0 < local_keep_ratio <= 1.0:
        raise ValueError("local_keep_ratio must be finite and in (0, 1]")
    if sparse_value_bits != 8:
        raise ValueError("block-local INT8 requires sparse_value_bits=8")
    if scale_bits != 32:
        raise ValueError("block-local INT8 requires scale_bits=32")


def _block_local_indices_for_selected(flat, block_table, block_ids, counts):
    """Vectorize local magnitude ordering only for blocks that fit the wire cap."""
    import torch

    if block_ids.numel() == 0 or block_ids.numel() != counts.numel():
        raise ValueError("selected block ids/counts must be non-empty and aligned")
    starts = block_table["starts"]
    sizes = block_table["sizes"]
    selected_sizes = sizes[block_ids]
    max_size = int(selected_sizes.max().item())
    order_matrix = torch.full(
        (int(block_ids.numel()), max_size), -1, dtype=torch.int32
    )
    for block_size in sorted(set(int(size) for size in selected_sizes.tolist())):
        rows = torch.nonzero(selected_sizes == block_size, as_tuple=False).view(-1)
        starts_device = starts[block_ids[rows]].to(device=flat.device)
        gather_indices = starts_device[:, None] + torch.arange(
            block_size, device=flat.device, dtype=torch.int64
        )[None, :]
        values = (
            flat[gather_indices]
            .detach()
            .abs()
            .to(dtype=torch.float32, device="cpu")
            .numpy()
        )
        # NumPy stable row sort is substantially faster than torch stable
        # argsort for tens of thousands of tiny (9/16-value) blocks while
        # preserving the local-index ascending tie rule.
        local_order = np.argsort(-values, axis=1, kind="stable").astype(
            np.int32, copy=False
        )
        order_matrix[rows, :block_size] = torch.from_numpy(local_order)
    rank_positions = torch.arange(max_size, dtype=torch.int64)[None, :]
    mask = rank_positions < counts[:, None]
    return order_matrix.to(dtype=torch.int64)[mask]


def _block_local_int8_compress(
    flat,
    sparse_layout,
    block_layout,
    block_table,
    *,
    sparse_bit_cap: int,
    local_keep_ratio: float,
    sparse_value_bits: int = 8,
    scale_bits: int = 32,
):
    """Encode a global block-RMS shortlist with block-local signed INT8 support.

    Blocks are visited by descending FP32 RMS.  Inside each visited block, the
    largest-magnitude local coordinates are retained.  The final block may be
    shortened so the encoded payload never exceeds ``sparse_bit_cap``.
    """
    import torch

    _validate_block_local_codec_args(
        sparse_bit_cap=sparse_bit_cap,
        local_keep_ratio=local_keep_ratio,
        sparse_value_bits=sparse_value_bits,
        scale_bits=scale_bits,
    )
    if int(flat.numel()) != int(block_table["sparse_dimension"]):
        raise ValueError("flat size does not match block table")
    if not torch.isfinite(flat).all():
        raise ValueError("block-local INT8 source contains non-finite values")

    rms = _block_rms_from_flat(
        flat, sparse_layout, int(block_table["linear_block_size"])
    )
    order = torch.argsort(
        rms.detach().to(dtype=torch.float64).cpu(), descending=True, stable=True
    )
    sizes = block_table["sizes"]
    starts = block_table["starts"]
    widths = block_table["index_widths"]
    layer_ids = block_table["layer_ids"]
    mandatory_framing_bits = int(block_table["layer_count_bits"].sum().item())
    used_bits = mandatory_framing_bits
    active_layers = set()
    selected_block_ids = []
    selected_counts = []
    support_modes = []
    support_bits_total = 0
    block_id_bits_total = 0
    value_bits_total = 0
    scale_bits_total = 0

    order_list = order.tolist()
    sizes_list = sizes.tolist()
    widths_list = widths.tolist()
    layer_ids_list = layer_ids.tolist()
    option_cache = {}
    for block_size in set(int(size) for size in sizes_list):
        target_count = max(1, int(math.ceil(block_size * local_keep_ratio)))
        option_cache[block_size] = [
            (candidate_count, *_block_local_support_bits(block_size, candidate_count))
            for candidate_count in range(target_count, 0, -1)
        ]
    minimum_increment_without_scale = min(
        int(widths_list[block_id])
        + int(option_cache[int(sizes_list[block_id])][-1][2])
        + sparse_value_bits
        for block_id in order_list
    )
    for block_id in order_list:
        block_size = int(sizes_list[block_id])
        layer_id = int(layer_ids_list[block_id])
        chosen_count = 0
        chosen_mode = None
        chosen_support_bits = None
        for candidate_count, mode, support_bits in option_cache[block_size]:
            incremental = (
                int(widths_list[block_id])
                + support_bits
                + candidate_count * sparse_value_bits
                + (scale_bits if layer_id not in active_layers else 0)
            )
            if used_bits + incremental <= sparse_bit_cap:
                chosen_count = candidate_count
                chosen_mode = mode
                chosen_support_bits = support_bits
                break
        if chosen_count == 0:
            continue
        selected_block_ids.append(block_id)
        selected_counts.append(chosen_count)
        support_modes.append(chosen_mode)
        used_bits += (
            int(widths_list[block_id])
            + int(chosen_support_bits)
            + chosen_count * sparse_value_bits
        )
        block_id_bits_total += int(widths_list[block_id])
        support_bits_total += int(chosen_support_bits)
        value_bits_total += chosen_count * sparse_value_bits
        if layer_id not in active_layers:
            active_layers.add(layer_id)
            used_bits += scale_bits
            scale_bits_total += scale_bits
        if used_bits + minimum_increment_without_scale > sparse_bit_cap:
            # No future block can fit without a new scale; stop once the cap is
            # saturated instead of scanning every lower-ranked block.
            break

    if not selected_block_ids:
        raise ValueError("sparse bit cap cannot encode one block-local INT8 value")

    selected_block_ids_t = torch.tensor(selected_block_ids, dtype=torch.int64)
    counts = torch.tensor(selected_counts, dtype=torch.int64)
    selected_layer_ids = layer_ids[selected_block_ids_t]
    layer_offsets = block_table["layer_offsets"]
    selected_local_block_ids = selected_block_ids_t - layer_offsets[selected_layer_ids]
    # Per-layer block counts frame the payload, so serialize block records grouped
    # by layer.  Within a layer, local block id ascending is deterministic.
    serialization_order = sorted(
        range(int(selected_block_ids_t.numel())),
        key=lambda pos: (
            int(selected_layer_ids[pos]), int(selected_local_block_ids[pos])
        ),
    )
    perm = torch.tensor(serialization_order, dtype=torch.int64)
    selected_block_ids_t = selected_block_ids_t[perm]
    selected_layer_ids = selected_layer_ids[perm]
    selected_local_block_ids = selected_local_block_ids[perm]
    counts = counts[perm]
    support_modes = [support_modes[pos] for pos in serialization_order]
    concatenated_local = _block_local_indices_for_selected(
        flat, block_table, selected_block_ids_t, counts
    )
    # Bitmap payloads imply ascending bit-position value order; local-ID payloads
    # carry their explicit order. Store the actual representation that was costed.
    support_data = []
    ordered_local_parts = []
    cursor = 0
    for block_id, count, mode in zip(
        selected_block_ids_t.tolist(), counts.tolist(), support_modes
    ):
        local = concatenated_local[cursor : cursor + int(count)]
        cursor += int(count)
        if mode == "bitmap":
            local = torch.sort(local).values
            bitmap = torch.zeros(int(sizes[block_id]), dtype=torch.bool)
            bitmap[local] = True
            support_data.append(bitmap)
        else:
            support_data.append(local.to(dtype=torch.int64))
        ordered_local_parts.append(local)
    concatenated_local = torch.cat(ordered_local_parts)
    flat_indices = torch.repeat_interleave(
        starts[selected_block_ids_t], counts
    ) + concatenated_local
    flat_indices = flat_indices.to(device=flat.device)
    value_layer_ids = torch.repeat_interleave(
        selected_layer_ids, counts
    ).to(device=flat.device)
    layer_count = len(block_table["layer_keys"])
    layer_block_counts = torch.bincount(
        selected_layer_ids, minlength=layer_count
    ).to(dtype=torch.int64)
    active_layer_ids = torch.nonzero(
        layer_block_counts > 0, as_tuple=False
    ).view(-1)
    scales = torch.empty(
        active_layer_ids.numel(), dtype=torch.float32, device=flat.device
    )
    qvalues = torch.empty(flat_indices.numel(), dtype=torch.int8, device=flat.device)
    source_values = flat[flat_indices]
    for scale_pos, layer_id in enumerate(active_layer_ids.tolist()):
        mask = value_layer_ids == layer_id
        max_abs = source_values[mask].abs().max()
        scale = (
            max_abs / 127.0
            if float(max_abs.detach().cpu()) > 0.0
            else max_abs.new_tensor(1.0)
        )
        scales[scale_pos] = scale.to(dtype=torch.float32)
        qvalues[mask] = torch.clamp(
            torch.round(source_values[mask] / scale), -127, 127
        ).to(dtype=torch.int8)

    payload = {
        "schema": "block_local_int8_v1",
        "layer_block_counts": layer_block_counts,
        "block_ids": selected_local_block_ids,
        "counts": counts,
        "support_modes": tuple(support_modes),
        "support_data": tuple(support_data),
        "qvalues": qvalues,
        "scales": scales,
    }
    stats = {
        "selected_blocks": int(selected_block_ids_t.numel()),
        "selected_coordinates": int(qvalues.numel()),
        "value_bits": int(value_bits_total),
        "scale_bits": int(scale_bits_total),
        "framing_bits": int(mandatory_framing_bits),
        "block_id_bits": int(block_id_bits_total),
        "support_bits": int(support_bits_total),
        "index_bits": int(mandatory_framing_bits + block_id_bits_total + support_bits_total),
        "payload_bits": int(used_bits),
        "sparse_bit_cap": int(sparse_bit_cap),
        "unused_bits": int(sparse_bit_cap - used_bits),
        "parameter_retention_ratio": int(qvalues.numel()) / max(int(flat.numel()), 1),
        "encoding": "block_local_bitmap_or_ids_int8_per_layer_scale",
    }
    return payload, stats


def _decode_block_local_int8_payload(payload, block_table, like):
    """Decode a block-local INT8 payload into the existing flat-pack shape."""
    import torch

    required = {
        "schema",
        "layer_block_counts",
        "block_ids",
        "counts",
        "support_modes",
        "support_data",
        "qvalues",
        "scales",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"block-local INT8 payload missing {sorted(missing)}")
    if payload["schema"] != "block_local_int8_v1":
        raise ValueError("unsupported block-local INT8 payload schema")
    for field in (
        "layer_block_counts",
        "block_ids",
        "counts",
        "qvalues",
        "scales",
    ):
        if not torch.is_tensor(payload[field]):
            raise ValueError(f"block-local payload {field} must be a tensor")
    layer_block_counts = payload["layer_block_counts"].detach().to(
        dtype=torch.int64, device="cpu"
    )
    local_block_ids = payload["block_ids"].detach().to(
        dtype=torch.int64, device="cpu"
    )
    counts = payload["counts"].detach().to(dtype=torch.int64, device="cpu")
    qvalues = payload["qvalues"]
    scales = payload["scales"]
    modes = tuple(payload["support_modes"])
    support_data = tuple(payload["support_data"])
    if any(not torch.is_tensor(support) for support in support_data):
        raise ValueError("block-local support records must be tensors")
    layer_count = len(block_table["layer_keys"])
    if qvalues.dtype != torch.int8:
        raise ValueError("block-local payload qvalues must be torch.int8")
    if layer_block_counts.ndim != 1 or int(layer_block_counts.numel()) != layer_count:
        raise ValueError("layer_block_counts must cover every sparse layer")
    if bool((layer_block_counts < 0).any()):
        raise ValueError("layer block counts must be non-negative")
    if bool((layer_block_counts > block_table["layer_block_counts"]).any()):
        raise ValueError("layer block count exceeds the layer capacity")
    record_count = int(layer_block_counts.sum().item())
    if local_block_ids.ndim != 1 or counts.ndim != 1 or qvalues.ndim != 1 or scales.ndim != 1 or int(local_block_ids.numel()) != record_count or int(counts.numel()) != record_count:
        raise ValueError("block ids/counts/values/scales have invalid shape or framing")
    if len(modes) != record_count or len(support_data) != record_count:
        raise ValueError("support records do not match layer framing")
    if record_count == 0:
        raise ValueError("block-local payload may not be empty")
    if int(counts.sum().item()) != int(qvalues.numel()):
        raise ValueError("block-local support/value lengths are inconsistent")
    active_layer_ids = torch.nonzero(
        layer_block_counts > 0, as_tuple=False
    ).view(-1)
    if int(scales.numel()) != int(active_layer_ids.numel()):
        raise ValueError("block-local scales must cover exactly the active layers")
    if not torch.isfinite(scales).all() or bool((scales <= 0).any()):
        raise ValueError("block-local scales must be finite and positive")

    layer_ids_for_records = torch.repeat_interleave(
        torch.arange(layer_count, dtype=torch.int64), layer_block_counts
    )
    layer_offsets = block_table["layer_offsets"]
    layer_capacities = block_table["layer_block_counts"]
    if bool((local_block_ids < 0).any()) or bool(
        (local_block_ids >= layer_capacities[layer_ids_for_records]).any()
    ):
        raise ValueError("local block id is out of range")
    global_block_ids = layer_offsets[layer_ids_for_records] + local_block_ids
    record_pairs = list(zip(layer_ids_for_records.tolist(), local_block_ids.tolist()))
    if len(record_pairs) != len(set(record_pairs)):
        raise ValueError("block-local payload contains duplicate block records")
    if record_pairs != sorted(record_pairs):
        raise ValueError("block-local payload records are not in canonical order")
    starts = block_table["starts"]
    sizes = block_table["sizes"]
    parts = []
    for block_id, count, mode, support in zip(
        global_block_ids.tolist(), counts.tolist(), modes, support_data
    ):
        block_size = int(sizes[block_id])
        if mode not in {"bitmap", "local_ids"}:
            raise ValueError("unknown block-local support mode")
        if not 0 < int(count) <= block_size:
            raise ValueError("block-local selected count is invalid")
        support = support.detach().to(device="cpu")
        if mode == "bitmap":
            if support.dtype != torch.bool or support.ndim != 1 or int(support.numel()) != block_size:
                raise ValueError("bitmap support must be a bool vector matching block size")
            local = torch.nonzero(support, as_tuple=False).view(-1).to(dtype=torch.int64)
            if int(local.numel()) != int(count):
                raise ValueError("bitmap popcount does not match selected count")
        else:
            local = support.to(dtype=torch.int64)
            if local.ndim != 1 or int(local.numel()) != int(count):
                raise ValueError("local-ID support length does not match selected count")
            if int(local.min()) < 0 or int(local.max()) >= block_size:
                raise ValueError("block-local index is out of range")
            if int(torch.unique(local).numel()) != int(local.numel()):
                raise ValueError("block-local indices must be unique within a block")
        parts.append(local + int(starts[block_id]))
    indices = torch.cat(parts).to(device=like.device)
    value_layer_ids = torch.repeat_interleave(
        layer_ids_for_records, counts
    ).to(device=qvalues.device)
    layer_to_scale = torch.full((layer_count,), -1, dtype=torch.int64)
    layer_to_scale[active_layer_ids] = torch.arange(active_layer_ids.numel())
    value_scale_ids = layer_to_scale[value_layer_ids.to(device="cpu")].to(
        device=qvalues.device
    )
    values = qvalues.to(dtype=like.dtype) * scales.to(
        device=qvalues.device, dtype=like.dtype
    )[value_scale_ids]
    if not torch.isfinite(values).all():
        raise ValueError("decoded block-local values contain non-finite values")
    return {"__flat__": (indices, values.to(device=like.device, dtype=like.dtype))}


def _downlink_block_local_int8_with_ef(
    delta_flat,
    residual_flat,
    sparse_layout,
    block_layout,
    block_table,
    *,
    sparse_bit_cap: int,
    local_keep_ratio: float,
    use_error_feedback: bool = True,
    sparse_value_bits: int = 8,
    scale_bits: int = 32,
):
    """Closed-loop quantized DL EF: residual is corrected minus decoded wire."""
    corrected = (
        delta_flat
        if residual_flat is None or not use_error_feedback
        else delta_flat + residual_flat
    )
    payload, stats = _block_local_int8_compress(
        corrected,
        sparse_layout,
        block_layout,
        block_table,
        sparse_bit_cap=sparse_bit_cap,
        local_keep_ratio=local_keep_ratio,
        sparse_value_bits=sparse_value_bits,
        scale_bits=scale_bits,
    )
    decoded = _decode_block_local_int8_payload(payload, block_table, corrected)
    sent = _scatter_blocks_into_flat(
        corrected.new_zeros(corrected.shape), sparse_layout, decoded, mode="add"
    )
    next_residual = corrected - sent if use_error_feedback else corrected.new_zeros(corrected.shape)
    return payload, decoded, next_residual, stats


def _iso_wire_coordinate_budget(
    block_payload_bits: int,
    sparse_dimension: int,
    value_bits: int = 32,
) -> dict:
    """Map a block sparse payload bit budget onto coordinate Top-K k.

    Coordinate encoding pays ``value_bits + ceil(log2(sparse_dim))`` per value,
    so matched ``k`` yields strictly lower parameter retention than the block
    retention that produced ``block_payload_bits`` (when index_bits > 0).
    """
    if sparse_dimension <= 0:
        raise ValueError("sparse_dimension must be positive")
    index_bits = (
        int(math.ceil(math.log2(sparse_dimension))) if sparse_dimension > 1 else 0
    )
    bits_per_coord = value_bits + index_bits
    if bits_per_coord <= 0:
        raise ValueError("bits_per_coord must be positive")
    k = int(block_payload_bits) // bits_per_coord
    k = max(0, min(k, sparse_dimension))
    return {
        "matched_sparse_payload_bits": int(block_payload_bits),
        "coordinate_index_bits": index_bits,
        "bits_per_coordinate": bits_per_coord,
        "coordinate_k": k,
        "coordinate_retention": k / sparse_dimension,
        "coordinate_sparse_bits": k * bits_per_coord,
    }


def _downlink_dual_channel_quota(
    gap,
    k: int,
    support_mask,
    support_share: float,
):
    """Select k coords from a per-client catch-up gap with dual-channel quota.

    Channel A (support_share · k): highest |gap| among support_mask coordinates.
    Channel B (remainder): highest |gap| among non-support coordinates.
    Leftover slots from either channel fill from the other channel's leftovers.
    Returns ``(indices[int32], values)``; residual conservation is caller's job
    (set selected coords of residual / base after applying the payload).
    """
    import torch

    if not 0 < k <= gap.numel():
        raise ValueError("downlink dual-channel k must be in [1, gap dimension]")
    if not 0.0 <= support_share <= 1.0:
        raise ValueError("support_share must be in [0, 1]")
    if not torch.isfinite(gap).all():
        raise ValueError("downlink gap contains non-finite values")
    if support_mask.shape != gap.shape:
        raise ValueError("support_mask must match gap shape")
    scores = gap.abs()
    support_k = int(round(k * support_share))
    support_k = max(0, min(k, support_k))
    cold_k = k - support_k

    support_scores = scores.clone()
    support_scores[~support_mask] = float("-inf")
    cold_scores = scores.clone()
    cold_scores[support_mask] = float("-inf")

    selected = []
    if support_k > 0 and bool(support_mask.any()):
        n_support = int(support_mask.sum().item())
        take = min(support_k, n_support)
        if take > 0:
            selected.append(_stable_topk(support_scores, take))
            leftover_support = support_k - take
        else:
            leftover_support = support_k
    else:
        leftover_support = support_k

    if cold_k > 0 and bool((~support_mask).any()):
        n_cold = int((~support_mask).sum().item())
        take = min(cold_k, n_cold)
        if take > 0:
            selected.append(_stable_topk(cold_scores, take))
            leftover_cold = cold_k - take
        else:
            leftover_cold = cold_k
    else:
        leftover_cold = cold_k

    # Fill leftovers from the opposite channel using remaining coords.
    already = (
        torch.unique(torch.cat(selected))
        if selected
        else torch.empty(0, dtype=torch.long, device=gap.device)
    )
    remaining_budget = leftover_support + leftover_cold
    if remaining_budget > 0:
        fill_scores = scores.clone()
        if already.numel() > 0:
            fill_scores[already] = float("-inf")
        # Only keep finite positive candidates.
        finite = torch.isfinite(fill_scores)
        n_available = int(finite.sum().item())
        take = min(remaining_budget, n_available)
        if take > 0:
            # Exclude -inf by setting them to a low finite value after mask.
            fill_scores = torch.where(
                finite, fill_scores, torch.full_like(fill_scores, float("-inf"))
            )
            selected.append(_stable_topk(fill_scores, take))

    if not selected:
        # Degenerate: fall back to plain top-k on |gap|.
        indices = _stable_topk(scores, k)
    else:
        indices = torch.unique(torch.cat(selected))
        if indices.numel() < k:
            fill_scores = scores.clone()
            fill_scores[indices] = float("-inf")
            extra = _stable_topk(fill_scores, k - int(indices.numel()))
            indices = torch.cat([indices, extra])
        if indices.numel() > k:
            # Keep the k largest |gap| among the selected union.
            sub_scores = scores[indices]
            order = torch.argsort(sub_scores, descending=True, stable=True)[:k]
            indices = indices[order]
    values = gap[indices].clone()
    return indices.to(torch.int32), values


def _scatter_flat_update(state, flat_delta, layout):
    """Build a new state dict with a flat sparse increment applied in place.

    Unlike ``_apply_flat_update`` this does not clone-then-add the whole state;
    it writes the reconstructed values directly so the caller controls the base.
    Returns a new state dict (clones), leaving ``state`` unmodified.
    """
    result = _clone_state(state)
    offset = 0
    for key, shape, count in layout:
        result[key] = flat_delta[offset : offset + count].reshape(shape).clone()
        offset += count
    if offset != flat_delta.numel():
        raise AssertionError("flat delta does not match sparse layout")
    return result


def _flatten_state(state, layout):
    """Flatten the sparse portion of a state dict following ``layout`` order."""
    import torch

    if not layout:
        reference = next(
            value
            for value in state.values()
            if torch.is_floating_point(value) or torch.is_complex(value)
        )
        return reference.new_empty(0)
    return torch.cat([state[key].reshape(-1) for key, _, _ in layout])


def _probe_layer_firing_rates(
    model,
    dataset,
    client_indices,
    device,
    batch_size: int,
    seed: int,
    max_batches,
):
    """Mean per-layer firing rate over a probe of one client's data.

    Runs the model with ``return_layer_activity=True`` over a deterministic
    probe (up to ``max_batches`` batches, or one batch when ``max_batches`` is
    None) and averages the ``[batch, num_layers]`` activity over samples to a
    ``[num_layers]`` vector. Returns a float32 CPU-independent tensor on
    ``device`` aligned with the model's spike-layer order.
    """
    import numpy as _np
    import torch
    from torch.utils.data import DataLoader, Subset

    probe_count = min(int(batch_size), int(client_indices.size))
    if probe_count <= 0:
        raise RuntimeError("downlink firing-rate probe received an empty client partition")
    rng = _np.random.default_rng(seed)
    probe_indices = rng.choice(
        _np.asarray(client_indices, dtype=_np.int64),
        size=probe_count,
        replace=int(client_indices.size) < probe_count,
    )
    probe_loader = DataLoader(
        Subset(dataset, probe_indices.tolist()),
        batch_size=probe_count,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    was_training = model.training
    model.eval()
    total = None
    observed = 0
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(probe_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            _, layer_rates = model(images, return_layer_activity=True)
            if layer_rates.ndim != 2:
                raise RuntimeError("model layer activity must have shape [batch, layers]")
            if total is None:
                total = torch.zeros(
                    layer_rates.shape[1], device=layer_rates.device, dtype=torch.float32
                )
            total.add_(layer_rates.detach().to(torch.float32).sum(dim=0))
            observed += int(layer_rates.shape[0])
            break  # one probe batch is enough for a stable per-layer estimate
    if was_training:
        model.train()
    if total is None or observed == 0:
        raise RuntimeError("downlink firing-rate probe produced no samples")
    return total.div_(observed)


def _channel_update_energy(update, layer_map, channel_map, channel_sizes):
    """Per-channel mean update energy ``mean(delta**2)`` over each channel.

    ``update`` is a flat per-coordinate tensor (sparse update). ``layer_map`` and
    ``channel_map`` assign each coordinate to a (layer, channel); ``channel_sizes``
    is the number of channels per spike layer. Returns a tuple of 1-D tensors.
    """
    import torch

    energy_per_coord = update.to(torch.float32).pow(2)
    result = []
    for layer_id, num_channels in enumerate(channel_sizes):
        mask = layer_map == layer_id
        if not mask.any():
            result.append(torch.zeros(num_channels, device=update.device))
            continue
        layer_energy = torch.zeros(num_channels, device=update.device)
        counts = torch.zeros(num_channels, device=update.device)
        channels = channel_map[mask]
        layer_energy.scatter_add_(0, channels, energy_per_coord[mask])
        counts.scatter_add_(0, channels, torch.ones_like(energy_per_coord[mask]))
        result.append(layer_energy / counts.clamp(min=1.0))
    return tuple(result)


def _probe_neuron_firing_rates(model, images, seed: int):
    """Return deterministic batch-mean channel rates through the model API."""
    import torch

    seed_everything(seed)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(images, return_neuron_activity=True)
    if was_training:
        model.train()
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError("return_neuron_activity must return (logits, layer rates)")
    rates = output[1]
    if not isinstance(rates, (tuple, list)) or any(rate.ndim != 2 for rate in rates):
        raise RuntimeError("neuron activity must be per-layer [batch, channel]")
    return tuple(rate.detach().to(torch.float32).mean(dim=0) for rate in rates)


def _layer_diagnostics(corrected, next_residual, indices, layout):
    """Lightweight per-state-key selection and residual diagnostics."""
    import torch

    selected = indices.to(torch.int64)
    per_layer = {}
    offset = 0
    for key, _, count in layout:
        start, end = offset, offset + count
        offset = end
        mask = (selected >= start) & (selected < end)
        chosen = selected[mask]
        segment = corrected[start:end]
        residual_segment = next_residual[start:end]
        update_energy = float(segment.pow(2).sum())
        residual_energy = float(residual_segment.pow(2).sum())
        retained = 0.0 if update_energy <= 0 else 1.0 - residual_energy / update_energy
        per_layer[key] = {
            "selected": int(chosen.numel()),
            "coordinates": int(count),
            "selection_rate": float(chosen.numel()) / count if count else 0.0,
            "update_l2": float(segment.norm()),
            "residual_l2": float(residual_segment.norm()),
            "retained_energy_ratio": float(retained),
            "zero_energy": bool(update_energy <= 0),
        }
    return per_layer


def _cache_metrics(method: str, cache_versions: dict[int, int], round_index: int):
    if method in CURRENT_ROUND_TOPK_METHODS:
        return {"cache_size": 0, "mean_cache_age": 0.0, "max_cache_age": 0}
    ages = [round_index + 1 - version for version in cache_versions.values()]
    return {
        "cache_size": len(cache_versions),
        "mean_cache_age": float(np.mean(ages)) if ages else 0.0,
        "max_cache_age": max(ages) if ages else 0,
    }


def _aggregate_sparse(payloads, weights, dimension: int):
    import torch

    if not payloads or len(payloads) != len(weights):
        raise ValueError("payloads and weights must be non-empty and aligned")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("aggregation weights must have positive mass")
    output = torch.zeros(
        dimension,
        dtype=payloads[0][1].dtype,
        device=payloads[0][1].device,
    )
    for (indices, values), weight in zip(payloads, weights):
        if indices is None:
            output.add_(values, alpha=float(weight) / total)
        else:
            output.index_add_(0, indices.to(torch.int64), values, alpha=float(weight) / total)
    return output


def _run_topk(
    config: dict,
    data_root: str,
    device_name: str,
    resume: bool = False,
    smoke: bool = False,
    *,
    allowed_methods: set[str],
    trainer_name: str,
) -> Path:
    method = str(config["paper"]["method"])
    if method not in allowed_methods:
        raise ValueError(f"{trainer_name} requires one of {sorted(allowed_methods)}; got {method}")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset

    info = resolve_device(device_name)
    device = activate_device(info)
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    rounds = 1 if smoke else int(config["training"]["rounds"])
    execution_backend = str(
        config["model"].get("execution_backend", "legacy_stepwise")
    ).lower()
    strict_value = config["model"].get("execution_backend_strict", False)
    if not isinstance(strict_value, bool):
        raise ValueError("model.execution_backend_strict must be a boolean")
    execution_backend_strict = strict_value
    configured_local_epochs = int(config["training"]["local_epochs"])
    smoke_requires_two_epochs = method in {
        "training_integrated_credit_topk_saw_snn",
        "training_integrated_credit_topk_saw_ef_snn",
        "training_integrated_credit_topk_snn",
        "training_integrated_credit_topk_ef_snn",
        "anchor_credit_topk_ef_snn",
        "anchor_credit_topk_ef_ema_snn",
        "double_anchor_credit_topk_ef_snn",
        "double_anchor_neuron_topk_ef_snn",
    }
    local_epochs = (
        min(configured_local_epochs, 2 if smoke_requires_two_epochs else 1)
        if smoke
        else configured_local_epochs
    )
    timesteps = (
        int(config["model"]["timesteps"])
        if (not smoke or execution_backend_strict)
        else 1
    )
    max_batches = 1 if smoke else None
    dataset_name, train_set, test_set, model_builder = load_protocol_dataset_and_model(
        config, data_root, timesteps=timesteps
    )
    clients = int(config["federation"]["clients"])
    configured_candidates = int(config["federation"]["candidate_clients"])
    if clients <= 0 or configured_candidates <= 0 or configured_candidates > clients:
        raise ValueError("federation candidate_clients must be in [1, clients]")
    candidates_per_round = 2 if smoke else configured_candidates
    configured_aggregation = int(
        config["federation"].get("aggregation_clients", configured_candidates)
    )
    if configured_aggregation != configured_candidates:
        raise ValueError(
            "Top-k trainer aggregates every candidate; aggregation_clients must equal candidate_clients"
        )
    configured_dense_equivalents = float(
        config["federation"]["dense_upload_equivalents"]
    )
    if configured_dense_equivalents <= 0:
        raise ValueError("federation.dense_upload_equivalents must be positive")
    requested_dense_equivalents = 1.0 if smoke else configured_dense_equivalents
    batch_size = int(config["training"]["batch_size"])
    partition = partition_protocol_labels(config, train_set.targets, min_samples=1)
    partitions = partition.partitions

    if execution_backend_strict and execution_backend != "packed_aspy":
        raise ValueError(
            "model.execution_backend_strict is supported only for packed_aspy"
        )
    if execution_backend_strict and device.type != "npu":
        raise ValueError("strict packed_aspy qualification requires an NPU device")
    if execution_backend == "npugraph":
        if device.type != "npu":
            raise ValueError("model.execution_backend=npugraph requires an NPU device")
        # spikingjelly_npu intentionally rejects nondeterministic training graph
        # capture: hard spike thresholds amplified backend-dependent kernel order
        # in qualification.  This is an explicit opt-in global training policy.
        torch.use_deterministic_algorithms(True, warn_only=False)
    model = model_builder().to(device)
    state_device = _accelerator_state_device(config, device)
    global_state = _clone_state(model.state_dict(), state_device)
    sparse_layout, dense_affine_layout, dense_buffer_layout = _state_layouts(
        model, global_state
    )
    sparse_dimension = sum(count for _, _, count in sparse_layout)
    dense_affine_dimension = sum(count for _, _, count in dense_affine_layout)
    dense_buffer_dimension = sum(count for _, _, count in dense_buffer_layout)
    parameter_dimension = (
        sparse_dimension + dense_affine_dimension + dense_buffer_dimension
    )
    model_bytes = sum(value.numel() * value.element_size() for value in global_state.values())
    compression = config.get("compression", {})
    value_bits = _positive_int(compression["value_bits"], "compression.value_bits")
    # Downlink config must be parsed before layouts that depend on it (spike
    # layer map) and before run_signature / communication accounting.
    downlink = config.get("downlink", {})
    downlink_compression = bool(downlink.get("downlink_compression", False))
    downlink_topk_ratio = float(downlink.get("downlink_topk_ratio", 0.1))
    downlink_credit_mode = str(downlink.get("downlink_credit_mode", "layer_firing"))
    downlink_credit_temp = float(downlink.get("downlink_credit_temp", 1.0))
    downlink_credit_ema_decay = float(
        downlink.get("downlink_credit_ema_decay", 0.5)
    )
    downlink_credit_fn = str(downlink.get("downlink_credit_fn", "exp"))
    downlink_ef = bool(downlink.get("downlink_ef", True))
    neuron_method = method == "double_anchor_neuron_topk_ef_snn"
    if method in {
        "double_credit_topk_ef_snn",
        "double_anchor_credit_topk_ef_snn",
        "double_anchor_neuron_topk_ef_snn",
    } and not downlink_compression:
        raise ValueError(
            f"{method} requires downlink.downlink_compression=true"
        )
    if method in {
        "dual_topk_fedavg_snn",
        "dual_topk_fedavg_ef_snn",
        *DUAL_BUDGET_METHODS,
    } and not downlink_compression:
        raise ValueError(
            f"{method} requires downlink.downlink_compression=true"
        )
    # Dual equal top-k: force credit_mode=none so downlink is pure magnitude top-k.
    # No-EF dual forces downlink_ef=False; EF dual forces downlink_ef=True.
    if method in {"dual_topk_fedavg_snn", "dual_topk_fedavg_ef_snn"}:
        downlink_credit_mode = "none"
        downlink_ef = method == "dual_topk_fedavg_ef_snn"
    # Dual-budget matrix defaults: pure magnitude selection, no temporal credit.
    # dual_global_topk_ef_snn is the exception: server downlink EF is required.
    if method in DUAL_BUDGET_METHODS:
        if method in BLOCK_LOCAL_INT8_METHODS:
            if downlink_credit_mode != "none":
                raise ValueError("block-local INT8 requires downlink_credit_mode=none")
            if not downlink_ef:
                raise ValueError("block-local INT8 requires downlink.downlink_ef=true")
        else:
            downlink_credit_mode = "none"
            if method in DUAL_GLOBAL_SERVER_EF_METHODS:
                downlink_ef = True
            else:
                downlink_ef = False
    dual_channel_support_share = float(downlink.get("support_share", 0.7))
    if method == "dual_channel_quota_dual_topk_snn":
        if not 0.0 <= dual_channel_support_share <= 1.0:
            raise ValueError("downlink.support_share must be in [0, 1]")
    per_client_gap_downlink = method in PER_CLIENT_GAP_DOWNLINK_METHODS
    dual_global_model_downlink = method in DUAL_GLOBAL_MODEL_DOWNLINK_METHODS
    dual_global_server_ef = method in DUAL_GLOBAL_SERVER_EF_METHODS
    block_dual = method in BLOCK_DUAL_METHODS
    block_local_int8 = method in BLOCK_LOCAL_INT8_METHODS
    architecture_block_method = method in ARCHITECTURE_BLOCK_METHODS
    linear_block_size = int(
        config.get("compression", {}).get("linear_block_size", DEFAULT_BLOCK_LINEAR_SIZE)
    )
    if linear_block_size <= 0:
        raise ValueError("compression.linear_block_size must be positive")
    if block_local_int8:
        required_compression = {
            "method": "block_rms_local_magnitude_int8",
            "allocation": "exact_wire_cap",
            "value_bits": 32,
            "sparse_value_bits": 8,
            "scale_bits": 32,
            "error_feedback": False,
            "bntt_affine_upload": "dense",
            "bntt_buffer_upload": "dense",
            "normalization_dense_cost_in_budget": True,
            "ranking_score": "block_rms_then_local_magnitude",
        }
        mismatches = {
            key: {"configured": compression.get(key), "required": required}
            for key, required in required_compression.items()
            if compression.get(key) != required
        }
        if mismatches:
            raise ValueError(
                "block-local INT8 configuration mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
        if execution_backend != "packed_eager":
            raise ValueError(
                "block-local INT8 CIFAR-10 identity requires "
                "model.execution_backend=packed_eager"
            )
    if downlink_compression:
        if not block_local_int8 and not 0.0 < downlink_topk_ratio <= 1.0:
            raise ValueError("downlink.downlink_topk_ratio must be in (0, 1]")
        if downlink_credit_mode not in {
            "none",
            "layer_firing",
            "neuron_firing",
            "channel_energy",
        }:
            raise ValueError(
                "downlink.downlink_credit_mode must be none, layer_firing, neuron_firing, or channel_energy"
            )
        if not math.isfinite(downlink_credit_temp) or downlink_credit_temp < 0:
            raise ValueError("downlink.downlink_credit_temp must be finite and non-negative")
        if (
            not math.isfinite(downlink_credit_ema_decay)
            or not 0.0 <= downlink_credit_ema_decay < 1.0
        ):
            raise ValueError(
                "downlink.downlink_credit_ema_decay must be finite and in [0, 1)"
            )
        if downlink_credit_fn not in {"exp", "linear"}:
            raise ValueError("downlink.downlink_credit_fn must be 'exp' or 'linear'")
        if neuron_method and downlink_credit_mode not in {"neuron_firing", "channel_energy"}:
            raise ValueError(
                "double_anchor_neuron_topk_ef_snn requires downlink_credit_mode=neuron_firing or channel_energy"
            )
        if method in {
            "double_credit_topk_ef_snn",
            "dual_topk_fedavg_ef_snn",
            "double_anchor_credit_topk_ef_snn",
            "double_anchor_neuron_topk_ef_snn",
            *DUAL_GLOBAL_SERVER_EF_METHODS,
        } and not downlink_ef:
            raise ValueError(f"{method} requires downlink.downlink_ef=true")
    downlink_coordinates = (
        max(1, int(round(sparse_dimension * downlink_topk_ratio)))
        if downlink_compression
        else sparse_dimension
    )
    downlink_spike_order = None
    sparse_layer_index_map = None
    sparse_neuron_layer_map = None
    sparse_neuron_channel_map = None
    if downlink_compression:
        downlink_spike_order = _spike_layer_order(model)
        sparse_layer_index_map = _build_layer_index_map(
            sparse_layout, downlink_spike_order
        ).to(state_device)
        if neuron_method:
            channel_sizes = [
                int(size[0] if isinstance(size, (tuple, list)) else size)
                for size in getattr(model, "spike_channel_sizes", ())
            ]
            if len(channel_sizes) != len(downlink_spike_order):
                raise ValueError("neuron method requires model.spike_channel_sizes")
            sparse_neuron_layer_map, sparse_neuron_channel_map = _build_neuron_coordinate_map(
                sparse_layout, downlink_spike_order, channel_sizes
            )
            sparse_neuron_layer_map = sparse_neuron_layer_map.to(state_device)
            sparse_neuron_channel_map = sparse_neuron_channel_map.to(state_device)
    structured_credit_values_per_client = (
        sum(int(size) for size in getattr(model, "spike_channel_sizes", ()))
        if neuron_method
        else 0
    )
    upload_coordinates_per_round_override = _optional_positive_int(
        compression.get("upload_coordinates_per_round"),
        "compression.upload_coordinates_per_round",
    )
    uplink_topk_ratio = compression.get("uplink_topk_ratio")
    if method in BLOCK_LOCAL_INT8_METHODS:
        if uplink_topk_ratio is not None or upload_coordinates_per_round_override is not None:
            raise ValueError(
                "block-local INT8 uses exact total_wire_budget_ratio, not an uplink coordinate target"
            )
    elif uplink_topk_ratio is not None:
        uplink_topk_ratio = float(uplink_topk_ratio)
        if not 0.0 < uplink_topk_ratio <= 1.0:
            raise ValueError("compression.uplink_topk_ratio must be in (0, 1]")
        if upload_coordinates_per_round_override is not None:
            raise ValueError(
                "set only one of compression.upload_coordinates_per_round "
                "and compression.uplink_topk_ratio"
            )
        per_client_coords = max(1, int(round(sparse_dimension * uplink_topk_ratio)))
        upload_coordinates_per_round_override = (
            candidates_per_round * per_client_coords
        )
    planning_method = (
        "dual_global_topk_ef_snn" if method in ARCHITECTURE_BLOCK_METHODS else method
    )
    communication_plan = _communication_plan(
        planning_method,
        sparse_dimension=sparse_dimension,
        dense_affine_dimension=dense_affine_dimension,
        dense_buffer_dimension=dense_buffer_dimension,
        candidates_per_round=candidates_per_round,
        requested_dense_upload_equivalents=requested_dense_equivalents,
        value_bits=value_bits,
        structured_credit_values_per_client=structured_credit_values_per_client,
        upload_coordinates_per_round_override=upload_coordinates_per_round_override,
    )
    index_bits = int(communication_plan["global_index_bits"])
    upload_coordinates_per_round = int(communication_plan["upload_coordinates_per_round"])
    # Block dual: architecture-aligned blocks for both UL and DL; coordinate
    # index_bits are replaced by per-layer block IDs in realized accounting.
    block_layout = None
    block_target_coordinates_per_client = None
    block_table = None
    block_local_sparse_bit_cap = None
    block_local_downlink_sparse_bit_cap = None
    block_local_keep_ratio = None
    block_local_sparse_value_bits = None
    block_local_scale_bits = None
    block_local_total_budget_ratio = None
    mandatory_dense_bits_per_client = (dense_affine_dimension + dense_buffer_dimension) * value_bits
    if architecture_block_method:
        block_layout = _architecture_block_layout(sparse_layout, linear_block_size)
        block_table = _precompute_block_flat_table(block_layout, sparse_layout)
        if block_dual:
            block_target_coordinates_per_client = max(
                1,
                int(
                    round(
                        sparse_dimension
                        * float(uplink_topk_ratio or downlink_topk_ratio)
                    )
                ),
            )
            # Prefer explicit uplink_topk_ratio; fall back to per-client override.
            if uplink_topk_ratio is not None:
                block_target_coordinates_per_client = max(
                    1, int(round(sparse_dimension * uplink_topk_ratio))
                )
            elif upload_coordinates_per_round_override is not None:
                block_target_coordinates_per_client = max(
                    1,
                    int(
                        round(
                            upload_coordinates_per_round_override
                            / max(candidates_per_round, 1)
                        )
                    ),
                )
            # Downlink uses the same coordinate-retention target (block-encoded).
            if downlink_compression:
                downlink_coordinates = max(
                    1, int(round(sparse_dimension * downlink_topk_ratio))
                )
            # Placeholder index_bits for resolved_config; realized bits use block IDs.
            index_bits = 0
            communication_plan = dict(communication_plan)
            communication_plan["encoding"] = "block_id_per_layer"
            communication_plan["block_target_coordinates_per_client"] = (
                block_target_coordinates_per_client
            )
            communication_plan["linear_block_size"] = linear_block_size
            communication_plan["global_index_bits"] = 0
            communication_plan["total_blocks"] = int(block_table["num_blocks"])
            communication_plan["note"] = (
                "block dual: sparse payload = values×32 + per-layer block IDs; "
                "BNTT dense both ways; server whole-block EF on shared residual"
            )
    if block_local_int8:
        block_local_sparse_value_bits = _positive_int(
            compression.get("sparse_value_bits", 8), "compression.sparse_value_bits"
        )
        block_local_scale_bits = _positive_int(
            compression.get("scale_bits", 32), "compression.scale_bits"
        )
        block_local_keep_ratio = float(compression.get("block_local_keep_ratio", 0.5))
        block_local_total_budget_ratio = float(
            compression.get("total_wire_budget_ratio", 0.01)
        )
        if not math.isfinite(block_local_total_budget_ratio) or not 0.0 < block_local_total_budget_ratio <= 1.0:
            raise ValueError("compression.total_wire_budget_ratio must be finite and in (0, 1]")
        _validate_block_local_codec_args(
            sparse_bit_cap=1,
            local_keep_ratio=block_local_keep_ratio,
            sparse_value_bits=block_local_sparse_value_bits,
            scale_bits=block_local_scale_bits,
        )
        full_model_reference_bits = int(model_bytes * 8)
        total_bit_cap_per_client = int(
            math.floor(full_model_reference_bits * block_local_total_budget_ratio)
        )
        block_local_sparse_bit_cap = total_bit_cap_per_client - mandatory_dense_bits_per_client
        configured_downlink_total_budget_ratio = float(
            downlink.get("total_wire_budget_ratio", block_local_total_budget_ratio)
        )
        if not math.isfinite(configured_downlink_total_budget_ratio) or not 0.0 < configured_downlink_total_budget_ratio <= 1.0:
            raise ValueError("downlink.total_wire_budget_ratio must be finite and in (0, 1]")
        downlink_total_bit_cap_per_client = int(
            math.floor(full_model_reference_bits * configured_downlink_total_budget_ratio)
        )
        block_local_downlink_sparse_bit_cap = (
            downlink_total_bit_cap_per_client - mandatory_dense_bits_per_client
        )
        minimum_sparse_framing = int(block_table["layer_count_bits"].sum().item())
        minimum_sparse_value = min(
            int(width)
            + _block_local_support_bits(int(size), 1)[1]
            + block_local_sparse_value_bits
            + block_local_scale_bits
            for width, size in zip(
                block_table["index_widths"].tolist(), block_table["sizes"].tolist()
            )
        )
        minimum_sparse_payload = minimum_sparse_framing + minimum_sparse_value
        if block_local_sparse_bit_cap < minimum_sparse_payload:
            raise ValueError("1% uplink budget cannot encode one block-local INT8 value after dense BNTT")
        if block_local_downlink_sparse_bit_cap < minimum_sparse_payload:
            raise ValueError("1% downlink budget cannot encode one block-local INT8 value after dense BNTT")
        # The exact cap replaces nominal coordinate retention; old ratio keys are
        # intentionally rejected so the new identity cannot silently inherit
        # historical whole-block semantics.
        if "downlink_topk_ratio" in downlink:
            raise ValueError("block-local INT8 uses exact downlink total_wire_budget_ratio, not a coordinate target")
        allowed_downlink_keys = {
            "downlink_compression",
            "downlink_credit_mode",
            "downlink_credit_temp",
            "downlink_ef",
            "total_wire_budget_ratio",
        }
        unknown_downlink_keys = sorted(set(downlink) - allowed_downlink_keys)
        if unknown_downlink_keys:
            raise ValueError(
                "unsupported block-local INT8 downlink keys: "
                + ", ".join(unknown_downlink_keys)
            )
        block_target_coordinates_per_client = None
        downlink_coordinates = None
        index_bits = 0
        communication_plan = {
            "encoding": "block_local_bitmap_or_ids_int8_per_layer_scale",
            "full_model_reference_bits_per_client": full_model_reference_bits,
            "total_wire_budget_ratio": block_local_total_budget_ratio,
            "total_bit_cap_per_client": total_bit_cap_per_client,
            "mandatory_dense_bits_per_client": mandatory_dense_bits_per_client,
            "sparse_bit_cap_per_client": block_local_sparse_bit_cap,
            "downlink_total_wire_budget_ratio": configured_downlink_total_budget_ratio,
            "downlink_total_bit_cap_per_client": downlink_total_bit_cap_per_client,
            "downlink_sparse_bit_cap_per_client": block_local_downlink_sparse_bit_cap,
            "sparse_value_bits": block_local_sparse_value_bits,
            "scale_bits": block_local_scale_bits,
            "block_local_keep_ratio": block_local_keep_ratio,
            "linear_block_size": linear_block_size,
            "total_blocks": int(block_table["num_blocks"]),
            "global_index_bits": 0,
            "note": (
                "exact per-client/per-direction cap includes dense BNTT; sparse wire "
                "uses block IDs + local bitmap/IDs + signed INT8 + one FP32 scale per active layer"
            ),
        }
    if block_local_int8:
        if compression.get("ranking_score") != "block_rms_then_local_magnitude":
            raise ValueError(
                "block-local INT8 requires compression.ranking_score="
                "block_rms_then_local_magnitude"
            )
        ranking_score = "block_rms_then_local_magnitude"
    else:
        ranking_score = _ranking_score(compression)
    uses_credit_scores = _uses_credit_scores(method)
    credit_strength = float(compression.get("credit_strength", 0.75))
    credit_transform = str(compression.get("credit_transform", "sqrt"))
    credit_mode = (
        _credit_mode_for_method(method, compression) if uses_credit_scores else "disabled"
    )
    if credit_mode in {"training_integrated", "anchor"} and local_epochs < 2:
        raise ValueError(
            "training-integrated/anchor Credit-TopK requires local_epochs >= 2"
        )
    credit_ema_beta = float(compression.get("credit_ema_beta", 0.8))
    if uses_credit_scores and not 0.0 <= credit_ema_beta < 1.0:
        raise ValueError("compression.credit_ema_beta must be in [0, 1)")
    credit_uses_ema = _credit_uses_ema(method) if uses_credit_scores else False
    credit_probe_per_class = int(compression.get("credit_probe_per_class", 4))
    probe_indices = {}
    if credit_mode == "probe":
        probe_indices = {
            client_id: _fixed_class_probe_indices(
                train_set.targets,
                partitions[client_id],
                credit_probe_per_class,
                seed + 700_000 + client_id,
            )
            for client_id in range(clients)
        }
    global_step_size = float(config["training"].get("global_step_size", 1.0))
    checkpoint_every = int(config["training"].get("checkpoint_every", 25))
    memory_cleanup_interval = int(config["training"].get("memory_cleanup_interval", 5))
    if memory_cleanup_interval <= 0:
        raise ValueError("training.memory_cleanup_interval must be positive")
    learning_rate = float(config["training"]["learning_rate"])
    momentum = float(config["training"].get("momentum", 0.0))
    weight_decay = float(config["training"].get("weight_decay", 0.0))
    if momentum < 0:
        raise ValueError("training.momentum must be non-negative")
    if weight_decay < 0:
        raise ValueError("training.weight_decay must be non-negative")
    run_signature = {
        "method": method,
        "dataset": copy.deepcopy(config["dataset"]),
        "model": copy.deepcopy(config["model"]),
        "seed": seed,
        "effective_timesteps": timesteps,
        "effective_local_epochs": local_epochs,
        "effective_candidates_per_round": candidates_per_round,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "global_step_size": global_step_size,
        "compression": copy.deepcopy(compression),
        "downlink": copy.deepcopy(downlink),
        "ranking_score": ranking_score,
        "ranking_epsilon": RANKING_SCORE_EPSILON,
        "ranking_tie_policy": "score_desc_index_asc",
        "ranking_normalization_unit": "sparse_state_tensor",
        "ranking_upload_value_semantics": "original_corrected_value",
        "communication_plan": communication_plan,
        "smoke": smoke,
    }

    output = (
        Path(config["output"]["root"]) / "smoke" / method / info.backend / f"seed={seed}"
        if smoke
        else result_dir(config)
    )
    output.mkdir(parents=True, exist_ok=True)
    resolved = copy.deepcopy(config)
    resolved["runtime"] = {
        "device": info.resolved,
        "backend": info.backend,
        "smoke": smoke,
        "effective_rounds": rounds,
        "effective_timesteps": timesteps,
        "effective_local_epochs": local_epochs,
        "effective_candidates_per_round": candidates_per_round,
        "trained_clients_per_round": candidates_per_round,
        "aggregation_clients_per_round": candidates_per_round,
        "configured_selected_clients": config["federation"].get("selected_clients"),
        "dataset_name": dataset_name,
        "parameter_dimension": parameter_dimension,
        "sparse_parameter_dimension": sparse_dimension,
        "dense_bntt_affine_dimension": dense_affine_dimension,
        "dense_bntt_buffer_dimension": dense_buffer_dimension,
        "normalization_upload_policy": "dense_affine_and_buffers",
        "model_bytes": model_bytes,
        "value_bits": value_bits,
        "configured_requested_dense_upload_equivalents": configured_dense_equivalents,
        **communication_plan,
        "credit_scores_enabled": uses_credit_scores,
        "credit_transform": credit_transform,
        "credit_mode": credit_mode,
        "credit_ema_beta": credit_ema_beta if credit_uses_ema else None,
        "credit_probe_per_class": (
            credit_probe_per_class if credit_mode == "probe" else None
        ),
        "credit_extra_forward_passes_per_client": (
            2 if credit_mode in {"full_scan", "probe"} else 0
        ),
        "neuron_probe_forward_batches_per_client": 1 if neuron_method else 0,
        "structured_credit_values_per_client": structured_credit_values_per_client,
        "memory_cleanup_interval": memory_cleanup_interval,
        "state_storage": "accelerator",
        "partition_metadata": partition.metadata,
        "error_feedback": method not in NO_ERROR_FEEDBACK_METHODS and method != "dense_saw_snn",
        "staleness_aware_weighting": method not in CURRENT_ROUND_TOPK_METHODS,
        "downlink_compression": downlink_compression,
        "downlink_topk_ratio": downlink_topk_ratio if downlink_compression else None,
        "downlink_coordinates": downlink_coordinates if downlink_compression else None,
        "downlink_credit_mode": downlink_credit_mode if downlink_compression else None,
        "downlink_credit_temp": downlink_credit_temp if downlink_compression else None,
        "downlink_credit_ema_decay": (
            downlink_credit_ema_decay if neuron_method else None
        ),
        "downlink_credit_fn": downlink_credit_fn if neuron_method else None,
        "downlink_ef": downlink_ef if downlink_compression else None,
        "downlink_policy": (
            "per_client_gap_residual"
            if per_client_gap_downlink and method == "gap_residual_dual_topk_snn"
            else (
                "per_client_dual_channel_quota"
                if method == "dual_channel_quota_dual_topk_snn"
                else (
                    "shared_model_magnitude_topk"
                    if dual_global_model_downlink
                    else (
                        (
                            "shared_block_local_int8_delta_topk_server_ef"
                            if block_local_int8
                            else "shared_block_delta_topk_server_ef"
                        )
                        if architecture_block_method
                        else (
                            "shared_delta_magnitude_topk_server_ef"
                            if dual_global_server_ef
                            else (
                                "shared_delta_magnitude_topk"
                                if downlink_compression
                                else None
                            )
                        )
                    )
                )
            )
        ),
        "uplink_encoding": (
            "block_local_bitmap_or_ids_int8_per_layer_scale"
            if block_local_int8
            else ("block_id_per_layer" if block_dual else "global_coordinate")
        ),
        "downlink_encoding": (
            "block_local_bitmap_or_ids_int8_per_layer_scale"
            if block_local_int8
            else (
                "block_id_per_layer"
                if block_dual
                else ("global_coordinate" if downlink_compression else None)
            )
        ),
        "block_dual": block_dual,
        "block_local_int8": block_local_int8,
        "linear_block_size": linear_block_size if architecture_block_method else None,
        "block_target_coordinates_per_client": (
            block_target_coordinates_per_client if block_dual else None
        ),
        "sparse_value_bits": block_local_sparse_value_bits,
        "sparse_scale_bits": block_local_scale_bits,
        "block_local_keep_ratio": block_local_keep_ratio,
        "total_wire_budget_ratio": block_local_total_budget_ratio,
        "sparse_bit_cap_per_client": block_local_sparse_bit_cap,
        "downlink_sparse_bit_cap_per_client": block_local_downlink_sparse_bit_cap,
        "downlink_support_share": (
            dual_channel_support_share
            if method == "dual_channel_quota_dual_topk_snn"
            else None
        ),
        "periodic_full_downlink_refresh": False,
        "ranking_score": ranking_score,
        "ranking_epsilon": RANKING_SCORE_EPSILON,
        "ranking_tie_policy": "score_desc_index_asc",
        "ranking_normalization_unit": "sparse_state_tensor",
        "ranking_upload_value_semantics": (
            "decoded_signed_int8_value"
            if block_local_int8
            else "original_corrected_value"
        ),
        "code_commit": _code_commit(),
        "run_signature": run_signature,
    }
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    checkpoint_path = output / "latest.pt"
    metrics_path = output / "metrics.jsonl"
    start_round = 0
    cache_payloads = {}
    cache_dense_affine_payloads = {}
    cache_dense_buffer_payloads = {}
    cache_versions = {}
    residuals = {}
    credit_ema = {}
    completed_jobs = 0
    uploaded_bits = 0
    downloaded_bytes = 0
    # Server-side downlink state (Double-Credit-Topk-EF / dual-budget matrix).
    # ``downlink_base_state``: shared sparse reconstruction (legacy dual / dual_global).
    # ``client_base_states``: per-client sparse bases for D1/D2 gap residual.
    # ``downlink_residual``: unsent part of cumulative global increments (EF).
    # ``downlink_prev_global_flat``: previous full global sparse flat (for Δ).
    # EF contract: Δ_t = w^t − w^{t−1} (full global step), R ← R + Δ, send
    # top-k(R), R ← R − sent; client base ← base + sent. Using (w^t − base) as
    # Δ would double-count residual and break EF.
    # D1/D2 contract: G_i = w^global − base_i; rank |G_i| (optionally dual-channel);
    # send top-k of G_i; base_i[selected] ← global[selected]. No periodic full refresh.
    downlink_base_state = _clone_state(global_state)
    client_base_states = (
        {
            client_id: _clone_state(global_state)
            for client_id in range(clients)
        }
        if per_client_gap_downlink
        else None
    )
    downlink_residual = None  # lazily a flat tensor of shape [sparse_dimension]
    downlink_prev_rates = None  # last round's per-layer credit (fallback)
    downlink_ema_rates = None  # server EMA of per-channel neuron firing rates
    downlink_prev_global_flat = (
        _flatten_state(global_state, sparse_layout)
        if downlink_compression and not per_client_gap_downlink
        else None
    )
    # D2: union of uplink support from the previous round (empty at start).
    previous_uplink_support_mask = (
        torch.zeros(sparse_dimension, dtype=torch.bool, device=state_device)
        if method == "dual_channel_quota_dual_topk_snn"
        else None
    )
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("run_signature") != run_signature:
            raise ValueError("resume checkpoint was created by a different Top-k/SAW run")
        model.load_state_dict(checkpoint["model"])
        global_state = _clone_state(checkpoint["model"], state_device)
        checkpoint_round = int(checkpoint["round"])
        reconcile_metrics_for_resume(metrics_path, checkpoint_round)
        start_round = checkpoint_round + 1
        if method not in CURRENT_ROUND_TOPK_METHODS:
            cache_payloads = _move_tensor_tree(checkpoint["cache_payloads"], state_device)
            cache_dense_affine_payloads = _move_tensor_tree(
                checkpoint["cache_dense_affine_payloads"], state_device
            )
            cache_dense_buffer_payloads = _move_tensor_tree(
                checkpoint["cache_dense_buffer_payloads"], state_device
            )
            cache_versions = checkpoint["cache_versions"]
        if method not in NO_ERROR_FEEDBACK_METHODS and method != "dense_saw_snn":
            residuals = _move_tensor_tree(checkpoint["residuals"], state_device)
        if downlink_compression:
            if per_client_gap_downlink:
                if "client_base_states" not in checkpoint:
                    raise ValueError(
                        "resume checkpoint is incompatible with per-client gap downlink; "
                        "missing client_base_states"
                    )
                saved_bases = checkpoint["client_base_states"]
                client_base_states = {
                    int(client_id): _move_tensor_tree(base, state_device)
                    for client_id, base in saved_bases.items()
                }
                if set(client_base_states) != set(range(clients)):
                    raise ValueError(
                        "resume client_base_states does not cover all clients"
                    )
            else:
                required_downlink_state = {
                    "downlink_base_state",
                    "downlink_residual",
                    "downlink_prev_global_flat",
                }
                missing_downlink_state = sorted(
                    required_downlink_state - checkpoint.keys()
                )
                if missing_downlink_state:
                    raise ValueError(
                        "resume checkpoint is incompatible with compressed downlink; "
                        f"missing {missing_downlink_state}"
                    )
                downlink_base_state = _move_tensor_tree(
                    checkpoint["downlink_base_state"], state_device
                )
                saved_residual = checkpoint["downlink_residual"]
                downlink_residual = (
                    _move_tensor_tree(saved_residual, state_device)
                    if saved_residual is not None
                    else None
                )
                downlink_prev_global_flat = _move_tensor_tree(
                    checkpoint["downlink_prev_global_flat"], state_device
                )
                if downlink_prev_global_flat is None:
                    raise ValueError(
                        "compressed-downlink checkpoint has no previous global state"
                    )
            if method == "dual_channel_quota_dual_topk_snn":
                saved_mask = checkpoint.get("previous_uplink_support_mask")
                if saved_mask is None:
                    raise ValueError(
                        "resume checkpoint is incompatible with dual-channel quota; "
                        "missing previous_uplink_support_mask"
                    )
                previous_uplink_support_mask = _move_tensor_tree(
                    saved_mask, state_device
                ).to(dtype=torch.bool)
            if neuron_method:
                saved_ema_rates = checkpoint.get("downlink_ema_rates")
                if saved_ema_rates is None:
                    raise ValueError(
                        "resume checkpoint is incompatible with neuron EMA; "
                        "missing downlink_ema_rates"
                    )
                downlink_ema_rates = tuple(
                    _move_tensor_tree(rate, state_device) for rate in saved_ema_rates
                )
        credit_ema = {
            int(client_id): float(value)
            for client_id, value in checkpoint.get("credit_ema", {}).items()
        }
        completed_jobs = int(checkpoint["completed_jobs"])
        uploaded_bits = int(checkpoint["uploaded_bits"])
        downloaded_bytes = int(checkpoint["downloaded_bytes"])
    else:
        metrics_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
    if rounds < start_round:
        raise ValueError(
            f"configured rounds={rounds} precede checkpoint continuation round {start_round}"
        )

    criterion = nn.CrossEntropyLoss()
    local_model = model_builder().to(device)
    local_forward = model_forward_runner(local_model, batch_size)
    # Global eval reuses the same explicitly configured model path.
    eval_forward = model_forward_runner(model, batch_size)

    class ActivityForward(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, inputs):
            return self.wrapped(inputs, return_activity=True)

    local_activity_forward = (
        StaticBatchCudaGraph(ActivityForward(local_model), batch_size)
        if credit_mode in {"training_integrated", "anchor"}
        else None
    )
    resolved["runtime"].update(model_forward_runtime_metadata(local_forward))
    resolved["runtime"]["eval_forward_backend"] = eval_forward.backend
    resolved["runtime"]["activity_forward_backend"] = (
        local_activity_forward.backend if local_activity_forward is not None else None
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    for round_index in range(start_round, rounds):
        started = time.time()
        candidates = np.random.default_rng(seed + 3000 + round_index).choice(
            clients, size=candidates_per_round, replace=False
        ).tolist()
        # With downlink compression clients train from the sparse-reconstructed
        # model (what they were actually sent), not the full global model.
        # D1/D2 keep a distinct base per client; legacy dual shares one base.
        shared_base_state = (
            None
            if per_client_gap_downlink
            else (
                _clone_state(downlink_base_state)
                if downlink_compression
                else _clone_state(global_state)
            )
        )
        scores = {}
        raw_scores = {}
        sparse_updates = {}
        dense_affine_updates = {}
        dense_buffer_updates = {}
        losses = {}
        candidate_layer_rates = {}  # per-candidate per-layer firing rate (downlink credit)
        candidate_neuron_rates = {}  # per-candidate per-channel firing rate
        # Round-local union of this round's uplink support (for next-round D2).
        round_uplink_support_mask = (
            torch.zeros(sparse_dimension, dtype=torch.bool, device=state_device)
            if method == "dual_channel_quota_dual_topk_snn"
            else None
        )
        downlink_selected_coords = None
        downlink_residual_l2 = 0.0
        downlink_layer_weight_mean = 1.0
        downlink_unmapped_frac = None
        downlink_credit_ema_delta_l2 = None
        downlink_credit_ema_rate_l2 = None
        client_gap_l2_values = []
        downlink_selected_per_client = []
        round_downlink_block_stats = None
        round_uplink_block_stats = []

        # D1/D2: before local train, catch up each candidate from its own gap.
        if per_client_gap_downlink and downlink_compression:
            global_flat = _flatten_state(global_state, sparse_layout)
            support_mask = (
                previous_uplink_support_mask
                if previous_uplink_support_mask is not None
                else torch.zeros(
                    sparse_dimension, dtype=torch.bool, device=state_device
                )
            )
            for client_id in candidates:
                base_flat = _flatten_state(client_base_states[client_id], sparse_layout)
                gap = global_flat - base_flat
                client_gap_l2_values.append(float(gap.norm().detach().cpu()))
                if method == "dual_channel_quota_dual_topk_snn":
                    indices, values = _downlink_dual_channel_quota(
                        gap,
                        downlink_coordinates,
                        support_mask,
                        dual_channel_support_share,
                    )
                else:
                    # D1 pure gap residual: top-k by |G_i|.
                    gap_scores = gap.abs()
                    indices = _stable_topk(gap_scores, downlink_coordinates).to(
                        torch.int32
                    )
                    values = gap[indices.to(torch.int64)].clone()
                # Apply catch-up: base[selected] becomes global[selected].
                idx64 = indices.to(torch.int64)
                new_base_flat = base_flat.clone()
                new_base_flat[idx64] = global_flat[idx64]
                client_base_states[client_id] = _scatter_flat_update(
                    client_base_states[client_id], new_base_flat, sparse_layout
                )
                # Dense affine + buffers always full-sync (same as legacy dual).
                for key, _, _ in dense_affine_layout:
                    client_base_states[client_id][key] = global_state[key].detach().clone()
                for key, _, _ in dense_buffer_layout:
                    client_base_states[client_id][key] = global_state[key].detach().clone()
                downlink_selected_per_client.append(int(indices.numel()))
            downlink_selected_coords = (
                int(sum(downlink_selected_per_client) / max(len(downlink_selected_per_client), 1))
                if downlink_selected_per_client
                else 0
            )
            residual_gaps = []
            for client_id in candidates:
                base_flat = _flatten_state(client_base_states[client_id], sparse_layout)
                residual_gaps.append(float((global_flat - base_flat).norm().detach().cpu()))
            downlink_residual_l2 = (
                float(sum(residual_gaps) / max(len(residual_gaps), 1))
                if residual_gaps
                else 0.0
            )

        for client_id in candidates:
            if per_client_gap_downlink:
                base_state = client_base_states[client_id]
            else:
                base_state = shared_base_state
            local_model.load_state_dict(base_state)
            rate_seed = seed + 100_000 * round_index + client_id
            if credit_mode in {"full_scan", "probe"}:
                credit_indices = (
                    probe_indices[client_id]
                    if credit_mode == "probe"
                    else partitions[client_id]
                )
                before = _class_firing_rates(
                    local_model,
                    train_set,
                    credit_indices,
                    device,
                    batch_size,
                    rate_seed,
                    max_batches,
                )
            local_model.train()
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            loader = DataLoader(
                Subset(train_set, partitions[client_id].tolist()),
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                generator=torch.Generator().manual_seed(seed + round_index * clients + client_id),
            )
            client_losses = []
            integrated_early_sums = torch.zeros(10, dtype=torch.float32, device=device)
            integrated_early_counts = torch.zeros(10, dtype=torch.float32, device=device)
            integrated_late_sums = torch.zeros(10, dtype=torch.float32, device=device)
            integrated_late_counts = torch.zeros(10, dtype=torch.float32, device=device)
            training_stream_seed = seed + 400_000 * round_index + client_id
            anchor_images = None
            anchor_labels = None
            anchor_rate_seed = None
            if credit_mode == "anchor":
                anchor_pool = np.asarray(partitions[client_id], dtype=np.int64)
                anchor_rng = np.random.default_rng(
                    seed + 900_000 + 1000 * round_index + client_id
                )
                anchor_indices = anchor_rng.choice(
                    anchor_pool,
                    size=batch_size,
                    replace=anchor_pool.size < batch_size,
                )
                anchor_loader = DataLoader(
                    Subset(train_set, anchor_indices.tolist()),
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,
                )
                anchor_images, anchor_labels = next(iter(anchor_loader))
                anchor_images = anchor_images.to(device)
                anchor_labels = anchor_labels.to(device)
                anchor_rate_seed = seed + 600_000 + 100_000 * round_index + client_id
            seed_everything(training_stream_seed)
            for local_epoch in range(local_epochs):
                if credit_mode == "anchor" and local_epoch == 0:
                    # Anchor batch is injected as the first training batch of the
                    # local round.  The before/after forwards share one rate seed
                    # (common random numbers), and the training RNG stream is
                    # restored afterwards so non-anchor batches are unaffected.
                    seed_everything(anchor_rate_seed)
                    optimizer.zero_grad(set_to_none=True)
                    logits, sample_rates = local_activity_forward(anchor_images)
                    _accumulate_class_activity(
                        integrated_early_sums,
                        integrated_early_counts,
                        anchor_labels,
                        sample_rates,
                    )
                    loss = criterion(logits, anchor_labels)
                    loss.backward()
                    optimizer.step()
                    client_losses.append(loss.detach())
                    seed_everything(training_stream_seed)
                for batch_index, (images, labels) in enumerate(loader):
                    if max_batches is not None and batch_index >= max_batches:
                        break
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    if credit_mode == "training_integrated":
                        logits, sample_rates = local_activity_forward(images)
                        if local_epoch == 0:
                            _accumulate_class_activity(
                                integrated_early_sums,
                                integrated_early_counts,
                                labels,
                                sample_rates,
                            )
                        if local_epoch == local_epochs - 1:
                            _accumulate_class_activity(
                                integrated_late_sums,
                                integrated_late_counts,
                                labels,
                                sample_rates,
                            )
                    else:
                        logits = local_forward(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    client_losses.append(loss.detach())
                if credit_mode == "anchor" and local_epoch == local_epochs - 1:
                    # The same anchor batch is re-injected as the last training
                    # batch, reusing the same rate seed for a paired comparison.
                    seed_everything(anchor_rate_seed)
                    optimizer.zero_grad(set_to_none=True)
                    logits, sample_rates = local_activity_forward(anchor_images)
                    _accumulate_class_activity(
                        integrated_late_sums,
                        integrated_late_counts,
                        anchor_labels,
                        sample_rates,
                    )
                    loss = criterion(logits, anchor_labels)
                    loss.backward()
                    optimizer.step()
                    client_losses.append(loss.detach())
                    seed_everything(training_stream_seed)
            if credit_mode in {"full_scan", "probe"}:
                after = _class_firing_rates(
                    local_model,
                    train_set,
                    credit_indices,
                    device,
                    batch_size,
                    rate_seed,
                    max_batches,
                )
                observed_credit = float((after - before).abs().sum())
            elif credit_mode in {"training_integrated", "anchor"}:
                early = _finish_class_activity(
                    integrated_early_sums, integrated_early_counts
                )
                late = _finish_class_activity(
                    integrated_late_sums, integrated_late_counts
                )
                observed_credit = float((late - early).abs().sum().detach().cpu())
            if uses_credit_scores:
                raw_scores[client_id] = observed_credit
                if credit_uses_ema:
                    scores[client_id] = _update_credit_ema(
                        credit_ema,
                        client_id,
                        observed_credit,
                        credit_ema_beta,
                    )
                else:
                    scores[client_id] = observed_credit
            local_state = _clone_state(local_model.state_dict(), state_device)
            if neuron_method and downlink_credit_mode == "neuron_firing":
                candidate_neuron_rates[client_id] = _probe_neuron_firing_rates(
                    local_model,
                    anchor_images,
                    seed + 800_000 + 100_000 * round_index + client_id,
                )
                seed_everything(training_stream_seed)
            if downlink_compression and downlink_credit_mode == "layer_firing" and not neuron_method:
                candidate_layer_rates[client_id] = _probe_layer_firing_rates(
                    local_model,
                    train_set,
                    partitions[client_id],
                    device,
                    batch_size,
                    seed + 500_000 * round_index + client_id,
                    max_batches,
                )
            sparse_updates[client_id] = _flatten_difference(
                local_state, base_state, sparse_layout
            )
            dense_affine_updates[client_id] = _flatten_difference(
                local_state, base_state, dense_affine_layout
            )
            dense_buffer_updates[client_id] = _flatten_difference(
                local_state, base_state, dense_buffer_layout
            )
            if neuron_method and downlink_credit_mode == "channel_energy":
                candidate_neuron_rates[client_id] = _channel_update_energy(
                    sparse_updates[client_id],
                    sparse_neuron_layer_map,
                    sparse_neuron_channel_map,
                    [int(size[0] if isinstance(size, (tuple, list)) else size)
                     for size in getattr(model, "spike_channel_sizes", ())],
                )
            loss_values = torch.stack(client_losses).cpu().tolist()
            losses[client_id] = sum(loss_values) / len(loss_values)
            del optimizer, local_state, client_losses, loss_values

        if method == "dense_saw_snn":
            budgets = {client_id: sparse_dimension for client_id in candidates}
        elif block_dual:
            # Per-client coordinate retention target (filled with whole blocks).
            budgets = {
                client_id: int(block_target_coordinates_per_client)
                for client_id in candidates
            }
        elif block_local_int8:
            # Exact sparse wire cap after mandatory dense BNTT state.
            budgets = {
                client_id: int(block_local_sparse_bit_cap)
                for client_id in candidates
            }
        elif uses_credit_scores:
            budgets = allocate_topk_budget(
                scores,
                upload_coordinates_per_round,
                credit_strength,
                credit_transform,
                sparse_dimension,
            )
        else:
            budgets = equal_topk_budget(candidates, upload_coordinates_per_round)
        current_payloads = {}
        current_block_packs = {}
        current_dense_affine = {}
        current_dense_buffer = {}
        round_layer_diagnostics = {}
        round_uplink_block_stats = []
        round_uplink_block_stats_by_client = {}
        compression_time = 0.0
        for client_id in candidates:
            if method == "dense_saw_snn":
                indices, values, residual = None, sparse_updates[client_id].clone(), None
                corrected = None
            elif block_dual:
                compress_started = time.perf_counter()
                update = sparse_updates.pop(client_id)
                packed, selected_blocks, bstats = _compress_blocks_no_ef(
                    update,
                    sparse_layout,
                    block_layout,
                    budgets[client_id],
                    linear_block_size=linear_block_size,
                    block_table=block_table,
                )
                compression_time += time.perf_counter() - compress_started
                residual = None
                corrected = update
                indices, values = None, None
                current_block_packs[client_id] = packed
                round_uplink_block_stats.append(bstats)
                round_uplink_block_stats_by_client[client_id] = bstats
            elif block_local_int8:
                compress_started = time.perf_counter()
                update = sparse_updates.pop(client_id)
                payload, bstats = _block_local_int8_compress(
                    update,
                    sparse_layout,
                    block_layout,
                    block_table,
                    sparse_bit_cap=budgets[client_id],
                    local_keep_ratio=block_local_keep_ratio,
                    sparse_value_bits=block_local_sparse_value_bits,
                    scale_bits=block_local_scale_bits,
                )
                decoded_pack = _decode_block_local_int8_payload(
                    payload, block_table, update
                )
                compression_time += time.perf_counter() - compress_started
                residual = None
                corrected = update
                indices, values = None, None
                current_block_packs[client_id] = decoded_pack
                round_uplink_block_stats.append(bstats)
                round_uplink_block_stats_by_client[client_id] = bstats
            elif method in NO_ERROR_FEEDBACK_METHODS:
                compress_started = time.perf_counter()
                update = sparse_updates.pop(client_id)
                indices, values, _ = _compress_with_error_feedback(
                    update, None, budgets[client_id], sparse_layout, ranking_score
                )
                compression_time += time.perf_counter() - compress_started
                residual = None
                corrected = update
            else:
                compress_started = time.perf_counter()
                update = sparse_updates.pop(client_id)
                old_residual = residuals.get(client_id)
                corrected = update if old_residual is None else update + old_residual
                neuron_weight = None
                if neuron_method:
                    neuron_weight = _neuron_coordinate_weight(
                        candidate_neuron_rates[client_id],
                        sparse_neuron_layer_map,
                        sparse_neuron_channel_map,
                        downlink_credit_temp,
                        downlink_credit_fn,
                    )
                indices, values, residual = _compress_with_error_feedback(
                    update,
                    old_residual,
                    budgets[client_id],
                    sparse_layout,
                    ranking_score,
                    coordinate_weight=neuron_weight,
                )
                compression_time += time.perf_counter() - compress_started
            if indices is not None:
                round_layer_diagnostics[client_id] = _layer_diagnostics(
                    corrected, residual if residual is not None else torch.zeros_like(corrected),
                    indices, sparse_layout
                )
                if round_uplink_support_mask is not None:
                    round_uplink_support_mask[indices.to(torch.int64)] = True
            if not architecture_block_method:
                current_payloads[client_id] = (indices, values)
            current_dense_affine[client_id] = dense_affine_updates.pop(client_id)
            current_dense_buffer[client_id] = dense_buffer_updates.pop(client_id)
            if residual is not None:
                residuals[client_id] = residual
            if method in CURRENT_ROUND_TOPK_METHODS:
                continue
            cache_payloads[client_id] = (indices, values)
            cache_dense_affine_payloads[client_id] = current_dense_affine[client_id]
            cache_dense_buffer_payloads[client_id] = current_dense_buffer[client_id]
            cache_versions[client_id] = round_index

        if method in CURRENT_ROUND_TOPK_METHODS:
            aggregate_clients = sorted(candidates)
            weights = [1.0] * len(aggregate_clients)
            aggregate_payloads = current_payloads
            aggregate_dense_affine_payloads = current_dense_affine
            aggregate_dense_buffer_payloads = current_dense_buffer
        else:
            weight_map = harmonic_cache_weights(cache_versions, round_index)
            aggregate_clients = sorted(weight_map)
            weights = [weight_map[client_id] for client_id in aggregate_clients]
            aggregate_payloads = cache_payloads
            aggregate_dense_affine_payloads = cache_dense_affine_payloads
            aggregate_dense_buffer_payloads = cache_dense_buffer_payloads
        if architecture_block_method:
            like = _flatten_state(global_state, sparse_layout)
            agg_pack = _aggregate_block_packs(
                [current_block_packs[client_id] for client_id in aggregate_clients],
                weights,
                sparse_layout,
                like=like,
            )
            aggregate = _flat_from_block_pack(agg_pack, sparse_layout, like)
        else:
            aggregate = _aggregate_sparse(
                [aggregate_payloads[client_id] for client_id in aggregate_clients],
                weights,
                sparse_dimension,
            )
        aggregate_dense_affine = _aggregate_sparse(
            [(None, aggregate_dense_affine_payloads[client_id]) for client_id in aggregate_clients],
            weights,
            dense_affine_dimension,
        )
        aggregate_dense_buffer = _aggregate_sparse(
            [(None, aggregate_dense_buffer_payloads[client_id]) for client_id in aggregate_clients],
            weights,
            dense_buffer_dimension,
        )
        global_state = _apply_flat_update(
            global_state, aggregate, sparse_layout, global_step_size
        )
        global_state = _apply_flat_update(
            global_state,
            aggregate_dense_affine,
            dense_affine_layout,
            global_step_size,
        )
        global_state = _apply_flat_update(
            global_state,
            aggregate_dense_buffer,
            dense_buffer_layout,
            global_step_size,
        )
        model.load_state_dict(global_state)

        # Advance D2 support mask for the next round's dual-channel priority.
        if round_uplink_support_mask is not None:
            previous_uplink_support_mask = round_uplink_support_mask.detach().clone()

        # ---- Downlink (server -> client) ----
        # Legacy dual / dual_global_topk share one reconstructed base and run
        # after aggregation. D1/D2 already applied per-client catch-up before
        # train above; skip the shared path for those methods.
        if downlink_compression and not per_client_gap_downlink:
            # Keep legacy metric defaults when re-entering shared path.
            downlink_selected_coords = None
            downlink_residual_l2 = 0.0
            downlink_layer_weight_mean = 1.0
            downlink_unmapped_frac = None
            downlink_credit_ema_delta_l2 = None
            downlink_credit_ema_rate_l2 = None
            global_flat = _flatten_state(global_state, sparse_layout)
            round_downlink_block_stats = None
            if block_dual:
                # Historical Scheme A + server whole-block EF on shared Δ residual.
                if downlink_prev_global_flat is None:
                    downlink_prev_global_flat = global_flat.detach().clone()
                delta = global_flat - downlink_prev_global_flat
                packed, selected_blocks, next_residual, bstats = (
                    _downlink_block_topk_with_ef(
                        delta,
                        downlink_residual,
                        sparse_layout,
                        block_layout,
                        downlink_coordinates,
                        use_error_feedback=downlink_ef,
                        linear_block_size=linear_block_size,
                        block_table=block_table,
                    )
                )
                downlink_residual = next_residual
                downlink_residual_l2 = float(downlink_residual.norm().detach().cpu())
                downlink_selected_coords = int(bstats["selected_coordinates"])
                round_downlink_block_stats = bstats
                base_flat = _flatten_state(downlink_base_state, sparse_layout)
                new_base_flat = _scatter_blocks_into_flat(
                    base_flat, sparse_layout, packed, mode="add"
                )
                downlink_base_state = _scatter_flat_update(
                    downlink_base_state, new_base_flat, sparse_layout
                )
                for key, _, _ in dense_affine_layout:
                    downlink_base_state[key] = global_state[key].detach().clone()
                for key, _, _ in dense_buffer_layout:
                    downlink_base_state[key] = global_state[key].detach().clone()
                downlink_prev_global_flat = global_flat.detach().clone()
            elif block_local_int8:
                # New closed loop: rank corrected FP32, transmit actual signed
                # INT8 codes, reconstruct from decoded wire, and retain both
                # sparsification and quantization error in the server residual.
                if downlink_prev_global_flat is None:
                    downlink_prev_global_flat = global_flat.detach().clone()
                delta = global_flat - downlink_prev_global_flat
                _, decoded_pack, next_residual, bstats = (
                    _downlink_block_local_int8_with_ef(
                        delta,
                        downlink_residual,
                        sparse_layout,
                        block_layout,
                        block_table,
                        sparse_bit_cap=block_local_downlink_sparse_bit_cap,
                        local_keep_ratio=block_local_keep_ratio,
                        use_error_feedback=downlink_ef,
                        sparse_value_bits=block_local_sparse_value_bits,
                        scale_bits=block_local_scale_bits,
                    )
                )
                downlink_residual = next_residual
                downlink_residual_l2 = float(downlink_residual.norm().detach().cpu())
                downlink_selected_coords = int(bstats["selected_coordinates"])
                round_downlink_block_stats = bstats
                base_flat = _flatten_state(downlink_base_state, sparse_layout)
                new_base_flat = _scatter_blocks_into_flat(
                    base_flat, sparse_layout, decoded_pack, mode="add"
                )
                downlink_base_state = _scatter_flat_update(
                    downlink_base_state, new_base_flat, sparse_layout
                )
                for key, _, _ in dense_affine_layout:
                    downlink_base_state[key] = global_state[key].detach().clone()
                for key, _, _ in dense_buffer_layout:
                    downlink_base_state[key] = global_state[key].detach().clone()
                downlink_prev_global_flat = global_flat.detach().clone()
            elif dual_global_model_downlink:
                # dual_global_topk: rank |model| (global sparse weights), send
                # those coordinates' *current global values* as absolute sparse
                # writes onto the shared client base (not a delta residual).
                model_scores = global_flat.abs()
                indices = _stable_topk(model_scores, downlink_coordinates).to(
                    torch.int32
                )
                idx64 = indices.to(torch.int64)
                values = global_flat[idx64].clone()
                base_flat = _flatten_state(downlink_base_state, sparse_layout)
                new_base_flat = base_flat.clone()
                new_base_flat[idx64] = values
                downlink_base_state = _scatter_flat_update(
                    downlink_base_state, new_base_flat, sparse_layout
                )
                for key, _, _ in dense_affine_layout:
                    downlink_base_state[key] = global_state[key].detach().clone()
                for key, _, _ in dense_buffer_layout:
                    downlink_base_state[key] = global_state[key].detach().clone()
                downlink_selected_coords = int(indices.numel())
                residual_gap = global_flat - new_base_flat
                downlink_residual_l2 = float(residual_gap.norm().detach().cpu())
                downlink_prev_global_flat = global_flat.detach().clone()
            else:
                # Shared Δ path (dual_topk_fedavg / credit EF variants).
                if downlink_prev_global_flat is None:
                    downlink_prev_global_flat = global_flat.detach().clone()
                delta = global_flat - downlink_prev_global_flat
                if neuron_method and candidate_neuron_rates:
                    neuron_rates = tuple(
                        torch.stack(
                            [candidate_neuron_rates[c][layer] for c in candidates]
                        )
                        .mean(dim=0)
                        .to(state_device)
                        for layer in range(len(candidate_neuron_rates[candidates[0]]))
                    )
                    previous_ema_rates = downlink_ema_rates
                    if previous_ema_rates is None:
                        # Initialize from the first observation so decay=0 exactly
                        # reproduces the original raw-rate ranking without a zero bias.
                        downlink_ema_rates = tuple(
                            rate.detach().clone() for rate in neuron_rates
                        )
                        ema_delta = tuple(
                            torch.zeros_like(rate) for rate in neuron_rates
                        )
                    else:
                        downlink_ema_rates = tuple(
                            downlink_credit_ema_decay * old
                            + (1.0 - downlink_credit_ema_decay) * new
                            for old, new in zip(previous_ema_rates, neuron_rates)
                        )
                        ema_delta = tuple(
                            new_ema - old
                            for new_ema, old in zip(
                                downlink_ema_rates, previous_ema_rates
                            )
                        )
                    downlink_credit_ema_delta_l2 = math.sqrt(
                        sum(
                            float(value.square().sum().detach().cpu())
                            for value in ema_delta
                        )
                    )
                    downlink_credit_ema_rate_l2 = math.sqrt(
                        sum(
                            float(value.square().sum().detach().cpu())
                            for value in downlink_ema_rates
                        )
                    )
                    credit_weight = _neuron_coordinate_weight(
                        downlink_ema_rates,
                        sparse_neuron_layer_map,
                        sparse_neuron_channel_map,
                        downlink_credit_temp,
                        downlink_credit_fn,
                    ).to(dtype=delta.dtype)
                    layer_rates = None
                elif downlink_credit_mode == "layer_firing" and candidate_layer_rates:
                    stacked = torch.stack(
                        [
                            candidate_layer_rates[c]
                            for c in candidates
                            if c in candidate_layer_rates
                        ]
                    )
                    layer_rates = stacked.mean(dim=0).to(state_device)
                    downlink_prev_rates = layer_rates
                elif (
                    downlink_credit_mode == "layer_firing"
                    and downlink_prev_rates is not None
                ):
                    layer_rates = downlink_prev_rates
                else:
                    layer_rates = None
                num_spike_layers = (
                    len(downlink_spike_order) if downlink_spike_order is not None else 0
                )
                if neuron_method and candidate_neuron_rates:
                    downlink_layer_weight_mean = float(
                        credit_weight.mean().detach().cpu()
                    )
                elif (
                    layer_rates is not None
                    and sparse_layer_index_map is not None
                    and layer_rates.numel() == num_spike_layers
                ):
                    credit_weight = _credit_coordinate_weight(
                        layer_rates, sparse_layer_index_map, downlink_credit_temp
                    )
                    downlink_layer_weight_mean = float(
                        credit_weight.mean().detach().cpu()
                    )
                else:
                    credit_weight = torch.ones(
                        sparse_dimension, device=delta.device, dtype=delta.dtype
                    )
                active_downlink_map = (
                    sparse_neuron_layer_map
                    if neuron_method
                    else sparse_layer_index_map
                )
                if (
                    active_downlink_map is not None
                    and active_downlink_map.numel() > 0
                ):
                    downlink_unmapped_frac = float(
                        (active_downlink_map < 0)
                        .to(torch.float32)
                        .mean()
                        .detach()
                        .cpu()
                    )
                indices, values, next_residual = _downlink_topk_with_credit(
                    delta,
                    downlink_residual,
                    credit_weight,
                    downlink_coordinates,
                    downlink_ef,
                )
                downlink_residual = next_residual
                downlink_residual_l2 = float(downlink_residual.norm().detach().cpu())
                downlink_selected_coords = int(indices.numel())
                # Apply only the sent coordinates onto the client-held reconstruction.
                base_flat = _flatten_state(downlink_base_state, sparse_layout)
                sent_flat = torch.zeros_like(base_flat)
                sent_flat[indices.to(torch.int64)] = values
                new_base_flat = base_flat + sent_flat
                downlink_base_state = _scatter_flat_update(
                    downlink_base_state, new_base_flat, sparse_layout
                )
                # Non-sparse (dense affine + buffers) are sent in full each round.
                for key, _, _ in dense_affine_layout:
                    downlink_base_state[key] = global_state[key].detach().clone()
                for key, _, _ in dense_buffer_layout:
                    downlink_base_state[key] = global_state[key].detach().clone()
                # Advance full-global reference for next round's Δ.
                downlink_prev_global_flat = global_flat.detach().clone()

        accuracy, test_loss = _evaluate(
            model,
            test_set,
            device,
            batch_size,
            2 if smoke else None,
            model_forward=eval_forward,
        )
        if execution_backend_strict:
            training_route = getattr(local_forward, "last_route", None)
            evaluation_route = getattr(eval_forward, "last_route", None)
            training_lif_routes = tuple(
                getattr(local_model, "last_lif_routes", ())
            )
            evaluation_lif_routes = tuple(
                getattr(model, "last_lif_routes", ())
            )
            training_native = len(training_lif_routes) == 6 and all(
                getattr(route, "backend", None) == "aspy"
                and getattr(route, "requested_backend", None) == "aspy"
                for route in training_lif_routes
            )
            evaluation_native = len(evaluation_lif_routes) == 6 and all(
                getattr(route, "backend", None) == "aspy"
                and getattr(route, "requested_backend", None) == "aspy"
                for route in evaluation_lif_routes
            )
            if (
                training_route != "packed_aspy"
                or evaluation_route != "packed_aspy"
                or not training_native
                or not evaluation_native
            ):
                raise RuntimeError(
                    "strict packed_aspy qualification requires six native training "
                    "and evaluation LIF routes; "
                    f"training={training_route!r}/{len(training_lif_routes)}, "
                    f"eval={evaluation_route!r}/{len(evaluation_lif_routes)}"
                )

        if architecture_block_method:
            # Realized UL sparse bits = sum of per-client architecture-block payloads.
            uplink_value_bits = sum(
                int(s["value_bits"]) for s in round_uplink_block_stats
            )
            uplink_scale_bits = sum(
                int(s.get("scale_bits", 0)) for s in round_uplink_block_stats
            )
            uplink_index_bits = sum(
                int(s["index_bits"]) for s in round_uplink_block_stats
            )
            uplink_sparse_bits = uplink_value_bits + uplink_scale_bits + uplink_index_bits
            dense_affine_bits = (
                candidates_per_round * dense_affine_dimension * value_bits
            )
            dense_buffer_bits = (
                candidates_per_round * dense_buffer_dimension * value_bits
            )
            data_upload_bits = (
                uplink_sparse_bits + dense_affine_bits + dense_buffer_bits
            )
            round_communication = {
                "sparse_upload_bits": uplink_sparse_bits,
                "uplink_value_bits": uplink_value_bits,
                "uplink_scale_bits": uplink_scale_bits,
                "uplink_index_bits": uplink_index_bits,
                "dense_affine_upload_bits": dense_affine_bits,
                "dense_buffer_upload_bits": dense_buffer_bits,
                "data_upload_bits": data_upload_bits,
                "credit_payload_bits": 0,
                "structured_credit_payload_bits": 0,
                "total_upload_bits": data_upload_bits,
                "encoding": (
                    "block_local_bitmap_or_ids_int8_per_layer_scale"
                    if block_local_int8
                    else "block_id_per_layer"
                ),
                "uplink_selected_blocks": sum(
                    int(s["selected_blocks"]) for s in round_uplink_block_stats
                ),
                "uplink_selected_coordinates": sum(
                    int(s["selected_coordinates"]) for s in round_uplink_block_stats
                ),
            }
            if round_downlink_block_stats is not None:
                round_communication["downlink_value_bits"] = int(
                    round_downlink_block_stats["value_bits"]
                )
                round_communication["downlink_scale_bits"] = int(
                    round_downlink_block_stats.get("scale_bits", 0)
                )
                round_communication["downlink_index_bits"] = int(
                    round_downlink_block_stats["index_bits"]
                )
                round_communication["downlink_payload_bits"] = int(
                    round_downlink_block_stats["payload_bits"]
                )
                round_communication["downlink_selected_blocks"] = int(
                    round_downlink_block_stats["selected_blocks"]
                )
            if block_local_int8:
                realized_caps = [
                    int(s["payload_bits"]) + mandatory_dense_bits_per_client
                    for s in round_uplink_block_stats
                ]
                if any(bits > int(communication_plan["total_bit_cap_per_client"]) for bits in realized_caps):
                    raise AssertionError("realized block-local uplink exceeded the exact total wire cap")
                if round_downlink_block_stats is not None:
                    realized_downlink = (
                        int(round_downlink_block_stats["payload_bits"])
                        + mandatory_dense_bits_per_client
                    )
                    if realized_downlink > int(communication_plan["downlink_total_bit_cap_per_client"]):
                        raise AssertionError("realized block-local downlink exceeded the exact total wire cap")
            # Architecture-block methods skip coordinate communication-plan equality checks.
        else:
            round_communication = _round_communication(
                method,
                budgets,
                sparse_dimension=sparse_dimension,
                dense_affine_dimension=dense_affine_dimension,
                dense_buffer_dimension=dense_buffer_dimension,
                value_bits=value_bits,
                index_bits=index_bits,
                structured_credit_values_per_client=structured_credit_values_per_client,
            )
            if int(round_communication["data_upload_bits"]) != int(
                communication_plan["planned_data_upload_bits_per_round"]
            ):
                raise AssertionError(
                    "realized data upload does not match the communication plan"
                )
            if int(round_communication["credit_payload_bits"]) != int(
                communication_plan["planned_credit_payload_bits_per_round"]
            ):
                raise AssertionError(
                    "realized credit upload does not match the communication plan"
                )
            if int(round_communication["structured_credit_payload_bits"]) != int(
                communication_plan["planned_structured_credit_payload_bits_per_round"]
            ):
                raise AssertionError(
                    "realized structured-credit upload does not match the communication plan"
                )
        round_payload_bits = int(round_communication["total_upload_bits"])
        uploaded_bits += round_payload_bits
        if downlink_compression:
            # Downlink sends sparse payload + dense affine/buffer in full, per candidate.
            dense_state_bytes = sum(
                global_state[key].numel() * global_state[key].element_size()
                for key, _, _ in (*dense_affine_layout, *dense_buffer_layout)
            )
            if architecture_block_method and round_downlink_block_stats is not None:
                sparse_downlink_bytes = math.ceil(
                    int(round_downlink_block_stats["payload_bits"]) / 8
                )
            else:
                sparse_downlink_bytes = math.ceil(
                    downlink_selected_coords * (value_bits + index_bits) / 8
                )
            downloaded_bytes += candidates_per_round * (
                sparse_downlink_bytes + dense_state_bytes
            )
        else:
            downloaded_bytes += candidates_per_round * model_bytes
        completed_jobs += candidates_per_round
        cache_metrics = _cache_metrics(method, cache_versions, round_index)
        layer_diag_summary = {}
        if round_layer_diagnostics:
            all_keys = next(iter(round_layer_diagnostics.values())).keys()
            for key in all_keys:
                selected = sum(
                    round_layer_diagnostics[c][key]["selected"] for c in round_layer_diagnostics
                )
                update_l2 = sum(
                    round_layer_diagnostics[c][key]["update_l2"] ** 2
                    for c in round_layer_diagnostics
                ) ** 0.5
                residual_l2 = sum(
                    round_layer_diagnostics[c][key]["residual_l2"] ** 2
                    for c in round_layer_diagnostics
                ) ** 0.5
                zero_energy = sum(
                    1 for c in round_layer_diagnostics if round_layer_diagnostics[c][key]["zero_energy"]
                )
                layer_diag_summary[key] = {
                    "selected": int(selected),
                    "update_l2": float(update_l2),
                    "residual_l2": float(residual_l2),
                    "residual_update_ratio": float(residual_l2 / update_l2) if update_l2 > 0 else 0.0,
                    "zero_energy_clients": int(zero_energy),
                }
        record = {
            "round": round_index,
            "candidates": candidates,
            "aggregate_clients": len(aggregate_clients),
            **cache_metrics,
            "staleness_aware_weighting": method not in CURRENT_ROUND_TOPK_METHODS,
            "error_feedback": method not in NO_ERROR_FEEDBACK_METHODS and method != "dense_saw_snn",
            "credit_scores_applicable": uses_credit_scores,
            "credit_mode": credit_mode,
            "credit_raw_scores": raw_scores,
            "credit_scores": scores,
            "client_upload_coordinates": (
                {
                    client_id: int(
                        round_uplink_block_stats_by_client[client_id][
                            "selected_coordinates"
                        ]
                    )
                    for client_id in candidates
                }
                if architecture_block_method
                else budgets
            ),
            "client_sparse_bit_caps": (
                {client_id: int(budgets[client_id]) for client_id in candidates}
                if block_local_int8
                else None
            ),
            "client_uplink_codec_stats": (
                {
                    client_id: {
                        key: value
                        for key, value in round_uplink_block_stats_by_client[
                            client_id
                        ].items()
                        if key != "selected_block_ids"
                    }
                    for client_id in candidates
                }
                if block_local_int8
                else None
            ),
            "mean_upload_density": (
                (
                    sum(int(s["selected_coordinates"]) for s in round_uplink_block_stats)
                    / (candidates_per_round * sparse_dimension)
                )
                if architecture_block_method and round_uplink_block_stats
                else (
                    sum(budgets.values())
                    / (candidates_per_round * sparse_dimension)
                )
            ),
            "uplink_encoding": (
                "block_local_bitmap_or_ids_int8_per_layer_scale"
                if block_local_int8
                else ("block_id_per_layer" if block_dual else "global_coordinate")
            ),
            "downlink_encoding": (
                "block_local_bitmap_or_ids_int8_per_layer_scale"
                if block_local_int8
                else (
                    "block_id_per_layer"
                    if block_dual
                    else ("global_coordinate" if downlink_compression else None)
                )
            ),
            "dense_bntt_affine_coordinates": (
                candidates_per_round * dense_affine_dimension
            ),
            "dense_bntt_buffer_coordinates": (
                candidates_per_round * dense_buffer_dimension
            ),
            "requested_dense_upload_equivalents": requested_dense_equivalents,
            "total_wire_budget_ratio": block_local_total_budget_ratio,
            "total_bit_cap_per_client": (
                communication_plan.get("total_bit_cap_per_client")
                if block_local_int8
                else None
            ),
            "sparse_bit_cap_per_client": block_local_sparse_bit_cap,
            "downlink_total_bit_cap_per_client": (
                communication_plan.get("downlink_total_bit_cap_per_client")
                if block_local_int8
                else None
            ),
            "downlink_sparse_bit_cap_per_client": block_local_downlink_sparse_bit_cap,
            **round_communication,
            "train_loss": sum(losses.values()) / len(losses),
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "training_forward_route": str(getattr(local_forward, "last_route", None)),
            "training_lif_route": str(getattr(local_model, "last_lif_route", None)),
            "training_lif_backends": [
                getattr(route, "backend", None)
                for route in getattr(local_model, "last_lif_routes", ())
            ],
            "eval_forward_route": str(getattr(eval_forward, "last_route", None)),
            "eval_lif_backends": [
                getattr(route, "backend", None)
                for route in getattr(model, "last_lif_routes", ())
            ],
            "completed_client_jobs": completed_jobs,
            "cumulative_upload_bytes": math.ceil(uploaded_bits / 8),
            "cumulative_download_bytes": downloaded_bytes,
            "ranking_score": ranking_score,
            "downlink_compression": downlink_compression,
            "downlink_selected_coords": downlink_selected_coords,
            "downlink_residual_l2": downlink_residual_l2,
            "downlink_layer_weight_mean": downlink_layer_weight_mean,
            "downlink_unmapped_frac": downlink_unmapped_frac,
            "downlink_credit_ema_decay": (
                downlink_credit_ema_decay if neuron_method else None
            ),
            "downlink_credit_fn": downlink_credit_fn if neuron_method else None,
            "downlink_credit_ema_delta_l2": downlink_credit_ema_delta_l2,
            "downlink_credit_ema_rate_l2": downlink_credit_ema_rate_l2,
            "downlink_mean_pre_catchup_gap_l2": (
                float(sum(client_gap_l2_values) / max(len(client_gap_l2_values), 1))
                if client_gap_l2_values
                else None
            ),
            "downlink_support_share": (
                dual_channel_support_share
                if method == "dual_channel_quota_dual_topk_snn"
                else None
            ),
            "periodic_full_downlink_refresh": False,
            "layer_diagnostics": layer_diag_summary,
            "compression_seconds": compression_time,
            "seconds": time.time() - started,
        }
        _append_jsonl(metrics_path, record)
        if (round_index + 1) % checkpoint_every == 0 or round_index + 1 == rounds:
            checkpoint_state = {
                "round": round_index,
                "model": _move_tensor_tree(global_state, "cpu"),
                "credit_ema": credit_ema,
                "completed_jobs": completed_jobs,
                "uploaded_bits": uploaded_bits,
                "downloaded_bytes": downloaded_bytes,
                "run_signature": run_signature,
            }
            if method not in CURRENT_ROUND_TOPK_METHODS:
                checkpoint_state.update(
                    {
                        "cache_payloads": _move_tensor_tree(cache_payloads, "cpu"),
                        "cache_dense_affine_payloads": _move_tensor_tree(
                            cache_dense_affine_payloads, "cpu"
                        ),
                        "cache_dense_buffer_payloads": _move_tensor_tree(
                            cache_dense_buffer_payloads, "cpu"
                        ),
                        "cache_versions": cache_versions,
                    }
                )
            if method not in NO_ERROR_FEEDBACK_METHODS and method != "dense_saw_snn":
                checkpoint_state["residuals"] = _move_tensor_tree(residuals, "cpu")
            if downlink_compression:
                if per_client_gap_downlink:
                    checkpoint_state["client_base_states"] = {
                        client_id: _move_tensor_tree(base, "cpu")
                        for client_id, base in client_base_states.items()
                    }
                    if method == "dual_channel_quota_dual_topk_snn":
                        checkpoint_state["previous_uplink_support_mask"] = (
                            _move_tensor_tree(previous_uplink_support_mask, "cpu")
                            if previous_uplink_support_mask is not None
                            else None
                        )
                else:
                    checkpoint_state["downlink_base_state"] = _move_tensor_tree(
                        downlink_base_state, "cpu"
                    )
                    checkpoint_state["downlink_residual"] = (
                        _move_tensor_tree(downlink_residual, "cpu")
                        if downlink_residual is not None
                        else None
                    )
                    checkpoint_state["downlink_prev_global_flat"] = (
                        _move_tensor_tree(downlink_prev_global_flat, "cpu")
                        if downlink_prev_global_flat is not None
                        else None
                    )
                if neuron_method:
                    checkpoint_state["downlink_ema_rates"] = (
                        tuple(
                            _move_tensor_tree(rate, "cpu")
                            for rate in downlink_ema_rates
                        )
                        if downlink_ema_rates is not None
                        else None
                    )
            _atomic_torch_save(
                torch,
                checkpoint_state,
                checkpoint_path,
            )
            del checkpoint_state
        print(json.dumps(record, sort_keys=True), flush=True)
        del (
            aggregate,
            aggregate_dense_affine,
            aggregate_dense_buffer,
            base_state,
            current_payloads,
            current_dense_affine,
            current_dense_buffer,
            sparse_updates,
            dense_affine_updates,
            dense_buffer_updates,
            losses,
            scores,
        )
        if (round_index + 1) % memory_cleanup_interval == 0:
            _release_cpu_memory()
    return output


def run_topk_saw(
    config: dict,
    data_root: str,
    device_name: str,
    resume: bool = False,
    smoke: bool = False,
) -> Path:
    return _run_topk(
        config,
        data_root,
        device_name,
        resume,
        smoke,
        allowed_methods=SAW_METHODS,
        trainer_name="Top-k SAW trainer",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Communication-matched Top-k SAW trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rounds", type=int, help="override training.rounds for diagnostics")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.rounds is not None:
        if args.rounds <= 0:
            parser.error("--rounds must be positive")
        config["training"]["rounds"] = args.rounds
    output = run_topk_saw(config, args.data_root, args.device, args.resume, args.smoke)
    print(f"output={output}")


if __name__ == "__main__":
    main()
