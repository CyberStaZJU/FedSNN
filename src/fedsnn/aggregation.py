from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TypeVar

T = TypeVar("T")


def normalized_weights(sample_counts: Sequence[int]) -> list[float]:
    total = sum(sample_counts)
    if total <= 0 or any(x < 0 for x in sample_counts):
        raise ValueError("sample counts must be non-negative with positive sum")
    return [x / total for x in sample_counts]


def weighted_average(states: Sequence[Mapping[str, T]], weights: Sequence[float]) -> dict[str, T]:
    if not states or len(states) != len(weights):
        raise ValueError("states and weights must be non-empty and have equal length")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("all states must have identical keys")
    scale = sum(weights)
    return {
        key: sum((state[key] * (weight / scale) for state, weight in zip(states, weights)))
        for key in states[0]
    }


def afedsnn_mixing_weight(
    sample_fraction: float,
    information_age: float,
    spike_rate: float,
    spike_mean: float,
    spike_std: float,
    kappa: float = 1.0,
) -> float:
    """AFedSNN equations 16, 18, and 19 from the supplied paper.

    The paper's printed Gaussian prefactor is ambiguous (`sqrt(2*pi*sigma)`
    versus the conventional `sqrt(2*pi)*sigma`). We preserve the printed
    expression here and expose it in the resolved run metadata.
    """
    if not 0 <= sample_fraction <= 1:
        raise ValueError("sample_fraction must be in [0, 1]")
    if information_age < 0 or spike_std <= 0 or kappa < 0:
        raise ValueError("invalid AFedSNN weight parameters")
    age_weight = 1.0 / (1.0 + math.exp(information_age))
    spike_weight = math.exp(-((spike_rate - spike_mean) ** 2) / (2.0 * spike_std**2))
    spike_weight /= math.sqrt(2.0 * math.pi * spike_std)
    return kappa * age_weight * sample_fraction * spike_weight


def sfedca_select(scores: Mapping[int, float], selected_clients: int) -> list[int]:
    """SFedCA algorithm 1, step 11: select largest firing-rate differences."""
    if not 0 < selected_clients <= len(scores):
        raise ValueError("invalid number of selected clients")
    return [client for client, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:selected_clients]]


def harmonic_cache_weights(versions: Mapping[int, int], current_version: int) -> dict[int, float]:
    """Normalize SAW's inverse-age weights over the valid update cache."""
    if not versions:
        raise ValueError("at least one cached update is required")
    masses = {}
    for client_id, source_version in versions.items():
        age = current_version + 1 - int(source_version)
        if age <= 0:
            raise ValueError("cached update versions cannot be in the future")
        masses[int(client_id)] = 1.0 / age
    total = sum(masses.values())
    return {client_id: mass / total for client_id, mass in masses.items()}


def subtract_torch_state_dicts(local_state, base_state):
    """Return floating-point model differences while ignoring bookkeeping buffers."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("PyTorch is required for state-dict differences") from exc
    if set(local_state) != set(base_state):
        raise ValueError("local and base states must have identical keys")
    return {
        key: local_state[key] - base_state[key]
        for key in local_state
        if torch.is_floating_point(local_state[key]) or torch.is_complex(local_state[key])
    }


def apply_torch_updates(global_state, updates, weights, step_size: float = 1.0):
    """Apply a normalized weighted average of model differences to a state dict."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("PyTorch is required for state-dict updates") from exc
    if not updates or len(updates) != len(weights):
        raise ValueError("updates and weights must be non-empty and have equal length")
    scale = float(sum(weights))
    if scale <= 0 or any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative with positive sum")
    floating_keys = {
        key for key, value in global_state.items() if torch.is_floating_point(value) or torch.is_complex(value)
    }
    if any(set(update) != floating_keys for update in updates):
        raise ValueError("every update must contain all floating-point state keys")
    result = {key: value.clone() for key, value in global_state.items()}
    for key in floating_keys:
        for update, weight in zip(updates, weights):
            result[key].add_(update[key], alpha=step_size * float(weight) / scale)
    return result


def allocate_topk_budget(
    scores: Mapping[int, float],
    total_coordinates: int,
    credit_strength: float = 0.75,
    credit_transform: str = "sqrt",
    max_coordinates_per_client: int | None = None,
) -> dict[int, int]:
    """Allocate a fixed Top-k budget while retaining an equal-share floor."""
    if (
        not scores
        or total_coordinates < len(scores)
        or not 0 <= credit_strength <= 1
        or credit_transform not in {"raw", "sqrt"}
        or (
            max_coordinates_per_client is not None
            and (
                isinstance(max_coordinates_per_client, bool)
                or not isinstance(max_coordinates_per_client, int)
                or max_coordinates_per_client < 1
                or total_coordinates > len(scores) * max_coordinates_per_client
            )
        )
    ):
        raise ValueError("invalid Top-k allocation inputs")
    clients = sorted(scores)
    nonnegative = {client: max(float(scores[client]), 0.0) for client in clients}
    transformed = (
        {client: math.sqrt(nonnegative[client]) for client in clients}
        if credit_transform == "sqrt"
        else nonnegative
    )
    mass = sum(transformed.values())
    if mass == 0:
        proportions = {client: 1.0 / len(clients) for client in clients}
    else:
        proportions = {
            client: (1.0 - credit_strength) / len(clients)
            + credit_strength * transformed[client] / mass
            for client in clients
        }
    allocation = {client: 1 for client in clients}
    capacity = {
        client: (max_coordinates_per_client or total_coordinates) - 1
        for client in clients
    }
    remainder = total_coordinates - len(clients)
    while remainder:
        active = [client for client in clients if capacity[client] > 0]
        if not active:
            raise AssertionError("Top-k per-client caps cannot accommodate the budget")
        active_mass = sum(proportions[client] for client in active)
        raw = (
            {client: remainder / len(active) for client in active}
            if active_mass == 0
            else {
                client: remainder * proportions[client] / active_mass
                for client in active
            }
        )
        grants = {
            client: min(capacity[client], int(math.floor(raw[client])))
            for client in active
        }
        granted = sum(grants.values())
        if granted == 0:
            order = sorted(
                active,
                key=lambda client: (-(raw[client] - math.floor(raw[client])), client),
            )
            grants[order[0]] = 1
            granted = 1
        for client, grant in grants.items():
            allocation[client] += grant
            capacity[client] -= grant
        remainder -= granted
    if sum(allocation.values()) != total_coordinates:
        raise AssertionError("Top-k allocation did not conserve its budget")
    return allocation


def equal_topk_budget(client_ids: Sequence[int], total_coordinates: int) -> dict[int, int]:
    if not client_ids or total_coordinates < len(client_ids):
        raise ValueError("invalid equal Top-k allocation inputs")
    clients = sorted(int(client_id) for client_id in client_ids)
    quotient, remainder = divmod(total_coordinates, len(clients))
    return {client: quotient + int(index < remainder) for index, client in enumerate(clients)}


def average_torch_state_dicts(states, weights):
    """Average model state dictionaries safely, including integer BN buffers."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("PyTorch is required for state-dict aggregation") from exc
    if not states or len(states) != len(weights):
        raise ValueError("states and weights must be non-empty and have equal length")
    scale = float(sum(weights))
    if scale <= 0 or any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative with positive sum")
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("all states must have identical keys")
    result = {}
    for key in states[0]:
        first = states[0][key]
        if torch.is_floating_point(first) or torch.is_complex(first):
            value = torch.zeros_like(first)
            for state, weight in zip(states, weights):
                value.add_(state[key], alpha=float(weight) / scale)
            result[key] = value
        else:
            # BatchNorm's num_batches_tracked is bookkeeping, not a parameter.
            result[key] = first.clone()
    return result


def mix_torch_state_dicts(global_state, local_state, mixing_weight: float):
    """AFedSNN equation 20 with integer buffers kept from the global model."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for state-dict mixing") from exc
    if not 0 <= mixing_weight <= 1:
        raise ValueError("mixing_weight must be in [0, 1]")
    if set(global_state) != set(local_state):
        raise ValueError("state dictionaries must have identical keys")
    result = {}
    for key, global_value in global_state.items():
        local_value = local_state[key]
        if torch.is_floating_point(global_value) or torch.is_complex(global_value):
            result[key] = global_value * (1.0 - mixing_weight) + local_value * mixing_weight
        else:
            result[key] = global_value.clone()
    return result
