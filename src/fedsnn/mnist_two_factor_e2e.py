"""Exploratory async MNIST probe for eligibility-informed staleness.

Supports two registered tracks on the same harder async protocol (v2):

1. ``idea_c_mnist_two_factor_e2e_v2`` — three methods, two-factor only.
2. ``idea_c_mnist_four_method_e2e_v2`` — four methods including third-factor
   (RMS × |local-error|), same N/R/α/delays as (1).

Independent of formal Stage-1A / SHD archives. Descriptive ACC only; no
mechanical IDEA_C_CONTINUE.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .break_gate_e2e import (
    _apply_flat,
    _dataloader_kwargs,
    _flatten,
    _layout,
    _loader,
    _state_delta,
    _unflatten,
    descriptive_e2e_summary,
    seed_rows_from_metrics,
)
from .config import load_config, result_dir
from .data import load_mnist_unit_interval
from .device import activate_device, resolve_device, seed_everything
from .idea_c_stage1a import (
    CostLedger,
    CorrectionCoefficients,
    IDEA_C_METHOD,
    QuantizedNeuronSummary,
    correction_for_method,
    dequantize_rms_u8,
    expand_postsynaptic,
    protocol_identity,
    quantize_rms_u8,
)
from .models import build_mnist_2conv2fc_bntt
from .partition import dirichlet_partition
from .protocol import clone_state
from .third_factor_oracle import THIRD_FACTOR_METHOD
from .train import _append_jsonl, _code_commit

# v2 harder async protocol shared by two-factor and four-method tracks.
# v1 (N=5/R=50/α=0.5/delays≤3) remains historical under idea_c_mnist_two_factor_e2e_v1.
TRACK_TWO_FACTOR = "idea_c_mnist_two_factor_e2e_v2"
TRACK_FOUR_METHOD = "idea_c_mnist_four_method_e2e_v2"
# Default export names stay on the historical two-factor track identity.
REGISTERED_TRACK = TRACK_TWO_FACTOR
REGISTERED_FIDELITY = TRACK_TWO_FACTOR
REGISTERED_SEEDS = (2, 3, 4)
REGISTERED_CLIENTS = 10
REGISTERED_ROUNDS = 100
REGISTERED_ALPHA = 0.1
# Harder than v1 (1/1/2/3/3): more slow/very_slow clients and larger delay_updates.
DELAY_CLASSES = (
    "fast",
    "fast",
    "medium",
    "medium",
    "slow",
    "slow",
    "slow",
    "very_slow",
    "very_slow",
    "very_slow",
)
DELAYS = (2, 2, 4, 4, 6, 8, 8, 12, 14, 16)
# Length-20 schedule covering all 10 clients; biased slightly toward fast clients
# so slow arrivals still see substantial server drift.
SCHEDULE = (
    0,
    1,
    2,
    4,
    7,
    0,
    3,
    5,
    8,
    1,
    2,
    6,
    9,
    0,
    3,
    4,
    7,
    1,
    5,
    8,
)
TWO_FACTOR_METHODS = (
    "drift_age",
    "activity_drift_age",
    IDEA_C_METHOD,
)
FOUR_METHODS = (
    "drift_age",
    "activity_drift_age",
    IDEA_C_METHOD,
    THIRD_FACTOR_METHOD,
)
# Backward-compatible default for imports/tests that expect three-method COMPARE_METHODS.
COMPARE_METHODS = TWO_FACTOR_METHODS
TWO_FACTOR_METHOD = IDEA_C_METHOD
# Spiking-parameter layers that receive postsynaptic eligibility correction.
SPIKE_WEIGHT_LAYERS = ("conv1.weight", "conv2.weight", "fc1.weight")
DEFAULT_COEFFICIENTS = CorrectionCoefficients(
    age=0.12,
    drift=0.35,
    eligibility=-0.1,
    interaction=0.2,
)


@dataclass(frozen=True)
class MultiLayerEligibilitySummary:
    """Per-layer postsynaptic eligibility + matched activity payloads for BNTT."""

    layers: Mapping[str, tuple[QuantizedNeuronSummary, QuantizedNeuronSummary]]
    samples: int
    timesteps: int

    @property
    def payload_bits(self) -> int:
        return sum(elig.payload_bits for elig, _act in self.layers.values())

    @property
    def activity_payload_bits(self) -> int:
        return sum(act.payload_bits for _elig, act in self.layers.values())

    def eligibility_for(self, layer: str) -> QuantizedNeuronSummary:
        if layer not in self.layers:
            raise KeyError(layer)
        return self.layers[layer][0]

    def activity_for(self, layer: str) -> QuantizedNeuronSummary:
        if layer not in self.layers:
            raise KeyError(layer)
        return self.layers[layer][1]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _track_profile(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve which MNIST exploratory track this config belongs to."""
    fidelity = str(config["paper"].get("fidelity", ""))
    methods = tuple(config.get("compare_methods") or ())
    paper_method = str(config["paper"].get("method", ""))
    notes = config.get("notes") or {}
    if fidelity == TRACK_FOUR_METHOD or paper_method == "mnist_four_method_e2e":
        return {
            "track": TRACK_FOUR_METHOD,
            "fidelity": TRACK_FOUR_METHOD,
            "paper_method": "mnist_four_method_e2e",
            "methods": FOUR_METHODS,
            "includes_third_factor": True,
            "exploratory_note": "mnist_four_method_exploratory",
            "third_factor_out_of_scope": False,
        }
    if fidelity == TRACK_TWO_FACTOR or paper_method == "mnist_two_factor_three_method_e2e":
        return {
            "track": TRACK_TWO_FACTOR,
            "fidelity": TRACK_TWO_FACTOR,
            "paper_method": "mnist_two_factor_three_method_e2e",
            "methods": TWO_FACTOR_METHODS,
            "includes_third_factor": False,
            "exploratory_note": "mnist_two_factor_exploratory",
            "third_factor_out_of_scope": True,
        }
    # Fall back for tests that only set notes + fidelity-like fields.
    if notes.get("mnist_four_method_exploratory") is True or methods == FOUR_METHODS:
        return {
            "track": TRACK_FOUR_METHOD,
            "fidelity": TRACK_FOUR_METHOD,
            "paper_method": "mnist_four_method_e2e",
            "methods": FOUR_METHODS,
            "includes_third_factor": True,
            "exploratory_note": "mnist_four_method_exploratory",
            "third_factor_out_of_scope": False,
        }
    return {
        "track": TRACK_TWO_FACTOR,
        "fidelity": TRACK_TWO_FACTOR,
        "paper_method": "mnist_two_factor_three_method_e2e",
        "methods": TWO_FACTOR_METHODS,
        "includes_third_factor": False,
        "exploratory_note": "mnist_two_factor_exploratory",
        "third_factor_out_of_scope": True,
    }


def _validate(config: Mapping[str, Any]) -> None:
    notes = config.get("notes", {})
    profile = _track_profile(config)
    _require(
        notes.get(profile["exploratory_note"]) is True
        or notes.get("mnist_two_factor_exploratory") is True
        or notes.get("mnist_four_method_exploratory") is True,
        f"requires notes.{profile['exploratory_note']} (or shared exploratory flag)",
    )
    _require(notes.get("formal_stage1a_immutable") is True, "must freeze formal Stage-1A as immutable")
    _require(notes.get("oracle_gate_bypassed") is True, "must explicitly declare oracle gate bypass")
    _require(notes.get("mechanical_e2e_gate_disabled") is True, "mechanical e2e gate must be disabled")
    _require(notes.get("rerun_authorized_stage1a") is not True, "must not authorize formal Stage-1A rerun")
    if profile["includes_third_factor"]:
        _require(
            notes.get("third_factor_out_of_scope") is not True,
            "four-method track must not declare third_factor_out_of_scope",
        )
        _require(
            notes.get("third_factor_exploratory") is True,
            "four-method track requires notes.third_factor_exploratory",
        )
    else:
        _require(notes.get("third_factor_out_of_scope") is True, "third-factor must be declared out of scope")
    _require(
        str(config["paper"].get("fidelity", "")) == profile["fidelity"],
        f"fidelity mismatch (expected {profile['fidelity']})",
    )
    _require(
        str(config["paper"]["method"]) == profile["paper_method"],
        f"wrong paper.method (expected {profile['paper_method']})",
    )
    _require(str(config["stage"]["mode"]) == "end_to_end", "stage.mode must be end_to_end")
    _require(str(config["dataset"]["name"]).lower() == "mnist", "dataset must be mnist")
    _require(
        int(config["federation"]["clients"]) == REGISTERED_CLIENTS,
        f"requires exactly {REGISTERED_CLIENTS} clients",
    )
    _require(
        tuple(config["federation"]["delay_classes"]) == DELAY_CLASSES,
        "delay classes must remain preregistered harder table",
    )
    delay_updates = tuple(int(x) for x in config["federation"]["delay_updates"])
    _require(delay_updates == DELAYS, "delay_updates must match harder DELAYS table")
    _require(len(delay_updates) == REGISTERED_CLIENTS, "delay_updates length must equal clients")
    _require(
        abs(float(config["dataset"]["alpha"]) - REGISTERED_ALPHA) < 1e-12,
        f"dataset.alpha must be {REGISTERED_ALPHA}",
    )
    _require(
        int(config["training"]["rounds"]) == REGISTERED_ROUNDS,
        f"training.rounds must be {REGISTERED_ROUNDS}",
    )
    _require(str(config["model"]["name"]) == "mnist_2conv2fc_bntt", "model must be mnist_2conv2fc_bntt")
    methods = tuple(config.get("compare_methods") or profile["methods"])
    _require(methods == profile["methods"], f"compare_methods must be {profile['methods']}")
    if not profile["includes_third_factor"] and "third_factor" in json.dumps(methods):
        raise ValueError("third-factor method is out of scope for the two-factor MNIST track")
    _require(
        set(SCHEDULE) == set(range(REGISTERED_CLIENTS)),
        "SCHEDULE must cover every client id",
    )
    _require(len(DELAYS) == REGISTERED_CLIENTS, "DELAYS length must equal clients")


def load_mnist_two_factor_coefficients(config: Mapping[str, Any]) -> dict[str, CorrectionCoefficients]:
    gate = config.get("gate") or {}
    if gate.get("two_factor_calibration_manifest_path"):
        raise ValueError("MNIST probe uses grid-centre defaults; no SHD calibration manifests")
    if gate.get("third_factor_calibration_manifest_path"):
        raise ValueError("MNIST probe uses grid-centre defaults; no SHD calibration manifests")
    DEFAULT_COEFFICIENTS.validate()
    methods = tuple(config.get("compare_methods") or _track_profile(config)["methods"])
    return {method: DEFAULT_COEFFICIENTS for method in methods}


def _single_block_eligibility_rms(
    presynaptic: Any,
    postsynaptic: Any,
    *,
    trace_decay: float,
) -> Any:
    """Two-factor RMS → one value per postsynaptic unit.

    ``presynaptic``: [B, T, P], ``postsynaptic``: [B, T, C].
    """
    import torch

    pre = torch.as_tensor(presynaptic)
    post = torch.as_tensor(postsynaptic)
    if pre.ndim != 3 or post.ndim != 3:
        raise ValueError("eligibility factors must be rank-three")
    if pre.shape[:2] != post.shape[:2]:
        raise ValueError("batch/time mismatch between pre and post")
    if not 0.0 <= float(trace_decay) < 1.0:
        raise ValueError("trace_decay must be in [0,1)")
    pre_trace = torch.zeros_like(pre[:, 0])
    accum = torch.zeros(post.shape[-1], dtype=torch.float64, device=post.device)
    for step in range(pre.shape[1]):
        pre_trace = float(trace_decay) * pre_trace + pre[:, step]
        factor = post[:, step]
        # [B,C,P] squared mean over B,P → [C]
        accum = accum + (
            factor.unsqueeze(-1) * pre_trace.unsqueeze(1)
        ).to(torch.float64).square().mean(dim=(0, 2))
    return (accum / pre.shape[1]).sqrt().to(post.dtype)


def capture_mnist_bntt_two_factor_eligibility(
    model: Any,
    images: Any,
    *,
    labels: Any | None = None,
    third_factor: bool = False,
    trace_decay: float = 0.9,
    surrogate_beta: float | None = None,
) -> MultiLayerEligibilitySummary:
    """Capture channel/neuron eligibility for MNIST 2Conv2FC+BNTT.

    Replays the public forward LIF path without mutating training state.
    Eligibility is defined only on spiking layers ``conv1``, ``conv2``, ``fc1``
    (matching ``structured_spike_parameter_map``). The non-spiking readout
    ``fc2`` is age-only at correction time.

    When ``third_factor=True``, each layer's postsynaptic surrogate is scaled by
    a local-error third factor before the RMS reduction:

    - ``fc1``: ``|(one_hot - softmax) @ fc2.weight|`` (shape [B, 128])
    - ``conv2`` / ``conv1``: mean absolute readout error projected through
      ``fc2`` then back-propagated as a scalar per sample, broadcast onto the
      channel post factor (keeps nonnegative RMS codec; matches SHD form of
      absolute local-error modulation without inventing full spatial BP).

    Activity payloads remain unmodulated spike rates for capacity parity.
    """
    import math as _math
    import torch
    from torch import nn

    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("MNIST eligibility expects [B,1,28,28] images")
    if third_factor and labels is None:
        raise ValueError("third_factor capture requires labels")
    beta = float(
        surrogate_beta
        if surrogate_beta is not None
        else getattr(model, "surrogate_beta", 2.0)
    )
    threshold = float(model.threshold)
    decay = float(model.membrane_decay)
    timesteps = int(model.timesteps)
    batch = int(images.shape[0])

    def _apply_bntt(module, inputs):
        if module.training and inputs.ndim == 2 and inputs.shape[0] == 1:
            return nn.functional.batch_norm(
                inputs,
                module.running_mean,
                module.running_var,
                module.weight,
                module.bias,
                training=False,
                momentum=module.momentum,
                eps=module.eps,
            )
        return module(inputs)

    def _atan_post(centered):
        return (beta / 2.0) / (1.0 + (_math.pi * beta * centered / 2.0).square())

    membranes = [
        images.new_zeros((batch, 32, 28, 28)),
        images.new_zeros((batch, 64, 14, 14)),
        images.new_zeros((batch, 128)),
    ]
    pre_steps = {"conv1.weight": [], "conv2.weight": [], "fc1.weight": []}
    post_steps = {"conv1.weight": [], "conv2.weight": [], "fc1.weight": []}
    spike_steps = {"conv1.weight": [], "conv2.weight": [], "fc1.weight": []}

    was_training = bool(model.training)
    model.eval()
    third_fc1 = None  # [B, 128]
    third_scalar = None  # [B, 1] broadcast onto conv channel posts
    try:
        with torch.no_grad():
            if third_factor:
                # Separate forward for logits used only as third-factor modulator.
                # Poisson noise differs from the eligibility replay below; this
                # matches the SHD feedforward pattern (modulator from one pass).
                logits = model(images)
                labels_t = torch.as_tensor(labels, device=images.device, dtype=torch.long)
                if labels_t.ndim != 1 or int(labels_t.shape[0]) != batch:
                    raise ValueError("labels must be a rank-one class index per sample")
                probabilities = torch.softmax(logits.detach(), dim=-1)
                targets = nn.functional.one_hot(
                    labels_t, num_classes=int(logits.shape[-1])
                ).to(probabilities.dtype)
                # fc1 postsynaptic third factor: same form as SHD hidden.
                third_fc1 = ((targets - probabilities) @ model.fc2.weight.detach()).abs()
                # Conv layers: scalar per-sample |error| mean for channel modulation.
                third_scalar = third_fc1.mean(dim=1, keepdim=True)

            for timestep in range(timesteps):
                # Poisson encode (same as model.forward).
                spikes_in = (torch.rand_like(images) <= images).to(images.dtype)
                pre_steps["conv1.weight"].append(spikes_in.flatten(1))  # [B,784]

                current = _apply_bntt(model.bntt1[timestep], model.conv1(spikes_in))
                charged = decay * membranes[0] + current
                centered = charged - threshold
                spikes = (centered >= 0).to(images.dtype)
                membranes[0] = charged - spikes * threshold
                # Channel postsynaptic factor / activity: mean over spatial dims.
                post_c1 = _atan_post(centered).flatten(2).mean(2)
                if third_scalar is not None:
                    post_c1 = post_c1 * third_scalar
                post_steps["conv1.weight"].append(post_c1)
                spike_steps["conv1.weight"].append(spikes.flatten(2).mean(2))

                pooled = nn.functional.avg_pool2d(spikes, 2)
                pre_steps["conv2.weight"].append(pooled.flatten(1))

                current = _apply_bntt(model.bntt2[timestep], model.conv2(pooled))
                charged = decay * membranes[1] + current
                centered = charged - threshold
                spikes = (centered >= 0).to(images.dtype)
                membranes[1] = charged - spikes * threshold
                post_c2 = _atan_post(centered).flatten(2).mean(2)
                if third_scalar is not None:
                    post_c2 = post_c2 * third_scalar
                post_steps["conv2.weight"].append(post_c2)
                spike_steps["conv2.weight"].append(spikes.flatten(2).mean(2))

                flat = nn.functional.avg_pool2d(spikes, 2).flatten(1)
                pre_steps["fc1.weight"].append(flat)

                current = _apply_bntt(model.bntt_fc1[timestep], model.fc1(flat))
                charged = decay * membranes[2] + current
                centered = charged - threshold
                spikes = (centered >= 0).to(images.dtype)
                membranes[2] = charged - spikes * threshold
                post_fc1 = _atan_post(centered)
                if third_fc1 is not None:
                    post_fc1 = post_fc1 * third_fc1
                post_steps["fc1.weight"].append(post_fc1)
                spike_steps["fc1.weight"].append(spikes)
    finally:
        if was_training:
            model.train()

    layers: dict[str, tuple[QuantizedNeuronSummary, QuantizedNeuronSummary]] = {}
    for layer in SPIKE_WEIGHT_LAYERS:
        pre_bt = torch.stack(pre_steps[layer], dim=1)
        post_bt = torch.stack(post_steps[layer], dim=1)
        spike_bt = torch.stack(spike_steps[layer], dim=1)
        elig = _single_block_eligibility_rms(pre_bt, post_bt, trace_decay=trace_decay)
        activity = spike_bt.to(torch.float64).mean(dim=(0, 1))
        layers[layer] = (quantize_rms_u8(elig), quantize_rms_u8(activity))
    return MultiLayerEligibilitySummary(
        layers=layers,
        samples=batch,
        timesteps=timesteps,
    )


def capture_mnist_bntt_third_factor_eligibility(
    model: Any,
    images: Any,
    labels: Any,
    *,
    trace_decay: float = 0.9,
    surrogate_beta: float | None = None,
) -> MultiLayerEligibilitySummary:
    """Third-factor modulated eligibility for MNIST BNTT (wrapper)."""
    return capture_mnist_bntt_two_factor_eligibility(
        model,
        images,
        labels=labels,
        third_factor=True,
        trace_decay=trace_decay,
        surrogate_beta=surrogate_beta,
    )


def _feature_tensors(
    layer: str,
    parameter: Any,
    drift: Any,
    summary: MultiLayerEligibilitySummary,
):

    drift = drift.abs()
    eligibility = dequantize_rms_u8(
        summary.eligibility_for(layer), device=parameter.device, dtype=parameter.dtype
    )
    activity = dequantize_rms_u8(
        summary.activity_for(layer), device=parameter.device, dtype=parameter.dtype
    )
    return drift, expand_postsynaptic(eligibility, parameter), expand_postsynaptic(activity, parameter)


def _eligibility_mode_for_method(method: str) -> str:
    if method == THIRD_FACTOR_METHOD:
        return "third_factor"
    # drift/activity/two-factor all capture two-factor summaries so activity
    # payload parity is preserved; drift_age ignores eligibility features.
    return "two_factor"


def _correction_method_name(method: str) -> str:
    """Map proposed method names onto correction_for_method's frozen set."""
    if method == THIRD_FACTOR_METHOD:
        return IDEA_C_METHOD
    if method in {"drift_age", "activity_drift_age", IDEA_C_METHOD}:
        return method
    raise ValueError(f"unsupported method: {method}")


def _buffer_keys(model: Any) -> tuple[str, ...]:
    """BNTT running stats and other non-parameter tensors in state_dict."""
    return tuple(name for name, _ in model.named_buffers())


def _merge_buffers(
    state: Mapping[str, Any],
    local: Mapping[str, Any],
    buffer_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Copy client BN buffers into the server state after a parameter update.

    Parameter tensors are corrected for staleness via the weight layout path;
    BNTT ``running_mean`` / ``running_var`` / ``num_batches_tracked`` are not
    in that layout. Leaving them at init zeros makes ``model.eval()`` collapse
    logits to a near-uniform prior (test_acc ≈ 0.098) even when local train
    loss falls. Async one-arrival therefore takes the arriving client's buffers
    wholesale (no age attenuation on stats).
    """
    import torch

    result = clone_state(dict(state), next(iter(state.values())).device)
    for name in buffer_keys:
        if name not in local:
            raise KeyError(f"local state missing buffer {name}")
        if name not in result:
            raise KeyError(f"server state missing buffer {name}")
        result[name] = torch.as_tensor(local[name]).detach().to(
            device=result[name].device, dtype=result[name].dtype
        ).clone()
    return result


def _evaluate(model, dataset, device, batch_size, config, max_batches=None):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **{k: v for k, v in _dataloader_kwargs(config).items() if k != "shuffle"},
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += float(criterion(logits, labels).item())
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.shape[0])
    if total == 0:
        raise RuntimeError("evaluation produced no samples")
    return correct / total, loss_sum / total


def _train_local(
    builder: Any,
    base_state: Mapping[str, Any],
    dataset: Any,
    indices: Any,
    device: Any,
    config: Mapping[str, Any],
    seed: int,
    *,
    eligibility_mode: str = "two_factor",
    max_batches: int | None = None,
) -> tuple[dict[str, Any], MultiLayerEligibilitySummary, float, int, int, dict[str, Any]]:
    import torch
    from torch import nn

    if eligibility_mode not in {"two_factor", "third_factor"}:
        raise ValueError(f"unknown eligibility_mode={eligibility_mode}")
    model = builder().to(device)
    model.load_state_dict(base_state)
    training = config["training"]
    loader, ordered_indices = _loader(dataset, indices, int(training["batch_size"]), seed, config)
    optimizer_name = str(training.get("optimizer", "sgd")).strip().lower()
    lr = float(training["learning_rate"])
    weight_decay = float(training.get("weight_decay", 0.0))
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(training.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"unsupported optimizer={optimizer_name}")
    criterion = nn.CrossEntropyLoss()
    loss_sum = None
    summary: MultiLayerEligibilitySummary | None = None
    steps = samples = 0
    layer_rate_sum = None
    for _epoch in range(int(training["local_epochs"])):
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if summary is None:
                summary = capture_mnist_bntt_two_factor_eligibility(
                    model,
                    images,
                    labels=labels if eligibility_mode == "third_factor" else None,
                    third_factor=eligibility_mode == "third_factor",
                    trace_decay=float(config["eligibility"]["trace_decay"]),
                    surrogate_beta=float(config["model"].get("surrogate_beta", 2.0)),
                )
            optimizer.zero_grad(set_to_none=True)
            logits, layer_rates = model(images, return_layer_activity=True)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                rates = layer_rates.mean(dim=0).detach().cpu()
                layer_rate_sum = rates if layer_rate_sum is None else layer_rate_sum + rates
            detached = loss.detach()
            loss_sum = detached if loss_sum is None else loss_sum + detached
            steps += 1
            samples += int(labels.shape[0])
    if steps == 0 or loss_sum is None or summary is None:
        raise RuntimeError("local training produced no batches/summary")
    mean_loss = float((loss_sum / steps).item())
    mean_layer_rates = (layer_rate_sum / steps).tolist() if layer_rate_sum is not None else []
    budget = {
        "seed": int(seed),
        "local_epochs": int(training["local_epochs"]),
        "max_batches": None if max_batches is None else int(max_batches),
        "steps": int(steps),
        "samples": int(samples),
        "ordered_indices_sha256": __import__("hashlib")
        .sha256(ordered_indices.tobytes())
        .hexdigest(),
        "eligibility_mode": eligibility_mode,
        "optimizer": optimizer_name,
        "layer_spike_rates_mean": mean_layer_rates,
        "hidden_spike_rate_mean": (
            float(sum(mean_layer_rates) / len(mean_layer_rates)) if mean_layer_rates else 0.0
        ),
    }
    return (
        clone_state(model.state_dict(), next(iter(base_state.values())).device),
        summary,
        mean_loss,
        steps,
        samples,
        budget,
    )


def _model_builder(config: Mapping[str, Any]):
    model = config["model"]
    return lambda: build_mnist_2conv2fc_bntt(
        timesteps=int(model["timesteps"]),
        classes=int(model.get("classes", 10)),
        surrogate_beta=float(model.get("surrogate_beta", 2.0)),
        threshold=float(model.get("threshold", 1.0)),
        membrane_decay=float(model.get("membrane_decay", 0.95)),
        track_runtime_activity=True,
    )


def _partitions(train_set: Any, config: Mapping[str, Any]) -> list[np.ndarray]:
    dataset = config["dataset"]
    if dataset["partition"] != "dirichlet":
        raise ValueError("MNIST two-factor e2e requires Dirichlet partition")
    labels = np.asarray(train_set.targets)
    return dirichlet_partition(
        labels,
        int(config["federation"]["clients"]),
        float(dataset["alpha"]),
        int(config["training"]["seed"]),
        min_samples=1,
    )


def _end_to_end(
    config: Mapping[str, Any],
    train_set: Any,
    test_set: Any,
    partitions: list[np.ndarray],
    builder: Any,
    device: Any,
    smoke: bool,
) -> list[dict[str, Any]]:
    profile = _track_profile(config)
    methods = tuple(config.get("compare_methods") or profile["methods"])
    selected_coefficients = load_mnist_two_factor_coefficients(config)
    seed = int(config["training"]["seed"])
    rounds = 2 if smoke else int(config["training"]["rounds"])
    records: list[dict[str, Any]] = []
    reference_model = builder().to(device)
    reference_state = clone_state(reference_model.state_dict(), device)
    for method in methods:
        model = builder().to(device)
        state = clone_state(reference_state, device)
        model.load_state_dict(state)
        layout = _layout(model)
        buffer_keys = _buffer_keys(model)
        if not any(key.startswith("bntt") and "running_" in key for key in buffer_keys):
            raise RuntimeError(
                "MNIST BNTT model has no bntt running_* buffers; eval would be invalid"
            )
        n_clients = REGISTERED_CLIENTS
        client_states = [clone_state(state, device) for _ in range(n_clients)]
        client_versions = [0] * n_clients
        ledger = CostLedger()
        eligibility_mode = _eligibility_mode_for_method(method)
        method_coefficients = selected_coefficients[method]
        for server_update in range(rounds):
            client_id = SCHEDULE[server_update % len(SCHEDULE)]
            base = client_states[client_id]
            local, summary, loss, steps, samples, audit = _train_local(
                builder,
                base,
                train_set,
                partitions[client_id],
                device,
                config,
                seed + server_update * 100 + client_id,
                eligibility_mode=eligibility_mode,
                max_batches=1 if smoke else None,
            )
            delta_by_layer = _unflatten(_state_delta(local, base, layout), layout)
            age = max(1, server_update - client_versions[client_id] + DELAYS[client_id])
            corrected = {}
            correction_min = 1.0
            correction_max = 0.0
            for layer, parameter in delta_by_layer.items():
                if layer in SPIKE_WEIGHT_LAYERS:
                    drift, eligibility, activity = _feature_tensors(
                        layer, parameter, state[layer] - base[layer], summary
                    )
                    correction = correction_for_method(
                        _correction_method_name(method),
                        age=age,
                        drift=drift,
                        eligibility=eligibility,
                        activity=activity,
                        layer_age_rate=method_coefficients.age,
                        coefficients=method_coefficients,
                    )
                    corrected[layer] = parameter * correction
                    correction_min = min(correction_min, float(correction.min()))
                    correction_max = max(correction_max, float(correction.max()))
                else:
                    scalar = math.exp(-method_coefficients.age * age)
                    corrected[layer] = parameter * scalar
                    correction_min = min(correction_min, scalar)
                    correction_max = max(correction_max, scalar)
            # Weights: staleness-corrected delta. BNTT buffers: full client copy.
            state = _merge_buffers(
                _apply_flat(state, _flatten(corrected, layout), layout),
                local,
                buffer_keys,
            )
            model.load_state_dict(state)
            accuracy, test_loss = _evaluate(
                model,
                test_set,
                device,
                int(config["training"]["batch_size"]),
                config,
                1 if smoke else None,
            )
            client_states[client_id] = clone_state(state, device)
            client_versions[client_id] = server_update + 1
            ledger.add_summary(summary)
            if summary.activity_payload_bits != summary.payload_bits:
                raise RuntimeError("activity and eligibility payloads are not matched")
            ledger.local_training_steps += steps
            ledger.eligibility_factor_multiply_adds += (
                samples
                * int(config["model"]["timesteps"])
                * int(sum(model.spike_channel_sizes))
            )
            records.append(
                {
                    "seed": seed,
                    "method": method,
                    "server_update": server_update,
                    "client_id": client_id,
                    "age": age,
                    "delay_class": config["federation"]["delay_classes"][client_id],
                    "train_loss": loss,
                    "test_loss": test_loss,
                    "test_accuracy": accuracy,
                    "eligibility_mode": eligibility_mode,
                    "architecture": "mnist_2conv2fc_bntt",
                    "hidden_spike_rate_mean": float(audit.get("hidden_spike_rate_mean", 0.0)),
                    "layer_spike_rates_mean": list(audit.get("layer_spike_rates_mean") or []),
                    "correction_min": correction_min,
                    "correction_max": correction_max,
                    "optimizer": str(config["training"].get("optimizer", "sgd")),
                    "coefficients": {
                        "age": method_coefficients.age,
                        "drift": method_coefficients.drift,
                        "eligibility": method_coefficients.eligibility,
                        "interaction": method_coefficients.interaction,
                    },
                    "cost": ledger.as_record(),
                }
            )
    return records


def run_mnist_two_factor_e2e(
    config: dict[str, Any],
    data_root: str | Path,
    device_spec: str,
    *,
    smoke: bool = False,
) -> Path:
    _validate(config)
    profile = _track_profile(config)
    methods = tuple(config.get("compare_methods") or profile["methods"])
    coefficients = load_mnist_two_factor_coefficients(config)
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    device_info = resolve_device(device_spec)
    device = activate_device(device_info)
    download = bool(config["dataset"].get("download", False))
    train_set = load_mnist_unit_interval(data_root, train=True, download=download)
    test_set = load_mnist_unit_interval(data_root, train=False, download=download)
    partitions = _partitions(train_set, config)
    builder = _model_builder(config)
    output = result_dir(config)
    if smoke:
        output = output.with_name(f"{output.name}__smoke")
    output.mkdir(parents=True, exist_ok=True)
    code_identity = _code_commit()
    resolved = copy.deepcopy(config)
    resolved["runtime"] = {
        "device": str(device),
        "smoke": smoke,
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "code_identity": code_identity,
        "protocol_identity": protocol_identity(config, code_identity),
        "track": profile["track"],
        "oracle_gate_bypassed": True,
        "mechanical_e2e_gate_disabled": True,
        "third_factor_out_of_scope": profile["third_factor_out_of_scope"],
        "includes_third_factor": profile["includes_third_factor"],
        "compare_methods": list(methods),
        "coefficient_source": "stage1a_grid_centre_default",
        "frozen_coefficients": {
            method: {
                "age": coeffs.age,
                "drift": coeffs.drift,
                "eligibility": coeffs.eligibility,
                "interaction": coeffs.interaction,
            }
            for method, coeffs in coefficients.items()
        },
    }
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    (output / "frozen_coefficients.json").write_text(
        json.dumps(resolved["runtime"]["frozen_coefficients"], indent=2) + "\n",
        encoding="utf-8",
    )
    records = _end_to_end(config, train_set, test_set, partitions, builder, device, smoke)
    path = output / ("metrics.smoke.jsonl" if smoke else "metrics.jsonl")
    if path.exists():
        path.unlink()
    for row in records:
        _append_jsonl(path, row)
    return output


def run_mnist_two_factor_e2e_file(
    config_path: str | Path,
    data_root: str | Path,
    device_spec: str = "auto",
    *,
    smoke: bool = False,
) -> Path:
    return run_mnist_two_factor_e2e(
        load_config(config_path), data_root, device_spec, smoke=smoke
    )


__all__ = [
    "COMPARE_METHODS",
    "DEFAULT_COEFFICIENTS",
    "DELAYS",
    "DELAY_CLASSES",
    "FOUR_METHODS",
    "REGISTERED_ALPHA",
    "REGISTERED_CLIENTS",
    "REGISTERED_FIDELITY",
    "REGISTERED_ROUNDS",
    "REGISTERED_SEEDS",
    "REGISTERED_TRACK",
    "SCHEDULE",
    "SPIKE_WEIGHT_LAYERS",
    "THIRD_FACTOR_METHOD",
    "TRACK_FOUR_METHOD",
    "TRACK_TWO_FACTOR",
    "TWO_FACTOR_METHOD",
    "TWO_FACTOR_METHODS",
    "capture_mnist_bntt_third_factor_eligibility",
    "capture_mnist_bntt_two_factor_eligibility",
    "descriptive_e2e_summary",
    "load_mnist_two_factor_coefficients",
    "run_mnist_two_factor_e2e",
    "run_mnist_two_factor_e2e_file",
    "seed_rows_from_metrics",
]
