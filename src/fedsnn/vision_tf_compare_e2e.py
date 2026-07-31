"""Cross-dataset exploratory compare: drift_age vs TF vs TF+long.

Datasets:
  - MNIST + mnist_2conv2fc_bntt
  - Fashion-MNIST + fashion_mnist_2conv2fc_bntt
  - CIFAR-10 + fedsnn_alexnet_bntt

Tracks (paper.fidelity):
  - idea_c_vision_tf_compare_v1: undertrained (do not relaunch)
  - idea_c_vision_tf_compare_v2: R=100, N=10, α=0.1, learnable HPs (Fashion/CIFAR)
  - idea_c_vision_tf_compare_r200: R=200 only, N=10, α=0.1
  - idea_c_vision_tf_compare_n8: N=8, R=100, α=0.1
  - idea_c_vision_tf_compare_a05: N=10, R=100, α=0.5 (milder Dirichlet)

Default async federation (v2/r200): N=10, α=0.1, delays 2–16.
n8: N=8, α=0.1, delays (2,2,4,4,8,8,12,16).
a05: N=10, α=0.5, same delays as v2. Descriptive ACC only;
formal Stage-1A never rewritten.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from .break_gate_e2e import (
    _apply_flat,
    _flatten,
    _layout,
    _loader,
    _state_delta,
    _unflatten,
)
from .config import load_config, result_dir
from .data import (
    load_cifar10_unit_interval,
    load_fashion_mnist_unit_interval,
    load_mnist_unit_interval,
)
from .device import activate_device, resolve_device, seed_everything
from .idea_c_stage1a import (
    CostLedger,
    CorrectionCoefficients,
    IDEA_C_METHOD,
    correction_for_method,
    dequantize_rms_u8,
    expand_postsynaptic,
    protocol_identity,
    quantize_rms_u8,
)
from .models import (
    build_fashion_mnist_2conv2fc_bntt,
    build_fedsnn_alexnet_bntt,
    build_mnist_2conv2fc_bntt,
)
from .mnist_module_ablation_e2e import (
    DEFAULT_COEFFICIENTS,
    METHOD_DRIFT,
    METHOD_TF,
    METHOD_TF_LONG,
    ModuleFlags,
    ModuleEligibilitySummary,
    LayerEligibilityPayload,
    _eligibility_block,
    capture_mnist_module_eligibility,
)
from .mnist_two_factor_e2e import (
    DELAYS as DELAYS_N10,
    DELAY_CLASSES as DELAY_CLASSES_N10,
    REGISTERED_ALPHA,
    REGISTERED_CLIENTS as CLIENTS_N10,
    REGISTERED_SEEDS,
    SCHEDULE as SCHEDULE_N10,
    _buffer_keys,
    _evaluate,
    _merge_buffers,
    _partitions,
)
from .protocol import clone_state
from .train import _append_jsonl, _code_commit

# Default export for unit tests (v2 R=100). Other tracks use TRACK_ROUNDS + protocol tables.
REGISTERED_TRACK = "idea_c_vision_tf_compare_v2"
REGISTERED_FIDELITY = REGISTERED_TRACK
REGISTERED_PAPER_METHOD = "vision_tf_compare_e2e"
LEGACY_UNDERTRAINED_TRACK = "idea_c_vision_tf_compare_v1"
R200_TRACK = "idea_c_vision_tf_compare_r200"
N8_TRACK = "idea_c_vision_tf_compare_n8"
A05_TRACK = "idea_c_vision_tf_compare_a05"

# Back-compat aliases (N=10 protocol used by v1/v2/r200 and unit tests)
REGISTERED_CLIENTS = CLIENTS_N10
DELAYS = DELAYS_N10
DELAY_CLASSES = DELAY_CLASSES_N10
SCHEDULE = SCHEDULE_N10

# fidelity -> server rounds
TRACK_ROUNDS = {
    LEGACY_UNDERTRAINED_TRACK: 100,
    REGISTERED_TRACK: 100,
    R200_TRACK: 200,
    N8_TRACK: 100,
    A05_TRACK: 100,
}

# Dirichlet partition alpha by track (default harder α=0.1)
TRACK_ALPHA = {
    LEGACY_UNDERTRAINED_TRACK: REGISTERED_ALPHA,
    REGISTERED_TRACK: REGISTERED_ALPHA,
    R200_TRACK: REGISTERED_ALPHA,
    N8_TRACK: REGISTERED_ALPHA,
    A05_TRACK: 0.5,
}

# N=8 protocol: keep 2 per delay class (F/M/S/VS), drop two slow/very_slow clients
# from the N=10 table while preserving the same delay extremes (2–16).
CLIENTS_N8 = 8
DELAY_CLASSES_N8 = (
    "fast",
    "fast",
    "medium",
    "medium",
    "slow",
    "slow",
    "very_slow",
    "very_slow",
)
DELAYS_N8 = (2, 2, 4, 4, 8, 8, 12, 16)
# Length-16 schedule covering all 8 clients; mild fast bias (same spirit as N=10).
SCHEDULE_N8 = (
    0,
    1,
    2,
    4,
    6,
    0,
    3,
    5,
    7,
    1,
    2,
    4,
    0,
    3,
    5,
    6,
)

COMPARE_METHODS = (
    METHOD_DRIFT,
    METHOD_TF,
    METHOD_TF_LONG,
)

GRAY_SPIKE_LAYERS = ("conv1.weight", "conv2.weight", "fc1.weight")
CIFAR_SPIKE_LAYERS = (
    "convs.0.weight",
    "convs.1.weight",
    "convs.2.weight",
    "convs.3.weight",
    "convs.4.weight",
    "fc1.weight",
)

DATASET_PROFILES = {
    "mnist": {
        "model_name": "mnist_2conv2fc_bntt",
        "architecture": "mnist_2conv2fc_bntt",
        "spike_layers": GRAY_SPIKE_LAYERS,
        "input_shape": (1, 28, 28),
        "default_timesteps": 2,
        "default_lr": 0.1,
        "default_local_epochs": 1,
        "default_batch_size": 64,
        # MNIST was already learnable under LE=1; R-only ablation keeps LE=1.
        "min_local_epochs": 1,
        "min_timesteps": 2,
    },
    "fashion_mnist": {
        "model_name": "fashion_mnist_2conv2fc_bntt",
        "architecture": "fashion_mnist_2conv2fc_bntt",
        "spike_layers": GRAY_SPIKE_LAYERS,
        "input_shape": (1, 28, 28),
        "default_timesteps": 4,
        "default_lr": 0.1,
        "default_local_epochs": 5,
        "default_batch_size": 64,
        "min_local_epochs": 5,
        "min_timesteps": 4,
    },
    "cifar10": {
        "model_name": "fedsnn_alexnet_bntt",
        "architecture": "fedsnn_alexnet_bntt",
        "spike_layers": CIFAR_SPIKE_LAYERS,
        "input_shape": (3, 32, 32),
        "default_timesteps": 4,
        "default_lr": 0.05,
        "default_local_epochs": 5,
        "default_batch_size": 128,
        "min_local_epochs": 5,
        "min_timesteps": 4,
    },
}


def _flags_for_method(method: str) -> ModuleFlags:
    if method == METHOD_DRIFT:
        return ModuleFlags(third_factor=False)
    if method == METHOD_TF:
        return ModuleFlags(third_factor=True)
    if method == METHOD_TF_LONG:
        return ModuleFlags(third_factor=True, separate_long_history=True)
    raise ValueError(f"unsupported compare method: {method}")


def _correction_method_name(method: str) -> str:
    if method == METHOD_DRIFT:
        return "drift_age"
    if method in {METHOD_TF, METHOD_TF_LONG}:
        return IDEA_C_METHOD
    raise ValueError(f"unsupported method: {method}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _dataset_name(config: Mapping[str, Any]) -> str:
    name = str(config["dataset"]["name"]).lower()
    if name in {"fashion-mnist", "fashionmnist"}:
        return "fashion_mnist"
    if name in {"cifar-10", "cifar_10"}:
        return "cifar10"
    return name


def _fidelity(config: Mapping[str, Any]) -> str:
    return str(config["paper"].get("fidelity", ""))


def _expected_rounds(config: Mapping[str, Any]) -> int:
    fidelity = _fidelity(config)
    _require(fidelity in TRACK_ROUNDS, f"unsupported fidelity {fidelity}; known={sorted(TRACK_ROUNDS)}")
    return int(TRACK_ROUNDS[fidelity])


@dataclass(frozen=True)
class AsyncProtocol:
    clients: int
    delay_classes: tuple[str, ...]
    delays: tuple[int, ...]
    schedule: tuple[int, ...]


PROTOCOL_N10 = AsyncProtocol(
    clients=CLIENTS_N10,
    delay_classes=DELAY_CLASSES_N10,
    delays=DELAYS_N10,
    schedule=SCHEDULE_N10,
)
PROTOCOL_N8 = AsyncProtocol(
    clients=CLIENTS_N8,
    delay_classes=DELAY_CLASSES_N8,
    delays=DELAYS_N8,
    schedule=SCHEDULE_N8,
)


def _async_protocol(config: Mapping[str, Any]) -> AsyncProtocol:
    fidelity = _fidelity(config)
    if fidelity == N8_TRACK:
        return PROTOCOL_N8
    return PROTOCOL_N10


def _expected_alpha(config: Mapping[str, Any]) -> float:
    fidelity = _fidelity(config)
    _require(fidelity in TRACK_ALPHA, f"unsupported fidelity {fidelity} for alpha; known={sorted(TRACK_ALPHA)}")
    return float(TRACK_ALPHA[fidelity])


def _profile(config: Mapping[str, Any]) -> dict[str, Any]:
    name = _dataset_name(config)
    _require(name in DATASET_PROFILES, f"unsupported dataset {name}")
    return DATASET_PROFILES[name]


def _validate(config: Mapping[str, Any]) -> None:
    notes = config.get("notes") or {}
    profile = _profile(config)
    fidelity = _fidelity(config)
    expected_rounds = _expected_rounds(config)
    expected_alpha = _expected_alpha(config)
    protocol = _async_protocol(config)
    _require(notes.get("vision_tf_compare_exploratory") is True, "requires vision compare exploratory note")
    _require(notes.get("third_factor_active") is True, "third-factor must be active")
    _require(notes.get("formal_stage1a_immutable") is True, "Stage-1A immutable")
    _require(notes.get("oracle_gate_bypassed") is True, "oracle gate must be bypassed")
    _require(notes.get("mechanical_e2e_gate_disabled") is True, "mechanical e2e gate disabled")
    _require(notes.get("rerun_authorized_stage1a") is not True, "must not authorize Stage-1A rerun")
    _require(
        fidelity != LEGACY_UNDERTRAINED_TRACK,
        "v1 undertrained track is frozen; use v2/r200/n8/a05",
    )
    _require(str(config["paper"]["method"]) == REGISTERED_PAPER_METHOD, "wrong paper.method")
    _require(str(config["stage"]["mode"]) == "end_to_end", "stage.mode must be end_to_end")
    _require(
        int(config["federation"]["clients"]) == protocol.clients,
        f"requires {protocol.clients} clients for fidelity={fidelity}",
    )
    _require(
        tuple(config["federation"]["delay_classes"]) == protocol.delay_classes,
        "delay classes mismatch",
    )
    delay_updates = tuple(int(x) for x in config["federation"]["delay_updates"])
    _require(delay_updates == protocol.delays, "delay_updates mismatch")
    _require(len(protocol.delays) == protocol.clients, "protocol delays length != clients")
    _require(set(protocol.schedule) == set(range(protocol.clients)), "schedule must cover every client")
    _require(
        abs(float(config["dataset"]["alpha"]) - expected_alpha) < 1e-12,
        f"alpha must be {expected_alpha} for fidelity={fidelity}",
    )
    if "label_skew_alpha" in config["dataset"]:
        _require(
            abs(float(config["dataset"]["label_skew_alpha"]) - expected_alpha) < 1e-12,
            f"label_skew_alpha must match alpha={expected_alpha}",
        )
    _require(
        int(config["training"]["rounds"]) == expected_rounds,
        f"rounds must be {expected_rounds} for fidelity={fidelity}",
    )
    _require(str(config["model"]["name"]) == profile["model_name"], "model mismatch for dataset")
    _require(
        int(config["model"]["timesteps"]) >= int(profile["min_timesteps"]),
        f"timesteps must be >= {profile['min_timesteps']} for {_dataset_name(config)}",
    )
    _require(
        int(config["training"]["local_epochs"]) >= int(profile["min_local_epochs"]),
        f"local_epochs must be >= {profile['min_local_epochs']} for {_dataset_name(config)}",
    )
    methods = tuple(config.get("compare_methods") or COMPARE_METHODS)
    _require(methods == COMPARE_METHODS, f"compare_methods must be {COMPARE_METHODS}")


def load_coefficients(config: Mapping[str, Any]) -> dict[str, CorrectionCoefficients]:
    gate = config.get("gate") or {}
    if gate.get("two_factor_calibration_manifest_path") or gate.get("third_factor_calibration_manifest_path"):
        raise ValueError("vision compare uses grid-centre defaults; no SHD calibration manifests")
    DEFAULT_COEFFICIENTS.validate()
    return {method: DEFAULT_COEFFICIENTS for method in COMPARE_METHODS}


def _apply_bntt(module: Any, inputs: Any) -> Any:
    from torch import nn

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


def capture_alexnet_module_eligibility(
    model: Any,
    images: Any,
    labels: Any | None,
    *,
    flags: ModuleFlags,
    surrogate_beta: float | None = None,
) -> ModuleEligibilitySummary:
    """Capture AlexNet-BNTT eligibility (postsynaptic TF / TF+long)."""
    import math as _math
    import torch
    from torch import nn

    if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError("CIFAR eligibility expects [B,3,32,32]")
    if flags.third_factor and labels is None:
        raise ValueError("third_factor requires labels")
    if flags.connection_level or flags.signed_timing:
        raise ValueError("vision compare track only supports TF and TF+long (no signed/conn)")

    beta = float(
        surrogate_beta if surrogate_beta is not None else getattr(model, "surrogate_beta", 2.0)
    )
    threshold = float(model.threshold)
    decay = float(model.membrane_decay)
    timesteps = int(model.timesteps)
    batch = int(images.shape[0])

    def _atan_post(centered):
        return (beta / 2.0) / (1.0 + (_math.pi * beta * centered / 2.0).square())

    conv_shapes = [
        (64, 32, 32),
        (192, 16, 16),
        (384, 8, 8),
        (256, 8, 8),
        (256, 8, 8),
    ]
    membranes = [images.new_zeros((batch, *shape)) for shape in conv_shapes]
    membrane_fc1 = images.new_zeros((batch, 1024))

    pre_steps = {layer: [] for layer in CIFAR_SPIKE_LAYERS}
    post_steps = {layer: [] for layer in CIFAR_SPIKE_LAYERS}
    spike_steps = {layer: [] for layer in CIFAR_SPIKE_LAYERS}

    was_training = bool(model.training)
    model.eval()
    third_fc1 = None
    third_scalar = None
    try:
        with torch.no_grad():
            if flags.third_factor:
                logits = model(images)
                labels_t = torch.as_tensor(labels, device=images.device, dtype=torch.long)
                probabilities = torch.softmax(logits.detach(), dim=-1)
                targets = nn.functional.one_hot(
                    labels_t, num_classes=int(logits.shape[-1])
                ).to(probabilities.dtype)
                third_fc1 = ((targets - probabilities) @ model.fc2.weight.detach()).abs()
                third_scalar = third_fc1.mean(dim=1, keepdim=True)

            for timestep in range(timesteps):
                spikes = (torch.rand_like(images) <= images).to(images.dtype)
                for index, convolution in enumerate(model.convs):
                    layer = f"convs.{index}.weight"
                    pre_steps[layer].append(spikes.flatten(1))
                    current = _apply_bntt(
                        model.bntt_convs[index][timestep], convolution(spikes)
                    )
                    charged = decay * membranes[index] + current
                    centered = charged - threshold
                    spikes = (centered >= 0).to(images.dtype)
                    membranes[index] = charged - spikes * threshold
                    post = _atan_post(centered).flatten(2).mean(2)
                    if third_scalar is not None:
                        post = post * third_scalar
                    post_steps[layer].append(post)
                    spike_steps[layer].append(spikes.flatten(2).mean(2))
                    if index in model.pool_after:
                        spikes = nn.functional.avg_pool2d(spikes, 2)

                flat = spikes.flatten(1)
                pre_steps["fc1.weight"].append(flat)
                current = _apply_bntt(model.bntt_fc1[timestep], model.fc1(flat))
                charged = decay * membrane_fc1 + current
                centered = charged - threshold
                spikes = (centered >= 0).to(images.dtype)
                membrane_fc1 = charged - spikes * threshold
                post = _atan_post(centered)
                if third_fc1 is not None:
                    post = post * third_fc1
                post_steps["fc1.weight"].append(post)
                spike_steps["fc1.weight"].append(spikes)
    finally:
        if was_training:
            model.train()

    layers: dict[str, LayerEligibilityPayload] = {}
    for layer in CIFAR_SPIKE_LAYERS:
        pre_bt = torch.stack(pre_steps[layer], dim=1)
        post_bt = torch.stack(post_steps[layer], dim=1)
        spike_bt = torch.stack(spike_steps[layer], dim=1)
        elig, _ = _eligibility_block(pre_bt, post_bt, flags=flags)
        activity = spike_bt.to(torch.float64).mean(dim=(0, 1))
        layers[layer] = LayerEligibilityPayload(
            kind="postsynaptic",
            eligibility=quantize_rms_u8(elig),
            activity=quantize_rms_u8(activity),
            out_features=int(elig.numel()),
            in_features=1,
        )
    return ModuleEligibilitySummary(
        layers=layers,
        samples=batch,
        timesteps=timesteps,
        flags=flags,
    )


def _capture_eligibility(
    dataset_name: str,
    model: Any,
    images: Any,
    labels: Any | None,
    flags: ModuleFlags,
    config: Mapping[str, Any],
) -> ModuleEligibilitySummary:
    beta = float(config["model"].get("surrogate_beta", 2.0))
    if dataset_name in {"mnist", "fashion_mnist"}:
        return capture_mnist_module_eligibility(
            model, images, labels, flags=flags, surrogate_beta=beta
        )
    if dataset_name == "cifar10":
        return capture_alexnet_module_eligibility(
            model, images, labels, flags=flags, surrogate_beta=beta
        )
    raise ValueError(f"unsupported dataset for capture: {dataset_name}")


def _feature_tensors(
    layer: str,
    parameter: Any,
    drift: Any,
    summary: ModuleEligibilitySummary,
):
    drift = drift.abs()
    payload = summary.layers[layer]
    elig = dequantize_rms_u8(
        payload.eligibility, device=parameter.device, dtype=parameter.dtype
    )
    act = dequantize_rms_u8(
        payload.activity, device=parameter.device, dtype=parameter.dtype
    )
    if payload.kind != "postsynaptic":
        raise ValueError("vision compare expects postsynaptic eligibility only")
    eligibility = expand_postsynaptic(elig, parameter)
    activity = expand_postsynaptic(act, parameter)
    return drift, eligibility, activity


def _model_builder(config: Mapping[str, Any]) -> Callable[[], Any]:
    profile = _profile(config)
    name = profile["model_name"]
    kwargs = dict(
        timesteps=int(config["model"]["timesteps"]),
        classes=int(config["model"].get("classes", 10)),
        surrogate_beta=float(config["model"].get("surrogate_beta", 2.0)),
        threshold=float(config["model"].get("threshold", 1.0)),
        membrane_decay=float(config["model"].get("membrane_decay", 0.95)),
        track_runtime_activity=True,
    )
    if name == "mnist_2conv2fc_bntt":
        return lambda: build_mnist_2conv2fc_bntt(**kwargs)
    if name == "fashion_mnist_2conv2fc_bntt":
        return lambda: build_fashion_mnist_2conv2fc_bntt(**kwargs)
    if name == "fedsnn_alexnet_bntt":
        return lambda: build_fedsnn_alexnet_bntt(**kwargs)
    raise ValueError(f"unsupported model {name}")


def _load_dataset(dataset_name: str, data_root: str | Path, download: bool):
    if dataset_name == "mnist":
        train_set = load_mnist_unit_interval(data_root, train=True, download=download)
        test_set = load_mnist_unit_interval(data_root, train=False, download=download)
        return train_set, test_set
    if dataset_name == "fashion_mnist":
        train_set = load_fashion_mnist_unit_interval(data_root, train=True, download=download)
        test_set = load_fashion_mnist_unit_interval(data_root, train=False, download=download)
        return train_set, test_set
    if dataset_name == "cifar10":
        train_set = load_cifar10_unit_interval(data_root, train=True, download=download)
        test_set = load_cifar10_unit_interval(data_root, train=False, download=download)
        return train_set, test_set
    raise ValueError(f"unsupported dataset {dataset_name}")


def _train_local(
    builder: Any,
    base_state: Mapping[str, Any],
    dataset: Any,
    indices: Any,
    device: Any,
    config: Mapping[str, Any],
    seed: int,
    *,
    dataset_name: str,
    flags: ModuleFlags,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], ModuleEligibilitySummary, float, int, int, dict[str, Any]]:
    import torch
    from torch import nn

    model = builder().to(device)
    model.load_state_dict(base_state)
    training = config["training"]
    loader, ordered_indices = _loader(dataset, indices, int(training["batch_size"]), seed, config)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training["learning_rate"]),
        momentum=float(training.get("momentum", 0.0)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()
    loss_sum = None
    summary: ModuleEligibilitySummary | None = None
    steps = samples = 0
    layer_rate_sum = None
    for _epoch in range(int(training["local_epochs"])):
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if summary is None:
                summary = _capture_eligibility(
                    dataset_name,
                    model,
                    images,
                    labels if flags.third_factor else None,
                    flags,
                    config,
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
    audit = {
        "eligibility_mode": "third_factor" if flags.third_factor else "capture_for_parity",
        "module_flags": {
            "third_factor": flags.third_factor,
            "signed_timing": flags.signed_timing,
            "connection_level": flags.connection_level,
            "separate_long_history": flags.separate_long_history,
        },
        "layer_spike_rates_mean": mean_layer_rates,
        "hidden_spike_rate_mean": (
            float(sum(mean_layer_rates) / len(mean_layer_rates)) if mean_layer_rates else 0.0
        ),
        "ordered_indices_sha256": __import__("hashlib")
        .sha256(ordered_indices.tobytes())
        .hexdigest(),
    }
    return (
        clone_state(model.state_dict(), next(iter(base_state.values())).device),
        summary,
        mean_loss,
        steps,
        samples,
        audit,
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
    coefficients = load_coefficients(config)
    dataset_name = _dataset_name(config)
    profile = _profile(config)
    protocol = _async_protocol(config)
    spike_layers = profile["spike_layers"]
    seed = int(config["training"]["seed"])
    rounds = 2 if smoke else int(config["training"]["rounds"])
    records: list[dict[str, Any]] = []
    reference_model = builder().to(device)
    reference_state = clone_state(reference_model.state_dict(), device)
    for method in COMPARE_METHODS:
        flags = _flags_for_method(method)
        model = builder().to(device)
        state = clone_state(reference_state, device)
        model.load_state_dict(state)
        layout = _layout(model)
        buffer_keys = _buffer_keys(model)
        if not any("running_" in key for key in buffer_keys):
            raise RuntimeError("BNTT model missing running_* buffers")
        client_states = [clone_state(state, device) for _ in range(protocol.clients)]
        client_versions = [0] * protocol.clients
        ledger = CostLedger()
        method_coefficients = coefficients[method]
        for server_update in range(rounds):
            client_id = protocol.schedule[server_update % len(protocol.schedule)]
            base = client_states[client_id]
            local, summary, loss, steps, samples, audit = _train_local(
                builder,
                base,
                train_set,
                partitions[client_id],
                device,
                config,
                seed + server_update * 100 + client_id,
                dataset_name=dataset_name,
                flags=flags,
                max_batches=1 if smoke else None,
            )
            delta_by_layer = _unflatten(_state_delta(local, base, layout), layout)
            age = max(1, server_update - client_versions[client_id] + protocol.delays[client_id])
            corrected = {}
            correction_min = 1.0
            correction_max = 0.0
            for layer, parameter in delta_by_layer.items():
                if layer in spike_layers:
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
            ledger.summary_payload_bits += summary.payload_bits
            ledger.activity_baseline_payload_bits += summary.activity_payload_bits
            if summary.activity_payload_bits != summary.payload_bits:
                raise RuntimeError("activity and eligibility payloads are not matched")
            ledger.local_training_steps += steps
            ledger.eligibility_factor_multiply_adds += (
                samples * int(config["model"]["timesteps"]) * int(sum(model.spike_channel_sizes))
            )
            records.append(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "method": method,
                    "server_update": server_update,
                    "client_id": client_id,
                    "age": age,
                    "delay_class": config["federation"]["delay_classes"][client_id],
                    "train_loss": loss,
                    "test_loss": test_loss,
                    "test_accuracy": accuracy,
                    "eligibility_mode": audit["eligibility_mode"],
                    "module_flags": audit["module_flags"],
                    "architecture": profile["architecture"],
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


def run_vision_tf_compare_e2e(
    config: dict[str, Any],
    data_root: str | Path,
    device_spec: str,
    *,
    smoke: bool = False,
) -> Path:
    _validate(config)
    coefficients = load_coefficients(config)
    dataset_name = _dataset_name(config)
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    device_info = resolve_device(device_spec)
    device = activate_device(device_info)
    download = bool(config["dataset"].get("download", False))
    train_set, test_set = _load_dataset(dataset_name, data_root, download)
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
        "dataset": dataset_name,
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "code_identity": code_identity,
        "protocol_identity": protocol_identity(config, code_identity),
        "track": _fidelity(config),
        "rounds": int(config["training"]["rounds"]),
        "clients": _async_protocol(config).clients,
        "delays": list(_async_protocol(config).delays),
        "dirichlet_alpha": _expected_alpha(config),
        "oracle_gate_bypassed": True,
        "mechanical_e2e_gate_disabled": True,
        "third_factor_active": True,
        "compare_methods": list(COMPARE_METHODS),
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


def run_vision_tf_compare_e2e_file(
    config_path: str | Path,
    data_root: str | Path,
    device_spec: str = "auto",
    *,
    smoke: bool = False,
) -> Path:
    return run_vision_tf_compare_e2e(
        load_config(config_path), data_root, device_spec, smoke=smoke
    )


__all__ = [
    "A05_TRACK",
    "CLIENTS_N8",
    "COMPARE_METHODS",
    "DATASET_PROFILES",
    "DELAYS_N8",
    "DELAY_CLASSES_N8",
    "N8_TRACK",
    "R200_TRACK",
    "REGISTERED_FIDELITY",
    "REGISTERED_PAPER_METHOD",
    "REGISTERED_SEEDS",
    "REGISTERED_TRACK",
    "SCHEDULE_N8",
    "TRACK_ALPHA",
    "TRACK_ROUNDS",
    "capture_alexnet_module_eligibility",
    "load_coefficients",
    "run_vision_tf_compare_e2e",
    "run_vision_tf_compare_e2e_file",
]
