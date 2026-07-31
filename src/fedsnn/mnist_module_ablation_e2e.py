"""MNIST third-factor + module leave-one-in ablation (exploratory).

User (2026-07-28): put third-factor back on the active algorithm list, then probe
whether any of the retired modules helps on top of third-factor:

1. signed timing
2. connection-level
3. separate long history

Protocol matches MNIST v2 harder async settings (N=10, R=100, α=0.1, delays 2–16).
Descriptive ACC only; formal Stage-1A archive is never rewritten.
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
from .mnist_two_factor_e2e import (
    DELAYS,
    DELAY_CLASSES,
    REGISTERED_ALPHA,
    REGISTERED_CLIENTS,
    REGISTERED_ROUNDS,
    REGISTERED_SEEDS,
    SCHEDULE,
    SPIKE_WEIGHT_LAYERS,
    _buffer_keys,
    _evaluate,
    _merge_buffers,
    _model_builder,
    _partitions,
)
from .protocol import clone_state
from .third_factor_oracle import THIRD_FACTOR_METHOD
from .train import _append_jsonl, _code_commit

REGISTERED_TRACK = "idea_c_mnist_tf_module_ablation_v1"
REGISTERED_FIDELITY = REGISTERED_TRACK
REGISTERED_PAPER_METHOD = "mnist_tf_module_ablation_e2e"

# Combo track: long+connection and long+signed on TF base (leave-one-in v1 frozen).
COMBO_TRACK = "idea_c_mnist_tf_module_combo_v1"
COMBO_FIDELITY = COMBO_TRACK
COMBO_PAPER_METHOD = "mnist_tf_module_combo_e2e"

# Leave-one-in: base third-factor plus each module alone.
METHOD_DRIFT = "drift_age"
METHOD_TF = THIRD_FACTOR_METHOD
METHOD_TF_SIGNED = "third_factor_signed_timing"
METHOD_TF_CONN = "third_factor_connection_level"
METHOD_TF_LONG = "third_factor_long_history"
METHOD_TF_LONG_CONN = "third_factor_long_connection"
METHOD_TF_LONG_SIGNED = "third_factor_long_signed"

COMPARE_METHODS = (
    METHOD_DRIFT,
    METHOD_TF,
    METHOD_TF_SIGNED,
    METHOD_TF_CONN,
    METHOD_TF_LONG,
)

# Focused combo panel: pure long base + two pairwise stacks.
COMBO_METHODS = (
    METHOD_DRIFT,
    METHOD_TF_LONG,
    METHOD_TF_LONG_CONN,
    METHOD_TF_LONG_SIGNED,
)

# Historical recurrent decays (from retired recurrent_eligibility).
LONG_PRE_DECAY = 0.85
LONG_POST_DECAY = 0.70
LONG_HISTORY_DECAY = 0.95
SHARED_TRACE_DECAY = 0.9

DEFAULT_COEFFICIENTS = CorrectionCoefficients(
    age=0.12,
    drift=0.35,
    eligibility=-0.1,
    interaction=0.2,
)


@dataclass(frozen=True)
class ModuleFlags:
    third_factor: bool = True
    signed_timing: bool = False
    connection_level: bool = False
    separate_long_history: bool = False


def _flags_for_method(method: str) -> ModuleFlags:
    if method == METHOD_DRIFT:
        # Capture still runs for payload parity; drift correction zeros eligibility.
        return ModuleFlags(third_factor=False)
    if method == METHOD_TF:
        return ModuleFlags(third_factor=True)
    if method == METHOD_TF_SIGNED:
        return ModuleFlags(third_factor=True, signed_timing=True)
    if method == METHOD_TF_CONN:
        return ModuleFlags(third_factor=True, connection_level=True)
    if method == METHOD_TF_LONG:
        return ModuleFlags(third_factor=True, separate_long_history=True)
    if method == METHOD_TF_LONG_CONN:
        return ModuleFlags(
            third_factor=True,
            connection_level=True,
            separate_long_history=True,
        )
    if method == METHOD_TF_LONG_SIGNED:
        return ModuleFlags(
            third_factor=True,
            signed_timing=True,
            separate_long_history=True,
        )
    raise ValueError(f"unsupported ablation method: {method}")


def _track_profile(config: Mapping[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    """Return (track, fidelity, paper_method, methods) for leave-one-in or combo."""
    fidelity = str(config["paper"].get("fidelity", ""))
    if fidelity == REGISTERED_FIDELITY:
        return REGISTERED_TRACK, REGISTERED_FIDELITY, REGISTERED_PAPER_METHOD, COMPARE_METHODS
    if fidelity == COMBO_FIDELITY:
        return COMBO_TRACK, COMBO_FIDELITY, COMBO_PAPER_METHOD, COMBO_METHODS
    raise ValueError(
        f"unsupported fidelity {fidelity!r}; expected "
        f"{REGISTERED_FIDELITY!r} or {COMBO_FIDELITY!r}"
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate(config: Mapping[str, Any]) -> None:
    notes = config.get("notes") or {}
    track, fidelity, paper_method, expected_methods = _track_profile(config)
    exploratory_key = (
        "mnist_tf_module_combo_exploratory"
        if track == COMBO_TRACK
        else "mnist_tf_module_ablation_exploratory"
    )
    _require(notes.get(exploratory_key) is True, f"requires {exploratory_key} note")
    _require(notes.get("third_factor_active") is True, "third-factor must be re-listed as active")
    _require(notes.get("formal_stage1a_immutable") is True, "Stage-1A immutable")
    _require(notes.get("oracle_gate_bypassed") is True, "oracle gate must be bypassed")
    _require(notes.get("mechanical_e2e_gate_disabled") is True, "mechanical e2e gate disabled")
    _require(notes.get("rerun_authorized_stage1a") is not True, "must not authorize Stage-1A rerun")
    _require(str(config["paper"].get("fidelity", "")) == fidelity, "fidelity mismatch")
    _require(paper_method == str(config["paper"]["method"]), "wrong paper.method")
    _require(str(config["stage"]["mode"]) == "end_to_end", "stage.mode must be end_to_end")
    _require(str(config["dataset"]["name"]).lower() == "mnist", "dataset must be mnist")
    _require(int(config["federation"]["clients"]) == REGISTERED_CLIENTS, "requires 10 clients")
    _require(tuple(config["federation"]["delay_classes"]) == DELAY_CLASSES, "delay classes mismatch")
    delay_updates = tuple(int(x) for x in config["federation"]["delay_updates"])
    _require(delay_updates == DELAYS, "delay_updates mismatch")
    _require(abs(float(config["dataset"]["alpha"]) - REGISTERED_ALPHA) < 1e-12, "alpha must be 0.1")
    _require(int(config["training"]["rounds"]) == REGISTERED_ROUNDS, "rounds must be 100")
    _require(str(config["model"]["name"]) == "mnist_2conv2fc_bntt", "model mismatch")
    methods = tuple(config.get("compare_methods") or expected_methods)
    _require(methods == expected_methods, f"compare_methods must be {expected_methods}")


def load_coefficients(config: Mapping[str, Any]) -> dict[str, CorrectionCoefficients]:
    gate = config.get("gate") or {}
    if gate.get("two_factor_calibration_manifest_path") or gate.get("third_factor_calibration_manifest_path"):
        raise ValueError("module ablation uses grid-centre defaults; no SHD calibration manifests")
    DEFAULT_COEFFICIENTS.validate()
    _, _, _, methods = _track_profile(config)
    return {method: DEFAULT_COEFFICIENTS for method in methods}


@dataclass
class LayerEligibilityPayload:
    """Per-layer eligibility + matched activity.

    ``kind`` is ``postsynaptic`` (vector length = out channels/neurons) or
    ``connection`` (matrix matching weight shape after reshape to [out, -1]).
    """

    kind: str
    eligibility: QuantizedNeuronSummary
    activity: QuantizedNeuronSummary
    out_features: int
    in_features: int

    @property
    def payload_bits(self) -> int:
        return self.eligibility.payload_bits

    @property
    def activity_payload_bits(self) -> int:
        return self.activity.payload_bits


@dataclass(frozen=True)
class ModuleEligibilitySummary:
    layers: Mapping[str, LayerEligibilityPayload]
    samples: int
    timesteps: int
    flags: ModuleFlags

    @property
    def payload_bits(self) -> int:
        return sum(layer.payload_bits for layer in self.layers.values())

    @property
    def activity_payload_bits(self) -> int:
        return sum(layer.activity_payload_bits for layer in self.layers.values())


def _eligibility_block(
    pre_bt: Any,
    post_bt: Any,
    *,
    flags: ModuleFlags,
) -> tuple[Any, Any]:
    """Return (eligibility_feature, activity) for one block.

    pre_bt: [B,T,P], post_bt: [B,T,C]
    - postsynaptic: eligibility [C]
    - connection: eligibility [C,P]
    activity always [C] (mean spike rate over B,T) for cost parity bookkeeping.
    """
    import torch

    pre = torch.as_tensor(pre_bt)
    post = torch.as_tensor(post_bt)
    if pre.ndim != 3 or post.ndim != 3:
        raise ValueError("factors must be rank-three")
    if pre.shape[:2] != post.shape[:2]:
        raise ValueError("batch/time mismatch")
    batch, timesteps, pre_units = pre.shape
    post_units = post.shape[-1]

    if flags.separate_long_history:
        pre_decay = LONG_PRE_DECAY
        post_decay = LONG_POST_DECAY
        history_decay = LONG_HISTORY_DECAY
    else:
        pre_decay = SHARED_TRACE_DECAY
        post_decay = SHARED_TRACE_DECAY
        history_decay = 0.0  # pure accumulation of step contributions

    pre_trace = torch.zeros(batch, pre_units, device=pre.device, dtype=pre.dtype)
    post_trace = torch.zeros(batch, post_units, device=post.device, dtype=post.dtype)

    if flags.connection_level:
        history = torch.zeros(post_units, pre_units, device=pre.device, dtype=torch.float64)
    else:
        history = torch.zeros(post_units, device=pre.device, dtype=torch.float64)

    for step in range(timesteps):
        pre_now = pre[:, step]
        post_now = post[:, step]
        if flags.connection_level:
            # causal: post(t) ⊗ pre_trace(t-1); anti: post_trace(t-1) ⊗ pre(t)
            if flags.signed_timing:
                causal = post_now.unsqueeze(-1) * pre_trace.unsqueeze(1)
                anti = post_trace.unsqueeze(-1) * pre_now.unsqueeze(1)
                timing = (causal - anti).abs()
            else:
                timing = post_now.unsqueeze(-1) * pre_trace.unsqueeze(1)
            # Mean over batch of squared timing → [C,P]
            step_feat = timing.to(torch.float64).square().mean(dim=0)
        else:
            # Postsynaptic: avoid materializing [B,C,P].
            # mean_{b,p} (post_bc * pre_bp)^2 = mean_b [post_bc^2 * mean_p pre_bp^2]
            if flags.signed_timing:
                # |causal - anti|^2 averaged; expand only when needed for signed.
                # Use identity: (a-b)^2 = a^2 + b^2 - 2ab with a=post*pre_tr, b=post_tr*pre
                pre_tr_energy = pre_trace.square().mean(dim=-1, keepdim=True)  # [B,1]
                pre_energy = pre_now.square().mean(dim=-1, keepdim=True)
                post_sq = post_now.square()
                post_tr_sq = post_trace.square()
                # Cross-term is accumulated exactly below.
                # exact: mean_p pre_tr_p * pre_p
                pre_tr_dot_pre = (pre_trace * pre_now).mean(dim=-1, keepdim=True)
                step_feat = (
                    post_sq * pre_tr_energy
                    + post_tr_sq * pre_energy
                    - 2.0 * post_now * post_trace * pre_tr_dot_pre
                ).mean(dim=0).to(torch.float64).clamp_min(0)
            else:
                pre_energy = pre_trace.square().mean(dim=-1, keepdim=True)  # [B,1]
                step_feat = (post_now.square() * pre_energy).mean(dim=0).to(torch.float64)

        if history_decay > 0.0:
            history = history_decay * history + step_feat
        else:
            history = history + step_feat

        pre_trace = pre_decay * pre_trace + pre_now
        post_trace = post_decay * post_trace + post_now

    if history_decay > 0.0:
        # EMA-style history already normalized by (1 - history_decay) roughly;
        # take sqrt of mean-ish magnitude. Use abs already nonnegative.
        elig = history.clamp_min(0).sqrt().to(post.dtype)
    else:
        elig = (history / max(timesteps, 1)).clamp_min(0).sqrt().to(post.dtype)

    activity = post.to(torch.float64).mean(dim=(0, 1))
    return elig, activity


def capture_mnist_module_eligibility(
    model: Any,
    images: Any,
    labels: Any | None,
    *,
    flags: ModuleFlags,
    surrogate_beta: float | None = None,
) -> ModuleEligibilitySummary:
    """Capture MNIST BNTT eligibility under module flags (third-factor base)."""
    import math as _math
    import torch
    from torch import nn

    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("MNIST eligibility expects [B,1,28,28]")
    if flags.third_factor and labels is None:
        raise ValueError("third_factor requires labels")

    beta = float(
        surrogate_beta if surrogate_beta is not None else getattr(model, "surrogate_beta", 2.0)
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

    def _conv_pre_features(spatial: Any, conv: Any) -> Any:
        """Pre features matching weight view [out, in*kH*kW].

        connection-level: unfold patches, mean over spatial locations → [B, in*k*k]
        so connection eligibility [C,P] aligns with conv weight reshape.
        postsynaptic: full flatten is fine (P is reduced inside eligibility block).
        """
        if flags.connection_level:
            patches = nn.functional.unfold(
                spatial,
                kernel_size=conv.kernel_size,
                dilation=conv.dilation,
                padding=conv.padding,
                stride=conv.stride,
            )
            return patches.mean(dim=2)
        return spatial.flatten(1)

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
                spikes_in = (torch.rand_like(images) <= images).to(images.dtype)
                pre_steps["conv1.weight"].append(
                    _conv_pre_features(spikes_in, model.conv1)
                )

                current = _apply_bntt(model.bntt1[timestep], model.conv1(spikes_in))
                charged = decay * membranes[0] + current
                centered = charged - threshold
                spikes = (centered >= 0).to(images.dtype)
                membranes[0] = charged - spikes * threshold
                post_c1 = _atan_post(centered).flatten(2).mean(2)
                if third_scalar is not None:
                    post_c1 = post_c1 * third_scalar
                post_steps["conv1.weight"].append(post_c1)
                spike_steps["conv1.weight"].append(spikes.flatten(2).mean(2))

                pooled = nn.functional.avg_pool2d(spikes, 2)
                pre_steps["conv2.weight"].append(
                    _conv_pre_features(pooled, model.conv2)
                )

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

    layers: dict[str, LayerEligibilityPayload] = {}
    for layer in SPIKE_WEIGHT_LAYERS:
        pre_bt = torch.stack(pre_steps[layer], dim=1)
        post_bt = torch.stack(post_steps[layer], dim=1)
        # Prefer spike activity for activity baseline when available.
        spike_bt = torch.stack(spike_steps[layer], dim=1)
        elig, _ = _eligibility_block(pre_bt, post_bt, flags=flags)
        activity = spike_bt.to(torch.float64).mean(dim=(0, 1))
        if flags.connection_level:
            # elig [C,P]; activity remains [C] — pad activity payload by tiling
            # so cost ledger can compare bits at the same order of magnitude.
            c, p = int(elig.shape[0]), int(elig.shape[1])
            act_matrix = activity.reshape(c, 1).expand(c, p).contiguous()
            layers[layer] = LayerEligibilityPayload(
                kind="connection",
                eligibility=quantize_rms_u8(elig.reshape(-1)),
                activity=quantize_rms_u8(act_matrix.reshape(-1)),
                out_features=c,
                in_features=p,
            )
        else:
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
    if payload.kind == "connection":
        # Reshape to weight layout [out, ...]
        elig = elig.reshape(payload.out_features, payload.in_features)
        act = act.reshape(payload.out_features, payload.in_features)
        weight_view = parameter.reshape(parameter.shape[0], -1)
        if elig.shape != weight_view.shape:
            raise ValueError(
                f"connection eligibility shape {tuple(elig.shape)} "
                f"!= weight view {tuple(weight_view.shape)} for {layer}"
            )
        eligibility = elig.reshape_as(parameter)
        activity = act.reshape_as(parameter)
    else:
        eligibility = expand_postsynaptic(elig, parameter)
        activity = expand_postsynaptic(act, parameter)
    return drift, eligibility, activity


def _correction_method_name(method: str) -> str:
    if method == METHOD_DRIFT:
        return "drift_age"
    # All TF* methods use the eligibility_informed_staleness correction form.
    if method in {
        METHOD_TF,
        METHOD_TF_SIGNED,
        METHOD_TF_CONN,
        METHOD_TF_LONG,
        METHOD_TF_LONG_CONN,
        METHOD_TF_LONG_SIGNED,
    }:
        return IDEA_C_METHOD
    raise ValueError(f"unsupported method: {method}")


def _train_local(
    builder: Any,
    base_state: Mapping[str, Any],
    dataset: Any,
    indices: Any,
    device: Any,
    config: Mapping[str, Any],
    seed: int,
    *,
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
                summary = capture_mnist_module_eligibility(
                    model,
                    images,
                    labels if flags.third_factor else None,
                    flags=flags,
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
    audit = {
        "eligibility_mode": (
            "third_factor"
            if flags.third_factor
            else "two_factor_capture_for_parity"
        ),
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
    _, _, _, methods = _track_profile(config)
    seed = int(config["training"]["seed"])
    rounds = 2 if smoke else int(config["training"]["rounds"])
    records: list[dict[str, Any]] = []
    reference_model = builder().to(device)
    reference_state = clone_state(reference_model.state_dict(), device)
    for method in methods:
        flags = _flags_for_method(method)
        model = builder().to(device)
        state = clone_state(reference_state, device)
        model.load_state_dict(state)
        layout = _layout(model)
        buffer_keys = _buffer_keys(model)
        if not any(key.startswith("bntt") and "running_" in key for key in buffer_keys):
            raise RuntimeError("MNIST BNTT missing running_* buffers")
        client_states = [clone_state(state, device) for _ in range(REGISTERED_CLIENTS)]
        client_versions = [0] * REGISTERED_CLIENTS
        ledger = CostLedger()
        method_coefficients = coefficients[method]
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
                flags=flags,
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
            # CostLedger.add_summary expects EligibilitySummary; book payload manually.
            ledger.summary_payload_bits += summary.payload_bits
            ledger.activity_baseline_payload_bits += summary.activity_payload_bits
            if summary.activity_payload_bits != summary.payload_bits:
                # connection-level pads activity to match; still require equality
                raise RuntimeError("activity and eligibility payloads are not matched")
            ledger.local_training_steps += steps
            ledger.eligibility_factor_multiply_adds += (
                samples * int(config["model"]["timesteps"]) * int(sum(model.spike_channel_sizes))
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
                    "eligibility_mode": audit["eligibility_mode"],
                    "module_flags": audit["module_flags"],
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


def run_mnist_module_ablation_e2e(
    config: dict[str, Any],
    data_root: str | Path,
    device_spec: str,
    *,
    smoke: bool = False,
) -> Path:
    _validate(config)
    track, _, _, methods = _track_profile(config)
    coefficients = load_coefficients(config)
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
        "track": track,
        "oracle_gate_bypassed": True,
        "mechanical_e2e_gate_disabled": True,
        "third_factor_active": True,
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


def run_mnist_module_ablation_e2e_file(
    config_path: str | Path,
    data_root: str | Path,
    device_spec: str = "auto",
    *,
    smoke: bool = False,
) -> Path:
    return run_mnist_module_ablation_e2e(
        load_config(config_path), data_root, device_spec, smoke=smoke
    )


__all__ = [
    "COMPARE_METHODS",
    "COMBO_FIDELITY",
    "COMBO_METHODS",
    "COMBO_PAPER_METHOD",
    "COMBO_TRACK",
    "DEFAULT_COEFFICIENTS",
    "METHOD_DRIFT",
    "METHOD_TF",
    "METHOD_TF_CONN",
    "METHOD_TF_LONG",
    "METHOD_TF_LONG_CONN",
    "METHOD_TF_LONG_SIGNED",
    "METHOD_TF_SIGNED",
    "ModuleFlags",
    "REGISTERED_FIDELITY",
    "REGISTERED_PAPER_METHOD",
    "REGISTERED_SEEDS",
    "REGISTERED_TRACK",
    "capture_mnist_module_eligibility",
    "descriptive_e2e_summary",
    "load_coefficients",
    "run_mnist_module_ablation_e2e",
    "run_mnist_module_ablation_e2e_file",
    "seed_rows_from_metrics",
]
