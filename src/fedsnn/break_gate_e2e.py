"""Exploratory break-gate end-to-end for Idea C two-factor vs third-factor.

This track deliberately bypasses the multi-layer Oracle hard gate.  Formal
Stage-1A remains immutable (``STOP_ORACLE_FAIL`` / ``SKIPPED_BY_ORACLE_GATE``).
There is no mechanical e2e PASS that upgrades the formal archive: results are
descriptive accuracy comparisons only.  Absence of ACC advantage may be used to
terminate the innovation.

Active methods in one fair seed run:

1. ``drift_age``
2. ``activity_drift_age``
3. ``eligibility_informed_staleness`` (two-factor)
4. ``third_factor_eligibility_informed_staleness`` (third-factor)
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .config import load_config, result_dir
from .device import activate_device, resolve_device, seed_everything
from .idea_c_stage1a import (
    CostLedger,
    CorrectionCoefficients,
    EligibilitySummary,
    IDEA_C_METHOD,
    correction_for_method,
    dequantize_rms_u8,
    expand_postsynaptic,
    protocol_identity,
    trapezoid_auc,
    capture_recurrent_lif_eligibility,
)
from .partition import dirichlet_partition
from .protocol import clone_state
from .shd import load_shd
from .shd_model import build_recurrent_lif_shd
from .third_factor_oracle import (
    THIRD_FACTOR_METHOD,
    capture_third_factor_lif_eligibility,
)
from .train import _append_jsonl, _code_commit

REGISTERED_TRACK = "idea_c_break_gate_e2e_v1"
REGISTERED_FIDELITY = "idea_c_break_gate_e2e_v1"
REGISTERED_SEEDS = (2, 3, 4, 5, 6)
COMPARE_METHODS = (
    "drift_age",
    "activity_drift_age",
    IDEA_C_METHOD,
    THIRD_FACTOR_METHOD,
)
BASELINE_METHODS = ("drift_age", "activity_drift_age")
TWO_FACTOR_METHOD = IDEA_C_METHOD
SCHEDULE = (0, 1, 0, 2, 1, 0, 3, 1, 4, 2)
DELAYS = (1, 1, 2, 3, 3)


def _layout(model: Any) -> tuple[tuple[str, tuple[int, ...], int], ...]:
    return tuple((name, tuple(value.shape), value.numel()) for name, value in model.named_parameters())


def _flatten(values: Mapping[str, Any], layout: Any):
    import torch

    return torch.cat([values[name].reshape(-1) for name, _shape, _count in layout])


def _unflatten(values: Any, layout: Any) -> dict[str, Any]:
    result = {}
    offset = 0
    for name, shape, count in layout:
        result[name] = values[offset : offset + count].reshape(shape)
        offset += count
    if offset != values.numel():
        raise ValueError("flat tensor does not match model layout")
    return result


def _state_delta(local: Mapping[str, Any], base: Mapping[str, Any], layout: Any):
    return _flatten({name: local[name] - base[name] for name, _shape, _count in layout}, layout)


def _apply_flat(state: Mapping[str, Any], update: Any, layout: Any):
    result = clone_state(dict(state), next(iter(state.values())).device)
    for name, tensor in _unflatten(update, layout).items():
        result[name].add_(tensor)
    return result


def _dataloader_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Host-side DataLoader options; safe defaults keep determinism for workers=0."""
    training = config.get("training", {})
    num_workers = int(training.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("training.num_workers must be >= 0")
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "drop_last": False,
        "shuffle": False,
    }
    if num_workers > 0:
        prefetch = int(training.get("prefetch_factor", 2))
        if prefetch <= 0:
            raise ValueError("training.prefetch_factor must be positive when num_workers > 0")
        kwargs["prefetch_factor"] = prefetch
        # Default false: break-gate rebuilds many short-lived subset loaders.
        kwargs["persistent_workers"] = bool(training.get("persistent_workers", False))
        # Spawn is safer with NPU/mmap than fork; indices are pre-materialized.
        kwargs["multiprocessing_context"] = "spawn"
    return kwargs


def _loader(
    dataset: Any,
    indices: Any,
    batch_size: int,
    seed: int,
    config: Mapping[str, Any],
    shuffle: bool = True,
):
    import torch
    from torch.utils.data import DataLoader, Subset

    ordered = np.asarray(indices, dtype=np.int64)
    if shuffle:
        permutation = torch.randperm(len(ordered), generator=torch.Generator().manual_seed(seed)).numpy()
        ordered = ordered[permutation]
    return DataLoader(
        Subset(dataset, ordered.tolist()),
        batch_size=batch_size,
        **_dataloader_kwargs(config),
    ), ordered


def _model_builder(config: Mapping[str, Any], train_set: Any):
    model = config["model"]
    input_units = int(getattr(train_set, "input_units", model.get("input_units", 700)))
    classes = int(getattr(train_set, "classes", model.get("classes", 20)))
    return lambda: build_recurrent_lif_shd(
        input_units=input_units,
        hidden_units=int(model["hidden_units"]),
        classes=classes,
        tau=float(model["tau"]),
        threshold=float(model["threshold"]),
        surrogate_beta=float(model["surrogate_beta"]),
    )


def _partitions(train_set: Any, config: Mapping[str, Any]) -> list[np.ndarray]:
    dataset = config["dataset"]
    if dataset["partition"] != "dirichlet" or not math.isclose(
        float(dataset["alpha"]), float(dataset["label_skew_alpha"])
    ):
        raise ValueError("break-gate e2e requires the preregistered label-skew Dirichlet partition")
    return dirichlet_partition(
        train_set.targets,
        int(config["federation"]["clients"]),
        float(dataset["alpha"]),
        int(config["training"]["seed"]),
        min_samples=1,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate(config: Mapping[str, Any]) -> None:
    notes = config.get("notes", {})
    _require(notes.get("break_gate_exploratory") is True, "break-gate e2e requires notes.break_gate_exploratory=true")
    _require(notes.get("formal_stage1a_immutable") is True, "must freeze formal Stage-1A as immutable")
    _require(notes.get("oracle_gate_bypassed") is True, "must explicitly declare oracle gate bypass")
    _require(notes.get("mechanical_e2e_gate_disabled") is True, "mechanical e2e gate must be disabled")
    _require(notes.get("rerun_authorized_stage1a") is not True, "must not authorize formal Stage-1A rerun")
    _require(str(config["paper"].get("fidelity", "")) == REGISTERED_FIDELITY, "fidelity mismatch")
    _require(str(config["paper"]["method"]) == "break_gate_four_method_e2e", "wrong paper.method")
    _require(str(config["stage"]["mode"]) == "end_to_end", "stage.mode must be end_to_end")
    _require(int(config["federation"]["clients"]) == 5, "requires exactly five clients")
    _require(
        tuple(config["federation"]["delay_classes"]) == ("fast", "fast", "medium", "slow", "slow"),
        "delay classes must remain preregistered",
    )
    _require(
        int(config["model"]["tbptt_steps"]) == int(config["model"]["timesteps"]),
        "one full-sequence truncated-BPTT window is required",
    )
    methods = tuple(config.get("compare_methods") or COMPARE_METHODS)
    _require(methods == COMPARE_METHODS, "compare_methods must be the frozen four-method set")
    for retired in ("signed_timing", "connection_level", "separate_long_history"):
        if retired in config.get("eligibility", {}):
            raise ValueError(f"retired eligibility option {retired!r} is not allowed")
    gate = config.get("gate", {})
    _require("two_factor_calibration_manifest_path" in gate, "missing two-factor calibration path")
    _require("third_factor_calibration_manifest_path" in gate, "missing third-factor calibration path")


def _coefficients_from_manifest(manifest: Mapping[str, Any], method: str) -> CorrectionCoefficients:
    methods = manifest.get("methods") or {}
    if method not in methods:
        # Third-factor Oracle aliases proposed under IDEA_C_METHOD keys.
        if method == THIRD_FACTOR_METHOD and IDEA_C_METHOD in methods:
            method = IDEA_C_METHOD
        else:
            raise ValueError(f"calibration manifest missing method {method}")
    selected = methods[method].get("selected_coefficients")
    if not isinstance(selected, Mapping):
        raise ValueError(f"calibration manifest missing selected_coefficients for {method}")
    coefficients = CorrectionCoefficients(
        age=float(selected["age"]),
        drift=float(selected["drift"]),
        eligibility=float(selected["eligibility"]),
        interaction=float(selected["interaction"]),
    )
    coefficients.validate()
    return coefficients


def _load_seed_manifest(path: Path, seed: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"calibration manifest absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "calibration_manifests" in payload:
        matching = [item for item in payload["calibration_manifests"] if int(item.get("seed", -1)) == seed]
        if len(matching) != 1:
            raise ValueError(f"aggregate calibration manifest does not uniquely match seed={seed}")
        payload = matching[0]
    elif "per_seed" in payload and isinstance(payload["per_seed"], list):
        matching = [item for item in payload["per_seed"] if int(item.get("seed", -1)) == seed]
        if len(matching) == 1:
            payload = matching[0]
    manifest_seed = payload.get("seed")
    if manifest_seed is not None and int(manifest_seed) != seed:
        raise ValueError(f"calibration seed mismatch: expected {seed}, got {manifest_seed}")
    if payload.get("calibration_status") != "PASS":
        raise ValueError(f"calibration_status is not PASS for {path}")
    if payload.get("equal_candidate_counts") is not True:
        raise ValueError(f"equal_candidate_counts audit failed for {path}")
    return payload


def load_frozen_coefficients(config: Mapping[str, Any]) -> dict[str, CorrectionCoefficients]:
    """Load per-seed frozen coefficients from independent Oracle calibrations."""
    seed = int(config["training"]["seed"])
    gate = config["gate"]
    two_factor = _load_seed_manifest(Path(gate["two_factor_calibration_manifest_path"]), seed)
    third_factor = _load_seed_manifest(Path(gate["third_factor_calibration_manifest_path"]), seed)
    selected = {
        "drift_age": _coefficients_from_manifest(two_factor, "drift_age"),
        "activity_drift_age": _coefficients_from_manifest(two_factor, "activity_drift_age"),
        TWO_FACTOR_METHOD: _coefficients_from_manifest(two_factor, TWO_FACTOR_METHOD),
        THIRD_FACTOR_METHOD: _coefficients_from_manifest(third_factor, THIRD_FACTOR_METHOD),
    }
    return selected


def _feature_tensors(
    layer: str,
    parameter: Any,
    drift: Any,
    summary: EligibilitySummary,
):
    import torch

    drift = drift.abs()
    if layer == "input.weight":
        eligibility = dequantize_rms_u8(summary.input, device=parameter.device, dtype=parameter.dtype)
    elif layer == "recurrent.weight":
        eligibility = dequantize_rms_u8(summary.recurrent, device=parameter.device, dtype=parameter.dtype)
    else:
        eligibility = torch.zeros(parameter.shape[0], device=parameter.device, dtype=parameter.dtype)
    activity_summary = summary.activity_for_layer(layer)
    activity = dequantize_rms_u8(activity_summary, device=parameter.device, dtype=parameter.dtype)
    return drift, expand_postsynaptic(eligibility, parameter), expand_postsynaptic(activity, parameter)


def _train_local(
    builder: Any,
    base_state: Mapping[str, Any],
    dataset: Any,
    indices: Any,
    device: Any,
    config: Mapping[str, Any],
    seed: int,
    *,
    eligibility_mode: str,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], EligibilitySummary, float, int, int, dict[str, Any]]:
    import torch
    from torch import nn

    if eligibility_mode not in {"two_factor", "third_factor", "none"}:
        raise ValueError(f"unknown eligibility_mode={eligibility_mode}")
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
    summary: EligibilitySummary | None = None
    steps = samples = 0
    for _epoch in range(int(training["local_epochs"])):
        for batch_index, (events, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            # non_blocking helps when host prefetch fills pinned/pageable host
            # buffers while the previous NPU step runs.
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if summary is None and eligibility_mode != "none":
                if eligibility_mode == "two_factor":
                    summary = capture_recurrent_lif_eligibility(
                        model,
                        events,
                        trace_decay=float(config["eligibility"]["trace_decay"]),
                        surrogate_beta=float(config["model"]["surrogate_beta"]),
                    )
                else:
                    summary = capture_third_factor_lif_eligibility(
                        model,
                        events,
                        labels,
                        trace_decay=float(config["eligibility"]["trace_decay"]),
                        surrogate_beta=float(config["model"]["surrogate_beta"]),
                    )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(events), labels)
            loss.backward()
            optimizer.step()
            detached = loss.detach()
            loss_sum = detached if loss_sum is None else loss_sum + detached
            steps += 1
            samples += int(labels.shape[0])
    if steps == 0 or loss_sum is None:
        raise RuntimeError("local training produced no batches")
    if summary is None:
        # Baselines still need an activity/eligibility-shaped summary for payload parity
        # and activity_drift_age. Capture two-factor once after the last local batch path
        # is impossible here; re-run a no-grad capture on a fresh forward of first batch
        # indices is expensive. Instead require eligibility capture for all methods that
        # need features — callers set mode appropriately.
        raise RuntimeError("local training produced no eligibility summary")
    # Single host sync for the mean local loss instead of per-batch .cpu().
    mean_loss = float((loss_sum / steps).item())
    budget = {
        "seed": int(seed),
        "local_epochs": int(training["local_epochs"]),
        "max_batches": None if max_batches is None else int(max_batches),
        "steps": int(steps),
        "samples": int(samples),
        "ordered_indices_sha256": __import__("hashlib").sha256(ordered_indices.tobytes()).hexdigest(),
        "eligibility_mode": eligibility_mode,
    }
    return (
        clone_state(model.state_dict(), next(iter(base_state.values())).device),
        summary,
        mean_loss,
        steps,
        samples,
        budget,
    )


def _evaluate(
    model: Any,
    dataset: Any,
    device: Any,
    batch_size: int,
    config: Mapping[str, Any],
    max_batches: int | None = None,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total = 0
    loss_sum = None
    correct_sum = None
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, **_dataloader_kwargs(config))
        for batch_index, (events, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(events)
            batch_loss = criterion(logits, labels)
            batch_correct = (logits.argmax(1) == labels).sum()
            loss_sum = batch_loss if loss_sum is None else loss_sum + batch_loss
            correct_sum = batch_correct if correct_sum is None else correct_sum + batch_correct
            total += int(labels.shape[0])
    if total == 0 or loss_sum is None or correct_sum is None:
        raise RuntimeError("empty evaluation")
    # One host sync at the end of eval instead of per-batch transfers.
    return float(correct_sum.item()) / total, float(loss_sum.item()) / total


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
    raise ValueError(f"unsupported break-gate method: {method}")


def _end_to_end(
    config: Mapping[str, Any],
    train_set: Any,
    test_set: Any,
    partitions: list[np.ndarray],
    builder: Any,
    device: Any,
    smoke: bool,
) -> list[dict[str, Any]]:
    selected_coefficients = load_frozen_coefficients(config)
    seed = int(config["training"]["seed"])
    rounds = 2 if smoke else int(config["training"]["rounds"])
    records: list[dict[str, Any]] = []
    reference_model = builder().to(device)
    reference_state = clone_state(reference_model.state_dict(), device)
    for method in COMPARE_METHODS:
        model = builder().to(device)
        state = clone_state(reference_state, device)
        model.load_state_dict(state)
        layout = _layout(model)
        client_states = [clone_state(state, device) for _ in range(5)]
        client_versions = [0] * 5
        ledger = CostLedger()
        eligibility_mode = _eligibility_mode_for_method(method)
        method_coefficients = selected_coefficients[method]
        for server_update in range(rounds):
            client_id = SCHEDULE[server_update % len(SCHEDULE)]
            base = client_states[client_id]
            local, summary, loss, steps, samples, _training_audit = _train_local(
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
                if layer in {"input.weight", "recurrent.weight"}:
                    drift, eligibility, activity = _feature_tensors(
                        layer, parameter, state[layer] - base[layer], summary
                    )
                    layer_age_rate = (
                        method_coefficients.age if layer == "input.weight" else method_coefficients.drift
                    )
                    correction = correction_for_method(
                        _correction_method_name(method),
                        age=age,
                        drift=drift,
                        eligibility=eligibility,
                        activity=activity,
                        layer_age_rate=layer_age_rate,
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
            state = _apply_flat(state, _flatten(corrected, layout), layout)
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
                * int(config["model"]["hidden_units"])
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
                    "eligibility_input_min": summary.input.minimum,
                    "eligibility_input_max": summary.input.maximum,
                    "eligibility_recurrent_min": summary.recurrent.minimum,
                    "eligibility_recurrent_max": summary.recurrent.maximum,
                    "correction_min": correction_min,
                    "correction_max": correction_max,
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


def descriptive_e2e_summary(
    seed_rows: Sequence[Mapping[str, Any]],
    *,
    proposed_method: str,
    target_accuracy: float = 0.70,
) -> dict[str, Any]:
    """Descriptive ACC comparison only — no mechanical pass upgrades formal state."""
    if proposed_method not in {TWO_FACTOR_METHOD, THIRD_FACTOR_METHOD}:
        raise ValueError(f"proposed_method must be two-factor or third-factor, got {proposed_method}")
    if len(seed_rows) != 5 or len({int(row["seed"]) for row in seed_rows}) != 5:
        raise ValueError("descriptive summary expects exactly five distinct seeds")
    auc_diffs = []
    update_reductions = []
    final_diffs = []
    activity_diffs = []
    per_seed = []
    for row in seed_rows:
        methods = row["methods"]
        for method in ("drift_age", "activity_drift_age", proposed_method):
            if method not in methods:
                raise ValueError(f"seed {row['seed']} missing method {method}")
        full = methods[proposed_method]
        drift = methods["drift_age"]
        activity = methods["activity_drift_age"]
        drift_auc = float(drift.get("accuracy_auc", trapezoid_auc(drift["accuracy_curve"])))
        full_auc = float(full.get("accuracy_auc", trapezoid_auc(full["accuracy_curve"])))
        auc_rel = (full_auc - drift_auc) / max(abs(drift_auc), 1e-12)
        drift_updates = float(drift.get("updates_to_target", math.inf))
        full_updates = float(full.get("updates_to_target", math.inf))
        reduction = (
            (drift_updates - full_updates) / max(drift_updates, 1.0)
            if math.isfinite(drift_updates)
            else 0.0
        )
        final_diff = float(full["final_accuracy"]) - float(drift["final_accuracy"])
        activity_diff = float(full["final_accuracy"]) - float(activity["final_accuracy"])
        auc_diffs.append(auc_rel)
        update_reductions.append(reduction)
        final_diffs.append(final_diff)
        activity_diffs.append(activity_diff)
        per_seed.append(
            {
                "seed": int(row["seed"]),
                "relative_auc_vs_drift": auc_rel,
                "update_reduction_vs_drift": reduction,
                "final_acc_vs_drift": final_diff,
                "final_acc_vs_activity": activity_diff,
                "final_accuracy": float(full["final_accuracy"]),
                "drift_final_accuracy": float(drift["final_accuracy"]),
                "activity_final_accuracy": float(activity["final_accuracy"]),
            }
        )
    return {
        "stage": "break_gate_e2e_descriptive",
        "proposed_method": proposed_method,
        "mechanical_gate_disabled": True,
        "formal_stage1a_immutable": True,
        "target_accuracy": float(target_accuracy),
        "statistics": {
            "mean_relative_auc_improvement_vs_drift": float(np.mean(auc_diffs)),
            "mean_update_reduction_vs_drift": float(np.mean(update_reductions)),
            "mean_final_accuracy_difference_vs_drift": float(np.mean(final_diffs)),
            "mean_final_accuracy_difference_vs_activity": float(np.mean(activity_diffs)),
            "seeds_auc_positive_vs_drift": int(sum(value > 0 for value in auc_diffs)),
            "seeds_final_positive_vs_drift": int(sum(value > 0 for value in final_diffs)),
            "seeds_final_positive_vs_activity": int(sum(value > 0 for value in activity_diffs)),
        },
        "per_seed": per_seed,
        "innovation_death_hint": (
            "no_acc_advantage_vs_drift"
            if float(np.mean(final_diffs)) <= 0 and float(np.mean(auc_diffs)) <= 0
            else "acc_or_auc_mean_nonnegative_descriptive_only"
        ),
    }


def seed_rows_from_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    target_accuracy: float = 0.70,
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for row in records:
        grouped.setdefault(int(row["seed"]), {}).setdefault(str(row["method"]), []).append(dict(row))
    seed_rows = []
    for seed, methods in sorted(grouped.items()):
        payload = {}
        for method in COMPARE_METHODS:
            method_rows = sorted(methods.get(method, []), key=lambda item: int(item["server_update"]))
            if not method_rows:
                raise ValueError(f"seed {seed} missing method {method}")
            curve = [float(item["test_accuracy"]) for item in method_rows]
            updates_to_target = next(
                (
                    int(item["server_update"]) + 1
                    for item in method_rows
                    if float(item["test_accuracy"]) >= target_accuracy
                ),
                float("inf"),
            )
            payload[method] = {
                "accuracy_curve": curve,
                "accuracy_auc": trapezoid_auc(curve),
                "final_accuracy": curve[-1],
                "updates_to_target": updates_to_target,
            }
        seed_rows.append({"seed": seed, "methods": payload})
    return seed_rows


def run_break_gate_e2e(
    config: dict[str, Any],
    data_root: str | Path,
    device_spec: str,
    *,
    smoke: bool = False,
) -> Path:
    """Run one seed of the four-method break-gate e2e track."""
    _validate(config)
    # Touch coefficients early so missing manifests fail before SHD load.
    coefficients = load_frozen_coefficients(config)
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    device_info = resolve_device(device_spec)
    device = activate_device(device_info)
    dataset_cfg = config["dataset"]
    cache_dir = dataset_cfg.get("cache_dir")
    require_cache = bool(dataset_cfg.get("require_cache", False))
    train_set, test_set, metadata = load_shd(
        data_root,
        timesteps=int(config["model"]["timesteps"]),
        duration=float(dataset_cfg.get("duration_seconds", 1.0)),
        binary=bool(dataset_cfg.get("binary_bins", True)),
        cache_dir=cache_dir,
        require_cache=require_cache,
    )
    if require_cache and metadata.get("data_backend") != "npy_mmap_uint8":
        raise RuntimeError(
            f"require_cache=true but data_backend={metadata.get('data_backend')!r}; "
            "expected npy_mmap_uint8 binned cache"
        )
    partitions = _partitions(train_set, config)
    builder = _model_builder(config, train_set)
    output = result_dir(config)
    if smoke:
        output = output.with_name(f"{output.name}__smoke")
    output.mkdir(parents=True, exist_ok=True)
    code_identity = _code_commit()
    resolved = copy.deepcopy(config)
    resolved["runtime"] = {
        "device": str(device),
        "smoke": smoke,
        "dataset_metadata": metadata,
        "code_identity": code_identity,
        "protocol_identity": protocol_identity(config, code_identity),
        "track": REGISTERED_TRACK,
        "oracle_gate_bypassed": True,
        "mechanical_e2e_gate_disabled": True,
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
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
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


def run_break_gate_e2e_file(
    config_path: str | Path,
    data_root: str | Path,
    device_spec: str = "auto",
    *,
    smoke: bool = False,
) -> Path:
    return run_break_gate_e2e(load_config(config_path), data_root, device_spec, smoke=smoke)
