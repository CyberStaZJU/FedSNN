"""Independent multi-layer Oracle for third-factor eligibility.

This track is deliberately separate from the frozen Idea C Stage-1A archive
(``STOP_ORACLE_FAIL`` / ``SKIPPED_BY_ORACLE_GATE``).  Active Idea C eligibility
options are only:

1. two-factor postsynaptic RMS (formal Stage-1A), and
2. the same RMS modulated by a local-error third factor (this module).

No other mechanism switches (connection-level, signed timing, separate long
history, etc.) are part of the active option set.
"""

from __future__ import annotations

from dataclasses import asdict
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .config import load_config, result_dir
from .device import activate_device, resolve_device, seed_everything
from .idea_c_stage1a import (
    COEFFICIENT_GRID_SCALES,
    CostLedger,
    CorrectionCoefficients,
    EligibilitySummary,
    IDEA_C_METHOD,
    ORACLE_BASELINES,
    OraclePair,
    coefficient_grid,
    dequantize_rms_u8,
    eligibility_rms_from_factors,
    evaluate_oracle_pair,
    expand_postsynaptic,
    oracle_verdict,
    protocol_identity,
    quantize_rms_u8,
    select_calibrated_coefficients,
)
from .partition import dirichlet_partition
from .protocol import clone_state
from .shd import load_shd
from .shd_model import build_recurrent_lif_shd
from .train import _append_jsonl, _code_commit

THIRD_FACTOR_METHOD = "third_factor_eligibility_informed_staleness"
REGISTERED_TRACK = "idea_c_third_factor_oracle_v1"
REGISTERED_ESTIMATOR = "third_factor_modulated_two_factor_rms"
REGISTERED_GRANULARITY = "postsynaptic_neuron_input_recurrent_split"
REGISTERED_SEEDS = (2, 3, 4, 5, 6)
REGISTERED_TRACE_DECAY = 0.9
REGISTERED_SUMMARY_BITS = 8


def capture_third_factor_lif_eligibility(
    model: Any,
    inputs: Any,
    labels: Any,
    *,
    trace_decay: float = 0.9,
    surrogate_beta: float | None = None,
) -> EligibilitySummary:
    """Capture postsynaptic RMS eligibility modulated by local-error third factor.

    The Stage-1A two-factor pre/post structure is retained for payload parity.
    The only change is that the postsynaptic surrogate is scaled by the absolute
    local-error third factor ``|(one_hot - softmax) @ readout.weight|`` before the
    RMS reduction.  Fresh updates never enter this feature.
    """
    import math as _math
    import torch

    labels = torch.as_tensor(labels, device=inputs.device, dtype=torch.long)
    if labels.ndim != 1 or labels.shape[0] != inputs.shape[0]:
        raise ValueError("labels must be a rank-one class index per sample")

    input_currents: list[Any] = []
    recurrent_currents: list[Any] = []
    recurrent_inputs: list[Any] = []

    def input_hook(_module, _args, output):
        input_currents.append(output)

    def recurrent_hook(_module, args, output):
        recurrent_inputs.append(args[0])
        recurrent_currents.append(output)

    handles = [
        model.input.register_forward_hook(input_hook),
        model.recurrent.register_forward_hook(recurrent_hook),
    ]
    try:
        logits = model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    timesteps = int(inputs.shape[1])
    if len(input_currents) != timesteps or len(recurrent_currents) != timesteps:
        raise RuntimeError("SHD model hooks did not observe exactly one call per timestep")

    probabilities = torch.softmax(logits.detach(), dim=-1)
    targets = torch.nn.functional.one_hot(labels, num_classes=logits.shape[-1]).to(probabilities.dtype)
    # Absolute third factor keeps the Stage-1A nonnegative RMS codec/correction.
    third_factor = ((targets - probabilities) @ model.readout.weight.detach()).abs()

    membrane = torch.zeros_like(input_currents[0])
    post_steps = []
    spike_steps = []
    beta = float(surrogate_beta if surrogate_beta is not None else getattr(model, "surrogate_beta", 2.0))
    for input_current, recurrent_current in zip(input_currents, recurrent_currents):
        membrane = membrane + (input_current + recurrent_current - membrane) / float(model.tau)
        centered = membrane - float(model.threshold)
        post = (beta / 2.0) / (1.0 + (_math.pi * beta * centered / 2.0).square())
        spikes = (centered >= 0).to(membrane.dtype)
        post_steps.append(post * third_factor)
        spike_steps.append(spikes)
        membrane = membrane * (1.0 - spikes.detach())

    post = torch.stack(post_steps, dim=1)
    recurrent_pre = torch.stack(recurrent_inputs, dim=1)
    input_rms, recurrent_rms = eligibility_rms_from_factors(
        inputs, recurrent_pre, post, trace_decay=trace_decay
    )
    spikes = torch.stack(spike_steps, dim=1).to(torch.float64)
    recurrent_pre = recurrent_pre.to(torch.float64)
    activity_input = spikes.mean(dim=(0, 1))
    activity_recurrent = (spikes * recurrent_pre.abs()).mean(dim=(0, 1))
    return EligibilitySummary(
        input=quantize_rms_u8(input_rms),
        recurrent=quantize_rms_u8(recurrent_rms),
        samples=int(inputs.shape[0]),
        timesteps=timesteps,
        activity_input=quantize_rms_u8(activity_input),
        activity_recurrent=quantize_rms_u8(activity_recurrent),
    )


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


def _loader(dataset: Any, indices: Any, batch_size: int, seed: int, shuffle: bool = True):
    import torch
    from torch.utils.data import DataLoader, Subset

    ordered = np.asarray(indices, dtype=np.int64)
    if shuffle:
        permutation = torch.randperm(len(ordered), generator=torch.Generator().manual_seed(seed)).numpy()
        ordered = ordered[permutation]
    return DataLoader(
        Subset(dataset, ordered.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
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


def _train_local(
    builder: Any,
    base_state: Mapping[str, Any],
    dataset: Any,
    indices: Any,
    device: Any,
    config: Mapping[str, Any],
    seed: int,
    *,
    max_batches: int | None = None,
    capture_summary: bool = True,
) -> tuple[dict[str, Any], EligibilitySummary | None, float, int, int, dict[str, Any]]:
    import torch
    from torch import nn

    model = builder().to(device)
    model.load_state_dict(base_state)
    training = config["training"]
    loader, ordered_indices = _loader(dataset, indices, int(training["batch_size"]), seed)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training["learning_rate"]),
        momentum=float(training.get("momentum", 0.0)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()
    losses = []
    summary = None
    steps = samples = 0
    for _epoch in range(int(training["local_epochs"])):
        for batch_index, (events, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            events, labels = events.to(device), labels.to(device)
            if capture_summary and summary is None:
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
            losses.append(float(loss.detach().cpu()))
            steps += 1
            samples += labels.numel()
    if not losses:
        raise RuntimeError("local training produced no batches")
    if capture_summary and summary is None:
        raise RuntimeError("local training produced no eligibility summary")
    budget = {
        "seed": int(seed),
        "local_epochs": int(training["local_epochs"]),
        "max_batches": None if max_batches is None else int(max_batches),
        "steps": int(steps),
        "samples": int(samples),
        "ordered_indices_sha256": __import__("hashlib").sha256(ordered_indices.tobytes()).hexdigest(),
    }
    return (
        clone_state(model.state_dict(), next(iter(base_state.values())).device),
        summary,
        float(np.mean(losses)),
        steps,
        samples,
        budget,
    )


def _partitions(train_set: Any, config: Mapping[str, Any]) -> list[np.ndarray]:
    dataset = config["dataset"]
    if dataset["partition"] != "dirichlet" or not math.isclose(float(dataset["alpha"]), float(dataset["label_skew_alpha"])):
        raise ValueError("third-factor Oracle requires the preregistered label-skew Dirichlet partition")
    return dirichlet_partition(
        train_set.targets,
        int(config["federation"]["clients"]),
        float(dataset["alpha"]),
        int(config["training"]["seed"]),
        min_samples=1,
    )


def _correction_coefficients(config: Mapping[str, Any]) -> CorrectionCoefficients:
    values = config["correction"]
    coefficients = CorrectionCoefficients(
        age=float(values["age"]),
        drift=float(values["drift"]),
        eligibility=float(values["eligibility"]),
        interaction=float(values["interaction"]),
    )
    coefficients.validate()
    return coefficients


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


def _split_client_indices(indices: np.ndarray, seed: int, client_id: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ordered = np.asarray(indices, dtype=np.int64)
    if len(ordered) < 2:
        raise ValueError("each client needs at least two samples for calibration/heldout")
    rng = np.random.default_rng(seed + 30_000 + client_id)
    shuffled = ordered[rng.permutation(len(ordered))]
    calibration_count = max(1, len(shuffled) // 2)
    calibration = np.sort(shuffled[:calibration_count])
    heldout = np.sort(shuffled[calibration_count:])
    if len(heldout) == 0 or np.intersect1d(calibration, heldout).size:
        raise RuntimeError("calibration and heldout indices must be non-empty and disjoint")
    def digest(values):
        return __import__("hashlib").sha256(values.tobytes()).hexdigest()
    return calibration, heldout, {
        "client_id": client_id,
        "calibration_count": int(len(calibration)),
        "heldout_count": int(len(heldout)),
        "calibration_indices_sha256": digest(calibration),
        "heldout_indices_sha256": digest(heldout),
        "disjoint": True,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate(config: Mapping[str, Any]) -> None:
    notes = config.get("notes", {})
    if notes.get("formal_stage1a_immutable") is not True:
        raise ValueError("third-factor Oracle must explicitly freeze formal Stage-1A as immutable")
    if notes.get("rerun_authorized_stage1a") is True:
        raise ValueError("third-factor Oracle must not authorize Stage-1A rerun")
    if notes.get("end_to_end_authorized") is True:
        raise ValueError("end-to-end is gated; third-factor Oracle configs must keep it unauthorized until PASS")
    if config["paper"]["method"] != THIRD_FACTOR_METHOD:
        raise ValueError("wrong method for third-factor multi-layer Oracle")
    if str(config["paper"].get("fidelity", "")) != REGISTERED_TRACK:
        raise ValueError("third-factor Oracle fidelity/track mismatch")
    if int(config["federation"]["clients"]) != 5:
        raise ValueError("protocol requires exactly five clients")
    if tuple(config["federation"]["delay_classes"]) != ("fast", "fast", "medium", "slow", "slow"):
        raise ValueError("delay classes must remain preregistered fast/fast/medium/slow/slow")
    if int(config["model"]["tbptt_steps"]) != int(config["model"]["timesteps"]):
        raise ValueError("one full-sequence truncated-BPTT window is required")
    eligibility = config["eligibility"]
    _require(str(eligibility["estimator"]) == REGISTERED_ESTIMATOR, "eligibility estimator mismatch")
    _require(str(eligibility["granularity"]) == REGISTERED_GRANULARITY, "eligibility granularity mismatch")
    _require(int(eligibility["summary_bits"]) == REGISTERED_SUMMARY_BITS, "summary bits mismatch")
    _require(math.isclose(float(eligibility["trace_decay"]), REGISTERED_TRACE_DECAY), "trace decay mismatch")
    _require(eligibility.get("third_factor") is True, "third factor must be enabled")
    # Retired mechanism keys must not re-enter active configs.
    for retired in ("signed_timing", "connection_level", "separate_long_history", "timing_rule",
                    "pre_trace_decay", "post_trace_decay", "history_decay"):
        if retired in eligibility:
            raise ValueError(
                f"retired eligibility option {retired!r} is not part of the active "
                "two-factor / third-factor option set"
            )
    oracle = config["oracle"]
    for key in ("context_batches", "calibration_batches", "heldout_batches"):
        if int(oracle.get(key, 0)) != 1:
            raise ValueError(f"oracle.{key} must equal 1")
    if tuple(float(value) for value in oracle.get("coefficient_grid_scales", ())) != COEFFICIENT_GRID_SCALES:
        raise ValueError("requires the frozen common three-candidate coefficient grid")
    if int(config["training"]["seed"]) not in REGISTERED_SEEDS:
        raise ValueError(f"seed must be one of {REGISTERED_SEEDS}")
    if str(config["stage"]["mode"]) != "oracle":
        raise ValueError("this entry only runs multi-layer Oracle; end-to-end requires a later PASS gate")
    _correction_coefficients(config)


def _remap_method_keys(row: dict[str, Any]) -> dict[str, Any]:
    """Expose the proposed method under both track-local and Stage-1A gate keys."""
    methods = dict(row["methods"])
    if IDEA_C_METHOD not in methods and THIRD_FACTOR_METHOD in methods:
        methods[IDEA_C_METHOD] = methods[THIRD_FACTOR_METHOD]
    if THIRD_FACTOR_METHOD not in methods and IDEA_C_METHOD in methods:
        methods[THIRD_FACTOR_METHOD] = methods[IDEA_C_METHOD]
    row = dict(row)
    row["methods"] = methods
    return row


def _oracle_rows(
    config: Mapping[str, Any], train_set: Any, partitions: list[np.ndarray], builder: Any, device: Any, smoke: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], CostLedger, dict[str, Any]]:
    seed = int(config["training"]["seed"])
    base_model = builder().to(device)
    initial = clone_state(base_model.state_dict(), device)
    layout = _layout(base_model)
    calibration_rows: list[dict[str, Any]] = []
    heldout_pairs: list[OraclePair] = []
    ledger = CostLedger()
    clients = range(1 if smoke else 5)
    bucket_ages = {"light": 1, "medium": 2, "heavy": 3}
    base_coefficients = _correction_coefficients(config)
    candidates = coefficient_grid(base_coefficients)
    split_audit = []
    snapshot_count = 0

    for client_id in clients:
        calibration_indices, heldout_indices, client_audit = _split_client_indices(
            partitions[client_id], seed, client_id
        )
        split_audit.append(client_audit)
        context_client_ids = tuple(index for index in range(5) if index != client_id)
        snapshots = [initial]
        evolving = initial
        for version in range(1, 4):
            context_client = context_client_ids[(version - 1) % len(context_client_ids)]
            local, _summary, _loss, _steps, _samples, _audit = _train_local(
                builder, evolving, train_set, partitions[context_client], device, config,
                seed + 10_000 + client_id * 100 + version,
                max_batches=int(config["oracle"]["context_batches"]),
                capture_summary=False,
            )
            evolving = _apply_flat(evolving, _state_delta(local, evolving, layout), layout)
            snapshots.append(evolving)
        snapshot_count += len(snapshots)

        for split_name, split_indices, max_batches in (
            ("calibration", calibration_indices, int(config["oracle"]["calibration_batches"])),
            ("heldout", heldout_indices, int(config["oracle"]["heldout_batches"])),
        ):
            pair_seed = seed + (20_000 if split_name == "calibration" else 40_000) + client_id
            stale_local, summary, _loss, steps, samples, stale_audit = _train_local(
                builder, snapshots[0], train_set, split_indices, device, config,
                pair_seed, max_batches=max_batches, capture_summary=True,
            )
            if summary is None:
                raise RuntimeError("stale training missing third-factor eligibility summary")
            stale_delta = _state_delta(stale_local, snapshots[0], layout)
            ledger.add_summary(summary)
            if summary.activity_payload_bits != summary.payload_bits:
                raise RuntimeError("activity and eligibility payloads are not matched")
            ledger.local_training_steps += steps
            ledger.eligibility_factor_multiply_adds += (
                samples
                * int(config["model"]["timesteps"])
                * int(config["model"]["hidden_units"])
                * (int(config["model"].get("input_units", 700)) + int(config["model"]["hidden_units"]))
            )
            stale_by_layer = _unflatten(stale_delta, layout)
            for bucket, age in bucket_ages.items():
                fresh_local, _fresh_summary, _fresh_loss, fresh_steps, _fresh_samples, fresh_audit = _train_local(
                    builder, snapshots[age], train_set, split_indices, device, config,
                    pair_seed, max_batches=max_batches, capture_summary=False,
                )
                paired_fields = ("seed", "local_epochs", "max_batches", "steps", "samples", "ordered_indices_sha256")
                identical_pairing = all(stale_audit[field] == fresh_audit[field] for field in paired_fields)
                pairing_audit = {
                    "identical_pairing": identical_pairing,
                    "split": split_name,
                    "seed": pair_seed,
                    "ordered_indices_sha256": stale_audit["ordered_indices_sha256"],
                    "local_epochs": stale_audit["local_epochs"],
                    "max_batches": stale_audit["max_batches"],
                    "steps": stale_audit["steps"],
                    "samples": stale_audit["samples"],
                }
                if not identical_pairing:
                    raise RuntimeError("stale/fresh local training pairing diverged")
                ledger.oracle_fresh_training_steps += fresh_steps
                fresh_by_layer = _unflatten(_state_delta(fresh_local, snapshots[age], layout), layout)
                for layer in ("input.weight", "recurrent.weight"):
                    parameter = stale_by_layer[layer]
                    drift = snapshots[age][layer] - snapshots[0][layer]
                    drift_feature, eligibility, activity = _feature_tensors(layer, parameter, drift, summary)
                    pair = OraclePair(
                        pair_id=f"seed={seed}/client={client_id}/{split_name}/{bucket}/{layer}",
                        seed=seed, client_id=client_id, layer=layer,
                        staleness_bucket=bucket, age=age,
                        stale_update=parameter, fresh_update=fresh_by_layer[layer],
                        parameter_drift=drift_feature, eligibility=eligibility, activity=activity,
                        pairing_audit=pairing_audit, context_client_ids=context_client_ids,
                    )
                    if split_name == "calibration":
                        for candidate_index, candidate in enumerate(candidates):
                            row = evaluate_oracle_pair(pair, coefficients=candidate)
                            row = _remap_method_keys(row)
                            row["methods"][THIRD_FACTOR_METHOD] = row["methods"][IDEA_C_METHOD]
                            row["data_split"] = "calibration"
                            row["candidate"] = {
                                "index": candidate_index,
                                "scale": COEFFICIENT_GRID_SCALES[candidate_index],
                                "coefficients": asdict(candidate),
                            }
                            row["eligibility_estimator"] = REGISTERED_ESTIMATOR
                            calibration_rows.append(row)
                    else:
                        heldout_pairs.append(pair)
                    ledger.correction_element_ops += parameter.numel() * len(ORACLE_BASELINES) * 8

    selected, selection_audit = select_calibrated_coefficients(calibration_rows)
    heldout_rows = []
    for pair in heldout_pairs:
        row = evaluate_oracle_pair(pair, coefficients_by_method=selected)
        row = _remap_method_keys(row)
        row["methods"][THIRD_FACTOR_METHOD] = row["methods"][IDEA_C_METHOD]
        row["data_split"] = "heldout"
        row["calibration_audit"] = "PASS"
        row["eligibility_estimator"] = REGISTERED_ESTIMATOR
        heldout_rows.append(row)
    ledger.model_snapshot_bytes = sum(
        value.numel() * value.element_size() for value in initial.values()
    ) * snapshot_count
    manifest = {
        "seed": seed,
        "track": REGISTERED_TRACK,
        "method": THIRD_FACTOR_METHOD,
        "eligibility_estimator": REGISTERED_ESTIMATOR,
        "calibration_status": "PASS",
        "selection_metric": "calibration_mean_relative_l2_only",
        "split_audit": split_audit,
        "calibration_pair_count": len(calibration_rows),
        "heldout_pair_count": len(heldout_rows),
        "common_candidate_scales": list(COEFFICIENT_GRID_SCALES),
        "equal_candidate_counts": len({item["candidate_count"] for item in selection_audit.values()}) == 1,
        "methods": selection_audit,
        "formal_stage1a_boundary": {
            "formal_terminal_state": "STOP_ORACLE_FAIL",
            "end_to_end_status": "SKIPPED_BY_ORACLE_GATE",
            "archive_modified": False,
            "rerun_authorized": False,
        },
    }
    return calibration_rows, heldout_rows, ledger, manifest


def run_third_factor_oracle(
    config: dict[str, Any],
    data_root: str | Path,
    device_spec: str,
    *,
    smoke: bool = False,
) -> Path:
    """Run the independent third-factor multi-layer held-out Oracle."""
    _validate(config)
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    device_info = resolve_device(device_spec)
    device = activate_device(device_info)
    train_set, test_set, metadata = load_shd(
        data_root,
        timesteps=int(config["model"]["timesteps"]),
        duration=float(config["dataset"].get("duration_seconds", 1.0)),
        binary=bool(config["dataset"].get("binary_bins", True)),
    )
    del test_set  # Oracle only; evaluation deferred to a future end-to-end gate.
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
    }
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    calibration_rows, heldout_rows, ledger, manifest = _oracle_rows(
        config, train_set, partitions, builder, device, smoke
    )
    calibration_path = output / ("calibration_pairs.smoke.jsonl" if smoke else "calibration_pairs.jsonl")
    heldout_path = output / ("oracle_pairs.smoke.jsonl" if smoke else "oracle_pairs.jsonl")
    for path, rows in ((calibration_path, calibration_rows), (heldout_path, heldout_rows)):
        if path.exists():
            path.unlink()
        for row in rows:
            _append_jsonl(path, row)
    manifest_path = output / ("calibration_manifest.smoke.json" if smoke else "calibration_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    cost_path = output / ("cost.smoke.json" if smoke else "cost.json")
    cost_path.write_text(json.dumps(ledger.as_record(), indent=2) + "\n", encoding="utf-8")
    if not smoke and len({row["seed"] for row in heldout_rows}) == 5:
        verdict = oracle_verdict(heldout_rows).to_dict()
        verdict["track"] = REGISTERED_TRACK
        verdict["method"] = THIRD_FACTOR_METHOD
        verdict["formal_stage1a_boundary"] = manifest["formal_stage1a_boundary"]
        (output / "oracle_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    return output


def run_third_factor_oracle_file(
    config_path: str | Path,
    data_root: str | Path,
    device_spec: str = "auto",
    *,
    smoke: bool = False,
) -> Path:
    return run_third_factor_oracle(load_config(config_path), data_root, device_spec, smoke=smoke)
