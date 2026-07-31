"""Frozen Idea C stage-1A primitives for eligibility-informed staleness correction.

This module is intentionally independent of experiment I/O.  It implements the
8-bit postsynaptic summaries, bounded monotone correction, leakage-safe oracle
records, cost accounting, and the preregistered oracle/end-to-end verdicts.
Archived Oracle metrics remain tied to their recorded code identity; final
acceptance is read-only, and later hardening cannot retroactively support a
positive experimental claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


IDEA_C_METHOD = "eligibility_informed_staleness"
FINAL_ORACLE_TERMINAL_STATE = "STOP_ORACLE_FAIL"
SKIPPED_END_TO_END_STATE = "SKIPPED_BY_ORACLE_GATE"
FINAL_ACCEPTANCE_EVIDENCE = "saved_artifacts_only"
FINAL_ACCEPTANCE_REVIEW_MODE = "manifest_and_hashes_read_only"
ORACLE_BASELINES = (
    "no_correction",
    "scalar_age",
    "per_layer_age",
    "drift_age",
    "activity_drift_age",
    IDEA_C_METHOD,
    "eligibility_age_no_drift",
    "eligibility_drift_age_no_interaction",
    "shuffled_eligibility_drift_age",
)
CALIBRATED_METHODS = tuple(method for method in ORACLE_BASELINES if method != "no_correction")
COEFFICIENT_GRID_SCALES = (0.5, 1.0, 1.5)
DEPLOYMENT_FEATURES = frozenset(
    {"age", "layer", "parameter_drift", "eligibility_input", "eligibility_recurrent"}
)
FORBIDDEN_DEPLOYMENT_FEATURES = frozenset(
    {"fresh_update", "fresh_delta", "oracle_target", "future_model", "future_gradient"}
)


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class QuantizedNeuronSummary:
    """Per-postsynaptic-neuron 8-bit affine quantization payload."""

    values: tuple[int, ...]
    minimum: float
    maximum: float
    bits_per_value: int = 8

    @property
    def payload_bits(self) -> int:
        # Two float32 endpoints plus the uint8 values.  Shape/layer identifiers
        # are charged by CostLedger rather than hidden in this object.
        return 64 + self.bits_per_value * len(self.values)


def quantize_rms_u8(values: Any) -> QuantizedNeuronSummary:
    """Quantize a finite nonnegative neuron vector to a real uint8 payload."""
    import torch

    tensor = torch.as_tensor(values).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if tensor.numel() == 0 or not torch.isfinite(tensor).all() or bool((tensor < 0).any()):
        raise ValueError("eligibility RMS must be a non-empty finite nonnegative vector")
    minimum = float(tensor.min())
    maximum = float(tensor.max())
    if maximum == minimum:
        encoded = torch.zeros_like(tensor, dtype=torch.uint8)
    else:
        encoded = torch.round((tensor - minimum) * (255.0 / (maximum - minimum))).clamp(0, 255).to(torch.uint8)
    return QuantizedNeuronSummary(tuple(int(x) for x in encoded.tolist()), minimum, maximum)


def dequantize_rms_u8(summary: QuantizedNeuronSummary, *, device: Any = "cpu", dtype: Any = None):
    import torch

    if summary.bits_per_value != 8 or not summary.values:
        raise ValueError("only non-empty 8-bit summaries are supported")
    if not math.isfinite(summary.minimum) or not math.isfinite(summary.maximum) or summary.minimum > summary.maximum:
        raise ValueError("invalid summary range")
    target_dtype = dtype or torch.float32
    encoded = torch.tensor(summary.values, dtype=torch.float64, device=device)
    if summary.maximum == summary.minimum:
        decoded = torch.full_like(encoded, summary.minimum)
    else:
        decoded = summary.minimum + encoded * ((summary.maximum - summary.minimum) / 255.0)
    return decoded.to(dtype=target_dtype)


@dataclass(frozen=True)
class EligibilitySummary:
    input: QuantizedNeuronSummary
    recurrent: QuantizedNeuronSummary
    samples: int
    timesteps: int
    activity_input: QuantizedNeuronSummary | None = None
    activity_recurrent: QuantizedNeuronSummary | None = None

    @property
    def payload_bits(self) -> int:
        return self.input.payload_bits + self.recurrent.payload_bits

    @property
    def activity_payload_bits(self) -> int:
        if self.activity_input is None or self.activity_recurrent is None:
            raise ValueError("both block-specific activity payloads are required")
        return self.activity_input.payload_bits + self.activity_recurrent.payload_bits

    def activity_for_layer(self, layer: str) -> QuantizedNeuronSummary:
        if layer == "input.weight" and self.activity_input is not None:
            return self.activity_input
        if layer == "recurrent.weight" and self.activity_recurrent is not None:
            return self.activity_recurrent
        raise ValueError(f"block-specific activity payload is unavailable for {layer}")


def eligibility_rms_from_factors(
    input_presynaptic: Any,
    recurrent_presynaptic: Any,
    postsynaptic: Any,
    *,
    trace_decay: float,
) -> tuple[Any, Any]:
    """Compute two-factor eligibility RMS, reset at every sample boundary.

    Inputs are ``[batch,time,input]``, ``[batch,time,hidden]`` and
    ``[batch,time,hidden]``.  Presynaptic traces are local to each sample and
    are multiplied by the postsynaptic surrogate factor.  RMS is then reduced
    over sample, time, and presynaptic coordinates, leaving one value per
    postsynaptic neuron for input and recurrent blocks separately.
    """
    import torch

    x = torch.as_tensor(input_presynaptic)
    r = torch.as_tensor(recurrent_presynaptic)
    post = torch.as_tensor(postsynaptic)
    if x.ndim != 3 or r.ndim != 3 or post.ndim != 3:
        raise ValueError("eligibility factors must be rank-three [batch,time,units]")
    if x.shape[:2] != post.shape[:2] or r.shape != post.shape:
        raise ValueError("eligibility factor batch/time/hidden dimensions disagree")
    if not 0.0 <= float(trace_decay) < 1.0:
        raise ValueError("trace_decay must be in [0,1)")
    if not all(torch.isfinite(value).all() for value in (x, r, post)):
        raise ValueError("eligibility factors must be finite")
    x_trace = torch.zeros_like(x[:, 0])
    r_trace = torch.zeros_like(r[:, 0])
    input_sq = torch.zeros(post.shape[-1], dtype=torch.float64, device=post.device)
    recurrent_sq = torch.zeros_like(input_sq)
    for step in range(x.shape[1]):
        x_trace = float(trace_decay) * x_trace + x[:, step]
        r_trace = float(trace_decay) * r_trace + r[:, step]
        factor = post[:, step]
        input_sq += (factor.unsqueeze(-1) * x_trace.unsqueeze(1)).to(torch.float64).square().mean(dim=(0, 2))
        recurrent_sq += (factor.unsqueeze(-1) * r_trace.unsqueeze(1)).to(torch.float64).square().mean(dim=(0, 2))
    return (input_sq / x.shape[1]).sqrt().to(post.dtype), (recurrent_sq / x.shape[1]).sqrt().to(post.dtype)


def capture_recurrent_lif_eligibility(
    model: Any,
    inputs: Any,
    *,
    trace_decay: float = 0.9,
    surrogate_beta: float | None = None,
) -> EligibilitySummary:
    """Capture two-factor summaries from the public recurrent SHD model.

    Forward hooks observe the public ``input`` and ``recurrent`` linear modules.
    Their summed currents reconstruct the LIF membrane trajectory exactly; no
    parameter-wise eligibility is retained or transmitted.
    """
    import math as _math
    import torch

    input_currents: list[Any] = []
    recurrent_currents: list[Any] = []
    recurrent_inputs: list[Any] = []

    def input_hook(_module, _args, output):
        input_currents.append(output)

    def recurrent_hook(_module, args, output):
        recurrent_inputs.append(args[0])
        recurrent_currents.append(output)

    handles = [model.input.register_forward_hook(input_hook), model.recurrent.register_forward_hook(recurrent_hook)]
    try:
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    timesteps = int(inputs.shape[1])
    if len(input_currents) != timesteps or len(recurrent_currents) != timesteps:
        raise RuntimeError("SHD model hooks did not observe exactly one call per timestep")
    membrane = torch.zeros_like(input_currents[0])
    post_steps = []
    spike_steps = []
    beta = float(surrogate_beta if surrogate_beta is not None else getattr(model, "surrogate_beta", 2.0))
    for input_current, recurrent_current in zip(input_currents, recurrent_currents):
        membrane = membrane + (input_current + recurrent_current - membrane) / float(model.tau)
        centered = membrane - float(model.threshold)
        post = (beta / 2.0) / (1.0 + (_math.pi * beta * centered / 2.0).square())
        spikes = (centered >= 0).to(membrane.dtype)
        post_steps.append(post)
        spike_steps.append(spikes)
        membrane = membrane * (1.0 - spikes.detach())
    post = torch.stack(post_steps, dim=1)
    recurrent_pre = torch.stack(recurrent_inputs, dim=1)
    input_rms, recurrent_rms = eligibility_rms_from_factors(
        inputs, recurrent_pre, post, trace_decay=trace_decay
    )
    spikes = torch.stack(spike_steps, dim=1).to(torch.float64)
    recurrent_pre = recurrent_pre.to(torch.float64)
    # Two independently informative same-batch activity summaries occupy the
    # same input/recurrent block slots as eligibility.  The input-associated
    # slot measures postsynaptic firing; the recurrent-associated slot measures
    # firing conditioned on recurrent presynaptic activity.
    activity_input = spikes.mean(dim=(0, 1))
    recurrent_gate = recurrent_pre.abs()
    activity_recurrent = (spikes * recurrent_gate).mean(dim=(0, 1))
    return EligibilitySummary(
        input=quantize_rms_u8(input_rms),
        recurrent=quantize_rms_u8(recurrent_rms),
        samples=int(inputs.shape[0]),
        timesteps=timesteps,
        activity_input=quantize_rms_u8(activity_input),
        activity_recurrent=quantize_rms_u8(activity_recurrent),
    )


def expand_postsynaptic(values: Any, parameter: Any):
    """Expand one scalar per output neuron to a parameter tensor."""
    import torch

    values = torch.as_tensor(values, device=parameter.device, dtype=parameter.dtype).reshape(-1)
    if parameter.ndim == 0 or parameter.shape[0] != values.numel():
        raise ValueError("postsynaptic summary does not match parameter output dimension")
    return values.reshape((-1,) + (1,) * (parameter.ndim - 1)).expand_as(parameter)


@dataclass(frozen=True)
class CorrectionCoefficients:
    age: float = 0.12
    drift: float = 0.35
    eligibility: float = 0.25
    interaction: float = 0.50

    def validate(self) -> None:
        values = {name: float(value) for name, value in asdict(self).items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("correction coefficients must be finite")
        # Eligibility is deliberately not assigned a direction.  For normalized
        # eligibility in [0,1], only require the effective drift slope to remain
        # nonnegative and the age rate to remain nonnegative over the unit square.
        drift_slopes = (values["drift"], values["drift"] + values["interaction"])
        if min(drift_slopes) < 0.0:
            raise ValueError("effective drift slope must be nonnegative for eligibility in [0,1]")
        corner_rates = (
            values["age"],
            values["age"] + values["eligibility"],
            values["age"] + values["drift"],
            values["age"] + values["drift"] + values["eligibility"] + values["interaction"],
        )
        if min(corner_rates) < 0.0:
            raise ValueError("age rate must be nonnegative for normalized drift and eligibility")


def bounded_monotone_correction(
    *,
    age: float,
    drift: Any,
    eligibility: Any,
    coefficients: CorrectionCoefficients = CorrectionCoefficients(),
):
    """Return ``c∈[0,1]`` with explicit eligibility×drift interaction.

    ``c = exp(-age*(a + d*drift + e*eligibility + x*eligibility*drift))``.
    Eligibility coefficients may be signed: its direction is learned/frozen,
    not prescribed.  Validation guarantees non-increasing age and drift over
    normalized eligibility in ``[0,1]`` and permits attenuation only.
    """
    import torch

    coefficients.validate()
    age = _finite_nonnegative(age, "age")
    drift = torch.as_tensor(drift)
    eligibility = torch.as_tensor(eligibility, device=drift.device, dtype=drift.dtype)
    if drift.shape != eligibility.shape or not drift.is_floating_point():
        raise ValueError("drift and eligibility must be aligned floating tensors")
    if not torch.isfinite(drift).all() or not torch.isfinite(eligibility).all():
        raise ValueError("correction inputs must be finite")
    if bool((drift < 0).any()) or bool((eligibility < 0).any()):
        raise ValueError("correction inputs must be nonnegative")
    if bool((drift > 1).any()) or bool((eligibility > 1).any()):
        raise ValueError("correction inputs must be normalized to [0,1]")
    exponent = -age * (
        coefficients.age
        + coefficients.drift * drift
        + coefficients.eligibility * eligibility
        + coefficients.interaction * eligibility * drift
    )
    return torch.exp(exponent).clamp_(0.0, 1.0)


def normalize_nonnegative(values: Any, epsilon: float = 1e-12):
    import torch

    tensor = torch.as_tensor(values)
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all() or bool((tensor < 0).any()):
        raise ValueError("normalization requires finite nonnegative floating values")
    scale = tensor.detach().max().clamp_min(float(epsilon))
    return tensor / scale


def correction_for_method(
    method: str,
    *,
    age: float,
    drift: Any,
    eligibility: Any,
    activity: Any,
    layer_age_rate: float = 0.12,
    coefficients: CorrectionCoefficients = CorrectionCoefficients(),
):
    """Apply a frozen capacity/payload-matched baseline or ablation."""
    import torch

    if method not in ORACLE_BASELINES:
        raise ValueError(f"unsupported Idea C correction method: {method}")
    drift = normalize_nonnegative(drift)
    eligibility = normalize_nonnegative(eligibility)
    activity = normalize_nonnegative(activity)
    if drift.shape != eligibility.shape or drift.shape != activity.shape:
        raise ValueError("all per-parameter features must have identical shapes")
    age = _finite_nonnegative(age, "age")
    if method == "no_correction":
        return torch.ones_like(drift)
    if method == "scalar_age":
        return torch.full_like(drift, math.exp(-coefficients.age * age))
    if method == "per_layer_age":
        return torch.full_like(drift, math.exp(-_finite_nonnegative(layer_age_rate, "layer_age_rate") * age))
    if method == "drift_age":
        return bounded_monotone_correction(
            age=age,
            drift=drift,
            eligibility=torch.zeros_like(eligibility),
            coefficients=CorrectionCoefficients(coefficients.age, coefficients.drift, 0.0, 0.0),
        )
    if method == "activity_drift_age":
        return bounded_monotone_correction(age=age, drift=drift, eligibility=activity, coefficients=coefficients)
    if method == "eligibility_age_no_drift":
        return bounded_monotone_correction(
            age=age,
            drift=torch.zeros_like(drift),
            eligibility=eligibility,
            coefficients=CorrectionCoefficients(coefficients.age, 0.0, coefficients.eligibility, 0.0),
        )
    if method == "eligibility_drift_age_no_interaction":
        return bounded_monotone_correction(
            age=age,
            drift=drift,
            eligibility=eligibility,
            coefficients=CorrectionCoefficients(
                coefficients.age, coefficients.drift, coefficients.eligibility, 0.0
            ),
        )
    return bounded_monotone_correction(age=age, drift=drift, eligibility=eligibility, coefficients=coefficients)


def coefficient_grid(
    base: CorrectionCoefficients,
    scales: Sequence[float] = COEFFICIENT_GRID_SCALES,
) -> tuple[CorrectionCoefficients, ...]:
    """Return the same tiny deterministic three-candidate grid for every tunable method."""
    if tuple(float(scale) for scale in scales) != COEFFICIENT_GRID_SCALES:
        raise ValueError("Idea C coefficient grid must remain the frozen three-candidate grid")
    candidates = []
    for scale in COEFFICIENT_GRID_SCALES:
        candidate = CorrectionCoefficients(**{
            name: float(value) * scale for name, value in asdict(base).items()
        })
        candidate.validate()
        candidates.append(candidate)
    return tuple(candidates)


def select_calibrated_coefficients(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str] = CALIBRATED_METHODS,
) -> tuple[dict[str, CorrectionCoefficients], dict[str, Any]]:
    """Select each method only by mean calibration relative-L2, with stable ties."""
    if not rows:
        raise ValueError("coefficient selection requires calibration rows")
    selected: dict[str, CorrectionCoefficients] = {}
    audit: dict[str, Any] = {}
    expected_count = None
    expected_rows_per_candidate = None
    for method in methods:
        scores: dict[int, list[float]] = {}
        coefficients: dict[int, CorrectionCoefficients] = {}
        for row in rows:
            candidate = row.get("candidate")
            if method not in row.get("methods", {}) or not isinstance(candidate, Mapping):
                raise ValueError("calibration row is missing candidate method data")
            index = int(candidate["index"])
            coefficient = CorrectionCoefficients(**candidate["coefficients"])
            coefficient.validate()
            scores.setdefault(index, []).append(float(row["methods"][method]["relative_l2"]))
            coefficients[index] = coefficient
        candidate_count = len(scores)
        if candidate_count != len(COEFFICIENT_GRID_SCALES):
            raise ValueError("every tunable method requires exactly three calibration candidates")
        if expected_count is None:
            expected_count = candidate_count
        if candidate_count != expected_count:
            raise ValueError("calibration candidate counts are not equal")
        rows_per_candidate = {len(values) for values in scores.values()}
        if len(rows_per_candidate) != 1:
            raise ValueError("calibration candidates do not have equal pair counts")
        if expected_rows_per_candidate is None:
            expected_rows_per_candidate = next(iter(rows_per_candidate))
        if next(iter(rows_per_candidate)) != expected_rows_per_candidate:
            raise ValueError("tunable methods do not have equal calibration pair counts")
        means = {index: float(np.mean(values)) for index, values in scores.items()}
        chosen = min(means, key=lambda index: (means[index], index))
        selected[method] = coefficients[chosen]
        audit[method] = {
            "candidate_count": candidate_count,
            "pairs_per_candidate": expected_rows_per_candidate,
            "candidate_mean_relative_l2": {str(index): means[index] for index in sorted(means)},
            "selected_candidate_index": chosen,
            "selected_coefficients": asdict(coefficients[chosen]),
        }
    return selected, audit


@dataclass(frozen=True)
class OraclePair:
    pair_id: str
    seed: int
    client_id: int
    layer: str
    staleness_bucket: str
    age: int
    stale_update: Any
    fresh_update: Any
    parameter_drift: Any
    eligibility: Any
    activity: Any
    pairing_audit: Mapping[str, Any] | None = None
    context_client_ids: tuple[int, ...] = ()
    deployment_features: frozenset[str] = DEPLOYMENT_FEATURES

    def validate_no_leakage(self) -> None:
        if self.deployment_features & FORBIDDEN_DEPLOYMENT_FEATURES:
            raise ValueError("fresh/oracle information leaked into deployment features")
        if not self.deployment_features <= DEPLOYMENT_FEATURES:
            raise ValueError("unregistered deployment feature")
        if self.age <= 0 or self.staleness_bucket not in {"light", "medium", "heavy"}:
            raise ValueError("invalid stale/fresh pairing metadata")
        if self.pairing_audit is not None and self.pairing_audit.get("identical_pairing") is not True:
            raise ValueError("stale/fresh pairing audit failed")
        if self.client_id in self.context_client_ids:
            raise ValueError("held-out client leaked into snapshot context")


def relative_l2(prediction: Any, target: Any, epsilon: float = 1e-12) -> float:
    import torch

    prediction = torch.as_tensor(prediction)
    target = torch.as_tensor(target, device=prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape or not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("oracle vectors must be aligned and finite")
    return float(torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target).clamp_min(epsilon))


def evaluate_oracle_pair(
    pair: OraclePair,
    methods: Sequence[str] = ORACLE_BASELINES,
    coefficients: CorrectionCoefficients = CorrectionCoefficients(),
    coefficients_by_method: Mapping[str, CorrectionCoefficients] | None = None,
) -> dict[str, Any]:
    import torch

    pair.validate_no_leakage()
    stale = torch.as_tensor(pair.stale_update)
    fresh = torch.as_tensor(pair.fresh_update, device=stale.device, dtype=stale.dtype)
    drift = torch.as_tensor(pair.parameter_drift, device=stale.device, dtype=stale.dtype)
    eligibility = torch.as_tensor(pair.eligibility, device=stale.device, dtype=stale.dtype)
    activity = torch.as_tensor(pair.activity, device=stale.device, dtype=stale.dtype)
    if not (stale.shape == fresh.shape == drift.shape == eligibility.shape == activity.shape):
        raise ValueError("oracle pair tensors must have identical shapes")
    rows = {}
    shuffled = eligibility.reshape(-1).roll(1).reshape_as(eligibility)
    for method in methods:
        feature = shuffled if method == "shuffled_eligibility_drift_age" else eligibility
        method_coefficients = (coefficients_by_method or {}).get(method, coefficients)
        layer_age_rate = (
            method_coefficients.age
            if pair.layer == "input.weight"
            else method_coefficients.drift
        )
        correction = correction_for_method(
            IDEA_C_METHOD if method == "shuffled_eligibility_drift_age" else method,
            age=pair.age,
            drift=drift,
            eligibility=feature,
            activity=activity,
            layer_age_rate=layer_age_rate,
            coefficients=method_coefficients,
        )
        rows[method] = {
            "relative_l2": relative_l2(stale * correction, fresh),
            "correction_min": float(correction.min()),
            "correction_max": float(correction.max()),
        }
    return {
        "pair_id": pair.pair_id,
        "seed": pair.seed,
        "client_id": pair.client_id,
        "layer": pair.layer,
        "staleness_bucket": pair.staleness_bucket,
        "age": pair.age,
        "methods": rows,
        "pairing_audit": dict(pair.pairing_audit or {}),
        "context_client_ids": list(pair.context_client_ids),
        "leakage_audit": "PASS",
    }


@dataclass
class CostLedger:
    summary_payload_bits: int = 0
    activity_baseline_payload_bits: int = 0
    correction_metadata_bits: int = 0
    model_snapshot_bytes: int = 0
    eligibility_factor_multiply_adds: int = 0
    correction_element_ops: int = 0
    local_training_steps: int = 0
    oracle_fresh_training_steps: int = 0

    def add_summary(self, summary: EligibilitySummary, *, layer_identifier_bits: int = 16) -> None:
        identifiers = 2 * int(layer_identifier_bits)
        self.summary_payload_bits += summary.payload_bits + identifiers
        self.activity_baseline_payload_bits += summary.activity_payload_bits + identifiers

    def as_record(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class Verdict:
    stage: str
    passed: bool
    terminal_state: str
    checks: Mapping[str, bool]
    statistics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "terminal_state": self.terminal_state,
            "checks": dict(self.checks),
            "statistics": dict(self.statistics),
        }


def paired_bootstrap_ci(
    differences: Sequence[float], *, seed: int = 20260727, samples: int = 10000
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or samples <= 0:
        raise ValueError("bootstrap requires at least two finite paired differences")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _seed_means(rows: Sequence[Mapping[str, Any]], method: str, metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(float(row["methods"][method][metric]))
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def oracle_verdict(rows: Sequence[Mapping[str, Any]]) -> Verdict:
    """Mechanically apply the frozen five-part oracle hard gate."""
    if not rows:
        raise ValueError("oracle verdict requires held-out rows")
    required = {"drift_age", "activity_drift_age", IDEA_C_METHOD}
    for row in rows:
        if (
            not required <= row["methods"].keys()
            or row.get("leakage_audit") != "PASS"
            or row.get("data_split") != "heldout"
            or row.get("calibration_audit") != "PASS"
        ):
            raise ValueError("oracle rows are incomplete or missing calibration PASS")
    full = _seed_means(rows, IDEA_C_METHOD, "relative_l2")
    drift = _seed_means(rows, "drift_age", "relative_l2")
    activity = _seed_means(rows, "activity_drift_age", "relative_l2")
    seeds = sorted(set(full) & set(drift) & set(activity))
    if len(seeds) != 5:
        raise ValueError("oracle hard gate requires exactly five aligned seeds")
    relative_improvements = [(drift[s] - full[s]) / max(drift[s], 1e-12) for s in seeds]
    absolute_differences = [drift[s] - full[s] for s in seeds]
    ci = paired_bootstrap_ci(absolute_differences)
    direction_count = sum(value > 0 for value in absolute_differences)
    bucket_differences: dict[str, float] = {}
    for bucket in ("light", "medium", "heavy"):
        selected = [row for row in rows if row["staleness_bucket"] == bucket]
        if not selected:
            raise ValueError(f"missing oracle staleness bucket: {bucket}")
        bucket_differences[bucket] = float(np.mean([
            row["methods"]["drift_age"]["relative_l2"]
            - row["methods"][IDEA_C_METHOD]["relative_l2"]
            for row in selected
        ]))
    improved_buckets = sum(value > 0 for value in bucket_differences.values())
    activity_differences = [activity[s] - full[s] for s in seeds]
    layers = sorted({str(row["layer"]) for row in rows})
    layer_improvements = {
        layer: float(np.mean([
            row["methods"]["drift_age"]["relative_l2"]
            - row["methods"][IDEA_C_METHOD]["relative_l2"]
            for row in rows if str(row["layer"]) == layer
        ]))
        for layer in layers
    }
    positive_layers = sum(value > 0 for value in layer_improvements.values())
    checks = {
        "mean_relative_l2_improvement_at_least_5pct": float(np.mean(relative_improvements)) >= 0.05,
        "at_least_4_of_5_seeds_same_direction": direction_count >= 4,
        "bootstrap_95ci_excludes_zero": ci[0] > 0.0,
        "at_least_two_buckets_improve": improved_buckets >= 2,
        "heavy_bucket_not_significantly_worse": bucket_differences["heavy"] >= 0.0,
        "beats_cost_matched_activity": sum(value > 0 for value in activity_differences) >= 4 and float(np.mean(activity_differences)) > 0,
        "benefit_not_single_layer": len(layers) >= 2 and positive_layers >= 2,
    }
    passed = all(checks.values())
    return Verdict(
        "oracle",
        passed,
        "ORACLE_PASS_RUN_END_TO_END" if passed else FINAL_ORACLE_TERMINAL_STATE,
        checks,
        {
            "seeds": seeds,
            "mean_relative_improvement": float(np.mean(relative_improvements)),
            "paired_absolute_difference_ci95": ci,
            "direction_count": direction_count,
            "bucket_differences": bucket_differences,
            "layer_improvements": layer_improvements,
        },
    )


def trapezoid_auc(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("AUC requires at least two finite values")
    return float(np.trapezoid(array) if hasattr(np, "trapezoid") else np.trapz(array))


def end_to_end_verdict(seed_rows: Sequence[Mapping[str, Any]]) -> Verdict:
    """Apply the frozen asynchronous SHD end-to-end gate."""
    if len(seed_rows) != 5 or len({int(row["seed"]) for row in seed_rows}) != 5:
        raise ValueError("end-to-end gate requires exactly five seeds")
    auc_diffs = []
    update_reductions = []
    final_diffs = []
    activity_diffs = []
    for row in seed_rows:
        methods = row["methods"]
        for method in ("drift_age", "activity_drift_age", IDEA_C_METHOD):
            if method not in methods:
                raise ValueError("end-to-end row is missing a required method")
        full = methods[IDEA_C_METHOD]
        drift = methods["drift_age"]
        activity = methods["activity_drift_age"]
        drift_auc = float(drift.get("accuracy_auc", trapezoid_auc(drift["accuracy_curve"])))
        full_auc = float(full.get("accuracy_auc", trapezoid_auc(full["accuracy_curve"])))
        auc_diffs.append((full_auc - drift_auc) / max(abs(drift_auc), 1e-12))
        drift_updates = float(drift.get("updates_to_target", math.inf))
        full_updates = float(full.get("updates_to_target", math.inf))
        update_reductions.append((drift_updates - full_updates) / max(drift_updates, 1.0) if math.isfinite(drift_updates) else 0.0)
        final_diffs.append(float(full["final_accuracy"]) - float(drift["final_accuracy"]))
        activity_diffs.append(float(full["final_accuracy"]) - float(activity["final_accuracy"]))
    primary = [max(auc, reduction) for auc, reduction in zip(auc_diffs, update_reductions)]
    ci = paired_bootstrap_ci(primary)
    checks = {
        "auc_plus_2pct_or_updates_minus_10pct": float(np.mean(auc_diffs)) >= 0.02 or float(np.mean(update_reductions)) >= 0.10,
        "at_least_4_of_5_seeds_same_direction": sum(value > 0 for value in primary) >= 4,
        "bootstrap_95ci_excludes_zero": ci[0] > 0,
        "final_accuracy_not_significantly_worse": paired_bootstrap_ci(final_diffs)[0] >= -0.01,
        "beats_cost_matched_activity": sum(value > 0 for value in activity_diffs) >= 4 and float(np.mean(activity_diffs)) > 0,
    }
    passed = all(checks.values())
    return Verdict(
        "end_to_end",
        passed,
        "IDEA_C_CONTINUE" if passed else "STOP_END_TO_END_FAIL",
        checks,
        {
            "mean_relative_auc_improvement": float(np.mean(auc_diffs)),
            "mean_update_reduction": float(np.mean(update_reductions)),
            "primary_ci95": ci,
            "mean_final_accuracy_difference": float(np.mean(final_diffs)),
        },
    )


def gate_end_to_end(oracle: Verdict) -> None:
    if oracle.stage != "oracle" or not oracle.passed:
        raise RuntimeError(SKIPPED_END_TO_END_STATE)


def protocol_identity(config: Mapping[str, Any], code_identity: str) -> str:
    payload = json.dumps(
        {"config": config, "code_identity": str(code_identity)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
