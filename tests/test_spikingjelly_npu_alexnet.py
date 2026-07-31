"""Parity and routing gates for the opt-in SpikingJelly NPU AlexNet path."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml


torch = pytest.importorskip("torch")

from fedsnn.config import load_config  # noqa: E402
from fedsnn.models import build_fedsnn_alexnet_bntt  # noqa: E402
from fedsnn.runtime import (  # noqa: E402
    SpikingJellyAlexNetForward,
    model_forward_runner,
    model_forward_runtime_metadata,
)


pytest.importorskip("spikingjelly_npu")


def _models(
    *,
    training: bool = True,
    backend: str = "packed_eager",
    strict: bool = False,
    timesteps: int = 2,
):
    reference = build_fedsnn_alexnet_bntt(
        timesteps=timesteps,
        track_runtime_activity=False,
        execution_backend="legacy_stepwise",
    )
    packed = build_fedsnn_alexnet_bntt(
        timesteps=timesteps,
        track_runtime_activity=False,
        execution_backend=backend,
        execution_backend_strict=strict,
    )
    packed.load_state_dict(reference.state_dict())
    reference.train(training)
    packed.train(training)
    return reference, packed


def _snapshot_gradients(model):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def test_packed_backend_preserves_legacy_state_dict_keys():
    reference, packed = _models(training=False)
    assert tuple(reference.state_dict()) == tuple(packed.state_dict())
    assert all(not key.startswith("model.") for key in packed.state_dict())


@pytest.mark.parametrize("backend", ["packed_eager", "packed_aspy"])
def test_actual_alexnet_encoded_packed_matches_stepwise_training_step(backend):
    reference, packed = _models(training=True, backend=backend)
    encoded = torch.rand(2, 2, 3, 32, 32).ge(0.5).float()
    labels = torch.tensor([1, 7])
    criterion = torch.nn.CrossEntropyLoss()

    reference.zero_grad(set_to_none=True)
    packed.zero_grad(set_to_none=True)
    reference_logits, reference_patterns = reference.forward_encoded_stepwise(
        encoded, return_neuron_patterns=True
    )
    packed_logits, packed_patterns = packed.forward_encoded_packed(
        encoded, return_neuron_patterns=True
    )
    reference_loss = criterion(reference_logits, labels)
    packed_loss = criterion(packed_logits, labels)
    reference_loss.backward()
    packed_loss.backward()

    torch.testing.assert_close(packed_logits, reference_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(packed_loss, reference_loss, rtol=1e-6, atol=1e-7)
    assert len(reference_patterns) == len(packed_patterns) == 6
    for expected, actual in zip(reference_patterns, packed_patterns):
        torch.testing.assert_close(actual, expected)

    reference_gradients = _snapshot_gradients(reference)
    packed_gradients = _snapshot_gradients(packed)
    assert reference_gradients.keys() == packed_gradients.keys()
    for name in reference_gradients:
        expected = reference_gradients[name]
        actual = packed_gradients[name]
        assert (expected is None) == (actual is None), name
        if expected is not None:
            torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)

    reference_buffers = dict(reference.named_buffers())
    packed_buffers = dict(packed.named_buffers())
    assert reference_buffers.keys() == packed_buffers.keys()
    for name in reference_buffers:
        torch.testing.assert_close(packed_buffers[name], reference_buffers[name])

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    packed_optimizer = torch.optim.SGD(packed.parameters(), lr=0.05)
    reference_optimizer.step()
    packed_optimizer.step()
    for (reference_name, expected), (packed_name, actual) in zip(
        reference.named_parameters(), packed.named_parameters()
    ):
        assert reference_name == packed_name
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize("backend", ["packed_eager", "packed_aspy"])
def test_packed_forward_uses_same_seeded_poisson_stream_as_legacy(backend):
    reference, packed = _models(training=False, backend=backend)
    images = torch.rand(2, 3, 32, 32)

    torch.manual_seed(234)
    expected = reference(images)
    torch.manual_seed(234)
    actual = packed(images)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", ["packed_eager", "packed_aspy"])
def test_packed_diagnostics_stay_on_legacy_path(backend):
    reference, packed = _models(training=False, backend=backend)
    images = torch.rand(2, 3, 32, 32)

    torch.manual_seed(17)
    expected_logits, expected_patterns = reference(
        images, return_neuron_patterns=True
    )
    torch.manual_seed(17)
    actual_logits, actual_patterns = packed(images, return_neuron_patterns=True)

    torch.testing.assert_close(actual_logits, expected_logits)
    for expected, actual in zip(expected_patterns, actual_patterns):
        torch.testing.assert_close(actual, expected)


def test_model_forward_runner_keeps_default_legacy_and_selects_packed():
    legacy = build_fedsnn_alexnet_bntt(
        timesteps=1, track_runtime_activity=False
    )
    packed = build_fedsnn_alexnet_bntt(
        timesteps=1,
        track_runtime_activity=False,
        execution_backend="packed_eager",
    )
    aspy = build_fedsnn_alexnet_bntt(
        timesteps=1,
        track_runtime_activity=False,
        execution_backend="packed_aspy",
    )
    assert model_forward_runner(legacy, 4).__class__.__name__ == (
        "StaticBatchAcceleratorGraph"
    )
    assert isinstance(model_forward_runner(packed, 4), SpikingJellyAlexNetForward)
    aspy_runner = model_forward_runner(aspy, 4)
    assert isinstance(aspy_runner, SpikingJellyAlexNetForward)
    assert aspy_runner.backend == "packed_aspy"
    metadata = model_forward_runtime_metadata(aspy_runner)
    assert metadata["training_forward_backend"] == "packed_aspy"
    assert metadata["partial_batch_backend"] == "packed_aspy"
    assert metadata["requested_temporal_backend"] == "aspy_exact_decay_lif"


def test_packed_aspy_strict_cpu_rejects_before_execution():
    _, aspy = _models(training=False, backend="packed_aspy", strict=True)
    runner = SpikingJellyAlexNetForward(aspy, 2, "packed_aspy")

    with pytest.raises(RuntimeError, match="requires an NPU tensor"):
        runner(torch.rand(2, 3, 32, 32))


def test_packed_aspy_cpu_fallback_is_observable_and_exact():
    reference, aspy = _models(training=False, backend="packed_aspy")
    images = torch.rand(2, 3, 32, 32)
    runner = SpikingJellyAlexNetForward(aspy, 2, "packed_aspy")

    torch.manual_seed(811)
    expected = reference(images)
    torch.manual_seed(811)
    actual = runner(images)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert runner.last_route == "packed_aspy_fallback"
    assert aspy.last_lif_route.requested_backend == "aspy"
    assert aspy.last_lif_route.backend == "torch"
    assert "requires an NPU tensor" in aspy.last_lif_route.reason


def test_npugraph_runner_requires_npu_model():
    model = build_fedsnn_alexnet_bntt(
        timesteps=1,
        track_runtime_activity=False,
        execution_backend="npugraph",
    )
    # Device monkeypatching is intentionally avoided: on CPU the runner must
    # reject before constructing a graph, which is the safety contract.
    with pytest.raises(ValueError, match="requires an NPU model"):
        SpikingJellyAlexNetForward(model, 4, "npugraph")


@pytest.mark.npu
def test_actual_alexnet_packed_aspy_native_training_eval_remainder_and_diagnostics():
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    device = torch.device(f"npu:{int(os.environ.get('ASCEND_DEVICE_ID', '0'))}")
    torch.npu.set_device(device)
    reference, aspy = _models(
        training=True,
        backend="packed_aspy",
        strict=True,
        timesteps=4,
    )
    assert tuple(reference.state_dict()) == tuple(aspy.state_dict())
    reference = reference.to(device)
    aspy = aspy.to(device)
    full_batch = 128
    runner = SpikingJellyAlexNetForward(aspy, full_batch, "packed_aspy")
    images = torch.rand(full_batch, 3, 32, 32, device=device)
    torch.manual_seed(20260902)
    encoded = reference.encode_poisson_sequence(images)
    labels = torch.arange(full_batch, device=device).remainder(10)
    criterion = torch.nn.CrossEntropyLoss()

    reference.zero_grad(set_to_none=True)
    aspy.zero_grad(set_to_none=True)
    expected, expected_patterns = reference.forward_encoded_stepwise(
        encoded, return_neuron_patterns=True
    )
    actual, actual_patterns = aspy.forward_encoded_packed(
        encoded, return_neuron_patterns=True
    )
    expected_loss = criterion(expected, labels)
    actual_loss = criterion(actual, labels)
    expected_loss.backward()
    actual_loss.backward()
    torch.npu.synchronize(device)

    direct_routes = tuple(aspy.last_lif_routes)
    assert aspy.last_lif_route.backend == "aspy"
    assert aspy.last_lif_route.accelerated
    assert len(direct_routes) == 6
    assert all(route.backend == "aspy" for route in direct_routes)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-7)
    assert len(expected_patterns) == len(actual_patterns) == 6
    for expected_pattern, actual_pattern in zip(expected_patterns, actual_patterns):
        torch.testing.assert_close(actual_pattern, expected_pattern, rtol=0, atol=0)

    for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
        reference.named_parameters(), aspy.named_parameters()
    ):
        assert expected_name == actual_name
        if expected_parameter.grad is None:
            assert actual_parameter.grad is None
        else:
            # Ascend's native reverse-time reduction differs slightly from the
            # eager convolution gradient order at actual client batch shapes.
            # The user-approved numerical gate covers the observed full-batch
            # 1.23009e-5 and remainder-batch 2.58684e-5 absolute differences;
            # this is tolerance equivalence, not bitwise equivalence.
            torch.testing.assert_close(
                actual_parameter.grad,
                expected_parameter.grad,
                rtol=5e-5,
                atol=3e-5,
            )
    for (expected_name, expected_buffer), (actual_name, actual_buffer) in zip(
        reference.named_buffers(), aspy.named_buffers()
    ):
        assert expected_name == actual_name
        torch.testing.assert_close(actual_buffer, expected_buffer)

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    aspy_optimizer = torch.optim.SGD(aspy.parameters(), lr=0.05)
    reference_optimizer.step()
    aspy_optimizer.step()
    for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
        reference.named_parameters(), aspy.named_parameters()
    ):
        assert expected_name == actual_name
        torch.testing.assert_close(
            actual_parameter,
            expected_parameter,
            rtol=5e-5,
            atol=5e-6,
        )

    # The real client ends with a 42-sample training remainder. Reset both models
    # to one identical post-full-batch state, then qualify its backward, BNTT
    # buffers, native routes, and optimizer update explicitly rather than relying
    # only on finite-loss coverage in the client benchmark.
    aspy.load_state_dict(reference.state_dict())
    remainder_batch = 42
    remainder_images = torch.rand(
        remainder_batch, 3, 32, 32, device=device
    )
    remainder_labels = torch.arange(
        remainder_batch, device=device
    ).remainder(10)
    torch.manual_seed(20260904)
    remainder_encoded = reference.encode_poisson_sequence(remainder_images)
    reference.zero_grad(set_to_none=True)
    aspy.zero_grad(set_to_none=True)
    expected_remainder = reference.forward_encoded_stepwise(remainder_encoded)
    actual_remainder = aspy.forward_encoded_packed(remainder_encoded)
    expected_remainder_loss = criterion(expected_remainder, remainder_labels)
    actual_remainder_loss = criterion(actual_remainder, remainder_labels)
    expected_remainder_loss.backward()
    actual_remainder_loss.backward()
    torch.npu.synchronize(device)

    assert len(aspy.last_lif_routes) == 6
    assert all(route.backend == "aspy" for route in aspy.last_lif_routes)
    torch.testing.assert_close(
        actual_remainder, expected_remainder, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        actual_remainder_loss,
        expected_remainder_loss,
        rtol=1e-6,
        atol=1e-7,
    )
    for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
        reference.named_parameters(), aspy.named_parameters()
    ):
        assert expected_name == actual_name
        if expected_parameter.grad is None:
            assert actual_parameter.grad is None
        else:
            # Same approved actual-shape numerical gate as the full batch.
            torch.testing.assert_close(
                actual_parameter.grad,
                expected_parameter.grad,
                rtol=5e-5,
                atol=3e-5,
            )
    for (expected_name, expected_buffer), (actual_name, actual_buffer) in zip(
        reference.named_buffers(), aspy.named_buffers()
    ):
        assert expected_name == actual_name
        torch.testing.assert_close(actual_buffer, expected_buffer)

    reference_optimizer.step()
    aspy_optimizer.step()
    for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
        reference.named_parameters(), aspy.named_parameters()
    ):
        assert expected_name == actual_name
        torch.testing.assert_close(
            actual_parameter,
            expected_parameter,
            rtol=5e-5,
            atol=5e-6,
        )

    # The optimizer-update gate above intentionally accepts a small numerical
    # tolerance. Do not feed those non-identical parameters into another hard
    # threshold trajectory and misclassify spike amplification as a route
    # failure. Route qualification starts from one exactly shared state.
    aspy.load_state_dict(reference.state_dict())
    runner_training_images = torch.rand(full_batch, 3, 32, 32, device=device)
    reference.train()
    aspy.train()
    torch.manual_seed(20260903)
    expected_runner_training = reference(runner_training_images)
    # NPU random kernels are asynchronous. Complete the reference Poisson
    # stream before rewinding the shared generator for the packed path.
    torch.npu.synchronize(device)
    torch.manual_seed(20260903)
    actual_runner_training = runner(runner_training_images)
    torch.npu.synchronize(device)
    torch.testing.assert_close(
        actual_runner_training,
        expected_runner_training,
        rtol=1e-5,
        atol=1e-6,
    )
    assert runner.last_route == "packed_aspy"
    assert len(aspy.last_lif_routes) == 6
    assert all(route.backend == "aspy" for route in aspy.last_lif_routes)

    aspy.eval()
    reference.eval()
    for batch in (full_batch, 42, 1):
        eval_images = torch.rand(batch, 3, 32, 32, device=device)
        torch.manual_seed(20260910 + batch)
        expected_eval = reference(eval_images)
        torch.npu.synchronize(device)
        torch.manual_seed(20260910 + batch)
        actual_eval = runner(eval_images)
        torch.npu.synchronize(device)
        torch.testing.assert_close(actual_eval, expected_eval, rtol=1e-5, atol=1e-6)
        assert actual_eval.shape == (batch, 10)
        assert runner.last_route == "packed_aspy"
        assert len(aspy.last_lif_routes) == 6
        assert all(route.backend == "aspy" for route in aspy.last_lif_routes)

    diagnostic_images = torch.rand(1, 3, 32, 32, device=device)
    torch.manual_seed(20260999)
    expected_diagnostic, expected_diagnostic_patterns = reference(
        diagnostic_images, return_neuron_patterns=True
    )
    torch.npu.synchronize(device)
    torch.manual_seed(20260999)
    diagnostic_output, patterns = runner(
        diagnostic_images, return_neuron_patterns=True
    )
    torch.testing.assert_close(diagnostic_output, expected_diagnostic, rtol=0, atol=0)
    assert len(patterns) == len(expected_diagnostic_patterns) == 6
    for expected_pattern, actual_pattern in zip(
        expected_diagnostic_patterns, patterns
    ):
        torch.testing.assert_close(actual_pattern, expected_pattern, rtol=0, atol=0)
    assert runner.last_route == "eager_diagnostic"


@pytest.mark.npu
def test_packed_aspy_exact_yaml_runs_real_topk_trainer_smoke(tmp_path):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    from fedsnn.train_topk import run_topk

    device_index = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "smoke"
        / "spikingjelly_npu_packed_aspy_seed2.yaml"
    )
    config = load_config(config_path)
    config["output"]["root"] = str(tmp_path / "runs")
    data_root = os.environ.get("FEDSNN_TEST_DATA_ROOT")
    if not data_root:
        pytest.skip("FEDSNN_TEST_DATA_ROOT is required for the real trainer smoke")

    output = run_topk(
        config,
        data_root,
        f"npu:{device_index}",
        resume=False,
        smoke=True,
    )
    resolved = yaml.safe_load((output / "resolved_config.yaml").read_text())
    rows = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert resolved["model"]["execution_backend"] == "packed_aspy"
    assert resolved["model"]["execution_backend_strict"] is True
    assert resolved["runtime"]["effective_timesteps"] == 4
    assert resolved["runtime"]["execution_backend_strict"] is True
    assert len(rows) == 1
    assert rows[0]["training_forward_route"] == "packed_aspy"
    assert rows[0]["eval_forward_route"] == "packed_aspy"
    assert rows[0]["training_lif_backends"] == ["aspy"] * 6
    assert rows[0]["eval_lif_backends"] == ["aspy"] * 6
    assert torch.isfinite(torch.tensor(rows[0]["test_loss"]))
    assert torch.isfinite(torch.tensor(rows[0]["test_accuracy"]))


@pytest.mark.npu
def test_actual_alexnet_npugraph_matches_packed_and_routes_remainders_eager():
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    device = torch.device(f"npu:{int(os.environ.get('ASCEND_DEVICE_ID', '0'))}")
    torch.npu.set_device(device)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        packed = build_fedsnn_alexnet_bntt(
            timesteps=2,
            track_runtime_activity=False,
            execution_backend="packed_eager",
        ).to(device).train()
        graph = build_fedsnn_alexnet_bntt(
            timesteps=2,
            track_runtime_activity=False,
            execution_backend="npugraph",
        ).to(device).train()
        graph.load_state_dict(packed.state_dict())
        runner = SpikingJellyAlexNetForward(graph, 2, "npugraph")
        encoded = torch.rand(2, 2, 3, 32, 32, device=device).ge(0.5).float()
        labels = torch.tensor([1, 7], device=device)
        criterion = torch.nn.CrossEntropyLoss()

        packed.zero_grad(set_to_none=True)
        graph.zero_grad(set_to_none=True)
        expected = packed.forward_encoded_packed(encoded)
        actual = runner._runner(encoded.transpose(0, 1).contiguous())
        expected_loss = criterion(expected, labels)
        actual_loss = criterion(actual, labels)
        expected_loss.backward()
        actual_loss.backward()
        torch.npu.synchronize()

        assert runner._runner.last_route.backend == "npugraph"
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0.0, atol=0.0)
        for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
            packed.named_parameters(), graph.named_parameters()
        ):
            assert expected_name == actual_name
            if expected_parameter.grad is None:
                assert actual_parameter.grad is None
            else:
                torch.testing.assert_close(
                    actual_parameter.grad, expected_parameter.grad, rtol=0.0, atol=0.0
                )
        for (expected_name, expected_buffer), (actual_name, actual_buffer) in zip(
            packed.named_buffers(), graph.named_buffers()
        ):
            assert expected_name == actual_name
            torch.testing.assert_close(actual_buffer, expected_buffer, rtol=0.0, atol=0.0)

        remainder = encoded[:, :1].transpose(0, 1).contiguous()
        remainder_output = runner._runner(remainder)
        assert remainder_output.shape == (1, 10)
        assert runner._runner.last_route.backend == "eager"
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
