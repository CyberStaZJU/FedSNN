from __future__ import annotations

from typing import Any


class StaticBatchAcceleratorGraph:
    """Run full accelerator batches through one reusable execution graph.

    Partial batches stay in eager mode so heterogeneous client sizes do not
    trigger a separate graph capture for every remainder shape.
    """

    def __init__(self, model: Any, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.batch_size = int(batch_size)
        parameter = next(model.parameters())
        self.device_type = parameter.device.type
        self.enabled = self.device_type in {"cuda", "npu"}
        self._compiled = None

    @property
    def backend(self) -> str:
        if self.device_type == "cuda":
            return "cudagraphs"
        if self.device_type == "npu":
            return "npugraph"
        return "eager"

    def __call__(self, inputs, *args, **kwargs):
        if (
            self.enabled
            and int(inputs.shape[0]) == self.batch_size
            and not args
            and not kwargs
        ):
            if self._compiled is None:
                import torch

                if self.device_type == "cuda":
                    self._compiled = torch.compile(
                        self.model,
                        backend="cudagraphs",
                        fullgraph=True,
                        dynamic=False,
                    )
                else:
                    # torch_npu patches the supplied module's ``forward`` in
                    # place.  Capture a wrapper so the real model keeps its
                    # eager forward for partial batches and activity queries.
                    class ForwardOnly(torch.nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model

                        def forward(self, batch):
                            return self.model(batch)

                    capture_wrapper = ForwardOnly(self.model)
                    self._compiled = torch.npu.make_graphed_callables(
                        capture_wrapper,
                        (inputs,),
                        num_warmup_iters=3,
                    )
            return self._compiled(inputs)
        return self.model(inputs, *args, **kwargs)


def accelerator_graph_runtime_metadata(
    runner: StaticBatchAcceleratorGraph,
) -> dict[str, Any]:
    return {
        "training_forward_backend": runner.backend,
        "accelerator_graph_static_batch_size": (
            runner.batch_size if runner.enabled else None
        ),
        "partial_batch_backend": "eager",
        "accelerator_graph_semantics": "exact eager math with advancing RNG replay",
    }


class SpikingJellyAlexNetForward:
    """Opt-in packed/NPUGraph runner for the FedSNN AlexNet+BNTT model.

    Poisson encoding deliberately happens before graph routing.  The capturable
    model therefore consumes a fixed batch-first current sequence and contains
    no RNG operation.  Parameters remain owned by ``model`` so checkpoints and
    optimizers retain the legacy state-dict identity.
    """

    def __init__(self, model: Any, batch_size: int, backend: str):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        backend = str(backend).lower()
        if backend not in {"packed_eager", "packed_aspy", "npugraph"}:
            raise ValueError(
                "SpikingJelly AlexNet backend must be packed_eager, "
                "packed_aspy, or npugraph"
            )
        required = ("encode_poisson_sequence", "forward_encoded_packed")
        if any(not hasattr(model, name) for name in required):
            raise ValueError("model does not expose the packed AlexNet execution seam")
        self.model = model
        self.batch_size = int(batch_size)
        self.requested_backend = backend
        self._runner = None
        self._adapter = None
        self.last_route = None

        if backend == "npugraph":
            try:
                import torch
                from spikingjelly_npu.npu import StaticGraphRunner
            except ImportError as exc:
                raise RuntimeError(
                    "model.execution_backend=npugraph requires spikingjelly_npu"
                ) from exc
            if next(model.parameters()).device.type != "npu":
                raise ValueError("spikingjelly_npu NPUGraph requires an NPU model")
            # Ascend Conv2D otherwise selects the legacy aclop/internal-format
            # path, which torch-npu cannot record inside an NPUGraph.  This is
            # process-global, so it is changed only for the explicit npugraph
            # backend before any capture is attempted.
            torch.npu.config.allow_internal_format = False
            if not torch.are_deterministic_algorithms_enabled():
                raise ValueError(
                    "spikingjelly_npu training graph requires "
                    "torch.use_deterministic_algorithms(True)"
                )

            class BatchFirstEncoded(torch.nn.Module):
                _spikingjelly_npu_graph_safe = True

                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(self, batch_first_sequence):
                    return self.inner.forward_encoded_packed(
                        batch_first_sequence.transpose(0, 1).contiguous()
                    )

            self._adapter = BatchFirstEncoded(model)
            self._runner = StaticGraphRunner(
                self._adapter,
                batch_size=self.batch_size,
                strict=True,
                allow_training=True,
            )

    @property
    def backend(self) -> str:
        return self.requested_backend

    def __call__(self, inputs, *args, **kwargs):
        if args or kwargs:
            # Diagnostic calls stay on the authoritative legacy model path.
            self.last_route = "eager_diagnostic"
            return self.model(inputs, *args, **kwargs)
        encoded = self.model.encode_poisson_sequence(inputs)
        if self._runner is None:
            output = self.model.forward_encoded_packed(encoded)
            if self.requested_backend == "packed_aspy":
                lif_routes = tuple(getattr(self.model, "last_lif_routes", ()))
                all_native = len(lif_routes) == 6 and all(
                    getattr(route, "backend", "torch") == "aspy"
                    and getattr(route, "requested_backend", None) == "aspy"
                    for route in lif_routes
                )
                self.last_route = (
                    "packed_aspy" if all_native else "packed_aspy_fallback"
                )
                if (
                    not all_native
                    and bool(getattr(self.model, "execution_backend_strict", False))
                ):
                    details = [
                        {
                            "index": index,
                            "requested": getattr(route, "requested_backend", None),
                            "backend": getattr(route, "backend", None),
                            "reason": getattr(route, "reason", None),
                        }
                        for index, route in enumerate(lif_routes)
                    ]
                    raise RuntimeError(
                        "strict packed_aspy execution requires six native LIF "
                        f"routes; observed={len(lif_routes)}, details={details}"
                    )
            else:
                self.last_route = "packed_eager"
            return output
        output = self._runner(encoded.transpose(0, 1).contiguous())
        self.last_route = self._runner.last_route
        return output


def model_forward_runner(model: Any, batch_size: int):
    """Select the legacy or opt-in SpikingJelly runner from model metadata."""
    backend = str(getattr(model, "execution_backend", "legacy_stepwise")).lower()
    if backend == "legacy_stepwise":
        return StaticBatchAcceleratorGraph(model, batch_size)
    return SpikingJellyAlexNetForward(model, batch_size, backend)


def model_forward_runtime_metadata(runner: Any) -> dict[str, Any]:
    if isinstance(runner, StaticBatchAcceleratorGraph):
        return accelerator_graph_runtime_metadata(runner)
    return {
        "training_forward_backend": runner.backend,
        "accelerator_graph_static_batch_size": (
            runner.batch_size if runner.backend == "npugraph" else None
        ),
        "partial_batch_backend": (
            "eager_packed" if runner.backend == "npugraph" else runner.backend
        ),
        "accelerator_graph_semantics": (
            "poisson_outside_graph; packed T*N AlexNet; exact decay-LIF AsPy "
            "route when requested; deterministic training gate for NPUGraph"
        ),
        "requested_temporal_backend": (
            "aspy_exact_decay_lif" if runner.backend == "packed_aspy" else None
        ),
        "execution_backend_strict": bool(
            getattr(runner.model, "execution_backend_strict", False)
        ),
        "temporal_backend_fallback_policy": (
            "fail"
            if bool(getattr(runner.model, "execution_backend_strict", False))
            else "allow_observable"
        ),
        "diagnostic_forward_backend": "legacy_eager",
        "expected_native_lif_calls_per_forward": (
            6 if runner.backend == "packed_aspy" else None
        ),
        "spikingjelly_npu_integration": True,
    }


# Backward-compatible names for existing trainers and external scripts.
StaticBatchCudaGraph = StaticBatchAcceleratorGraph
cuda_graph_runtime_metadata = accelerator_graph_runtime_metadata
