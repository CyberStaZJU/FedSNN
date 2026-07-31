from __future__ import annotations

from typing import Any


def resolve_snn_backend(requested: str = "auto") -> str:
    """Resolve multi-step SNN backend for feedforward SHD probes.

    Returns one of: ``legacy``, ``pure``, ``torch``, ``cupy``, ``triton``.

    - ``legacy``: original Python ``for t`` loop (tests / parity).
    - ``pure``: pure-PyTorch multi-step (one Linear over ``B*T``, then LIF scan).
    - ``torch`` / ``cupy`` / ``triton``: SpikingJelly multi-step LIF backends.
    - ``auto``: ``triton`` → ``cupy`` → SpikingJelly ``torch`` → ``pure``.
    """
    name = str(requested or "auto").strip().lower()
    if name in {"legacy", "python", "for_loop"}:
        return "legacy"
    if name in {"pure", "pure_torch", "multistep_pure"}:
        return "pure"
    if name not in {"auto", "torch", "cupy", "triton", "sj_torch", "multi_step"}:
        raise ValueError(
            f"unsupported snn backend={requested!r}; "
            "use auto|legacy|pure|torch|cupy|triton"
        )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for the SHD model") from exc

    def _sj_ok() -> bool:
        try:
            from spikingjelly.activation_based import neuron, surrogate  # noqa: F401

            return True
        except Exception:
            return False

    def _backend_constructible(backend: str) -> bool:
        if not _sj_ok():
            return False
        try:
            from spikingjelly.activation_based import neuron, surrogate

            node = neuron.LIFNode(
                tau=2.0,
                decay_input=True,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(alpha=2.0),
                detach_reset=True,
                step_mode="m",
                backend=backend,
            )
            return backend in getattr(node, "supported_backends", (backend,))
        except Exception:
            return False

    cuda = bool(torch.cuda.is_available())
    if name in {"torch", "sj_torch", "multi_step"}:
        if _sj_ok():
            return "torch"
        return "pure"
    if name == "cupy":
        if not _backend_constructible("cupy"):
            raise RuntimeError("SpikingJelly cupy backend unavailable (install cupy + CUDA)")
        return "cupy"
    if name == "triton":
        if not _backend_constructible("triton"):
            raise RuntimeError("SpikingJelly triton backend unavailable")
        return "triton"
    # auto: prefer CuPy fused multi-step when available; otherwise SpikingJelly
    # torch multi-step (always correct, no Triton JIT tax on small SHD kernels).
    # Triton remains available via explicit snn_backend=triton.
    if cuda and _backend_constructible("cupy"):
        return "cupy"
    if _sj_ok():
        return "torch"
    return "pure"


def build_recurrent_lif_shd(
    *,
    input_units: int = 700,
    hidden_units: int = 128,
    classes: int = 20,
    tau: float = 20.0,
    threshold: float = 1.0,
    surrogate_beta: float = 2.0,
):
    """Build a recurrent LIF network with a non-spiking linear readout for SHD.

    Inputs have shape ``[batch, time, input_units]``. The recurrent connection
    consumes the previous time step's spikes and has no bias. Logits are the
    temporal mean of the linear readout membrane current. ``return_spikes``
    exposes the complete ``[batch, time, hidden]`` pattern for shared SHD
    diagnostics and downstream algorithms.
    """

    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("PyTorch is required for the SHD model") from exc
    if min(input_units, hidden_units, classes) <= 0:
        raise ValueError("model dimensions must be positive")
    if tau <= 1 or threshold <= 0 or surrogate_beta <= 0:
        raise ValueError("tau must exceed 1 and threshold/beta must be positive")

    class ATanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return (inputs >= 0).to(inputs.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (inputs,) = ctx.saved_tensors
            denominator = 1.0 + (
                math.pi * surrogate_beta * inputs / 2.0
            ).square()
            return grad_output * (surrogate_beta / 2.0) / denominator

    class RecurrentLIFSHD(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_units = int(input_units)
            self.hidden_units = int(hidden_units)
            self.classes = int(classes)
            self.tau = float(tau)
            self.threshold = float(threshold)
            self.input = nn.Linear(input_units, hidden_units, bias=True)
            self.recurrent = nn.Linear(hidden_units, hidden_units, bias=False)
            self.readout = nn.Linear(hidden_units, classes, bias=True)
            self.spike_parameter_layers = {
                "input.weight": "hidden",
                "input.bias": "hidden",
                "recurrent.weight": "hidden",
                "readout.weight": "readout",
                "readout.bias": "readout",
            }
            self.snn_backend = "legacy"
            self.architecture = "recurrent_lif_shd"

        def forward(self, inputs, *, return_spikes: bool = False):
            if inputs.ndim != 3 or inputs.shape[2] != self.input_units:
                raise ValueError(
                    f"SHD model expects [batch,time,{self.input_units}] inputs"
                )
            batch, timesteps, _ = inputs.shape
            membrane = inputs.new_zeros((batch, self.hidden_units))
            previous_spikes = inputs.new_zeros((batch, self.hidden_units))
            spike_steps = []
            logits = inputs.new_zeros((batch, self.classes))
            for step in range(timesteps):
                current = self.input(inputs[:, step]) + self.recurrent(previous_spikes)
                membrane = membrane + (current - membrane) / self.tau
                spikes = ATanSpike.apply(membrane - self.threshold)
                membrane = membrane * (1.0 - spikes.detach())
                logits = logits + self.readout(spikes)
                spike_steps.append(spikes)
                previous_spikes = spikes
            logits = logits / timesteps
            if return_spikes:
                return logits, torch.stack(spike_steps, dim=1)
            return logits

    return RecurrentLIFSHD()


def build_feedforward_lif_shd(
    *,
    input_units: int = 700,
    hidden_units: int = 50,
    classes: int = 5,
    tau: float = 20.0,
    threshold: float = 1.0,
    surrogate_beta: float = 2.0,
    backend: str = "auto",
    dynamics: str = "leaky",
    weight_init: str = "default",
    current_decay: float | None = None,
    voltage_decay: float | None = None,
):
    """Build a feedforward (non-recurrent) LIF/IF network for simplified SHD probes.

    Same public surface as the recurrent SHD model (input + readout, optional
    spike return) but **no** ``recurrent`` module.  Intended for literature-
    aligned simple-task exploratory tracks (e.g. 5-class SHD + small hidden),
    not for rewriting the frozen Stage-1A recurrent protocol.

    Dynamics
    --------
    - ``leaky`` (default): discrete leaky LIF
      ``V ← V + (I − V) / tau`` with hard reset (historical Stage-1A style).
    - ``chaki_if``: Chaki et al. (arXiv:2303.00928) Table I discrete form with
      current decay α=0 and voltage decay β=1, i.e. non-leaky integrator
      ``I = W x`` (no current memory) and ``V ← V + I`` (IF). Surrogate BP still
      applies. Output remains a non-spiking linear temporal mean (probe
      simplification; paper also applies LIF-like dynamics to the output).

    Weight init
    -----------
    - ``default``: PyTorch ``nn.Linear`` defaults.
    - ``chaki``: weight ~ N(0, 1), bias 0 (Table I mean/scale 0/1).

    Backend (desktop GPU path):
    - ``auto`` / ``triton`` / ``cupy`` / ``torch``: SpikingJelly multi-step
      (``step_mode='m'``); Linear is multi-step so one GEMM covers all timesteps.
      ``chaki_if`` uses ``IFNode``; ``leaky`` uses ``LIFNode``.
    - ``pure``: pure-PyTorch multi-step (fused Linear over ``B*T`` + scan).
    - ``legacy``: original Python ``for t`` loop (parity / no SpikingJelly).
    """

    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("PyTorch is required for the SHD model") from exc
    if min(input_units, hidden_units, classes) <= 0:
        raise ValueError("model dimensions must be positive")
    if threshold <= 0 or surrogate_beta <= 0:
        raise ValueError("threshold and surrogate_beta must be positive")

    dynamics_name = str(dynamics or "leaky").strip().lower()
    if dynamics_name in {"chaki", "chaki_if", "if", "integrator", "alpha0_beta1"}:
        dynamics_name = "chaki_if"
    elif dynamics_name in {"leaky", "lif", "leaky_lif"}:
        dynamics_name = "leaky"
    else:
        raise ValueError(
            f"unsupported dynamics={dynamics!r}; use leaky|chaki_if"
        )

    if dynamics_name == "leaky":
        if tau <= 1:
            raise ValueError("tau must exceed 1 for leaky dynamics")
        alpha = 0.0  # no separate current state in this discretisation
        beta = 1.0 - 1.0 / float(tau)
    else:
        # Chaki Table I: α=0, β=1. tau is unused but kept for attribute parity.
        alpha = 0.0 if current_decay is None else float(current_decay)
        beta = 1.0 if voltage_decay is None else float(voltage_decay)
        if not math.isclose(alpha, 0.0) or not math.isclose(beta, 1.0):
            # Keep the door open for mild ablations, but only α=0/β=1 is tested.
            if alpha < 0.0 or alpha >= 1.0 or beta <= 0.0 or beta > 1.0:
                raise ValueError("current_decay/voltage_decay out of range for chaki_if")

    init_name = str(weight_init or "default").strip().lower()
    if init_name not in {"default", "chaki", "normal_1"}:
        raise ValueError(f"unsupported weight_init={weight_init!r}; use default|chaki")
    if init_name == "normal_1":
        init_name = "chaki"

    resolved = resolve_snn_backend(backend)
    use_sj = resolved in {"torch", "cupy", "triton"}

    class ATanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return (inputs >= 0).to(inputs.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (inputs,) = ctx.saved_tensors
            denominator = 1.0 + (
                math.pi * surrogate_beta * inputs / 2.0
            ).square()
            return grad_output * (surrogate_beta / 2.0) / denominator

    def _apply_weight_init(module: Any) -> None:
        if init_name != "chaki":
            return
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if parameter.ndim >= 2:
                    # Table I: mean 0, scale factor (std) 1.
                    parameter.normal_(mean=0.0, std=1.0)
                else:
                    parameter.zero_()

    def _membrane_step(membrane: Any, current: Any) -> Any:
        if dynamics_name == "chaki_if":
            # V[m+1] = β V[m] + I[m] with β=1, I = W x (α=0 ⇒ no current IIR).
            return beta * membrane + current
        return membrane + (current - membrane) / float(tau)

    def _lif_scan(current_bt: Any) -> Any:
        """LIF/IF over time for ``current_bt`` shaped ``[batch, time, hidden]``."""
        batch, timesteps, hidden = current_bt.shape
        membrane = current_bt.new_zeros((batch, hidden))
        spike_steps = []
        for step in range(timesteps):
            membrane = _membrane_step(membrane, current_bt[:, step])
            spikes = ATanSpike.apply(membrane - float(threshold))
            membrane = membrane * (1.0 - spikes.detach())
            spike_steps.append(spikes)
        return torch.stack(spike_steps, dim=1)

    if use_sj:
        from spikingjelly.activation_based import functional, layer, neuron, surrogate

        class FeedforwardLIFSHD(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_units = int(input_units)
                self.hidden_units = int(hidden_units)
                self.classes = int(classes)
                self.tau = float(tau)
                self.threshold = float(threshold)
                self.surrogate_beta = float(surrogate_beta)
                self.dynamics = dynamics_name
                self.current_decay = float(alpha)
                self.voltage_decay = float(beta) if dynamics_name == "chaki_if" else float(
                    1.0 - 1.0 / float(tau)
                )
                self.weight_init = init_name
                self.snn_backend = resolved
                self.architecture = "feedforward_lif_shd"
                # Multi-step Linear: one GEMM over the time dimension.
                self.input = layer.Linear(
                    input_units, hidden_units, bias=True, step_mode="m"
                )
                if dynamics_name == "chaki_if":
                    # IFNode: V ← V + x  (matches α=0, β=1 integrator).
                    self.lif = neuron.IFNode(
                        v_threshold=float(threshold),
                        v_reset=0.0,
                        surrogate_function=surrogate.ATan(alpha=float(surrogate_beta)),
                        detach_reset=True,
                        step_mode="m",
                        backend=resolved,
                        store_v_seq=False,
                    )
                else:
                    self.lif = neuron.LIFNode(
                        tau=float(tau),
                        decay_input=True,
                        v_threshold=float(threshold),
                        v_reset=0.0,
                        surrogate_function=surrogate.ATan(alpha=float(surrogate_beta)),
                        detach_reset=True,
                        step_mode="m",
                        backend=resolved,
                        store_v_seq=False,
                    )
                # Readout stays plain Linear so state_dict keys match legacy and
                # temporal-mean logits equal mean-then-readout for a linear layer.
                self.readout = nn.Linear(hidden_units, classes, bias=True)
                self.spike_parameter_layers = {
                    "input.weight": "hidden",
                    "input.bias": "hidden",
                    "readout.weight": "readout",
                    "readout.bias": "readout",
                }
                _apply_weight_init(self)

            def reset_state(self) -> None:
                functional.reset_net(self)

            def input_current_sequence(self, inputs: Any) -> Any:
                """Return input currents shaped ``[batch, time, hidden]``."""
                if inputs.ndim != 3 or inputs.shape[2] != self.input_units:
                    raise ValueError(
                        f"SHD model expects [batch,time,{self.input_units}] inputs"
                    )
                # SpikingJelly multi-step expects [T, B, *].
                x_seq = inputs.transpose(0, 1).contiguous()
                current_seq = self.input(x_seq)
                return current_seq.transpose(0, 1).contiguous()

            def forward(self, inputs, *, return_spikes: bool = False):
                if inputs.ndim != 3 or inputs.shape[2] != self.input_units:
                    raise ValueError(
                        f"SHD model expects [batch,time,{self.input_units}] inputs"
                    )
                self.reset_state()
                x_seq = inputs.transpose(0, 1).contiguous()  # [T,B,F]
                current_seq = self.input(x_seq)  # [T,B,H]
                spike_seq = self.lif(current_seq)  # [T,B,H]
                spikes_bt = spike_seq.transpose(0, 1).contiguous()  # [B,T,H]
                logits = self.readout(spikes_bt.mean(dim=1))
                if return_spikes:
                    return logits, spikes_bt
                return logits

        return FeedforwardLIFSHD()

    # Pure PyTorch paths (no SpikingJelly): multi-step GEMM, or legacy for-loop.
    class FeedforwardLIFSHD(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_units = int(input_units)
            self.hidden_units = int(hidden_units)
            self.classes = int(classes)
            self.tau = float(tau)
            self.threshold = float(threshold)
            self.surrogate_beta = float(surrogate_beta)
            self.dynamics = dynamics_name
            self.current_decay = float(alpha)
            self.voltage_decay = (
                float(beta)
                if dynamics_name == "chaki_if"
                else float(1.0 - 1.0 / float(tau))
            )
            self.weight_init = init_name
            self.snn_backend = resolved  # "legacy" or "pure"
            self.architecture = "feedforward_lif_shd"
            self.input = nn.Linear(input_units, hidden_units, bias=True)
            self.readout = nn.Linear(hidden_units, classes, bias=True)
            self.spike_parameter_layers = {
                "input.weight": "hidden",
                "input.bias": "hidden",
                "readout.weight": "readout",
                "readout.bias": "readout",
            }
            _apply_weight_init(self)

        def reset_state(self) -> None:
            return None

        def input_current_sequence(self, inputs: Any) -> Any:
            if inputs.ndim != 3 or inputs.shape[2] != self.input_units:
                raise ValueError(
                    f"SHD model expects [batch,time,{self.input_units}] inputs"
                )
            batch, timesteps, features = inputs.shape
            if self.snn_backend == "legacy":
                return torch.stack(
                    [self.input(inputs[:, step]) for step in range(timesteps)],
                    dim=1,
                )
            flat = inputs.reshape(batch * timesteps, features)
            return self.input(flat).reshape(batch, timesteps, self.hidden_units)

        def forward(self, inputs, *, return_spikes: bool = False):
            if inputs.ndim != 3 or inputs.shape[2] != self.input_units:
                raise ValueError(
                    f"SHD model expects [batch,time,{self.input_units}] inputs"
                )
            batch, timesteps, features = inputs.shape
            if self.snn_backend == "legacy":
                membrane = inputs.new_zeros((batch, self.hidden_units))
                spike_steps = []
                logits = inputs.new_zeros((batch, self.classes))
                for step in range(timesteps):
                    current = self.input(inputs[:, step])
                    membrane = _membrane_step(membrane, current)
                    spikes = ATanSpike.apply(membrane - self.threshold)
                    membrane = membrane * (1.0 - spikes.detach())
                    logits = logits + self.readout(spikes)
                    spike_steps.append(spikes)
                logits = logits / timesteps
                if return_spikes:
                    return logits, torch.stack(spike_steps, dim=1)
                return logits

            # Multi-step pure torch: one Linear over B*T, then elementwise LIF/IF.
            current_bt = self.input(inputs.reshape(batch * timesteps, features)).reshape(
                batch, timesteps, self.hidden_units
            )
            spikes_bt = _lif_scan(current_bt)
            logits = self.readout(spikes_bt.mean(dim=1))
            if return_spikes:
                return logits, spikes_bt
            return logits

    return FeedforwardLIFSHD()
