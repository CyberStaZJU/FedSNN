from __future__ import annotations


def build_fedlec_vgg9(
    timesteps: int = 4,
    classes: int = 10,
    tau: float = 2.0,
    threshold: float = 1.0,
    surrogate_beta: float = 2.0,
    track_runtime_activity: bool = True,
):
    """Device-agnostic port of FedLEC's official multi-step S-VGG9.

    The official model repeats each static image across ``T`` steps, uses one
    shared BatchNorm per convolution (not BNTT), max pooling, bias-free weight
    layers, ATan-surrogate LIF neurons, and mean output voltage as logits. The
    implementation is stateless across calls, matching the upstream
    ``functional.reset_net`` after every batch while remaining usable on CUDA,
    NPU, and CPU without a runtime SpikingJelly dependency.
    """
    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("PyTorch is required for FedLEC models") from exc

    if timesteps <= 0 or classes <= 0:
        raise ValueError("timesteps and classes must be positive")
    if tau <= 1.0 or threshold <= 0 or surrogate_beta <= 0:
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

    class FedLECVGG9(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = timesteps
            self.classes = classes
            self.tau = tau
            self.threshold = threshold
            channels = (64, 64, 128, 128, 256, 256, 256)
            in_channels = (3, *channels[:-1])
            self.convs = nn.ModuleList(
                nn.Conv2d(source, target, 3, padding=1, bias=False)
                for source, target in zip(in_channels, channels)
            )
            self.norms = nn.ModuleList(nn.BatchNorm2d(channel) for channel in channels)
            self.fc1 = nn.Linear(4 * 4 * 256, 1024, bias=False)
            self.fc2 = nn.Linear(1024, classes, bias=False)
            self.pool_after = frozenset({1, 3, 6})
            self.spike_layer_sizes = (
                64 * 32 * 32,
                64 * 32 * 32,
                128 * 16 * 16,
                128 * 16 * 16,
                256 * 8 * 8,
                256 * 8 * 8,
                256 * 8 * 8,
                1024,
            )
            self.last_firing_rate = 0.0
            self.last_sample_firing_rate = None

        @staticmethod
        def _time_flatten(values):
            return values.flatten(0, 1)

        def _lif_sequence(self, currents):
            membrane = torch.zeros_like(currents[0])
            spikes = []
            for step in range(self.timesteps):
                membrane = membrane + (currents[step] - membrane) / self.tau
                output = ATanSpike.apply(membrane - self.threshold)
                membrane = membrane * (1.0 - output.detach())
                spikes.append(output)
            return torch.stack(spikes, dim=0)

        @staticmethod
        def _batch_norm(norm, values):
            flat = values.flatten(0, 1)
            if norm.training and flat.shape[0] == 1:
                # Only reachable in degenerate one-step smoke tests. Real FedLEC
                # runs have T*B > 1 and follow ordinary training-mode BatchNorm.
                flat = nn.functional.batch_norm(
                    flat,
                    norm.running_mean,
                    norm.running_var,
                    norm.weight,
                    norm.bias,
                    training=False,
                    momentum=norm.momentum,
                    eps=norm.eps,
                )
            else:
                flat = norm(flat)
            return flat.reshape(values.shape[0], values.shape[1], *flat.shape[1:])

        def forward(
            self,
            inputs,
            return_activity: bool = False,
            return_layer_activity: bool = False,
        ):
            if inputs.ndim != 4 or inputs.shape[1:] != (3, 32, 32):
                raise ValueError("FedLEC VGG9 expects NCHW CIFAR images")
            batch = inputs.shape[0]
            output = inputs.unsqueeze(0).expand(self.timesteps, *inputs.shape)
            collect_activity = (
                track_runtime_activity or return_activity or return_layer_activity
            )
            layer_activity = []

            for index, (convolution, norm) in enumerate(zip(self.convs, self.norms)):
                current = convolution(self._time_flatten(output)).reshape(
                    self.timesteps, batch, -1, output.shape[-2], output.shape[-1]
                )
                output = self._lif_sequence(self._batch_norm(norm, current))
                if collect_activity:
                    layer_activity.append(output.flatten(2).mean(2).mean(0))
                if index in self.pool_after:
                    pooled = nn.functional.max_pool2d(self._time_flatten(output), 2)
                    output = pooled.reshape(
                        self.timesteps, batch, *pooled.shape[1:]
                    )

            output = output.flatten(2)
            current = self.fc1(self._time_flatten(output)).reshape(
                self.timesteps, batch, 1024
            )
            output = self._lif_sequence(current)
            if collect_activity:
                layer_activity.append(output.mean(2).mean(0))
            logits = self.fc2(self._time_flatten(output)).reshape(
                self.timesteps, batch, self.classes
            ).mean(0)

            if collect_activity:
                layer_rates = torch.stack(layer_activity, dim=1)
                sample_rates = layer_rates.mean(dim=1)
                if track_runtime_activity:
                    self.last_sample_firing_rate = sample_rates.detach()
                    self.last_firing_rate = float(sample_rates.mean().detach().cpu())
                if return_layer_activity:
                    return logits, layer_rates
                if return_activity:
                    return logits, sample_rates
            return logits

    return FedLECVGG9()


def build_snn_cifar10(
    timesteps: int = 12,
    classes: int = 10,
    surrogate_beta: float = 2.0,
    track_runtime_activity: bool = True,
):
    """Build the SFedCA-style AlexNet reconstruction under the shared name.

    This is the former ``sfedca_alexnet`` implementation: five biased
    convolutions, a biased 1024-unit hidden classifier, no normalization,
    Poisson input, max pooling, ATan-surrogate IF neurons with hard reset, and
    output spike counts as logits. The per-layer activity interface is retained
    for Credit-TopK-SAW without changing the model's forward semantics.
    """
    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("PyTorch is required for SNN models") from exc
    if timesteps <= 0 or classes <= 0 or surrogate_beta <= 0:
        raise ValueError("timesteps, classes, and surrogate_beta must be positive")

    class ATanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return (inputs >= 0).to(inputs.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (inputs,) = ctx.saved_tensors
            denominator = 1.0 + (math.pi * surrogate_beta * inputs / 2.0).square()
            return grad_output * (surrogate_beta / 2.0) / denominator

    class SNNCIFAR10(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = timesteps
            self.classes = classes
            self.threshold = 1.0
            channels = (64, 192, 384, 256, 256)
            in_channels = (3, *channels[:-1])
            self.convs = nn.ModuleList(
                nn.Conv2d(source, target, 3, padding=1, bias=True)
                for source, target in zip(in_channels, channels)
            )
            self.fc1 = nn.Linear(256 * 4 * 4, 1024, bias=True)
            self.fc2 = nn.Linear(1024, classes, bias=True)
            self.spike_layer_sizes = (
                64 * 32 * 32,
                192 * 16 * 16,
                384 * 8 * 8,
                256 * 8 * 8,
                256 * 8 * 8,
                1024,
                classes,
            )
            self.last_firing_rate = 0.0
            self.last_sample_firing_rate = None

        def _if(self, current, membrane):
            charged = membrane + current
            spikes = ATanSpike.apply(charged - self.threshold)
            membrane = charged * (1.0 - spikes.detach())
            return spikes, membrane

        def forward(
            self,
            inputs,
            return_activity: bool = False,
            return_layer_activity: bool = False,
        ):
            if inputs.ndim != 4 or inputs.shape[1:] != (3, 32, 32):
                raise ValueError("SNN-CIFAR10 expects NCHW CIFAR images")
            if track_runtime_activity and (
                inputs.min().detach().item() < 0
                or inputs.max().detach().item() > 1
            ):
                raise ValueError("SNN-CIFAR10 Poisson encoding expects pixels in [0, 1]")
            batch = inputs.shape[0]
            shapes = [(64, 32), (192, 16), (384, 8), (256, 8), (256, 8)]
            membranes = [inputs.new_zeros((batch, c, s, s)) for c, s in shapes]
            membrane_fc1 = inputs.new_zeros((batch, 1024))
            membrane_fc2 = inputs.new_zeros((batch, classes))
            logits = inputs.new_zeros((batch, classes))
            layer_activity = [inputs.new_zeros(batch) for _ in self.spike_layer_sizes]

            for _ in range(self.timesteps):
                output = (torch.rand_like(inputs) <= inputs).to(inputs.dtype)
                for index, convolution in enumerate(self.convs):
                    current = convolution(output)
                    output, membranes[index] = self._if(current, membranes[index])
                    layer_activity[index] = (
                        layer_activity[index] + output.flatten(1).mean(1)
                    )
                    if index in (0, 1, 4):
                        output = nn.functional.max_pool2d(output, 2)
                output = output.flatten(1)
                hidden_current = self.fc1(output)
                output, membrane_fc1 = self._if(hidden_current, membrane_fc1)
                layer_activity[5] = layer_activity[5] + output.mean(1)
                output, membrane_fc2 = self._if(self.fc2(output), membrane_fc2)
                logits = logits + output
                layer_activity[6] = layer_activity[6] + output.mean(1)

            layer_rates = torch.stack(layer_activity, dim=1) / self.timesteps
            sample_rates = layer_rates.mean(dim=1)
            if track_runtime_activity:
                self.last_sample_firing_rate = sample_rates.detach()
                self.last_firing_rate = float(sample_rates.mean().detach().cpu())
            if return_layer_activity:
                return logits, layer_rates
            if return_activity:
                return logits, sample_rates
            return logits

    return SNNCIFAR10()


def build_sfedca_vgg9(
    timesteps: int = 12,
    classes: int = 10,
    surrogate_beta: float = 2.0,
    track_runtime_activity: bool = True,
):
    """Reconstruct a VGG-9 SNN for CIFAR-10 under the shared SFedCA SNN recipe.

    Same methodology as ``build_sfedca_vgg5`` / ``build_snn_cifar10``: IF
    neurons (threshold=1.0, hard reset), ATan-surrogate (beta=2.0), Poisson
    input encoding, no normalization, biased convolutions, max pooling, and
    spike-count accumulation as logits. The convolution skeleton follows
    FedLEC's VGG-9 channel layout (64-64-128-128-256-256-256, max pooling after
    convs 1/3/6) but the BN-free LIF / static-repeat recipe is replaced by the
    SFedCA SNN recipe so the backbone can be compared apples-to-apples with the
    other SFedCA-protocol nets. The per-layer activity interface is retained for
    Credit-TopK-SAW firing-rate credit.
    """
    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("PyTorch is required for SFedCA models") from exc
    if timesteps <= 0 or classes <= 0 or surrogate_beta <= 0:
        raise ValueError("timesteps, classes, and surrogate_beta must be positive")

    class ATanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return (inputs >= 0).to(inputs.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (inputs,) = ctx.saved_tensors
            denominator = 1.0 + (math.pi * surrogate_beta * inputs / 2.0).square()
            return grad_output * (surrogate_beta / 2.0) / denominator

    class SFedCAVGG9(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = timesteps
            self.classes = classes
            self.threshold = 1.0
            channels = (64, 64, 128, 128, 256, 256, 256)
            in_channels = (3, *channels[:-1])
            self.convs = nn.ModuleList(
                nn.Conv2d(source, target, 3, padding=1, bias=True)
                for source, target in zip(in_channels, channels)
            )
            self.fc1 = nn.Linear(256 * 4 * 4, 1024, bias=True)
            self.fc2 = nn.Linear(1024, classes, bias=True)
            self.pool_after = frozenset({1, 3, 6})
            self.spike_layer_sizes = (
                64 * 32 * 32,
                64 * 32 * 32,
                128 * 16 * 16,
                128 * 16 * 16,
                256 * 8 * 8,
                256 * 8 * 8,
                256 * 8 * 8,
                1024,
                classes,
            )
            self.last_firing_rate = 0.0
            self.last_sample_firing_rate = None

        def _if(self, current, membrane):
            charged = membrane + current
            spikes = ATanSpike.apply(charged - self.threshold)
            return spikes, charged * (1.0 - spikes.detach())

        def forward(
            self,
            inputs,
            return_activity: bool = False,
            return_layer_activity: bool = False,
        ):
            if inputs.ndim != 4 or inputs.shape[1:] != (3, 32, 32):
                raise ValueError("SFedCA VGG-9 expects NCHW CIFAR images")
            if track_runtime_activity and (
                inputs.min().detach().item() < 0
                or inputs.max().detach().item() > 1
            ):
                raise ValueError("SFedCA Poisson encoding expects pixels in [0, 1]")
            batch = inputs.shape[0]
            conv_shapes = [
                (64, 32, 32),
                (64, 32, 32),
                (128, 16, 16),
                (128, 16, 16),
                (256, 8, 8),
                (256, 8, 8),
                (256, 8, 8),
            ]
            membranes = [inputs.new_zeros((batch, *s)) for s in conv_shapes]
            membrane_fc1 = inputs.new_zeros((batch, 1024))
            membrane_fc2 = inputs.new_zeros((batch, classes))
            logits = inputs.new_zeros((batch, classes))
            layer_activity = [inputs.new_zeros(batch) for _ in self.spike_layer_sizes]

            for _ in range(self.timesteps):
                output = (torch.rand_like(inputs) <= inputs).to(inputs.dtype)
                for index, convolution in enumerate(self.convs):
                    current = convolution(output)
                    output, membranes[index] = self._if(current, membranes[index])
                    layer_activity[index] = (
                        layer_activity[index] + output.flatten(1).mean(1)
                    )
                    if index in self.pool_after:
                        output = nn.functional.max_pool2d(output, 2)
                output = output.flatten(1)
                hidden_current = self.fc1(output)
                output, membrane_fc1 = self._if(hidden_current, membrane_fc1)
                layer_activity[7] = layer_activity[7] + output.mean(1)
                output, membrane_fc2 = self._if(self.fc2(output), membrane_fc2)
                logits = logits + output
                layer_activity[8] = layer_activity[8] + output.mean(1)

            layer_rates = torch.stack(layer_activity, dim=1) / self.timesteps
            sample_rates = layer_rates.mean(dim=1)
            if track_runtime_activity:
                self.last_sample_firing_rate = sample_rates.detach()
                self.last_firing_rate = float(sample_rates.mean().detach().cpu())
            if return_layer_activity:
                return logits, layer_rates
            if return_activity:
                return logits, sample_rates
            return logits

    return SFedCAVGG9()


def build_sfedca_mnist(
    timesteps: int = 12,
    classes: int = 10,
    surrogate_beta: float = 2.0,
    track_runtime_activity: bool = True,
):
    """Reconstruct SFedCA's two-convolution, two-FC MNIST SNN.

    The paper reports only the layer types. We use conventional 32/64
    convolution channels and a 128-unit hidden layer, with IF hard reset,
    Poisson input encoding, max pooling, and the paper's ATan surrogate.
    """
    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for SFedCA models") from exc
    if timesteps <= 0 or surrogate_beta <= 0:
        raise ValueError("timesteps and surrogate_beta must be positive")

    class ATanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return (inputs >= 0).to(inputs.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (inputs,) = ctx.saved_tensors
            denominator = 1.0 + (math.pi * surrogate_beta * inputs / 2.0).square()
            return grad_output * (surrogate_beta / 2.0) / denominator

    class SFedCAMNIST(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = timesteps
            self.threshold = 1.0
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, classes)
            self.spike_layer_sizes = (32 * 28 * 28, 64 * 14 * 14, 128, classes)
            self.last_firing_rate = 0.0
            self.last_sample_firing_rate = None

        def _if(self, current, membrane):
            charged = membrane + current
            spikes = ATanSpike.apply(charged - self.threshold)
            return spikes, charged * (1.0 - spikes.detach())

        def forward(
            self,
            inputs,
            return_activity: bool = False,
            return_layer_activity: bool = False,
        ):
            if inputs.ndim != 4 or inputs.shape[1:] != (1, 28, 28):
                raise ValueError("SFedCA MNIST model expects NCHW inputs of shape (1, 28, 28)")
            if track_runtime_activity and (
                inputs.min().detach().item() < 0
                or inputs.max().detach().item() > 1
            ):
                raise ValueError("SFedCA Poisson encoding expects pixels in [0, 1]")
            batch = inputs.shape[0]
            membrane1 = inputs.new_zeros((batch, 32, 28, 28))
            membrane2 = inputs.new_zeros((batch, 64, 14, 14))
            membrane3 = inputs.new_zeros((batch, 128))
            membrane4 = inputs.new_zeros((batch, classes))
            logits = inputs.new_zeros((batch, classes))
            layer_activity = [inputs.new_zeros(batch) for _ in self.spike_layer_sizes]

            for _ in range(self.timesteps):
                output = (torch.rand_like(inputs) <= inputs).to(inputs.dtype)
                output, membrane1 = self._if(self.conv1(output), membrane1)
                layer_activity[0] = layer_activity[0] + output.flatten(1).mean(1)
                output = nn.functional.max_pool2d(output, 2)
                output, membrane2 = self._if(self.conv2(output), membrane2)
                layer_activity[1] = layer_activity[1] + output.flatten(1).mean(1)
                output = nn.functional.max_pool2d(output, 2).flatten(1)
                output, membrane3 = self._if(self.fc1(output), membrane3)
                layer_activity[2] = layer_activity[2] + output.mean(1)
                output, membrane4 = self._if(self.fc2(output), membrane4)
                layer_activity[3] = layer_activity[3] + output.mean(1)
                logits = logits + output

            layer_rates = torch.stack(layer_activity, dim=1) / self.timesteps
            sample_rates = layer_rates.mean(dim=1)
            if track_runtime_activity:
                self.last_sample_firing_rate = sample_rates.detach()
                self.last_firing_rate = float(sample_rates.mean().detach().cpu())
            if return_layer_activity:
                return logits, layer_rates
            if return_activity:
                return logits, sample_rates
            return logits

    return SFedCAMNIST()


def build_sfedca_vgg5(
    timesteps: int = 12,
    classes: int = 10,
    surrogate_beta: float = 2.0,
    track_runtime_activity: bool = True,
):
    """Reconstruct SFedCA's VGG-5 SNN for Fashion-MNIST.

    The convolution layout follows the FedSNN upstream VGG-5 definition
    (64-pool-128-128-pool). SFedCA does not publish its classifier width, so
    this reconstruction uses one 1024-unit hidden FC layer before the output.
    """
    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for SFedCA models") from exc
    if timesteps <= 0 or surrogate_beta <= 0:
        raise ValueError("timesteps and surrogate_beta must be positive")

    class ATanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return (inputs >= 0).to(inputs.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (inputs,) = ctx.saved_tensors
            denominator = 1.0 + (math.pi * surrogate_beta * inputs / 2.0).square()
            return grad_output * (surrogate_beta / 2.0) / denominator

    class SFedCAVGG5(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = timesteps
            self.threshold = 1.0
            self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
            self.fc1 = nn.Linear(128 * 7 * 7, 1024)
            self.fc2 = nn.Linear(1024, classes)
            self.spike_layer_sizes = (
                64 * 28 * 28,
                128 * 14 * 14,
                128 * 14 * 14,
                1024,
                classes,
            )
            self.last_firing_rate = 0.0
            self.last_sample_firing_rate = None

        def _if(self, current, membrane):
            charged = membrane + current
            spikes = ATanSpike.apply(charged - self.threshold)
            return spikes, charged * (1.0 - spikes.detach())

        def forward(
            self,
            inputs,
            return_activity: bool = False,
            return_layer_activity: bool = False,
        ):
            if inputs.ndim != 4 or inputs.shape[1:] != (1, 28, 28):
                raise ValueError("SFedCA VGG-5 expects NCHW inputs of shape (1, 28, 28)")
            if track_runtime_activity and (
                inputs.min().detach().item() < 0
                or inputs.max().detach().item() > 1
            ):
                raise ValueError("SFedCA Poisson encoding expects pixels in [0, 1]")
            batch = inputs.shape[0]
            membranes = [
                inputs.new_zeros((batch, 64, 28, 28)),
                inputs.new_zeros((batch, 128, 14, 14)),
                inputs.new_zeros((batch, 128, 14, 14)),
                inputs.new_zeros((batch, 1024)),
                inputs.new_zeros((batch, classes)),
            ]
            logits = inputs.new_zeros((batch, classes))
            layer_activity = [inputs.new_zeros(batch) for _ in self.spike_layer_sizes]

            for _ in range(self.timesteps):
                output = (torch.rand_like(inputs) <= inputs).to(inputs.dtype)
                output, membranes[0] = self._if(self.conv1(output), membranes[0])
                layer_activity[0] = layer_activity[0] + output.flatten(1).mean(1)
                output = nn.functional.max_pool2d(output, 2)
                output, membranes[1] = self._if(self.conv2(output), membranes[1])
                layer_activity[1] = layer_activity[1] + output.flatten(1).mean(1)
                output, membranes[2] = self._if(self.conv3(output), membranes[2])
                layer_activity[2] = layer_activity[2] + output.flatten(1).mean(1)
                output = nn.functional.max_pool2d(output, 2).flatten(1)
                output, membranes[3] = self._if(self.fc1(output), membranes[3])
                layer_activity[3] = layer_activity[3] + output.mean(1)
                output, membranes[4] = self._if(self.fc2(output), membranes[4])
                layer_activity[4] = layer_activity[4] + output.mean(1)
                logits = logits + output

            layer_rates = torch.stack(layer_activity, dim=1) / self.timesteps
            sample_rates = layer_rates.mean(dim=1)
            if track_runtime_activity:
                self.last_sample_firing_rate = sample_rates.detach()
                self.last_firing_rate = float(sample_rates.mean().detach().cpu())
            if return_layer_activity:
                return logits, layer_rates
            if return_activity:
                return logits, sample_rates
            return logits

    return SFedCAVGG5()


def build_fedsnn_mnist_bntt(
    timesteps: int = 3,
    classes: int = 10,
    surrogate_beta: float = 2.0,
    threshold: float = 1.0,
    membrane_decay: float = 0.95,
    bntt_eps: float = 1e-4,
    bntt_momentum: float = 0.1,
    track_runtime_activity: bool = True,
):
    """Build an independent two-convolution, two-FC MNIST SNN with BNTT."""
    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for MNIST+BNTT") from exc
    if timesteps <= 0 or classes <= 0 or surrogate_beta <= 0 or threshold <= 0:
        raise ValueError("timesteps, classes, surrogate_beta, and threshold must be positive")
    if not 0.0 <= membrane_decay <= 1.0:
        raise ValueError("membrane_decay must be in [0, 1]")

    class ATanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return (inputs >= 0).to(inputs.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (inputs,) = ctx.saved_tensors
            denominator = 1.0 + (math.pi * surrogate_beta * inputs / 2.0).square()
            return grad_output * (surrogate_beta / 2.0) / denominator

    class FedSNNMNISTBNTT(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = timesteps
            self.threshold = threshold
            self.membrane_decay = membrane_decay
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
            self.fc1 = nn.Linear(64 * 7 * 7, 128, bias=False)
            self.fc2 = nn.Linear(128, classes, bias=False)
            self.bntt1 = nn.ModuleList([
                nn.BatchNorm2d(32, eps=bntt_eps, momentum=bntt_momentum)
                for _ in range(timesteps)
            ])
            self.bntt2 = nn.ModuleList([
                nn.BatchNorm2d(64, eps=bntt_eps, momentum=bntt_momentum)
                for _ in range(timesteps)
            ])
            self.bntt_fc1 = nn.ModuleList([
                nn.BatchNorm1d(128, eps=bntt_eps, momentum=bntt_momentum)
                for _ in range(timesteps)
            ])
            self.spike_layer_sizes = (32 * 28 * 28, 64 * 14 * 14, 128)
            self.spike_channel_sizes = (32, 64, 128)
            self.structured_spike_parameter_map = {
                "conv1.weight": 0,
                "conv2.weight": 1,
                "fc1.weight": 2,
            }
            self.last_firing_rate = 0.0
            self.last_sample_firing_rate = None

        @staticmethod
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

        def _lif(self, current, membrane):
            charged = self.membrane_decay * membrane + current
            spikes = ATanSpike.apply(charged - self.threshold)
            return spikes, charged - spikes.detach() * self.threshold

        def forward(
            self,
            inputs,
            return_activity: bool = False,
            return_layer_activity: bool = False,
            return_neuron_activity: bool = False,
            return_neuron_patterns: bool = False,
        ):
            if inputs.ndim != 4 or inputs.shape[1:] != (1, 28, 28):
                raise ValueError("MNIST+BNTT expects NCHW inputs of shape (1, 28, 28)")
            if return_neuron_activity and return_neuron_patterns:
                raise ValueError(
                    "return_neuron_activity and return_neuron_patterns are mutually exclusive"
                )
            if track_runtime_activity and (
                inputs.min().detach().item() < 0
                or inputs.max().detach().item() > 1
            ):
                raise ValueError("MNIST+BNTT Poisson encoding expects pixels in [0, 1]")
            batch = inputs.shape[0]
            membranes = [
                inputs.new_zeros((batch, 32, 28, 28)),
                inputs.new_zeros((batch, 64, 14, 14)),
                inputs.new_zeros((batch, 128)),
                inputs.new_zeros((batch, classes)),
            ]
            logits = inputs.new_zeros((batch, classes))
            layer_activity = [inputs.new_zeros(batch) for _ in self.spike_layer_sizes]
            neuron_activity = (
                [inputs.new_zeros((batch, channels)) for channels in self.spike_channel_sizes]
                if return_neuron_activity
                else None
            )
            neuron_patterns = (
                [[] for _ in self.spike_channel_sizes] if return_neuron_patterns else None
            )

            for timestep in range(self.timesteps):
                output = (torch.rand_like(inputs) <= inputs).to(inputs.dtype)
                current = self._apply_bntt(self.bntt1[timestep], self.conv1(output))
                output, membranes[0] = self._lif(current, membranes[0])
                if neuron_activity is not None:
                    neuron_activity[0] = neuron_activity[0] + output.flatten(2).mean(2)
                if neuron_patterns is not None:
                    neuron_patterns[0].append(output)
                layer_activity[0] = layer_activity[0] + output.flatten(1).mean(1)
                output = nn.functional.avg_pool2d(output, 2)
                current = self._apply_bntt(self.bntt2[timestep], self.conv2(output))
                output, membranes[1] = self._lif(current, membranes[1])
                if neuron_activity is not None:
                    neuron_activity[1] = neuron_activity[1] + output.flatten(2).mean(2)
                if neuron_patterns is not None:
                    neuron_patterns[1].append(output)
                layer_activity[1] = layer_activity[1] + output.flatten(1).mean(1)
                output = nn.functional.avg_pool2d(output, 2).flatten(1)
                current = self._apply_bntt(self.bntt_fc1[timestep], self.fc1(output))
                output, membranes[2] = self._lif(current, membranes[2])
                if neuron_activity is not None:
                    neuron_activity[2] = neuron_activity[2] + output
                if neuron_patterns is not None:
                    neuron_patterns[2].append(output)
                layer_activity[2] = layer_activity[2] + output.mean(1)
                membranes[3] = membranes[3] + self.fc2(output)
                logits = logits + membranes[3]

            logits = logits / self.timesteps
            layer_rates = torch.stack(layer_activity, dim=1) / self.timesteps
            sample_rates = layer_rates.mean(dim=1)
            if track_runtime_activity:
                self.last_sample_firing_rate = sample_rates.detach()
                self.last_firing_rate = float(sample_rates.mean().detach().cpu())
            if return_neuron_activity:
                return logits, tuple(rate / self.timesteps for rate in neuron_activity)
            if return_neuron_patterns:
                return logits, tuple(
                    torch.stack(patterns, dim=1) for patterns in neuron_patterns
                )
            if return_layer_activity:
                return logits, layer_rates
            if return_activity:
                return logits, sample_rates
            return logits

    return FedSNNMNISTBNTT()


def build_mnist_2conv2fc_bntt(
    timesteps: int = 12,
    classes: int = 10,
    surrogate_beta: float = 2.0,
    threshold: float = 1.0,
    membrane_decay: float = 0.95,
    bntt_eps: float = 1e-4,
    bntt_momentum: float = 0.1,
    track_runtime_activity: bool = True,
):
    """Build the unified two-convolution, two-FC MNIST SNN with BNTT."""
    return build_fedsnn_mnist_bntt(
        timesteps=timesteps,
        classes=classes,
        surrogate_beta=surrogate_beta,
        threshold=threshold,
        membrane_decay=membrane_decay,
        bntt_eps=bntt_eps,
        bntt_momentum=bntt_momentum,
        track_runtime_activity=track_runtime_activity,
    )


def build_fashion_mnist_2conv2fc_bntt(
    timesteps: int = 12,
    classes: int = 10,
    surrogate_beta: float = 2.0,
    threshold: float = 1.0,
    membrane_decay: float = 0.95,
    bntt_eps: float = 1e-4,
    bntt_momentum: float = 0.1,
    track_runtime_activity: bool = True,
):
    """Build the Fashion-MNIST semantic alias of the shared 2Conv2FC+BNTT model."""
    return build_mnist_2conv2fc_bntt(
        timesteps=timesteps,
        classes=classes,
        surrogate_beta=surrogate_beta,
        threshold=threshold,
        membrane_decay=membrane_decay,
        bntt_eps=bntt_eps,
        bntt_momentum=bntt_momentum,
        track_runtime_activity=track_runtime_activity,
    )


def build_fashion_mnist_vgg5_bntt(
    timesteps: int = 4,
    classes: int = 10,
    surrogate_beta: float = 2.0,
    threshold: float = 1.0,
    membrane_decay: float = 0.95,
    bntt_eps: float = 1e-4,
    bntt_momentum: float = 0.1,
    track_runtime_activity: bool = True,
):
    """Fashion-MNIST VGG-5 + BNTT (1×28×28).

    Topology matches SFedCA VGG-5 (64-pool-128-128-pool + FC1024 + readout),
    while the training recipe matches other FedSNN BNTT models: Poisson
    encoding, bias-free weights, per-timestep BatchNorm, avg-pool, LIF with
    soft reset, mean-voltage logits. Spike layers: conv1/2/3 + fc1 (fc2 is
    non-spiking readout).
    """
    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for Fashion VGG-5+BNTT") from exc
    if timesteps <= 0 or surrogate_beta <= 0 or threshold <= 0:
        raise ValueError("timesteps, surrogate_beta, and threshold must be positive")
    if not 0.0 <= membrane_decay <= 1.0:
        raise ValueError("membrane_decay must be in [0, 1]")

    class ATanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return (inputs >= 0).to(inputs.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (inputs,) = ctx.saved_tensors
            denominator = 1.0 + (math.pi * surrogate_beta * inputs / 2.0).square()
            return grad_output * (surrogate_beta / 2.0) / denominator

    class FashionMNISTVGG5BNTT(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = timesteps
            self.threshold = threshold
            self.membrane_decay = membrane_decay
            self.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
            self.conv3 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
            self.fc1 = nn.Linear(128 * 7 * 7, 1024, bias=False)
            self.fc2 = nn.Linear(1024, classes, bias=False)
            self.bntt1 = nn.ModuleList(
                [
                    nn.BatchNorm2d(64, eps=bntt_eps, momentum=bntt_momentum)
                    for _ in range(timesteps)
                ]
            )
            self.bntt2 = nn.ModuleList(
                [
                    nn.BatchNorm2d(128, eps=bntt_eps, momentum=bntt_momentum)
                    for _ in range(timesteps)
                ]
            )
            self.bntt3 = nn.ModuleList(
                [
                    nn.BatchNorm2d(128, eps=bntt_eps, momentum=bntt_momentum)
                    for _ in range(timesteps)
                ]
            )
            self.bntt_fc1 = nn.ModuleList(
                [
                    nn.BatchNorm1d(1024, eps=bntt_eps, momentum=bntt_momentum)
                    for _ in range(timesteps)
                ]
            )
            self.spike_layer_sizes = (
                64 * 28 * 28,
                128 * 14 * 14,
                128 * 14 * 14,
                1024,
            )
            self.spike_channel_sizes = (64, 128, 128, 1024)
            self.structured_spike_parameter_map = {
                "conv1.weight": 0,
                "conv2.weight": 1,
                "conv3.weight": 2,
                "fc1.weight": 3,
            }
            self.last_firing_rate = 0.0
            self.last_sample_firing_rate = None

        @staticmethod
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

        def _lif(self, current, membrane):
            charged = self.membrane_decay * membrane + current
            spikes = ATanSpike.apply(charged - self.threshold)
            return spikes, charged - spikes.detach() * self.threshold

        def forward(
            self,
            inputs,
            return_activity: bool = False,
            return_layer_activity: bool = False,
            return_neuron_activity: bool = False,
            return_neuron_patterns: bool = False,
        ):
            if inputs.ndim != 4 or inputs.shape[1:] != (1, 28, 28):
                raise ValueError(
                    "Fashion VGG-5+BNTT expects NCHW inputs of shape (1, 28, 28)"
                )
            if return_neuron_activity and return_neuron_patterns:
                raise ValueError(
                    "return_neuron_activity and return_neuron_patterns are mutually exclusive"
                )
            if track_runtime_activity and (
                inputs.min().detach().item() < 0 or inputs.max().detach().item() > 1
            ):
                raise ValueError(
                    "Fashion VGG-5+BNTT Poisson encoding expects pixels in [0, 1]"
                )
            batch = inputs.shape[0]
            membranes = [
                inputs.new_zeros((batch, 64, 28, 28)),
                inputs.new_zeros((batch, 128, 14, 14)),
                inputs.new_zeros((batch, 128, 14, 14)),
                inputs.new_zeros((batch, 1024)),
                inputs.new_zeros((batch, classes)),
            ]
            logits = inputs.new_zeros((batch, classes))
            layer_activity = [inputs.new_zeros(batch) for _ in self.spike_layer_sizes]
            neuron_activity = (
                [
                    inputs.new_zeros((batch, channels))
                    for channels in self.spike_channel_sizes
                ]
                if return_neuron_activity
                else None
            )
            neuron_patterns = (
                [[] for _ in self.spike_channel_sizes] if return_neuron_patterns else None
            )

            for timestep in range(self.timesteps):
                output = (torch.rand_like(inputs) <= inputs).to(inputs.dtype)
                current = self._apply_bntt(self.bntt1[timestep], self.conv1(output))
                output, membranes[0] = self._lif(current, membranes[0])
                if neuron_activity is not None:
                    neuron_activity[0] = neuron_activity[0] + output.flatten(2).mean(2)
                if neuron_patterns is not None:
                    neuron_patterns[0].append(output)
                layer_activity[0] = layer_activity[0] + output.flatten(1).mean(1)
                output = nn.functional.avg_pool2d(output, 2)
                current = self._apply_bntt(self.bntt2[timestep], self.conv2(output))
                output, membranes[1] = self._lif(current, membranes[1])
                if neuron_activity is not None:
                    neuron_activity[1] = neuron_activity[1] + output.flatten(2).mean(2)
                if neuron_patterns is not None:
                    neuron_patterns[1].append(output)
                layer_activity[1] = layer_activity[1] + output.flatten(1).mean(1)
                current = self._apply_bntt(self.bntt3[timestep], self.conv3(output))
                output, membranes[2] = self._lif(current, membranes[2])
                if neuron_activity is not None:
                    neuron_activity[2] = neuron_activity[2] + output.flatten(2).mean(2)
                if neuron_patterns is not None:
                    neuron_patterns[2].append(output)
                layer_activity[2] = layer_activity[2] + output.flatten(1).mean(1)
                output = nn.functional.avg_pool2d(output, 2).flatten(1)
                current = self._apply_bntt(self.bntt_fc1[timestep], self.fc1(output))
                output, membranes[3] = self._lif(current, membranes[3])
                if neuron_activity is not None:
                    neuron_activity[3] = neuron_activity[3] + output
                if neuron_patterns is not None:
                    neuron_patterns[3].append(output)
                layer_activity[3] = layer_activity[3] + output.mean(1)
                membranes[4] = membranes[4] + self.fc2(output)
                logits = logits + membranes[4]

            logits = logits / self.timesteps
            layer_rates = torch.stack(layer_activity, dim=1) / self.timesteps
            sample_rates = layer_rates.mean(dim=1)
            if track_runtime_activity:
                self.last_sample_firing_rate = sample_rates.detach()
                self.last_firing_rate = float(sample_rates.mean().detach().cpu())
            if return_neuron_activity:
                return logits, tuple(rate / self.timesteps for rate in neuron_activity)
            if return_neuron_patterns:
                return logits, tuple(
                    torch.stack(patterns, dim=1) for patterns in neuron_patterns
                )
            if return_layer_activity:
                return logits, layer_rates
            if return_activity:
                return logits, sample_rates
            return logits

    return FashionMNISTVGG5BNTT()


def build_fedsnn_vgg5_bntt(
    timesteps: int = 12,
    classes: int = 10,
    surrogate_beta: float = 2.0,
    threshold: float = 1.0,
    membrane_decay: float = 0.95,
    bntt_eps: float = 1e-4,
    bntt_momentum: float = 0.1,
    track_runtime_activity: bool = True,
):
    """Build the FedSNN-style CIFAR-10 VGG-5 with BNTT.

    VGG-5 follows the upstream 64-pool-128-128-pool layout (three
    convolutions and two fully connected layers). Each hidden layer owns an
    independent BatchNorm module for every simulation timestep. Membranes are
    local to one forward call, preventing state leakage across mini-batches.
    """
    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for VGG-5+BNTT") from exc
    if timesteps <= 0 or surrogate_beta <= 0 or threshold <= 0:
        raise ValueError("timesteps, surrogate_beta, and threshold must be positive")
    if not 0.0 <= membrane_decay <= 1.0:
        raise ValueError("membrane_decay must be in [0, 1]")

    class ATanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return (inputs >= 0).to(inputs.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (inputs,) = ctx.saved_tensors
            denominator = 1.0 + (math.pi * surrogate_beta * inputs / 2.0).square()
            return grad_output * (surrogate_beta / 2.0) / denominator

    class FedSNNVGG5BNTT(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = timesteps
            self.threshold = threshold
            self.membrane_decay = membrane_decay
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
            self.conv3 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
            self.fc1 = nn.Linear(128 * 8 * 8, 1024, bias=False)
            self.fc2 = nn.Linear(1024, classes, bias=False)
            self.bntt1 = nn.ModuleList([
                nn.BatchNorm2d(64, eps=bntt_eps, momentum=bntt_momentum)
                for _ in range(timesteps)
            ])
            self.bntt2 = nn.ModuleList([
                nn.BatchNorm2d(128, eps=bntt_eps, momentum=bntt_momentum)
                for _ in range(timesteps)
            ])
            self.bntt3 = nn.ModuleList([
                nn.BatchNorm2d(128, eps=bntt_eps, momentum=bntt_momentum)
                for _ in range(timesteps)
            ])
            self.bntt_fc1 = nn.ModuleList([
                nn.BatchNorm1d(1024, eps=bntt_eps, momentum=bntt_momentum)
                for _ in range(timesteps)
            ])
            self.spike_layer_sizes = (
                64 * 32 * 32,
                128 * 16 * 16,
                128 * 16 * 16,
                1024,
            )
            self.last_firing_rate = 0.0
            self.last_sample_firing_rate = None

        @staticmethod
        def _apply_bntt(module, inputs):
            # BatchNorm1d cannot estimate variance from a singleton local batch.
            # In that edge case, use the accumulated running statistics.
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

        def _lif(self, current, membrane):
            charged = self.membrane_decay * membrane + current
            spikes = ATanSpike.apply(charged - self.threshold)
            return spikes, charged - spikes.detach() * self.threshold

        def forward(
            self,
            inputs,
            return_activity: bool = False,
            return_layer_activity: bool = False,
        ):
            if inputs.ndim != 4 or inputs.shape[1:] != (3, 32, 32):
                raise ValueError("VGG-5+BNTT expects NCHW inputs of shape (3, 32, 32)")
            if track_runtime_activity and (
                inputs.min().detach().item() < 0
                or inputs.max().detach().item() > 1
            ):
                raise ValueError("VGG-5+BNTT Poisson encoding expects pixels in [0, 1]")
            batch = inputs.shape[0]
            membranes = [
                inputs.new_zeros((batch, 64, 32, 32)),
                inputs.new_zeros((batch, 128, 16, 16)),
                inputs.new_zeros((batch, 128, 16, 16)),
                inputs.new_zeros((batch, 1024)),
                inputs.new_zeros((batch, classes)),
            ]
            logits = inputs.new_zeros((batch, classes))
            layer_activity = [inputs.new_zeros(batch) for _ in self.spike_layer_sizes]

            for timestep in range(self.timesteps):
                output = (torch.rand_like(inputs) <= inputs).to(inputs.dtype)
                current = self._apply_bntt(self.bntt1[timestep], self.conv1(output))
                output, membranes[0] = self._lif(current, membranes[0])
                layer_activity[0] = layer_activity[0] + output.flatten(1).mean(1)
                output = nn.functional.avg_pool2d(output, 2)
                current = self._apply_bntt(self.bntt2[timestep], self.conv2(output))
                output, membranes[1] = self._lif(current, membranes[1])
                layer_activity[1] = layer_activity[1] + output.flatten(1).mean(1)
                current = self._apply_bntt(self.bntt3[timestep], self.conv3(output))
                output, membranes[2] = self._lif(current, membranes[2])
                layer_activity[2] = layer_activity[2] + output.flatten(1).mean(1)
                output = nn.functional.avg_pool2d(output, 2).flatten(1)
                current = self._apply_bntt(self.bntt_fc1[timestep], self.fc1(output))
                output, membranes[3] = self._lif(current, membranes[3])
                layer_activity[3] = layer_activity[3] + output.mean(1)
                membranes[4] = membranes[4] + self.fc2(output)
                logits = logits + membranes[4]

            logits = logits / self.timesteps
            layer_rates = torch.stack(layer_activity, dim=1) / self.timesteps
            sample_rates = layer_rates.mean(dim=1)
            if track_runtime_activity:
                self.last_sample_firing_rate = sample_rates.detach()
                self.last_firing_rate = float(sample_rates.mean().detach().cpu())
            if return_layer_activity:
                return logits, layer_rates
            if return_activity:
                return logits, sample_rates
            return logits

    return FedSNNVGG5BNTT()


def build_fedsnn_vgg9_bntt(
    timesteps: int = 6,
    classes: int = 10,
    surrogate_beta: float = 2.0,
    threshold: float = 1.0,
    membrane_decay: float = 0.95,
    bntt_eps: float = 1e-4,
    bntt_momentum: float = 0.1,
    track_runtime_activity: bool = True,
):
    """Build a FedSNN-recipe CIFAR-10 VGG-9 with per-timestep BNTT.

    Same training recipe as ``build_fedsnn_vgg5_bntt`` (Poisson input encoding,
    ATan-surrogate LIF neurons with soft membrane decay, per-layer per-timestep
    BatchNorm, bias-free weights, average pooling, mean-voltage logits) but on
    the VGG-9 skeleton shared with ``build_fedlec_vgg9`` / ``build_sfedca_vgg9``
    (channels 64-64-128-128-256-256-256, pooling after convs 1/3/6, 1024-wide
    hidden fully connected layer). Membranes are local to one forward call.
    """
    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for VGG-9+BNTT") from exc
    if timesteps <= 0 or surrogate_beta <= 0 or threshold <= 0:
        raise ValueError("timesteps, surrogate_beta, and threshold must be positive")
    if not 0.0 <= membrane_decay <= 1.0:
        raise ValueError("membrane_decay must be in [0, 1]")

    class ATanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return (inputs >= 0).to(inputs.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (inputs,) = ctx.saved_tensors
            denominator = 1.0 + (math.pi * surrogate_beta * inputs / 2.0).square()
            return grad_output * (surrogate_beta / 2.0) / denominator

    class FedSNNVGG9BNTT(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = timesteps
            self.threshold = threshold
            self.membrane_decay = membrane_decay
            channels = (64, 64, 128, 128, 256, 256, 256)
            in_channels = (3, *channels[:-1])
            self.convs = nn.ModuleList(
                nn.Conv2d(source, target, 3, padding=1, bias=False)
                for source, target in zip(in_channels, channels)
            )
            self.bntt_convs = nn.ModuleList(
                nn.ModuleList(
                    nn.BatchNorm2d(channel, eps=bntt_eps, momentum=bntt_momentum)
                    for _ in range(timesteps)
                )
                for channel in channels
            )
            self.fc1 = nn.Linear(256 * 4 * 4, 1024, bias=False)
            self.fc2 = nn.Linear(1024, classes, bias=False)
            self.bntt_fc1 = nn.ModuleList(
                nn.BatchNorm1d(1024, eps=bntt_eps, momentum=bntt_momentum)
                for _ in range(timesteps)
            )
            self.pool_after = frozenset({1, 3, 6})
            self.spike_layer_sizes = (
                64 * 32 * 32,
                64 * 32 * 32,
                128 * 16 * 16,
                128 * 16 * 16,
                256 * 8 * 8,
                256 * 8 * 8,
                256 * 8 * 8,
                1024,
            )
            self.last_firing_rate = 0.0
            self.last_sample_firing_rate = None

        @staticmethod
        def _apply_bntt(module, inputs):
            # BatchNorm1d cannot estimate variance from a singleton local batch.
            # In that edge case, use the accumulated running statistics.
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

        def _lif(self, current, membrane):
            charged = self.membrane_decay * membrane + current
            spikes = ATanSpike.apply(charged - self.threshold)
            return spikes, charged - spikes.detach() * self.threshold

        def forward(
            self,
            inputs,
            return_activity: bool = False,
            return_layer_activity: bool = False,
        ):
            if inputs.ndim != 4 or inputs.shape[1:] != (3, 32, 32):
                raise ValueError("VGG-9+BNTT expects NCHW inputs of shape (3, 32, 32)")
            if track_runtime_activity and (
                inputs.min().detach().item() < 0
                or inputs.max().detach().item() > 1
            ):
                raise ValueError("VGG-9+BNTT Poisson encoding expects pixels in [0, 1]")
            batch = inputs.shape[0]
            conv_shapes = [
                (64, 32, 32),
                (64, 32, 32),
                (128, 16, 16),
                (128, 16, 16),
                (256, 8, 8),
                (256, 8, 8),
                (256, 8, 8),
            ]
            membranes = [inputs.new_zeros((batch, *s)) for s in conv_shapes]
            membrane_fc1 = inputs.new_zeros((batch, 1024))
            membrane_out = inputs.new_zeros((batch, classes))
            logits = inputs.new_zeros((batch, classes))
            layer_activity = [inputs.new_zeros(batch) for _ in self.spike_layer_sizes]

            for timestep in range(self.timesteps):
                output = (torch.rand_like(inputs) <= inputs).to(inputs.dtype)
                for index, convolution in enumerate(self.convs):
                    current = self._apply_bntt(
                        self.bntt_convs[index][timestep], convolution(output)
                    )
                    output, membranes[index] = self._lif(current, membranes[index])
                    layer_activity[index] = (
                        layer_activity[index] + output.flatten(1).mean(1)
                    )
                    if index in self.pool_after:
                        output = nn.functional.avg_pool2d(output, 2)
                output = output.flatten(1)
                current = self._apply_bntt(self.bntt_fc1[timestep], self.fc1(output))
                output, membrane_fc1 = self._lif(current, membrane_fc1)
                layer_activity[7] = layer_activity[7] + output.mean(1)
                membrane_out = membrane_out + self.fc2(output)
                logits = logits + membrane_out

            logits = logits / self.timesteps
            layer_rates = torch.stack(layer_activity, dim=1) / self.timesteps
            sample_rates = layer_rates.mean(dim=1)
            if track_runtime_activity:
                self.last_sample_firing_rate = sample_rates.detach()
                self.last_firing_rate = float(sample_rates.mean().detach().cpu())
            if return_layer_activity:
                return logits, layer_rates
            if return_activity:
                return logits, sample_rates
            return logits

    return FedSNNVGG9BNTT()


def build_fedsnn_alexnet_bntt(
    timesteps: int = 4,
    classes: int = 10,
    surrogate_beta: float = 2.0,
    threshold: float = 1.0,
    membrane_decay: float = 0.95,
    bntt_eps: float = 1e-4,
    bntt_momentum: float = 0.1,
    track_runtime_activity: bool = True,
    execution_backend: str = "legacy_stepwise",
    execution_backend_strict: bool = False,
):
    """Build a FedSNN-recipe CIFAR-10 AlexNet with per-timestep BNTT.

    Combines the AlexNet-style channel skeleton from ``build_snn_cifar10``
    (64-192-384-256-256, pooling after convs 0/1/4, 1024-wide hidden FC) with
    the BNTT training recipe from ``build_fedsnn_vgg9_bntt`` (Poisson encoding,
    ATan-surrogate LIF with soft membrane decay, per-layer per-timestep
    BatchNorm, bias-free weights, average pooling, mean-voltage logits).
    Average pooling is intentional recipe packaging even though historical
    AlexNet used max pooling; membranes are local to one forward call.
    """
    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for AlexNet+BNTT") from exc
    if timesteps <= 0 or surrogate_beta <= 0 or threshold <= 0:
        raise ValueError("timesteps, surrogate_beta, and threshold must be positive")
    if not 0.0 <= membrane_decay <= 1.0:
        raise ValueError("membrane_decay must be in [0, 1]")
    execution_backend = str(execution_backend).lower()
    if execution_backend not in {
        "legacy_stepwise",
        "packed_eager",
        "packed_aspy",
        "npugraph",
    }:
        raise ValueError(
            "execution_backend must be legacy_stepwise, packed_eager, "
            "packed_aspy, or npugraph"
        )

    class ATanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return (inputs >= 0).to(inputs.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (inputs,) = ctx.saved_tensors
            denominator = 1.0 + (math.pi * surrogate_beta * inputs / 2.0).square()
            return grad_output * (surrogate_beta / 2.0) / denominator

    class FedSNNAlexNetBNTT(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = timesteps
            self.threshold = threshold
            self.membrane_decay = membrane_decay
            channels = (64, 192, 384, 256, 256)
            in_channels = (3, *channels[:-1])
            self.convs = nn.ModuleList(
                nn.Conv2d(source, target, 3, padding=1, bias=False)
                for source, target in zip(in_channels, channels)
            )
            self.bntt_convs = nn.ModuleList(
                nn.ModuleList(
                    nn.BatchNorm2d(channel, eps=bntt_eps, momentum=bntt_momentum)
                    for _ in range(timesteps)
                )
                for channel in channels
            )
            self.fc1 = nn.Linear(256 * 4 * 4, 1024, bias=False)
            self.fc2 = nn.Linear(1024, classes, bias=False)
            self.bntt_fc1 = nn.ModuleList(
                nn.BatchNorm1d(1024, eps=bntt_eps, momentum=bntt_momentum)
                for _ in range(timesteps)
            )
            self.pool_after = frozenset({0, 1, 4})
            self.spike_layer_sizes = (
                64 * 32 * 32,
                192 * 16 * 16,
                384 * 8 * 8,
                256 * 8 * 8,
                256 * 8 * 8,
                1024,
            )
            self.spike_channel_sizes = (64, 192, 384, 256, 256, 1024)
            # Explicit parameter-to-spike mapping for structured selection.  Do
            # not infer this from state-dict order: more complex backbones can
            # contain non-spiking projection weights between measured layers.
            self.structured_spike_parameter_map = {
                **{f"convs.{index}.weight": index for index in range(5)},
                "fc1.weight": 5,
            }
            self.last_firing_rate = 0.0
            self.last_sample_firing_rate = None
            self.execution_backend = execution_backend
            self.execution_backend_strict = bool(execution_backend_strict)
            self.last_lif_route = None
            self.last_lif_routes = ()
            self._current_lif_routes = None
            self._packed_decay_lif = None
            if execution_backend == "packed_aspy":
                try:
                    from spikingjelly_npu.activation_based import surrogate
                    from spikingjelly_npu.fedsnn import DecayLIF
                except ImportError as exc:
                    raise RuntimeError(
                        "packed_aspy AlexNet execution requires spikingjelly_npu"
                    ) from exc
                self._packed_decay_lif = DecayLIF(
                    membrane_decay=float(self.membrane_decay),
                    v_threshold=float(self.threshold),
                    surrogate_function=surrogate.ATan(alpha=float(surrogate_beta)),
                    backend="aspy",
                    backend_strict=self.execution_backend_strict,
                )
            # Every membrane is created and consumed within one forward.  This
            # declaration is used only by the opt-in spikingjelly_npu graph
            # runner; diagnostic forwards still bypass graph capture.
            self._spikingjelly_npu_graph_safe = True

        @staticmethod
        def _apply_bntt(module, inputs):
            # BatchNorm1d cannot estimate variance from a singleton local batch.
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

        def _lif(self, current, membrane):
            charged = self.membrane_decay * membrane + current
            spikes = ATanSpike.apply(charged - self.threshold)
            return spikes, charged - spikes.detach() * self.threshold

        def encode_poisson_sequence(self, inputs):
            """Generate the exact legacy Poisson stream as ``[T,N,C,H,W]``."""
            if inputs.ndim != 4 or inputs.shape[1:] != (3, 32, 32):
                raise ValueError(
                    "AlexNet+BNTT expects NCHW inputs of shape (3, 32, 32)"
                )
            return torch.stack(
                [
                    (torch.rand_like(inputs) <= inputs).to(inputs.dtype)
                    for _ in range(self.timesteps)
                ]
            )

        @staticmethod
        def _packed_ann(module, sequence):
            """Apply a stateless ANN module over the packed ``T*N`` axis."""
            try:
                from spikingjelly_npu.activation_based.functional import (
                    seq_to_ann_forward,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "packed AlexNet execution requires spikingjelly_npu"
                ) from exc
            return seq_to_ann_forward(sequence, module)

        def _bntt_sequence(self, modules, sequence):
            return torch.stack(
                [self._apply_bntt(modules[t], sequence[t]) for t in range(self.timesteps)]
            )

        def _lif_sequence(self, currents):
            if self._packed_decay_lif is not None:
                outputs = self._packed_decay_lif(currents.contiguous())
                self.last_lif_route = self._packed_decay_lif.last_backend_route
                if self._current_lif_routes is not None:
                    self._current_lif_routes.append(self.last_lif_route)
                return outputs
            membrane = torch.zeros_like(currents[0])
            outputs = []
            for timestep in range(self.timesteps):
                spikes, membrane = self._lif(currents[timestep], membrane)
                outputs.append(spikes)
            self.last_lif_route = "torch_stepwise"
            return torch.stack(outputs)

        def forward_encoded_packed(self, encoded, return_neuron_patterns: bool = False):
            """Run AlexNet+BNTT with stateless ANN operators packed over T*N."""
            if encoded.ndim != 5 or encoded.shape[0] != self.timesteps:
                raise ValueError(
                    "encoded AlexNet input must have shape [T,N,3,32,32]"
                )
            output = encoded
            patterns = [] if return_neuron_patterns else None
            self._current_lif_routes = [] if self._packed_decay_lif is not None else None
            for index, convolution in enumerate(self.convs):
                currents = self._packed_ann(convolution, output)
                currents = self._bntt_sequence(self.bntt_convs[index], currents)
                output = self._lif_sequence(currents)
                if patterns is not None:
                    patterns.append(output.detach().transpose(0, 1).to(torch.bool))
                if index in self.pool_after:
                    output = self._packed_ann(
                        lambda tensor: nn.functional.avg_pool2d(tensor, 2), output
                    )
            output = self._packed_ann(nn.Flatten(start_dim=1), output)
            currents = self._packed_ann(self.fc1, output)
            currents = self._bntt_sequence(self.bntt_fc1, currents)
            output = self._lif_sequence(currents)
            if patterns is not None:
                patterns.append(output.detach().transpose(0, 1).to(torch.bool))
            readout = self._packed_ann(self.fc2, output)
            # Preserve legacy cumulative readout: mean_t(sum_{j<=t} readout_j).
            logits = readout.cumsum(dim=0).sum(dim=0) / self.timesteps
            if self._current_lif_routes is not None:
                self.last_lif_routes = tuple(self._current_lif_routes)
                self._current_lif_routes = None
            if patterns is not None:
                return logits, tuple(patterns)
            return logits

        def forward_encoded_stepwise(self, encoded, return_neuron_patterns: bool = False):
            """Reference path for parity using a fixed encoded sequence."""
            if encoded.ndim != 5 or encoded.shape[0] != self.timesteps:
                raise ValueError(
                    "encoded AlexNet input must have shape [T,N,3,32,32]"
                )
            batch = encoded.shape[1]
            conv_shapes = [
                (64, 32, 32),
                (192, 16, 16),
                (384, 8, 8),
                (256, 8, 8),
                (256, 8, 8),
            ]
            membranes = [encoded.new_zeros((batch, *shape)) for shape in conv_shapes]
            membrane_fc1 = encoded.new_zeros((batch, 1024))
            membrane_out = encoded.new_zeros((batch, self.fc2.out_features))
            logits = encoded.new_zeros((batch, self.fc2.out_features))
            patterns = [[] for _ in range(6)] if return_neuron_patterns else None
            for timestep in range(self.timesteps):
                output = encoded[timestep]
                for index, convolution in enumerate(self.convs):
                    current = self._apply_bntt(
                        self.bntt_convs[index][timestep], convolution(output)
                    )
                    output, membranes[index] = self._lif(current, membranes[index])
                    if patterns is not None:
                        patterns[index].append(output.detach().to(torch.bool))
                    if index in self.pool_after:
                        output = nn.functional.avg_pool2d(output, 2)
                current = self._apply_bntt(
                    self.bntt_fc1[timestep], self.fc1(output.flatten(1))
                )
                output, membrane_fc1 = self._lif(current, membrane_fc1)
                if patterns is not None:
                    patterns[5].append(output.detach().to(torch.bool))
                membrane_out = membrane_out + self.fc2(output)
                logits = logits + membrane_out
            logits = logits / self.timesteps
            if patterns is not None:
                return logits, tuple(torch.stack(values, dim=1) for values in patterns)
            return logits

        def forward(
            self,
            inputs,
            return_activity: bool = False,
            return_layer_activity: bool = False,
            return_neuron_activity: bool = False,
            return_neuron_patterns: bool = False,
        ):
            if inputs.ndim != 4 or inputs.shape[1:] != (3, 32, 32):
                raise ValueError(
                    "AlexNet+BNTT expects NCHW inputs of shape (3, 32, 32)"
                )
            if track_runtime_activity and (
                inputs.min().detach().item() < 0
                or inputs.max().detach().item() > 1
            ):
                raise ValueError(
                    "AlexNet+BNTT Poisson encoding expects pixels in [0, 1]"
                )
            if return_neuron_activity and return_neuron_patterns:
                raise ValueError(
                    "return_neuron_activity and return_neuron_patterns are mutually exclusive"
                )
            diagnostic = any(
                (
                    return_activity,
                    return_layer_activity,
                    return_neuron_activity,
                    return_neuron_patterns,
                )
            )
            if self.execution_backend != "legacy_stepwise" and not diagnostic:
                encoded = self.encode_poisson_sequence(inputs)
                return self.forward_encoded_packed(encoded)
            batch = inputs.shape[0]
            conv_shapes = [
                (64, 32, 32),
                (192, 16, 16),
                (384, 8, 8),
                (256, 8, 8),
                (256, 8, 8),
            ]
            membranes = [inputs.new_zeros((batch, *s)) for s in conv_shapes]
            membrane_fc1 = inputs.new_zeros((batch, 1024))
            membrane_out = inputs.new_zeros((batch, classes))
            logits = inputs.new_zeros((batch, classes))
            layer_activity = [inputs.new_zeros(batch) for _ in self.spike_layer_sizes]
            neuron_activity = (
                [inputs.new_zeros((batch, shape[0])) for shape in conv_shapes]
                + [inputs.new_zeros((batch, 1024))]
                if return_neuron_activity
                else None
            )
            neuron_patterns = (
                [[] for _ in range(len(conv_shapes) + 1)]
                if return_neuron_patterns
                else None
            )

            for timestep in range(self.timesteps):
                output = (torch.rand_like(inputs) <= inputs).to(inputs.dtype)
                for index, convolution in enumerate(self.convs):
                    current = self._apply_bntt(
                        self.bntt_convs[index][timestep], convolution(output)
                    )
                    output, membranes[index] = self._lif(current, membranes[index])
                    layer_activity[index] = (
                        layer_activity[index] + output.flatten(1).mean(1)
                    )
                    if neuron_activity is not None:
                        neuron_activity[index] = (
                            neuron_activity[index] + output.flatten(2).mean(2)
                        )
                    if neuron_patterns is not None:
                        neuron_patterns[index].append(output.detach().to(torch.bool))
                    if index in self.pool_after:
                        output = nn.functional.avg_pool2d(output, 2)
                output = output.flatten(1)
                current = self._apply_bntt(self.bntt_fc1[timestep], self.fc1(output))
                output, membrane_fc1 = self._lif(current, membrane_fc1)
                layer_activity[5] = layer_activity[5] + output.mean(1)
                if neuron_activity is not None:
                    neuron_activity[5] = neuron_activity[5] + output
                if neuron_patterns is not None:
                    neuron_patterns[5].append(output.detach().to(torch.bool))
                membrane_out = membrane_out + self.fc2(output)
                logits = logits + membrane_out

            logits = logits / self.timesteps
            layer_rates = torch.stack(layer_activity, dim=1) / self.timesteps
            sample_rates = layer_rates.mean(dim=1)
            if track_runtime_activity:
                self.last_sample_firing_rate = sample_rates.detach()
                self.last_firing_rate = float(sample_rates.mean().detach().cpu())
            if return_neuron_activity:
                return logits, tuple(rate / self.timesteps for rate in neuron_activity)
            if return_neuron_patterns:
                return logits, tuple(
                    torch.stack(patterns, dim=1) for patterns in neuron_patterns
                )
            if return_layer_activity:
                return logits, layer_rates
            if return_activity:
                return logits, sample_rates
            return logits

    return FedSNNAlexNetBNTT()


def build_fedsnn_resnet18_bntt(
    timesteps: int = 4,
    classes: int = 100,
    surrogate_beta: float = 2.0,
    threshold: float = 1.0,
    membrane_decay: float = 0.95,
    bntt_eps: float = 1e-4,
    bntt_momentum: float = 0.1,
    track_runtime_activity: bool = True,
):
    """ResNet-18 + BNTT for CIFAR-100.

    SEW-style residual (skip adds to current before LIF) with per-timestep BNTT.
    Architecture: conv1(64) -> [64,64]x2 -> [128,128]x2 -> [256,256]x2 ->
    [512,512]x2 -> avgpool -> fc(classes).
    """
    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for ResNet-18+BNTT") from exc
    if timesteps <= 0 or surrogate_beta <= 0 or threshold <= 0:
        raise ValueError("timesteps, surrogate_beta, and threshold must be positive")
    if not 0.0 <= membrane_decay <= 1.0:
        raise ValueError("membrane_decay must be in [0, 1]")

    class ATanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return (inputs >= 0).to(inputs.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (inputs,) = ctx.saved_tensors
            denominator = 1.0 + (math.pi * surrogate_beta * inputs / 2.0).square()
            return grad_output * (surrogate_beta / 2.0) / denominator

    class BasicBlock(nn.Module):
        """ResNet BasicBlock with per-timestep BNTT on both convs."""

        def __init__(self, in_ch, out_ch, stride):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bntt1 = nn.ModuleList(
                nn.BatchNorm2d(out_ch, eps=bntt_eps, momentum=bntt_momentum)
                for _ in range(timesteps)
            )
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
            self.bntt2 = nn.ModuleList(
                nn.BatchNorm2d(out_ch, eps=bntt_eps, momentum=bntt_momentum)
                for _ in range(timesteps)
            )
            # Skip: 1x1 conv if shape changes, else identity
            self.skip = (
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
                if (in_ch != out_ch or stride != 1)
                else None
            )

    class FedSNNResNet18BNTT(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = timesteps
            self.threshold = threshold
            self.membrane_decay = membrane_decay

            # conv1
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.bntt_conv1 = nn.ModuleList(
                nn.BatchNorm2d(64, eps=bntt_eps, momentum=bntt_momentum)
                for _ in range(timesteps)
            )

            # Residual layers (2 blocks each)
            self.layer1 = nn.Sequential(BasicBlock(64, 64, 1), BasicBlock(64, 64, 1))
            self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128, 1))
            self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256, 1))
            self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512, 1))

            self.fc = nn.Linear(512, classes, bias=False)

            self.spike_channel_sizes = (
                64,                            # conv1
                64, 64, 64, 64,               # layer1: 2 blocks x 2 convs
                128, 128, 128, 128,           # layer2
                256, 256, 256, 256,           # layer3
                512, 512, 512, 512,           # layer4
            )
            self.spike_layer_sizes = (
                64 * 32 * 32,                  # conv1
                64 * 32 * 32, 64 * 32 * 32, 64 * 32 * 32, 64 * 32 * 32,  # layer1
                128 * 16 * 16, 128 * 16 * 16, 128 * 16 * 16, 128 * 16 * 16,  # layer2
                256 * 8 * 8, 256 * 8 * 8, 256 * 8 * 8, 256 * 8 * 8,  # layer3
                512 * 4 * 4, 512 * 4 * 4, 512 * 4 * 4, 512 * 4 * 4,  # layer4
            )
            spike_map = {"conv1.weight": 0}
            spike_index = 1
            for layer_index in range(1, 5):
                for block_index in range(2):
                    prefix = f"layer{layer_index}.{block_index}"
                    spike_map[f"{prefix}.conv1.weight"] = spike_index
                    spike_index += 1
                    spike_map[f"{prefix}.conv2.weight"] = spike_index
                    spike_index += 1
                    block = getattr(self, f"layer{layer_index}")[block_index]
                    if block.skip is not None:
                        # A projection contributes to the block's second LIF
                        # output, so its channels use that output's drift.
                        spike_map[f"{prefix}.skip.weight"] = spike_index - 1
            self.structured_spike_parameter_map = spike_map
            self.last_firing_rate = 0.0
            self.last_sample_firing_rate = None

        @staticmethod
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

        def _lif(self, current, membrane):
            charged = self.membrane_decay * membrane + current
            spikes = ATanSpike.apply(charged - self.threshold)
            return spikes, charged - spikes.detach() * self.threshold

        def forward(
            self,
            inputs,
            return_activity: bool = False,
            return_layer_activity: bool = False,
            return_neuron_activity: bool = False,
            return_neuron_patterns: bool = False,
        ):
            if inputs.ndim != 4 or inputs.shape[1:] != (3, 32, 32):
                raise ValueError("ResNet-18+BNTT expects NCHW (3, 32, 32)")
            if track_runtime_activity and (
                inputs.min().detach().item() < 0 or inputs.max().detach().item() > 1
            ):
                raise ValueError("Poisson encoding expects pixels in [0, 1]")
            if return_neuron_activity and return_neuron_patterns:
                raise ValueError(
                    "return_neuron_activity and return_neuron_patterns are mutually exclusive"
                )
            batch = inputs.shape[0]
            logits = inputs.new_zeros((batch, classes))
            layer_activity = [inputs.new_zeros(batch) for _ in self.spike_layer_sizes]
            neuron_activity = (
                [inputs.new_zeros((batch, ch)) for ch in self.spike_channel_sizes]
                if return_neuron_activity
                else None
            )
            neuron_patterns = (
                [[] for _ in self.spike_channel_sizes]
                if return_neuron_patterns
                else None
            )
            # Membranes for each spike-producing layer
            mem = [inputs.new_zeros((batch, ch, s, s)) for ch, s in [
                (64, 32), (64, 32), (64, 32), (64, 32), (64, 32),
                (128, 16), (128, 16), (128, 16), (128, 16),
                (256, 8), (256, 8), (256, 8), (256, 8),
                (512, 4), (512, 4), (512, 4), (512, 4),
            ]]

            for timestep in range(self.timesteps):
                spike = (torch.rand_like(inputs) <= inputs).to(inputs.dtype)
                idx = 0
                # conv1
                current = self._apply_bntt(self.bntt_conv1[timestep], self.conv1(spike))
                spike, mem[idx] = self._lif(current, mem[idx])
                layer_activity[idx] += spike.flatten(1).mean(1)
                if neuron_activity is not None:
                    neuron_activity[idx] += spike.flatten(2).mean(2)
                if neuron_patterns is not None:
                    neuron_patterns[idx].append(spike.detach().to(torch.bool))
                idx += 1
                # Residual layers (each block has 2 spike-producing convs)
                for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                    for block in layer:
                        block_input = spike  # store for skip connection
                        # conv1 -> LIF
                        current = self._apply_bntt(block.bntt1[timestep], block.conv1(spike))
                        spike, mem[idx] = self._lif(current, mem[idx])
                        layer_activity[idx] += spike.flatten(1).mean(1)
                        if neuron_activity is not None:
                            neuron_activity[idx] += spike.flatten(2).mean(2)
                        if neuron_patterns is not None:
                            neuron_patterns[idx].append(spike.detach().to(torch.bool))
                        idx += 1
                        # conv2 -> SEW ADD residual -> LIF
                        current = self._apply_bntt(block.bntt2[timestep], block.conv2(spike))
                        skip = block.skip(block_input) if block.skip is not None else block_input
                        current = current + skip
                        spike, mem[idx] = self._lif(current, mem[idx])
                        layer_activity[idx] += spike.flatten(1).mean(1)
                        if neuron_activity is not None:
                            neuron_activity[idx] += spike.flatten(2).mean(2)
                        if neuron_patterns is not None:
                            neuron_patterns[idx].append(spike.detach().to(torch.bool))
                        idx += 1
                # avgpool + fc (readout, no spike)
                output = nn.functional.adaptive_avg_pool2d(spike, (1, 1)).flatten(1)
                logits = logits + self.fc(output)

            logits = logits / self.timesteps
            layer_rates = torch.stack(layer_activity, dim=1) / self.timesteps
            sample_rates = layer_rates.mean(dim=1)
            if track_runtime_activity:
                self.last_sample_firing_rate = sample_rates.detach()
                self.last_firing_rate = float(sample_rates.mean().detach().cpu())
            if return_neuron_activity:
                return logits, tuple(r / self.timesteps for r in neuron_activity)
            if return_neuron_patterns:
                return logits, tuple(
                    torch.stack(patterns, dim=1) for patterns in neuron_patterns
                )
            if return_layer_activity:
                return logits, layer_rates
            if return_activity:
                return logits, sample_rates
            return logits

    return FedSNNResNet18BNTT()


def build_afedsnn_cifar10net(timesteps: int = 10, tau: float = 2.0, classes: int = 10):
    """Reconstruct AFedSNN's six-convolution CIFAR10Net.

    The paper only says CIFAR10Net extends its tabulated MNISTNet to six
    convolutions. We preserve the 2048 -> 100 -> voting head and use
    SpikingJelly-equivalent LIF charging with hard reset to zero.
    """
    try:
        import math
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for AFedSNN models") from exc
    if timesteps <= 0 or tau <= 1:
        raise ValueError("timesteps must be positive and tau must exceed one")

    class ATanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return (inputs >= 0).to(inputs.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (inputs,) = ctx.saved_tensors
            alpha = 2.0
            return grad_output * (alpha / 2.0) / (1.0 + (math.pi * alpha * inputs / 2.0).square())

    class AFedSNNCifar10Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = timesteps
            self.tau = tau
            channels = (32, 32, 64, 64, 128, 128)
            inputs = (3, *channels[:-1])
            self.convs = nn.ModuleList(
                [nn.Conv2d(cin, cout, 3, padding=1, bias=False) for cin, cout in zip(inputs, channels)]
            )
            self.norms = nn.ModuleList([nn.BatchNorm2d(channel) for channel in channels])
            self.fc1 = nn.Linear(128 * 4 * 4, 2048)
            self.fc2 = nn.Linear(2048, classes * 10)
            self.last_firing_rate = 0.0
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(module.weight)

        def _lif(self, current, membrane):
            charged = membrane + (current - membrane) / self.tau
            spikes = ATanSpike.apply(charged - 1.0)
            return spikes, charged * (1.0 - spikes.detach())

        def forward(self, images):
            if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
                raise ValueError("AFedSNN CIFAR10Net expects NCHW CIFAR images")
            if images.min().detach().item() < 0 or images.max().detach().item() > 1:
                raise ValueError("AFedSNN Poisson encoding expects pixels in [0, 1]")
            batch = images.shape[0]
            shapes = [(32, 32), (32, 32), (64, 16), (64, 16), (128, 8), (128, 8)]
            memories = [images.new_zeros((batch, c, size, size)) for c, size in shapes]
            memory_fc1 = images.new_zeros((batch, 2048))
            memory_fc2 = images.new_zeros((batch, classes * 10))
            votes = images.new_zeros((batch, classes * 10))
            spike_count = images.new_zeros(())
            neuron_steps = 0
            for _ in range(self.timesteps):
                output = (torch.rand_like(images) <= images).to(images.dtype)
                for index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                    output, memories[index] = self._lif(norm(conv(output)), memories[index])
                    spike_count = spike_count + output.detach().sum()
                    neuron_steps += output.numel()
                    if index in (1, 3, 5):
                        output = nn.functional.max_pool2d(output, 2)
                output = output.flatten(1)
                output, memory_fc1 = self._lif(self.fc1(output), memory_fc1)
                spike_count = spike_count + output.detach().sum()
                neuron_steps += output.numel()
                output, memory_fc2 = self._lif(self.fc2(output), memory_fc2)
                votes = votes + output
                spike_count = spike_count + output.detach().sum()
                neuron_steps += output.numel()
            self.last_firing_rate = float((spike_count / neuron_steps).detach().cpu())
            return votes.reshape(batch, classes, 10).mean(dim=2)

    return AFedSNNCifar10Net()
