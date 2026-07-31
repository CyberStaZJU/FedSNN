from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    resolved: str
    backend: str


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for device resolution") from exc
    return torch


def resolve_device(requested: str = "auto") -> DeviceInfo:
    """Resolve CPU, CUDA, or Ascend NPU without assuming a CUDA-only runtime."""
    requested = requested.lower()

    # Configuration validation and orchestration should remain usable on a
    # lightweight controller machine that does not have PyTorch installed.
    if requested == "cpu":
        return DeviceInfo(requested, "cpu", "cpu")

    torch = _load_torch()

    if requested == "auto":
        try:
            import torch_npu  # noqa: F401
        except (ImportError, OSError):
            pass
        if hasattr(torch, "npu") and torch.npu.is_available():
            index = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
            return DeviceInfo(requested, f"npu:{index}", "npu")
        if torch.cuda.is_available():
            return DeviceInfo(requested, "cuda:0", "cuda")
        return DeviceInfo(requested, "cpu", "cpu")

    backend = requested.split(":", 1)[0]
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return DeviceInfo(requested, requested if ":" in requested else "cuda:0", "cuda")
    if backend == "npu":
        try:
            import torch_npu  # noqa: F401
        except (ImportError, OSError) as exc:
            raise RuntimeError("NPU was requested but torch_npu could not be loaded") from exc
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("NPU was requested but is unavailable")
        return DeviceInfo(requested, requested if ":" in requested else "npu:0", "npu")
    raise ValueError(f"Unsupported device: {requested}")


def activate_device(info: DeviceInfo):
    """Set the process default accelerator before any global RNG initialization."""
    torch = _load_torch()
    device = torch.device(info.resolved)
    if info.backend == "cuda":
        torch.cuda.set_device(device)
    elif info.backend == "npu":
        # NPUGraph cannot capture legacy ACLop Conv2D kernels.  Disabling
        # internal formats before models and tensors are created selects the
        # capturable ACLNN path.  This is also slightly faster in eager mode
        # for the small, fixed-shape SNN convolutions used here.
        torch.npu.config.allow_internal_format = False
        torch.npu.set_device(device)
    return device


def seed_everything(seed: int, deterministic: bool = True) -> None:
    torch = _load_torch()
    if deterministic:
        # Required by deterministic CuBLAS kernels on CUDA >= 10.2. Set this
        # before the first matrix multiplication in every experiment process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
