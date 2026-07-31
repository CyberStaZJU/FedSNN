from __future__ import annotations

from pathlib import Path


def load_cifar10(root: str | Path, train: bool, download: bool = False):
    """Load CIFAR-10 with the normalization used by official FedSNN code."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("torchvision is required for CIFAR-10") from exc
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    return datasets.CIFAR10(str(root), train=train, download=download, transform=transform)


def load_cifar10_unit_interval(root: str | Path, train: bool, download: bool = False):
    """Load [0, 1] pixels for the Poisson encoders used by SFedCA/AFedSNN."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("torchvision is required for CIFAR-10") from exc
    return datasets.CIFAR10(str(root), train=train, download=download, transform=transforms.ToTensor())


def load_cifar100_unit_interval(root: str | Path, train: bool, download: bool = False):
    """Load CIFAR-100 pixels in [0, 1] for Poisson-encoded SNNs."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("torchvision is required for CIFAR-100") from exc
    return datasets.CIFAR100(
        str(root), train=train, download=download, transform=transforms.ToTensor()
    )


def load_mnist_unit_interval(root: str | Path, train: bool, download: bool = False):
    """Load MNIST pixels in [0, 1] for SFedCA's Poisson encoder."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("torchvision is required for MNIST") from exc
    return datasets.MNIST(str(root), train=train, download=download, transform=transforms.ToTensor())


def load_fashion_mnist_unit_interval(root: str | Path, train: bool, download: bool = False):
    """Load Fashion-MNIST pixels in [0, 1] for SFedCA's Poisson encoder."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("torchvision is required for Fashion-MNIST") from exc
    return datasets.FashionMNIST(
        str(root), train=train, download=download, transform=transforms.ToTensor()
    )
