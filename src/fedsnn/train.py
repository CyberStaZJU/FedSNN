from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _atomic_torch_save(torch, payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _evaluate(
    model,
    dataset,
    device,
    batch_size: int,
    max_batches: int | None = None,
    model_forward=None,
) -> tuple[float, float]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    model.eval()
    forward = model if model_forward is None else model_forward
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    total_loss = 0.0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            logits = forward(images)
            total_loss += float(criterion(logits, labels).cpu())
            correct += int((logits.argmax(dim=1) == labels).sum().cpu())
            total += labels.numel()
    return correct / total, total_loss / total

