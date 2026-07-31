from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .aggregation import average_torch_state_dicts, sfedca_select
from .config import load_config, result_dir
from .device import activate_device, resolve_device, seed_everything
from .protocol import (
    cpu_tensor_tree,
    load_protocol_dataset_and_model,
    partition_protocol_labels,
    reconcile_metrics_for_resume,
)
from .runtime import StaticBatchCudaGraph, cuda_graph_runtime_metadata
from .train import _append_jsonl, _atomic_torch_save, _code_commit, _evaluate


def _class_firing_rates(model, dataset, indices, device, batch_size: int, seed: int, max_batches=None):
    import torch
    from torch.utils.data import DataLoader, Subset

    seed_everything(seed)
    model.eval()
    # Keep the per-batch class statistics on the accelerator.  Ascend does not
    # provide native float64 kernels, so use the model's float32 precision there;
    # CPU and CUDA retain the previous float64 accumulation precision.
    device = torch.device(device)
    accumulator_dtype = torch.float32 if device.type == "npu" else torch.float64
    sums = torch.zeros(10, dtype=accumulator_dtype, device=device)
    counts = torch.zeros(10, dtype=accumulator_dtype, device=device)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            _, sample_rates = model(images, return_activity=True)
            sample_rates = sample_rates.detach().to(dtype=accumulator_dtype)
            sums.scatter_add_(0, labels, sample_rates)
            counts.scatter_add_(0, labels, torch.ones_like(sample_rates))
    rates = torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums))
    return rates.cpu().double()


def run_sfedca(config: dict, data_root: str, device_name: str, resume: bool = False, smoke: bool = False) -> Path:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset

    if config["paper"]["method"] != "sfedca":
        raise ValueError("SFedCA trainer requires an SFedCA config")
    info = resolve_device(device_name)
    device = activate_device(info)
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    rounds = 1 if smoke else int(config["training"]["rounds"])
    timesteps = 1 if smoke else int(config["model"]["timesteps"])
    local_epochs = 1 if smoke else int(config["training"]["local_epochs"])
    candidate_count = 2 if smoke else int(config["federation"]["candidate_clients"])
    selected_count = 1 if smoke else int(config["federation"]["selected_clients"])
    max_batches = 1 if smoke else None
    dataset_name, train_set, test_set, model_builder = load_protocol_dataset_and_model(
        config,
        data_root,
        timesteps=timesteps,
        track_runtime_activity=False,
    )
    clients = int(config["federation"]["clients"])
    batch_size = int(config["training"]["batch_size"])
    checkpoint_every = int(config["training"].get("checkpoint_every", 25))
    if checkpoint_every <= 0:
        raise ValueError("training.checkpoint_every must be positive")
    partition = partition_protocol_labels(config, train_set.targets, min_samples=1)
    partitions = partition.partitions

    model = model_builder().to(device)
    model_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    credit_bytes_per_client = 10 * 8
    run_signature = {
        "method": "sfedca",
        "dataset": copy.deepcopy(config["dataset"]),
        "model": copy.deepcopy(config["model"]),
        "seed": seed,
        "effective_timesteps": timesteps,
        "effective_local_epochs": local_epochs,
        "effective_candidate_clients": candidate_count,
        "effective_selected_clients": selected_count,
        "checkpoint_every": checkpoint_every,
        "batch_size": batch_size,
        "optimizer": str(config["training"].get("optimizer", "sgd")),
        "learning_rate": float(config["training"]["learning_rate"]),
        "momentum": float(config["training"].get("momentum", 0.0)),
        "smoke": smoke,
    }

    output = (
        Path(config["output"]["root"]) / "smoke" / "sfedca_reconstructed" / info.backend / f"seed={seed}"
        if smoke
        else result_dir(config)
    )
    output.mkdir(parents=True, exist_ok=True)
    resolved = copy.deepcopy(config)
    model_name = str(config["model"]["name"])
    if model_name == "fedlec_vgg9_snn":
        reconstruction_assumptions = {
            "architecture": model_name,
            "source": "FedLEC_official_S_VGG9",
            "input_encoding": "static_repeat",
            "neuron": "lif",
            "tau": float(config["model"].get("tau", 2.0)),
            "batch_norm": "shared_across_timesteps",
            "pooling": "max",
            "bias": False,
            "initialization": "pytorch_layer_default",
            "sgd_momentum": float(config["training"].get("momentum", 0.0)),
        }
        matched_poisson_stream = "not_applicable_static_repeat"
    else:
        reconstruction_assumptions = {
            "architecture": model_name,
            "normalization": "none",
            "pooling": "max",
            "bias": True,
            "initialization": "pytorch_layer_default",
            "sgd_momentum": float(config["training"].get("momentum", 0.0)),
        }
        matched_poisson_stream = True
    resolved["runtime"] = {
        "device": info.resolved,
        "backend": info.backend,
        "state_storage": "accelerator",
        "code_commit": _code_commit(),
        "implementation": f"reconstructed_{dataset_name}_{config['model']['name']}",
        "reconstruction_assumptions": reconstruction_assumptions,
        "matched_poisson_stream_for_credit": matched_poisson_stream,
        "smoke": smoke,
        "effective_rounds": rounds,
        "effective_timesteps": timesteps,
        "effective_local_epochs": local_epochs,
        "model_bytes": model_bytes,
        "credit_payload_assumption": "ten float64 class firing-rate differences per candidate",
        "partition_metadata": partition.metadata,
        "run_signature": run_signature,
    }
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    checkpoint_path = output / "latest.pt"
    metrics_path = output / "metrics.jsonl"
    start_round = 0
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("run_signature") != run_signature:
            raise ValueError("resume checkpoint was created by a different SFedCA run")
        model.load_state_dict(checkpoint["model"])
        checkpoint_round = int(checkpoint["round"])
        reconcile_metrics_for_resume(metrics_path, checkpoint_round)
        start_round = checkpoint_round + 1
    else:
        metrics_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
    if rounds < start_round:
        raise ValueError(
            f"configured rounds={rounds} precede checkpoint continuation round {start_round}"
        )

    criterion = nn.CrossEntropyLoss()
    local_model = model_builder().to(device)
    local_forward = StaticBatchCudaGraph(local_model, batch_size)
    resolved["runtime"].update(cuda_graph_runtime_metadata(local_forward))
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    for round_index in range(start_round, rounds):
        started = time.time()
        candidates = np.random.default_rng(seed + 3000 + round_index).choice(
            clients, size=candidate_count, replace=False
        ).tolist()
        scores = {}
        candidate_states = {}
        candidate_losses = {}
        for client_id in candidates:
            rate_seed = seed + 100_000 * round_index + client_id
            before = _class_firing_rates(
                model, train_set, partitions[client_id], device, batch_size, rate_seed, max_batches
            )
            local_model.load_state_dict(model.state_dict())
            local_model.train()
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=float(config["training"]["learning_rate"]),
                momentum=float(config["training"].get("momentum", 0.0)),
            )
            generator = torch.Generator().manual_seed(seed + round_index * clients + client_id)
            loader = DataLoader(
                Subset(train_set, partitions[client_id].tolist()),
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                generator=generator,
            )
            losses = []
            seed_everything(seed + 400_000 * round_index + client_id)
            for _ in range(local_epochs):
                for batch_index, (images, labels) in enumerate(loader):
                    if max_batches is not None and batch_index >= max_batches:
                        break
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(local_forward(images), labels)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.detach())
            after = _class_firing_rates(
                local_model, train_set, partitions[client_id], device, batch_size, rate_seed, max_batches
            )
            scores[client_id] = float((after - before).abs().sum())
            loss_values = torch.stack(losses).cpu().tolist()
            candidate_losses[client_id] = sum(loss_values) / len(loss_values)
            candidate_states[client_id] = {
                key: value.detach().clone()
                for key, value in local_model.state_dict().items()
            }
            del optimizer

        selected = sfedca_select(scores, selected_count)
        model.load_state_dict(average_torch_state_dicts([candidate_states[i] for i in selected], [1.0] * len(selected)))
        accuracy, test_loss = _evaluate(model, test_set, device, batch_size, 2 if smoke else None)
        record = {
            "round": round_index,
            "candidates": candidates,
            "selected_clients": selected,
            "credit_scores": scores,
            "train_loss": sum(candidate_losses[i] for i in selected) / len(selected),
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "completed_client_jobs": (round_index + 1) * candidate_count,
            "cumulative_upload_bytes": (round_index + 1)
            * (selected_count * model_bytes + candidate_count * credit_bytes_per_client),
            "cumulative_download_bytes": (round_index + 1) * candidate_count * model_bytes,
            "seconds": time.time() - started,
        }
        _append_jsonl(metrics_path, record)
        if (round_index + 1) % checkpoint_every == 0 or round_index + 1 == rounds:
            checkpoint = {
                "round": round_index,
                "model": cpu_tensor_tree(model.state_dict()),
                "run_signature": run_signature,
            }
            _atomic_torch_save(torch, checkpoint, checkpoint_path)
        print(json.dumps(record, sort_keys=True), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstructed SFedCA trainer")
    parser.add_argument("--config", default="configs/papers/sfedca_cifar10.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rounds", type=int, help="override training.rounds for diagnostics")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.rounds is not None:
        if args.rounds <= 0:
            parser.error("--rounds must be positive")
        config["training"]["rounds"] = args.rounds
    output = run_sfedca(config, args.data_root, args.device, args.resume, args.smoke)
    print(f"output={output}")


if __name__ == "__main__":
    main()
