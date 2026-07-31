from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .train_topk_saw import CURRENT_ROUND_TOPK_METHODS, _run_topk


def run_topk(
    config: dict,
    data_root: str,
    device_name: str,
    resume: bool = False,
    smoke: bool = False,
) -> Path:
    return _run_topk(
        config,
        data_root,
        device_name,
        resume,
        smoke,
        allowed_methods=CURRENT_ROUND_TOPK_METHODS,
        trainer_name="Top-k trainer",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Communication-matched Top-k trainer")
    parser.add_argument("--config", required=True)
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
    output = run_topk(config, args.data_root, args.device, args.resume, args.smoke)
    print(f"output={output}")


if __name__ == "__main__":
    main()
