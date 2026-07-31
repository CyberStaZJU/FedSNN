from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = {"paper", "dataset", "federation", "model", "training", "output"}

# Display / result-path canonical names (2026-07-28 method-set pivot).
# Pre-rename YAML / result trees may still use the legacy short name.
METHOD_NAME_ALIASES = {
    "global_topk_noef": "global_topk",
}


def canonical_method_name(method: str) -> str:
    """Map legacy method short names to the active result-directory names."""
    return METHOD_NAME_ALIASES.get(str(method), str(method))


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    missing = REQUIRED_SECTIONS - config.keys()
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    federation = config["federation"]
    total = int(federation["clients"])
    selected = int(federation["selected_clients"])
    if total <= 0 or not 0 < selected <= total:
        raise ValueError("selected_clients must be in [1, clients]")
    if config["dataset"]["partition"] == "dirichlet":
        alpha = float(config["dataset"]["alpha"])
        if alpha <= 0:
            raise ValueError("Dirichlet alpha must be positive")
    if int(config["model"]["timesteps"]) <= 0:
        raise ValueError("model.timesteps must be positive")
    if int(config["training"]["rounds"]) <= 0:
        raise ValueError("training.rounds must be positive")


def result_dir(config: dict[str, Any], root: str | Path = ".") -> Path:
    output = config["output"]
    method = canonical_method_name(output.get("method_name", config["paper"]["method"]))
    seed = int(config["training"]["seed"])
    base = Path(root) / output["root"] / output["track"]
    # Legacy configs include a separate partition/setting directory. The
    # dataset/group/method/seed=N layout introduced in 2026-07 omits it.
    if "setting" in output:
        base = base / output["setting"]
    return base / method / f"seed={seed}"
