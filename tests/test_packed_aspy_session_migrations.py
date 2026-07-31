"""Config gates for the Scheme A strict packed-AsPy migrations."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fedsnn.config import load_config
from fedsnn.protocol import load_protocol_dataset_and_model

ROOT = Path(__file__).resolve().parents[1]
SCHEME_CONFIGS = (
    ROOT / "configs/experiments/cifar10_t4_block_dual_ef_dir_v1/packed_aspy_qualification/block_2pct_seed2.yaml",
    ROOT / "configs/experiments/cifar10_t4_block_dual_ef_dir_v1/packed_aspy_qualification/coordinate_2pct_seed2.yaml",
)


@pytest.mark.parametrize("path", SCHEME_CONFIGS)
def test_scheme_a_migration_configs_drive_shipped_protocol_builder(path, monkeypatch):
    config = load_config(path)
    assert config["model"]["execution_backend"] == "packed_aspy"
    assert config["model"]["execution_backend_strict"] is True
    assert config["notes"]["packed_aspy_qualification_only"] is True
    assert config["notes"]["public_migration_id"].startswith("scheme_a_")

    sentinel = object()
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("fedsnn.protocol.build_fedsnn_alexnet_bntt", fake_build)

    class FakeDataset:
        def __init__(self):
            self.targets = [0]

    monkeypatch.setattr(
        "fedsnn.protocol.load_cifar10_unit_interval",
        lambda *args, **kwargs: FakeDataset(),
    )
    _, _, _, builder = load_protocol_dataset_and_model(
        config, ROOT / "unused-data", timesteps=4
    )
    assert builder() is sentinel
    assert captured["execution_backend"] == "packed_aspy"
    assert captured["execution_backend_strict"] is True


def test_historical_scheme_a_results_are_not_retargeted():
    historical = yaml.safe_load(
        (
            ROOT
            / "configs/experiments/cifar10_t4_block_dual_ef_dir_v1/tight_2pct_v1/formal/seed_2/symmetric_block_dual_topk_ef_2pct.yaml"
        ).read_text()
    )
    migrated = yaml.safe_load(SCHEME_CONFIGS[0].read_text())
    assert historical["model"]["execution_backend"] == "packed_eager"
    assert migrated["output"] != historical["output"]
    assert migrated["notes"]["execution_backend_policy"] == "strict_packed_aspy_qualification"
