import pandas as pd
import pytest
from conftest import market_fixture

from garl_trading.config import FrameworkConfig, ModelsConfig
from garl_trading.experiment.artifacts import ArtifactStore


def test_artifact_store_saves_config_and_market_snapshot(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[experiment]\nname = 'test'\n", encoding="utf-8")
    store = ArtifactStore(tmp_path / "artifacts", "test")
    store.initialise(FrameworkConfig(), config_path)
    store.save_market_data(market_fixture(("AAA",), periods=10))
    store.add_tuning_parameters(
        {
            "baseline": "garl_ddal",
            "fold": 0,
            "fold_kind": "walk_forward",
            "seed": 42,
        },
        "ALL",
        {"learning_rate": 0.0003, "rollout_length": 32},
    )
    store.flush()
    assert (store.path / "manifest.json").exists()
    assert (store.path / "config.toml").exists()
    assert (store.path / "data" / "prices.csv").exists()
    tuning = pd.read_csv(store.path / "tuning_parameters.csv")
    assert set(tuning["parameter"]) == {"learning_rate", "rollout_length"}
    assert set(tuning["ticker"]) == {"ALL"}


def test_cuda_index_is_a_valid_configured_device():
    config = FrameworkConfig(models=ModelsConfig(device="cuda:1"))
    config.validate()


def test_rl_tcn_receptive_field_must_cover_the_lookback():
    config = FrameworkConfig(
        models=ModelsConfig(
            lookback=20,
            rl_encoder_kernel_size=2,
            rl_encoder_dilations=(1, 2, 4),
        )
    )
    with pytest.raises(ValueError, match="receptive field"):
        config.validate()
