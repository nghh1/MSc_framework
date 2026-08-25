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
    dates = pd.bdate_range("2024-01-02", periods=2)
    store.add_predictions(
        {"baseline": "random_forest", "fold": 0, "fold_kind": "walk_forward"},
        "AAA",
        pd.Series([0.01, -0.02], index=dates),
        pd.Series([0.02, -0.01], index=dates),
    )
    store.add_result(
        {"baseline": "garl_ddal", "fold": 0, "fold_kind": "walk_forward"},
        {"sharpe": 0.5},
        pd.DataFrame({"AAA": [0.0, 1.0]}, index=dates),
        pd.Series([100_000.0, 101_000.0], index=dates),
        trades=pd.DataFrame(
            {
                "date": [dates[1]],
                "ticker": ["AAA"],
                "pretrade_position": [0.0],
                "target_position": [1.0],
                "executed_change": [1.0],
                "execution_price": [101.0],
                "transaction_cost": [0.0007],
                "short_borrow_cost": [0.0],
            }
        ),
    )
    store.flush()
    assert (store.path / "manifest.json").exists()
    assert (store.path / "config.toml").exists()
    assert (store.path / "data" / "prices.csv").exists()
    prices = pd.read_csv(store.path / "data" / "prices.csv")
    assert "date" in prices.columns
    tuning = pd.read_csv(store.path / "tuning_parameters.csv")
    assert set(tuning["parameter"]) == {"learning_rate", "rollout_length"}
    assert set(tuning["ticker"]) == {"ALL"}
    predictions = pd.read_csv(store.path / "predictions.csv")
    assert set(predictions.columns) >= {"prediction", "actual_return", "ticker"}
    assert len(predictions) == 2
    trades = pd.read_csv(store.path / "trades.csv")
    assert trades.loc[0, "executed_change"] == 1.0


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


def test_turnover_penalty_multiplier_cannot_understate_real_cost():
    config = FrameworkConfig(models=ModelsConfig(turnover_penalty_multiplier=0.5))
    with pytest.raises(ValueError, match="turnover_penalty_multiplier"):
        config.validate()


def test_supervised_risk_aversion_must_be_positive():
    config = FrameworkConfig(models=ModelsConfig(supervised_risk_aversion=0.0))
    with pytest.raises(ValueError, match="supervised_risk_aversion"):
        config.validate()
