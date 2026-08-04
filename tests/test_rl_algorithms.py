import numpy as np
import torch
from conftest import market_fixture

from garl_trading.data import build_dataset
from garl_trading.data.features import FEATURE_COLUMNS
from garl_trading.rl import (
    train_independent_a2c,
    train_independent_dqn,
    train_independent_ppo,
    train_joint_a2c,
    train_joint_dqn,
    train_joint_ppo,
)
from garl_trading.rl.core import RewardEarlyStopper, TemporalFeatureExtractor


def test_rl_temporal_extractor_is_causal_and_preserves_position_state():
    torch.manual_seed(3)
    extractor = TemporalFeatureExtractor(
        observation_size=20 * 3 + 1,
        lookback=20,
        channels=8,
        dropout=0.0,
    ).eval()
    original = torch.randn(1, 20, 3)
    changed = original.clone()
    changed[:, 12:, :] += 100

    first = extractor.temporal_states(original)
    second = extractor.temporal_states(changed)
    assert torch.allclose(first[:, :, :12], second[:, :, :12])

    observation = torch.cat([original.reshape(1, -1), torch.tensor([[0.5]])], dim=1)
    encoded = extractor(observation)
    assert encoded.shape == (1, 9)
    assert np.isclose(float(encoded[0, -1].detach()), 0.5)


def test_joint_a2c_ppo_and_dqn_and_independent_ppo_dqn_emit_position_matrices():
    dataset = build_dataset(market_fixture(("AAA", "BBB"), periods=280))
    split = 50
    train_features = {
        ticker: dataset.features[ticker].iloc[:split].loc[:, FEATURE_COLUMNS]
        for ticker in dataset.tickers
    }
    train_closes = {
        ticker: dataset.prices[ticker]["close"].iloc[:split] for ticker in dataset.tickers
    }
    test_features = {
        ticker: dataset.features[ticker].iloc[split : split + 5].loc[:, FEATURE_COLUMNS]
        for ticker in dataset.tickers
    }
    context = {
        ticker: dataset.features[ticker].iloc[split - 4 : split].loc[:, FEATURE_COLUMNS]
        for ticker in dataset.tickers
    }
    common = {
        "levels": (-1.0, 0.0, 1.0),
        "lookback": 5,
        "epochs": 1,
        "rollout_length": 4,
        "learning_rate": 3e-4,
        "gamma": 0.95,
        "cost_rate": 0.0007,
        "seed": 4,
    }
    for trainer in (
        train_joint_a2c,
        train_joint_ppo,
        train_independent_ppo,
        train_joint_dqn,
        train_independent_dqn,
    ):
        policy = trainer(train_features, train_closes, **common)
        positions = policy.positions(test_features, context=context)
        assert positions.shape == (5, 2)
        assert positions.abs().max().max() <= 1
        models = policy.models.values() if isinstance(policy.models, dict) else (policy.models,)
        assert all(hasattr(model, "extractor") for model in models)


def test_early_stopping_checkpointing_starts_after_minimum_epochs():
    model = torch.nn.Linear(1, 1)
    stopper = RewardEarlyStopper(
        patience=2,
        min_delta=0.0,
        minimum_epochs=3,
    )

    assert not stopper.update(0, 1.0, model)
    assert not stopper.update(1, 0.0, model)
    assert not stopper.update(2, 0.0, model)
    assert stopper.best_state is None
    assert stopper.best_epoch is None

    assert not stopper.update(3, 1.0, model)
    assert not stopper.update(4, 0.0, model)
    assert stopper.update(5, 0.0, model)

    assert stopper.best_epoch == 3
    assert stopper.stop_epoch == 5


def test_early_stopping_respects_algorithm_checkpoint_gate():
    model = torch.nn.Linear(1, 1)
    stopper = RewardEarlyStopper(patience=2, min_delta=0.0, minimum_epochs=1)

    assert not stopper.update(0, 1.0, model)
    assert not stopper.update(1, 2.0, model, checkpoint_eligible=False)
    assert stopper.best_state is None

    assert not stopper.update(2, 1.5, model, checkpoint_eligible=True)
    assert stopper.best_epoch == 2


def test_independent_a2c_retains_per_agent_training_diagnostics():
    dataset = build_dataset(market_fixture(("AAA", "BBB"), periods=280))
    features = {
        ticker: dataset.features[ticker].iloc[:50].loc[:, FEATURE_COLUMNS]
        for ticker in dataset.tickers
    }
    closes = {
        ticker: dataset.prices[ticker]["close"].iloc[:50] for ticker in dataset.tickers
    }
    policy = train_independent_a2c(
        features,
        closes,
        levels=(-1.0, 0.0, 1.0),
        lookback=5,
        epochs=1,
        rollout_length=4,
        learning_rate=3e-4,
        gamma=0.95,
        cost_rate=0.0007,
        seed=4,
    )

    assert {row["agent"] for row in policy.diagnostics} == {"AAA", "BBB"}
