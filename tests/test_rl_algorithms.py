from conftest import market_fixture

from garl_trading.data import build_dataset
from garl_trading.data.features import FEATURE_COLUMNS
from garl_trading.rl import (
    train_independent_dqn,
    train_independent_ppo,
    train_joint_dqn,
    train_joint_ppo,
)


def test_ppo_and_dqn_joint_and_independent_policies_emit_position_matrices():
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
    for trainer in (train_joint_ppo, train_independent_ppo, train_joint_dqn, train_independent_dqn):
        policy = trainer(train_features, train_closes, **common)
        positions = policy.positions(test_features, context=context)
        assert positions.shape == (5, 2)
        assert positions.abs().max().max() <= 1
