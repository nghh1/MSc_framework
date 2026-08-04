import pandas as pd
import torch
from conftest import market_fixture

from garl_trading.data import build_dataset
from garl_trading.data.features import FEATURE_COLUMNS
from garl_trading.garl.ddal import (
    GradientPiece,
    return_relevance,
    train_garl_ddal,
    weighted_average,
)
from garl_trading.rl.core import initialise_asset_actor_critics


def test_gradient_average_includes_declared_weights():
    pieces = [
        GradientPiece([torch.tensor([1.0])], "A", 0, 1.0),
        GradientPiece([torch.tensor([3.0])], "B", 0, 1.0)
    ]
    averaged = weighted_average(pieces)
    assert torch.allclose(averaged[0], torch.tensor([2.0]))


def test_ddal_relevance_uses_absolute_return_correlation():
    index = pd.bdate_range("2020-01-01", periods=5)
    closes = {
        "A": pd.Series([100, 101, 103, 102, 105], index=index),
        "B": pd.Series([50, 50.5, 51.5, 51, 52.5], index=index)
    }
    relevance = return_relevance(closes)
    assert relevance[("A", "A")] == 1.0
    assert relevance[("A", "B")] > 0.99


def test_garl_and_independent_ablation_share_reproducible_initialisation_contract():
    first = initialise_asset_actor_critics(("A", "B"), 3, 2, 42, torch.device("cpu"))
    second = initialise_asset_actor_critics(("A", "B"), 3, 2, 42, torch.device("cpu"))
    first_a = next(first["A"].parameters())
    first_b = next(first["B"].parameters())
    second_a = next(second["A"].parameters())
    assert not torch.allclose(first_a, first_b)
    assert torch.allclose(first_a, second_a)


def test_garl_checkpoints_become_eligible_only_after_shared_updates():
    dataset = build_dataset(market_fixture(("AAA", "BBB"), periods=280, seed=6))
    features = {
        ticker: dataset.features[ticker].iloc[:50].loc[:, FEATURE_COLUMNS]
        for ticker in dataset.tickers
    }
    closes = {
        ticker: dataset.prices[ticker]["close"].iloc[:50] for ticker in dataset.tickers
    }
    policy = train_garl_ddal(
        features,
        closes,
        levels=(-1.0, 0.0, 1.0),
        lookback=5,
        epochs=4,
        rollout_length=4,
        learning_rate=3e-4,
        gamma=0.95,
        cost_rate=0.0007,
        seed=4,
        share_after_fraction=0.25,
        share_every=2,
        minimum_train_epochs=1,
    )

    for agent in dataset.tickers:
        rows = [row for row in policy.diagnostics if row["agent"] == agent]
        assert any(row["checkpoint_eligible"] for row in rows)
        seen_shared_update = False
        for row in rows:
            seen_shared_update |= row["shared_update"]
            assert row["checkpoint_eligible"] == seen_shared_update
