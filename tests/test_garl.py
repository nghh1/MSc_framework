import pandas as pd
import torch

from garl_trading.garl.ddal import GradientPiece, return_relevance, weighted_average
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
