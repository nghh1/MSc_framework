import torch
from garl_trading.garl.ddal import GradientPiece, weighted_average


def test_gradient_average_includes_declared_weights():
    pieces = [
        GradientPiece([torch.tensor([1.0])], "A", 0, 1.0),
        GradientPiece([torch.tensor([3.0])], "B", 0, 1.0)
    ]
    averaged = weighted_average(pieces)
    assert torch.allclose(averaged[0], torch.tensor([2.0]))

