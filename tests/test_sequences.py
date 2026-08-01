import torch

from garl_trading.models.supervised.sequences import TCN_custom, TorchSequenceForecaster


def test_tcn_hidden_states_are_strictly_causal():
    torch.manual_seed(3)
    model = TCN_custom(n_features=3, hidden=8, dropout=0.0).eval()
    original = torch.randn(1, 20, 3)
    changed = original.clone()
    changed[:, 12:, :] += 100
    first = model.network(original.transpose(1, 2))
    second = model.network(changed.transpose(1, 2))
    assert torch.allclose(first[:, :, :12], second[:, :, :12])


def test_sequence_forecaster_resolves_auto_device():
    model = TorchSequenceForecaster(device="auto")
    assert model.device.type in {"cpu", "cuda", "mps"}
