import numpy as np
import pandas as pd
import torch

from garl_trading.models import ModelContext
from garl_trading.models.supervised.sequences import (LSTMForecaster, TCN_custom,
                                                      TorchSequenceForecaster, Transformer_custom)


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


def test_sequence_defaults_use_two_layers_and_fixed_dropout():
    model = LSTMForecaster(device="cpu")
    assert model.layers == 2
    assert model.dropout == 0.2


def test_transformer_is_causal_and_emits_one_value_per_sequence():
    torch.manual_seed(7)
    model = Transformer_custom(
        n_features=3,
        hidden=8,
        heads=2,
        layers=2,
        dropout=0.0,
        max_length=20).eval()
    original = torch.randn(1, 20, 3)
    changed = original.clone()
    changed[:, 12:, :] += 100
    projected = model.project(original) + model.position[:, :20]
    changed_projected = model.project(changed) + model.position[:, :20]
    mask = torch.triu(torch.ones(20, 20, dtype=torch.bool), diagonal=1)
    first = model.encoder(projected, mask=mask)
    second = model.encoder(changed_projected, mask=mask)
    assert torch.allclose(first[:, :12], second[:, :12], atol=1e-6)
    assert model(original).shape == (1,)


def test_sequence_targets_are_scaled_on_train_and_predictions_are_inverted():
    index = pd.bdate_range("2020-01-01", periods=40)
    features = pd.DataFrame(
        {"a": np.linspace(-1.0, 1.0, len(index)), "b": np.sin(np.arange(len(index)))},
        index=index)
    targets = pd.Series(np.linspace(0.01, 0.03, len(index)), index=index)
    model = TorchSequenceForecaster(
        lookback=5, hidden=8, dropout=0.0, epochs=1, seed=3, device="cpu").fit(features, targets)

    assert np.isclose(model.target_mean, targets.mean())
    assert np.isclose(model.target_std, targets.std(ddof=0))
    with torch.no_grad():
        for parameter in model.model.parameters():
            parameter.zero_()
    predicted = model.predict_returns(
        features.iloc[-5:],
        context=ModelContext(features.iloc[-9:-5]))
    assert np.allclose(predicted, model.target_mean)
