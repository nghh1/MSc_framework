import numpy as np
import pandas as pd
import torch

from garl_trading.models import ModelContext
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


def test_sequence_targets_are_scaled_on_train_and_predictions_are_inverted():
    index = pd.bdate_range("2020-01-01", periods=40)
    features = pd.DataFrame(
        {"a": np.linspace(-1.0, 1.0, len(index)), "b": np.sin(np.arange(len(index)))},
        index=index,
    )
    targets = pd.Series(np.linspace(0.01, 0.03, len(index)), index=index)
    model = TorchSequenceForecaster(
        lookback=5, hidden=8, dropout=0.0, epochs=1, seed=3, device="cpu"
    ).fit(features, targets)

    assert np.isclose(model.target_mean, targets.mean())
    assert np.isclose(model.target_std, targets.std(ddof=0))
    with torch.no_grad():
        for parameter in model.model.parameters():
            parameter.zero_()
    predicted = model.predict_returns(
        features.iloc[-5:],
        context=ModelContext(features.iloc[-9:-5]),
    )
    assert np.allclose(predicted, model.target_mean)
