import numpy as np
from conftest import market_fixture

from garl_trading.data import build_dataset
from garl_trading.data.features import FEATURE_COLUMNS
from garl_trading.models import ModelContext, create_forecaster


def test_all_supervised_models_fit_and_emit_finite_positions():
    dataset = build_dataset(market_fixture(("AAA",), periods=300))
    features = dataset.features["AAA"].loc[:, FEATURE_COLUMNS]
    targets = dataset.features["AAA"]["target_return"]
    train = np.arange(0, 70)
    context_positions = np.arange(51, 70)
    test = np.arange(70, 75)
    context = ModelContext(features.iloc[context_positions], targets.iloc[context_positions])
    parameters = {
        "arimax_static": {"p": 0, "q": 0},
        "arimax_rolling": {"p": 0, "q": 0, "window": 60, "refit_every": 2},
        "random_forest": {"n_estimators": 10, "min_samples_leaf": 2},
        "lstm": {"lookback": 5, "hidden": 16, "epochs": 1, "device": "cpu"},
        "tcn": {"lookback": 5, "hidden": 16, "epochs": 1, "device": "cpu"},
        "tft": {"lookback": 5, "hidden": 16, "epochs": 1, "device": "cpu"},
    }
    for name, params in parameters.items():
        model = create_forecaster(name, seed=4, **params)
        model.fit(features.iloc[train], targets.iloc[train])
        positions = model.predict_positions(
            features.iloc[test],
            context=context,
            realised_targets=targets.iloc[test],
        )
        assert len(positions) == len(test)
        assert np.isfinite(positions).all()
        assert positions.abs().max() <= 1
