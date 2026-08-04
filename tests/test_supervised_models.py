import numpy as np
from conftest import market_fixture

from garl_trading.data import build_dataset
from garl_trading.data.features import FEATURE_COLUMNS
from garl_trading.models import ModelContext, create_forecaster
from garl_trading.models.supervised.arimax import RollingARIMAX


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


def test_rolling_arimax_assimilates_observations_between_parameter_refits():
    dataset = build_dataset(market_fixture(("AAA",), periods=320, seed=9))
    features = dataset.features["AAA"].loc[:, FEATURE_COLUMNS]
    targets = dataset.features["AAA"]["target_return"]
    train = np.arange(0, 90)
    test = np.arange(90, 93)
    model = RollingARIMAX(p=1, d=0, q=0, trend="n", window=90, refit_every=10)
    model.fit(features.iloc[train], targets.iloc[train])

    low_first_return = targets.iloc[test].copy()
    high_first_return = targets.iloc[test].copy()
    low_first_return.iloc[0] = -0.25
    high_first_return.iloc[0] = 0.25
    low_forecast = model.predict_returns(
        features.iloc[test], realised_targets=low_first_return
    )
    high_forecast = model.predict_returns(
        features.iloc[test], realised_targets=high_first_return
    )

    assert np.isclose(low_forecast.iloc[0], high_forecast.iloc[0])
    assert not np.isclose(low_forecast.iloc[1], high_forecast.iloc[1])
