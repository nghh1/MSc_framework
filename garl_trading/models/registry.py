from __future__ import annotations

from .base import ForecastModel
from .supervised import (
    LSTMForecaster,
    RandomForestForecaster,
    RollingARIMAX,
    StaticARIMAX,
    TCNForecaster,
    TFTForecaster,
)


def create_forecaster(name: str, *, seed: int = 42, **params) -> ForecastModel:
    constructors = {
        "arimax_static": StaticARIMAX,
        "arimax_rolling": RollingARIMAX,
        "random_forest": RandomForestForecaster,
        "lstm": LSTMForecaster,
        "tcn": TCNForecaster,
        "tft": TFTForecaster,
    }
    if name not in constructors:
        raise KeyError(f"Unknown forecaster: {name}")
    if name in {"random_forest", "lstm", "tcn", "tft"}:
        params.setdefault("seed", seed)
    return constructors[name](**params)

