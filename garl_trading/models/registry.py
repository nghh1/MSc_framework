from __future__ import annotations

import numpy as np

from .base import ForecastModel
from .supervised import (
    LSTMForecaster,
    RandomForestForecaster,
    RollingARIMAX,
    StaticARIMAX,
    TCNForecaster,
    TransformerForecaster,
)


def create_forecaster(name: str, *, seed: int = 42, **params) -> ForecastModel:
    risk_aversion = float(params.pop("risk_aversion", 10.0))
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("risk_aversion must be finite and positive.")
    constructors = {
        "arimax_static": StaticARIMAX,
        "arimax_rolling": RollingARIMAX,
        "random_forest": RandomForestForecaster,
        "lstm": LSTMForecaster,
        "tcn": TCNForecaster,
        "transformer": TransformerForecaster,
    }
    if name not in constructors:
        raise KeyError(f"Unknown forecaster: {name}")
    if name in {"random_forest", "lstm", "tcn", "transformer"}:
        params.setdefault("seed", seed)
    model = constructors[name](**params)
    model.risk_aversion = risk_aversion
    return model
