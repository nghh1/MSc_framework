from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelContext:
    features: pd.DataFrame
    realised_targets: pd.Series | None = None
    target_horizon: int = 1


class ForecastModel(ABC):
    """Common contract for all supervised return forecasters."""

    def __init__(self) -> None:
        self.return_variance = 0.01**2
        self.risk_aversion = 10.0

    @abstractmethod
    def fit(self, features: pd.DataFrame, targets: pd.Series) -> ForecastModel: ...

    @abstractmethod
    def predict_returns(self, features: pd.DataFrame, context: ModelContext | None = None,
                        realised_targets: pd.Series | None = None) -> pd.Series: ...

    def set_return_variance(self, targets: pd.Series) -> None:
        variance = float(targets.dropna().var(ddof=1))
        self.return_variance = (
            variance if np.isfinite(variance) and variance > 1e-10 else 0.01**2
        )

    def predict_positions(self, features: pd.DataFrame, context: ModelContext | None = None,
                          realised_targets: pd.Series | None = None) -> pd.Series:
        prediction = self.predict_returns(
            features, context=context, realised_targets=realised_targets)
        return self.positions_from_predictions(prediction)

    def positions_from_predictions(self, prediction: pd.Series) -> pd.Series:
        """Map forecasts to constrained mean-variance positions using training variance."""
        position = prediction / (self.risk_aversion * self.return_variance)
        return pd.Series(position, index=prediction.index).fillna(0.0).clip(-1, 1)
