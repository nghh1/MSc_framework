from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ModelContext:
    features: pd.DataFrame
    realised_targets: pd.Series | None = None


class ForecastModel(ABC):
    """Common contract for all supervised return forecasters."""

    def __init__(self) -> None:
        self.signal_scale = 0.01

    @abstractmethod
    def fit(self, features: pd.DataFrame, targets: pd.Series) -> ForecastModel:
        ...

    @abstractmethod
    def predict_returns(self, features: pd.DataFrame, context: ModelContext | None = None, realised_targets: pd.Series | None = None) -> pd.Series:
        ...

    def set_signal_scale(self, targets: pd.Series) -> None:
        scale = float(targets.dropna().std())
        self.signal_scale = scale if np.isfinite(scale) and scale > 1e-8 else 0.01

    def predict_positions(self, features: pd.DataFrame, context: ModelContext | None = None, realised_targets: pd.Series | None = None) -> pd.Series:
        prediction = self.predict_returns(features, context=context, realised_targets=realised_targets)
        position = np.tanh(prediction / (2.5 * self.signal_scale))
        return pd.Series(position, index=features.index).fillna(0.0).clip(-1, 1)

