from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from ..base import ForecastModel, ModelContext


def effective_trend(d: int, trend: str) -> str:
    return "n" if d > 0 and trend == "c" else trend


class StaticARIMAX(ForecastModel):
    def __init__(self, p: int = 1, d: int = 0, q: int = 1, trend: str = "c") -> None:
        super().__init__()
        self.order = (p, d, q)
        self.trend = effective_trend(d, trend)
        self.result = None
        self.columns: list[str] = []

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> StaticARIMAX:
        valid = features.notna().all(axis=1) & targets.notna()
        x, y = features.loc[valid], targets.loc[valid]
        self.columns = list(x.columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result = ARIMA(
                y.to_numpy(),
                exog=x.to_numpy(),
                order=self.order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
        self.set_return_variance(y)
        return self

    def predict_returns(
        self,
        features: pd.DataFrame,
        context: ModelContext | None = None,
        realised_targets: pd.Series | None = None,
    ) -> pd.Series:
        if self.result is None:
            raise RuntimeError("Model is not fitted.")
        x = features.loc[:, self.columns].fillna(0.0)
        prediction = self.result.get_forecast(steps=len(x), exog=x.to_numpy()).predicted_mean
        return pd.Series(np.asarray(prediction), index=x.index)


class RollingARIMAX(ForecastModel):
    """Causal rolling ARIMAX with identical update behavior in tuning and testing."""

    def __init__(
        self,
        p: int = 1,
        d: int = 0,
        q: int = 1,
        trend: str = "c",
        window: int = 252,
        refit_every: int = 10,
    ) -> None:
        super().__init__()
        self.order = (p, d, q)
        self.trend = effective_trend(d, trend)
        self.window = window
        self.refit_every = refit_every
        self.history_x = pd.DataFrame()
        self.history_y = pd.Series(dtype=float)
        self.columns: list[str] = []

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> RollingARIMAX:
        valid = features.notna().all(axis=1) & targets.notna()
        self.columns = list(features.columns)
        self.history_x = features.loc[valid, self.columns].tail(self.window).copy()
        self.history_y = targets.loc[valid].tail(self.window).copy()
        self.set_return_variance(self.history_y)
        return self

    def predict_returns(
        self,
        features: pd.DataFrame,
        context: ModelContext | None = None,
        realised_targets: pd.Series | None = None,
    ) -> pd.Series:
        x_history = self.history_x.copy()
        y_history = self.history_y.copy()
        if context is not None and context.realised_targets is not None:
            context_valid = context.features.notna().all(axis=1) & context.realised_targets.notna()
            x_history = pd.concat([x_history, context.features.loc[context_valid, self.columns]])
            y_history = pd.concat([y_history, context.realised_targets.loc[context_valid]])
            x_history = x_history.loc[~x_history.index.duplicated(keep="last")].sort_index()
            y_history = y_history.loc[~y_history.index.duplicated(keep="last")].sort_index()

        x_test = features.loc[:, self.columns].fillna(0.0)
        observed = realised_targets.reindex(x_test.index) if realised_targets is not None else None
        predictions: dict[pd.Timestamp, float] = {}
        fitted = None
        for i, date in enumerate(x_test.index):
            if i > 0:
                previous = x_test.index[i - 1]
                x_history = pd.concat([x_history, x_test.loc[[previous]]])
                value = observed.loc[previous] if observed is not None else predictions[previous]
                y_history = pd.concat([y_history, pd.Series([value], index=[previous])])
                if fitted is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fitted = fitted.append(
                            np.asarray([float(value)]),
                            exog=x_test.loc[[previous]].to_numpy(),
                            refit=False,
                        )
            if fitted is None or i % self.refit_every == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = ARIMA(
                        y_history.tail(self.window).fillna(0.0).to_numpy(),
                        exog=x_history.tail(self.window).fillna(0.0).to_numpy(),
                        order=self.order,
                        trend=self.trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()
            forecast = fitted.get_forecast(steps=1, exog=x_test.loc[[date]].to_numpy())
            predictions[date] = float(np.asarray(forecast.predicted_mean)[0])
        return pd.Series(predictions).reindex(x_test.index)
