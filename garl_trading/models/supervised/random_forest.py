from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..base import ForecastModel, ModelContext


class RandomForestForecaster(ForecastModel):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        max_features: str | float = "sqrt",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=seed,
            n_jobs=-1,
        )
        self.columns: list[str] = []

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> RandomForestForecaster:
        valid = features.notna().all(axis=1) & targets.notna()
        self.columns = list(features.columns)
        self.model.fit(features.loc[valid, self.columns], targets.loc[valid])
        self.set_return_variance(targets.loc[valid])
        return self

    def predict_returns(
        self,
        features: pd.DataFrame,
        context: ModelContext | None = None,
        realised_targets: pd.Series | None = None,
    ) -> pd.Series:
        values = self.model.predict(features.loc[:, self.columns].fillna(0.0))
        return pd.Series(values, index=features.index)
