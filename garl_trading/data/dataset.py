from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import FEATURE_COLUMNS, build_features


@dataclass(frozen=True)
class MarketDataset:
    prices: dict[str, pd.DataFrame]
    features: dict[str, pd.DataFrame]
    index: pd.DatetimeIndex

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(self.prices)

    def slice(self, positions) -> MarketDataset:
        index = self.index[positions]
        return MarketDataset(
            prices={t: f.loc[index] for t, f in self.prices.items()},
            features={t: f.loc[index] for t, f in self.features.items()},
            index=index
        )


def build_dataset(raw: dict[str, pd.DataFrame], horizon: int = 5) -> MarketDataset:
    features = {ticker: build_features(frame, horizon) for ticker, frame in raw.items()}
    common = None
    for ticker, frame in features.items():
        valid = frame.loc[:, FEATURE_COLUMNS].notna().all(axis=1)
        valid &= raw[ticker]["close"].reindex(frame.index).notna()
        ticker_index = frame.index[valid]
        common = ticker_index if common is None else common.intersection(ticker_index)
    if common is None or common.empty:
        raise ValueError("No common feature-valid dates across the universe.")
    common = pd.DatetimeIndex(common).sort_values()
    aligned_prices = {ticker: frame.loc[common] for ticker, frame in raw.items()}
    aligned_features = {ticker: frame.loc[common] for ticker, frame in features.items()}
    return MarketDataset(prices=aligned_prices, features=aligned_features, index=common)
