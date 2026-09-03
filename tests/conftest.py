from __future__ import annotations

import numpy as np
import pandas as pd


def market_fixture(tickers: tuple[str, ...] = ("AAA",), periods: int = 760, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Deterministic in-memory OHLCV fixture; never exposed as a data provider."""
    dates = pd.bdate_range("2018-01-01", periods=periods)
    master = np.random.default_rng(seed)
    market = master.normal(0.0002, 0.008, periods)
    result = {}
    for ticker in tickers:
        rng = np.random.default_rng(master.integers(0, 2**32 - 1))
        returns = 0.7 * market + rng.normal(0, 0.008, periods)
        close = 100 * np.exp(np.cumsum(returns))
        open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.002, periods))
        spread = abs(rng.normal(0.005, 0.002, periods)) * close
        result[ticker] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) + spread,
                "low": np.maximum(0.01, np.minimum(open_, close) - spread),
                "close": close,
                "volume": rng.lognormal(np.log(5_000_000), 0.3, periods)
            },
            index=dates)
    return result
