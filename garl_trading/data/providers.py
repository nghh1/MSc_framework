from __future__ import annotations

import time
from collections.abc import Iterable

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def normalise(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(col[0]).lower() for col in frame.columns]
    else:
        frame.columns = [str(col).lower() for col in frame.columns]
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{ticker}: missing columns {sorted(missing)}")
    frame = frame.loc[:, REQUIRED_COLUMNS].astype(float).sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    frame.index.name = "date"
    if frame.empty or (frame["close"] <= 0).any():
        raise ValueError(f"{ticker}: invalid or empty price history")
    return frame


def download_yahoo(
    tickers: Iterable[str], start: str, end: str, adjust_prices: bool = True, retries: int = 3
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        error: Exception | None = None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=adjust_prices,
                    progress=False,
                    threads=False,
                )
                result[ticker] = normalise(raw, ticker)
                break
            except Exception as exc:  # noqa: BLE001 - retry transient vendor/parser failures
                error = exc
                time.sleep(1.0)
        else:
            raise RuntimeError(f"Unable to download {ticker}: {error}")
    return result


def load_market_data(
    tickers: Iterable[str], start: str, end: str, adjust_prices: bool = True
) -> dict[str, pd.DataFrame]:
    return download_yahoo(tickers, start, end, adjust_prices=adjust_prices)
