from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "ret_1", "ret_5", "ret_10",
    "sma_ratio_10", "sma_ratio_30", "sma_ratio_200",
    "ema_ratio_12", "ema_ratio_26",
    "rsi_14", "macd", "macd_signal",
    "bb_zscore", "atr_14_norm", "volatility_10", "volatility_30",
    "volume_z_20", "obv_slope_10",
)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def build_features(frame: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    close, high, low, volume = (
        frame["close"], frame["high"], frame["low"], frame["volume"]
    )
    out = pd.DataFrame(index=frame.index)
    for period in (1, 5, 10):
        out[f"ret_{period}"] = close.pct_change(period, fill_method=None)

    for period in (10, 30, 200):
        out[f"sma_ratio_{period}"] = close / close.rolling(period).mean() - 1
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["ema_ratio_12"] = close / ema12 - 1
    out["ema_ratio_26"] = close / ema26 - 1
    out["rsi_14"] = (_rsi(close) - 50) / 50
    out["macd"] = (ema12 - ema26) / close
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    middle = close.rolling(20).mean()
    width = close.rolling(20).std().replace(0, np.nan)
    out["bb_zscore"] = (close - middle) / (2 * width)
    previous = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - previous).abs(), (low - previous).abs()],
        axis=1,
    ).max(axis=1)
    out["atr_14_norm"] = true_range.rolling(14).mean() / close
    out["volatility_10"] = out["ret_1"].rolling(10).std()
    out["volatility_30"] = out["ret_1"].rolling(30).std()

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std().replace(0, np.nan)
    out["volume_z_20"] = ((volume - volume_mean) / volume_std).clip(-4, 4)
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    out["obv_slope_10"] = (
        obv.diff(10) / volume.rolling(10).mean().replace(0, np.nan) / 10
    )
    out["target_return"] = close.shift(-horizon) / close - 1
    return out.replace([np.inf, -np.inf], np.nan)

