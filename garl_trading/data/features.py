from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "ret_1",
    "ret_5",
    "ret_10",
    "roc_20",
    "sma_ratio_10",
    "sma_ratio_30",
    "sma_ratio_200",
    "ema_ratio_12",
    "ema_ratio_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_zscore",
    "atr_14_norm",
    "adx_14",
    "volatility_10",
    "volatility_30",
    "volume_z_20",
    "obv_slope_10",
)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
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


def adx(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Causal Wilder-style Average Directional Index scaled to [0, 1]."""
    high, low, close = frame["high"], frame["low"], frame["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denominator
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / 100


def build_features(frame: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    close, high, low, volume = frame["close"], frame["high"], frame["low"], frame["volume"]
    out = pd.DataFrame(index=frame.index)
    for period in (1, 5, 10):
        out[f"ret_{period}"] = close.pct_change(period, fill_method=None)
    # ret_10 is already the conventional 10-day ROC; use 20 days to add information.
    out["roc_20"] = close.pct_change(20, fill_method=None)

    for period in (10, 30, 200):
        out[f"sma_ratio_{period}"] = close / close.rolling(period).mean() - 1
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["ema_ratio_12"] = close / ema12 - 1
    out["ema_ratio_26"] = close / ema26 - 1
    out["rsi_14"] = (rsi(close) - 50) / 50
    out["macd"] = (ema12 - ema26) / close
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    middle = close.rolling(20).mean()
    width = close.rolling(20).std().replace(0, np.nan)
    out["bb_zscore"] = (close - middle) / (2 * width)
    previous = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    out["atr_14_norm"] = true_range.rolling(14).mean() / close
    out["adx_14"] = adx(frame)
    out["volatility_10"] = out["ret_1"].rolling(10).std()
    out["volatility_30"] = out["ret_1"].rolling(30).std()

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std().replace(0, np.nan)
    out["volume_z_20"] = ((volume - volume_mean) / volume_std).clip(-5, 5)
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    out["obv_slope_10"] = obv.diff(10) / volume.rolling(10).mean().replace(0, np.nan) / 10
    out["target_return"] = close.shift(-horizon) / close - 1
    return out.replace([np.inf, -np.inf], np.nan)
