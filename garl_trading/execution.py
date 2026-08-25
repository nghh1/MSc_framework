from __future__ import annotations

import numpy as np
import pandas as pd

MAX_ABS_EXPOSURE = 1.0


def thresholded_target(
    desired: float | pd.Series,
    current: float | pd.Series,
    threshold: float,
) -> float | pd.Series:
    """Keep the drifted position when a proposed rebalance is economically immaterial."""
    if threshold < 0:
        raise ValueError("Rebalance threshold cannot be negative.")
    difference = abs(desired - current)
    if isinstance(difference, pd.Series):
        return desired.where(difference > threshold, current)
    return current if difference <= threshold else desired


def bounded_exposure(value: float | pd.Series) -> float | pd.Series:
    """Enforce the common long/short exposure constraint."""
    if isinstance(value, pd.Series):
        return value.clip(-MAX_ABS_EXPOSURE, MAX_ABS_EXPOSURE)
    return float(np.clip(value, -MAX_ABS_EXPOSURE, MAX_ABS_EXPOSURE))


def limited_net_return(value: float | pd.Series) -> float | pd.Series:
    """Apply limited liability: a sleeve cannot lose more than its capital."""
    if isinstance(value, pd.Series):
        return value.clip(lower=-1.0)
    return max(float(value), -1.0)


def drifted_exposure(
    held: float | pd.Series,
    asset_return: float | pd.Series,
    gross_return: float | pd.Series,
) -> float | pd.Series:
    """Return price-driven pre-trade exposure before the next execution event."""
    wealth_factor = 1.0 + gross_return
    if isinstance(wealth_factor, pd.Series):
        alive = wealth_factor > 0
        result = held * (1.0 + asset_return) / wealth_factor.where(alive, 1.0)
        result = result.where(alive, 0.0)
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise FloatingPointError("Non-finite exposure after portfolio transition.")
        return result
    if wealth_factor <= 0:
        return 0.0
    result = float(held * (1.0 + asset_return) / wealth_factor)
    if not np.isfinite(result):
        raise FloatingPointError("Non-finite exposure after RL transition.")
    return result
