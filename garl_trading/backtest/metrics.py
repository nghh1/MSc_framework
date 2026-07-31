from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def summarize(
    equity: pd.Series,
    returns: pd.Series,
    positions: pd.DataFrame,
    turnover: pd.Series,
) -> dict[str, float]:
    volatility = returns.std(ddof=1)
    downside = returns.clip(upper=0)
    downside_deviation = np.sqrt((downside**2).mean())
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    years = max(len(returns) / TRADING_DAYS, 1 / TRADING_DAYS)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    sharpe = 0.0 if not volatility else returns.mean() / volatility * np.sqrt(TRADING_DAYS)
    sortino = (
        0.0 if not downside_deviation else returns.mean() / downside_deviation * np.sqrt(TRADING_DAYS)
    )
    max_drawdown = float(drawdown.min())
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_volatility": float(volatility * np.sqrt(TRADING_DAYS)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(cagr / abs(max_drawdown)) if max_drawdown else 0.0,
        "turnover_daily": float(turnover.mean()),
        "gross_exposure": float(positions.abs().mean(axis=1).mean()),
        "net_exposure": float(positions.mean(axis=1).mean()),
    }

