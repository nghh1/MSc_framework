from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def summarise(
    equity: pd.Series,
    returns: pd.Series,
    positions: pd.DataFrame,
    turnover: pd.Series,
    gross_returns: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> dict[str, float]:
    volatility = returns.std(ddof=1)
    downside = returns.clip(upper=0)
    downside_deviation = np.sqrt((downside**2).mean())
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    years = max(len(returns) / TRADING_DAYS, 1 / TRADING_DAYS)
    total_return = float((1 + returns).prod() - 1)
    cagr = (1 + total_return) ** (1 / years) - 1
    sharpe = 0.0 if not volatility else returns.mean() / volatility * np.sqrt(TRADING_DAYS)
    sortino = (
        0.0
        if not downside_deviation
        else returns.mean() / downside_deviation * np.sqrt(TRADING_DAYS)
    )
    max_drawdown = float(drawdown.min())
    nonzero = returns[returns != 0]
    losses = abs(float(returns[returns < 0].sum()))
    value_at_risk = float(returns.quantile(0.05))
    tail = returns[returns <= value_at_risk]
    gross_total_return = (
        float((1 + gross_returns).prod() - 1) if gross_returns is not None else total_return
    )
    total_cost = float(costs.sum()) if costs is not None else 0.0
    return {
        "total_return": total_return,
        "annual_return": float(returns.mean() * TRADING_DAYS),
        "cagr": float(cagr),
        "annual_volatility": float(volatility * np.sqrt(TRADING_DAYS)),
        "annual_downside_deviation": float(downside_deviation * np.sqrt(TRADING_DAYS)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(cagr / abs(max_drawdown)) if max_drawdown else 0.0,
        "turnover_daily": float(turnover.mean()),
        "turnover_annual": float(turnover.mean() * TRADING_DAYS),
        "gross_exposure": float(positions.abs().mean(axis=1).mean()),
        "net_exposure": float(positions.mean(axis=1).mean()),
        "positive_day_rate": float((nonzero > 0).mean()) if len(nonzero) else 0.0,
        "profit_factor": float(returns[returns > 0].sum() / losses) if losses else 0.0,
        "value_at_risk_95": value_at_risk,
        "conditional_value_at_risk_95": float(tail.mean()) if len(tail) else value_at_risk,
        "skewness": float(returns.skew()) if len(returns) > 2 else 0.0,
        "excess_kurtosis": float(returns.kurt()) if len(returns) > 3 else 0.0,
        "ulcer_index": float(np.sqrt(np.mean(np.square(drawdown)))),
        "gross_total_return": gross_total_return,
        "cost_drag": float(gross_total_return - total_return),
        "total_cost": total_cost,
        "annual_cost": float(costs.mean() * TRADING_DAYS) if costs is not None else 0.0,
    }
