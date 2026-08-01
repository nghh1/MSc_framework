from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import summarise


@dataclass(frozen=True)
class PortfolioResult:
    equity: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    held_positions: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    metrics: dict[str, float]


def run_portfolio(
    closes: pd.DataFrame,
    target_positions: pd.DataFrame,
    initial_capital: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    short_borrow_bps_annual: float = 0.0,
) -> PortfolioResult:

    closes = closes.astype(float).sort_index()
    positions = target_positions.reindex(index=closes.index, columns=closes.columns).fillna(0.0)
    positions = positions.clip(-1.0, 1.0)
    held = positions.shift(1).fillna(0.0)
    weights = held / max(1, closes.shape[1])
    asset_returns = closes.pct_change(fill_method=None).fillna(0.0)
    gross = (weights * asset_returns).sum(axis=1)
    turnover_by_asset = held.diff().abs().fillna(held.abs())
    turnover = turnover_by_asset.sum(axis=1) / max(1, closes.shape[1])
    trade_cost = turnover * (transaction_cost_bps + slippage_bps) / 10_000
    short_exposure = held.clip(upper=0).abs().sum(axis=1) / max(1, closes.shape[1])
    borrow_cost = short_exposure * short_borrow_bps_annual / 10_000 / 252
    costs = trade_cost + borrow_cost
    net = gross - costs
    equity = initial_capital * (1 + net).cumprod()
    metrics = summarise(equity, net, held, turnover, gross_returns=gross, costs=costs)
    return PortfolioResult(equity, net, gross, held, turnover, costs, metrics)
