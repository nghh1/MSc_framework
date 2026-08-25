from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from garl_trading.execution import (
    bounded_exposure,
    drifted_exposure,
    limited_net_return,
    thresholded_target,
)

from .metrics import summarise


@dataclass(frozen=True)
class PortfolioResult:
    equity: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    held_positions: pd.DataFrame
    held_weights: pd.DataFrame
    cash_exposure: pd.Series
    turnover: pd.Series
    costs: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]


TRADE_COLUMNS = [
    "date",
    "ticker",
    "pretrade_position",
    "target_position",
    "executed_change",
    "execution_price",
    "transaction_cost",
    "short_borrow_cost",
]


def trade_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=TRADE_COLUMNS)


def finish_result(
    initial_capital: float,
    net_returns: pd.Series,
    gross_returns: pd.Series,
    held_weights: pd.DataFrame,
    cash_exposure: pd.Series,
    turnover: pd.Series,
    costs: pd.Series,
    trades: pd.DataFrame,
) -> PortfolioResult:
    assets = max(1, held_weights.shape[1])
    held_positions = held_weights * assets
    equity = initial_capital * (1 + net_returns).cumprod()
    metrics = summarise(
        equity,
        net_returns,
        held_positions,
        turnover,
        gross_returns=gross_returns,
        costs=costs,
        cash_exposure=cash_exposure
    )
    return PortfolioResult(
        equity,
        net_returns,
        gross_returns,
        held_positions,
        held_weights,
        cash_exposure,
        turnover,
        costs,
        trades,
        metrics
    )


def run_portfolio(
    closes: pd.DataFrame,
    target_positions: pd.DataFrame,
    initial_capital: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    short_borrow_bps_annual: float = 0.0,
    rebalance_threshold: float = 0.0,
    decision_interval: int = 1,
) -> PortfolioResult:
    """Execute delayed targets only on scheduled decision dates."""
    closes = closes.astype(float).sort_index()
    if closes.empty or not np.isfinite(closes.to_numpy()).all() or (closes <= 0).any().any():
        raise ValueError("Active backtests require finite, strictly positive close prices.")
    positions = target_positions.reindex(index=closes.index, columns=closes.columns).fillna(0.0)
    positions = positions.clip(-1.0, 1.0)
    executed_positions = positions.shift(1).fillna(0.0)
    asset_returns = closes.pct_change(fill_method=None).fillna(0.0)
    cost_rate = (transaction_cost_bps + slippage_bps) / 10000
    borrow_rate = short_borrow_bps_annual / 10000 / 252
    if decision_interval < 1:
        raise ValueError("decision_interval must be positive.")

    # Each stock is a fixed equal-capital sleeve. This makes the execution transition exactly
    # reproducible inside each independent/GARL stock environment.
    pretrade_positions = pd.Series(0.0, index=closes.columns)
    weight_rows: list[pd.Series] = []
    gross_values: list[float] = []
    net_values: list[float] = []
    cash_values: list[float] = []
    turnover_values: list[float] = []
    cost_values: list[float] = []
    trade_records: list[dict] = []

    for step, date in enumerate(closes.index):
        decision_due = step > 0 and (step - 1) % decision_interval == 0
        proposed = (
            thresholded_target(
                executed_positions.loc[date], pretrade_positions, rebalance_threshold
            )
            if decision_due
            else pretrade_positions.copy()
        )
        # Price drift can push a short sleeve beyond its permitted leverage before the next
        # scheduled decision. Forced deleveraging is treated as a real, costed trade.
        desired = bounded_exposure(proposed)
        sleeve_turnover = (desired - pretrade_positions).abs()
        sleeve_gross = desired * asset_returns.loc[date]
        transaction_cost = sleeve_turnover * cost_rate
        borrow_cost = desired.clip(upper=0).abs() * borrow_rate
        sleeve_cost = transaction_cost + borrow_cost
        sleeve_net = limited_net_return(sleeve_gross - sleeve_cost)
        gross_return = float(sleeve_gross.mean())
        net_return = float(sleeve_net.mean())
        turnover = float(sleeve_turnover.mean())
        total_cost = float(sleeve_cost.mean())
        cash_weight = float((1.0 - desired).mean())

        for ticker in closes.columns[sleeve_turnover.to_numpy() > 1e-12]:
            trade_records.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "pretrade_position": float(pretrade_positions[ticker]),
                    "target_position": float(desired[ticker]),
                    "executed_change": float(desired[ticker] - pretrade_positions[ticker]),
                    "execution_price": float(closes.loc[date, ticker]),
                    "transaction_cost": float(transaction_cost[ticker] / closes.shape[1]),
                    "short_borrow_cost": float(borrow_cost[ticker] / closes.shape[1]),
                }
            )

        pretrade_positions = drifted_exposure(
            desired, asset_returns.loc[date], sleeve_gross
        )
        pretrade_positions = pretrade_positions.where(sleeve_net > -1.0, 0.0)
        if (
            not np.isfinite([gross_return, net_return, turnover, total_cost]).all()
            or net_return < -1.0
        ):
            raise FloatingPointError(f"Invalid active portfolio transition on {date}.")
        weight_rows.append(desired / max(1, closes.shape[1]))
        gross_values.append(gross_return)
        net_values.append(net_return)
        cash_values.append(cash_weight)
        turnover_values.append(turnover)
        cost_values.append(total_cost)

    held_weights = pd.DataFrame(weight_rows, index=closes.index, columns=closes.columns)
    gross = pd.Series(gross_values, index=closes.index, name="gross_return")
    net = pd.Series(net_values, index=closes.index, name="net_return")
    cash = pd.Series(cash_values, index=closes.index, name="cash_exposure")
    turnover = pd.Series(turnover_values, index=closes.index, name="turnover")
    costs = pd.Series(cost_values, index=closes.index, name="cost")
    return finish_result(
        initial_capital=initial_capital,
        net_returns=net,
        gross_returns=gross,
        held_weights=held_weights,
        cash_exposure=cash,
        turnover=turnover,
        costs=costs,
        trades=trade_frame(trade_records),
    )


def run_buy_and_hold(
    closes: pd.DataFrame,
    initial_capital: float,
    transaction_cost_bps: float,
    slippage_bps: float
) -> PortfolioResult:
    """Allocate equal capital once, retain fixed shares, and allow portfolio weights to drift."""
    closes = closes.astype(float).sort_index()
    asset_returns = closes.pct_change(fill_method=None).fillna(0.0)
    cost_rate = (transaction_cost_bps + slippage_bps) / 10000
    equal_weight = pd.Series(1.0 / max(1, closes.shape[1]), index=closes.columns)
    pretrade = pd.Series(0.0, index=closes.columns)

    weight_rows: list[pd.Series] = []
    gross_values: list[float] = []
    net_values: list[float] = []
    cash_values: list[float] = []
    turnover_values: list[float] = []
    cost_values: list[float] = []
    trade_records: list[dict] = []

    for step, date in enumerate(closes.index):
        if step == 0:
            desired = pretrade.copy()
        elif step == 1:
            desired = equal_weight.copy()
        else:
            desired = pretrade.copy()
        turnover = float((desired - pretrade).abs().sum())
        cash_weight = float(1.0 - desired.sum())
        gross_return = float((desired * asset_returns.loc[date]).sum())
        total_cost = turnover * cost_rate
        net_return = gross_return - total_cost

        changes = desired - pretrade
        for ticker in closes.columns[changes.abs().to_numpy() > 1e-12]:
            trade_records.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "pretrade_position": float(pretrade[ticker] * closes.shape[1]),
                    "target_position": float(desired[ticker] * closes.shape[1]),
                    "executed_change": float(changes[ticker] * closes.shape[1]),
                    "execution_price": float(closes.loc[date, ticker]),
                    "transaction_cost": float(abs(changes[ticker]) * cost_rate),
                    "short_borrow_cost": 0.0,
                }
            )

        denominator = max(1.0 + net_return, 1e-12)
        pretrade = desired * (1.0 + asset_returns.loc[date]) / denominator
        weight_rows.append(desired)
        gross_values.append(gross_return)
        net_values.append(net_return)
        cash_values.append(cash_weight)
        turnover_values.append(turnover)
        cost_values.append(total_cost)

    held_weights = pd.DataFrame(weight_rows, index=closes.index, columns=closes.columns)
    gross = pd.Series(gross_values, index=closes.index, name="gross_return")
    net = pd.Series(net_values, index=closes.index, name="net_return")
    cash = pd.Series(cash_values, index=closes.index, name="cash_exposure")
    turnover = pd.Series(turnover_values, index=closes.index, name="turnover")
    costs = pd.Series(cost_values, index=closes.index, name="cost")
    return finish_result(
        initial_capital=initial_capital,
        net_returns=net,
        gross_returns=gross,
        held_weights=held_weights,
        cash_exposure=cash,
        turnover=turnover,
        costs=costs,
        trades=trade_frame(trade_records),
    )


def run_equal_weight_rebalanced(
    closes: pd.DataFrame,
    initial_capital: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> PortfolioResult:
    """Rebalance to equal portfolio weights daily and charge drift-induced turnover."""
    closes = closes.astype(float).sort_index()
    asset_returns = closes.pct_change(fill_method=None).fillna(0.0)
    cost_rate = (transaction_cost_bps + slippage_bps) / 10000
    equal_weight = pd.Series(1.0 / max(1, closes.shape[1]), index=closes.columns)
    pretrade = pd.Series(0.0, index=closes.columns)

    weight_rows: list[pd.Series] = []
    gross_values: list[float] = []
    net_values: list[float] = []
    cash_values: list[float] = []
    turnover_values: list[float] = []
    cost_values: list[float] = []
    trade_records: list[dict] = []

    for step, date in enumerate(closes.index):
        desired = pretrade.copy() if step == 0 else equal_weight.copy()
        turnover = float((desired - pretrade).abs().sum())
        gross_return = float((desired * asset_returns.loc[date]).sum())
        total_cost = turnover * cost_rate
        net_return = gross_return - total_cost

        changes = desired - pretrade
        for ticker in closes.columns[changes.abs().to_numpy() > 1e-12]:
            trade_records.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "pretrade_position": float(pretrade[ticker] * closes.shape[1]),
                    "target_position": float(desired[ticker] * closes.shape[1]),
                    "executed_change": float(changes[ticker] * closes.shape[1]),
                    "execution_price": float(closes.loc[date, ticker]),
                    "transaction_cost": float(abs(changes[ticker]) * cost_rate),
                    "short_borrow_cost": 0.0,
                }
            )

        denominator = max(1.0 + net_return, 1e-12)
        pretrade = desired * (1.0 + asset_returns.loc[date]) / denominator
        weight_rows.append(desired)
        gross_values.append(gross_return)
        net_values.append(net_return)
        cash_values.append(float(1.0 - desired.sum()))
        turnover_values.append(turnover)
        cost_values.append(total_cost)

    held_weights = pd.DataFrame(weight_rows, index=closes.index, columns=closes.columns)
    return finish_result(
        initial_capital=initial_capital,
        net_returns=pd.Series(net_values, index=closes.index, name="net_return"),
        gross_returns=pd.Series(gross_values, index=closes.index, name="gross_return"),
        held_weights=held_weights,
        cash_exposure=pd.Series(cash_values, index=closes.index, name="cash_exposure"),
        turnover=pd.Series(turnover_values, index=closes.index, name="turnover"),
        costs=pd.Series(cost_values, index=closes.index, name="cost"),
        trades=trade_frame(trade_records),
    )
