import numpy as np
import pandas as pd

from garl_trading.backtest import run_buy_and_hold, run_portfolio
from garl_trading.rl.core import TradingState


def test_backtest_delays_positions_one_bar_and_charges_turnover():
    index = pd.bdate_range("2023-01-01", periods=4)
    closes = pd.DataFrame({"AAA": [100.0, 110.0, 110.0, 110.0]}, index=index)
    positions = pd.DataFrame({"AAA": [1.0, 1.0, 0.0, 0.0]}, index=index)
    result = run_portfolio(
        closes, positions, initial_capital=100.0, transaction_cost_bps=10.0, slippage_bps=0.0
    )
    assert result.held_positions.iloc[0, 0] == 0
    assert result.held_positions.iloc[1, 0] == 1
    assert np.isclose(result.returns.iloc[1], 0.10 - 0.001)
    assert result.costs.sum() > 0


def test_all_models_share_equal_weight_portfolio_contract():
    index = pd.bdate_range("2023-01-01", periods=3)
    closes = pd.DataFrame({"A": [100, 110, 110], "B": [100, 100, 90]}, index=index)
    positions = pd.DataFrame(1.0, index=index, columns=closes.columns)
    result = run_portfolio(
        closes, positions, initial_capital=100, transaction_cost_bps=0, slippage_bps=0
    )
    assert np.isclose(result.gross_returns.iloc[1], 0.05)


def test_reporting_metrics_include_cost_and_tail_risk_fields():
    index = pd.bdate_range("2023-01-01", periods=5)
    closes = pd.DataFrame({"AAA": [100, 103, 101, 104, 102]}, index=index)
    positions = pd.DataFrame(1.0, index=index, columns=closes.columns)
    result = run_portfolio(closes, positions, 100, transaction_cost_bps=10, slippage_bps=5)
    required = {
        "annual_return",
        "annual_downside_deviation",
        "positive_day_rate",
        "profit_factor",
        "value_at_risk_95",
        "conditional_value_at_risk_95",
        "ulcer_index",
        "cost_drag",
        "total_cost"
    }
    assert required.issubset(result.metrics)
    assert result.metrics["total_cost"] > 0


def test_buy_and_hold_is_not_daily_equal_weight_rebalancing():
    index = pd.bdate_range("2023-01-01", periods=4)
    closes = pd.DataFrame(
        {"A": [100.0, 100.0, 200.0, 200.0], "B": [100.0, 100.0, 100.0, 200.0]},
        index=index
    )
    rebalanced = run_portfolio(
        closes,
        pd.DataFrame(1.0, index=index, columns=closes.columns),
        initial_capital=100.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0
    )
    buy_hold = run_buy_and_hold(
        closes,
        initial_capital=100.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0
    )
    assert np.isclose(buy_hold.metrics["total_return"], 1.0)
    assert np.isclose(rebalanced.metrics["total_return"], 1.25)
    assert not np.allclose(buy_hold.held_weights, rebalanced.held_weights)


def test_uninvested_cash_has_zero_return_and_is_reported():
    index = pd.bdate_range("2023-01-01", periods=4)
    closes = pd.DataFrame({"A": [100.0] * 4}, index=index)
    result = run_portfolio(
        closes,
        pd.DataFrame(0.0, index=index, columns=closes.columns),
        initial_capital=100.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0
    )
    assert np.allclose(result.returns, 0.0)
    assert np.allclose(result.cash_exposure, 1.0)
    assert np.isclose(result.metrics["sharpe"], 0.0)


def test_rl_sleeve_reward_matches_backtest_turnover_and_return():
    index = pd.bdate_range("2023-01-01", periods=4)
    closes = pd.DataFrame({"A": [100.0, 110.0, 99.0, 99.0]}, index=index)
    targets = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0]}, index=index)
    result = run_portfolio(closes, targets, 100.0, transaction_cost_bps=10.0, slippage_bps=0.0)
    state = TradingState.create(
        np.zeros((4, 1), dtype=np.float32),
        closes["A"].to_numpy(dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        lookback=1,
        cost_rate=0.001,
        short_borrow_rate=0.0
    )
    _, first_reward, _ = state.step(1)
    _, second_reward, _ = state.step(1)
    assert np.isclose(first_reward, result.returns.iloc[1])
    assert np.isclose(second_reward, result.returns.iloc[2])
