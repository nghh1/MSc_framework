import numpy as np
import pandas as pd
import pytest

from garl_trading.backtest import run_buy_and_hold, run_equal_weight_rebalanced, run_portfolio
from garl_trading.rl.core import TradingState


def test_backtest_delays_positions_one_bar_and_charges_turnover():
    index = pd.bdate_range("2023-01-01", periods=4)
    closes = pd.DataFrame({"AAA": [100.0, 110.0, 110.0, 110.0]}, index=index)
    positions = pd.DataFrame({"AAA": [1.0, 1.0, 0.0, 0.0]}, index=index)
    result = run_portfolio(
        closes, positions, initial_capital=100.0, transaction_cost_bps=10.0, slippage_bps=0.0)
    assert result.held_positions.iloc[0, 0] == 0
    assert result.held_positions.iloc[1, 0] == 1
    assert np.isclose(result.returns.iloc[1], 0.10 - 0.001)
    assert result.costs.sum() > 0


def test_all_models_share_equal_weight_portfolio_contract():
    index = pd.bdate_range("2023-01-01", periods=3)
    closes = pd.DataFrame({"A": [100, 110, 110], "B": [100, 100, 90]}, index=index)
    positions = pd.DataFrame(1.0, index=index, columns=closes.columns)
    result = run_portfolio(
        closes, positions, initial_capital=100, transaction_cost_bps=0, slippage_bps=0)
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
        index=index)
    rebalanced = run_equal_weight_rebalanced(
        closes,
        initial_capital=100.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0)
    buy_hold = run_buy_and_hold(
        closes,
        initial_capital=100.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0)
    assert np.isclose(buy_hold.metrics["total_return"], 1.0)
    assert np.isclose(rebalanced.metrics["total_return"], 1.25)
    assert not np.allclose(buy_hold.held_weights, rebalanced.held_weights)
    assert np.count_nonzero(buy_hold.turnover.to_numpy()) == 1
    assert np.count_nonzero(rebalanced.turnover.to_numpy()) > 1
    assert np.allclose(buy_hold.held_weights.iloc[0], 0.0)
    assert np.allclose(buy_hold.held_weights.iloc[1], 0.5)


def test_uninvested_cash_has_zero_return_and_is_reported():
    index = pd.bdate_range("2023-01-01", periods=4)
    closes = pd.DataFrame({"A": [100.0] * 4}, index=index)
    result = run_portfolio(
        closes,
        pd.DataFrame(0.0, index=index, columns=closes.columns),
        initial_capital=100.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0)
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
        short_borrow_rate=0.0)
    _, first_reward, _ = state.step(1)
    _, second_reward, _ = state.step(1)
    assert np.isclose(first_reward, result.returns.iloc[1])
    assert np.isclose(second_reward, result.returns.iloc[2])


def test_common_rebalance_threshold_matches_rl_reward_and_backtest():
    index = pd.bdate_range("2023-01-01", periods=3)
    closes = pd.DataFrame({"A": [100.0, 110.0, 110.0]}, index=index)
    targets = pd.DataFrame({"A": [0.05, 0.05, 0.05]}, index=index)
    result = run_portfolio(
        closes,
        targets,
        100.0,
        transaction_cost_bps=10.0,
        slippage_bps=0.0,
        rebalance_threshold=0.10)
    state = TradingState.create(
        np.zeros((3, 1), dtype=np.float32),
        closes["A"].to_numpy(dtype=np.float32),
        np.asarray([0.0, 0.05], dtype=np.float32),
        lookback=1,
        cost_rate=0.001,
        short_borrow_rate=0.0,
        rebalance_threshold=0.10)
    _, reward, _ = state.step(1)
    assert np.isclose(reward, result.returns.iloc[1])
    assert np.isclose(reward, 0.0)
    assert np.isclose(result.turnover.sum(), 0.0)


def test_active_execution_only_trades_after_scheduled_decisions_and_records_orders():
    index = pd.bdate_range("2024-01-02", periods=8)
    closes = pd.DataFrame({"A": 100.0}, index=index)
    targets = pd.DataFrame(
        {"A": [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0]}, index=index)
    result = run_portfolio(
        closes,
        targets,
        initial_capital=100.0,
        transaction_cost_bps=10.0,
        slippage_bps=0.0,
        decision_interval=3)
    traded_dates = result.turnover.index[result.turnover > 0].tolist()
    assert traded_dates == [index[1], index[4], index[7]]
    assert result.trades["date"].tolist() == traded_dates
    assert np.allclose(
        result.trades.groupby("date")["transaction_cost"].sum(),
        result.costs.loc[traded_dates])


def test_five_day_rl_reward_matches_compounded_backtest_return():
    index = pd.bdate_range("2024-02-01", periods=7)
    closes = pd.DataFrame(
        {"A": [100.0, 102.0, 101.0, 104.0, 103.0, 106.0, 106.0]}, index=index)
    targets = pd.DataFrame(1.0, index=index, columns=["A"])
    result = run_portfolio(
        closes,
        targets,
        initial_capital=100.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        decision_interval=5)
    state = TradingState.create(
        np.zeros((7, 1), dtype=np.float32),
        closes["A"].to_numpy(dtype=np.float32),
        np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
        lookback=1,
        cost_rate=0.0,
        short_borrow_rate=0.0,
        decision_interval=5)
    _, reward, _ = state.step(2)
    expected = (1.0 + result.returns.iloc[1:6]).prod() - 1.0
    assert np.isclose(reward, expected)


def test_short_squeeze_forces_costed_cover_without_exposure_explosion():
    index = pd.bdate_range("2008-10-10", periods=7)
    closes = pd.DataFrame(
        {"MS": [100.0, 100.0, 186.983453, 226.650000, 190.0, 195.0, 198.0]},
        index=index)
    targets = pd.DataFrame(-1.0, index=index, columns=["MS"])
    result = run_portfolio(
        closes,
        targets,
        initial_capital=100.0,
        transaction_cost_bps=10.0,
        slippage_bps=0.0,
        decision_interval=5)
    state = TradingState.create(
        np.zeros((7, 1), dtype=np.float32),
        closes["MS"].to_numpy(dtype=np.float32),
        np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
        lookback=1,
        cost_rate=0.001,
        short_borrow_rate=0.0,
        decision_interval=5)
    _, reward, _ = state.step(0)

    assert np.isfinite(result.returns).all()
    assert result.held_positions.abs().to_numpy().max() <= 1.0
    assert result.metrics["gross_exposure"] <= 1.0
    forced = result.trades[result.trades["pretrade_position"].abs() > 1.0]
    assert not forced.empty
    assert (forced["transaction_cost"] > 0).all()
    expected = (1.0 + result.returns.iloc[1:6]).prod() - 1.0
    assert np.isclose(reward, expected, atol=1e-6)


def test_active_backtest_rejects_invalid_close_prices():
    index = pd.bdate_range("2024-01-02", periods=3)
    closes = pd.DataFrame({"A": [100.0, np.nan, 101.0]}, index=index)
    targets = pd.DataFrame(0.0, index=index, columns=["A"])
    with pytest.raises(ValueError, match="finite, strictly positive"):
        run_portfolio(closes, targets, 100.0, 0.0, 0.0)
