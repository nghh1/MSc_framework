import numpy as np
import pandas as pd

from garl_trading.backtest import run_portfolio


def test_backtest_delays_positions_one_bar_and_charges_turnover():
    index = pd.bdate_range("2023-01-01", periods=4)
    closes = pd.DataFrame({"AAA": [100.0, 110.0, 110.0, 110.0]}, index=index)
    positions = pd.DataFrame({"AAA": [1.0, 1.0, 0.0, 0.0]}, index=index)
    result = run_portfolio(
        closes,
        positions,
        initial_capital=100.0,
        transaction_cost_bps=10.0,
        slippage_bps=0.0,
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
        closes,
        positions,
        initial_capital=100,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    assert np.isclose(result.gross_returns.iloc[1], 0.05)

