import pandas as pd
from garl_trading.reporting.visualise import summary


def test_summary_reports_seed_and_fold_uncertainty():
    metrics = pd.DataFrame(
        {
            "baseline": ["A", "A", "B", "B"],
            "total_return": [0.1, 0.2, 0.0, 0.1],
            "cagr": [0.1, 0.2, 0.0, 0.1],
            "annual_volatility": [0.2] * 4,
            "sharpe": [0.5, 1.0, 0.0, 0.5],
            "sortino": [0.7, 1.2, 0.1, 0.6],
            "max_drawdown": [-0.2, -0.1, -0.3, -0.2],
            "turnover_daily": [0.1, 0.2, 0.1, 0.1],
            "gross_exposure": [1.0] * 4,
            "fold": [0, 1, 0, 1],
            "fold_kind": ["walk_forward"] * 4,
            "repetition": [0] * 4
        }
    )
    summary = summary(metrics, 0.95)
    assert set(summary["baseline"]) == {"A", "B"}
    assert (summary["sharpe_ci"] > 0).all()
