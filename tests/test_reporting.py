import json

import numpy as np
import pandas as pd

from garl_trading.reporting.visualise import build_report
from garl_trading.reporting.visualise import summary as build_summary


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
            "repetition": [0] * 4,
        }
    )
    result = build_summary(metrics, 0.95)
    assert set(result["baseline"]) == {"A", "B"}
    assert (result["sharpe_ci"] > 0).all()


def test_report_emits_each_dissertation_figure_as_a_separate_file(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "data").mkdir(parents=True)
    report_dir = run_dir / "report"
    report_dir.mkdir()
    dates = pd.bdate_range("2020-01-02", periods=24)
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100 * np.cumprod(1 + np.linspace(-0.01, 0.012, len(dates))),
            "volume": 1_000_000.0,
        }
    )
    prices.to_csv(run_dir / "data" / "prices.csv", index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps({"config": {"execution": {"initial_capital": 100_000}}}),
        encoding="utf-8",
    )

    metric_rows = []
    equity_rows = []
    position_rows = []
    daily_rows = []
    periods = [
        (0, "walk_forward", dates[:12], dates[0], dates[3]),
        (1, "final_holdout", dates[12:], dates[0], dates[11]),
    ]
    for baseline, multiplier in (("buy_and_hold", 1.0), ("garl_ddal", 0.7)):
        for fold, kind, period, train_start, train_end in periods:
            net_returns = multiplier * np.linspace(-0.01, 0.012, len(period))
            equity = 100_000 * np.cumprod(1 + net_returns)
            metadata = {
                "baseline": baseline,
                "fold": fold,
                "fold_kind": kind,
                "repetition": 0,
                "seed": 42,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": period[0],
                "test_end": period[-1],
            }
            metric_rows.append(
                {
                    **metadata,
                    "total_return": equity[-1] / 100_000 - 1,
                    "annual_return": np.mean(net_returns) * 252,
                    "cagr": np.mean(net_returns) * 252,
                    "annual_volatility": np.std(net_returns) * np.sqrt(252),
                    "sharpe": np.mean(net_returns) / np.std(net_returns) * np.sqrt(252),
                    "sortino": 0.5,
                    "max_drawdown": -0.05,
                    "calmar": 1.0,
                    "turnover_daily": 0.1,
                    "gross_exposure": 1.0,
                    "cost_drag": 0.01,
                }
            )
            for date, value, net_return in zip(period, equity, net_returns, strict=True):
                equity_rows.append({**metadata, "date": date, "equity": value})
                position_rows.append(
                    {**metadata, "date": date, "ticker": "AAA", "position": multiplier}
                )
                daily_rows.append(
                    {
                        **metadata,
                        "date": date,
                        "net_return": net_return,
                        "gross_return": net_return + 0.0001,
                        "cost": 0.0001,
                        "turnover": 0.1,
                    }
                )
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(equity_rows).to_csv(run_dir / "equity.csv", index=False)
    pd.DataFrame(position_rows).to_csv(run_dir / "positions.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(run_dir / "daily_returns.csv", index=False)

    build_report(run_dir)

    expected_figures = {
        "sharpe_ranking.png",
        "return_vs_drawdown.png",
        "cumulative_returns_net.png",
        "turnover.png",
        "fold_stability.png",
        "data_split_timeline.png",
        "sharpe_over_time.png",
        "cost_sensitivity.png",
        "crash_period_cumulative_returns.png",
    }
    assert expected_figures == {path.name for path in report_dir.glob("*.png")}
    assert (report_dir / "performance_comparison.csv").exists()
    assert (report_dir / "sharpe_over_time.md").exists()
