import json

import numpy as np
import pandas as pd

from garl_trading.reporting.visualise import (BASELINE_STYLES, build_report, close_price_matrix,
                                              cost_sensitivity_scope, cumulative_seed_summary,
                                              display_label, plot_rl_seed_sharpe_distribution,
                                              representative_runs, rl_seed_sharpe_rows,
                                              trade_timing_summary, training_diagnostic_series)
from garl_trading.reporting.visualise import summary as build_summary


def test_every_baseline_has_a_unique_colour_and_common_solid_style():
    expected = {
        "buy_and_hold",
        "equal_weight_rebalanced",
        "arimax_static",
        "arimax_rolling",
        "random_forest",
        "lstm",
        "tcn",
        "transformer",
        "single_a2c",
        "single_ppo",
        "single_dqn",
        "independent_a2c",
        "independent_ppo",
        "independent_dqn",
        "garl_ddal",
        "selective_garl_ddal",
    }
    assert set(BASELINE_STYLES) == expected
    assert len({style[0] for style in BASELINE_STYLES.values()}) == len(expected)
    assert all(style[1] == "-" for style in BASELINE_STYLES.values())
    assert all(style[2] == "" for style in BASELINE_STYLES.values())
    assert all(style[3] == "" for style in BASELINE_STYLES.values())


def test_publication_labels_distinguish_joint_rl_and_ddqn():
    assert display_label("single_a2c") == "Joint A2C"
    assert display_label("single_dqn") == "Joint DDQN"
    assert display_label("garl_ddal") == "GARL-DDAL"
    assert display_label("equal_weight_rebalanced") == "Daily equal-weight rebalancing"


def test_close_price_matrix_uses_datetime_axis():
    prices = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "ticker": ["AAA", "AAA"],
            "close": [100.0, 101.0]
        })
    closes = close_price_matrix(prices)
    assert isinstance(closes.index, pd.DatetimeIndex)
    assert closes.loc[pd.Timestamp("2024-01-03"), "AAA"] == 101.0


def test_cumulative_seed_summary_reports_empirical_seed_dispersion():
    dates = pd.bdate_range("2024-01-02", periods=3)
    frame = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "repetition": [0] * 3 + [1] * 3,
            "seed": [42] * 3 + [43] * 3,
            "equity": [100, 110, 121, 100, 90, 81]
        })
    result = cumulative_seed_summary(frame)
    assert (result["seed_count"] == 2).all()
    assert np.isclose(result.loc[dates[-1], "mean"], 1.01)
    assert result.loc[dates[-1], "lower"] < result.loc[dates[-1], "mean"]
    assert result.loc[dates[-1], "upper"] > result.loc[dates[-1], "mean"]


def test_rl_seed_selector_and_distribution_use_final_holdout(tmp_path):
    metrics = pd.DataFrame(
        {
            "baseline": ["garl_ddal"] * 5 + ["buy_and_hold"] * 2,
            "fold": [0, 1, 1, 1, 1, 0, 1],
            "fold_kind": [
                "walk_forward",
                "final_holdout",
                "final_holdout",
                "final_holdout",
                "final_holdout",
                "walk_forward",
                "final_holdout",
            ],
            "repetition": [0, 0, 1, 2, 3, 0, 0],
            "seed": [42, 42, 43, 44, 45, 42, 42],
            "sharpe": [0.1, 0.1, 0.3, 0.7, 0.9, 0.2, 0.8]
        })
    rows, label = rl_seed_sharpe_rows(metrics)
    assert label == "final holdout"
    assert set(rows["baseline"]) == {"garl_ddal"}
    path = tmp_path / "seed_sharpe.png"
    assert plot_rl_seed_sharpe_distribution(metrics, path)
    assert path.exists()


def test_representative_run_is_observed_seed_nearest_median_sharpe():
    metrics = pd.DataFrame(
        {
            "baseline": ["garl_ddal"] * 4,
            "fold": [5] * 4,
            "fold_kind": ["final_holdout"] * 4,
            "repetition": [0, 1, 2, 3],
            "seed": [42, 43, 44, 45],
            "sharpe": [0.1, 0.3, 0.7, 0.9]
        })
    selected = representative_runs(metrics).iloc[0]
    assert selected.seed == 43
    assert selected.sharpe == 0.3
    assert selected.run_count == 4


def test_training_diagnostic_uses_completed_epochs_and_equal_run_weighting():
    diagnostics = pd.DataFrame(
        {
            "baseline": ["garl_ddal"] * 5,
            "fold": [0] * 5,
            "repetition": [0, 0, 1, 1, 1],
            "seed": [42, 42, 43, 43, 43],
            "epoch": [0, 0, 0, 1, 99],
            "agent": ["A", "B", "A", "A", "A"],
            "loss": [1.0, 3.0, 4.0, np.nan, 2.0]
        })
    result = training_diagnostic_series(diagnostics, "loss")
    assert result["completed_epoch"].tolist() == [1, 100]
    assert np.isclose(result.iloc[0]["loss"], 3.0)


def test_cost_sensitivity_uses_one_explicit_evaluation_scope():
    sensitivity = pd.DataFrame(
        {
            "baseline": ["garl_ddal"] * 4,
            "fold": [0, 0, 5, 5],
            "fold_kind": ["walk_forward", "walk_forward", "final_holdout", "final_holdout"],
            "repetition": [0, 1, 0, 1],
            "seed": [42, 43, 42, 43],
            "cost_bps": [5] * 4,
            "sharpe": [0.1, 0.3, 0.8, 1.0]
        })
    scoped, label = cost_sensitivity_scope(sensitivity)
    assert label == "final holdout (mean across training seeds)"
    assert set(scoped["fold_kind"]) == {"final_holdout"}
    assert np.isclose(scoped["sharpe"].mean(), 0.9)


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
        })
    result = build_summary(metrics, 0.95)
    assert set(result["baseline"]) == {"A", "B"}
    assert (result["sharpe_ci"] > 0).all()


def test_final_holdout_interval_is_seed_scoped_and_not_fabricated_for_single_run():
    metrics = pd.DataFrame(
        {
            "baseline": ["buy_and_hold", "garl_ddal", "garl_ddal"],
            "fold_kind": ["final_holdout"] * 3,
            "sharpe": [0.8, 0.5, 0.9]
        })
    result = build_summary(metrics, 0.95).set_index("baseline")
    assert np.isnan(result.loc["buy_and_hold", "sharpe_ci"])
    assert result.loc["buy_and_hold", "interval_basis"] == "not_estimable"
    assert result.loc["garl_ddal", "sharpe_ci"] > 0
    assert result.loc["garl_ddal", "interval_basis"] == "training_seed"


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
            "volume": 1_000_000.0
        })
    prices.to_csv(run_dir / "data" / "prices.csv", index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps({"config": {"execution": {"initial_capital": 100_000}}}),
        encoding="utf-8")

    metric_rows = []
    equity_rows = []
    position_rows = []
    daily_rows = []
    trade_rows = []
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
                "test_end": period[-1]
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
                    "cash_exposure": 0.0,
                    "cost_drag": 0.01
                })
            for date, value, net_return in zip(period, equity, net_returns, strict=True):
                equity_rows.append({**metadata, "date": date, "equity": value})
                position_rows.append(
                    {**metadata, "date": date, "ticker": "AAA", "position": multiplier})
                daily_rows.append(
                    {
                        **metadata,
                        "date": date,
                        "net_return": net_return,
                        "gross_return": net_return + 0.0001,
                        "cost": 0.0001,
                        "turnover": 0.1,
                        "cash_exposure": 0.0
                    })
                if baseline == "garl_ddal" and date in {period[1], period[6]}:
                    trade_rows.append(
                        {
                            **metadata,
                            "date": date,
                            "ticker": "AAA",
                            "pretrade_position": 0.0,
                            "target_position": 1.0 if date == period[1] else 0.0,
                            "executed_change": 1.0 if date == period[1] else -1.0,
                            "execution_price": float(
                                prices.loc[prices["date"] == date, "close"].iloc[0]),
                            "transaction_cost": 0.0007,
                            "short_borrow_cost": 0.0
                        })
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(equity_rows).to_csv(run_dir / "equity.csv", index=False)
    pd.DataFrame(position_rows).to_csv(run_dir / "positions.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(run_dir / "daily_returns.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(run_dir / "trades.csv", index=False)
    prediction_rows = []
    for date, actual in zip(dates[12:], np.linspace(-0.01, 0.012, 12), strict=True):
        prediction_rows.append(
            {
                "date": date,
                "ticker": "AAA",
                "prediction": actual * 0.6,
                "actual_return": actual,
                "baseline": "random_forest",
                "fold": 1,
                "fold_kind": "final_holdout",
                "repetition": 0,
                "seed": 42
            })
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False)

    build_report(run_dir)

    expected_figures = {
        "sharpe_ranking.png",
        "return_vs_drawdown.png",
        "cumulative_returns_passive.png",
        "cumulative_returns_group_agent_rl.png",
        "turnover.png",
        "fold_stability.png",
        "data_split_timeline.png",
        "sharpe_over_time_passive.png",
        "sharpe_over_time_group_agent_rl.png",
        "cost_sensitivity_passive.png",
        "cost_sensitivity_group_agent_rl.png",
        "crash_period_cumulative_returns_passive.png",
        "crash_period_cumulative_returns_group_agent_rl.png",
        "prediction_vs_actual_random_forest.png",
        "trade_actions_garl_ddal_AAA.png",
    }
    assert expected_figures == {path.name for path in report_dir.glob("*.png")}
    assert (report_dir / "performance_comparison.csv").exists()
    assert (report_dir / "sharpe_over_time.md").exists()
    assert (report_dir / "trade_timing_summary.csv").exists()


def test_trade_timing_summary_uses_executed_trade_direction():
    dates = pd.bdate_range("2024-01-02", periods=30)
    prices = pd.DataFrame(
        {"date": dates, "ticker": "AAA", "close": np.arange(100.0, 130.0)})
    metadata = {
        "baseline": "garl_ddal",
        "fold": 1,
        "fold_kind": "final_holdout",
        "repetition": 0,
        "seed": 42
    }
    trades = pd.DataFrame(
        [
            {**metadata, "date": dates[20], "ticker": "AAA", "executed_change": 1.0},
            {**metadata, "date": dates[22], "ticker": "AAA", "executed_change": -1.0},
        ])
    result = trade_timing_summary(trades, prices).set_index("side")
    assert set(result.index) == {"buy", "sell"}
    assert result.loc["buy", "directional_hit_rate_5"] == 1.0
    assert result.loc["sell", "directional_hit_rate_5"] == 0.0
    assert result.loc["buy", "runs"] == 1
    assert result.loc["buy", "trades_per_run_mean"] == 1
    assert "matched_random_hit_rate_5" in result
