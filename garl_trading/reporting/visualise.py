from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from garl_trading.backtest import run_portfolio

COLOURS = {
    "buy_and_hold": "#8A94A6",
    "arimax_static": "#4378BF",
    "arimax_rolling": "#2457A6",
    "random_forest": "#29A37A",
    "lstm": "#E09F3E",
    "tcn": "#C56B42",
    "tft": "#A44A8B",
    "single_a2c": "#7559B3",
    "single_ppo": "#6147A6",
    "single_dqn": "#493886",
    "independent_a2c": "#B15A8A",
    "independent_ppo": "#9C4878",
    "independent_dqn": "#7E355F",
    "garl_ddal": "#D1495B",
}

SUMMARY_METRICS = [
    "total_return",
    "annual_return",
    "cagr",
    "annual_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "turnover_daily",
    "gross_exposure",
    "cost_drag",
]


def colour(name: str) -> str:
    return COLOURS.get(name, "#52616B")


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)


def _format_dates(axis: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _analysis_rows(metrics: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "fold_kind" in metrics and (metrics["fold_kind"] == "final_holdout").any():
        return metrics[metrics["fold_kind"] == "final_holdout"].copy(), "final_holdout"
    if {"fold", "repetition"}.issubset(metrics.columns):
        numeric = [column for column in SUMMARY_METRICS if column in metrics]
        grouped = metrics.groupby(["baseline", "fold"], as_index=False)[numeric].mean()
        return grouped, "walk_forward_fold_mean"
    return metrics.copy(), "all_rows"


def summary(metrics: pd.DataFrame, confidence: float) -> pd.DataFrame:
    analysis, scope = _analysis_rows(metrics)
    numeric = [column for column in SUMMARY_METRICS if column in analysis]
    rows = []
    for baseline, group in analysis.groupby("baseline", sort=False):
        row: dict[str, float | int | str] = {
            "baseline": baseline,
            "observations": len(group),
            "scope": scope,
        }
        for column in numeric:
            values = group[column].dropna()
            mean = float(values.mean()) if len(values) else np.nan
            sem = float(values.sem()) if len(values) > 1 else 0.0
            critical = (
                float(stats.t.ppf((1 + confidence) / 2, len(values) - 1))
                if len(values) > 1
                else 0.0
            )
            row[f"{column}_mean"] = mean
            row[f"{column}_ci"] = sem * critical
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values("sharpe_mean", ascending=False) if len(result) else result


def plot_sharpe_ranking(report: pd.DataFrame, path: Path) -> None:
    names = report["baseline"].tolist()
    fig, axis = plt.subplots(figsize=(10, max(5, 0.42 * len(names))), constrained_layout=True)
    y = np.arange(len(names))
    axis.barh(
        y,
        report["sharpe_mean"],
        xerr=report["sharpe_ci"],
        color=[colour(name) for name in names],
        capsize=3,
    )
    axis.set_yticks(y, names)
    axis.invert_yaxis()
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Annualised Sharpe ratio")
    axis.set_title("Out-of-sample Sharpe ratio with confidence interval")
    _style_axis(axis)
    _save(fig, path)


def plot_return_drawdown(report: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for row in report.itertuples(index=False):
        axis.scatter(
            abs(row.max_drawdown_mean),
            row.cagr_mean,
            s=70,
            color=colour(row.baseline),
            label=row.baseline,
        )
    axis.set_xlabel("Absolute maximum drawdown")
    axis.set_ylabel("CAGR")
    axis.set_title("Out-of-sample return versus drawdown")
    axis.legend(fontsize=8, ncol=2)
    _style_axis(axis)
    _save(fig, path)


def _chosen_equity(equity: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if (equity["fold_kind"] == "final_holdout").any():
        chosen = equity[equity["fold_kind"] == "final_holdout"].copy()
        label = "final holdout"
    else:
        fold = equity.loc[equity["fold_kind"] == "walk_forward", "fold"].max()
        chosen = equity[(equity["fold_kind"] == "walk_forward") & (equity["fold"] == fold)].copy()
        label = f"walk-forward fold {fold}"
    chosen["date"] = pd.to_datetime(chosen["date"])
    return chosen, label


def plot_cumulative_returns(equity: pd.DataFrame, path: Path) -> None:
    chosen, label = _chosen_equity(equity)
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for baseline, group in chosen.groupby("baseline"):
        curves = group.pivot_table(index="date", columns="repetition", values="equity")
        curves = curves.divide(curves.iloc[0])
        cumulative = curves.mean(axis=1) - 1
        axis.plot(cumulative.index, cumulative, label=baseline, color=colour(baseline), lw=1.5)
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_ylabel("Net cumulative return")
    axis.set_title(f"Cumulative returns after transaction costs — {label}")
    _format_dates(axis)
    _style_axis(axis)
    axis.legend(fontsize=8, ncol=2)
    _save(fig, path)


def plot_turnover(report: pd.DataFrame, path: Path) -> None:
    names = report["baseline"].tolist()
    fig, axis = plt.subplots(figsize=(10, max(5, 0.42 * len(names))), constrained_layout=True)
    y = np.arange(len(names))
    axis.barh(y, report["turnover_daily_mean"], color=[colour(name) for name in names])
    axis.set_yticks(y, names)
    axis.invert_yaxis()
    axis.set_xlabel("Fraction of portfolio traded per day")
    axis.set_title("Average daily turnover")
    _style_axis(axis)
    _save(fig, path)


def plot_fold_stability(metrics: pd.DataFrame, path: Path) -> pd.DataFrame:
    walk = metrics[metrics["fold_kind"] == "walk_forward"]
    pivot = walk.pivot_table(index="baseline", columns="fold", values="sharpe", aggfunc="mean")
    fig, axis = plt.subplots(figsize=(10, max(4, 0.5 * len(pivot))), constrained_layout=True)
    image = axis.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    axis.set_xticks(np.arange(len(pivot.columns)), [f"Fold {value}" for value in pivot.columns])
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            axis.text(
                column,
                row,
                "—" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
            )
    axis.set_title("Sharpe ratio stability across walk-forward folds")
    fig.colorbar(image, ax=axis, label="Sharpe ratio")
    _save(fig, path)
    return pivot


def plot_split_timeline(metrics: pd.DataFrame, path: Path) -> None:
    required = {"fold", "fold_kind", "train_start", "train_end", "test_start", "test_end"}
    if not required.issubset(metrics.columns):
        return
    folds = metrics[list(required)].drop_duplicates().sort_values("test_start")
    for column in ("train_start", "train_end", "test_start", "test_end"):
        folds[column] = pd.to_datetime(folds[column])
    fig, axis = plt.subplots(figsize=(11, max(4, 0.65 * len(folds))), constrained_layout=True)
    for row_number, row in enumerate(folds.itertuples(index=False)):
        train_start = mdates.date2num(row.train_start)
        test_start = mdates.date2num(row.test_start)
        axis.broken_barh(
            [(train_start, mdates.date2num(row.train_end) - train_start)],
            (row_number - 0.3, 0.6),
            facecolors="#5B8FF9",
        )
        axis.broken_barh(
            [(test_start, mdates.date2num(row.test_end) - test_start)],
            (row_number - 0.3, 0.6),
            facecolors="#F6BD16" if row.fold_kind == "walk_forward" else "#E8684A",
        )
    labels = [f"{row.fold_kind.replace('_', ' ')} {row.fold}" for row in folds.itertuples()]
    axis.set_yticks(np.arange(len(folds)), labels)
    axis.set_title("Purged walk-forward training and evaluation timeline")
    axis.set_xlabel("Date")
    _format_dates(axis)
    _style_axis(axis)
    _save(fig, path)


def sharpe_over_time_table(metrics: pd.DataFrame) -> pd.DataFrame:
    walk = metrics[metrics["fold_kind"] == "walk_forward"].copy()
    walk["test_start"] = pd.to_datetime(walk["test_start"])
    return walk.pivot_table(
        index=["fold", "test_start", "test_end"],
        columns="baseline",
        values="sharpe",
        aggfunc="mean",
    ).reset_index()


def plot_sharpe_over_time(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        return
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    baseline_columns = [c for c in table if c not in {"fold", "test_start", "test_end"}]
    dates = pd.to_datetime(table["test_start"])
    for baseline in baseline_columns:
        axis.plot(dates, table[baseline], marker="o", label=baseline, color=colour(baseline))
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_ylabel("Mean Sharpe ratio")
    axis.set_title("Sharpe ratios over successive out-of-sample periods")
    _format_dates(axis)
    _style_axis(axis)
    axis.legend(fontsize=8, ncol=2)
    _save(fig, path)


def plot_crash_period(daily: pd.DataFrame, path: Path) -> int | None:
    if daily.empty:
        return None
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    benchmark = daily[daily["baseline"] == "buy_and_hold"]
    if benchmark.empty:
        return None
    year_counts = benchmark.groupby("year")["date"].nunique()
    complete_years = year_counts[year_counts >= 126].index
    candidates = benchmark[benchmark["year"].isin(complete_years)]
    if candidates.empty:
        candidates = benchmark
    yearly = candidates.groupby("year")["net_return"].apply(lambda values: (1 + values).prod() - 1)
    crash_year = int(yearly.idxmin())
    crash = daily[daily["year"] == crash_year]
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for baseline, group in crash.groupby("baseline"):
        mean_return = group.pivot_table(
            index="date", columns="repetition", values="net_return"
        ).mean(axis=1)
        cumulative = (1 + mean_return).cumprod() - 1
        axis.plot(cumulative.index, cumulative, label=baseline, color=colour(baseline), lw=1.5)
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_ylabel("Net cumulative return")
    axis.set_title(f"Cumulative returns during the worst buy-and-hold year ({crash_year})")
    _format_dates(axis)
    _style_axis(axis)
    axis.legend(fontsize=8, ncol=2)
    _save(fig, path)
    return crash_year


def cost_sensitivity(run_dir: Path, positions: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    prices = pd.read_csv(run_dir / "data" / "prices.csv", parse_dates=["date"])
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    initial_capital = manifest["config"]["execution"]["initial_capital"]
    rows = []
    keys = ["baseline", "fold", "fold_kind", "repetition", "seed"]
    for key, group in positions.groupby(keys, dropna=False):
        metadata = dict(zip(keys, key, strict=True))
        matrix = group.pivot(index="date", columns="ticker", values="position")
        close_window = closes.reindex(index=matrix.index, columns=matrix.columns)
        for total_cost in (0, 5, 10, 20, 40):
            result = run_portfolio(
                close_window,
                matrix,
                initial_capital=initial_capital,
                transaction_cost_bps=total_cost,
                slippage_bps=0,
            )
            rows.append({**metadata, "cost_bps": total_cost, "sharpe": result.metrics["sharpe"]})
    sensitivity = pd.DataFrame(rows)
    if path is not None:
        fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for baseline, group in sensitivity.groupby("baseline"):
            curve = group.groupby("cost_bps")["sharpe"].mean()
            axis.plot(curve.index, curve.values, marker="o", label=baseline, color=colour(baseline))
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_title("Sharpe ratio under common trading-cost scenarios")
        axis.set_xlabel("Transaction cost and slippage (bps per unit turnover)")
        axis.set_ylabel("Mean out-of-sample Sharpe ratio")
        _style_axis(axis)
        axis.legend(fontsize=8, ncol=2)
        _save(fig, path)
    return sensitivity


def performance_table(report: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "total_return_mean": "cumulative_return",
        "annual_return_mean": "annual_return",
        "annual_volatility_mean": "annual_volatility",
        "sharpe_mean": "sharpe_ratio",
        "sortino_mean": "sortino_ratio",
        "max_drawdown_mean": "max_drawdown",
        "cost_drag_mean": "cost_drag",
    }
    available = ["baseline", *[column for column in columns if column in report]]
    return report[available].rename(columns=columns)


def markdown_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include="number"):
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(map(str, row)) + " |"
        for row in formatted.itertuples(index=False, name=None)
    )
    return f"{header}\n{separator}\n{body}"


def build_report(
    run_dir: str | Path,
    confidence: float = 0.95,
    formats: tuple[str, ...] = ("png", "csv", "md"),
) -> None:
    run_dir = Path(run_dir)
    report_dir = run_dir / "report"
    report_dir.mkdir(exist_ok=True)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    equity = pd.read_csv(run_dir / "equity.csv", parse_dates=["date"])
    positions = pd.read_csv(run_dir / "positions.csv", parse_dates=["date"])
    daily_path = run_dir / "daily_returns.csv"
    daily = pd.read_csv(daily_path, parse_dates=["date"]) if daily_path.exists() else pd.DataFrame()

    report = summary(metrics, confidence)
    comparison = performance_table(report)
    sharpe_table = sharpe_over_time_table(metrics)
    sensitivity = cost_sensitivity(
        run_dir,
        positions,
        report_dir / "cost_sensitivity.png" if "png" in formats else None,
    )

    if "csv" in formats:
        report.to_csv(report_dir / "summary.csv", index=False)
        comparison.to_csv(report_dir / "performance_comparison.csv", index=False)
        sharpe_table.to_csv(report_dir / "sharpe_over_time.csv", index=False)
        sensitivity.to_csv(report_dir / "cost_sensitivity.csv", index=False)
    if "md" in formats:
        (report_dir / "performance_comparison.md").write_text(
            markdown_table(comparison), encoding="utf-8"
        )
        (report_dir / "sharpe_over_time.md").write_text(
            markdown_table(sharpe_table), encoding="utf-8"
        )

    crash_year = None
    if "png" in formats:
        plot_sharpe_ranking(report, report_dir / "sharpe_ranking.png")
        plot_return_drawdown(report, report_dir / "return_vs_drawdown.png")
        plot_cumulative_returns(equity, report_dir / "cumulative_returns_net.png")
        plot_turnover(report, report_dir / "turnover.png")
        plot_fold_stability(metrics, report_dir / "fold_stability.png")
        plot_split_timeline(metrics, report_dir / "data_split_timeline.png")
        plot_sharpe_over_time(sharpe_table, report_dir / "sharpe_over_time.png")
        crash_year = plot_crash_period(daily, report_dir / "crash_period_cumulative_returns.png")

    scope = report["scope"].iloc[0].replace("_", " ") if len(report) else "unavailable"
    crash_note = (
        f"The optional crash-period figure uses {crash_year}, the worst available buy-and-hold year."
        if crash_year is not None
        else "The crash-period figure was skipped because daily return artifacts were unavailable."
    )
    narrative = (
        "# Experiment summary\n\n"
        f"Headline scope: {scope}. Confidence level: {confidence:.0%}. "
        "All equity and cumulative-return figures are net of configured transaction costs and slippage.\n\n"
        "## Performance comparison\n\n"
        f"{markdown_table(comparison)}\n\n"
        "## Reporting design\n\n"
        "Each analytical view is saved as a separate figure so it can be placed, captioned, and "
        "scaled independently in the dissertation. The split timeline documents the experimental "
        "protocol; Sharpe ranking and fold paths address level and stability; net cumulative returns "
        "show economic magnitude; drawdown, turnover, and cost sensitivity cover risk and implementability.\n\n"
        f"{crash_note}\n"
    )
    if "md" in formats:
        (report_dir / "summary.md").write_text(narrative, encoding="utf-8")
