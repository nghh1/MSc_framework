from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from garl_trading.backtest import run_buy_and_hold, run_equal_weight_rebalanced, run_portfolio

BASELINE_STYLES = {
    "buy_and_hold": ("#000000", "-", "", ""),
    "equal_weight_rebalanced": ("#7F7F7F", "-", "", ""),
    "arimax_static": ("#0072B2", "-", "", ""),
    "arimax_rolling": ("#56B4E9", "-", "", ""),
    "random_forest": ("#009E73", "-", "", ""),
    "lstm": ("#E69F00", "-", "", ""),
    "tcn": ("#A65628", "-", "", ""),
    "transformer": ("#8C6BB1", "-", "", ""),
    "single_a2c": ("#332288", "-", "", ""),
    "single_ppo": ("#117733", "-", "", ""),
    "single_dqn": ("#882255", "-", "", ""),
    "independent_a2c": ("#88CCEE", "-", "", ""),
    "independent_ppo": ("#44AA99", "-", "", ""),
    "independent_dqn": ("#AA4499", "-", "", ""),
    "garl_ddal": ("#E41A1C", "-", "", ""),
    "selective_garl_ddal": ("#6A3D9A", "-", "", "")
}
BASELINE_ORDER = tuple(BASELINE_STYLES)
COLOURS = {name: values[0] for name, values in BASELINE_STYLES.items()}
DEFAULT_STYLE = ("#52616B", "-", "", "")
BASELINE_LABELS = {
    "buy_and_hold": "Equal-weight buy-and-hold",
    "equal_weight_rebalanced": "Daily equal-weight rebalancing",
    "arimax_static": "Static ARIMAX",
    "arimax_rolling": "Rolling ARIMAX",
    "random_forest": "Random Forest",
    "lstm": "LSTM",
    "tcn": "TCN",
    "transformer": "Transformer",
    "single_a2c": "Joint A2C",
    "single_ppo": "Joint PPO",
    "single_dqn": "Joint DDQN",
    "independent_a2c": "Independent A2C",
    "independent_ppo": "Independent PPO",
    "independent_dqn": "Independent DDQN",
    "garl_ddal": "GARL-DDAL",
    "selective_garl_ddal": "Selective GARL-DDAL"
}
PASSIVE_BASELINES = ("buy_and_hold", "equal_weight_rebalanced")
BASELINE_FAMILIES = {
    "passive": PASSIVE_BASELINES,
    "statistical": ("arimax_static", "arimax_rolling"),
    "supervised_learning": ("random_forest", "lstm", "tcn", "transformer"),
    "joint_rl": ("single_a2c", "single_ppo", "single_dqn"),
    "independent_rl": ("independent_a2c", "independent_ppo", "independent_dqn"),
    "group_agent_rl": ("garl_ddal", "selective_garl_ddal")
}
FAMILY_TITLES = {
    "passive": "Passive allocation",
    "statistical": "Statistical forecasting",
    "supervised_learning": "Supervised and deep sequence learning",
    "joint_rl": "Joint reinforcement learning",
    "independent_rl": "Independent reinforcement learning",
    "group_agent_rl": "Group-agent reinforcement learning"
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
    "cash_exposure",
    "cost_drag",
]


def colour(name: str) -> str:
    return COLOURS.get(name, "#52616B")


def display_label(name: str) -> str:
    """Return a publication-facing label without changing stored identifiers."""
    return BASELINE_LABELS.get(name, name.replace("_", " ").title())


def style(name: str) -> tuple[str, str, str, str]:
    return BASELINE_STYLES.get(name, DEFAULT_STYLE)


def ordered_baselines(values) -> list[str]:
    order = {name: number for number, name in enumerate(BASELINE_ORDER)}
    return sorted(set(values), key=lambda name: (order.get(name, len(order)), name))


def family_selection(values, family: str, *, passive_reference: bool = True) -> list[str]:
    available = set(values)
    members = [name for name in BASELINE_FAMILIES[family] if name in available]
    if not members:
        return []
    if family != "passive" and passive_reference:
        members = [name for name in PASSIVE_BASELINES if name in available] + members
    return ordered_baselines(members)


def line_style(name: str, observations: int) -> dict:
    line_colour, _, _, _ = style(name)
    is_garl = name in {"garl_ddal", "selective_garl_ddal"}
    is_benchmark = name in {"buy_and_hold", "equal_weight_rebalanced"}
    return {
        "color": line_colour,
        "linestyle": "-",
        "linewidth": 2.5 if is_garl else 2.1 if is_benchmark else 1.6,
        "zorder": 4 if is_garl else 3 if is_benchmark else 2
    }


def apply_bar_styles(bars, names: list[str]) -> None:
    for bar, _ in zip(bars, names, strict=True):
        bar.set_edgecolor("#222222")
        bar.set_linewidth(0.45)


def style_axis(axis: plt.Axes) -> None:
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)


def format_dates(axis: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def format_percent(axis: plt.Axes, which: str = "y") -> None:
    formatter = mticker.PercentFormatter(xmax=1.0, decimals=1)
    getattr(axis, f"{which}axis").set_major_formatter(formatter)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analysis_rows(metrics: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "fold_kind" in metrics and (metrics["fold_kind"] == "final_holdout").any():
        return metrics[metrics["fold_kind"] == "final_holdout"].copy(), "final_holdout"
    if {"fold", "repetition"}.issubset(metrics.columns):
        numeric = [column for column in SUMMARY_METRICS if column in metrics]
        grouped = metrics.groupby(["baseline", "fold"], as_index=False)[numeric].mean()
        return grouped, "walk_forward_fold_mean"
    return metrics.copy(), "all_rows"


def summary(metrics: pd.DataFrame, confidence: float) -> pd.DataFrame:
    analysis, scope = analysis_rows(metrics)
    numeric = [column for column in SUMMARY_METRICS if column in analysis]
    rows = []
    for baseline, group in analysis.groupby("baseline", sort=False):
        row: dict[str, float | int | str] = {
            "baseline": baseline,
            "observations": len(group),
            "scope": scope,
            "interval_basis": (
                "training_seed"
                if scope == "final_holdout" and len(group) > 1
                else "walk_forward_fold"
                if scope == "walk_forward_fold_mean" and len(group) > 1
                else "not_estimable"
            )
        }
        for column in numeric:
            values = group[column].dropna()
            mean = float(values.mean()) if len(values) else np.nan
            sem = float(values.sem()) if len(values) > 1 else np.nan
            critical = (
                float(stats.t.ppf((1 + confidence) / 2, len(values) - 1))
                if len(values) > 1
                else np.nan
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
    xerr = report["sharpe_ci"].fillna(0.0).to_numpy()
    if {"sharpe_ci_lower", "sharpe_ci_upper"}.issubset(report.columns):
        means = report["sharpe_mean"].to_numpy()
        xerr = np.vstack(
            [
                np.maximum(means - report["sharpe_ci_lower"].to_numpy(), 0),
                np.maximum(report["sharpe_ci_upper"].to_numpy() - means, 0),
            ])
    bars = axis.barh(
        y,
        report["sharpe_mean"],
        xerr=xerr,
        color=[colour(name) for name in names],
        capsize=3)
    apply_bar_styles(bars, names)
    axis.set_yticks(y, [display_label(name) for name in names])
    axis.invert_yaxis()
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Annualised Sharpe ratio")
    bases = set(report.get("interval_basis", pd.Series(dtype=str)).dropna())
    if "training_seed" in bases:
        title = (
            "Final-holdout Sharpe ratio\n"
            "Error bars: 95% intervals across repeated RL training seeds"
        )
    elif "walk_forward_fold" in bases:
        title = "Out-of-sample Sharpe ratio (interval across test folds)"
    else:
        title = "Out-of-sample Sharpe ratio"
    axis.set_title(title)
    style_axis(axis)
    save(fig, path)


def chosen_evaluation_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Select the final holdout, or the latest available walk-forward fold."""
    if "fold_kind" in frame and (frame["fold_kind"] == "final_holdout").any():
        return frame[frame["fold_kind"] == "final_holdout"].copy(), "final holdout"
    if {"fold_kind", "fold"}.issubset(frame.columns):
        walk = frame[frame["fold_kind"] == "walk_forward"]
        if not walk.empty:
            fold = walk["fold"].max()
            return walk[walk["fold"] == fold].copy(), f"walk-forward fold {fold}"
    return frame.copy(), "all available evaluations"


def rl_seed_sharpe_rows(metrics: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return raw Sharpe observations for stochastic baselines in one evaluation period."""
    chosen, label = chosen_evaluation_rows(metrics)
    run_keys = [column for column in ("repetition", "seed") if column in chosen]
    if not run_keys or "sharpe" not in chosen:
        return chosen.iloc[0:0].copy(), label
    counts = chosen.groupby("baseline")[run_keys].apply(lambda values: len(values.drop_duplicates()))
    stochastic = counts[counts > 1].index
    columns = ["baseline", *run_keys, "sharpe"]
    return chosen[chosen["baseline"].isin(stochastic)].loc[:, columns].dropna(), label


def plot_rl_seed_sharpe_distribution(metrics: pd.DataFrame, path: Path) -> bool:
    rows, label = rl_seed_sharpe_rows(metrics)
    if rows.empty:
        return False
    names = rows.groupby("baseline")["sharpe"].median().sort_values(ascending=False).index.tolist()
    values = [rows.loc[rows["baseline"] == name, "sharpe"].to_numpy() for name in names]
    fig, axis = plt.subplots(
        figsize=(10, max(4.5, 0.55 * len(names))), constrained_layout=True)
    boxes = axis.boxplot(
        values,
        orientation="horizontal",
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#444444", "linewidth": 0.9},
        capprops={"color": "#444444", "linewidth": 0.9})
    axis.set_yticks(
        np.arange(1, len(names) + 1),
        [display_label(name) for name in names])
    for box, name in zip(boxes["boxes"], names, strict=True):
        box.set_facecolor(colour(name))
        box.set_edgecolor("#222222")
        box.set_linewidth(0.6)
        box.set_alpha(0.82)
    axis.invert_yaxis()
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Annualised Sharpe ratio")
    axis.set_title(f"RL Sharpe distribution across training seeds — {label}")
    style_axis(axis)
    save(fig, path)
    return True


def plot_return_drawdown(report: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for row in report.itertuples(index=False):
        point_colour, _, _, _ = style(row.baseline)
        axis.scatter(
            abs(row.max_drawdown_mean),
            row.cagr_mean,
            s=70,
            color=point_colour,
            marker="o",
            edgecolor="#222222",
            linewidth=0.5,
            label=display_label(row.baseline))
    axis.set_xlabel("Absolute maximum drawdown (%)")
    axis.set_ylabel("CAGR (%)")
    axis.set_title("Out-of-sample return versus drawdown")
    axis.legend(fontsize=8, ncol=2)
    format_percent(axis, "x")
    format_percent(axis, "y")
    style_axis(axis)
    save(fig, path)


def chosen_equity(equity: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    chosen, label = chosen_evaluation_rows(equity)
    chosen["date"] = pd.to_datetime(chosen["date"])
    return chosen, label


def close_price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Return closes on a true datetime axis for time-series plotting."""
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.pivot_table(
        index="date",
        columns="ticker",
        values="close",
        aggfunc="last").sort_index()


def cumulative_seed_summary(group: pd.DataFrame) -> pd.DataFrame:
    """Summarise normalized equity paths without treating seeds as market samples."""
    run_keys = [column for column in ("repetition", "seed") if column in group]
    columns = run_keys or None
    curves = group.pivot_table(index="date", columns=columns, values="equity")
    if isinstance(curves, pd.Series):
        curves = curves.to_frame("run")
    curves = curves.sort_index().divide(curves.iloc[0])
    result = pd.DataFrame(index=curves.index)
    result["mean"] = curves.mean(axis=1)
    result["lower"] = curves.quantile(0.10, axis=1)
    result["upper"] = curves.quantile(0.90, axis=1)
    result["seed_count"] = curves.shape[1]
    return result


def plot_cumulative_returns(equity: pd.DataFrame, path: Path, baselines: list[str],
                            family_title: str) -> None:
    chosen, label = chosen_equity(equity)
    chosen = chosen[chosen["baseline"].isin(baselines)]
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    has_seed_bands = False
    for baseline in baselines:
        group = chosen[chosen["baseline"] == baseline]
        seed_summary = cumulative_seed_summary(group)
        cumulative = seed_summary["mean"] - 1
        if int(seed_summary["seed_count"].iloc[0]) > 1:
            has_seed_bands = True
            axis.fill_between(
                seed_summary.index,
                (seed_summary["lower"] - 1).to_numpy(),
                (seed_summary["upper"] - 1).to_numpy(),
                color=colour(baseline),
                alpha=0.12,
                linewidth=0,
                zorder=1)
        axis.plot(
            cumulative.index,
            cumulative,
            label=display_label(baseline),
            **line_style(baseline, len(cumulative)))
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_ylabel("Net cumulative return")
    subtitle = "\nShading: 10th–90th percentile across training seeds" if has_seed_bands else ""
    axis.set_title(
        f"{family_title}: cumulative returns after transaction costs — {label}{subtitle}")
    format_dates(axis)
    format_percent(axis)
    style_axis(axis)
    axis.legend(fontsize=8, ncol=2)
    save(fig, path)


def plot_family_cumulative_returns(equity: pd.DataFrame, report_dir: Path) -> list[Path]:
    paths = []
    for family in BASELINE_FAMILIES:
        baselines = family_selection(equity["baseline"], family)
        if not baselines:
            continue
        path = report_dir / f"cumulative_returns_{family}.png"
        plot_cumulative_returns(equity, path, baselines, FAMILY_TITLES[family])
        paths.append(path)
    return paths


def plot_turnover(report: pd.DataFrame, path: Path) -> None:
    names = report["baseline"].tolist()
    fig, axis = plt.subplots(figsize=(10, max(5, 0.42 * len(names))), constrained_layout=True)
    y = np.arange(len(names))
    bars = axis.barh(y, report["turnover_daily_mean"], color=[colour(name) for name in names])
    apply_bar_styles(bars, names)
    axis.set_yticks(y, [display_label(name) for name in names])
    axis.invert_yaxis()
    axis.set_xlabel("Portfolio traded per day (%)")
    axis.set_title("Average daily turnover")
    format_percent(axis, "x")
    style_axis(axis)
    save(fig, path)


def plot_training_diagnostic(diagnostics: pd.DataFrame, value: str, path: Path, ylabel: str) -> None:
    if diagnostics.empty or value not in diagnostics:
        return
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    grouped = training_diagnostic_series(diagnostics, value)
    if grouped.empty:
        plt.close(fig)
        return
    for baseline in ordered_baselines(grouped["baseline"]):
        frame = grouped[grouped["baseline"] == baseline]
        axis.plot(
            frame["completed_epoch"],
            frame[value],
            label=display_label(baseline),
            **line_style(baseline, len(frame)))
    axis.set_xlim(1, int(grouped["completed_epoch"].max()))
    axis.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    axis.set_xlabel("Completed training epochs")
    axis.set_ylabel(ylabel)
    title = f"RL {ylabel.lower()} during training"
    if value == "loss":
        title += "\nLoss scales are algorithm-specific and not directly comparable"
    axis.set_title(title)
    style_axis(axis)
    axis.legend(fontsize=8, ncol=2)
    save(fig, path)


def training_diagnostic_series(diagnostics: pd.DataFrame, value: str) -> pd.DataFrame:
    """Average agents within runs before averaging equally across runs."""
    if diagnostics.empty or value not in diagnostics:
        return pd.DataFrame(columns=["baseline", "epoch", value, "completed_epoch"])
    run_keys = [
        column
        for column in ("baseline", "fold", "repetition", "seed", "epoch")
        if column in diagnostics
    ]
    run_level = diagnostics.groupby(run_keys, as_index=False, dropna=False)[value].mean()
    grouped = run_level.groupby(["baseline", "epoch"], as_index=False)[value].mean()
    grouped = grouped.dropna(subset=[value]).sort_values(["baseline", "epoch"])
    grouped["completed_epoch"] = grouped["epoch"].astype(int) + 1
    return grouped


def training_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    keys = ["baseline", "fold", "repetition", "seed"]
    runs = diagnostics.groupby(keys, as_index=False).agg(
        epochs_completed=("epoch", lambda values: int(values.max()) + 1),
        best_training_reward=("training_reward", "max"),
        final_training_reward=("training_reward", "last"))
    if "early_stopped" in diagnostics:
        stopped = diagnostics.groupby(keys)["early_stopped"].any().rename("early_stopped")
        runs = runs.merge(stopped.reset_index(), on=keys, how="left")
    return runs


def plot_fold_stability(metrics: pd.DataFrame, path: Path) -> pd.DataFrame:
    walk = metrics[metrics["fold_kind"] == "walk_forward"]
    pivot = walk.pivot_table(index="baseline", columns="fold", values="sharpe", aggfunc="mean")
    fig, axis = plt.subplots(figsize=(10, max(4, 0.5 * len(pivot))), constrained_layout=True)
    image = axis.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    axis.set_xticks(np.arange(len(pivot.columns)), [f"Fold {value}" for value in pivot.columns])
    axis.set_yticks(
        np.arange(len(pivot.index)),
        [display_label(name) for name in pivot.index])
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            axis.text(
                column,
                row,
                "—" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center")
    axis.set_title("Sharpe ratio stability across walk-forward folds")
    fig.colorbar(image, ax=axis, label="Sharpe ratio")
    save(fig, path)
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
            facecolors="#5B8FF9")
        axis.broken_barh(
            [(test_start, mdates.date2num(row.test_end) - test_start)],
            (row_number - 0.3, 0.6),
            facecolors="#F6BD16" if row.fold_kind == "walk_forward" else "#E8684A")
    labels = [
        "Final holdout"
        if row.fold_kind == "final_holdout"
        else f"Walk-forward fold {row.fold}"
        for row in folds.itertuples()
    ]
    axis.set_yticks(np.arange(len(folds)), labels)
    axis.set_title(
        "Purged walk-forward training and evaluation timeline\n"
        "White boundary gaps denote the configured purge interval")
    axis.set_xlabel("Date")
    axis.legend(
        handles=[
            Patch(facecolor="#5B8FF9", label="Training"),
            Patch(facecolor="#F6BD16", label="Walk-forward evaluation"),
            Patch(facecolor="#E8684A", label="Final holdout evaluation"),
        ],
        loc="upper left",
        fontsize=8,
        frameon=True)
    format_dates(axis)
    style_axis(axis)
    save(fig, path)


def sharpe_over_time_table(metrics: pd.DataFrame) -> pd.DataFrame:
    walk = metrics[metrics["fold_kind"] == "walk_forward"].copy()
    walk["test_start"] = pd.to_datetime(walk["test_start"])
    return walk.pivot_table(
        index=["fold", "test_start", "test_end"],
        columns="baseline",
        values="sharpe",
        aggfunc="mean").reset_index()


def plot_sharpe_over_time(table: pd.DataFrame, path: Path, baselines: list[str], family_title: str) -> None:
    if table.empty:
        return
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    dates = pd.to_datetime(table["test_start"])
    for baseline in baselines:
        axis.plot(
            dates,
            table[baseline],
            label=display_label(baseline),
            **line_style(baseline, len(dates)))
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_ylabel("Mean Sharpe ratio")
    axis.set_title(f"{family_title}: Sharpe ratios over out-of-sample periods")
    format_dates(axis)
    style_axis(axis)
    axis.legend(fontsize=8, ncol=2)
    save(fig, path)


def plot_family_sharpe_over_time(table: pd.DataFrame, report_dir: Path) -> list[Path]:
    available = [column for column in table if column not in {"fold", "test_start", "test_end"}]
    paths = []
    for family in BASELINE_FAMILIES:
        baselines = family_selection(available, family)
        if not baselines:
            continue
        path = report_dir / f"sharpe_over_time_{family}.png"
        plot_sharpe_over_time(table, path, baselines, FAMILY_TITLES[family])
        paths.append(path)
    return paths


def plot_crash_period(daily: pd.DataFrame, report_dir: Path) -> int | None:
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
    for family in BASELINE_FAMILIES:
        baselines = family_selection(crash["baseline"], family)
        if not baselines:
            continue
        fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
        for baseline in baselines:
            group = crash[crash["baseline"] == baseline]
            mean_return = group.pivot_table(
                index="date", columns="repetition", values="net_return").mean(axis=1)
            cumulative = (1 + mean_return).cumprod() - 1
            axis.plot(
                cumulative.index,
                cumulative,
                label=display_label(baseline),
                **line_style(baseline, len(cumulative)))
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_ylabel("Net cumulative return")
        axis.set_title(
            f"{FAMILY_TITLES[family]} during the worst buy-and-hold year ({crash_year})"
        )
        format_dates(axis)
        format_percent(axis)
        style_axis(axis)
        axis.legend(fontsize=8, ncol=2)
        save(fig, report_dir / f"crash_period_cumulative_returns_{family}.png")
    return crash_year


def plot_prediction_vs_actual(predictions: pd.DataFrame, report_dir: Path) -> list[Path]:
    """Create one final-evaluation forecast diagnostic per supervised baseline."""
    if predictions.empty:
        return []
    chosen, label = chosen_equity(predictions)
    paths = []
    for baseline in ordered_baselines(chosen["baseline"]):
        frame = chosen[chosen["baseline"] == baseline]
        daily = frame.groupby("date")[["prediction", "actual_return"]].mean().sort_index()
        smooth = daily.rolling(21, min_periods=5).mean()
        fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
        axis.plot(
            smooth.index,
            smooth["actual_return"],
            color="#333333",
            linewidth=1.8,
            label="Actual target-horizon return")
        axis.plot(
            smooth.index,
            smooth["prediction"],
            label="Predicted target-horizon return",
            **line_style(baseline, len(smooth)))
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.set_ylabel("Cross-stock mean return (%)")
        axis.set_title(
            f"Prediction versus actual return — {display_label(baseline)} "
            f"({label}, 21-day mean)"
        )
        format_dates(axis)
        format_percent(axis)
        style_axis(axis)
        axis.legend(fontsize=9)
        path = report_dir / f"prediction_vs_actual_{baseline}.png"
        save(fig, path)
        paths.append(path)
    return paths


def representative_runs(metrics: pd.DataFrame) -> pd.DataFrame:
    """Choose one observed run nearest each baseline's median Sharpe."""
    chosen, _ = chosen_evaluation_rows(metrics)
    required = {"baseline", "fold", "fold_kind", "repetition", "seed", "sharpe"}
    if not required.issubset(chosen.columns):
        return pd.DataFrame(columns=[*required, "run_count"])
    rows = []
    for baseline, group in chosen.groupby("baseline", sort=False):
        candidates = group.dropna(subset=["sharpe"]).copy()
        if candidates.empty:
            continue
        median = float(candidates["sharpe"].median())
        candidates["median_distance"] = (
            candidates["sharpe"] - median
        ).abs().round(12)
        selected = candidates.sort_values(
            ["median_distance", "seed", "repetition"], kind="stable").iloc[0]
        rows.append(
            {
                "baseline": baseline,
                "fold": selected["fold"],
                "fold_kind": selected["fold_kind"],
                "repetition": selected["repetition"],
                "seed": selected["seed"],
                "sharpe": selected["sharpe"],
                "run_count": len(candidates),
                "test_start": selected.get("test_start"),
                "test_end": selected.get("test_end")
            })
    return pd.DataFrame(rows)


def plot_trade_action_curves(trades: pd.DataFrame, prices: pd.DataFrame, metrics: pd.DataFrame,
                             report_dir: Path) -> list[Path]:
    """Plot one real run's actual executed changes on close prices."""
    if trades.empty or prices.empty or "executed_change" not in trades:
        return []
    chosen, label = chosen_equity(trades)
    chosen = chosen[
        ~chosen["baseline"].isin({"buy_and_hold", "equal_weight_rebalanced"})
    ]
    representatives = representative_runs(metrics)
    merge_keys = ["baseline", "fold", "fold_kind", "repetition", "seed"]
    chosen = chosen.merge(
        representatives[merge_keys], on=merge_keys, how="inner", validate="many_to_one")
    closes = close_price_matrix(prices)
    paths = []
    for baseline in ordered_baselines(chosen["baseline"]):
        frame = chosen[chosen["baseline"] == baseline]
        representative = representatives[representatives["baseline"] == baseline].iloc[0]
        run_note = (
            f"; representative seed {int(representative.seed)}, nearest median Sharpe"
            if int(representative.run_count) > 1
            else ""
        )
        for ticker in sorted(frame["ticker"].unique()):
            if ticker not in closes:
                continue
            start = pd.Timestamp(representative["test_start"])
            end = pd.Timestamp(representative["test_end"])
            price = closes[ticker].loc[start:end].dropna()
            if price.empty:
                continue
            ticker_trades = frame[frame["ticker"] == ticker].set_index("date")
            buy_dates = ticker_trades.index[ticker_trades["executed_change"] > 0]
            sell_dates = ticker_trades.index[ticker_trades["executed_change"] < 0]

            fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
            axis.plot(
                price.index,
                price,
                color="#334E68",
                linewidth=1.7,
                label="Close price",
                zorder=2)
            if len(buy_dates):
                axis.scatter(
                    buy_dates,
                    price.loc[buy_dates],
                    color="#009E73",
                    marker="^",
                    s=42,
                    edgecolor="white",
                    linewidth=0.35,
                    label="Buy / increase",
                    zorder=4)
            if len(sell_dates):
                axis.scatter(
                    sell_dates,
                    price.loc[sell_dates],
                    color="#D55E00",
                    marker="v",
                    s=42,
                    edgecolor="white",
                    linewidth=0.35,
                    label="Sell / reduce",
                    zorder=4)
            axis.set_ylabel("Close price")
            axis.set_title(
                f"Buy/sell decisions — {display_label(baseline)}, "
                f"{ticker} ({label}{run_note})")
            format_dates(axis)
            style_axis(axis)
            axis.legend(fontsize=9)
            path = report_dir / f"trade_actions_{baseline}_{ticker}.png"
            save(fig, path)
            paths.append(path)
    return paths


def trade_timing_summary(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Summarise causal price location and future movement after executed trades."""
    columns = [
        "baseline",
        "side",
        "runs",
        "trades_per_run_mean",
        "days_between_same_side_trades_mean",
        "range_position_20_mean",
        "forward_return_5_mean",
        "forward_return_20_mean",
        "favourable_excursion_5_mean",
        "adverse_excursion_5_mean",
        "directional_hit_rate_5",
        "directional_hit_rate_20",
        "matched_random_hit_rate_5",
        "directional_hit_lift_5"
    ]
    if trades.empty or prices.empty or "executed_change" not in trades:
        return pd.DataFrame(columns=columns)
    chosen, _ = chosen_equity(trades)
    market = prices.sort_values(["ticker", "date"]).copy()
    grouped = market.groupby("ticker", group_keys=False)
    rolling_low = grouped["close"].transform(lambda values: values.rolling(20).min())
    rolling_high = grouped["close"].transform(lambda values: values.rolling(20).max())
    market["range_position_20"] = (
        (market["close"] - rolling_low)
        / (rolling_high - rolling_low).replace(0, np.nan)
    )
    for horizon in (5, 20):
        market[f"forward_return_{horizon}"] = grouped["close"].transform(
            lambda values, h=horizon: values.shift(-h) / values - 1)
    future = pd.concat(
        [grouped["close"].shift(-offset) for offset in range(1, 6)], axis=1)
    market["future_max_5"] = future.max(axis=1) / market["close"] - 1
    market["future_min_5"] = future.min(axis=1) / market["close"] - 1
    merged = chosen.merge(
        market[
            [
                "date",
                "ticker",
                "range_position_20",
                "forward_return_5",
                "forward_return_20",
                "future_max_5",
                "future_min_5"
            ]
        ],
        on=["date", "ticker"],
        how="left",
        validate="many_to_one")
    merged["side"] = np.where(merged["executed_change"] > 0, "buy", "sell")
    buy = merged["side"] == "buy"
    merged["favourable_excursion_5"] = np.where(
        buy, merged["future_max_5"], -merged["future_min_5"])
    merged["adverse_excursion_5"] = np.where(
        buy, merged["future_min_5"], -merged["future_max_5"])
    merged["directional_hit_5"] = np.where(
        buy, merged["forward_return_5"] > 0, merged["forward_return_5"] < 0)
    merged["directional_hit_20"] = np.where(
        buy, merged["forward_return_20"] > 0, merged["forward_return_20"] < 0)
    run_columns = ["baseline", "fold", "fold_kind", "repetition", "seed", "side"]
    merged = merged.sort_values([*run_columns, "ticker", "date"])
    merged["days_between_same_side_trades"] = (
        merged.groupby([*run_columns, "ticker"])["date"].diff().dt.days
    )

    random_hit_rates: dict[tuple, float] = {}
    for run_key, run_trades in merged.groupby(run_columns, sort=False, dropna=False):
        baseline, fold, _, repetition, seed, side = run_key
        seed_value = (
            sum(str(baseline).encode("utf-8"))
            + 1009 * int(fold)
            + 9176 * int(repetition)
            + 37 * int(seed)
            + (1 if side == "buy" else 2)
        )
        rng = np.random.default_rng(seed_value)
        sampled_returns = []
        for ticker, ticker_trades in run_trades.groupby("ticker"):
            candidates = market[market["ticker"] == ticker]
            if "test_start" in ticker_trades and ticker_trades["test_start"].notna().any():
                candidates = candidates[
                    candidates["date"] >= pd.Timestamp(ticker_trades["test_start"].dropna().iloc[0])
                ]
            if "test_end" in ticker_trades and ticker_trades["test_end"].notna().any():
                candidates = candidates[
                    candidates["date"] <= pd.Timestamp(ticker_trades["test_end"].dropna().iloc[0])
                ]
            candidates = candidates.dropna(subset=["forward_return_5"])
            if candidates.empty:
                continue
            selected = rng.choice(
                len(candidates), size=len(ticker_trades), replace=len(ticker_trades) > len(candidates))
            sampled_returns.extend(candidates.iloc[selected]["forward_return_5"].tolist())
        if sampled_returns:
            values = np.asarray(sampled_returns)
            random_hit_rates[run_key] = float(
                np.mean(values > 0) if side == "buy" else np.mean(values < 0))

    per_run = (
        merged.groupby(run_columns, as_index=False, dropna=False)
        .agg(
            trades_per_run=("executed_change", "size"),
            days_between_same_side_trades=("days_between_same_side_trades", "mean"),
            range_position_20_mean=("range_position_20", "mean"),
            forward_return_5_mean=("forward_return_5", "mean"),
            forward_return_20_mean=("forward_return_20", "mean"),
            favourable_excursion_5_mean=("favourable_excursion_5", "mean"),
            adverse_excursion_5_mean=("adverse_excursion_5", "mean"),
            directional_hit_rate_5=("directional_hit_5", "mean"),
            directional_hit_rate_20=("directional_hit_20", "mean"))
    )
    per_run["matched_random_hit_rate_5"] = [
        random_hit_rates.get(tuple(row), np.nan)
        for row in per_run[run_columns].itertuples(index=False, name=None)
    ]
    per_run["directional_hit_lift_5"] = (
        per_run["directional_hit_rate_5"] - per_run["matched_random_hit_rate_5"]
    )
    return (
        per_run.groupby(["baseline", "side"], as_index=False)
        .agg(
            runs=("seed", "size"),
            trades_per_run_mean=("trades_per_run", "mean"),
            days_between_same_side_trades_mean=("days_between_same_side_trades", "mean"),
            range_position_20_mean=("range_position_20_mean", "mean"),
            forward_return_5_mean=("forward_return_5_mean", "mean"),
            forward_return_20_mean=("forward_return_20_mean", "mean"),
            favourable_excursion_5_mean=("favourable_excursion_5_mean", "mean"),
            adverse_excursion_5_mean=("adverse_excursion_5_mean", "mean"),
            directional_hit_rate_5=("directional_hit_rate_5", "mean"),
            directional_hit_rate_20=("directional_hit_rate_20", "mean"),
            matched_random_hit_rate_5=("matched_random_hit_rate_5", "mean"),
            directional_hit_lift_5=("directional_hit_lift_5", "mean"))
        .loc[:, columns]
    )


def cost_sensitivity(run_dir: Path, positions: pd.DataFrame) -> pd.DataFrame:
    prices = pd.read_csv(run_dir / "data" / "prices.csv", parse_dates=["date"])
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    execution = manifest["config"]["execution"]
    initial_capital = execution["initial_capital"]
    borrow_bps = execution.get("short_borrow_bps_annual", 0.0)
    rebalance_threshold = execution.get("rebalance_threshold", 0.0)
    decision_interval = execution.get("decision_interval", 1)
    rows = []
    keys = ["baseline", "fold", "fold_kind", "repetition", "seed"]
    for key, group in positions.groupby(keys, dropna=False):
        metadata = dict(zip(keys, key, strict=True))
        matrix = group.pivot(index="date", columns="ticker", values="position")
        close_window = closes.reindex(index=matrix.index, columns=matrix.columns)
        for total_cost in (0, 5, 10, 20, 40):
            if metadata["baseline"] == "buy_and_hold":
                result = run_buy_and_hold(
                    close_window,
                    initial_capital=initial_capital,
                    transaction_cost_bps=total_cost,
                    slippage_bps=0)
            elif metadata["baseline"] == "equal_weight_rebalanced":
                result = run_equal_weight_rebalanced(
                    close_window,
                    initial_capital=initial_capital,
                    transaction_cost_bps=total_cost,
                    slippage_bps=0)
            else:
                result = run_portfolio(
                    close_window,
                    matrix,
                    initial_capital=initial_capital,
                    transaction_cost_bps=total_cost,
                    slippage_bps=0,
                    short_borrow_bps_annual=borrow_bps,
                    rebalance_threshold=rebalance_threshold,
                    decision_interval=decision_interval)
            rows.append({**metadata, "cost_bps": total_cost, "sharpe": result.metrics["sharpe"]})
    sensitivity = pd.DataFrame(rows)
    return sensitivity


def plot_cost_sensitivity(sensitivity: pd.DataFrame, path: Path, baselines: list[str],
                          family_title: str, scope_label: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for baseline in baselines:
        group = sensitivity[sensitivity["baseline"] == baseline]
        curve = group.groupby("cost_bps")["sharpe"].mean()
        axis.plot(
            curve.index,
            curve.values,
            label=display_label(baseline),
            **line_style(baseline, len(curve)))
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_title(
        f"{family_title}: Sharpe ratio under common trading-cost scenarios\n{scope_label}")
    axis.set_xlabel("Transaction cost and slippage (bps per unit turnover)")
    axis.set_ylabel("Mean out-of-sample Sharpe ratio")
    style_axis(axis)
    axis.legend(fontsize=8, ncol=2)
    save(fig, path)


def cost_sensitivity_scope(sensitivity: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Use one explicit evaluation scope and avoid treating seeds as folds."""
    if "fold_kind" not in sensitivity:
        return sensitivity.copy(), "all available evaluation rows"
    if (sensitivity["fold_kind"] == "final_holdout").any():
        return (
            sensitivity[sensitivity["fold_kind"] == "final_holdout"].copy(),
            "final holdout (mean across training seeds)",
        )
    walk = sensitivity[sensitivity["fold_kind"] == "walk_forward"].copy()
    if walk.empty:
        return sensitivity.copy(), "all available evaluation rows"
    keys = ["baseline", "fold", "fold_kind", "cost_bps"]
    return (
        walk.groupby(keys, as_index=False)["sharpe"].mean(),
        "walk-forward folds (seed mean per fold)"
    )


def plot_family_cost_sensitivity(sensitivity: pd.DataFrame, report_dir: Path) -> list[Path]:
    scoped, scope_label = cost_sensitivity_scope(sensitivity)
    paths = []
    for family in BASELINE_FAMILIES:
        baselines = family_selection(scoped["baseline"], family)
        if not baselines:
            continue
        path = report_dir / f"cost_sensitivity_{family}.png"
        plot_cost_sensitivity(
            scoped, path, baselines, FAMILY_TITLES[family], scope_label)
        paths.append(path)
    return paths


def performance_table(report: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "total_return_mean": "cumulative_return",
        "annual_return_mean": "annual_return",
        "annual_volatility_mean": "annual_volatility",
        "sharpe_mean": "sharpe_ratio",
        "sortino_mean": "sortino_ratio",
        "max_drawdown_mean": "max_drawdown",
        "cost_drag_mean": "cost_drag"
    }
    available = ["baseline", *[column for column in columns if column in report]]
    return report[available].rename(columns=columns)


def markdown_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include="number"):
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(map(str, row)) + " |"
        for row in formatted.itertuples(index=False, name=None))
    return f"{header}\n{separator}\n{body}"


def build_report(run_dir: str | Path, confidence: float = 0.95,
                 formats: tuple[str, ...] = ("png", "csv", "md")) -> None:
    run_dir = Path(run_dir)
    report_dir = run_dir / "report"
    report_dir.mkdir(exist_ok=True)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    equity = pd.read_csv(run_dir / "equity.csv", parse_dates=["date"])
    positions = pd.read_csv(run_dir / "positions.csv", parse_dates=["date"])
    prices = pd.read_csv(run_dir / "data" / "prices.csv", parse_dates=["date"])
    daily_path = run_dir / "daily_returns.csv"
    daily = pd.read_csv(daily_path, parse_dates=["date"]) if daily_path.exists() else pd.DataFrame()
    trades_path = run_dir / "trades.csv"
    trades = (
        pd.read_csv(trades_path, parse_dates=["date"])
        if trades_path.exists()
        else pd.DataFrame()
    )
    diagnostics_path = run_dir / "training_diagnostics.csv"
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    predictions_path = run_dir / "predictions.csv"
    predictions = (
        pd.read_csv(predictions_path, parse_dates=["date"])
        if predictions_path.exists()
        else pd.DataFrame()
    )

    report = summary(metrics, confidence)
    comparison = performance_table(report)
    sharpe_table = sharpe_over_time_table(metrics)
    sensitivity_path = report_dir / "cost_sensitivity.csv"
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)
    else:
        sensitivity = cost_sensitivity(
            run_dir,
            positions)
    diagnostic_summary = training_summary(diagnostics)
    timing_summary = trade_timing_summary(trades, prices)
    _, sensitivity_scope_label = cost_sensitivity_scope(sensitivity)

    if "csv" in formats:
        report.to_csv(report_dir / "summary.csv", index=False)
        comparison.to_csv(report_dir / "performance_comparison.csv", index=False)
        sharpe_table.to_csv(report_dir / "sharpe_over_time.csv", index=False)
        sensitivity.to_csv(report_dir / "cost_sensitivity.csv", index=False)
        if not diagnostic_summary.empty:
            diagnostic_summary.to_csv(report_dir / "training_summary.csv", index=False)
        if not timing_summary.empty:
            timing_summary.to_csv(report_dir / "trade_timing_summary.csv", index=False)
    if "md" in formats:
        (report_dir / "performance_comparison.md").write_text(
            markdown_table(comparison), encoding="utf-8")
        (report_dir / "sharpe_over_time.md").write_text(
            markdown_table(sharpe_table), encoding="utf-8")
        if not timing_summary.empty:
            (report_dir / "trade_timing_summary.md").write_text(
                markdown_table(timing_summary), encoding="utf-8")

    crash_year = None
    if "png" in formats:
        plot_sharpe_ranking(report, report_dir / "sharpe_ranking.png")
        plot_rl_seed_sharpe_distribution(
            metrics, report_dir / "rl_seed_sharpe_distribution.png")
        plot_return_drawdown(report, report_dir / "return_vs_drawdown.png")
        plot_family_cumulative_returns(equity, report_dir)
        plot_turnover(report, report_dir / "turnover.png")
        plot_fold_stability(metrics, report_dir / "fold_stability.png")
        plot_split_timeline(metrics, report_dir / "data_split_timeline.png")
        plot_family_sharpe_over_time(sharpe_table, report_dir)
        plot_family_cost_sensitivity(sensitivity, report_dir)
        crash_year = plot_crash_period(daily, report_dir)
        plot_prediction_vs_actual(predictions, report_dir)
        plot_trade_action_curves(
            trades,
            prices,
            metrics,
            report_dir)
        if not diagnostics.empty:
            plot_training_diagnostic(
                diagnostics,
                "training_reward",
                report_dir / "training_reward.png",
                "Mean training reward")
            plot_training_diagnostic(
                diagnostics,
                "loss",
                report_dir / "training_loss.png",
                "Mean optimisation loss")

    scope = report["scope"].iloc[0].replace("_", " ") if len(report) else "unavailable"
    crash_note = (
        f"The optional crash-period figure uses {crash_year}, the worst available "
        "buy-and-hold year."
        if crash_year is not None
        else "The crash-period figure was skipped because daily return artifacts were unavailable."
    )
    narrative = (
        "# Experiment summary\n\n"
        f"Headline scope: {scope}. Confidence level: {confidence:.0%}. "
        "All equity and cumulative-return figures are net of configured transaction costs, "
        "slippage, and short-borrow costs. Sharpe and Sortino ratios assume a zero risk-free "
        "rate. Final-holdout intervals describe variation across stochastic training seeds only; "
        "they are left blank for deterministic methods and do not measure uncertainty across "
        "market regimes.\n\n"
        "## Performance comparison\n\n"
        f"{markdown_table(comparison)}\n\n"
        "## Reporting design\n\n"
        "Each analytical view is saved as a separate figure so it can be placed, captioned, and "
        "scaled independently in the dissertation. The split timeline documents the experimental "
        "protocol; Sharpe ranking and fold paths address level and stability; net cumulative "
        "returns show economic magnitude, with 10th–90th percentile bands describing stochastic "
        "training-seed dispersion. The separate RL boxplot shows the raw seed-level Sharpe "
        "distribution on the same market path; it does not quantify market-regime uncertainty. "
        "Drawdown, turnover, and cost sensitivity cover risk and implementability. Cost-sensitivity "
        f"figures use {sensitivity_scope_label}. Crowded line comparisons are separated by strategy "
        "family, with "
        "the passive baselines repeated as common reference curves. Prediction figures use "
        "out-of-sample forecasts in return units. Stochastic trade-action figures use the observed "
        "seed nearest the baseline's median Sharpe, rather than an average policy. They overlay only "
        "actual executed stock-level changes from the backtest on the corresponding stock-price "
        "curve; green upward triangles denote buy/increase decisions and "
        "red downward triangles denote sell/reduce decisions.\n\n"
        f"{crash_note}\n"
    )
    if "md" in formats:
        (report_dir / "summary.md").write_text(narrative, encoding="utf-8")
