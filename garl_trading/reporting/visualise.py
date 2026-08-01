from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
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
    "garl_ddal": "#D1495B"
}


def colour(name: str) -> str:
    return COLOURS.get(name, "#52616B")


def summary(metrics: pd.DataFrame, confidence: float) -> pd.DataFrame:
    numeric = [
        "total_return", "cagr", "annual_volatility", "sharpe", "sortino",
        "max_drawdown", "turnover_daily", "gross_exposure"
    ]
    if "fold_kind" in metrics and (metrics["fold_kind"] == "final_holdout").any():
        analysis = metrics[metrics["fold_kind"] == "final_holdout"].copy()
        scope = "final_holdout"
    elif {"fold", "repetition"}.issubset(metrics.columns):
        analysis = metrics.groupby(["baseline", "fold"], as_index=False)[numeric].mean()
        scope = "walk_forward_fold_mean"
    else:
        analysis = metrics.copy()
        scope = "all_rows"

    rows = []
    for baseline, group in analysis.groupby("baseline", sort=False):
        row = {"baseline": baseline, "observations": len(group)}
        for column in numeric:
            values = group[column].dropna()
            mean = values.mean()
            sem = values.sem() if len(values) > 1 else 0.0
            critical = stats.t.ppf((1 + confidence) / 2, len(values) - 1) if len(values) > 1 else 0
            row[f"{column}_mean"] = mean
            row[f"{column}_ci"] = sem * critical
        row["scope"] = scope
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sharpe_mean", ascending=False)


def overview(metrics: pd.DataFrame, equity: pd.DataFrame, summary: pd.DataFrame, path: Path):
    baselines = summary["baseline"].tolist()
    colours = [colour(name) for name in baselines]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("GARL benchmark: performance, risk, and implementation burden", fontsize=16)

    y = np.arange(len(baselines))
    axes[0, 0].barh(y, summary["sharpe_mean"], xerr=summary["sharpe_ci"], 
                    color=colours, alpha=0.9, capsize=3)
    axes[0, 0].set_yticks(y, baselines)
    axes[0, 0].invert_yaxis()
    axes[0, 0].axvline(0, color="#333333", linewidth=0.8)
    axes[0, 0].set_title("Out-of-sample Sharpe with confidence interval")

    for _, row in summary.iterrows():
        axes[0, 1].scatter(
            abs(row["max_drawdown_mean"]),
            row["cagr_mean"],
            s=70 + 800 * max(row["turnover_daily_mean"], 0),
            color=colour(row["baseline"]),
            label=row["baseline"],
            alpha=0.85,
            edgecolor="white"
        )
    axes[0, 1].set_xlabel("Absolute maximum drawdown")
    axes[0, 1].set_ylabel("CAGR")
    axes[0, 1].set_title("Return versus drawdown; bubble size = turnover")

    chosen_kind = "final_holdout" if (metrics["fold_kind"] == "final_holdout").any() else "walk_forward"
    chosen_fold = metrics.loc[metrics["fold_kind"] == chosen_kind, "fold"].max()
    subset = equity[(equity["fold_kind"] == chosen_kind) & (equity["fold"] == chosen_fold) & (equity["repetition"] == 0)]
    for baseline, group in subset.groupby("baseline"):
        group = group.sort_values("date")
        axes[1, 0].plot(pd.to_datetime(group["date"]), group["equity"] / group["equity"].iloc[0],
            label=baseline, color=colour(baseline), linewidth=1.6)
    axes[1, 0].set_title(f"Normalized equity — {chosen_kind.replace('_', ' ')}")
    axes[1, 0].set_ylabel("Growth of 1.0")
    axes[1, 0].legend(fontsize=7, ncol=2)

    axes[1, 1].barh(y, summary["turnover_daily_mean"], color=colours, alpha=0.9)
    axes[1, 1].set_yticks(y, baselines)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("Average daily turnover")
    axes[1, 1].set_xlabel("Fraction of portfolio traded")

    for axis in axes.flat:
        axis.grid(alpha=0.18)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fold_stability(metrics: pd.DataFrame, path: Path):
    walk = metrics[metrics["fold_kind"] == "walk_forward"]
    pivot = walk.pivot_table(index="baseline", columns="fold", values="sharpe", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(pivot))), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(np.arange(len(pivot.columns)), [f"Fold {value}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            ax.text(j, i, "—" if np.isnan(value) else f"{value:.2f}", ha="center", va="center")
    ax.set_title("Regime stability: mean Sharpe by walk-forward fold")
    fig.colorbar(image, ax=ax, label="Sharpe")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def cost_sensitivity(run_dir: Path, positions: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    prices = pd.read_csv(run_dir / "data" / "prices.csv", parse_dates=["date"])
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    initial_capital = manifest["config"]["execution"]["initial_capital"]
    scenarios = [0, 5, 10, 20, 40]
    rows = []
    keys = ["baseline", "fold", "fold_kind", "repetition", "seed"]
    for key, group in positions.groupby(keys, dropna=False):
        meta = dict(zip(keys, key))
        matrix = group.pivot(index="date", columns="ticker", values="position")
        matrix.index = pd.to_datetime(matrix.index)
        close_window = closes.reindex(index=matrix.index, columns=matrix.columns)
        for total_cost in scenarios:
            result = run_portfolio(close_window, matrix, initial_capital=initial_capital, 
                                   transaction_cost_bps=total_cost, slippage_bps=0)
            rows.append({**meta, "cost_bps": total_cost, "sharpe": result.metrics["sharpe"]})
    sensitivity = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for baseline, group in sensitivity.groupby("baseline"):
        curve = group.groupby("cost_bps")["sharpe"].mean()
        ax.plot(curve.index, curve.values, marker="o", label=baseline, color=colour(baseline))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Implementation robustness: Sharpe under trading-cost scenarios")
    ax.set_xlabel("Total transaction cost and slippage (bps per unit turnover)")
    ax.set_ylabel("Mean out-of-sample Sharpe")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    sensitivity.to_csv(run_dir / "report" / "cost_sensitivity.csv", index=False)


def markdown(summary: pd.DataFrame, metrics: pd.DataFrame, confidence: float) -> str:
    display = summary[["baseline", "observations", "sharpe_mean", "sharpe_ci", 
                       "cagr_mean", "max_drawdown_mean", "turnover_daily_mean"]].copy()
    display["sharpe"] = display.apply(lambda row: f"{row.sharpe_mean:.3f} ± {row.sharpe_ci:.3f}", axis=1)
    display["cagr"] = display["cagr_mean"].map(lambda value: f"{value:.2%}")
    display["max_drawdown"] = display["max_drawdown_mean"].map(lambda value: f"{value:.2%}")
    display["turnover"] = display["turnover_daily_mean"].map(lambda value: f"{value:.3f}")
    display = display[["baseline", "observations", "sharpe", "cagr", "max_drawdown", "turnover"]]
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = "\n".join("| " + " | ".join(str(value) for value in row) + " |"
                     for row in display.itertuples(index=False, name=None))
    table = f"{header}\n{separator}\n{body}"
    failures = int(metrics["sharpe"].isna().sum())
    scope = summary["scope"].iloc[0].replace("_", " ") if len(summary) else "unavailable"
    return (
        "# Experiment summary\n\n"
        f"Confidence level: {confidence:.0%}. Successful result rows: {len(metrics)}. "
        f"Rows with missing Sharpe: {failures}. Headline scope: {scope}.\n\n"
        "All headline metrics are computed from portfolio equity curves under the same execution "
        "assumptions. Confidence intervals summarise variation across folds and stochastic seeds.\n\n"
        + table
        + "\n\n"
        "## Reading the figures\n\n"
        "- `overview.png` combines performance, drawdown, turnover, and the final evaluation equity curves.\n"
        "- `fold_stability.png` shows whether a result survives different market regimes.\n"
        "- `cost_sensitivity.png` recomputes every saved position path at common cost assumptions.\n"
    )


def build_report(run_dir: str | Path, confidence: float = 0.95) -> None:
    run_dir = Path(run_dir)
    report_dir = run_dir / "report"
    report_dir.mkdir(exist_ok=True)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    equity = pd.read_csv(run_dir / "equity.csv", parse_dates=["date"])
    positions = pd.read_csv(run_dir / "positions.csv", parse_dates=["date"])
    summary = summary(metrics, confidence)
    summary.to_csv(report_dir / "summary.csv", index=False)
    overview(metrics, equity, summary, report_dir / "overview.png")
    fold_stability(metrics, report_dir / "fold_stability.png")
    cost_sensitivity(run_dir, positions, metrics, report_dir / "cost_sensitivity.png")
    (report_dir / "summary.md").write_text(markdown(summary, metrics, confidence), encoding="utf-8")
