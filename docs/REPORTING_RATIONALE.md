# Dissertation reporting design

The previous four-panel overview was convenient for quick inspection but unsuitable for a
dissertation: panels became small, shared legends were crowded, and each result could not be sized
or captioned independently. The report generator now writes exactly one analytical graph per image.

## Priority figures

1. `data_split_timeline.png` documents train, embargo, walk-forward test, and final-holdout timing.
2. `cumulative_returns_net.png` shows economic performance on the final holdout (or latest outer
   fold) after configured transaction costs and slippage. Dates use concise abbreviated labels.
3. `sharpe_ranking.png` shows the principal risk-adjusted ranking and uncertainty.
4. `sharpe_over_time.png` shows whether rankings persist across successive out-of-sample folds.
5. `fold_stability.png` gives the same fold evidence as a baseline-by-period heatmap.
6. `return_vs_drawdown.png` separates return magnitude from downside risk.
7. `turnover.png` shows implementability and explains sensitivity to trading costs.
8. `cost_sensitivity.png` replays saved positions at common cost levels.
9. `crash_period_cumulative_returns.png` is generated when daily returns exist. Its year is selected
   mechanically as the worst calendar year for buy-and-hold in the available evaluation sample.

The first three answer protocol, economic magnitude, and headline comparison. The remaining views
test stability, downside, and implementation robustness. This is why these graphs are included;
additional charts should answer a distinct research question rather than repeat the same ranking.

## Tables

- `performance_comparison.csv` and `.md` contain cumulative return, annual return, annual
  volatility, Sharpe, Sortino, maximum drawdown, and cost drag.
- `sharpe_over_time.csv` and `.md` contain mean Sharpe by out-of-sample period and baseline.
- `summary.csv` retains means, confidence intervals, observation counts, and analysis scope.
- `cost_sensitivity.csv` contains every replayed cost scenario.

Headline aggregation uses the untouched final holdout when present. Without it, repeated seeds are
first averaged within a fold and uncertainty is computed across folds, avoiding pseudo-replication
of highly related seed runs.

## Metric set

The core set remains total return, CAGR, annual volatility, Sharpe, Sortino, maximum drawdown,
Calmar, daily turnover, and gross/net exposure. Reporting also receives annual arithmetic return,
annual downside deviation, annual turnover, positive-day rate, profit factor, 95% historical VaR
and CVaR, skewness, excess kurtosis, ulcer index, gross return, total cost, annual cost, and cost drag.

No single metric should be treated as conclusive. Sharpe assumes a stable return distribution;
drawdown and ulcer index describe path risk; CVaR describes tail loss; turnover and cost drag show
whether paper performance survives implementation.
