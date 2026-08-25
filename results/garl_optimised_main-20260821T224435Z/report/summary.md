# Experiment summary

Headline scope: final holdout. Confidence level: 95%. All equity and cumulative-return figures are net of configured transaction costs, slippage, and short-borrow costs. Sharpe and Sortino ratios assume a zero risk-free rate. Final-holdout intervals describe variation across stochastic training seeds only; they are left blank for deterministic methods and do not measure uncertainty across market regimes.

## Performance comparison

| baseline | cumulative_return | annual_return | annual_volatility | sharpe_ratio | sortino_ratio | max_drawdown | cost_drag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| equal_weight_rebalanced | 1.6616 | 0.3459 | 0.1880 | 1.8398 | 2.7720 | -0.2259 | 0.0159 |
| buy_and_hold | 2.2806 | 0.4296 | 0.2503 | 1.7162 | 2.5998 | -0.2763 | 0.0023 |
| independent_ppo | 0.7961 | 0.1979 | 0.1216 | 1.6031 | 2.4588 | -0.1463 | 0.0410 |
| garl_ddal | 0.7017 | 0.1742 | 0.1155 | 1.4076 | 2.1000 | -0.1497 | 0.0433 |
| random_forest | 0.1621 | 0.0513 | 0.0438 | 1.1724 | 1.6694 | -0.0669 | 0.0652 |
| lstm | 0.1513 | 0.0494 | 0.0659 | 0.7498 | 1.1368 | -0.1047 | 0.0478 |
| independent_a2c | 0.3553 | 0.0737 | 0.1031 | 0.4925 | 0.8269 | -0.1915 | 0.0613 |
| selective_garl_ddal | 0.3807 | 0.0766 | 0.1074 | 0.3725 | 0.6945 | -0.2038 | 0.0648 |
| single_ppo | 0.1353 | 0.0276 | 0.0797 | 0.2676 | 0.5322 | -0.1682 | 0.0154 |
| transformer | 0.0190 | 0.0086 | 0.0677 | 0.1269 | 0.1888 | -0.1196 | 0.1869 |
| single_a2c | 0.0286 | -0.0037 | 0.0739 | -0.0152 | 0.1010 | -0.1756 | 0.0216 |
| arimax_static | -0.0965 | -0.0293 | 0.0978 | -0.2992 | -0.4531 | -0.1596 | 0.1620 |
| tcn | -0.1023 | -0.0337 | 0.0704 | -0.4779 | -0.6383 | -0.1970 | 0.2255 |
| independent_dqn | -0.1590 | -0.0569 | 0.0813 | -0.7173 | -0.9661 | -0.2263 | 0.2861 |
| single_dqn | -0.2825 | -0.1100 | 0.0882 | -1.2367 | -1.6602 | -0.3162 | 0.2225 |
| arimax_rolling | -0.2702 | -0.1021 | 0.0824 | -1.2388 | -1.5550 | -0.2837 | 0.2012 |

## Reporting design

Each analytical view is saved as a separate figure so it can be placed, captioned, and scaled independently in the dissertation. The split timeline documents the experimental protocol; Sharpe ranking and fold paths address level and stability; net cumulative returns show economic magnitude, with 10th–90th percentile bands describing stochastic training-seed dispersion. The separate RL boxplot shows the raw seed-level Sharpe distribution on the same market path; it does not quantify market-regime uncertainty. Drawdown, turnover, and cost sensitivity cover risk and implementability. Cost-sensitivity figures use final holdout (mean across training seeds). Crowded line comparisons are separated by strategy family, with the passive baselines repeated as common reference curves. Prediction figures use out-of-sample forecasts in return units. Stochastic trade-action figures use the observed seed nearest the baseline's median Sharpe, rather than an average policy. They overlay only target-position changes larger than the 0.10 no-trade threshold on the corresponding stock-price curve; green upward triangles denote buy/increase decisions and red downward triangles denote sell/reduce decisions.

The optional crash-period figure uses 2008, the worst available buy-and-hold year.
