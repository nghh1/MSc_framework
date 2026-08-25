# Experiment summary

Headline scope: final holdout. Confidence level: 95%. All equity and cumulative-return figures are net of configured transaction costs, slippage, and short-borrow costs. Sharpe and Sortino ratios assume a zero risk-free rate. Final-holdout intervals describe variation across stochastic training seeds only; they are left blank for deterministic methods and do not measure uncertainty across market regimes.

## Performance comparison

| baseline | cumulative_return | annual_return | annual_volatility | sharpe_ratio | sortino_ratio | max_drawdown | cost_drag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | 0.2845 | 0.0848 | 0.0412 | 2.0592 | 3.3265 | -0.0511 | 0.0142 |
| equal_weight_rebalanced | 1.6616 | 0.3459 | 0.1880 | 1.8398 | 2.7720 | -0.2259 | 0.0159 |
| independent_a2c | 1.2795 | 0.2872 | 0.1572 | 1.8353 | 2.8000 | -0.1932 | 0.0015 |
| garl_ddal | 1.2495 | 0.2792 | 0.1524 | 1.8327 | 2.7954 | -0.1841 | 0.0017 |
| independent_ppo | 1.2423 | 0.2807 | 0.1552 | 1.8100 | 2.7480 | -0.1909 | 0.0013 |
| selective_garl_ddal | 1.1688 | 0.2653 | 0.1458 | 1.8095 | 2.7481 | -0.1793 | 0.0014 |
| buy_and_hold | 2.2806 | 0.4296 | 0.2503 | 1.7162 | 2.5998 | -0.2763 | 0.0023 |
| single_ppo | 0.4266 | 0.1135 | 0.1049 | 1.0433 | 1.6199 | -0.1558 | 0.0195 |
| lstm | 0.1742 | 0.0570 | 0.0803 | 0.7097 | 1.1974 | -0.0753 | 0.0440 |
| single_a2c | 0.3432 | 0.0835 | 0.1011 | 0.7037 | 1.1331 | -0.1707 | 0.0148 |
| independent_dqn | 0.1933 | 0.0615 | 0.0934 | 0.6515 | 0.9783 | -0.1382 | 0.0590 |
| tcn | 0.0895 | 0.0317 | 0.0769 | 0.4119 | 0.6783 | -0.0754 | 0.0605 |
| transformer | 0.0059 | 0.0058 | 0.0885 | 0.0658 | 0.1106 | -0.1120 | 0.0509 |
| single_dqn | 0.0084 | 0.0036 | 0.0885 | 0.0550 | 0.1069 | -0.1790 | 0.0455 |
| arimax_static | -0.0144 | 0.0021 | 0.1188 | 0.0173 | 0.0269 | -0.1121 | 0.0388 |
| arimax_rolling | -0.0205 | -0.0043 | 0.0727 | -0.0590 | -0.0798 | -0.2068 | 0.0651 |

## Reporting design

Each analytical view is saved as a separate figure so it can be placed, captioned, and scaled independently in the dissertation. The split timeline documents the experimental protocol; Sharpe ranking and fold paths address level and stability; net cumulative returns show economic magnitude, with 10th–90th percentile bands describing stochastic training-seed dispersion. The separate RL boxplot shows the raw seed-level Sharpe distribution on the same market path; it does not quantify market-regime uncertainty. Drawdown, turnover, and cost sensitivity cover risk and implementability. Cost-sensitivity figures use final holdout (mean across training seeds). Crowded line comparisons are separated by strategy family, with the passive baselines repeated as common reference curves. Prediction figures use out-of-sample forecasts in return units. Stochastic trade-action figures use the observed seed nearest the baseline's median Sharpe, rather than an average policy. They overlay only actual executed stock-level changes from the backtest on the corresponding stock-price curve; green upward triangles denote buy/increase decisions and red downward triangles denote sell/reduce decisions.

The optional crash-period figure uses 2008, the worst available buy-and-hold year.
