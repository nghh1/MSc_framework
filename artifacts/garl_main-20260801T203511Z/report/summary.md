# Experiment summary

Headline scope: final holdout. Confidence level: 95%. All equity and cumulative-return figures are net of configured transaction costs and slippage.

## Performance comparison

| baseline | cumulative_return | annual_return | annual_volatility | sharpe_ratio | sortino_ratio | max_drawdown | cost_drag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| buy_and_hold | 2.3461 | 0.2653 | 0.2123 | 1.2500 | 1.8465 | -0.3091 | 0.0023 |
| random_forest | 0.0317 | 0.0063 | 0.0077 | 0.8171 | 1.3408 | -0.0111 | 0.0119 |
| tft | 0.0519 | 0.0104 | 0.0186 | 0.5552 | 0.8492 | -0.0230 | 0.0403 |
| arimax_static | 0.0502 | 0.0105 | 0.0360 | 0.2915 | 0.4346 | -0.0780 | 0.0571 |
| independent_a2c | 0.0663 | 0.0153 | 0.0770 | 0.1916 | 0.2799 | -0.1474 | 0.1917 |
| garl_ddal | 0.0600 | 0.0127 | 0.0851 | 0.1571 | 0.2478 | -0.1795 | 0.2004 |
| tcn | -0.0044 | -0.0007 | 0.0217 | -0.0305 | -0.0446 | -0.0434 | 0.0822 |
| independent_ppo | -0.0188 | -0.0070 | 0.0776 | -0.0942 | -0.1014 | -0.2058 | 0.1060 |
| lstm | -0.0233 | -0.0043 | 0.0307 | -0.1394 | -0.2075 | -0.0760 | 0.0785 |
| arimax_rolling | -0.0831 | -0.0166 | 0.0412 | -0.4028 | -0.5735 | -0.1344 | 0.0936 |
| single_ppo | -0.1572 | -0.0343 | 0.0723 | -0.4844 | -0.6475 | -0.2584 | 0.0879 |
| single_a2c | -0.1825 | -0.0422 | 0.0778 | -0.5497 | -0.7445 | -0.2673 | 0.1201 |
| independent_dqn | -0.2134 | -0.0479 | 0.0792 | -0.6160 | -0.8296 | -0.2903 | 0.3856 |
| single_dqn | -0.2165 | -0.0468 | 0.0697 | -0.6676 | -0.9021 | -0.2509 | 0.1786 |

## Reporting design

Each analytical view is saved as a separate figure so it can be placed, captioned, and scaled independently in the dissertation. The split timeline documents the experimental protocol; Sharpe ranking and fold paths address level and stability; net cumulative returns show economic magnitude; drawdown, turnover, and cost sensitivity cover risk and implementability.

The optional crash-period figure uses 2008, the worst available buy-and-hold year.
