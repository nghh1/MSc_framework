# Model and baseline rationale

This document records the modelling choices that must be fixed before results are inspected. It
is intended to support the dissertation methodology chapter and to prevent post-hoc architectural
changes in response to test performance.

## Shared prediction and execution contract

Every supervised model forecasts the next trading-day return for one stock. Forecasts are mapped
continuously to `[-1, 1]` positions using a training-target volatility scale. Every RL policy selects
one of `[-1, -0.5, 0, 0.5, 1]`. All target positions then enter the same equal-weight portfolio
engine, which delays execution by one bar and applies turnover-dependent transaction costs and
slippage. This common layer is more important for fairness than making every model equally large.

The 20-day lookback represents roughly one trading month. Hidden dimensions are deliberately
small and tuned only on inner folds because daily equity data provide thousands, not millions, of
independent observations. Larger networks would add variance and compute without a defensible
sample-size basis.

## LSTM

The LSTM uses one or more recurrent layers, the final hidden state, and a two-layer point-forecast
head. LSTM gates provide a direct baseline for persistent and decaying temporal relationships such
as volatility clustering. The final state is appropriate because the target is one-step-ahead rather
than a sequence. Dropout is active between recurrent layers only when more than one layer is used;
the hidden width, dropout, learning rate, and epochs are selected on inner folds.

## Temporal Convolutional Network

The TCN has four residual causal convolution blocks with kernel size 3 and dilations 1, 2, 4, and
8. Its receptive field is 31 observations, so it covers the configured 20-day lookback. Left-only
padding makes every intermediate state causal. Residual paths improve optimisation, while dropout
regularises a network that otherwise can fit short-lived patterns. The TCN contrasts recurrent
memory with a parallel, fixed receptive field under a similar hidden-width budget.

## Compact Temporal Fusion Transformer

This is explicitly a compact TFT-inspired forecaster, not a full reproduction of the original
multi-horizon TFT. A learned feature gate performs time-varying variable weighting, an LSTM encodes
local order, causal multi-head self-attention captures longer interactions, and a gated residual plus
layer normalisation stabilises the representation. A single linear head is used because the task is
one-step point forecasting and the dataset has no known-future covariates. Calling it `compact TFT`
in the dissertation avoids overstating equivalence to a full quantile, multi-horizon TFT.

## Statistical and machine-learning baselines

- Static ARIMAX is the interpretable linear time-series baseline. It tests whether autoregressive
  errors plus exogenous technical features are sufficient without online parameter adaptation.
- Rolling ARIMAX refits on a capped recent window and causally incorporates only returns known by
  the decision date. It tests adaptation to parameter drift.
- Random Forest captures nonlinear feature interactions without sequence-state assumptions and is
  robust on modest tabular samples. Depth and leaf-size constraints control overfitting.
- Equal-weight buy-and-hold is the passive investable reference under the same dates, capital, and
  execution convention.

## Non-GARL RL baselines

The state for each stock is the standardised 20-day feature window plus its current position. The
position is included because transaction costs make the problem state-dependent. Discrete symmetric
actions allow long, flat, and short exposure without giving one RL algorithm a different action
space.

The joint (`single_*`) policy concatenates all stock states, shares a two-layer representation, and
uses one action head per stock. It can learn cross-stock relationships but its parameterisation grows
with the universe. Independent policies train one network per stock. They cannot transfer knowledge,
but avoid negative transfer and provide the cleanest control for GARL.

- A2C is the direct on-policy actor-critic reference and the base learner used by GARL.
- PPO adds clipped policy updates and generalised advantage estimation, testing whether improved
  on-policy stability explains performance.
- DQN adds replay, epsilon-greedy exploration, and target networks, testing a value-based learner
  under the same discrete actions.

PPO and DQN are non-GARL baselines; they are not presented as extensions of Wu and Zeng's method.

## GARL and DDAL

GARL assigns one A2C agent to each stock-specific environment. Agents begin from the same parameter
state, first learn privately, and later share timestamped gradient pieces. DDAL weights selected
pieces by training maturity and task relevance; this implementation uses absolute training-return
correlation as a pre-declared stock-task relevance proxy. `independent_a2c` is therefore the direct
no-sharing ablation.

The code is a deterministic, single-process DDAL-style emulation. Gradient generation is scheduled
in epochs and exchange occurs every configured sharing interval; it is not a wall-clock asynchronous
distributed system. This distinction must be stated as a limitation. The implementation follows the
gradient-pool and weighted-average mechanism in Wu and Zeng's GARL paper:
<https://arxiv.org/abs/2202.05135>.

## Indicators: ADX and ROC

ADX(14) is included because it measures trend strength rather than direction and complements moving
average and MACD features. It is calculated causally with Wilder smoothing and scaled to `[0, 1]`.
ROC is mathematically the same as a percentage return over its horizon. The existing `ret_10` was
already ROC(10), so adding a second `roc_10` column would create exact duplication. The framework
therefore adds ROC(20), which represents a trading-month momentum horizon and is not identical to
the existing 1-, 5-, or 10-day returns.

Both additions increase the pre-registered feature set. They should be retained or removed before
the final full run, and an indicator ablation should be reported if the dissertation makes a claim
that they improve performance.
