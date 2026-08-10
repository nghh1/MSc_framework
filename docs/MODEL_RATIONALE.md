# Model and baseline rationale

This document records the modelling choices that must be fixed before results are inspected. It
is intended to support the dissertation methodology chapter and to prevent post-hoc architectural
changes in response to test performance.

## Shared prediction and execution contract

Every supervised model forecasts the next trading-day return for one stock. Forecasts are mapped
continuously to `[-1, 1]` positions using a training-target volatility scale. Deep sequence targets
are standardised using training-only moments and forecasts are converted back to return units before
position mapping. Every RL policy selects one of `[-1, 0, 1]`. Each stock controls one fixed
equal-capital sleeve. RL training and
backtesting use the same sleeve transition: the current exposure drifts with realised return, the
next target trades from that drifted exposure, and transaction, slippage, and short-borrow costs are
deducted before the sleeve returns are averaged. This exact shared contract is more important for
fairness than making every model equally large.

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
- Rolling ARIMAX causally assimilates every newly observed return into its state-space filter and
  periodically re-estimates parameters on a capped recent window. It therefore retains daily
  autoregressive state updates without paying the cost of a full parameter refit every day.
- Random Forest captures nonlinear feature interactions without sequence-state assumptions and is
  robust on modest tabular samples. Depth and leaf-size constraints control overfitting.
- Equal-weight buy-and-hold is the passive investable reference under the same dates, capital, and
  execution convention.

## Non-GARL RL baselines

The state for each stock is the standardised 20-day by 19-feature market window plus its current
position. The market window remains two-dimensional inside the policy and is encoded by four
residual causal TCN blocks with 32 channels, kernel size 3, and dilations 1, 2, 4, and 8. Their
31-observation receptive field covers the complete 20-day lookback. The current drifted position is
concatenated after temporal encoding because it is portfolio state rather than a market sequence.
It is included because transaction costs make the problem state-dependent.

Encoder dropout is fixed at zero. Random feature masking would make PPO's stored and recomputed
action likelihoods depend on different dropout masks and would therefore distort the clipped
likelihood ratio. Capacity is instead controlled through the 32-channel bottleneck and causal weight
sharing. Discrete symmetric actions allow long, flat, and short exposure without giving one RL
algorithm a different action space. The three-action design reduces exploration complexity for the
available sample size.

The joint (`single_*`) policy applies one weight-shared TCN encoder to every stock, concatenates the
nine compact stock representations, shares a two-layer portfolio representation, and uses one action
head per stock. This reduces the original immediate input compression from 3,429 flattened values to
297 encoded values. Independent policies train one complete TCN-policy network per stock. They cannot
transfer knowledge, but avoid negative transfer and provide the cleanest control for GARL.

- A2C is the direct on-policy actor-critic reference and the base learner used by GARL.
- PPO adds clipped policy updates and generalised advantage estimation, testing whether improved
  on-policy stability explains performance.
- DQN adds replay, epsilon-greedy exploration, and target networks. The joint version is a branching
  DQN with one Q-value head per stock and a shared representation; it is not an exhaustive joint
  action-value table over all `3^N` portfolio actions.

PPO and DQN are non-GARL baselines; they are not presented as extensions of Wu and Zeng's method.

## GARL and DDAL

GARL assigns one TCN-A2C agent to each stock-specific environment. Agents are independently
initialised with seeds `seed + stock_index`; independent A2C uses the identical encoder, policy,
value heads, construction, and seed contract, so corresponding agents start alike across methods
while agents within either method remain different. GARL agents first learn privately and later share
timestamped gradients across the complete encoder and actor-critic model. DDAL weights retrieved
pieces by training maturity and task relevance; absolute training-return correlation is the
pre-declared stock-task relevance proxy. `independent_a2c` is therefore the direct no-sharing
ablation.

The code reproduces Algorithm 1 with deterministic event-driven asynchrony. Every agent has an
independent simulated clock, local epoch, environment, model, optimiser, and FIFO knowledge queue.
After the private-learning threshold, every generated gradient is copied to every peer queue while
the originating agent continues learning locally. Each agent independently retrieves and removes
queued peer gradients on its own update schedule. On an integration epoch, its current local
gradient is combined with the retrieved experience/relevance-weighted peer gradients and exactly
one optimiser update is applied. This reproduces learning semantics but simulates
network/process timing in one process rather than claiming measured distributed-system speed. The
implementation follows the mechanism in Wu and Zeng's GARL paper:
<https://arxiv.org/abs/2202.05135>.

Sharing begins after 30% of each agent's local epochs and each agent independently consumes its
FIFO queue every two post-isolation local epochs. `garl_pool_size = 0` means that all currently queued pieces are
retrieved, matching the paper's experimental interpretation of `m` as the available pool size.

### Selective GARL extension

`selective_garl_ddal` preserves the original private-learning threshold, asynchronous queues,
A2C learner, architecture, seeds, costs, and training budget. It changes only the receiver-side
use of shared gradients. Task relevance is positive signed training-return correlation rather than
absolute correlation. A peer gradient is accepted only when its cosine alignment with the
receiver's current local gradient exceeds the predeclared threshold (zero by default). Accepted
pieces are weighted by training maturity, relevance, and alignment; the receiver's current local
gradient is always retained, and it performs a local update when no peer passes the gate.

The extension records candidate count, accepted count, acceptance rate, and mean accepted alignment
at every selective update. It is an application-specific negative-transfer safeguard, not part of
Wu and Zeng's original DDAL algorithm. Because it was designed after inspecting the first complete
holdout, its results on the existing periods are exploratory unless confirmed on a newly frozen
external dataset.

## RL tuning, stopping, and diagnostics

RL tuning fixes rollout length at 32 and evaluates a nine-point logarithmic learning-rate grid on
the latest embargoed pre-test validation segment. The same step budget applies to every RL method
and selected settings are reused for all ten evaluation seeds. The
TCN structure is fixed rather than added to the tuning search, limiting compute and avoiding another
layer of model-selection variance.

Every RL method completes 100 rollout epochs of 32 environment interactions, for a fixed
3,200-interaction budget per environment. Training-reward checkpoint selection is disabled because sequential rollout
rewards reflect changing historical regimes and are not a stationary validation criterion. The
final fixed-step parameters are evaluated. Reward, loss,
entropy or epsilon where applicable, checkpoint eligibility, asynchronous queue size, and sharing
events are saved for diagnosis. Loss magnitudes are algorithm-specific and must not be ranked across
A2C, PPO, and DQN.

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
