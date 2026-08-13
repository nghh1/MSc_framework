# Model and baseline rationale

This document records the modelling choices that must be fixed before results are inspected. It
is intended to support the dissertation methodology chapter and to prevent post-hoc architectural
changes in response to test performance.

## Shared prediction and execution contract

Every supervised model forecasts the next trading-day return for one stock. Forecasts are mapped
continuously to `[-1, 1]` using the constrained single-asset mean-variance rule
`position = clip(predicted_return / (risk_aversion * training_return_variance), -1, 1)`.
The shared risk-aversion coefficient is fixed at 10 for every model and stock, while variance is
estimated only from that model's outer-training targets. This is the unconstrained optimum for a
quadratic expected-utility approximation followed by the experiment's no-leverage/no-short-beyond-one
constraint. It replaces an arbitrary nonlinear signal squashing constant without using test data.
Deep sequence targets
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
Sequence-model dropout is fixed at 0.2 rather than tuned. Their Adam learning-rate search is bounded
to $[3\times10^{-4}, 1.5\times10^{-3}]$, avoiding both near-static and excessively aggressive fits.

## LSTM

The LSTM uses two recurrent layers, the final hidden state, and a two-layer point-forecast
head. LSTM gates provide a direct baseline for persistent and decaying temporal relationships such
as volatility clustering. The final state is appropriate because the target is one-step-ahead rather
than a sequence. Dropout is fixed at 0.2 between the recurrent layers; the hidden width, learning
rate, and epochs are selected on inner folds.

## Temporal Convolutional Network

The TCN has four residual causal blocks with kernel size 3 and dilations 1, 2, 4, and 8. Each block
contains two repetitions of causal convolution, weight normalisation, ReLU, and fixed 0.2 dropout
before its residual addition. Its receptive field covers the configured 20-day lookback. The TCN
contrasts recurrent memory with a parallel, fixed receptive field under a similar hidden-width budget.

## Encoder-only Transformer

The conventional encoder-only Transformer uses a linear projection and learned positional encoding,
followed by two pre-normalised causal Transformer
encoder blocks with multi-head self-attention, GELU feed-forward layers, and fixed 0.2 dropout. Layer
normalisation and a linear head map the final token to the one-step return forecast. A decoder is not
required because the task produces one value rather than an autoregressive output sequence.

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

- A2C is the direct on-policy actor-critic reference and the base learner used by GARL. It uses
  normalised generalised advantages but retains one unclipped update per rollout.
- PPO adds clipped policy updates and generalised advantage estimation, testing whether improved
  on-policy stability explains performance.
- DQN adds replay, epsilon-greedy exploration, Double-DQN targets, Huber loss, and target networks.
  The joint version is a branching
  DQN with one Q-value head per stock and a shared representation; it is not an exhaustive joint
  action-value table over all `3^N` portfolio actions.

All RL rewards contain actual execution costs. Inner validation may select a one- or two-times
turnover penalty during training; two is regularisation only, while evaluation always deducts the
actual cost exactly once. PPO and DQN are non-GARL baselines and are not presented as GARL variants.

## GARL and DDAL

GARL assigns one TCN-A2C agent to each stock-specific environment. Separate stock models are copied
from one common parameter template so corresponding tensor coordinates retain a defensible shared
meaning when raw gradients are exchanged. Independent A2C uses the same initialisation contract,
while every agent retains its own environment, optimiser, random stream, and subsequent updates.
GARL agents first learn privately and later share
timestamped gradients across the complete encoder and actor-critic model. DDAL weights retrieved
pieces by training maturity and task relevance; absolute training-return correlation is the
pre-declared stock-task relevance proxy. `independent_a2c` is therefore the direct no-sharing
ablation.

The code reproduces the central learning semantics of Algorithm 1 with deterministic event-driven
asynchrony. Every agent has an
independent simulated clock, local epoch, environment, model, optimiser, and knowledge queue.
After the private-learning threshold, every generated gradient is copied to every peer queue while
the originating agent continues learning locally. Each agent independently retrieves and removes
queued peer gradients on its own update schedule. Only the newest queued gradient from each source
is eligible, and an inner-validated pool bound limits stale transfer. On an integration epoch, its
current local gradient is combined with the retrieved experience/relevance-weighted peer gradients and exactly
one optimiser update is applied. This reproduces learning semantics but simulates
network/process timing in one process rather than claiming measured distributed-system speed. The
original DDAL weighted average exactly implements Wu and Zeng's equation: one half of the
experience-normalised gradient average plus one half of the relevance-normalised gradient average.
The surrounding implementation is an explicit adaptation: it simulates asynchronous timing in one
process, clips the A2C gradient norm, uses cross-stock return correlation for relevance, and combines
the receiver's current local gradient with queued peer pieces instead of replaying a stored local
gradient pool. Its A2C estimator uses 32-step normalised generalised advantages, value loss, entropy regularisation,
and gradient clipping, whereas the paper presents the core one-step advantage expression. It therefore
follows the paper's mechanism without claiming a bit-for-bit reproduction:
<https://arxiv.org/abs/2202.05135>.

Sharing begins after 30% of each agent's local epochs and each agent independently consumes recent
peer knowledge every two post-isolation local epochs. At most one current gradient per source is
eligible, and a common fixed pool size of three bounds how many sources enter one update for both
GARL variants.

### Selective GARL extension

`selective_garl_ddal` preserves the private-learning threshold, asynchronous queues, A2C learner,
architecture, costs, and training budget. It changes only the receiver-side
use of shared gradients. Task relevance is positive signed training-return correlation rather than
absolute correlation. A peer gradient is accepted only when its cosine alignment with the
receiver's current local gradient exceeds an inner-validated threshold in `{0.0, 0.05, 0.1}`. A zero
threshold therefore means any strictly positive alignment. Accepted peers
retain DDAL's additive experience/relevance weighting, multiplied by their positive gradient alignment
and renormalised within the accepted peer pool. The final update is a convex blend with a
fixed 50% peer share. If no peer passes the gate, the update is fully local. This is
a predeclared safeguard rather than a claim of mathematically optimal transfer.

The extension records candidate count, accepted count, acceptance rate, and mean accepted alignment
at every selective update. It is an application-specific negative-transfer safeguard, not part of
Wu and Zeng's original DDAL algorithm. Because it was designed after inspecting the first complete
holdout, its results on the existing periods are exploratory unless confirmed on a newly frozen
external dataset.

## RL tuning, stopping, and diagnostics

RL tuning fixes rollout length at 32. Each method evaluates nine predeclared profiles using three
learning rates from half to twice the declared base rate on the latest embargoed pre-test validation
segment. Turnover penalisation remains fixed at the actual configured cost multiplier of 1.0, so
tuning does not optimise against artificial costs that differ from the final backtest. Selective
GARL uses a balanced three-learning-rate by three-alignment-threshold grid with thresholds
`{0.0, 0.05, 0.1}`; its entropy weight, peer mixture, and pool size remain fixed at 0.01, 0.5,
and 3. Original GARL uses a balanced three-learning-rate by three-entropy-weight grid while fixing
the same pool size of 3. The same
step budget applies to every RL method
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
