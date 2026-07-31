"""A2C-based DDAL implementation for Group-Agent Reinforcement Learning.

This module reconstructs the GARL mechanism proposed by Wu and Zeng in their University of
Manchester paper. PPO and DQN live only in the non-GARL RL baseline modules.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from garl_trading.rl.core import (
    ActorCritic,
    a2c_gradient,
    apply_gradient,
    fit_feature_scalers,
    make_states,
)
from garl_trading.rl.trainers import RLPolicySet


@dataclass(frozen=True)
class GradientPiece:
    gradients: list[torch.Tensor]
    source: str
    epoch: int
    relevance: float


def _weighted_average(pieces: list[GradientPiece]) -> list[torch.Tensor]:
    experience = np.asarray([piece.epoch + 1 for piece in pieces], dtype=float)
    relevance = np.asarray([piece.relevance for piece in pieces], dtype=float)
    weights = 0.5 * experience / experience.sum() + 0.5 * relevance / relevance.sum()
    return [
        sum(float(weight) * piece.gradients[i] for weight, piece in zip(weights, pieces))
        for i in range(len(pieces[0].gradients))
    ]


def train_garl_ddal(
    features: dict[str, pd.DataFrame],
    closes: dict[str, pd.Series],
    *,
    levels: tuple[float, ...],
    lookback: int,
    epochs: int,
    rollout_length: int,
    learning_rate: float,
    gamma: float,
    cost_rate: float,
    seed: int,
    device: str = "cpu",
    share_after_fraction: float = 0.3,
    share_every: int = 4,
    pool_size: int | None = None,
) -> RLPolicySet:
    """Synchronous DDAL simulation with aligned model initialization.

    Every agent starts from the same parameter state. Gradient pieces are merged by generation
    epoch, and each sharing update includes the receiver's own latest gradient.
    """
    tickers = tuple(features)
    scalers = fit_feature_scalers(features)
    states = make_states(
        features, closes, scalers, levels=levels, lookback=lookback, cost_rate=cost_rate
    )
    observation_size = features[tickers[0]].shape[1] * lookback + 1
    torch.manual_seed(seed)
    template = ActorCritic(observation_size, len(levels)).to(device)
    models = {ticker: deepcopy(template) for ticker in tickers}
    optimizers = {
        ticker: torch.optim.Adam(models[ticker].parameters(), lr=learning_rate)
        for ticker in tickers
    }
    randoms = {
        ticker: np.random.default_rng(seed + i)
        for i, ticker in enumerate(tickers)
    }
    threshold = int(epochs * share_after_fraction)
    pool_size = pool_size or len(tickers)
    pending: dict[str, list[GradientPiece]] = {ticker: [] for ticker in tickers}

    for epoch in range(epochs):
        pieces = {}
        for ticker in tickers:
            gradient, _ = a2c_gradient(
                models[ticker],
                states[ticker],
                rollout_length=rollout_length,
                gamma=gamma,
                rng=randoms[ticker],
            )
            pieces[ticker] = GradientPiece(gradient, ticker, epoch, 1.0)

        if epoch < threshold:
            for ticker in tickers:
                apply_gradient(models[ticker], optimizers[ticker], pieces[ticker].gradients)
            continue

        for receiver in tickers:
            pending[receiver].extend(pieces.values())
        if (epoch + 1) % share_every:
            for ticker in tickers:
                apply_gradient(models[ticker], optimizers[ticker], pieces[ticker].gradients)
            continue

        for ticker in tickers:
            candidates = sorted(
                pending[ticker],
                key=lambda piece: (piece.epoch, piece.source == ticker),
                reverse=True,
            )
            own = pieces[ticker]
            chosen = [own]
            chosen.extend(piece for piece in candidates if piece.source != ticker)
            chosen = chosen[:pool_size]
            apply_gradient(models[ticker], optimizers[ticker], _weighted_average(chosen))
            pending[ticker].clear()
    return RLPolicySet("garl", models, scalers, tickers, levels, lookback)
