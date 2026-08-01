"""
A2C-based DDAL implementation for Group-Agent Reinforcement Learning.
This module reconstructs the GARL mechanism proposed by Wu and Zeng.
PPO and DQN algorithm live only in the non-GARL RL baseline modules.
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
from garl_trading.utils import resolve_torch_device


@dataclass(frozen=True)
class GradientPiece:
    gradients: list[torch.Tensor]
    source: str
    epoch: int
    relevance: float


def weighted_average(pieces: list[GradientPiece]) -> list[torch.Tensor]:
    experience = np.asarray([piece.epoch + 1 for piece in pieces], dtype=float)
    relevance = np.asarray([piece.relevance for piece in pieces], dtype=float)
    weights = 0.5 * experience / experience.sum() + 0.5 * relevance / relevance.sum()
    return [
        sum(float(weight) * piece.gradients[i] for weight, piece in zip(weights, pieces))
        for i in range(len(pieces[0].gradients))
    ]


def return_relevance(closes: dict[str, pd.Series]) -> dict[tuple[str, str], float]:
    """Map source/receiver similarity to DDAL relevance weights using return correlation."""
    returns = pd.DataFrame(closes).pct_change(fill_method=None)
    correlations = returns.corr().abs().fillna(0.0)
    return {
        (receiver, source): 1.0 if receiver == source else max(float(value), 1e-6)
        for receiver, row in correlations.iterrows()
        for source, value in row.items()
    }


def train_garl_ddal(
    features: dict[str, pd.DataFrame],
    closes: dict[str, pd.Series],
    levels: tuple[float, ...],
    lookback: int,
    epochs: int,
    rollout_length: int,
    learning_rate: float,
    gamma: float,
    cost_rate: float,
    seed: int,
    device: str = "auto",
    share_after_fraction: float = 0.3,
    share_every: int = 4,
    pool_size: int | None = None,
) -> RLPolicySet:
    """
    Deterministic single-process DDAL-style simulation with aligned model initialisation.

    Every agent starts from the same parameter state. Gradient pieces are merged by generation
    epoch, and each sharing update includes the receiver's own latest gradient. Absolute return
    correlation supplies the task-relevance term for stock-specific environments.
    """
    device = resolve_torch_device(device)
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
    randoms = {ticker: np.random.default_rng(seed + i) for i, ticker in enumerate(tickers)}
    threshold = int(epochs * share_after_fraction)
    pool_size = pool_size or len(tickers)
    relevance = return_relevance(closes)
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
            pending[receiver].extend(
                GradientPiece(
                    piece.gradients,
                    piece.source,
                    piece.epoch,
                    relevance[(receiver, piece.source)],
                )
                for piece in pieces.values()
            )
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
            apply_gradient(models[ticker], optimizers[ticker], weighted_average(chosen))
            pending[ticker].clear()
    return RLPolicySet("garl", models, scalers, tickers, levels, lookback)
