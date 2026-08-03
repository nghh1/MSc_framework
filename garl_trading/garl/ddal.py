"""
A2C-based DDAL implementation for Group-Agent Reinforcement Learning.
This module reconstructs the GARL mechanism proposed by Wu and Zeng.
PPO and DQN algorithm live only in the non-GARL RL baseline modules.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from garl_trading.rl.core import (
    RewardEarlyStopper,
    a2c_gradient,
    apply_gradient,
    fit_feature_scalers,
    initialise_asset_actor_critics,
    make_states
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
    short_borrow_bps_annual: float = 0.0,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-4,
    minimum_train_epochs: int = 30
) -> RLPolicySet:
    """
    Deterministic event-driven simulation of decentralised asynchronous DDAL queues.

    Agents have independent initial parameters and local clocks. After private learning, every
    generated gradient is copied into every agent's FIFO knowledge queue. Each agent independently
    retrieves and removes a batch from its queue on its own update schedule, matching Algorithm 1
    in Wu and Zeng as closely as possible without requiring distributed hardware.
    """
    device = resolve_torch_device(device)
    tickers = tuple(features)
    scalers = fit_feature_scalers(features)
    states = make_states(
        features, closes, scalers, levels=levels, lookback=lookback, cost_rate=cost_rate,
        short_borrow_bps_annual=short_borrow_bps_annual
    )
    observation_size = features[tickers[0]].shape[1] * lookback + 1
    models = initialise_asset_actor_critics(
        tickers, observation_size, len(levels), seed, device
    )
    optimizers = {
        ticker: torch.optim.Adam(models[ticker].parameters(), lr=learning_rate)
        for ticker in tickers
    }
    randoms = {ticker: np.random.default_rng(seed + i) for i, ticker in enumerate(tickers)}
    threshold = int(epochs * share_after_fraction)
    relevance = return_relevance(closes)
    queues: dict[str, list[GradientPiece]] = {ticker: [] for ticker in tickers}
    local_epochs = {ticker: 0 for ticker in tickers}
    scheduler = np.random.default_rng(seed + 100000)
    pace = {ticker: float(scheduler.uniform(0.75, 1.25)) for ticker in tickers}
    events = [
        (float(scheduler.exponential(pace[ticker])), number, ticker)
        for number, ticker in enumerate(tickers)
    ]
    heapq.heapify(events)
    event_number = len(events)
    diagnostics: list[dict] = []
    stoppers = {
        ticker: RewardEarlyStopper(
            early_stopping_patience, early_stopping_min_delta, minimum_train_epochs
        )
        for ticker in tickers
    }
    reward_history: dict[str, list[float]] = {ticker: [] for ticker in tickers}

    while events:
        simulation_time, _, ticker = heapq.heappop(events)
        epoch = local_epochs[ticker]
        if epoch >= epochs:
            continue
        gradient, loss, reward = a2c_gradient(
            models[ticker],
            states[ticker],
            rollout_length=rollout_length,
            gamma=gamma,
            rng=randoms[ticker]
        )
        shared_update = False
        if epoch < threshold:
            apply_gradient(models[ticker], optimizers[ticker], gradient)
        else:
            for receiver in tickers:
                queues[receiver].append(
                    GradientPiece(
                        gradient,
                        ticker,
                        epoch,
                        relevance[(receiver, ticker)]
                    )
                )
            if (epoch + 1) % share_every == 0 and queues[ticker]:
                take = (
                    len(queues[ticker])
                    if pool_size is None
                    else min(pool_size, len(queues[ticker]))
                )
                chosen = queues[ticker][:take]
                del queues[ticker][:take]
                apply_gradient(models[ticker], optimizers[ticker], weighted_average(chosen))
                shared_update = True
        diagnostics.append(
            {
                "epoch": epoch,
                "agent": ticker,
                "simulation_time": simulation_time,
                "training_reward": reward,
                "loss": loss,
                "queue_size": len(queues[ticker]),
                "shared_update": shared_update
            }
        )
        reward_history[ticker].append(reward)
        smoothed = float(np.mean(reward_history[ticker][-5:]))
        if stoppers[ticker].update(epoch, smoothed, models[ticker]):
            diagnostics[-1]["early_stopped"] = True
            stoppers[ticker].restore(models[ticker])
            local_epochs[ticker] = epochs
            continue
        local_epochs[ticker] += 1
        if local_epochs[ticker] < epochs:
            heapq.heappush(
                events,
                (
                    simulation_time + float(scheduler.exponential(pace[ticker])),
                    event_number,
                    ticker,
                )
            )
            event_number += 1
    for ticker in tickers:
        stoppers[ticker].restore(models[ticker])
    return RLPolicySet(
        "garl",
        models,
        scalers,
        tickers,
        levels,
        lookback,
        cost_rate,
        short_borrow_bps_annual / 10000 / 252,
        diagnostics
    )
