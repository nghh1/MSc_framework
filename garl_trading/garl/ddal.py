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


def gradient_cosine_similarity(
    first: list[torch.Tensor], second: list[torch.Tensor]
) -> float:
    """Cosine similarity between two complete model-gradient vectors."""
    dot = sum(float(torch.sum(left * right).item()) for left, right in zip(first, second))
    first_norm = np.sqrt(sum(float(torch.sum(value.square()).item()) for value in first))
    second_norm = np.sqrt(sum(float(torch.sum(value.square()).item()) for value in second))
    denominator = first_norm * second_norm
    return dot / denominator if denominator > 0 else 0.0


def selective_weighted_average(
    local: GradientPiece,
    pieces: list[GradientPiece],
    alignment_threshold: float = 0.05,
    peer_mix: float = 0.5,
) -> tuple[list[torch.Tensor], dict[str, float | int]]:
    """Blend a local anchor with positively relevant and aligned peer gradients.

    Accepted peers retain DDAL's equal experience/relevance weighting, with cosine
    alignment acting as an additional receiver-side confidence weight. The convex
    local/peer blend prevents a large queue from overwhelming the receiver's own signal.
    """
    if not 0 <= peer_mix <= 1:
        raise ValueError("peer_mix must lie in [0, 1].")
    candidates = [piece for piece in pieces if piece.source != local.source]
    scored = [
        (piece, gradient_cosine_similarity(local.gradients, piece.gradients))
        for piece in candidates
    ]
    accepted = [
        (piece, alignment)
        for piece, alignment in scored
        if piece.relevance > 0 and alignment > alignment_threshold
    ]
    accepted_alignments = [alignment for _, alignment in accepted]
    diagnostics = {
        "peer_candidates": len(candidates),
        "peer_accepted": len(accepted),
        "peer_acceptance_rate": len(accepted) / len(candidates) if candidates else 0.0,
        "mean_accepted_alignment": (
            float(np.mean(accepted_alignments)) if accepted_alignments else 0.0
        ),
        "local_gradient_weight": 1.0,
    }
    if not accepted or peer_mix == 0:
        return [gradient.clone() for gradient in local.gradients], diagnostics

    experience = np.asarray([piece.epoch + 1 for piece, _ in accepted], dtype=float)
    relevance = np.asarray([piece.relevance for piece, _ in accepted], dtype=float)
    alignment = np.asarray(accepted_alignments, dtype=float)
    ddal_weights = 0.5 * experience / experience.sum() + 0.5 * relevance / relevance.sum()
    peer_weights = ddal_weights * alignment
    peer_weights /= peer_weights.sum()
    peer_average = [
        sum(
            float(weight) * piece.gradients[index]
            for weight, (piece, _) in zip(peer_weights, accepted)
        )
        for index in range(len(local.gradients))
    ]
    diagnostics["local_gradient_weight"] = 1.0 - peer_mix
    return [
        (1.0 - peer_mix) * local_gradient + peer_mix * peer_gradient
        for local_gradient, peer_gradient in zip(local.gradients, peer_average)
    ], diagnostics


def return_relevance(closes: dict[str, pd.Series]) -> dict[tuple[str, str], float]:
    """Map source/receiver similarity to DDAL relevance weights using return correlation."""
    returns = pd.DataFrame(closes).pct_change(fill_method=None)
    correlations = returns.corr().abs().fillna(0.0)
    return {
        (receiver, source): 1.0 if receiver == source else max(float(value), 1e-6)
        for receiver, row in correlations.iterrows()
        for source, value in row.items()
    }


def positive_return_relevance(closes: dict[str, pd.Series]) -> dict[tuple[str, str], float]:
    """Use positive signed return correlation as the selective-GARL task relevance proxy."""
    returns = pd.DataFrame(closes).pct_change(fill_method=None)
    correlations = returns.corr().fillna(0.0).clip(lower=0.0)
    return {
        (receiver, source): 1.0 if receiver == source else float(value)
        for receiver, row in correlations.iterrows()
        for source, value in row.items()
    }


def latest_peer_pieces(
    queue: list[GradientPiece], pool_size: int | None
) -> list[GradientPiece]:
    """Consume at most the newest queued gradient from each source agent."""
    newest: dict[str, GradientPiece] = {}
    for piece in queue:
        previous = newest.get(piece.source)
        if previous is None or piece.epoch > previous.epoch:
            newest[piece.source] = piece
    queue.clear()
    selected = sorted(newest.values(), key=lambda piece: piece.epoch, reverse=True)
    return selected if pool_size is None else selected[:pool_size]


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
    encoder_channels: int = 32,
    encoder_kernel_size: int = 3,
    encoder_dilations: tuple[int, ...] = (1, 2, 4, 8),
    encoder_dropout: float = 0.0,
    gae_lambda: float = 0.95,
    entropy_weight: float = 0.01,
    turnover_penalty_multiplier: float = 1.0,
    share_after_fraction: float = 0.3,
    share_every: int = 4,
    pool_size: int | None = None,
    short_borrow_bps_annual: float = 0.0,
    rebalance_threshold: float = 0.0,
    decision_interval: int = 1,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-4,
    minimum_train_epochs: int = 30,
    selective: bool = False,
    alignment_threshold: float = 0.05,
    selective_peer_mix: float = 0.5,
) -> RLPolicySet:
    """
    Deterministic event-driven simulation of decentralised asynchronous DDAL queues.

    Agents start from a common parameter template, then use independent environments, random
    streams, optimisers, and local clocks. After private learning, generated gradients enter peer
    queues. At each sharing event, a receiver consumes only the newest gradient from each source,
    limiting staleness while preserving deterministic asynchronous event ordering.
    """
    device = resolve_torch_device(device)
    tickers = tuple(features)
    scalers = fit_feature_scalers(features)
    states = make_states(
        features, closes, scalers, levels=levels, lookback=lookback, cost_rate=cost_rate,
        turnover_penalty_multiplier=turnover_penalty_multiplier,
        short_borrow_bps_annual=short_borrow_bps_annual,
        rebalance_threshold=rebalance_threshold,
        decision_interval=decision_interval,
    )
    observation_size = features[tickers[0]].shape[1] * lookback + 1
    models = initialise_asset_actor_critics(
        tickers,
        observation_size,
        len(levels),
        seed,
        device,
        lookback=lookback,
        encoder_channels=encoder_channels,
        encoder_kernel_size=encoder_kernel_size,
        encoder_dilations=encoder_dilations,
        encoder_dropout=encoder_dropout,
    )
    optimizers = {
        ticker: torch.optim.Adam(models[ticker].parameters(), lr=learning_rate)
        for ticker in tickers
    }
    randoms = {ticker: np.random.default_rng(seed + i) for i, ticker in enumerate(tickers)}
    threshold = int(epochs * share_after_fraction)
    relevance = positive_return_relevance(closes) if selective else return_relevance(closes)
    queues: dict[str, list[GradientPiece]] = {ticker: [] for ticker in tickers}
    has_shared_update = {ticker: False for ticker in tickers}
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
            gamma=gamma**decision_interval,
            rng=randoms[ticker],
            gae_lambda=gae_lambda,
            entropy_weight=entropy_weight,
        )
        shared_update = False
        selection_diagnostics: dict[str, float | int] = {}
        update_kind = "local"
        if epoch < threshold:
            apply_gradient(models[ticker], optimizers[ticker], gradient)
        else:
            local_piece = GradientPiece(gradient, ticker, epoch, 1.0)
            for receiver in tickers:
                if receiver == ticker:
                    continue
                queues[receiver].append(
                    GradientPiece(
                        gradient,
                        ticker,
                        epoch,
                        relevance[(receiver, ticker)]
                    )
                )
            sharing_due = (epoch - threshold + 1) % share_every == 0
            if sharing_due and queues[ticker]:
                chosen = latest_peer_pieces(queues[ticker], pool_size)
                if selective:
                    averaged, selection_diagnostics = selective_weighted_average(
                        local_piece,
                        chosen,
                        alignment_threshold=alignment_threshold,
                        peer_mix=selective_peer_mix,
                    )
                    apply_gradient(models[ticker], optimizers[ticker], averaged)
                    shared_update = bool(selection_diagnostics["peer_accepted"])
                else:
                    apply_gradient(
                        models[ticker],
                        optimizers[ticker],
                        weighted_average([local_piece, *chosen]),
                    )
                    shared_update = True
                update_kind = "shared" if shared_update else "local"
                has_shared_update[ticker] |= shared_update
            else:
                apply_gradient(models[ticker], optimizers[ticker], gradient)
        diagnostics.append(
            {
                "epoch": epoch,
                "agent": ticker,
                "simulation_time": simulation_time,
                "training_reward": reward,
                "loss": loss,
                "queue_size": len(queues[ticker]),
                "shared_update": shared_update,
                "update_kind": update_kind,
                "checkpoint_eligible": has_shared_update[ticker],
                **selection_diagnostics,
            }
        )
        reward_history[ticker].append(reward)
        smoothed = float(np.mean(reward_history[ticker][-5:]))
        if stoppers[ticker].update(
            epoch,
            smoothed,
            models[ticker],
            checkpoint_eligible=has_shared_update[ticker],
        ):
            diagnostics[-1].update(
                {
                    "early_stopped": True,
                    "best_epoch": stoppers[ticker].best_epoch,
                    "stop_epoch": stoppers[ticker].stop_epoch,
                }
            )
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
        "selective_garl" if selective else "garl",
        models,
        scalers,
        tickers,
        levels,
        lookback,
        cost_rate,
        short_borrow_bps_annual / 10000 / 252,
        diagnostics,
        rebalance_threshold,
        decision_interval,
    )


def train_selective_garl_ddal(
    *args,
    alignment_threshold: float = 0.05,
    peer_mix: float = 0.5,
    **kwargs,
) -> RLPolicySet:
    """DDAL extension that rejects irrelevant or gradient-conflicting peer knowledge."""
    return train_garl_ddal(
        *args,
        selective=True,
        alignment_threshold=alignment_threshold,
        selective_peer_mix=peer_mix,
        **kwargs,
    )
