from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from garl_trading.utils import resolve_torch_device

from .core import (
    JointActorCritic,
    RewardEarlyStopper,
    a2c_gradient,
    apply_gradient,
    fit_feature_scalers,
    greedy_asset_positions,
    incremental_target,
    initialise_asset_actor_critics,
    make_states,
)


@dataclass
class RLPolicySet:
    kind: str
    models: dict[str, torch.nn.Module] | torch.nn.Module
    scalers: dict[str, StandardScaler]
    tickers: tuple[str, ...]
    levels: tuple[float, ...]
    lookback: int
    cost_rate: float = 0.0
    short_borrow_rate: float = 0.0
    diagnostics: list[dict] = field(default_factory=list)
    rebalance_threshold: float = 0.0
    decision_interval: int = 1

    def positions(
        self,
        features: dict[str, pd.DataFrame],
        context: dict[str, pd.DataFrame],
        closes: dict[str, pd.Series] | None = None
    ) -> pd.DataFrame:
        if self.kind == "joint":
            return joint_positions(self, features, context, closes=closes)
        series = {
            ticker: greedy_asset_positions(
                self.models[ticker],
                self.scalers[ticker],
                features[ticker],
                context[ticker],
                self.levels,
                self.lookback,
                closes[ticker] if closes is not None else None,
                self.cost_rate,
                self.short_borrow_rate,
                self.rebalance_threshold,
                self.decision_interval,
            )
            for ticker in self.tickers
        }
        return pd.DataFrame(series).reindex(features[self.tickers[0]].index)


def train_independent_a2c(
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
    short_borrow_bps_annual: float = 0.0,
    rebalance_threshold: float = 0.0,
    decision_interval: int = 1,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-4,
    minimum_train_epochs: int = 30
) -> RLPolicySet:
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
    optimizers, randoms = {}, {}
    for i, ticker in enumerate(tickers):
        optimizers[ticker] = torch.optim.Adam(models[ticker].parameters(), lr=learning_rate)
        randoms[ticker] = np.random.default_rng(seed + i)
    diagnostics = []
    stoppers = {
        ticker: RewardEarlyStopper(
            early_stopping_patience, early_stopping_min_delta, minimum_train_epochs
        )
        for ticker in tickers
    }
    reward_history: dict[str, list[float]] = {ticker: [] for ticker in tickers}
    active = set(tickers)
    for epoch in range(epochs):
        for ticker in tickers:
            if ticker not in active:
                continue
            gradients, loss, reward = a2c_gradient(
                models[ticker],
                states[ticker],
                rollout_length=rollout_length,
                gamma=gamma**decision_interval,
                rng=randoms[ticker],
                gae_lambda=gae_lambda,
                entropy_weight=entropy_weight,
            )
            apply_gradient(models[ticker], optimizers[ticker], gradients)
            reward_history[ticker].append(reward)
            row = {
                "epoch": epoch,
                "agent": ticker,
                "training_reward": reward,
                "loss": loss,
                "checkpoint_eligible": epoch + 1 > minimum_train_epochs,
            }
            smoothed = float(np.mean(reward_history[ticker][-5:]))
            if stoppers[ticker].update(epoch, smoothed, models[ticker]):
                row.update(
                    {
                        "early_stopped": True,
                        "best_epoch": stoppers[ticker].best_epoch,
                        "stop_epoch": stoppers[ticker].stop_epoch,
                    }
                )
                stoppers[ticker].restore(models[ticker])
                active.remove(ticker)
            diagnostics.append(row)
        if not active:
            break
    for ticker in tickers:
        stoppers[ticker].restore(models[ticker])
    return RLPolicySet(
        "independent",
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


def train_joint_a2c(
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
    short_borrow_bps_annual: float = 0.0,
    rebalance_threshold: float = 0.0,
    decision_interval: int = 1,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-4,
    minimum_train_epochs: int = 30
) -> RLPolicySet:
    device = resolve_torch_device(device)
    tickers = tuple(features)
    scalers = fit_feature_scalers(features)
    states = make_states(
        features, closes, scalers, levels, lookback, cost_rate,
        turnover_penalty_multiplier=turnover_penalty_multiplier,
        short_borrow_bps_annual=short_borrow_bps_annual,
        rebalance_threshold=rebalance_threshold,
        decision_interval=decision_interval,
    )
    per_asset_size = features[tickers[0]].shape[1] * lookback + 1
    torch.manual_seed(seed)
    model = JointActorCritic(
        per_asset_size,
        len(tickers),
        len(levels),
        lookback=lookback,
        encoder_channels=encoder_channels,
        encoder_kernel_size=encoder_kernel_size,
        encoder_dilations=encoder_dilations,
        encoder_dropout=encoder_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    observations = {ticker: states[ticker].observation() for ticker in tickers}

    diagnostics = []
    stopper = RewardEarlyStopper(
        early_stopping_patience, early_stopping_min_delta, minimum_train_epochs
    )
    for epoch in range(epochs):
        obs_buffer, action_buffer, reward_buffer, value_buffer, done_buffer = [], [], [], [], []
        for _ in range(rollout_length):
            joint_obs = np.concatenate([observations[ticker] for ticker in tickers])
            with torch.no_grad():
                logits, value = model(torch.tensor(joint_obs, device=device).unsqueeze(0))
                probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            actions = [
                int(rng.choice(len(levels), p=probabilities[i])) for i in range(len(tickers))
            ]
            rewards, dones = [], []
            for ticker, action in zip(tickers, actions):
                next_obs, reward, done = states[ticker].step(action)
                observations[ticker] = states[ticker].reset() if done else next_obs
                rewards.append(reward)
                dones.append(done)
            obs_buffer.append(joint_obs)
            action_buffer.append(actions)
            reward_buffer.append(float(np.mean(rewards)))
            value_buffer.append(float(value.item()))
            done_buffer.append(any(dones))

        obs_tensor = torch.tensor(np.stack(obs_buffer), dtype=torch.float32, device=device)
        action_tensor = torch.tensor(action_buffer, dtype=torch.long, device=device)
        logits, values = model(obs_tensor)
        with torch.no_grad():
            bootstrap_obs = np.concatenate([observations[ticker] for ticker in tickers])
            _, bootstrap = model(torch.tensor(bootstrap_obs, device=device).unsqueeze(0))
        advantages = np.zeros(len(reward_buffer), dtype=np.float32)
        running_advantage = 0.0
        next_value = float(bootstrap.item())
        for i in reversed(range(len(advantages))):
            mask = 0.0 if done_buffer[i] else 1.0
            following_value = next_value if i == len(advantages) - 1 else value_buffer[i + 1]
            discount = gamma**decision_interval
            delta = reward_buffer[i] + discount * following_value * mask - value_buffer[i]
            running_advantage = delta + discount * gae_lambda * mask * running_advantage
            advantages[i] = running_advantage
        returns = advantages + np.asarray(value_buffer, dtype=np.float32)
        return_tensor = torch.tensor(returns, device=device)
        advantage = torch.tensor(advantages, device=device)
        advantage = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-8
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(2, action_tensor[..., None]).squeeze(-1).sum(axis=1)
        entropy = -(torch.softmax(logits, dim=-1) * log_probs).sum(axis=-1).mean()
        loss = (
            -(chosen * advantage.detach()).mean()
            + 0.5 * nn.functional.mse_loss(values, return_tensor)
            - entropy_weight * entropy
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        diagnostics.append(
            {
                "epoch": epoch,
                "training_reward": float(np.mean(reward_buffer)),
                "loss": float(loss.item()),
                "entropy": float(entropy.item())
            }
        )
        smoothed = float(np.mean([row["training_reward"] for row in diagnostics[-5:]]))
        if stopper.update(epoch, smoothed, model):
            diagnostics[-1].update(
                {
                    "early_stopped": True,
                    "best_epoch": stopper.best_epoch,
                    "stop_epoch": stopper.stop_epoch,
                }
            )
            break
    stopper.restore(model)
    return RLPolicySet(
        "joint",
        model,
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


@torch.no_grad()
def joint_positions(
    policy: RLPolicySet,
    features: dict[str, pd.DataFrame],
    context: dict[str, pd.DataFrame],
    closes: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    model = policy.models
    model.eval()
    device = next(model.parameters()).device
    combined, locations, values = {}, {}, {}
    for ticker in policy.tickers:
        combined[ticker] = pd.concat([context[ticker], features[ticker]])
        combined[ticker] = (
            combined[ticker].loc[~combined[ticker].index.duplicated(keep="last")].sort_index()
        )
        locations[ticker] = {date: i for i, date in enumerate(combined[ticker].index)}
        values[ticker] = policy.scalers[ticker].transform(combined[ticker]).astype(np.float32)
    current = {ticker: 0.0 for ticker in policy.tickers}
    rows = []
    index = features[policy.tickers[0]].index
    for step, date in enumerate(index):
        if step % policy.decision_interval != 0:
            rows.append(dict(current))
            continue
        observations = []
        for ticker in policy.tickers:
            i = locations[ticker][date]
            window = values[ticker][max(0, i - policy.lookback + 1) : i + 1]
            if len(window) < policy.lookback:
                window = np.concatenate(
                    [np.repeat(window[:1], policy.lookback - len(window), axis=0), window]
                )
            observations.append(
                np.concatenate([window.reshape(-1), [current[ticker]]]).astype(np.float32)
            )
        output = model(torch.tensor(np.concatenate(observations), device=device).unsqueeze(0))
        logits = output[0] if isinstance(output, tuple) else output
        actions = torch.argmax(logits.squeeze(0), dim=-1).cpu().numpy()
        row = {}
        for ticker, action in zip(policy.tickers, actions):
            current[ticker] = incremental_target(
                current[ticker], action, np.asarray(policy.levels, dtype=float)
            )
            row[ticker] = current[ticker]
        rows.append(row)
    return pd.DataFrame(rows, index=index)
