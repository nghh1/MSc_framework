from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from garl_trading.utils import resolve_torch_device

from .core import (
    ActorCritic,
    JointActorCritic,
    a2c_gradient,
    apply_gradient,
    fit_feature_scalers,
    greedy_asset_positions,
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

    def positions(
        self, features: dict[str, pd.DataFrame], context: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        if self.kind == "joint":
            return joint_positions(self, features, context)
        series = {
            ticker: greedy_asset_positions(
                self.models[ticker],
                self.scalers[ticker],
                features[ticker],
                context[ticker],
                self.levels,
                self.lookback,
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
) -> RLPolicySet:
    device = resolve_torch_device(device)
    tickers = tuple(features)
    scalers = fit_feature_scalers(features)
    states = make_states(
        features, closes, scalers, levels=levels, lookback=lookback, cost_rate=cost_rate
    )
    models, optimizers, randoms = {}, {}, {}
    for i, ticker in enumerate(tickers):
        torch.manual_seed(seed + i)
        observation_size = features[ticker].shape[1] * lookback + 1
        models[ticker] = ActorCritic(observation_size, len(levels)).to(device)
        optimizers[ticker] = torch.optim.Adam(models[ticker].parameters(), lr=learning_rate)
        randoms[ticker] = np.random.default_rng(seed + i)
    for _ in range(epochs):
        for ticker in tickers:
            gradients, _ = a2c_gradient(
                models[ticker],
                states[ticker],
                rollout_length=rollout_length,
                gamma=gamma,
                rng=randoms[ticker],
            )
            apply_gradient(models[ticker], optimizers[ticker], gradients)
    return RLPolicySet("independent", models, scalers, tickers, levels, lookback)


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
) -> RLPolicySet:
    device = resolve_torch_device(device)
    tickers = tuple(features)
    scalers = fit_feature_scalers(features)
    states = make_states(features, closes, scalers, levels, lookback, cost_rate)
    per_asset_size = features[tickers[0]].shape[1] * lookback + 1
    torch.manual_seed(seed)
    model = JointActorCritic(per_asset_size, len(tickers), len(levels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    observations = {ticker: states[ticker].observation() for ticker in tickers}

    for _ in range(epochs):
        obs_buffer, action_buffer, reward_buffer, done_buffer = [], [], [], []
        for _ in range(rollout_length):
            joint_obs = np.concatenate([observations[ticker] for ticker in tickers])
            with torch.no_grad():
                logits, _ = model(torch.tensor(joint_obs, device=device).unsqueeze(0))
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
            done_buffer.append(any(dones))

        obs_tensor = torch.tensor(np.stack(obs_buffer), dtype=torch.float32, device=device)
        action_tensor = torch.tensor(action_buffer, dtype=torch.long, device=device)
        logits, values = model(obs_tensor)
        with torch.no_grad():
            bootstrap_obs = np.concatenate([observations[ticker] for ticker in tickers])
            _, bootstrap = model(torch.tensor(bootstrap_obs, device=device).unsqueeze(0))
        returns = np.zeros(len(reward_buffer), dtype=np.float32)
        running = float(bootstrap.item())
        for i in reversed(range(len(returns))):
            running = reward_buffer[i] + gamma * running * (not done_buffer[i])
            returns[i] = running
        return_tensor = torch.tensor(returns, device=device)
        advantage = return_tensor - values
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(2, action_tensor[..., None]).squeeze(-1).sum(axis=1)
        entropy = -(torch.softmax(logits, dim=-1) * log_probs).sum(axis=-1).mean()
        loss = (
            -(chosen * advantage.detach()).mean()
            + 0.5 * nn.functional.mse_loss(values, return_tensor)
            - 0.01 * entropy
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return RLPolicySet("joint", model, scalers, tickers, levels, lookback)


@torch.no_grad()
def joint_positions(
    policy: RLPolicySet, features: dict[str, pd.DataFrame], context: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    model = policy.models
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
    for date in index:
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
            current[ticker] = float(policy.levels[action])
            row[ticker] = current[ticker]
        rows.append(row)
    return pd.DataFrame(rows, index=index)
