from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .core import fit_feature_scalers, make_states
from .trainers import RLPolicySet


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, actions: int, hidden: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, actions),
        )

    def forward(self, observation):
        return self.network(observation)


class JointQNetwork(nn.Module):
    def __init__(self, per_asset_size: int, assets: int, actions: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(per_asset_size * assets, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, actions) for _ in range(assets)])

    def forward(self, observation):
        hidden = self.body(observation)
        return torch.stack([head(hidden) for head in self.heads], dim=1)


@dataclass
class Transition:
    observation: np.ndarray
    action: int | list[int]
    reward: float
    next_observation: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.items: deque[Transition] = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def append(self, transition: Transition) -> None:
        self.items.append(transition)

    def sample(self, size: int) -> list[Transition]:
        return self.random.sample(self.items, size)

    def __len__(self) -> int:
        return len(self.items)


def _epsilon(epoch: int, epochs: int, start: float, end: float, decay_fraction: float) -> float:
    decay = max(1, int(epochs * decay_fraction))
    progress = min(1.0, epoch / decay)
    return start + progress * (end - start)


def _independent_update(
    model: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    batch: list[Transition],
    gamma: float,
) -> None:
    device = next(model.parameters()).device
    observations = torch.tensor(
        np.stack([item.observation for item in batch]), dtype=torch.float32, device=device
    )
    actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([item.reward for item in batch], device=device)
    next_observations = torch.tensor(
        np.stack([item.next_observation for item in batch]), dtype=torch.float32, device=device
    )
    dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=device)
    predicted = model(observations).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        expected = rewards + gamma * target(next_observations).max(axis=1).values * (1 - dones)
    loss = nn.functional.mse_loss(predicted, expected)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def train_independent_dqn(
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
    epsilon_decay_fraction: float = 0.5,
) -> RLPolicySet:
    tickers = tuple(features)
    scalers = fit_feature_scalers(features)
    states = make_states(
        features, closes, scalers, levels=levels, lookback=lookback, cost_rate=cost_rate
    )
    observation_size = features[tickers[0]].shape[1] * lookback + 1
    models, targets, optimizers, buffers, randoms = {}, {}, {}, {}, {}
    for i, ticker in enumerate(tickers):
        torch.manual_seed(seed + i)
        models[ticker] = QNetwork(observation_size, len(levels)).to(device)
        targets[ticker] = QNetwork(observation_size, len(levels)).to(device)
        targets[ticker].load_state_dict(models[ticker].state_dict())
        optimizers[ticker] = torch.optim.Adam(models[ticker].parameters(), lr=learning_rate)
        buffers[ticker] = ReplayBuffer(5000, seed + i)
        randoms[ticker] = np.random.default_rng(seed + i)
    current = {ticker: states[ticker].observation() for ticker in tickers}

    for epoch in range(epochs):
        epsilon = _epsilon(epoch, epochs, 1.0, 0.05, epsilon_decay_fraction)
        for _ in range(rollout_length):
            for ticker in tickers:
                if randoms[ticker].random() < epsilon:
                    action = int(randoms[ticker].integers(0, len(levels)))
                else:
                    with torch.no_grad():
                        q_values = models[ticker](
                            torch.tensor(current[ticker], device=device).unsqueeze(0)
                        )
                    action = int(torch.argmax(q_values, dim=-1).item())
                next_observation, reward, done = states[ticker].step(action)
                buffers[ticker].append(
                    Transition(current[ticker], action, reward, next_observation, done)
                )
                current[ticker] = states[ticker].reset() if done else next_observation
                if len(buffers[ticker]) >= 64:
                    _independent_update(
                        models[ticker],
                        targets[ticker],
                        optimizers[ticker],
                        buffers[ticker].sample(64),
                        gamma,
                    )
        if (epoch + 1) % 10 == 0:
            for ticker in tickers:
                targets[ticker].load_state_dict(models[ticker].state_dict())
    return RLPolicySet("independent", models, scalers, tickers, levels, lookback)


def train_joint_dqn(
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
    epsilon_decay_fraction: float = 0.5,
) -> RLPolicySet:
    tickers = tuple(features)
    scalers = fit_feature_scalers(features)
    states = make_states(
        features, closes, scalers, levels=levels, lookback=lookback, cost_rate=cost_rate
    )
    per_asset_size = features[tickers[0]].shape[1] * lookback + 1
    torch.manual_seed(seed)
    model = JointQNetwork(per_asset_size, len(tickers), len(levels)).to(device)
    target = JointQNetwork(per_asset_size, len(tickers), len(levels)).to(device)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(5000, seed)
    rng = np.random.default_rng(seed)
    current = {ticker: states[ticker].observation() for ticker in tickers}

    for epoch in range(epochs):
        epsilon = _epsilon(epoch, epochs, 1.0, 0.05, epsilon_decay_fraction)
        for _ in range(rollout_length):
            joint_observation = np.concatenate([current[ticker] for ticker in tickers])
            if rng.random() < epsilon:
                actions = [int(rng.integers(0, len(levels))) for _ in tickers]
            else:
                with torch.no_grad():
                    q_values = model(
                        torch.tensor(joint_observation, device=device).unsqueeze(0)
                    )
                actions = torch.argmax(q_values.squeeze(0), dim=-1).cpu().tolist()
            rewards, dones = [], []
            for ticker, action in zip(tickers, actions):
                next_observation, reward, done = states[ticker].step(action)
                current[ticker] = states[ticker].reset() if done else next_observation
                rewards.append(reward)
                dones.append(done)
            next_joint = np.concatenate([current[ticker] for ticker in tickers])
            buffer.append(
                Transition(
                    joint_observation,
                    actions,
                    float(np.mean(rewards)),
                    next_joint,
                    any(dones),
                )
            )
            if len(buffer) >= 64:
                batch = buffer.sample(64)
                observations = torch.tensor(
                    np.stack([item.observation for item in batch]),
                    dtype=torch.float32,
                    device=device,
                )
                action_tensor = torch.tensor(
                    [item.action for item in batch], dtype=torch.long, device=device
                )
                rewards_tensor = torch.tensor(
                    [item.reward for item in batch], device=device
                )
                next_observations = torch.tensor(
                    np.stack([item.next_observation for item in batch]),
                    dtype=torch.float32,
                    device=device,
                )
                done_tensor = torch.tensor(
                    [item.done for item in batch], dtype=torch.float32, device=device
                )
                predicted = model(observations).gather(
                    2, action_tensor[..., None]
                ).squeeze(-1)
                with torch.no_grad():
                    future = target(next_observations).max(axis=-1).values
                    expected = (
                        rewards_tensor[:, None]
                        + gamma * future * (1 - done_tensor[:, None])
                    )
                loss = nn.functional.mse_loss(predicted, expected)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        if (epoch + 1) % 10 == 0:
            target.load_state_dict(model.state_dict())
    return RLPolicySet("joint", model, scalers, tickers, levels, lookback)

