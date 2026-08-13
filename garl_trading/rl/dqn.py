from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from garl_trading.utils import resolve_torch_device

from .core import RewardEarlyStopper, TemporalFeatureExtractor, fit_feature_scalers, make_states
from .trainers import RLPolicySet


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_size: int,
        actions: int,
        hidden: int = 64,
        *,
        lookback: int = 20,
        encoder_channels: int = 32,
        encoder_kernel_size: int = 3,
        encoder_dilations: tuple[int, ...] = (1, 2, 4, 8),
        encoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.extractor = TemporalFeatureExtractor(
            observation_size,
            lookback,
            channels=encoder_channels,
            kernel_size=encoder_kernel_size,
            dilations=encoder_dilations,
            dropout=encoder_dropout,
        )
        self.network = nn.Sequential(
            nn.Linear(self.extractor.output_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, actions)
        )

    def forward(self, observation):
        return self.network(self.extractor(observation))


class BranchingQNetwork(nn.Module):
    """Shared representation with one discrete Q-value branch per stock."""

    def __init__(
        self,
        per_asset_size: int,
        assets: int,
        actions: int,
        hidden: int = 128,
        *,
        lookback: int = 20,
        encoder_channels: int = 32,
        encoder_kernel_size: int = 3,
        encoder_dilations: tuple[int, ...] = (1, 2, 4, 8),
        encoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.assets = assets
        self.per_asset_size = per_asset_size
        self.extractor = TemporalFeatureExtractor(
            per_asset_size,
            lookback,
            channels=encoder_channels,
            kernel_size=encoder_kernel_size,
            dilations=encoder_dilations,
            dropout=encoder_dropout,
        )
        self.body = nn.Sequential(
            nn.Linear(self.extractor.output_size * assets, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, actions) for _ in range(assets)])

    def forward(self, observation):
        asset_observations = observation.reshape(-1, self.assets, self.per_asset_size)
        encoded = self.extractor(asset_observations.reshape(-1, self.per_asset_size))
        encoded = encoded.reshape(-1, self.assets * self.extractor.output_size)
        hidden = self.body(encoded)
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


def epsilon(epoch: int, epochs: int, start: float, end: float, decay_fraction: float) -> float:
    decay = max(1, int(epochs * decay_fraction))
    progress = min(1.0, epoch / decay)
    return start + progress * (end - start)


def independent_update(
    model: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    batch: list[Transition],
    gamma: float
) -> float:
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
        next_actions = model(next_observations).argmax(axis=1, keepdim=True)
        future = target(next_observations).gather(1, next_actions).squeeze(1)
        expected = rewards + gamma * future * (1 - dones)
    loss = nn.functional.smooth_l1_loss(predicted, expected)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def train_independent_dqn(
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
    epsilon_decay_fraction: float = 0.5,
    target_update_interval: int = 10,
    turnover_penalty_multiplier: float = 1.0,
    short_borrow_bps_annual: float = 0.0,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-4,
    minimum_train_epochs: int = 30
) -> RLPolicySet:
    if target_update_interval < 1:
        raise ValueError("target_update_interval must be positive.")
    device = resolve_torch_device(device)
    tickers = tuple(features)
    scalers = fit_feature_scalers(features)
    states = make_states(
        features, closes, scalers, levels=levels, lookback=lookback, cost_rate=cost_rate,
        turnover_penalty_multiplier=turnover_penalty_multiplier,
        short_borrow_bps_annual=short_borrow_bps_annual
    )
    observation_size = features[tickers[0]].shape[1] * lookback + 1
    models, targets, optimizers, buffers, randoms = {}, {}, {}, {}, {}
    for i, ticker in enumerate(tickers):
        torch.manual_seed(seed + i)
        network_parameters = {
            "lookback": lookback,
            "encoder_channels": encoder_channels,
            "encoder_kernel_size": encoder_kernel_size,
            "encoder_dilations": encoder_dilations,
            "encoder_dropout": encoder_dropout,
        }
        models[ticker] = QNetwork(
            observation_size, len(levels), **network_parameters
        ).to(device)
        targets[ticker] = QNetwork(
            observation_size, len(levels), **network_parameters
        ).to(device)
        targets[ticker].load_state_dict(models[ticker].state_dict())
        targets[ticker].eval()
        optimizers[ticker] = torch.optim.Adam(models[ticker].parameters(), lr=learning_rate)
        buffers[ticker] = ReplayBuffer(5000, seed + i)
        randoms[ticker] = np.random.default_rng(seed + i)
    current = {ticker: states[ticker].observation() for ticker in tickers}

    diagnostics = []
    exploration_decay_epochs = max(1, int(epochs * epsilon_decay_fraction))
    checkpoint_minimum_epochs = max(minimum_train_epochs, exploration_decay_epochs)
    stopper = RewardEarlyStopper(
        early_stopping_patience, early_stopping_min_delta, checkpoint_minimum_epochs
    )
    for epoch in range(epochs):
        epsilon_value = epsilon(epoch, epochs, 1.0, 0.05, epsilon_decay_fraction)
        epoch_rewards, epoch_losses = [], []
        for _ in range(rollout_length):
            for ticker in tickers:
                if randoms[ticker].random() < epsilon_value:
                    action = int(randoms[ticker].integers(0, len(levels)))
                else:
                    with torch.no_grad():
                        q_values = models[ticker](
                            torch.tensor(current[ticker], device=device).unsqueeze(0)
                        )
                    action = int(torch.argmax(q_values, dim=-1).item())
                next_observation, reward, done = states[ticker].step(action)
                epoch_rewards.append(reward)
                buffers[ticker].append(
                    Transition(current[ticker], action, reward, next_observation, done)
                )
                current[ticker] = states[ticker].reset() if done else next_observation
                if len(buffers[ticker]) >= 64:
                    epoch_losses.append(
                        independent_update(
                            models[ticker],
                            targets[ticker],
                            optimizers[ticker],
                            buffers[ticker].sample(64),
                            gamma
                        )
                    )
        if (epoch + 1) % target_update_interval == 0:
            for ticker in tickers:
                targets[ticker].load_state_dict(models[ticker].state_dict())
        diagnostics.append(
            {
                "epoch": epoch,
                "training_reward": float(np.mean(epoch_rewards)),
                "loss": float(np.mean(epoch_losses)) if epoch_losses else np.nan,
                "epsilon": epsilon_value,
                "checkpoint_eligible": epoch + 1 > checkpoint_minimum_epochs,
            }
        )
        smoothed = float(np.mean([row["training_reward"] for row in diagnostics[-5:]]))
        if stopper.update(epoch, smoothed, models):
            diagnostics[-1].update(
                {
                    "early_stopped": True,
                    "best_epoch": stopper.best_epoch,
                    "stop_epoch": stopper.stop_epoch,
                }
            )
            break
    stopper.restore(models)
    return RLPolicySet(
        "independent",
        models,
        scalers,
        tickers,
        levels,
        lookback,
        cost_rate,
        short_borrow_bps_annual / 10000 / 252,
        diagnostics
    )


def train_joint_dqn(
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
    epsilon_decay_fraction: float = 0.5,
    target_update_interval: int = 10,
    turnover_penalty_multiplier: float = 1.0,
    short_borrow_bps_annual: float = 0.0,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-4,
    minimum_train_epochs: int = 30
) -> RLPolicySet:
    if target_update_interval < 1:
        raise ValueError("target_update_interval must be positive.")
    device = resolve_torch_device(device)
    tickers = tuple(features)
    scalers = fit_feature_scalers(features)
    states = make_states(
        features, closes, scalers, levels=levels, lookback=lookback, cost_rate=cost_rate,
        turnover_penalty_multiplier=turnover_penalty_multiplier,
        short_borrow_bps_annual=short_borrow_bps_annual,
    )
    per_asset_size = features[tickers[0]].shape[1] * lookback + 1
    torch.manual_seed(seed)
    network_parameters = {
        "lookback": lookback,
        "encoder_channels": encoder_channels,
        "encoder_kernel_size": encoder_kernel_size,
        "encoder_dilations": encoder_dilations,
        "encoder_dropout": encoder_dropout,
    }
    model = BranchingQNetwork(
        per_asset_size, len(tickers), len(levels), **network_parameters
    ).to(device)
    target = BranchingQNetwork(
        per_asset_size, len(tickers), len(levels), **network_parameters
    ).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(5000, seed)
    rng = np.random.default_rng(seed)
    current = {ticker: states[ticker].observation() for ticker in tickers}

    diagnostics = []
    exploration_decay_epochs = max(1, int(epochs * epsilon_decay_fraction))
    checkpoint_minimum_epochs = max(minimum_train_epochs, exploration_decay_epochs)
    stopper = RewardEarlyStopper(
        early_stopping_patience, early_stopping_min_delta, checkpoint_minimum_epochs
    )
    for epoch in range(epochs):
        epsilon_value = epsilon(epoch, epochs, 1.0, 0.05, epsilon_decay_fraction)
        epoch_rewards, epoch_losses = [], []
        for _ in range(rollout_length):
            joint_observation = np.concatenate([current[ticker] for ticker in tickers])
            if rng.random() < epsilon_value:
                actions = [int(rng.integers(0, len(levels))) for _ in tickers]
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(joint_observation, device=device).unsqueeze(0))
                actions = torch.argmax(q_values.squeeze(0), dim=-1).cpu().tolist()
            rewards, dones = [], []
            for ticker, action in zip(tickers, actions):
                next_observation, reward, done = states[ticker].step(action)
                current[ticker] = states[ticker].reset() if done else next_observation
                rewards.append(reward)
                epoch_rewards.append(reward)
                dones.append(done)
            next_joint = np.concatenate([current[ticker] for ticker in tickers])
            buffer.append(
                Transition(
                    joint_observation, actions, float(np.mean(rewards)), next_joint, any(dones)
                )
            )
            if len(buffer) >= 64:
                batch = buffer.sample(64)
                observations = torch.tensor(
                    np.stack([item.observation for item in batch]),
                    dtype=torch.float32,
                    device=device
                )
                action_tensor = torch.tensor(
                    [item.action for item in batch], dtype=torch.long, device=device
                )
                rewards_tensor = torch.tensor([item.reward for item in batch], device=device)
                next_observations = torch.tensor(
                    np.stack([item.next_observation for item in batch]),
                    dtype=torch.float32,
                    device=device
                )
                done_tensor = torch.tensor(
                    [item.done for item in batch], dtype=torch.float32, device=device
                )
                predicted = model(observations).gather(2, action_tensor[..., None]).squeeze(-1)
                with torch.no_grad():
                    next_actions = model(next_observations).argmax(axis=-1, keepdim=True)
                    future = target(next_observations).gather(2, next_actions).squeeze(-1)
                    expected = rewards_tensor[:, None] + gamma * future * (1 - done_tensor[:, None])
                loss = nn.functional.smooth_l1_loss(predicted, expected)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.item()))
        if (epoch + 1) % target_update_interval == 0:
            target.load_state_dict(model.state_dict())
        diagnostics.append(
            {
                "epoch": epoch,
                "training_reward": float(np.mean(epoch_rewards)),
                "loss": float(np.mean(epoch_losses)) if epoch_losses else np.nan,
                "epsilon": epsilon_value,
                "checkpoint_eligible": epoch + 1 > checkpoint_minimum_epochs,
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
        diagnostics
    )
