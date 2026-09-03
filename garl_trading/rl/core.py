from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from garl_trading.execution import (bounded_exposure, drifted_exposure, limited_net_return,
                                    thresholded_target)


def incremental_target(current: float, action: int, levels: np.ndarray) -> float:
    """Interpret {-1, 0, +1} as decrease, hold, and increase."""
    delta = float(levels[action])
    proposed = float(np.clip(current + delta, levels.min(), levels.max()))
    return float(levels[np.argmin(np.abs(levels - proposed))])


class RewardEarlyStopper:
    def __init__(self, patience: int, min_delta: float, minimum_epochs: int):
        self.patience = patience
        self.min_delta = min_delta
        self.minimum_epochs = minimum_epochs
        self.best = -np.inf
        self.bad_epochs = 0
        self.best_state = None
        self.best_epoch: int | None = None
        self.stop_epoch: int | None = None

    def update(self, epoch: int, reward: float, models, *, checkpoint_eligible: bool = True) -> bool:
        if self.patience == 0:
            return False
        completed_epochs = epoch + 1
        if completed_epochs <= self.minimum_epochs or not checkpoint_eligible:
            self.bad_epochs = 0
            return False

        improved = reward > self.best + self.min_delta
        if improved:
            self.best = reward
            self.best_epoch = epoch

            if isinstance(models, dict):
                self.best_state = {
                    name: deepcopy(model.state_dict())
                    for name, model in models.items()
                }
            else:
                self.best_state = deepcopy(models.state_dict())
        if improved:
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        should_stop = self.bad_epochs >= self.patience
        if should_stop:
            self.stop_epoch = epoch
        return should_stop

    def restore(self, models) -> None:
        if self.patience == 0:
            return
        if self.best_state is None:
            return

        if isinstance(models, dict):
            for name, model in models.items():
                model.load_state_dict(self.best_state[name])
        else:
            models.load_state_dict(self.best_state)


class CausalTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int,
                 dropout: float) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        causal = nn.functional.pad(values, (self.left_padding, 0))
        return self.dropout(self.activation(self.conv(causal) + self.residual(values)))


class TemporalFeatureExtractor(nn.Module):
    """Causal TCN over the market window, followed by the current portfolio position."""

    def __init__(self, observation_size: int, lookback: int, channels: int = 32,
                 kernel_size: int = 3, dilations: tuple[int, ...] = (1, 2, 4, 8),
                 dropout: float = 0.0) -> None:
        super().__init__()
        market_size = observation_size - 1
        if lookback < 2 or market_size <= 0 or market_size % lookback:
            raise ValueError("Observation size must contain a complete temporal window plus position.")
        if channels < 1 or kernel_size < 2 or not dilations or any(value < 1 for value in dilations):
            raise ValueError("Invalid causal TCN encoder configuration.")
        self.observation_size = observation_size
        self.lookback = lookback
        self.feature_count = market_size // lookback
        self.channels = channels
        blocks = []
        in_channels = self.feature_count
        for dilation in dilations:
            blocks.append(
                CausalTemporalBlock(
                    in_channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout))
            in_channels = channels
        self.network = nn.Sequential(*blocks)
        self.output_size = channels + 1

    def temporal_states(self, market_window: torch.Tensor) -> torch.Tensor:
        if market_window.shape[-2:] != (self.lookback, self.feature_count):
            raise ValueError("Unexpected market-window shape for the temporal encoder.")
        return self.network(market_window.transpose(-1, -2))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.observation_size:
            raise ValueError("Unexpected observation width for the temporal encoder.")
        market = observation[..., :-1].reshape(-1, self.lookback, self.feature_count)
        temporal = self.temporal_states(market)[..., -1]
        position = observation[..., -1:].reshape(-1, 1)
        return torch.cat([temporal, position], dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, actions: int, hidden: int = 64, *, lookback: int = 20,
                 encoder_channels: int = 32, encoder_kernel_size: int = 3,
                 encoder_dilations: tuple[int, ...] = (1, 2, 4, 8), encoder_dropout: float = 0.0) -> None:
        super().__init__()
        self.extractor = TemporalFeatureExtractor(
            observation_size,
            lookback,
            channels=encoder_channels,
            kernel_size=encoder_kernel_size,
            dilations=encoder_dilations,
            dropout=encoder_dropout)
        self.body = nn.Sequential(
            nn.Linear(self.extractor.output_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh())
        self.policy = nn.Linear(hidden, actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, observation):
        hidden = self.body(self.extractor(observation))
        return self.policy(hidden), self.value(hidden).squeeze(-1)


class JointActorCritic(nn.Module):
    def __init__(self, per_asset_size: int, assets: int, actions: int, hidden: int = 128, *,
                 lookback: int = 20, encoder_channels: int = 32, encoder_kernel_size: int = 3,
                 encoder_dilations: tuple[int, ...] = (1, 2, 4, 8), encoder_dropout: float = 0.0) -> None:
        super().__init__()
        self.assets = assets
        self.per_asset_size = per_asset_size
        self.extractor = TemporalFeatureExtractor(
            per_asset_size,
            lookback,
            channels=encoder_channels,
            kernel_size=encoder_kernel_size,
            dilations=encoder_dilations,
            dropout=encoder_dropout)
        self.body = nn.Sequential(
            nn.Linear(self.extractor.output_size * assets, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh())
        self.policies = nn.ModuleList([nn.Linear(hidden, actions) for _ in range(assets)])
        self.value = nn.Linear(hidden, 1)

    def forward(self, observation):
        asset_observations = observation.reshape(-1, self.assets, self.per_asset_size)
        encoded = self.extractor(asset_observations.reshape(-1, self.per_asset_size))
        encoded = encoded.reshape(-1, self.assets * self.extractor.output_size)
        hidden = self.body(encoded)
        logits = torch.stack([head(hidden) for head in self.policies], dim=1)
        return logits, self.value(hidden).squeeze(-1)


@dataclass
class TradingState:
    features: np.ndarray
    closes: np.ndarray
    levels: np.ndarray
    lookback: int
    cost_rate: float
    short_borrow_rate: float
    rebalance_threshold: float
    decision_interval: int
    cursor: int
    position: float = 0.0
    target_position: float = 0.0

    @classmethod
    def create(cls, features: np.ndarray, closes: np.ndarray, levels: np.ndarray, lookback: int,
               cost_rate: float, short_borrow_rate: float, rebalance_threshold: float = 0.0,
               decision_interval: int = 1) -> TradingState:
        return cls(
            features,
            closes,
            levels,
            lookback,
            cost_rate,
            short_borrow_rate,
            rebalance_threshold,
            decision_interval,
            lookback - 1)

    def observation(self) -> np.ndarray:
        window = self.features[self.cursor - self.lookback + 1 : self.cursor + 1]
        return np.concatenate([window.reshape(-1), [self.target_position]]).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        desired = incremental_target(self.target_position, action, self.levels)
        proposed = thresholded_target(desired, self.position, self.rebalance_threshold)
        target = float(bounded_exposure(proposed))
        initial_turnover_cost = abs(target - self.position) * self.cost_rate
        exposure = target
        wealth_factor = 1.0
        end = min(self.cursor + self.decision_interval, len(self.closes) - 1)
        for next_cursor in range(self.cursor + 1, end + 1):
            held = float(bounded_exposure(exposure))
            forced_turnover_cost = abs(held - exposure) * self.cost_rate
            next_return = self.closes[next_cursor] / self.closes[next_cursor - 1] - 1
            borrow_cost = max(-held, 0.0) * self.short_borrow_rate
            net_return = held * next_return - borrow_cost - forced_turnover_cost
            if next_cursor == self.cursor + 1:
                net_return -= initial_turnover_cost
            net_return = float(limited_net_return(net_return))
            wealth_factor *= 1.0 + net_return
            gross_return = held * next_return
            exposure = (
                float(drifted_exposure(held, next_return, gross_return))
                if net_return > -1.0
                else 0.0
            )
            if not np.isfinite(wealth_factor):
                raise FloatingPointError("Non-finite wealth in RL transition.")
            if wealth_factor <= 0:
                exposure = 0.0
                break
        reward = wealth_factor - 1.0
        self.position = float(exposure)
        self.target_position = desired
        self.cursor = end
        done = self.cursor >= len(self.closes) - 1
        if done:
            return (
                np.zeros(self.features.shape[1] * self.lookback + 1, dtype=np.float32),
                float(reward),
                True,
            )
        return self.observation(), float(reward), False

    def reset(self) -> np.ndarray:
        self.cursor = self.lookback - 1
        self.position = 0.0
        self.target_position = 0.0
        return self.observation()


def fit_feature_scalers(features: dict[str, pd.DataFrame]) -> dict[str, StandardScaler]:
    return {ticker: StandardScaler().fit(frame) for ticker, frame in features.items()}


def initialise_asset_actor_critics(tickers: tuple[str, ...], observation_size: int, actions: int,
                                   seed: int, device: torch.device, *, lookback: int,
                                   encoder_channels: int = 32, encoder_kernel_size: int = 3,
                                   encoder_dilations: tuple[int, ...] = (1, 2, 4, 8),
                                   encoder_dropout: float = 0.0) -> dict[str, ActorCritic]:
    """Create separate stock agents from one common parameter template.

    The common start preserves parameter correspondence for raw GARL gradient transfer.
    Returned models remain independent objects with separate subsequent updates.
    """
    torch.manual_seed(seed)
    template = ActorCritic(
        observation_size,
        actions,
        lookback=lookback,
        encoder_channels=encoder_channels,
        encoder_kernel_size=encoder_kernel_size,
        encoder_dilations=encoder_dilations,
        encoder_dropout=encoder_dropout).to(device)
    template_state = template.state_dict()
    models = {}
    for ticker in tickers:
        model = ActorCritic(
            observation_size,
            actions,
            lookback=lookback,
            encoder_channels=encoder_channels,
            encoder_kernel_size=encoder_kernel_size,
            encoder_dilations=encoder_dilations,
            encoder_dropout=encoder_dropout).to(device)
        model.load_state_dict(template_state)
        models[ticker] = model
    return models


def make_states(features: dict[str, pd.DataFrame], closes: dict[str, pd.Series],
                scalers: dict[str, StandardScaler], levels: tuple[float, ...], lookback: int,
                cost_rate: float, turnover_penalty_multiplier: float = 1.0,
                short_borrow_bps_annual: float = 0.0, rebalance_threshold: float = 0.0,
                decision_interval: int = 1) -> dict[str, TradingState]:
    if turnover_penalty_multiplier < 1.0:
        raise ValueError("turnover_penalty_multiplier must be at least 1.0.")
    return {
        ticker: TradingState.create(
            scalers[ticker].transform(frame).astype(np.float32),
            closes[ticker].reindex(frame.index).to_numpy(dtype=np.float32),
            np.asarray(levels, dtype=np.float32),
            lookback,
            cost_rate * turnover_penalty_multiplier,
            short_borrow_bps_annual / 10000 / 252,
            rebalance_threshold,
            decision_interval)
        for ticker, frame in features.items()
    }


def a2c_gradient(model: ActorCritic, state: TradingState, rollout_length: int, gamma: float,
                 rng: np.random.Generator, gae_lambda: float = 0.95, entropy_weight: float = 0.01,
                 value_weight: float = 0.5) -> tuple[list[torch.Tensor], float, float]:
    device = next(model.parameters()).device
    observations, actions, rewards, rollout_values, dones = [], [], [], [], []
    observation = state.observation()
    for _ in range(rollout_length):
        with torch.no_grad():
            logits, value = model(torch.tensor(observation, device=device).unsqueeze(0))
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        action = int(rng.choice(len(probabilities), p=probabilities))
        next_observation, reward, done = state.step(action)
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        rollout_values.append(float(value.item()))
        dones.append(done)
        observation = state.reset() if done else next_observation

    obs_tensor = torch.tensor(np.stack(observations), dtype=torch.float32, device=device)
    action_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    logits, predicted_values = model(obs_tensor)
    with torch.no_grad():
        _, bootstrap = model(torch.tensor(observation, device=device).unsqueeze(0))
    advantages = np.zeros(len(rewards), dtype=np.float32)
    running_advantage = 0.0
    next_value = float(bootstrap.item())
    for i in reversed(range(len(rewards))):
        mask = 0.0 if dones[i] else 1.0
        following_value = next_value if i == len(rewards) - 1 else rollout_values[i + 1]
        delta = rewards[i] + gamma * following_value * mask - rollout_values[i]
        running_advantage = delta + gamma * gae_lambda * mask * running_advantage
        advantages[i] = running_advantage
    returns = advantages + np.asarray(rollout_values, dtype=np.float32)
    return_tensor = torch.tensor(returns, device=device)
    advantage_tensor = torch.tensor(advantages, device=device)
    advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
        advantage_tensor.std(unbiased=False) + 1e-8
    )
    log_probabilities = torch.log_softmax(logits, dim=-1)
    chosen = log_probabilities.gather(1, action_tensor[:, None]).squeeze(1)
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -(probabilities * log_probabilities).sum(axis=1).mean()
    loss = (
        -(chosen * advantage_tensor).mean()
        + value_weight * nn.functional.mse_loss(predicted_values, return_tensor)
        - entropy_weight * entropy
    )
    model.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    gradients = [
        parameter.grad.detach().clone()
        if parameter.grad is not None
        else torch.zeros_like(parameter)
        for parameter in model.parameters()
    ]
    return gradients, float(loss.item()), float(np.mean(rewards))


def apply_gradient(model: nn.Module, optimizer: torch.optim.Optimizer, gradients: list[torch.Tensor]) -> None:
    optimizer.zero_grad()
    for parameter, gradient in zip(model.parameters(), gradients):
        parameter.grad = gradient.clone()
    optimizer.step()


@torch.no_grad()
def greedy_asset_positions(model: nn.Module, scaler: StandardScaler, features: pd.DataFrame,
                           context: pd.DataFrame, levels: tuple[float, ...], lookback: int,
                           closes: pd.Series | None = None, cost_rate: float = 0.0,
                           short_borrow_rate: float = 0.0, rebalance_threshold: float = 0.0,
                           decision_interval: int = 1) -> pd.Series:
    model.eval()
    device = next(model.parameters()).device
    combined = pd.concat([context, features])
    combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
    values = scaler.transform(combined).astype(np.float32)
    location = {date: i for i, date in enumerate(combined.index)}
    current_target = 0.0
    output = {}
    dates = list(features.index)
    closes = closes.reindex(features.index) if closes is not None else None
    for step, date in enumerate(dates):
        i = location[date]
        start = max(0, i - lookback + 1)
        window = values[start : i + 1]
        if len(window) < lookback:
            window = np.concatenate([np.repeat(window[:1], lookback - len(window), axis=0), window])
        if step % decision_interval == 0:
            observation = np.concatenate([window.reshape(-1), [current_target]]).astype(
                np.float32)
            network_output = model(torch.tensor(observation, device=device).unsqueeze(0))
            logits = network_output[0] if isinstance(network_output, tuple) else network_output
            action = int(torch.argmax(logits, dim=-1).item())
            current_target = incremental_target(
                current_target, action, np.asarray(levels, dtype=float))
        output[date] = current_target
    return pd.Series(output)
