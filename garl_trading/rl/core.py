from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, actions: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        self.policy = nn.Linear(hidden, actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, observation):
        hidden = self.body(observation)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


class JointActorCritic(nn.Module):
    def __init__(self, per_asset_size: int, assets: int, actions: int, hidden: int = 128):
        super().__init__()
        self.assets = assets
        self.body = nn.Sequential(
            nn.Linear(per_asset_size * assets, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        self.policies = nn.ModuleList([nn.Linear(hidden, actions) for _ in range(assets)])
        self.value = nn.Linear(hidden, 1)

    def forward(self, observation):
        hidden = self.body(observation)
        logits = torch.stack([head(hidden) for head in self.policies], dim=1)
        return logits, self.value(hidden).squeeze(-1)


@dataclass
class TradingState:
    features: np.ndarray
    closes: np.ndarray
    levels: np.ndarray
    lookback: int
    cost_rate: float
    cursor: int
    position: float = 0.0

    @classmethod
    def create(cls, features: np.ndarray, closes: np.ndarray, levels: np.ndarray, 
               lookback: int, cost_rate: float) -> TradingState:
        return cls(features, closes, levels, lookback, cost_rate, lookback - 1)

    def observation(self) -> np.ndarray:
        window = self.features[self.cursor - self.lookback + 1:self.cursor + 1]
        return np.concatenate([window.reshape(-1), [self.position]]).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        target = float(self.levels[action])
        next_return = self.closes[self.cursor + 1] / self.closes[self.cursor] - 1
        reward = target * next_return - abs(target - self.position) * self.cost_rate
        self.position = target
        self.cursor += 1
        done = self.cursor >= len(self.closes) - 1
        if done:
            return np.zeros_like(self.observation()), float(reward), True
        return self.observation(), float(np.log(max(1 + reward, 1e-8))), False

    def reset(self) -> np.ndarray:
        self.cursor = self.lookback - 1
        self.position = 0.0
        return self.observation()


def fit_feature_scalers(features: dict[str, pd.DataFrame]) -> dict[str, StandardScaler]:
    return {ticker: StandardScaler().fit(frame) for ticker, frame in features.items()}


def make_states(features: dict[str, pd.DataFrame], closes: dict[str, pd.Series], 
                scalers: dict[str, StandardScaler], levels: tuple[float, ...], 
                lookback: int, cost_rate: float) -> dict[str, TradingState]:
    return {
        ticker: TradingState.create(
            scalers[ticker].transform(frame).astype(np.float32),
            closes[ticker].reindex(frame.index).to_numpy(dtype=np.float32),
            np.asarray(levels, dtype=np.float32),
            lookback,
            cost_rate
        )
        for ticker, frame in features.items()
    }


def a2c_gradient(model: ActorCritic, state: TradingState, rollout_length: int, gamma: float, 
                 rng: np.random.Generator, entropy_weight: float = 0.01, value_weight: float = 0.5) -> tuple[list[torch.Tensor], float]:
    device = next(model.parameters()).device
    observations, actions, rewards, dones = [], [], [], []
    observation = state.observation()
    for _ in range(rollout_length):
        with torch.no_grad():
            logits, _ = model(torch.tensor(observation, device=device).unsqueeze(0))
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        action = int(rng.choice(len(probabilities), p=probabilities))
        next_observation, reward, done = state.step(action)
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        observation = state.reset() if done else next_observation

    obs_tensor = torch.tensor(np.stack(observations), dtype=torch.float32, device=device)
    action_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    logits, values = model(obs_tensor)
    with torch.no_grad():
        _, bootstrap = model(torch.tensor(observation, device=device).unsqueeze(0))
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = float(bootstrap.item())
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running * (not dones[i])
        returns[i] = running
    return_tensor = torch.tensor(returns, device=device)
    advantages = return_tensor - values
    log_probabilities = torch.log_softmax(logits, dim=-1)
    chosen = log_probabilities.gather(1, action_tensor[:, None]).squeeze(1)
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -(probabilities * log_probabilities).sum(axis=1).mean()
    loss = (-(chosen * advantages.detach()).mean() + value_weight * 
            nn.functional.mse_loss(values, return_tensor) - entropy_weight * entropy)
    model.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    gradients = [
        parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros_like(parameter)
        for parameter in model.parameters()
    ]
    return gradients, float(loss.item())


def apply_gradient(model: nn.Module, optimizer: torch.optim.Optimizer, gradients: list[torch.Tensor]) -> None:
    optimizer.zero_grad()
    for parameter, gradient in zip(model.parameters(), gradients):
        parameter.grad = gradient.clone()
    optimizer.step()


@torch.no_grad()
def greedy_asset_positions(model: nn.Module, scaler: StandardScaler, features: pd.DataFrame, 
                           context: pd.DataFrame, levels: tuple[float, ...], lookback: int) -> pd.Series:
    device = next(model.parameters()).device
    combined = pd.concat([context, features])
    combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
    values = scaler.transform(combined).astype(np.float32)
    location = {date: i for i, date in enumerate(combined.index)}
    current = 0.0
    output = {}
    for date in features.index:
        i = location[date]
        start = max(0, i - lookback + 1)
        window = values[start:i + 1]
        if len(window) < lookback:
            window = np.concatenate([np.repeat(window[:1], lookback - len(window), axis=0), window])
        observation = np.concatenate([window.reshape(-1), [current]]).astype(np.float32)
        network_output = model(torch.tensor(observation, device=device).unsqueeze(0))
        logits = network_output[0] if isinstance(network_output, tuple) else network_output
        action = int(torch.argmax(logits, dim=-1).item())
        current = float(levels[action])
        output[date] = current
    return pd.Series(output)
