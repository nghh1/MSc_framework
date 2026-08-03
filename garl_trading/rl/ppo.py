from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from garl_trading.utils import resolve_torch_device

from .core import (
    ActorCritic,
    JointActorCritic,
    RewardEarlyStopper,
    TradingState,
    fit_feature_scalers,
    make_states,
)
from .trainers import RLPolicySet


def gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    next_value: float,
    gamma: float,
    gae_lambda: float
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        following = next_value if i == len(rewards) - 1 else values[i + 1]
        mask = 0.0 if dones[i] else 1.0
        delta = rewards[i] + gamma * following * mask - values[i]
        running = delta + gamma * gae_lambda * mask * running
        advantages[i] = running
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def asset_ppo_epoch(
    model: ActorCritic,
    state: TradingState,
    optimizer: torch.optim.Optimizer,
    rollout_length: int,
    gamma: float,
    rng: np.random.Generator,
    clip_epsilon: float,
    gae_lambda: float,
    update_epochs: int = 4
) -> dict[str, float]:
    device = next(model.parameters()).device
    observations, actions, rewards, values, old_log_probabilities, dones = [], [], [], [], [], []
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
        values.append(float(value.item()))
        old_log_probabilities.append(float(np.log(probabilities[action] + 1e-8)))
        dones.append(done)
        observation = state.reset() if done else next_observation

    with torch.no_grad():
        _, next_value = model(torch.tensor(observation, device=device).unsqueeze(0))
    advantages, returns = gae(rewards, values, dones, float(next_value.item()), gamma, gae_lambda)
    obs_tensor = torch.tensor(np.stack(observations), dtype=torch.float32, device=device)
    action_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    old_log_tensor = torch.tensor(old_log_probabilities, device=device)
    advantage_tensor = torch.tensor(advantages, device=device)
    advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
        advantage_tensor.std() + 1e-8
    )
    return_tensor = torch.tensor(returns, device=device)

    losses, entropies = [], []
    for _ in range(update_epochs):
        logits, predicted_values = model(obs_tensor)
        all_log_probabilities = torch.log_softmax(logits, dim=-1)
        log_probability = all_log_probabilities.gather(1, action_tensor[:, None]).squeeze(1)
        ratio = torch.exp(log_probability - old_log_tensor)
        unclipped = ratio * advantage_tensor
        clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage_tensor
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -(probabilities * all_log_probabilities).sum(axis=1).mean()
        loss = (
            -torch.minimum(unclipped, clipped).mean()
            + 0.5 * nn.functional.mse_loss(predicted_values, return_tensor)
            - 0.01 * entropy
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        entropies.append(float(entropy.item()))
    return {
        "training_reward": float(np.mean(rewards)),
        "loss": float(np.mean(losses)),
        "entropy": float(np.mean(entropies))
    }


def train_independent_ppo(
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
    clip_epsilon: float = 0.2,
    gae_lambda: float = 0.95,
    short_borrow_bps_annual: float = 0.0,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-4,
    minimum_train_epochs: int = 30
) -> RLPolicySet:
    device = resolve_torch_device(device)
    tickers = tuple(features)
    scalers = fit_feature_scalers(features)
    states = make_states(
        features, closes, scalers, levels=levels, lookback=lookback, cost_rate=cost_rate,
        short_borrow_bps_annual=short_borrow_bps_annual
    )
    models, optimizers, randoms = {}, {}, {}
    observation_size = features[tickers[0]].shape[1] * lookback + 1
    for i, ticker in enumerate(tickers):
        torch.manual_seed(seed + i)
        models[ticker] = ActorCritic(observation_size, len(levels)).to(device)
        optimizers[ticker] = torch.optim.Adam(models[ticker].parameters(), lr=learning_rate)
        randoms[ticker] = np.random.default_rng(seed + i)
    diagnostics = []
    stopper = RewardEarlyStopper(
        early_stopping_patience, early_stopping_min_delta, minimum_train_epochs
    )
    for epoch in range(epochs):
        epoch_rows = []
        for ticker in tickers:
            epoch_rows.append(
                asset_ppo_epoch(
                    models[ticker],
                    states[ticker],
                    optimizers[ticker],
                    rollout_length=rollout_length,
                    gamma=gamma,
                    rng=randoms[ticker],
                    clip_epsilon=clip_epsilon,
                    gae_lambda=gae_lambda
                )
            )
        row = {
            "epoch": epoch,
            **{
                key: float(np.mean([item[key] for item in epoch_rows]))
                for key in ("training_reward", "loss", "entropy")
            }
        }
        diagnostics.append(row)
        smoothed = float(np.mean([item["training_reward"] for item in diagnostics[-5:]]))
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


def train_joint_ppo(
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
    clip_epsilon: float = 0.2,
    gae_lambda: float = 0.95,
    short_borrow_bps_annual: float = 0.0,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-4,
    minimum_train_epochs: int = 30
) -> RLPolicySet:
    device = resolve_torch_device(device)
    tickers = tuple(features)
    scalers = fit_feature_scalers(features)
    states = make_states(
        features, closes, scalers, levels=levels, lookback=lookback, cost_rate=cost_rate,
        short_borrow_bps_annual=short_borrow_bps_annual
    )
    per_asset_size = features[tickers[0]].shape[1] * lookback + 1
    torch.manual_seed(seed)
    model = JointActorCritic(per_asset_size, len(tickers), len(levels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    current = {ticker: states[ticker].observation() for ticker in tickers}

    diagnostics = []
    stopper = RewardEarlyStopper(
        early_stopping_patience, early_stopping_min_delta, minimum_train_epochs
    )
    for epoch in range(epochs):
        observations, actions, rewards, values, old_logs, dones = [], [], [], [], [], []
        for _ in range(rollout_length):
            joint_observation = np.concatenate([current[ticker] for ticker in tickers])
            with torch.no_grad():
                logits, value = model(torch.tensor(joint_observation, device=device).unsqueeze(0))
                probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            selected = [
                int(rng.choice(len(levels), p=probabilities[i])) for i in range(len(tickers))
            ]
            step_rewards, step_dones = [], []
            for ticker, action in zip(tickers, selected):
                next_observation, reward, done = states[ticker].step(action)
                current[ticker] = states[ticker].reset() if done else next_observation
                step_rewards.append(reward)
                step_dones.append(done)
            observations.append(joint_observation)
            actions.append(selected)
            rewards.append(float(np.mean(step_rewards)))
            values.append(float(value.item()))
            old_logs.append(
                float(
                    sum(
                        np.log(probabilities[i, action] + 1e-8) for i, action in enumerate(selected)
                    )
                )
            )
            dones.append(any(step_dones))

        with torch.no_grad():
            bootstrap_observation = np.concatenate([current[ticker] for ticker in tickers])
            _, next_value = model(torch.tensor(bootstrap_observation, device=device).unsqueeze(0))
        advantages, returns = gae(
            rewards, values, dones, float(next_value.item()), gamma, gae_lambda
        )
        obs_tensor = torch.tensor(np.stack(observations), dtype=torch.float32, device=device)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=device)
        old_log_tensor = torch.tensor(old_logs, device=device)
        advantage_tensor = torch.tensor(advantages, device=device)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
            advantage_tensor.std() + 1e-8
        )
        return_tensor = torch.tensor(returns, device=device)
        update_losses, update_entropies = [], []
        for _ in range(4):
            logits, predicted_values = model(obs_tensor)
            log_probabilities = torch.log_softmax(logits, dim=-1)
            new_log = log_probabilities.gather(2, action_tensor[..., None]).squeeze(-1).sum(axis=1)
            ratio = torch.exp(new_log - old_log_tensor)
            objective = torch.minimum(
                ratio * advantage_tensor,
                torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage_tensor
            )
            probabilities = torch.softmax(logits, dim=-1)
            entropy = -(probabilities * log_probabilities).sum(axis=-1).mean()
            loss = (
                -objective.mean()
                + 0.5 * nn.functional.mse_loss(predicted_values, return_tensor)
                - 0.01 * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_losses.append(float(loss.item()))
            update_entropies.append(float(entropy.item()))
        diagnostics.append(
            {
                "epoch": epoch,
                "training_reward": float(np.mean(rewards)),
                "loss": float(np.mean(update_losses)),
                "entropy": float(np.mean(update_entropies))
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
