from __future__ import annotations

from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd

from garl_trading.backtest import run_portfolio
from garl_trading.garl import train_garl_ddal, train_selective_garl_ddal
from garl_trading.rl import (
    train_independent_a2c,
    train_independent_dqn,
    train_independent_ppo,
    train_joint_a2c,
    train_joint_dqn,
    train_joint_ppo,
)


def rl_candidate_profiles(name: str, learning_rate: float) -> list[dict[str, float | int]]:
    """Nine predeclared, compute-bounded profiles for each RL algorithm family."""
    rates = [learning_rate / 2, learning_rate, learning_rate * 2]
    if name.endswith("a2c"):
        entropy = [0.005, 0.01, 0.02]
        return [
            {
                "learning_rate": rates[index % 3],
                "entropy_weight": entropy[index // 3],
                "turnover_penalty_multiplier": 1.0,
            }
            for index in range(9)
        ]
    if name.endswith("ppo"):
        clipping = [0.1, 0.2, 0.3]
        return [
            {
                "learning_rate": rates[index % 3],
                "clip_epsilon": clipping[index // 3],
                "turnover_penalty_multiplier": 1.0,
            }
            for index in range(9)
        ]
    if name.endswith("dqn"):
        exploration = [0.3, 0.5, 0.7]
        target_intervals = [5, 10, 20]
        return [
            {
                "learning_rate": rates[index % 3],
                "epsilon_decay_fraction": exploration[index // 3],
                "target_update_interval": target_intervals[(index + index // 3) % 3],
                "turnover_penalty_multiplier": 1.0,
            }
            for index in range(9)
        ]
    if name == "garl_ddal":
        entropy = [0.005, 0.01, 0.02]
        return [
            {
                "learning_rate": rates[index % 3],
                "entropy_weight": entropy[index // 3],
                "pool_size": 3,
                "turnover_penalty_multiplier": 1.0,
            }
            for index in range(9)
        ]
    if name == "selective_garl_ddal":
        thresholds = [0.0, 0.05, 0.1]
        return [
            {
                "learning_rate": rates[index % 3],
                "entropy_weight": 0.01,
                "alignment_threshold": thresholds[index // 3],
                "peer_mix": 0.5,
                "pool_size": 3,
                "turnover_penalty_multiplier": 1.0,
            }
            for index in range(9)
        ]
    raise KeyError(name)


def tune_rl_policy(
    name: str,
    features: dict[str, pd.DataFrame],
    closes: dict[str, pd.Series],
    trials: int,
    seed: int,
    levels: tuple[float, ...],
    lookback: int,
    rollout_length: int,
    final_epochs: int,
    learning_rate: float,
    gamma: float,
    cost_rate: float,
    initial_capital: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    short_borrow_bps_annual: float,
    embargo_bars: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    minimum_train_epochs: int,
    garl_share_after_fraction: float,
    garl_share_every: int,
    garl_pool_size: int,
    selective_garl_alignment_threshold: float,
    selective_garl_peer_mix: float,
    encoder_channels: int = 32,
    encoder_kernel_size: int = 3,
    encoder_dilations: tuple[int, ...] = (1, 2, 4, 8),
    encoder_dropout: float = 0.0,
    device: str = "auto",
    objective_metric: str = "sharpe"
) -> dict:
    """Tune RL settings on the latest causal inner validation segment."""
    n = len(next(iter(features.values())))
    split = max(lookback + 50, int(n * 0.8))
    train_end = max(lookback + 20, split - embargo_bars)
    train_positions = np.arange(0, train_end)
    validation_positions = np.arange(split, n)
    if not len(validation_positions):
        return {"learning_rate": learning_rate}

    trainers: dict[str, Callable] = {
        "single_a2c": train_joint_a2c,
        "single_ppo": train_joint_ppo,
        "single_dqn": train_joint_dqn,
        "independent_a2c": train_independent_a2c,
        "independent_ppo": train_independent_ppo,
        "independent_dqn": train_independent_dqn,
        "garl_ddal": train_garl_ddal,
        "selective_garl_ddal": train_selective_garl_ddal,
    }
    trainer = trainers[name]
    train_features = {t: frame.iloc[train_positions] for t, frame in features.items()}
    train_closes = {t: series.iloc[train_positions] for t, series in closes.items()}
    validation_features = {t: frame.iloc[validation_positions] for t, frame in features.items()}
    validation_closes = {t: series.iloc[validation_positions] for t, series in closes.items()}
    context_positions = np.arange(max(0, split - lookback + 1), split)
    context = {t: frame.iloc[context_positions] for t, frame in features.items()}
    tune_epochs = max(10, min(30, final_epochs // 5))

    candidates = rl_candidate_profiles(name, learning_rate)

    def objective(trial: optuna.Trial) -> float:
        profile = int(trial.suggest_categorical("profile", list(range(len(candidates)))))
        params = candidates[profile]
        try:
            algorithm_parameters = {}
            if name in {"garl_ddal", "selective_garl_ddal"}:
                algorithm_parameters = {
                    "share_after_fraction": garl_share_after_fraction,
                    "share_every": garl_share_every,
                    "pool_size": garl_pool_size or None
                }
                if name == "selective_garl_ddal":
                    algorithm_parameters["alignment_threshold"] = (
                        selective_garl_alignment_threshold
                    )
                    algorithm_parameters["peer_mix"] = selective_garl_peer_mix
            algorithm_parameters.update(params)
            policy = trainer(
                train_features,
                train_closes,
                levels=levels,
                lookback=lookback,
                epochs=tune_epochs,
                rollout_length=rollout_length,
                gamma=gamma,
                cost_rate=cost_rate,
                seed=seed,
                device=device,
                encoder_channels=encoder_channels,
                encoder_kernel_size=encoder_kernel_size,
                encoder_dilations=encoder_dilations,
                encoder_dropout=encoder_dropout,
                short_borrow_bps_annual=short_borrow_bps_annual,
                early_stopping_patience=min(early_stopping_patience, tune_epochs),
                early_stopping_min_delta=early_stopping_min_delta,
                minimum_train_epochs=min(minimum_train_epochs, tune_epochs),
                **algorithm_parameters,
            )
            positions = policy.positions(
                validation_features,
                context=context,
                closes=validation_closes
            )
            result = run_portfolio(
                pd.DataFrame(validation_closes),
                positions,
                initial_capital,
                transaction_cost_bps,
                slippage_bps=slippage_bps,
                short_borrow_bps_annual=short_borrow_bps_annual
            )
            score = result.metrics[objective_metric]
            return float(score) if np.isfinite(score) else -10.0
        except Exception:  # noqa: BLE001 - invalid trial configurations are penalised
            return -10.0

    search_space = {"profile": list(range(len(candidates)))}
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler(search_space, seed=seed),
    )
    study.optimize(
        objective,
        n_trials=min(trials, len(search_space["profile"])),
        show_progress_bar=False,
    )
    return dict(candidates[int(study.best_params["profile"])])
