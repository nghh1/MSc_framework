from __future__ import annotations

from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd

from garl_trading.backtest import run_portfolio
from garl_trading.garl import train_garl_ddal
from garl_trading.rl import (
    train_independent_a2c,
    train_independent_dqn,
    train_independent_ppo,
    train_joint_a2c,
    train_joint_dqn,
    train_joint_ppo,
)


def tune_rl_policy(
    name: str,
    features: dict[str, pd.DataFrame],
    closes: dict[str, pd.Series],
    trials: int,
    seed: int,
    levels: tuple[float, ...],
    lookback: int,
    final_epochs: int,
    learning_rate: float,
    gamma: float,
    cost_rate: float,
    initial_capital: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    embargo_bars: int,
    device: str = "auto",
    objective_metric: str = "sharpe",
) -> dict:
    """Tune RL settings on the latest causal inner validation segment."""
    n = len(next(iter(features.values())))
    split = max(lookback + 50, int(n * 0.8))
    train_end = max(lookback + 20, split - embargo_bars)
    train_positions = np.arange(0, train_end)
    validation_positions = np.arange(split, n)
    if not len(validation_positions):
        return {"rollout_length": 32, "learning_rate": learning_rate}

    trainers: dict[str, Callable] = {
        "single_a2c": train_joint_a2c,
        "single_ppo": train_joint_ppo,
        "single_dqn": train_joint_dqn,
        "independent_a2c": train_independent_a2c,
        "independent_ppo": train_independent_ppo,
        "independent_dqn": train_independent_dqn,
        "garl_ddal": train_garl_ddal,
    }
    trainer = trainers[name]
    train_features = {t: frame.iloc[train_positions] for t, frame in features.items()}
    train_closes = {t: series.iloc[train_positions] for t, series in closes.items()}
    validation_features = {t: frame.iloc[validation_positions] for t, frame in features.items()}
    validation_closes = {t: series.iloc[validation_positions] for t, series in closes.items()}
    context_positions = np.arange(max(0, split - lookback + 1), split)
    context = {t: frame.iloc[context_positions] for t, frame in features.items()}
    tune_epochs = max(10, min(30, final_epochs // 5))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "rollout_length": trial.suggest_categorical("rollout_length", [16, 32, 64]),
            "learning_rate": trial.suggest_float(
                "learning_rate", learning_rate / 3, learning_rate * 3, log=True
            ),
        }
        try:
            policy = trainer(
                train_features,
                train_closes,
                levels=levels,
                lookback=lookback,
                epochs=tune_epochs,
                gamma=gamma,
                cost_rate=cost_rate,
                seed=seed,
                device=device,
                **params,
            )
            positions = policy.positions(validation_features, context=context)
            result = run_portfolio(
                pd.DataFrame(validation_closes),
                positions,
                initial_capital,
                transaction_cost_bps,
                slippage_bps=slippage_bps,
            )
            score = result.metrics[objective_metric]
            return float(score) if np.isfinite(score) else -10.0
        except Exception:  # noqa: BLE001 - invalid trial configurations are penalised
            return -10.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study.best_params
