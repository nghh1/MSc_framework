from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from garl_trading.backtest.engine import run_portfolio
from garl_trading.models import ModelContext, create_forecaster
from garl_trading.validation.walk_forward import Fold

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _parameters(name: str, trial: optuna.Trial) -> dict:
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50, step=10),
        }
    if name in {"arimax_static", "arimax_rolling"}:
        values = {
            "p": trial.suggest_int("p", 0, 2),
            "d": trial.suggest_int("d", 0, 1),
            "q": trial.suggest_int("q", 0, 2),
            "trend": trial.suggest_categorical("trend", ["n", "c"]),
        }
        if name == "arimax_rolling":
            values.update(
                {
                    "window": trial.suggest_categorical("window", [126, 252, 504]),
                    "refit_every": trial.suggest_categorical("refit_every", [5, 10, 20]),
                }
            )
        return values
    if name in {"lstm", "tcn", "tft"}:
        return {
            "hidden": trial.suggest_categorical("hidden", [16, 32, 64]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "epochs": trial.suggest_int("epochs", 10, 30, step=10),
        }
    return {}


def tune_forecaster(
    name: str,
    features: pd.DataFrame,
    targets: pd.Series,
    closes: pd.Series,
    inner_folds: list[Fold],
    *,
    trials: int,
    seed: int,
    initial_capital: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = _parameters(name, trial)
        scores = []
        for fold in inner_folds:
            try:
                model = create_forecaster(name, seed=seed, **params)
                model.fit(features.iloc[fold.train], targets.iloc[fold.train])
                context_positions = np.arange(max(0, fold.test[0] - 19), fold.test[0])
                context = ModelContext(
                    features.iloc[context_positions],
                    targets.iloc[context_positions],
                )
                test_features = features.iloc[fold.test]
                test_targets = targets.iloc[fold.test]
                positions = model.predict_positions(
                    test_features,
                    context=context,
                    realized_targets=test_targets,
                )
                result = run_portfolio(
                    closes.iloc[fold.test].to_frame("asset"),
                    positions.to_frame("asset"),
                    initial_capital=initial_capital,
                    transaction_cost_bps=transaction_cost_bps,
                    slippage_bps=slippage_bps,
                )
                score = result.metrics["sharpe"]
                scores.append(score if np.isfinite(score) else -10.0)
            except Exception:  # noqa: BLE001 - failed parameter sets receive a penalty
                scores.append(-10.0)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study.best_params
