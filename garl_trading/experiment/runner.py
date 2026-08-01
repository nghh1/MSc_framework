from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from garl_trading.backtest import run_portfolio
from garl_trading.config import FrameworkConfig
from garl_trading.data import build_dataset, load_market_data
from garl_trading.data.features import FEATURE_COLUMNS
from garl_trading.garl import train_garl_ddal
from garl_trading.models import ModelContext, create_forecaster
from garl_trading.rl import (
    train_independent_a2c,
    train_independent_dqn,
    train_independent_ppo,
    train_joint_a2c,
    train_joint_dqn,
    train_joint_ppo,
)
from garl_trading.tuning import tune_forecaster, tune_rl_policy
from garl_trading.validation import nested_folds, outer_folds

from .artifacts import ArtifactStore

LOGGER = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, config: FrameworkConfig, config_path: str | Path) -> None:
        self.config = config
        self.config_path = Path(config_path)
        self.rl_parameters: dict[tuple[str, int], dict] = {}

    def run(self) -> Path:
        cfg = self.config
        tickers = cfg.data.tickers
        raw = load_market_data(
            tickers, cfg.data.start, cfg.data.end, adjust_prices=cfg.data.adjust_prices
        )
        dataset = build_dataset(raw)
        min_train = min(cfg.validation.min_train_bars, max(252, len(dataset.index) // 3))
        folds, holdout = outer_folds(
            dataset.index,
            n_folds=cfg.validation.outer_folds,
            min_train_bars=min_train,
            max_train_bars=cfg.validation.max_train_bars,
            embargo=cfg.validation.embargo_bars,
            holdout_start=cfg.validation.final_holdout_start,
            use_holdout=cfg.validation.use_final_holdout,
        )
        evaluation_folds = folds + ([holdout] if holdout is not None else [])
        store = ArtifactStore(cfg.experiment.artifacts_dir, cfg.experiment.name)
        store.initialise(cfg, self.config_path)
        store.save_market_data(dataset.prices)

        supervised = cfg.models.supervised
        rl_names = cfg.models.rl
        repetitions = cfg.experiment.repetitions

        for fold in evaluation_folds:
            self.run_buy_hold(dataset, fold, store)
            for name in supervised:
                self.run_supervised(name, dataset, fold, store)
            for name in rl_names:
                for repetition in range(repetitions):
                    seed = cfg.experiment.seed + repetition
                    self.run_rl(name, dataset, fold, repetition, seed, store)
            store.flush()

        from garl_trading.reporting.visualise import build_report

        build_report(
            store.path,
            confidence=cfg.reporting.confidence_level,
            formats=cfg.reporting.formats,
        )
        return store.path

    def frames(self, dataset, positions):
        features = {
            ticker: dataset.features[ticker].iloc[positions].loc[:, FEATURE_COLUMNS]
            for ticker in dataset.tickers
        }
        closes = {
            ticker: dataset.prices[ticker]["close"].iloc[positions] for ticker in dataset.tickers
        }
        return features, closes

    def backtest(self, closes: dict[str, pd.Series], positions: pd.DataFrame):
        cfg = self.config.execution
        close_frame = pd.DataFrame(closes)
        return run_portfolio(
            close_frame,
            positions,
            initial_capital=cfg.initial_capital,
            transaction_cost_bps=cfg.transaction_cost_bps,
            slippage_bps=cfg.slippage_bps,
            short_borrow_bps_annual=cfg.short_borrow_bps_annual,
        )

    def metadata(self, name, fold, repetition, seed):
        return {
            "baseline": name,
            "fold": fold.number,
            "fold_kind": fold.kind,
            "repetition": repetition,
            "seed": seed,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
        }

    def run_buy_hold(self, dataset, fold, store):
        _, closes = self.frames(dataset, fold.test)
        positions = pd.DataFrame(1.0, index=dataset.index[fold.test], columns=dataset.tickers)
        result = self.backtest(closes, positions)
        store.add_result(
            metadata=self.metadata("buy_and_hold", fold, 0, self.config.experiment.seed),
            metrics=result.metrics,
            positions=positions,
            equity=result.equity,
            returns=result.returns,
            gross_returns=result.gross_returns,
            costs=result.costs,
            turnover=result.turnover,
        )

    def run_supervised(self, name, dataset, fold, store):
        cfg = self.config
        train_features, _ = self.frames(dataset, fold.train)
        test_features, test_closes = self.frames(dataset, fold.test)
        context_positions = np.arange(
            max(0, int(fold.test[0]) - cfg.models.lookback + 1), int(fold.test[0])
        )
        context_features, _ = self.frames(dataset, context_positions)
        position_columns = {}
        try:
            for ticker in dataset.tickers:
                targets = dataset.features[ticker]["target_return"]
                parameters = {}
                if cfg.tuning.enabled:
                    inner = nested_folds(
                        fold.train,
                        dataset.index,
                        n_folds=cfg.validation.inner_folds,
                        min_train_bars=min(
                            cfg.validation.min_train_bars, max(126, len(fold.train) // 3)
                        ),
                        max_train_bars=cfg.validation.max_train_bars,
                        embargo=cfg.validation.embargo_bars,
                    )
                    parameters = tune_forecaster(
                        name,
                        dataset.features[ticker].loc[:, FEATURE_COLUMNS],
                        targets,
                        dataset.prices[ticker]["close"],
                        inner,
                        trials=cfg.tuning.trials,
                        seed=cfg.experiment.seed,
                        initial_capital=cfg.execution.initial_capital / len(dataset.tickers),
                        transaction_cost_bps=cfg.execution.transaction_cost_bps,
                        slippage_bps=cfg.execution.slippage_bps,
                        objective_metric=cfg.tuning.objective,
                    )
                if name in {"lstm", "tcn", "tft"}:
                    parameters.setdefault("lookback", cfg.models.lookback)
                    parameters.setdefault("device", cfg.models.device)

                model = create_forecaster(name, seed=cfg.experiment.seed, **parameters)
                model.fit(train_features[ticker], targets.iloc[fold.train])
                context = ModelContext(context_features[ticker], targets.iloc[context_positions])
                position_columns[ticker] = model.predict_positions(
                    test_features[ticker],
                    context=context,
                    realised_targets=targets.iloc[fold.test],
                )
            positions = pd.DataFrame(position_columns).reindex(dataset.index[fold.test])
            result = self.backtest(test_closes, positions)
            store.add_result(
                metadata=self.metadata(name, fold, 0, cfg.experiment.seed),
                metrics=result.metrics,
                positions=positions,
                equity=result.equity,
                returns=result.returns,
                gross_returns=result.gross_returns,
                costs=result.costs,
                turnover=result.turnover,
            )
        except Exception as exc:
            LOGGER.exception("%s failed on fold %s", name, fold.number)
            store.add_failure(baseline=name, fold=fold.number, repetition=0, error=repr(exc))

    def run_rl(self, name, dataset, fold, repetition, seed, store):
        cfg = self.config
        train_features, train_closes = self.frames(dataset, fold.train)
        test_features, test_closes = self.frames(dataset, fold.test)
        context_positions = np.arange(
            max(0, int(fold.test[0]) - cfg.models.lookback + 1), int(fold.test[0])
        )
        context_features, _ = self.frames(dataset, context_positions)
        common = {
            "levels": cfg.execution.position_levels,
            "lookback": cfg.models.lookback,
            "epochs": cfg.models.train_epochs,
            "rollout_length": cfg.models.rollout_length,
            "learning_rate": cfg.models.learning_rate,
            "gamma": cfg.models.gamma,
            "cost_rate": (cfg.execution.transaction_cost_bps + cfg.execution.slippage_bps) / 10000,
            "seed": seed,
            "device": cfg.models.device,
        }
        try:
            tuning_key = (name, fold.number)
            if cfg.tuning.enabled:
                if tuning_key not in self.rl_parameters:
                    self.rl_parameters[tuning_key] = tune_rl_policy(
                        name,
                        train_features,
                        train_closes,
                        trials=cfg.tuning.rl_trials,
                        seed=cfg.experiment.seed,
                        levels=cfg.execution.position_levels,
                        lookback=cfg.models.lookback,
                        final_epochs=cfg.models.train_epochs,
                        learning_rate=cfg.models.learning_rate,
                        gamma=cfg.models.gamma,
                        cost_rate=common["cost_rate"],
                        initial_capital=cfg.execution.initial_capital,
                        transaction_cost_bps=cfg.execution.transaction_cost_bps,
                        slippage_bps=cfg.execution.slippage_bps,
                        embargo_bars=cfg.validation.embargo_bars,
                        device=cfg.models.device,
                        objective_metric=cfg.tuning.objective,
                    )
                common.update(self.rl_parameters[tuning_key])
            if name == "single_a2c":
                policy = train_joint_a2c(train_features, train_closes, **common)
            elif name == "single_ppo":
                policy = train_joint_ppo(train_features, train_closes, **common)
            elif name == "single_dqn":
                policy = train_joint_dqn(train_features, train_closes, **common)
            elif name == "independent_a2c":
                policy = train_independent_a2c(train_features, train_closes, **common)
            elif name == "independent_ppo":
                policy = train_independent_ppo(train_features, train_closes, **common)
            elif name == "independent_dqn":
                policy = train_independent_dqn(train_features, train_closes, **common)
            elif name == "garl_ddal":
                policy = train_garl_ddal(train_features, train_closes, **common)
            else:
                raise KeyError(name)
            positions = policy.positions(test_features, context=context_features)
            result = self.backtest(test_closes, positions)
            store.add_result(
                metadata=self.metadata(name, fold, repetition, seed),
                metrics=result.metrics,
                positions=positions,
                equity=result.equity,
                returns=result.returns,
                gross_returns=result.gross_returns,
                costs=result.costs,
                turnover=result.turnover,
            )
        except Exception as exc:
            LOGGER.exception("%s failed on fold %s seed %s", name, fold.number, seed)
            store.add_failure(
                baseline=name, fold=fold.number, repetition=repetition, seed=seed, error=repr(exc)
            )
