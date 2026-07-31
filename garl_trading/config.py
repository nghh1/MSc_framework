from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "garl_main"
    seed: int = 42
    repetitions: int = 5
    artifacts_dir: str = "artifacts"


@dataclass(frozen=True)
class DataConfig:
    start: str = "2001-01-01"
    end: str = "2025-12-31"
    tickers: tuple[str, ...] = (
        "NVDA", "AAPL", "MSFT", "JPM", "BAC", "MS", "CAT", "RTX", "BA"
    )
    adjust_prices: bool = True


@dataclass(frozen=True)
class ValidationConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    min_train_bars: int = 1260
    max_train_bars: int = 1864
    embargo_bars: int = 1
    final_holdout_start: str | None = "2021-01-01"
    use_final_holdout: bool = True


@dataclass(frozen=True)
class ExecutionConfig:
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    short_borrow_bps_annual: float = 0.0
    position_levels: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)


@dataclass(frozen=True)
class ModelsConfig:
    supervised: tuple[str, ...] = (
        "arimax_static", "arimax_rolling", "random_forest", "lstm", "tcn", "tft"
    )
    rl: tuple[str, ...] = (
        "single_a2c",
        "single_ppo",
        "single_dqn",
        "independent_a2c",
        "independent_ppo",
        "independent_dqn",
        "garl_ddal",
    )
    lookback: int = 20
    train_epochs: int = 100
    rollout_length: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.95


@dataclass(frozen=True)
class TuningConfig:
    enabled: bool = True
    trials: int = 15
    rl_trials: int = 5
    objective: str = "sharpe"


@dataclass(frozen=True)
class ReportingConfig:
    formats: tuple[str, ...] = ("png", "csv", "md")
    confidence_level: float = 0.95


@dataclass(frozen=True)
class FrameworkConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def validate(self) -> None:
        if not self.data.tickers:
            raise ValueError("At least one ticker is required.")
        if self.validation.outer_folds < 1 or self.validation.inner_folds < 1:
            raise ValueError("Fold counts must be positive.")
        if self.validation.max_train_bars < self.validation.min_train_bars:
            raise ValueError("max_train_bars must be >= min_train_bars.")
        if self.experiment.repetitions < 1:
            raise ValueError("repetitions must be positive.")
        levels = self.execution.position_levels
        if not levels or min(levels) < -1 or max(levels) > 1:
            raise ValueError("Position levels must lie in [-1, 1].")


_SECTIONS: dict[str, type] = {
    "experiment": ExperimentConfig,
    "data": DataConfig,
    "validation": ValidationConfig,
    "execution": ExecutionConfig,
    "models": ModelsConfig,
    "tuning": TuningConfig,
    "reporting": ReportingConfig,
}


def _coerce(section_cls: type, values: dict[str, Any]):
    normalized = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in values.items()
    }
    return section_cls(**normalized)


def load_config(path: str | Path = "configs/default.toml") -> FrameworkConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    sections = {
        name: _coerce(cls, raw.get(name, {}))
        for name, cls in _SECTIONS.items()
    }
    config = FrameworkConfig(**sections)
    config.validate()
    return config
