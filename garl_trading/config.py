from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "garl_main"
    seed: int = 42
    repetitions: int = 10
    artifacts_dir: str = "results"


@dataclass(frozen=True)
class DataConfig:
    start: str = "2001-01-01"
    end: str = "2025-12-31"
    tickers: tuple[str, ...] = ("NVDA", "AAPL", "MSFT", "JPM", "BAC", "MS", "CAT", "RTX", "BA")
    adjust_prices: bool = True
    target_horizon: int = 5


@dataclass(frozen=True)
class ValidationConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    min_train_bars: int = 1260
    max_train_bars: int = 1864
    embargo_bars: int = 5
    final_holdout_start: str | None = "2023-01-01"
    use_final_holdout: bool = True


@dataclass(frozen=True)
class ExecutionConfig:
    initial_capital: float = 100000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    short_borrow_bps_annual: float = 50.0
    position_levels: tuple[float, ...] = (-1.0, 0.0, 1.0)
    decision_interval: int = 5
    rebalance_threshold: float = 0.20


@dataclass(frozen=True)
class ModelsConfig:
    device: str = "auto"
    supervised: tuple[str, ...] = (
        "arimax_static",
        "arimax_rolling",
        "random_forest",
        "lstm",
        "tcn",
        "transformer",
    )
    rl: tuple[str, ...] = (
        "single_a2c",
        "single_ppo",
        "single_dqn",
        "independent_a2c",
        "independent_ppo",
        "independent_dqn",
        "garl_ddal",
        "selective_garl_ddal",
    )
    lookback: int = 20
    rl_feature_extractor: str = "tcn"
    rl_encoder_channels: int = 32
    rl_encoder_kernel_size: int = 3
    rl_encoder_dilations: tuple[int, ...] = (1, 2, 4, 8)
    rl_encoder_dropout: float = 0.0
    supervised_risk_aversion: float = 10.0
    train_epochs: int = 100
    rollout_length: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.95
    turnover_penalty_multiplier: float = 2.0
    # Zero disables checkpoint-based early stopping and keeps the final fixed-step model.
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 1e-4
    minimum_train_epochs: int = 30
    garl_share_after_fraction: float = 0.3
    garl_share_every: int = 2
    garl_pool_size: int = 3
    selective_garl_alignment_threshold: float = 0.0
    selective_garl_peer_mix: float = 0.5


@dataclass(frozen=True)
class TuningConfig:
    enabled: bool = True
    trials: int = 15
    rl_trials: int = 9
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
        if self.data.target_horizon < 1:
            raise ValueError("target_horizon must be positive.")
        if self.validation.outer_folds < 1 or self.validation.inner_folds < 1:
            raise ValueError("Fold counts must be positive.")
        if self.validation.max_train_bars < self.validation.min_train_bars:
            raise ValueError("max_train_bars must be >= min_train_bars.")
        if self.validation.embargo_bars < self.data.target_horizon:
            raise ValueError("embargo_bars must be at least target_horizon.")
        if self.experiment.repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if self.models.lookback < 2 or self.models.train_epochs < 1:
            raise ValueError("lookback must be >= 2 and train_epochs must be positive.")
        if self.models.rl_feature_extractor != "tcn":
            raise ValueError("The registered RL feature extractor must be 'tcn'.")
        if self.models.rl_encoder_channels < 1 or self.models.rl_encoder_kernel_size < 2:
            raise ValueError("RL TCN channels must be positive and kernel size must be >= 2.")
        if not self.models.rl_encoder_dilations or any(
            dilation < 1 for dilation in self.models.rl_encoder_dilations):
            raise ValueError("RL TCN dilations must be non-empty positive integers.")
        if not 0 <= self.models.rl_encoder_dropout < 1:
            raise ValueError("RL TCN dropout must lie in [0, 1).")
        receptive_field = 1 + (self.models.rl_encoder_kernel_size - 1) * sum(
            self.models.rl_encoder_dilations)
        if receptive_field < self.models.lookback:
            raise ValueError("RL TCN receptive field must cover the complete lookback window.")
        if self.models.rollout_length < 2 or not 0 < self.models.gamma <= 1:
            raise ValueError("rollout_length must be >= 2 and gamma must lie in (0, 1].")
        if self.models.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.models.supervised_risk_aversion <= 0:
            raise ValueError("supervised_risk_aversion must be positive.")
        if self.models.turnover_penalty_multiplier < 1:
            raise ValueError("turnover_penalty_multiplier must be at least 1.")
        if self.models.early_stopping_patience < 0 or self.models.minimum_train_epochs < 1:
            raise ValueError(
                "Early-stopping patience must be non-negative and minimum epochs positive.")
        if self.models.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative.")
        if not 0 <= self.models.garl_share_after_fraction < 1:
            raise ValueError("garl_share_after_fraction must lie in [0, 1).")
        if self.models.garl_share_every < 1 or self.models.garl_pool_size < 0:
            raise ValueError("GARL sharing interval must be positive and pool size non-negative.")
        if not -1 <= self.models.selective_garl_alignment_threshold < 1:
            raise ValueError("Selective GARL alignment threshold must lie in [-1, 1).")
        if not 0 <= self.models.selective_garl_peer_mix <= 1:
            raise ValueError("Selective GARL peer mix must lie in [0, 1].")
        if any(
            value < 0
            for value in (
                self.execution.transaction_cost_bps,
                self.execution.slippage_bps,
                self.execution.short_borrow_bps_annual,
            )):
            raise ValueError("Execution costs cannot be negative.")
        if not 0 <= self.execution.rebalance_threshold < 2:
            raise ValueError("rebalance_threshold must lie in [0, 2).")
        if self.execution.decision_interval < 1:
            raise ValueError("decision_interval must be positive.")
        valid_device = self.models.device in {"auto", "cpu", "mps", "cuda"}
        valid_device |= (
            self.models.device.startswith("cuda:")
            and self.models.device.removeprefix("cuda:").isdigit()
        )
        if not valid_device:
            raise ValueError("models.device must be auto, cpu, mps, cuda, or cuda:<index>.")
        levels = self.execution.position_levels
        if not levels or min(levels) < -1 or max(levels) > 1:
            raise ValueError("Position levels must lie in [-1, 1].")
        if tuple(levels) != (-1.0, 0.0, 1.0):
            raise ValueError(
                "Incremental RL actions require position_levels = [-1.0, 0.0, 1.0].")
        if self.tuning.objective not in {"sharpe", "sortino", "calmar", "total_return"}:
            raise ValueError("Unsupported tuning objective.")
        if not set(self.reporting.formats).issubset({"png", "csv", "md"}):
            raise ValueError("Reporting formats must be selected from png, csv, and md.")


SECTIONS: dict[str, type] = {
    "experiment": ExperimentConfig,
    "data": DataConfig,
    "validation": ValidationConfig,
    "execution": ExecutionConfig,
    "models": ModelsConfig,
    "tuning": TuningConfig,
    "reporting": ReportingConfig
}


def coerce(section_cls: type, values: dict[str, Any]):
    normalized = {
        key: tuple(value) if isinstance(value, list) else value for key, value in values.items()
    }
    return section_cls(**normalized)


def load_config(path: str | Path = "configs/default.toml") -> FrameworkConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    sections = {name: coerce(cls, raw.get(name, {})) for name, cls in SECTIONS.items()}
    config = FrameworkConfig(**sections)
    config.validate()
    return config
