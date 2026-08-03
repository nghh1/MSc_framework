from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from garl_trading.config import FrameworkConfig


class ArtifactStore:
    def __init__(self, root: str | Path, experiment_name: str) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{experiment_name}-{timestamp}"
        self.path = Path(root) / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "report").mkdir()
        self.metrics: list[dict] = []
        self.positions: list[pd.DataFrame] = []
        self.equity: list[pd.DataFrame] = []
        self.daily_returns: list[pd.DataFrame] = []
        self.training_diagnostics: list[pd.DataFrame] = []
        self.failures: list[dict] = []

    def initialise(self, config: FrameworkConfig, config_path: str | Path) -> None:
        shutil.copy2(config_path, self.path / "config.toml")
        manifest = {
            "run_id": self.run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "config": asdict(config)
        }
        (self.path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def save_market_data(self, prices: dict[str, pd.DataFrame]) -> None:
        data_dir = self.path / "data"
        data_dir.mkdir(exist_ok=True)
        long_frames = []
        for ticker, frame in prices.items():
            item = frame.reset_index()
            item.insert(1, "ticker", ticker)
            long_frames.append(item)
        pd.concat(long_frames, ignore_index=True).to_csv(data_dir / "prices.csv", index=False)

    def add_result(
        self,
        metadata: dict,
        metrics: dict,
        positions: pd.DataFrame,
        equity: pd.Series,
        returns: pd.Series | None = None,
        gross_returns: pd.Series | None = None,
        costs: pd.Series | None = None,
        turnover: pd.Series | None = None,
        cash_exposure: pd.Series | None = None,
    ) -> None:
        self.metrics.append({**metadata, **metrics})
        position_frame = positions.stack().rename("position").reset_index()
        position_frame.columns = ["date", "ticker", "position"]
        for key, value in metadata.items():
            position_frame[key] = value
        self.positions.append(position_frame)
        equity_frame = equity.rename("equity").reset_index()
        equity_frame.columns = ["date", "equity"]
        for key, value in metadata.items():
            equity_frame[key] = value
        self.equity.append(equity_frame)
        if returns is not None:
            daily = pd.DataFrame(
                {
                    "date": returns.index,
                    "net_return": returns.to_numpy(),
                    "gross_return": gross_returns.reindex(returns.index).to_numpy(),
                    "cost": costs.reindex(returns.index).to_numpy(),
                    "turnover": turnover.reindex(returns.index).to_numpy(),
                    "cash_exposure": cash_exposure.reindex(returns.index).to_numpy()
                }
            )
            for key, value in metadata.items():
                daily[key] = value
            self.daily_returns.append(daily)

    def add_failure(self, **values) -> None:
        self.failures.append(values)

    def add_training_diagnostics(self, metadata: dict, diagnostics: list[dict]) -> None:
        if not diagnostics:
            return
        frame = pd.DataFrame(diagnostics)
        for key, value in metadata.items():
            frame[key] = value
        self.training_diagnostics.append(frame)

    def flush(self) -> None:
        pd.DataFrame(self.metrics).to_csv(self.path / "metrics.csv", index=False)
        if self.positions:
            pd.concat(self.positions, ignore_index=True).to_csv(
                self.path / "positions.csv", index=False
            )
        if self.equity:
            pd.concat(self.equity, ignore_index=True).to_csv(self.path / "equity.csv", index=False)
        if self.daily_returns:
            pd.concat(self.daily_returns, ignore_index=True).to_csv(
                self.path / "daily_returns.csv", index=False
            )
        if self.training_diagnostics:
            pd.concat(self.training_diagnostics, ignore_index=True).to_csv(
                self.path / "training_diagnostics.csv", index=False
            )
        failure_columns = ["baseline", "fold", "repetition", "seed", "error"]
        pd.DataFrame(self.failures, columns=failure_columns).to_csv(
            self.path / "failures.csv", index=False
        )
