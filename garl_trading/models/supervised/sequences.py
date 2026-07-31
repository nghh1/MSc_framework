from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..base import ForecastModel, ModelContext


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.encoder = nn.LSTM(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        return self.head(hidden[-1]).squeeze(-1)


class _TCNNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(n_features, hidden, 3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        y = self.network(x.transpose(1, 2))
        y = y[..., : x.shape[1]]
        return self.head(y[:, :, -1]).squeeze(-1)


class _TFTLiteNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.feature_gate = nn.Sequential(nn.Linear(n_features, n_features), nn.Softmax(dim=-1))
        self.project = nn.Linear(n_features, hidden)
        self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x * self.feature_gate(x)
        encoded, _ = self.encoder(self.project(x))
        size = encoded.shape[1]
        mask = torch.triu(torch.ones(size, size, device=x.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(encoded, encoded, encoded, attn_mask=mask)
        return self.head(self.norm(encoded + attended)[:, -1]).squeeze(-1)


class TorchSequenceForecaster(ForecastModel):
    architecture = "lstm"

    def __init__(
        self,
        lookback: int = 20,
        hidden: int = 32,
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model: nn.Module | None = None
        self.columns: list[str] = []
        self.history = pd.DataFrame()

    def _network(self, n_features: int) -> nn.Module:
        if self.architecture == "lstm":
            return _LSTMNet(n_features, self.hidden, self.layers, self.dropout)
        if self.architecture == "tcn":
            return _TCNNet(n_features, self.hidden, self.dropout)
        return _TFTLiteNet(n_features, self.hidden, self.heads, self.dropout)

    def _windows(self, values: np.ndarray, targets: np.ndarray | None = None):
        x = np.stack(
            [values[i - self.lookback + 1:i + 1] for i in range(self.lookback - 1, len(values))]
        )
        if targets is None:
            return x
        return x, targets[self.lookback - 1:]

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> TorchSequenceForecaster:
        torch.manual_seed(self.seed)
        valid = features.notna().all(axis=1) & targets.notna()
        x, y = features.loc[valid], targets.loc[valid]
        self.columns = list(x.columns)
        scaled = self.scaler.fit_transform(x)
        xw, yw = self._windows(scaled, y.to_numpy(dtype=np.float32))
        if len(xw) < 20:
            raise ValueError("Insufficient sequence windows.")
        self.model = self._network(len(self.columns)).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        x_tensor = torch.tensor(xw, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(yw, dtype=torch.float32, device=self.device)
        self.model.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(x_tensor), device=self.device)
            for start in range(0, len(order), 64):
                batch = order[start:start + 64]
                loss = nn.functional.mse_loss(self.model(x_tensor[batch]), y_tensor[batch])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
        self.model.eval()
        self.history = x.tail(self.lookback - 1)
        self._set_signal_scale(y)
        return self

    @torch.no_grad()
    def predict_returns(
        self,
        features: pd.DataFrame,
        *,
        context: ModelContext | None = None,
        realized_targets: pd.Series | None = None,
    ) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        history = context.features if context is not None else self.history
        combined = pd.concat([history.loc[:, self.columns], features.loc[:, self.columns]])
        combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
        scaled = self.scaler.transform(combined)
        predictions = {}
        positions = {date: i for i, date in enumerate(combined.index)}
        for date in features.index:
            i = positions[date]
            if i + 1 < self.lookback:
                predictions[date] = 0.0
                continue
            window = torch.tensor(
                scaled[i - self.lookback + 1:i + 1],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            predictions[date] = float(self.model(window).item())
        return pd.Series(predictions).reindex(features.index)


class LSTMForecaster(TorchSequenceForecaster):
    architecture = "lstm"


class TCNForecaster(TorchSequenceForecaster):
    """Temporal Convolutional Network return forecaster."""

    architecture = "tcn"


class TFTForecaster(TorchSequenceForecaster):
    """Compact Temporal Fusion Transformer-style return forecaster."""

    architecture = "tft"
