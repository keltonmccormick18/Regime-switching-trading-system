"""TCN + LSTM combined model — used in high-volatility regimes.

Architecture:
  1. TCN core (_TCNCore from tcn.py): multi-scale causal feature extraction over the lookback window.
  2. LSTM: treats the TCN feature sequence as a temporal input to capture sequential dependencies.
  3. Dense head: maps the final LSTM hidden state to the horizon-step forecast.

Data flow:
  Input  (batch, n_features, lookback)
  TCN    (batch, tcn_ch, lookback)
  LSTM   input permuted to (batch, lookback, tcn_ch) → hidden (batch, lstm_hidden)
  Head   (batch, horizon)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel
from src.models.tcn import _TCNCore, _select_device, _train_loop, _train_loop_dual, _inference


class _TCNLSTMNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        tcn_channels: list[int],
        kernel_size: int,
        dropout: float,
        lstm_hidden: int,
        lstm_layers: int,
        horizon: int,
    ):
        super().__init__()
        self.tcn      = _TCNCore(n_features, tcn_channels, kernel_size, dropout)
        self.lstm     = nn.LSTM(
            input_size=tcn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head     = nn.Linear(lstm_hidden, horizon)
        self.dir_head = nn.Linear(lstm_hidden, 1)   # A1: direction classification head

    def forward_both(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (regression_output, direction_logit) sharing the TCN+LSTM backbone."""
        tcn_out = self.tcn(x)                  # (batch, tcn_ch, lookback)
        lstm_in = tcn_out.permute(0, 2, 1)     # (batch, lookback, tcn_ch)
        _, (h_n, _) = self.lstm(lstm_in)       # h_n: (lstm_layers, batch, lstm_hidden)
        h_last  = h_n[-1]                      # (batch, lstm_hidden)
        return self.head(h_last), self.dir_head(h_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, lookback)
        return self.forward_both(x)[0]         # (batch, horizon)


class TCNLSTMModel(BaseModel):
    """TCN + LSTM price forecaster — used in high-volatility regimes.

    The TCN captures multi-scale temporal patterns across the lookback window;
    the LSTM then models sequential dependencies in those extracted features.

    Args:
        n_features:   Number of input feature channels (must match X.shape[1]).
        lookback:     Sequence length (must match X.shape[2]).
        horizon:      Number of future time steps to predict (must match y.shape[1]).
        tcn_channels: TCN block output channels, one entry per block with dilation 2^i.
        lstm_hidden:  LSTM hidden state dimension.
        lstm_layers:  Number of stacked LSTM layers.
    """

    def __init__(
        self,
        n_features: int = 7,
        lookback: int = 64,
        horizon: int = 16,
        tcn_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
    ):
        self.n_features = n_features
        self.lookback = lookback
        self.horizon = horizon
        self.tcn_channels = tcn_channels or [32, 32, 32, 32]
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.device = _select_device()
        self.scaler = StandardScaler()
        self.net: _TCNLSTMNet | None = None

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        n, f, l = X.shape
        X2d = X.transpose(0, 2, 1).reshape(-1, f)
        X2d = self.scaler.fit_transform(X2d) if fit else self.scaler.transform(X2d)
        return X2d.reshape(n, l, f).transpose(0, 2, 1).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = self._scale(X, fit=True)
        self.net = _TCNLSTMNet(
            n_features=self.n_features,
            tcn_channels=self.tcn_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            lstm_hidden=self.lstm_hidden,
            lstm_layers=self.lstm_layers,
            horizon=self.horizon,
        ).to(self.device)
        _train_loop_dual(self.net, X, y, self.epochs, self.lr, self.batch_size, self.device)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._scale(X, fit=False)
        return _inference(self.net, X, self.device)

    def direction_confidence(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample direction certainty in [0, 1] from the direction head.

        Does NOT encode which direction — use direction_prob() for that.
        """
        X = self._scale(X, fit=False)
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.net.eval()
        with torch.no_grad():
            _, logits = self.net.forward_both(X_t)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        return (np.abs(probs - 0.5) * 2).astype(np.float32)

    def direction_prob(self, X: np.ndarray) -> np.ndarray:
        """Raw sigmoid probability from the direction head, shape (n,).

        > 0.5 → model predicts price HIGHER at horizon end; < 0.5 → LOWER.
        Use this (not direction_confidence) when you need both direction and
        certainty simultaneously.
        """
        X = self._scale(X, fit=False)
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.net.eval()
        with torch.no_grad():
            _, logits = self.net.forward_both(X_t)
        return torch.sigmoid(logits).squeeze(-1).cpu().numpy().astype(np.float32)
