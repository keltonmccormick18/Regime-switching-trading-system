"""Transformer forecasting model — adapted from the SDE-Modeling forecasting_models.py.

Key differences from the SDE version:
  - Input is multi-feature: (batch, n_features, lookback) instead of univariate (batch, lookback).
  - The linear input projection maps n_features → d_model instead of 1 → d_model.
  - Same causal mask, PreLN TransformerEncoder, and MLP head as the original.

Architecture:
  Input  (batch, n_features, lookback)
  Permute → (batch, lookback, n_features)
  Linear projection → (batch, lookback, d_model)
  + sinusoidal positional encoding
  TransformerEncoder (num_layers, nhead, dim_ff, PreLN, causal mask)
  Extract last token → (batch, d_model)
  MLP head: Linear → GELU → Linear → (batch, horizon)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel
from src.models.tcn import _select_device, _train_loop, _inference


# ---------------------------------------------------------------------------
# PyTorch modules
# ---------------------------------------------------------------------------

class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (same as SDE-Modeling version)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class _TransformerNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_ff: int,
        dropout: float,
        horizon: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = _SinusoidalPE(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # PreLN — more stable training, matches SDE-Modeling
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, lookback) — channels-first from Conv1d convention
        x = x.permute(0, 2, 1)                            # (batch, lookback, n_features)
        x = self.pos_enc(self.input_proj(x))               # (batch, lookback, d_model)
        mask = self._causal_mask(x.size(1), x.device)
        x = self.encoder(x, mask=mask, is_causal=True)    # (batch, lookback, d_model)
        return self.head(x[:, -1, :])                      # (batch, horizon)  — last token


# ---------------------------------------------------------------------------
# Sklearn-style wrapper
# ---------------------------------------------------------------------------

class TransformerModel(BaseModel):
    """Causal Transformer price forecaster.

    Identical training interface to TCNModel and TCNLSTMModel — pass to train_model() the same way.

    Args:
        n_features: Number of input feature channels (must match X.shape[1]).
        lookback:   Sequence length (must match X.shape[2]).
        horizon:    Steps ahead to predict (must match y.shape[1]).
        d_model:    Token embedding dimension (attention width).
        nhead:      Number of attention heads (d_model must be divisible by nhead).
        num_layers: Number of stacked TransformerEncoder layers.
        dim_ff:     Feed-forward hidden size inside each encoder layer.
    """

    def __init__(
        self,
        n_features: int = 7,
        lookback: int = 64,
        horizon: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        epochs: int = 50,
        lr: float = 3e-4,
        batch_size: int = 64,
    ):
        self.n_features = n_features
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.device = _select_device()
        self.scaler = StandardScaler()
        self.net: _TransformerNet | None = None

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        n, f, l = X.shape
        X2d = X.transpose(0, 2, 1).reshape(-1, f)
        X2d = self.scaler.fit_transform(X2d) if fit else self.scaler.transform(X2d)
        return X2d.reshape(n, l, f).transpose(0, 2, 1).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = self._scale(X, fit=True)
        self.net = _TransformerNet(
            n_features=self.n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_ff=self.dim_ff,
            dropout=self.dropout,
            horizon=self.horizon,
        ).to(self.device)

        _train_loop(self.net, X, y, self.epochs, self.lr, self.batch_size, self.device)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._scale(X, fit=False)
        return _inference(self.net, X, self.device)
