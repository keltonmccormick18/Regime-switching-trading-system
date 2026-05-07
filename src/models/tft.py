"""Temporal Fusion Transformer (TFT) — HIGH_VOL_BULL regime model.

Architecture (faithful to Lim et al. 2021):
  1. Per-feature input projection (scalar → d_model per time step)
  2. Variable Selection Network (VSN) with Gated Residual Networks (GRN)
     — learns soft feature importance weights at each time step
  3. LSTM encoder — captures short/medium-range temporal dynamics
  4. Multi-head self-attention with causal mask — long-range dependencies
  5. Three quantile heads (q10, q50, q90) trained with pinball loss

predict() returns the q50 (median) to satisfy the BaseModel interface.
predict_quantiles() returns all three quantiles: (n, 3, horizon).

Input shape:  (n_samples, n_features, lookback)  — channels-first
Output shape: (n_samples, horizon)                — median forecast
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.base import BaseModel
from src.models.tcn import _select_device

QUANTILES = [0.10, 0.50, 0.90]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _GRN(nn.Module):
    """Gated Residual Network — core non-linear block of TFT."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, context_dim: int | None = None):
        super().__init__()
        self.fc1      = nn.Linear(input_dim, hidden_dim)
        self.fc2      = nn.Linear(hidden_dim, output_dim)
        self.gate     = nn.Linear(input_dim, output_dim)
        self.norm     = nn.LayerNorm(output_dim)
        self.dropout  = nn.Dropout(dropout)
        self.context  = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if context is not None and self.context is not None:
            h = h + self.context(context)
        h    = self.dropout(self.fc2(F.elu(h)))
        gate = torch.sigmoid(self.gate(x))
        return self.norm(self.residual(x) + gate * h)


class _VSN(nn.Module):
    """Variable Selection Network — per-time-step soft feature selection."""

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.feature_grns = nn.ModuleList([
            _GRN(d_model, d_model, d_model, dropout) for _ in range(n_features)
        ])
        # Selection GRN: flattened features → per-feature weights
        self.selection_grn = _GRN(n_features * d_model, d_model, n_features, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, n_features, d_model)
        b, t, f, d = x.shape
        processed = torch.stack(
            [self.feature_grns[i](x[:, :, i, :]) for i in range(f)], dim=2
        )  # (b, t, f, d)
        weights = torch.softmax(
            self.selection_grn(x.reshape(b, t, f * d)), dim=-1
        )  # (b, t, f)
        out = (processed * weights.unsqueeze(-1)).sum(dim=2)  # (b, t, d)
        return out, weights


class _TFTNet(nn.Module):
    def __init__(
        self,
        n_features:       int,
        d_model:          int,
        nhead:            int,
        num_lstm_layers:  int,
        num_attn_layers:  int,
        dropout:          float,
        horizon:          int,
        n_quantiles:      int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.vsn        = _VSN(n_features, d_model, dropout)
        self.lstm       = nn.LSTM(
            d_model, d_model, num_lstm_layers, batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        enc_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.attn  = nn.TransformerEncoder(enc_layer, num_layers=num_attn_layers,
                                           enable_nested_tensor=False)
        self.heads    = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, horizon)
            )
            for _ in range(n_quantiles)
        ])
        self.dir_head = nn.Linear(d_model, 1)   # A1: direction classification head

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

    def forward_both(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Return (quantile_heads, direction_logit) sharing the full TFT backbone."""
        b, f, l = x.shape
        x_p    = x.permute(0, 2, 1)                      # (b, l, f)
        x_proj = self.input_proj(x_p.unsqueeze(-1))       # (b, l, f, d_model)
        x_vsn, _ = self.vsn(x_proj)                       # (b, l, d_model)
        x_lstm, _ = self.lstm(x_vsn)                      # (b, l, d_model)
        mask   = self._causal_mask(l, x_p.device)
        x_out  = self.attn(x_lstm, mask=mask, is_causal=True)  # (b, l, d_model)
        last   = x_out[:, -1, :]                           # (b, d_model)
        return [head(last) for head in self.heads], self.dir_head(last)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: (batch, n_features, lookback)
        return self.forward_both(x)[0]   # list of (b, horizon)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _pinball_loss(
    preds: list[torch.Tensor],
    target: torch.Tensor,
    quantiles: list[float],
) -> torch.Tensor:
    loss = torch.zeros(1, device=target.device)
    for q, pred in zip(quantiles, preds):
        err  = target - pred
        loss = loss + torch.mean(torch.max(q * err, (q - 1) * err))
    return loss


def _train_tft(
    net: _TFTNet,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> None:
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    loader    = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    net.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = net(xb)
            loss  = _pinball_loss(preds, yb, QUANTILES)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  TFT epoch {epoch + 1}/{epochs}  loss={total / len(loader):.4f}")


def _train_tft_dual(
    net: _TFTNet,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    dir_weight: float = 0.3,
) -> None:
    """Combined pinball (quantile regression) + BCE (direction) training for TFT.

    Shares the backbone via forward_both() — single forward pass per batch.
    """
    y_dir   = (y[:, -1] > 0).astype(np.float32)
    X_t     = torch.tensor(X,     dtype=torch.float32, device=device)
    y_t     = torch.tensor(y,     dtype=torch.float32, device=device)
    y_dir_t = torch.tensor(y_dir, dtype=torch.float32, device=device).unsqueeze(-1)

    loader    = DataLoader(TensorDataset(X_t, y_t, y_dir_t), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    bce_crit  = nn.BCEWithLogitsLoss()

    net.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb, ydb in loader:
            optimizer.zero_grad()
            preds, dir_logit = net.forward_both(xb)
            loss = (1.0 - dir_weight) * _pinball_loss(preds, yb, QUANTILES) + \
                   dir_weight * bce_crit(dir_logit, ydb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  TFT epoch {epoch + 1}/{epochs}  loss={total / len(loader):.4f}")


def _tft_inference(net: _TFTNet, X: np.ndarray, device: str) -> list[np.ndarray]:
    """Returns list of 3 arrays (q10, q50, q90), each shape (n, horizon)."""
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    net.eval()
    with torch.no_grad():
        preds = net(X_t)
    return [p.cpu().numpy() for p in preds]


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class TFTModel(BaseModel):
    """Temporal Fusion Transformer — used for HIGH_VOL_BULL regime.

    Extends the BaseModel interface with predict_quantiles() which returns
    (q10, q50, q90) prediction bands used by ConformalWrapper for tighter
    uncertainty estimates in volatile bull markets.
    """

    def __init__(
        self,
        n_features:      int = 7,
        lookback:        int = 64,
        horizon:         int = 16,
        d_model:         int = 64,
        nhead:           int = 4,
        num_lstm_layers: int = 1,
        num_attn_layers: int = 2,
        dropout:         float = 0.1,
        epochs:          int = 50,
        lr:              float = 3e-4,
        batch_size:      int = 64,
    ):
        self.n_features      = n_features
        self.lookback        = lookback
        self.horizon         = horizon
        self.d_model         = d_model
        self.nhead           = nhead
        self.num_lstm_layers = num_lstm_layers
        self.num_attn_layers = num_attn_layers
        self.dropout         = dropout
        self.epochs          = epochs
        self.lr              = lr
        self.batch_size      = batch_size

        self.device  = _select_device()
        self.scaler  = StandardScaler()
        self.net: _TFTNet | None = None

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        n, f, l = X.shape
        X2d = X.transpose(0, 2, 1).reshape(-1, f)
        X2d = self.scaler.fit_transform(X2d) if fit else self.scaler.transform(X2d)
        return X2d.reshape(n, l, f).transpose(0, 2, 1).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = self._scale(X, fit=True)
        self.net = _TFTNet(
            n_features=self.n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_lstm_layers=self.num_lstm_layers,
            num_attn_layers=self.num_attn_layers,
            dropout=self.dropout,
            horizon=self.horizon,
        ).to(self.device)
        _train_tft_dual(self.net, X, y, self.epochs, self.lr, self.batch_size, self.device)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns q50 (median) predictions — shape (n, horizon)."""
        X = self._scale(X, fit=False)
        qs = _tft_inference(self.net, X, self.device)
        return qs[1]  # median

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

    def predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        """Returns all three quantile forecasts — shape (n, 3, horizon).

        Axis 1: [q10, q50, q90]
        """
        X = self._scale(X, fit=False)
        qs = _tft_inference(self.net, X, self.device)
        return np.stack(qs, axis=1)
