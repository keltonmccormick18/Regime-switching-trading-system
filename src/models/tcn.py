"""Pure TCN model — causal dilated temporal convolutions with weight normalization.

Architecture mirrors tempconvnetwork.py from SDE-Modeling but accepts multi-feature input
and exposes a sklearn-style fit/predict interface via TCNModel.

Input shape:  (n_samples, n_features, lookback)  — channels-first for Conv1d
Output shape: (n_samples, horizon)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel


# ---------------------------------------------------------------------------
# PyTorch modules
# ---------------------------------------------------------------------------

class _CausalBlock(nn.Module):
    """Single causal dilated conv block with weight norm and residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        )
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal trim: symmetric padding produces L + pad extra steps on the right; keep only L.
        out = self.conv(x)[:, :, :x.size(2)]
        out = self.drop(self.relu(out))
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class _TCNCore(nn.Module):
    """Stack of causal blocks with exponentially increasing dilation (2^0, 2^1, ...).

    Args:
        n_features: Number of input channels (feature dimensions).
        channels:   Output channels per block, e.g. [32, 32, 32, 32].
    """

    def __init__(
        self,
        n_features: int,
        channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            layers.append(_CausalBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TCNNet(nn.Module):
    def __init__(self, n_features: int, channels: list[int], kernel_size: int, dropout: float, horizon: int):
        super().__init__()
        self.tcn      = _TCNCore(n_features, channels, kernel_size, dropout)
        self.head     = nn.Linear(channels[-1], horizon)
        self.dir_head = nn.Linear(channels[-1], 1)   # A1: direction classification head

    def forward_both(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (regression_output, direction_logit) sharing the TCN backbone."""
        out = self.tcn(x)[:, :, -1]   # (batch, channels[-1])
        return self.head(out), self.dir_head(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, lookback)
        return self.forward_both(x)[0]   # (batch, horizon)


# ---------------------------------------------------------------------------
# Sklearn-style wrapper
# ---------------------------------------------------------------------------

class TCNModel(BaseModel):
    """Pure TCN price forecaster — used in low-volatility regimes.

    Args:
        n_features: Number of input feature channels (must match X.shape[1]).
        lookback:   Sequence length (must match X.shape[2]).
        horizon:    Number of future time steps to predict (must match y.shape[1]).
    """

    def __init__(
        self,
        n_features: int = 7,
        lookback: int = 64,
        horizon: int = 16,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
    ):
        self.n_features = n_features
        self.lookback = lookback
        self.horizon = horizon
        self.channels = channels or [32, 32, 32, 32]
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.device = _select_device()
        self.scaler = StandardScaler()
        self.net: _TCNNet | None = None

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        n, f, l = X.shape
        X2d = X.transpose(0, 2, 1).reshape(-1, f)
        X2d = self.scaler.fit_transform(X2d) if fit else self.scaler.transform(X2d)
        return X2d.reshape(n, l, f).transpose(0, 2, 1).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = self._scale(X, fit=True)
        self.net = _TCNNet(
            self.n_features, self.channels, self.kernel_size, self.dropout, self.horizon
        ).to(self.device)
        _train_loop_dual(self.net, X, y, self.epochs, self.lr, self.batch_size, self.device)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._scale(X, fit=False)
        return _inference(self.net, X, self.device)

    def direction_confidence(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample direction certainty in [0, 1].

        Uses the direction classification head: |sigmoid(logit) - 0.5| * 2
        gives 0 when the model is uncertain (50/50) and 1 when it is certain.
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

        Values > 0.5 mean the model predicts price will be HIGHER at horizon
        end than the current bar; < 0.5 means LOWER.  Certainty is
        ``|prob - 0.5| * 2``.  This is the correct method to use when you
        need both direction AND certainty (e.g. agreement-gated confidence).
        """
        X = self._scale(X, fit=False)
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.net.eval()
        with torch.no_grad():
            _, logits = self.net.forward_both(X_t)
        return torch.sigmoid(logits).squeeze(-1).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Shared training utilities (reused by TCNLSTMModel)
# ---------------------------------------------------------------------------

def _select_device() -> str:
    # MPS (Apple Silicon Metal) is intentionally skipped — PyTorch's MPS backend
    # has a known SIGSEGV bug in the Metal GPU stream triggered by causal attention
    # masks and LSTM dropout ops used in these models (KERN_INVALID_ADDRESS @ 0x48).
    # CPU is fast enough for the batch sizes used here and is crash-free.
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _train_loop(
    net: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> None:
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    net.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")


def _inference(net: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    net.eval()
    with torch.no_grad():
        return net(X_t).cpu().numpy()


def _train_loop_dual(
    net: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    dir_weight: float = 0.3,
) -> None:
    """Combined MSE (regression) + BCE (direction) training loop.

    dir_weight controls the fraction of the loss attributed to direction:
      total_loss = (1 - dir_weight) * MSELoss + dir_weight * BCEWithLogitsLoss
    0.3 means the regression objective dominates while the direction head is
    trained as an auxiliary task.

    The network must expose a ``forward_both(x) → (regression, dir_logit)``
    method that shares the backbone and returns both heads in one pass.
    """
    y_dir   = (y[:, -1] > 0).astype(np.float32)          # (n,) binary direction at horizon end
    X_t     = torch.tensor(X,     dtype=torch.float32, device=device)
    y_t     = torch.tensor(y,     dtype=torch.float32, device=device)
    y_dir_t = torch.tensor(y_dir, dtype=torch.float32, device=device).unsqueeze(-1)

    loader    = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t, y_dir_t),
        batch_size=batch_size, shuffle=True,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_crit  = nn.MSELoss()
    bce_crit  = nn.BCEWithLogitsLoss()

    net.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb, ydb in loader:
            optimizer.zero_grad()
            reg_out, dir_logit = net.forward_both(xb)
            loss = (1.0 - dir_weight) * mse_crit(reg_out, yb) + dir_weight * bce_crit(dir_logit, ydb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")
