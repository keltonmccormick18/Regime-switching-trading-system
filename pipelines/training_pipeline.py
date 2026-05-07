"""Training pipeline — the main entry point for the full system.

Flow:
  1. Load raw OHLCV data (or read from cached parquet)
  2. Build features via pipeline.py  (log returns, rolling vol, RSI, SMA ratios, TDA norms)
  3. Detect current market regime     (TDA L1 norm + 200-day SMA → 4-state Regime enum)
  4. Select the regime-appropriate model (TCN+LSTM for high vol, TCN for low vol)
  5. Slice features into (X, y) sequence windows
  6. Train the selected model on X_train, y_train
  7. Call model.predict(X_test) → target array of predicted future close prices

  WANT: train on set that contains all market regimes, and predict by dynamically switching model based on current regime.

Usage:
    python -m pipelines.training_pipeline
    python -m pipelines.training_pipeline --ticker QQQ --start 2015-01-01 --epochs 50
    python -m pipelines.training_pipeline --from-parquet   # read cached features
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.ingestion.historical import load_data
from src.features.pipeline import build_features, FEATURE_COLS, MACRO_FEATURE_COLS
from src.models.regime import detect_regime, REGIME_MODELS
from src.models.train import train_model
from src.models.predict import predict

LOOKBACK    = 64   # input sequence length (trading days)
HORIZON     = 16   # forecast horizon (trading days, ~3 weeks)
TRAIN_RATIO = 0.80


def prepare_sequences(
    df: pd.DataFrame,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the feature DataFrame into sliding (X, y) windows.

    X: (n_samples, n_features, lookback)  — channels-first, ready for Conv1d
    y: (n_samples, horizon)               — future close prices (ground-truth targets)
    """
    # Use base features + any macro features present in df (B1 support)
    all_feature_cols = FEATURE_COLS + MACRO_FEATURE_COLS
    cols = [c for c in all_feature_cols if c in df.columns]
    features = df[cols].values.astype(np.float32)   # (T, n_features)
    prices   = df["close"].values.astype(np.float32) # (T,)

    X, y = [], []
    for i in range(lookback, len(features) - horizon + 1):
        X.append(features[i - lookback:i].T)             # (n_features, lookback)
        anchor = prices[i - 1]                            # last price in the lookback window
        fwd = prices[i:i + horizon] / anchor - 1.0        # (horizon,) forward returns
        y.append(fwd.astype(np.float32))

    return np.array(X), np.array(y)


def run(
    ticker: str = "SPY",
    start: str = "2010-01-01",
    end: str | None = None,
    use_tda: bool = False,
    from_parquet: bool = False,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    epochs: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full training + inference pipeline.

    Returns:
        target  — (n_test, horizon) predicted future close prices
        actuals — (n_test, horizon) ground-truth future close prices
    """
    # 1. Load
    if from_parquet:
        path = "data/features/features.parquet"
        print(f"[1/5] Reading features from {path}...")
        features = pd.read_parquet(path)
    else:
        print(f"[1/5] Loading {ticker} from {start}...")
        df = load_data(ticker, start=start, end=end)
        print("[2/5] Building features...")
        features = build_features(df, use_tda=use_tda)

    # 2. Detect regime
    print("[3/5] Detecting market regime...")
    regime = detect_regime(features)
    print(f"      Regime: {regime.value}")

    # 3. Prepare sequences
    print("[4/5] Preparing sequences and training model...")
    X, y = prepare_sequences(features, lookback=lookback, horizon=horizon)
    n_features = X.shape[1]

    split    = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 4. Select model class for this regime, then train
    model_cls = REGIME_MODELS[regime]
    print(f"      Model : {model_cls.__name__}")
    print(f"      Train : {len(X_train)} windows  |  Test: {len(X_test)} windows")

    model = train_model(
        model_cls,
        X_train,
        y_train,
        n_features=n_features,
        lookback=lookback,
        horizon=horizon,
        epochs=epochs,
    )

    # 5. Predict → target array
    print("[5/5] Predicting...")
    target = predict(model, X_test)
    print(f"      target shape: {target.shape}  (n_test_windows × horizon)")

    return target, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Trading System — Training Pipeline")
    parser.add_argument("--ticker",        default="SPY")
    parser.add_argument("--start",         default="2010-01-01")
    parser.add_argument("--end",           default=None)
    parser.add_argument("--use-tda",       action="store_true", help="Compute TDA features (slow)")
    parser.add_argument("--from-parquet",  action="store_true", help="Read cached features instead of downloading")
    parser.add_argument("--lookback",      type=int, default=LOOKBACK)
    parser.add_argument("--horizon",       type=int, default=HORIZON)
    parser.add_argument("--epochs",        type=int, default=30)
    args = parser.parse_args()

    target, actuals = run(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        use_tda=args.use_tda,
        from_parquet=args.from_parquet,
        lookback=args.lookback,
        horizon=args.horizon,
        epochs=args.epochs,
    )

    print(f"\nSample — last test window, next {target.shape[1]} trading days:")
    print("  Predicted:", target[-1].round(2))
    print("  Actual:   ", actuals[-1].round(2))
