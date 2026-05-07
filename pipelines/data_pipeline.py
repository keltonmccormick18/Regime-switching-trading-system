"""Data pipeline: download raw OHLCV data, build features, save to parquet.

Run this once (or on a schedule) to pre-compute the feature store.
The training pipeline can then read from parquet instead of re-downloading.

Usage:
    python -m pipelines.data_pipeline --ticker SPY --start 2010-01-01
    python -m pipelines.data_pipeline --ticker SPY --start 2010-01-01 --use-tda
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from src.ingestion.historical import load_data
from src.features.pipeline import build_features

OUTPUT_PATH = "data/features/features.parquet"


def run(ticker: str = "SPY", start: str = "2010-01-01", end: str | None = None, use_tda: bool = False) -> pd.DataFrame:
    print(f"Downloading {ticker} from {start}...")
    df = load_data(ticker, start=start, end=end)

    print(f"Building features (use_tda={use_tda})...")
    features = build_features(df, use_tda=use_tda)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    features.to_parquet(OUTPUT_PATH)
    print(f"Saved {len(features)} rows → {OUTPUT_PATH}")
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--use-tda", action="store_true")
    args = parser.parse_args()

    run(ticker=args.ticker, start=args.start, end=args.end, use_tda=args.use_tda)
