"""Model artifact store — save and load fitted models with versioning.

Each artifact is a pair of files written to a local directory (and optionally
mirrored to S3):

  {name}_{timestamp}.pt    — torch.save dict:
                               • state_dict  (PyTorch net weights)
                               • scaler      (pickled sklearn StandardScaler)
                               • class_name  (string — used to reconstruct the model)
                               • config      (dict of __init__ kwargs)

  {name}_{timestamp}.json  — sidecar metadata (no weights), used for fast listing
                               without loading the full .pt file.

S3 is optional: set ARTIFACT_BUCKET env var (or pass s3_bucket to ArtifactStore).
If boto3 is not installed or the bucket is not configured, everything stays local.

Usage:
    store = ArtifactStore()                        # local only
    store = ArtifactStore(s3_bucket="my-bucket")   # local + S3 mirror

    path = store.save(model, name="SPY_tcn_lstm",
                      metadata={"ticker": "SPY", "regime": "high_vol_bull"})

    model = store.load(path)                       # reconstruct full fitted model
    model = store.load("s3://my-bucket/models/…")  # load directly from S3

    entries = store.list_artifacts()               # fast — reads JSON sidecars only
    path    = store.latest("SPY_tcn_lstm")         # most recent artifact for a name prefix
"""
from __future__ import annotations

import json
import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.models.base import BaseModel

# Optional S3
try:
    import boto3
    _HAS_S3 = True
except ImportError:
    _HAS_S3 = False


# ---------------------------------------------------------------------------
# Model registry — maps class_name → (wrapper class, net factory)
# ---------------------------------------------------------------------------

def _build_registry():
    from src.models.tcn         import TCNModel,       _TCNNet
    from src.models.tcn_lstm    import TCNLSTMModel,   _TCNLSTMNet
    from src.models.transformer import TransformerModel, _TransformerNet

    def _make_tcn_net(cfg):
        return _TCNNet(cfg["n_features"], cfg["channels"], cfg["kernel_size"],
                       cfg["dropout"], cfg["horizon"])

    def _make_tcn_lstm_net(cfg):
        return _TCNLSTMNet(
            n_features   = cfg["n_features"],
            tcn_channels = cfg["tcn_channels"],
            kernel_size  = cfg["kernel_size"],
            dropout      = cfg["dropout"],
            lstm_hidden  = cfg["lstm_hidden"],
            lstm_layers  = cfg["lstm_layers"],
            horizon      = cfg["horizon"],
        )

    def _make_transformer_net(cfg):
        return _TransformerNet(
            n_features = cfg["n_features"],
            d_model    = cfg["d_model"],
            nhead      = cfg["nhead"],
            num_layers = cfg["num_layers"],
            dim_ff     = cfg["dim_ff"],
            dropout    = cfg["dropout"],
            horizon    = cfg["horizon"],
        )

    return {
        "TCNModel":         (TCNModel,          _make_tcn_net),
        "TCNLSTMModel":     (TCNLSTMModel,      _make_tcn_lstm_net),
        "TransformerModel": (TransformerModel,  _make_transformer_net),
    }


# Attributes captured from each model to fully reconstruct it.
_CONFIG_ATTRS = [
    "n_features", "lookback", "horizon", "epochs", "lr", "batch_size", "dropout",
    # TCN / TCN-LSTM
    "channels", "kernel_size", "tcn_channels", "lstm_hidden", "lstm_layers",
    # Transformer
    "d_model", "nhead", "num_layers", "dim_ff",
]


# ---------------------------------------------------------------------------
# ArtifactStore
# ---------------------------------------------------------------------------

class ArtifactStore:
    """Save, load, and list fitted model artifacts.

    Args:
        local_dir:  Directory for .pt and .json files. Created if absent.
        s3_bucket:  S3 bucket name. Reads ARTIFACT_BUCKET env var if not given.
                    Pass empty string "" to disable S3 entirely.
        s3_prefix:  Key prefix inside the bucket (default "models").
    """

    def __init__(
        self,
        local_dir:  str = "artifacts/models",
        s3_bucket:  str | None = None,
        s3_prefix:  str = "models",
    ):
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        bucket = s3_bucket if s3_bucket is not None else os.environ.get("ARTIFACT_BUCKET", "")
        self.s3_bucket = bucket or None
        self.s3_prefix = s3_prefix.rstrip("/")
        self._s3 = boto3.client("s3") if (self.s3_bucket and _HAS_S3) else None

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model:    BaseModel,
        name:     str,
        metadata: dict | None = None,
    ) -> str:
        """Persist a fitted model to disk (and optionally S3).

        Args:
            model:    A fitted BaseModel instance (TCNModel, TCNLSTMModel, TransformerModel).
            name:     Human-readable base name, e.g. "SPY_regime_bull".
            metadata: Extra key-value pairs stored in the JSON sidecar (ticker, regime, …).

        Returns:
            Absolute local path of the saved .pt file.
        """
        class_name = type(model).__name__
        config     = _extract_config(model)
        ts         = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        stem       = f"{name}_{ts}"

        pt_path   = self.local_dir / f"{stem}.pt"
        json_path = self.local_dir / f"{stem}.json"

        # --- .pt: weights + scaler + config ---
        torch.save({
            "class_name":  class_name,
            "config":      config,
            "state_dict":  model.net.state_dict(),
            "scaler":      pickle.dumps(model.scaler),
        }, pt_path)

        # --- .json: lightweight sidecar for listing ---
        sidecar = {
            "class_name":  class_name,
            "config":      config,
            "metadata":    metadata or {},
            "saved_at":    datetime.now(tz=timezone.utc).isoformat(),
            "pt_path":     str(pt_path),
            "name":        name,
        }
        json_path.write_text(json.dumps(sidecar, indent=2))

        # --- optional S3 mirror ---
        if self._s3:
            for path in (pt_path, json_path):
                key = f"{self.s3_prefix}/{path.name}"
                self._s3.upload_file(str(path), self.s3_bucket, key)

        return str(pt_path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, path: str) -> BaseModel:
        """Reconstruct a fitted model from a .pt file path or s3:// URI.

        The model is returned ready for inference (net in eval mode, scaler fitted).
        """
        local_path = self._resolve_local(path)
        ckpt = torch.load(local_path, map_location="cpu", weights_only=False)

        registry    = _build_registry()
        class_name  = ckpt["class_name"]
        config      = ckpt["config"]

        if class_name not in registry:
            raise ValueError(f"Unknown model class '{class_name}'. "
                             f"Known: {list(registry)}")

        model_cls, net_factory = registry[class_name]
        model = model_cls(**config)
        model.net    = net_factory(config).to(model.device)
        model.net.load_state_dict(ckpt["state_dict"])
        model.net.eval()
        model.scaler = pickle.loads(ckpt["scaler"])

        return model

    # ------------------------------------------------------------------
    # List / discover
    # ------------------------------------------------------------------

    def list_artifacts(self, name_prefix: str = "") -> list[dict]:
        """Return metadata for all saved artifacts (reads JSON sidecars only).

        Args:
            name_prefix: Filter to artifacts whose name starts with this string.

        Returns:
            List of sidecar dicts sorted by saved_at descending (newest first).
        """
        entries = []
        for json_path in sorted(self.local_dir.glob("*.json"), reverse=True):
            try:
                sidecar = json.loads(json_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if name_prefix and not sidecar.get("name", "").startswith(name_prefix):
                continue
            entries.append(sidecar)
        return entries

    def latest(self, name_prefix: str = "") -> str | None:
        """Return the local .pt path of the most recently saved artifact.

        Args:
            name_prefix: Restrict search to artifacts matching this name prefix.

        Returns:
            Absolute path string, or None if no matching artifact exists.
        """
        entries = self.list_artifacts(name_prefix=name_prefix)
        if not entries:
            return None
        return entries[0]["pt_path"]

    def delete(self, pt_path: str) -> None:
        """Remove a .pt artifact and its JSON sidecar from local disk (and S3 if enabled)."""
        pt   = Path(pt_path)
        json = pt.with_suffix(".json")
        for p in (pt, json):
            if p.exists():
                p.unlink()
        if self._s3:
            for p in (pt, json):
                key = f"{self.s3_prefix}/{p.name}"
                try:
                    self._s3.delete_object(Bucket=self.s3_bucket, Key=key)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_local(self, path: str) -> str:
        """If path is an s3:// URI, download to a temp local file and return its path."""
        if not path.startswith("s3://"):
            return path
        if not self._s3:
            raise RuntimeError("S3 not configured (install boto3 and set ARTIFACT_BUCKET).")
        # s3://bucket/prefix/file.pt
        parts    = path[5:].split("/", 1)
        bucket   = parts[0]
        key      = parts[1]
        filename = key.split("/")[-1]
        local    = self.local_dir / filename
        if not local.exists():
            self._s3.download_file(bucket, key, str(local))
        return str(local)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_config(model: BaseModel) -> dict:
    """Collect all __init__-style attributes from a model for serialisation."""
    return {attr: getattr(model, attr) for attr in _CONFIG_ATTRS if hasattr(model, attr)}
