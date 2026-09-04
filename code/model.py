from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


class BranchPreprocessor:
    def __init__(self) -> None:
        self.imputer = None
        self.scaler = None

    def transform(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        return self.scaler.transform(self.imputer.transform(flat)).reshape(shape).astype(np.float32)


class TargetPreprocessor:
    def inverse(self, y: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)


class PriceCurveFramework(nn.Module):
    def __init__(self, market_dim: int, history_dim: int, calendar_dim: int, hidden: int = 16) -> None:
        super().__init__()
        self.market_encoder = nn.GRU(market_dim, hidden, batch_first=True)
        self.history_encoder = nn.GRU(history_dim, hidden, batch_first=True)
        self.calendar_encoder = nn.GRU(calendar_dim, hidden // 2, batch_first=True)
        fused_dim = hidden + hidden + hidden // 2
        self.attn = nn.MultiheadAttention(fused_dim, num_heads=4, batch_first=True, dropout=0.05)
        self.norm1 = nn.LayerNorm(fused_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(fused_dim, fused_dim)
        )
        self.norm2 = nn.LayerNorm(fused_dim)
        self.head = nn.Sequential(nn.Linear(fused_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, market: torch.Tensor, history: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        market_h, _ = self.market_encoder(market)
        history_h, _ = self.history_encoder(history)
        calendar_h, _ = self.calendar_encoder(calendar)
        fused = torch.cat([market_h, history_h, calendar_h], dim=-1)
        attn_out, _ = self.attn(fused, fused, fused)
        fused = self.norm1(fused + attn_out)
        fused = self.norm2(fused + self.ffn(fused))
        return self.head(fused).squeeze(-1)


def _by_period(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    groups = [[c for c in columns if c.endswith(f"_t{i:02d}")] for i in range(1, 49)]
    if any(not group for group in groups):
        missing = [i + 1 for i, group in enumerate(groups) if not group]
        raise ValueError(f"Missing feature columns for periods {missing}")
    return np.stack([df[group].to_numpy(dtype=float) for group in groups], axis=1)


def _static_by_period(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(df), 48, 0), dtype=float)
    return np.repeat(df[columns].to_numpy(dtype=float)[:, None, :], 48, axis=1)


def load_preprocessors(path: Path) -> dict:
    # The original training script was executed as __main__; register the public
    # class definitions so its frozen joblib objects remain loadable.
    main = sys.modules["__main__"]
    main.BranchPreprocessor = BranchPreprocessor
    main.TargetPreprocessor = TargetPreprocessor
    return joblib.load(path)


def infer_local_expert(features_path: Path, weights_path: Path, preprocessors_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(features_path)
    bundle = load_preprocessors(preprocessors_path)
    cols = bundle["columns"]
    market = _by_period(raw, cols["market"])
    history = np.concatenate(
        [
            _by_period(raw, cols["history_price_lag"]),
            _by_period(raw, cols["history_vmd"]),
            _static_by_period(raw, cols["history_rolling"]),
        ],
        axis=-1,
    )
    calendar = _by_period(raw, cols["calendar"])
    base = raw[[f"price_lag_1d_t{i:02d}" for i in range(1, 49)]].to_numpy(dtype=float)
    market = bundle["preprocessors"]["market"].transform(market)
    history = bundle["preprocessors"]["history"].transform(history)
    calendar = bundle["preprocessors"]["calendar"].transform(calendar)

    model = PriceCurveFramework(market.shape[-1], history.shape[-1], calendar.shape[-1], hidden=16)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        residual = model(
            torch.tensor(market), torch.tensor(history), torch.tensor(calendar)
        ).numpy()
    pred = base + bundle["target"].inverse(residual)
    out = raw[["sample_id", "code", "target_date"]].copy()
    for i in range(48):
        out[f"pred_t{i + 1:02d}"] = pred[:, i]
    return out
