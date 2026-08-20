"""SpeedAdapter strategy: maps the predicted action chunk -> per-step speed_factor.

Three modes:
  - "off"        : factor == 1 everywhere. Default. Zero deps.
  - "heuristic"  : geometric, no training data needed. Slows down where the
                   action chunk's own predicted velocity is high (= fine
                   manipulation by the model's intent), speeds up where it's
                   low (= free-space motion). Works for ALL three FlexPi
                   variants identically since it consumes the chunk only.
                   The plan envisioned pointmap-based contact-zone slowdown
                   for 3d/unified-joint, but the model's return dict doesn't
                   currently expose pointmap (we don't touch src/flexpi/),
                   so chunk-velocity is the universal proxy.
  - "learned"    : small MLP over per-step velocity/jerk + state features.
                   Trained offline; see the FlexPi deploy docs.

The output factors[t] is the per-step multiplier on the smoother's dt_ref
(factor > 1 -> speed up, < 1 -> slow down). The smoother's QP consumes
this directly, so this module is purely numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SpeedAdapterConfig:
    mode: str = "off"
    factor_lo: float = 0.5
    factor_hi: float = 1.5
    ckpt: str | None = None
    sources: list[str] = field(default_factory=list)
    heuristic_alpha: float = 1.0
    heuristic_v_ref: float = 0.05
    smooth_window: int = 5


class SpeedAdapter:
    def __init__(self, cfg: SpeedAdapterConfig):
        self.cfg = cfg
        if cfg.mode not in ("off", "heuristic", "learned"):
            raise ValueError(f"unknown speed.mode: {cfg.mode!r}")
        self._mlp = None
        self._mlp_features: list[str] | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        if cfg.mode == "learned":
            self._mlp, self._mlp_features, self._feat_mean, self._feat_std = self._load_mlp(cfg.ckpt)

    def factors(
        self,
        chunk: np.ndarray,
        aux: dict[str, Any] | None = None,
    ) -> np.ndarray:
        T = chunk.shape[0]
        if self.cfg.mode == "off":
            return np.ones(T, dtype=np.float64)
        if self.cfg.mode == "heuristic":
            return self._heuristic(chunk)
        return self._learned(chunk, aux or {})

    def _heuristic(self, chunk: np.ndarray) -> np.ndarray:
        """speed_factor = clip(alpha * v_ref / (||v_step|| + eps), lo, hi).

        High predicted joint velocity -> low factor (slow down). Smoothed with
        a centered moving average of width `smooth_window`.
        """
        cfg = self.cfg
        chunk = np.asarray(chunk, dtype=np.float64)
        T = chunk.shape[0]
        if T < 2:
            return np.ones(T, dtype=np.float64)
        v = np.linalg.norm(chunk[1:] - chunk[:-1], axis=1)  # [T-1]
        v_full = np.concatenate([v[:1], v])  # [T]
        if cfg.smooth_window > 1:
            w = max(1, int(cfg.smooth_window))
            kernel = np.ones(w) / w
            v_full = np.convolve(v_full, kernel, mode="same")
        factors = cfg.heuristic_alpha * cfg.heuristic_v_ref / (v_full + 1e-6)
        return np.clip(factors, cfg.factor_lo, cfg.factor_hi)

    def _learned(self, chunk: np.ndarray, aux: dict[str, Any]) -> np.ndarray:
        if self._mlp is None:
            raise RuntimeError("learned SpeedAdapter not initialized; missing ckpt")
        import torch

        feats = self._build_features(chunk, aux)
        if self._feat_mean is not None and self._feat_std is not None:
            feats = (feats - self._feat_mean) / self._feat_std
        with torch.no_grad():
            x = torch.as_tensor(feats, dtype=torch.float32)
            y = self._mlp(x).cpu().numpy().reshape(-1)
        return np.clip(y, self.cfg.factor_lo, self.cfg.factor_hi)

    def _build_features(self, chunk: np.ndarray, aux: dict[str, Any]) -> np.ndarray:
        """Per-step features [T, F]. Features are deterministic from chunk:
            f0 = ||v_step||                          (current joint velocity)
            f1 = ||v_step|| smoothed                 (5-step moving avg)
            f2 = ||jerk_step|| (centered second diff)
            f3 = mean(||v_step||) over the chunk     (broadcast)
            f4 = max(||v_step||) over the chunk      (broadcast)
        Future versions can append `aux["state"]` or pointmap velocity if the
        model is modified to expose it.
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        T = chunk.shape[0]
        v = np.linalg.norm(chunk[1:] - chunk[:-1], axis=1)
        v = np.concatenate([v[:1], v])  # [T]
        v_smooth = np.convolve(v, np.ones(5) / 5, mode="same")
        jerk_inner = np.linalg.norm(chunk[2:] - 2 * chunk[1:-1] + chunk[:-2], axis=1)
        jerk = np.concatenate([jerk_inner[:1], jerk_inner, jerk_inner[-1:]])[:T]
        v_mean = np.full(T, float(v.mean()))
        v_max = np.full(T, float(v.max()))
        return np.stack([v, v_smooth, jerk, v_mean, v_max], axis=1)

    def _load_mlp(self, ckpt_path: str | None):
        if ckpt_path is None:
            raise ValueError("speed.mode=learned requires speed.ckpt")
        import torch

        path = Path(ckpt_path).expanduser().resolve()
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        feature_names = list(ckpt.get("feature_names", []))
        in_dim = int(ckpt["in_dim"])
        hidden = int(ckpt.get("hidden", 64))
        mlp = _SpeedMLP(in_dim=in_dim, hidden=hidden)
        mlp.load_state_dict(ckpt["state_dict"])
        mlp.eval()
        feat_mean = (
            np.asarray(ckpt["feature_mean"], dtype=np.float64)
            if "feature_mean" in ckpt else None
        )
        feat_std = (
            np.asarray(ckpt["feature_std"], dtype=np.float64)
            if "feature_std" in ckpt else None
        )
        return mlp, feature_names, feat_mean, feat_std


class _SpeedMLP:
    """Tiny PyTorch MLP. Deferred import: torch is only needed for `learned`."""

    def __new__(cls, in_dim: int, hidden: int = 64):
        import torch
        import torch.nn as nn

        class _Impl(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x).squeeze(-1)

        return _Impl()


def adapter_from_yaml(d: dict | None) -> SpeedAdapter:
    if not d:
        return SpeedAdapter(SpeedAdapterConfig())
    cfg = SpeedAdapterConfig(
        mode=d.get("mode", "off"),
        factor_lo=d.get("factor_lo", 0.5),
        factor_hi=d.get("factor_hi", 1.5),
        ckpt=d.get("ckpt"),
        sources=list(d.get("sources", [])),
        heuristic_alpha=d.get("heuristic_alpha", 1.0),
        heuristic_v_ref=d.get("heuristic_v_ref", d.get("heuristic_d_ref", 0.05)),
        smooth_window=int(d.get("smooth_window", 5)),
    )
    return SpeedAdapter(cfg)
