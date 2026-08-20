"""Smoke + correctness tests for the YAM server-side speed adapter.

Run from FlexPi root in the flexpi conda env:

    python -m pytest experiments/yam/flexpi_policy/test_server_speed.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.yam.flexpi_policy.server_speed import build_adapter_from_args


class _Args:
    speed_adapter = "off"
    speed_ckpt = None
    speed_factor_lo = 0.5
    speed_factor_hi = 1.5
    speed_heuristic_alpha = 1.0
    speed_heuristic_v_ref = 0.05
    speed_smooth_window = 5


def test_off_returns_none():
    args = _Args()
    assert build_adapter_from_args(args) is None


def test_heuristic_high_velocity_gives_low_factor():
    args = _Args()
    args.speed_adapter = "heuristic"
    a = build_adapter_from_args(args)
    assert a is not None

    # Joint slice with large per-step deltas → ||v|| >> v_ref → factor → lo.
    T = 32
    chunk = np.zeros((T, 12), dtype=np.float32)
    chunk[:, :] = np.linspace(0.0, 1.0, T)[:, None]  # Δ per step ≈ 1/31 per dim
    factors = a.factors(chunk)
    assert factors.shape == (T,)
    assert factors.mean() < 1.0, factors


def test_heuristic_low_velocity_gives_high_factor():
    args = _Args()
    args.speed_adapter = "heuristic"
    a = build_adapter_from_args(args)
    T = 32
    chunk = np.zeros((T, 12), dtype=np.float32)
    chunk[:, :] = np.linspace(0.0, 0.001, T)[:, None]  # near-static, ||v|| ≪ v_ref
    factors = a.factors(chunk)
    assert factors.mean() > 1.0, factors


def test_heuristic_factors_bounded():
    args = _Args()
    args.speed_adapter = "heuristic"
    a = build_adapter_from_args(args)
    rng = np.random.default_rng(42)
    chunk = rng.standard_normal(size=(32, 12)).astype(np.float32) * 0.5
    factors = a.factors(chunk)
    assert (factors >= 0.5).all()
    assert (factors <= 1.5).all()


def test_heuristic_smooth_transitions():
    """Factors should never exceed the factor range [lo, hi]; max step delta
    should be at most factor_range (= hi - lo)."""
    args = _Args()
    args.speed_adapter = "heuristic"
    a = build_adapter_from_args(args)

    T = 32
    # Slow -> fast transition without an artificial step. Smoothing window=5
    # spreads the velocity change over a few steps.
    chunk = np.cumsum(np.linspace(0.0001, 0.1, T)[:, None] * np.ones((1, 12)),
                      axis=0).astype(np.float32)
    factors = a.factors(chunk)
    max_step_delta = float(np.max(np.abs(np.diff(factors))))
    # Adjacent steps cannot move by more than the full factor range.
    assert max_step_delta <= (args.speed_factor_hi - args.speed_factor_lo), max_step_delta
    # And the transition should be at least somewhat smoothed — not the full
    # range in one step.
    assert max_step_delta < (args.speed_factor_hi - args.speed_factor_lo), max_step_delta


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
