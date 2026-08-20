"""Smoke + correctness tests for the YAM server-side temporal smoother.

Run from FlexPi root in the flexpi conda env:

    python -m pytest experiments/yam/flexpi_policy/test_server_smoother.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

osqp = pytest.importorskip("osqp")  # noqa: F841

from experiments.yam.flexpi_policy.server_smoother import (
    YAM_JOINT_OPTIM_DIMS,
    build_smoother_from_args,
)


class _Args:
    use_smoother = True
    smoother_dt_ref = 1.0 / 30.0
    smoother_dt_min = 1.0 / 60.0
    smoother_dt_max = 1.0 / 15.0
    smoother_lambda_acc = 10.0
    smoother_lambda_time = 1.0
    smoother_horizon = 32
    smoother_stride = 16
    smoother_optim_dims = ",".join(str(i) for i in YAM_JOINT_OPTIM_DIMS)


def _ramp_chunk_with_spike(T: int = 32, spike_idx: int = 10, spike: float = 0.3) -> np.ndarray:
    """32-dim chunk: zeros everywhere except the joint slice (20..32) gets a
    smooth ramp plus a single-step spike at ``spike_idx``."""
    rng = np.random.default_rng(0)
    chunk = np.zeros((T, 32), dtype=np.float32)
    ramp = np.linspace(0.0, 0.5, T, dtype=np.float32)[:, None]
    chunk[:, 20:32] = ramp + 0.01 * rng.standard_normal(size=(T, 12)).astype(np.float32)
    chunk[spike_idx, 20:32] += spike
    return chunk


def test_off_returns_none():
    args = _Args()
    args.use_smoother = False
    assert build_smoother_from_args(args) is None


def test_on_returns_smoother():
    s = build_smoother_from_args(_Args())
    assert s is not None
    assert s.cfg.optim_dims == YAM_JOINT_OPTIM_DIMS


def test_smoother_reduces_jerk():
    s = build_smoother_from_args(_Args())
    raw = _ramp_chunk_with_spike()
    smoothed = s.smooth(raw, speed_factors=None)

    # Jerk along the joint slice, p95.
    def p95_jerk(c: np.ndarray) -> float:
        j = c[:, 20:32]
        d3 = j[2:] - 2.0 * j[1:-1] + j[:-2]
        return float(np.percentile(np.linalg.norm(d3, axis=1), 95))

    p95_raw = p95_jerk(raw)
    p95_smooth = p95_jerk(smoothed)
    assert p95_smooth < p95_raw, (p95_raw, p95_smooth)


def test_smoother_preserves_path():
    """Positions should be preserved: every smoothed waypoint should be on the
    linear interpolation of the raw chunk's path. We check the smoothed
    waypoint is within the convex hull of [raw_min, raw_max] per dim."""
    s = build_smoother_from_args(_Args())
    raw = _ramp_chunk_with_spike()
    smoothed = s.smooth(raw, speed_factors=None)

    raw_min = raw.min(axis=0)
    raw_max = raw.max(axis=0)
    # All smoothed values lie within the per-dim range of the raw chunk
    # (with a tiny epsilon for float arithmetic).
    eps = 1e-4
    assert (smoothed >= raw_min - eps).all()
    assert (smoothed <= raw_max + eps).all()


def test_non_optim_dims_passthrough():
    """Dims outside optim_dims must be byte-identical to raw."""
    s = build_smoother_from_args(_Args())
    raw = _ramp_chunk_with_spike()
    smoothed = s.smooth(raw, speed_factors=None)
    # All dims 0..19 (EEF + grippers) should be untouched.
    for d in range(20):
        np.testing.assert_allclose(smoothed[:, d], raw[:, d])


def test_speed_factors_affect_timing():
    """Per-step factors > 1 should compress time (move faster);
    factors < 1 should stretch (move slower). Path stays."""
    s = build_smoother_from_args(_Args())
    raw = _ramp_chunk_with_spike()
    T = raw.shape[0]
    fast = np.full(T, 1.4)
    slow = np.full(T, 0.7)

    smoothed_fast = s.smooth(raw, speed_factors=fast)
    smoothed_slow = s.smooth(raw, speed_factors=slow)

    # When asked to go faster, the chunk reaches higher (further along) values
    # at the same uniform time index than when asked to go slower. Compare the
    # mean of the second half of the joint slice.
    second_half_fast = smoothed_fast[T // 2:, 20:32].mean()
    second_half_slow = smoothed_slow[T // 2:, 20:32].mean()
    assert second_half_fast > second_half_slow, (second_half_fast, second_half_slow)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
