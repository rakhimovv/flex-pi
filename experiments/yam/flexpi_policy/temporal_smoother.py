"""Temporal trajectory smoother (OSQP QP over inverse step-durations).

Port of `TimeParameterizationMPC` from
`third_party/realtime-vla-v2/server/optimizer.py`, with FlexPi-specific
defaults:
  - 14-DoF dual-arm RoboTwin layout, grippers (indices 6, 13) excluded.
  - 50 Hz dataset rate -> dt_ref = 0.02 s.
  - Optional per-step `speed_factors` multiplier on dt_ref (None == uniform);
    this is the integration point used by the SpeedAdapter strategy.

The QP minimizes:
    sum_k  lambda_time * (s_k - s_ref_k)^2
         + lambda_acc  * || dp_k * s_k - dp_{k-1} * s_{k-1} ||^2
subject to s_min <= s_k <= s_max, with optional v_max box on dp * s.

Positions are preserved exactly; only the per-step duration changes, then we
re-interpolate the waypoints onto a uniform dt_ref grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import osqp
import scipy.sparse as sp


@dataclass
class SmootherConfig:
    dt_ref: float = 1.0 / 50.0
    dt_min: float = 0.005
    dt_max: float = 0.1
    lambda_acc: float = 10.0
    lambda_time: float = 1.0
    lambda_v: float = 10.0
    v_max: float | None = None
    horizon: int = 32
    stride: int = 16
    optim_dims: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


class TemporalSmoother:
    def __init__(self, cfg: SmootherConfig):
        self.cfg = cfg
        self.s_ref = 1.0 / cfg.dt_ref
        self.s_min = 1.0 / cfg.dt_max
        self.s_max = 1.0 / cfg.dt_min

    def smooth(
        self,
        chunk: np.ndarray,
        speed_factors: np.ndarray | None = None,
    ) -> np.ndarray:
        """Smooth a [T, D] action chunk and return a [T, D] chunk on dt_ref grid.

        speed_factors: optional [T] array in [0.5, 1.5]. dt_ref_k = dt_ref / factor_k,
        i.e. factor > 1 speeds up locally, factor < 1 slows down.
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        T, D = chunk.shape
        wp = chunk[:, list(self.cfg.optim_dims)]
        dp = wp[1:] - wp[:-1]
        N = len(dp)
        if N < 2:
            return chunk.astype(np.float32)

        s_ref_per_step = self._build_s_ref(N, speed_factors)
        s_traj = self._roll_solve(dp, s_ref_per_step)

        dt_traj = 1.0 / s_traj
        t_orig = np.concatenate([[0.0], np.cumsum(dt_traj)])
        t_uniform = np.arange(T) * self.cfg.dt_ref
        t_uniform = np.clip(t_uniform, t_orig[0], t_orig[-1])

        out = chunk.copy()
        for j in self.cfg.optim_dims:
            out[:, j] = np.interp(t_uniform, t_orig, chunk[:, j])
        return out.astype(np.float32)

    def _build_s_ref(self, N: int, speed_factors: np.ndarray | None) -> np.ndarray:
        if speed_factors is None:
            return np.full(N, self.s_ref, dtype=np.float64)
        f = np.asarray(speed_factors, dtype=np.float64)
        if len(f) >= N + 1:
            f = 0.5 * (f[:-1] + f[1:])[:N]
        elif len(f) >= N:
            f = f[:N]
        else:
            f = np.concatenate([f, np.full(N - len(f), f[-1])])
        return self.s_ref * f

    def _roll_solve(self, dp: np.ndarray, s_ref_per_step: np.ndarray) -> np.ndarray:
        N = len(dp)
        H = min(self.cfg.horizon, N)
        stride = max(1, self.cfg.stride)
        out: list[float] = []
        k = 0
        while k < N:
            h = min(H, N - k)
            s = self._solve_qp(dp[k : k + h], s_ref_per_step[k : k + h])
            take = min(stride, h, N - k)
            out.extend(s[:take].tolist())
            k += take
        return np.asarray(out, dtype=np.float64)

    def _solve_qp(self, dp: np.ndarray, s_ref: np.ndarray) -> np.ndarray:
        h = len(dp)
        dp_norm = np.linalg.norm(dp, axis=1)
        scale_time = self.s_ref ** 2 + 1e-6
        scale_acc = float(np.mean((dp_norm * self.s_ref) ** 2)) + 1e-6
        l_time = self.cfg.lambda_time / scale_time
        l_acc = self.cfg.lambda_acc / scale_acc

        P = 2 * l_time * np.eye(h)
        for i in range(h - 1):
            d2_i = float(np.dot(dp[i], dp[i]))
            d2_i1 = float(np.dot(dp[i + 1], dp[i + 1]))
            d_cross = float(np.dot(dp[i], dp[i + 1]))
            P[i, i] += 2 * l_acc * d2_i
            P[i + 1, i + 1] += 2 * l_acc * d2_i1
            P[i, i + 1] -= 2 * l_acc * d_cross
            P[i + 1, i] -= 2 * l_acc * d_cross
        P_sp = sp.csc_matrix(P)
        q = -2 * l_time * s_ref

        A_blocks = [sp.eye(h)]
        l_vec = [self.s_min * np.ones(h)]
        u_vec = [self.s_max * np.ones(h)]
        if self.cfg.v_max is not None:
            A_blocks.append(sp.eye(h))
            l_vec.append(np.zeros(h))
            u_vec.append(self.cfg.v_max / (dp_norm + 1e-8))
        A = sp.vstack(A_blocks).tocsc()
        l = np.concatenate(l_vec)
        u = np.concatenate(u_vec)

        prob = osqp.OSQP()
        prob.setup(P=P_sp, q=q, A=A, l=l, u=u, verbose=False, polish=True)
        res = prob.solve()
        if res.info.status != "solved":
            return np.clip(s_ref, self.s_min, self.s_max)
        return np.clip(np.asarray(res.x, dtype=np.float64), self.s_min, self.s_max)


def smoother_from_yaml(d: dict) -> TemporalSmoother:
    cfg = SmootherConfig(
        dt_ref=d.get("dt_ref", SmootherConfig.dt_ref),
        dt_min=d.get("dt_min", SmootherConfig.dt_min),
        dt_max=d.get("dt_max", SmootherConfig.dt_max),
        lambda_acc=d.get("lambda_acc", SmootherConfig.lambda_acc),
        lambda_time=d.get("lambda_time", SmootherConfig.lambda_time),
        lambda_v=d.get("lambda_v", SmootherConfig.lambda_v),
        v_max=d.get("v_max", SmootherConfig.v_max),
        horizon=d.get("horizon", SmootherConfig.horizon),
        stride=d.get("stride", SmootherConfig.stride),
        optim_dims=tuple(d.get("optim_dims", SmootherConfig.optim_dims)),
    )
    return TemporalSmoother(cfg)
