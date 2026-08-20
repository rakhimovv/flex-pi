"""Thin wrapper exposing ``temporal_smoother`` to the WS server.

The smoother math lives in ``temporal_smoother.py`` (OSQP port of
realtime-vla-v2's ``TimeParameterizationMPC``). This wrapper just supplies
YAM-specific defaults and CLI plumbing.

YAM-specific choices vs. the RoboTwin default:
  - ``optim_dims = (20..31)``: the joint slice of the 32-D ``yam_eef``
    chunk. Production deploy uses ``action_source=model_joint``, which
    consumes only these dims. Smoothing the EEF slice (xyz + rot6d) is
    wasted work for that path; rot6d is non-Euclidean so ``||dp||`` is
    meaningless anyway.
  - ``dt_ref = 1/30``: matches raiden's 30 Hz control loop (not the 50 Hz
    dataset rate the RoboTwin default assumes).
"""
from __future__ import annotations

import argparse
from typing import Optional

# Adds ``flex-pi`` to sys.path -- already done by the caller
# (serve_yam_flexpi.py) before this module is imported.
from .temporal_smoother import SmootherConfig, TemporalSmoother


YAM_JOINT_OPTIM_DIMS = tuple(range(20, 32))  # l_joint(6) + r_joint(6)


def add_smoother_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("temporal smoother (off by default)")
    g.add_argument("--use-smoother", action="store_true",
                   help="Enable server-side OSQP temporal smoother on each chunk.")
    g.add_argument("--smoother-dt-ref", type=float, default=1.0 / 30.0,
                   help="Nominal step duration (s). Match raiden's --action_hz.")
    g.add_argument("--smoother-dt-min", type=float, default=1.0 / 60.0,
                   help="Lower bound on per-step duration. Allows 2x speedup.")
    g.add_argument("--smoother-dt-max", type=float, default=1.0 / 15.0,
                   help="Upper bound on per-step duration. Allows 2x slowdown.")
    g.add_argument("--smoother-lambda-acc", type=float, default=10.0,
                   help="Acceleration penalty weight (higher = smoother).")
    g.add_argument("--smoother-lambda-time", type=float, default=1.0,
                   help="Stay-near-dt_ref penalty weight.")
    g.add_argument("--smoother-horizon", type=int, default=32)
    g.add_argument("--smoother-stride", type=int, default=16)
    g.add_argument("--smoother-optim-dims", type=str,
                   default=",".join(str(i) for i in YAM_JOINT_OPTIM_DIMS),
                   help="Comma-separated dim indices into the 32-D yam_eef chunk.")


def build_smoother_from_args(args: argparse.Namespace) -> Optional[TemporalSmoother]:
    if not getattr(args, "use_smoother", False):
        return None
    dims = tuple(int(s) for s in str(args.smoother_optim_dims).split(",") if s.strip())
    cfg = SmootherConfig(
        dt_ref=float(args.smoother_dt_ref),
        dt_min=float(args.smoother_dt_min),
        dt_max=float(args.smoother_dt_max),
        lambda_acc=float(args.smoother_lambda_acc),
        lambda_time=float(args.smoother_lambda_time),
        horizon=int(args.smoother_horizon),
        stride=int(args.smoother_stride),
        optim_dims=dims,
    )
    return TemporalSmoother(cfg)
