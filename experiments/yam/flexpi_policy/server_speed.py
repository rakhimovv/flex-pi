"""Thin wrapper exposing ``speed_adapter`` to the WS server.

The adapter math lives in ``speed_adapter.py``. This wrapper
just supplies YAM-specific defaults and CLI plumbing.

YAM-specific note: the heuristic mode consumes only the joint slice of the
32-D ``yam_eef`` chunk (indices 20..31), so its velocity proxy
``||v_step||`` reflects actual motor motion. Passing the full 32-D chunk
would let the EEF rot6d dims (which are not Euclidean) dominate the norm
and give nonsensical factors.
"""
from __future__ import annotations

import argparse
from typing import Optional

from .speed_adapter import SpeedAdapter, SpeedAdapterConfig


def add_speed_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("speed adapter (off by default)")
    g.add_argument("--speed-adapter", default="off",
                   choices=["off", "heuristic", "learned"],
                   help="Per-step speed factor strategy. Fed into the smoother's s_ref.")
    g.add_argument("--speed-ckpt", default=None,
                   help="Required if --speed-adapter learned.")
    g.add_argument("--speed-factor-lo", type=float, default=0.5)
    g.add_argument("--speed-factor-hi", type=float, default=1.5)
    g.add_argument("--speed-heuristic-alpha", type=float, default=1.0)
    g.add_argument("--speed-heuristic-v-ref", type=float, default=0.05,
                   help="Reference joint-velocity (rad/step) for the heuristic proxy.")
    g.add_argument("--speed-smooth-window", type=int, default=5)


def build_adapter_from_args(args: argparse.Namespace) -> Optional[SpeedAdapter]:
    mode = str(getattr(args, "speed_adapter", "off"))
    if mode == "off":
        return None
    cfg = SpeedAdapterConfig(
        mode=mode,
        factor_lo=float(args.speed_factor_lo),
        factor_hi=float(args.speed_factor_hi),
        ckpt=args.speed_ckpt,
        heuristic_alpha=float(args.speed_heuristic_alpha),
        heuristic_v_ref=float(args.speed_heuristic_v_ref),
        smooth_window=int(args.speed_smooth_window),
    )
    return SpeedAdapter(cfg)
