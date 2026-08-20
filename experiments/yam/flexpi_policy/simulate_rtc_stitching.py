"""Simulate one RTC stitching cycle on recorded YAM data.

Loads the FlexPi policy, picks an episode + a frame ``T_obs``, runs inference
to get ``chunk_old``. Then advances to ``T_obs + offset`` (simulating inference
latency), reads obs from disk again, runs a second inference to get
``chunk_new``. Applies the RTC three-region merge (cosine blend over
``merge_steps`` overlap rows, hard-overwrite the rest, append non-overlap) and
emits three time-aligned trajectories:

    traj_old      = chunk_old[0:H]                                  # never replanned
    traj_hardcut  = chunk_old[0:offset] ⊕ chunk_new[0:H-offset]     # cold cut at T_recv
    traj_merged   = chunk_old[0:offset] ⊕ rtc_merge(old_tail, useful_new, merge_steps)

Compares per-step Δ and acceleration on the joint slice (dims 20..31, the
slice ``model_joint`` action source actually consumes) and saves both
``.npz`` arrays and an optional matplotlib PNG.

Usage::

    python -m experiments.yam.flexpi_policy.simulate_rtc_stitching \\
        --ckpt-path /path/to/ckpt/step_NNNN.pt \\
        --data-dir /path/to/data/stack_cup_100 \\
        --episode-idx 0 --frame-obs 300 --offset 4 --merge-steps 4 \\
        --torch-compile --offload-text-encoder \\
        --output-dir /tmp/rtc_sim
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Path bootstrap.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.yam.flexpi_policy.deploy_policy import (  # noqa: E402
    build_policy_from_checkpoint,
)
from experiments.yam.flexpi_policy.smoke_test import (  # noqa: E402
    build_obs_recorded,
    load_gt_action_chunk,
)


# ---------------------------------------------------------------- RTC merge


def rtc_merge(
    old_tail: np.ndarray,
    useful_new: np.ndarray,
    merge_steps: int,
) -> np.ndarray:
    """Apply the RTC three-region merge — pure function, no broker state.

    Mirrors ``RTCStepBroker._merge_response`` (which operates on the deque
    in-place). Here we work on plain arrays so the simulator can compare
    the result against the un-stitched baselines.

    Parameters
    ----------
    old_tail : np.ndarray, shape ``[L, D]``
        Remaining un-executed rows of the previously emitted chunk at
        receive time. Row 0 corresponds to the step that's about to fire
        when the new chunk arrives.
    useful_new : np.ndarray, shape ``[M, D]``
        Useful (post-offset) portion of the freshly-arrived chunk; row 0
        is for the same step as ``old_tail[0]``.
    merge_steps : int
        Cosine-blend window. Effective blend length is
        ``min(merge_steps, overlap)``.

    Returns
    -------
    np.ndarray, shape ``[max(L, M), D]``
        Stitched trajectory. Indexing:
          - rows ``[0, blend_n)`` are cosine blends of old/new
          - rows ``[blend_n, overlap)`` are pure new (hard-overwrite)
          - rows ``[overlap, M)`` are appended new
          - rows ``[overlap, L)`` (only when L > M) preserve old tail
    """
    if old_tail.ndim != 2 or useful_new.ndim != 2:
        raise ValueError("old_tail and useful_new must be 2-D [_, D]")
    if old_tail.shape[1] != useful_new.shape[1]:
        raise ValueError(
            f"D mismatch: old_tail.shape[1]={old_tail.shape[1]}, "
            f"useful_new.shape[1]={useful_new.shape[1]}"
        )
    if merge_steps < 0:
        raise ValueError(f"merge_steps must be >= 0; got {merge_steps}")

    L = old_tail.shape[0]
    M = useful_new.shape[0]
    D = old_tail.shape[1]
    overlap = min(L, M)
    blend_n = min(merge_steps, overlap)

    out_len = max(L, M)
    out = np.empty((out_len, D), dtype=np.float32)

    # Region 1: cosine blend.
    for i in range(blend_n):
        w = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / (blend_n + 1))
        out[i] = (1.0 - w) * old_tail[i].astype(np.float32) + w * useful_new[i].astype(np.float32)

    # Region 2: hard-overwrite rest of overlap.
    for i in range(blend_n, overlap):
        out[i] = useful_new[i].astype(np.float32)

    # Region 3a: appended new rows (only if M > L).
    for i in range(overlap, M):
        out[i] = useful_new[i].astype(np.float32)

    # Region 3b: preserved old tail past where new chunk reaches (only if L > M).
    for i in range(overlap, L):
        out[i] = old_tail[i].astype(np.float32)

    return out


# ---------------------------------------------------------------- metrics


def step_deltas(traj: np.ndarray, dims: Optional[slice] = None) -> np.ndarray:
    """Per-step Euclidean norm of ``traj[t+1] - traj[t]`` over ``dims``."""
    sub = traj if dims is None else traj[:, dims]
    if sub.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(np.diff(sub, axis=0).astype(np.float64), axis=1)


def acceleration(traj: np.ndarray, dims: Optional[slice] = None) -> np.ndarray:
    """Per-step Euclidean norm of ``traj[t+1] - 2*traj[t] + traj[t-1]``."""
    sub = traj if dims is None else traj[:, dims]
    if sub.shape[0] < 3:
        return np.zeros(0, dtype=np.float64)
    acc = sub[2:].astype(np.float64) - 2 * sub[1:-1].astype(np.float64) + sub[:-2].astype(np.float64)
    return np.linalg.norm(acc, axis=1)


def fmt_metrics(name: str, traj: np.ndarray, dims: Optional[slice]) -> str:
    d = step_deltas(traj, dims)
    a = acceleration(traj, dims)
    return (
        f"{name:<14} len={traj.shape[0]:3d}  "
        f"max|Δ|={d.max() if d.size else 0:.4f}  "
        f"rms|Δ|={float(np.sqrt(np.mean(d ** 2))) if d.size else 0:.4f}  "
        f"max|acc|={a.max() if a.size else 0:.4f}  "
        f"rms|acc|={float(np.sqrt(np.mean(a ** 2))) if a.size else 0:.4f}"
    )


# ---------------------------------------------------------------- plot


def _maybe_plot(
    *,
    traj_old: np.ndarray,
    traj_hardcut: np.ndarray,
    traj_merged: np.ndarray,
    gt_chunk: Optional[np.ndarray],
    offset: int,
    merge_steps: int,
    blend_n: int,
    dims_plot: range,
    out_path: Path,
    title: str,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[sim] matplotlib not available; skipping plot")
        return False

    H = traj_old.shape[0]
    n_dims = len(dims_plot)
    n_cols = 4
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.0 * n_rows), sharex=True)
    axes = np.atleast_2d(axes)

    time_axis = np.arange(H)
    for ax_idx, joint_dim in enumerate(dims_plot):
        ax = axes[ax_idx // n_cols, ax_idx % n_cols]
        ax.plot(time_axis, traj_old[:, joint_dim], color="0.5", linestyle="-",
                label="old (no replan)", alpha=0.6, linewidth=1.0)
        ax.plot(time_axis, traj_hardcut[:, joint_dim], color="C3", linestyle="--",
                label="hard cut (no merge)", alpha=0.85, linewidth=1.0)
        ax.plot(time_axis, traj_merged[:, joint_dim], color="C0", linestyle="-",
                label="merged (RTC)", linewidth=1.5)
        if gt_chunk is not None and gt_chunk.shape[1] > joint_dim:
            t_gt = np.arange(min(gt_chunk.shape[0], H))
            ax.plot(t_gt, gt_chunk[: len(t_gt), joint_dim], color="C2",
                    linestyle=":", label="ground truth", alpha=0.7, linewidth=1.0)
        ax.axvline(offset, color="0.3", linestyle=":", alpha=0.6)
        if blend_n > 0:
            ax.axvspan(offset, offset + blend_n, color="C0", alpha=0.08)
        ax.set_title(f"dim {joint_dim}")
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(loc="best", fontsize=8)

    # Hide unused subplots.
    for ax_idx in range(n_dims, n_rows * n_cols):
        axes[ax_idx // n_cols, ax_idx % n_cols].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt-path", required=True, help="Path to FlexPi .pt checkpoint")
    p.add_argument("--data-dir", required=True, help="LeRobot dataset root (e.g. stack_cup_100/)")
    p.add_argument("--episode-idx", type=int, default=0)
    p.add_argument("--frame-obs", type=int, default=300,
                   help="T_obs — frame at which we 'send' the first obs")
    p.add_argument("--offset", type=int, default=4,
                   help="Steps between submit and receive (simulated inference latency)")
    p.add_argument("--merge-steps", type=int, default=4,
                   help="Cosine-blend window; 0 = hard cut at boundary")

    # Model build args (mirrors serve_yam_flexpi.py).
    p.add_argument("--action-horizon", type=int, default=32)
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--torch-compile", action="store_true",
                   help="Enable torch.compile (~50 s warmup but 3x faster inference)")
    p.add_argument("--torch-compile-mode", default="reduce-overhead",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--offload-text-encoder", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)

    # Output.
    p.add_argument("--output-dir", default="/tmp/rtc_sim")
    p.add_argument("--plot-dims", default="20-31",
                   help="Range of action dims to plot, e.g. '20-31' or '0-2,9-11'.")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-gt", action="store_true", help="Skip ground-truth chunk loading.")
    args = p.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    H = int(args.action_horizon)
    offset = int(args.offset)
    if offset < 0 or offset >= H:
        raise SystemExit(f"--offset must be in [0, action_horizon={H}); got {offset}")

    # 1. Build policy.
    print(f"[sim] Loading model from {args.ckpt_path}", flush=True)
    t0 = time.perf_counter()
    policy = build_policy_from_checkpoint(
        args.ckpt_path,
        action_horizon=H,
        num_inference_steps=args.num_inference_steps,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
        mixed_precision=args.mixed_precision,
        offload_text_encoder=args.offload_text_encoder,
        device=args.device,
        seed=args.seed,
    )
    print(f"[sim] Policy ready in {time.perf_counter() - t0:.1f} s", flush=True)

    # 2. Inference at T_obs.
    data_dir = Path(args.data_dir).expanduser().resolve()
    print(f"\n[sim] Loading obs @ episode={args.episode_idx} frame={args.frame_obs}", flush=True)
    obs_old, instruction = build_obs_recorded(data_dir, args.episode_idx, args.frame_obs)
    print(f"[sim] instruction: {instruction!r}", flush=True)

    t0 = time.perf_counter()
    chunk_old = policy.infer_action_chunk(obs_old, instruction)
    print(f"[sim] chunk_old shape={chunk_old.shape} in {time.perf_counter() - t0:.2f} s", flush=True)
    if chunk_old.shape != (H, 32):
        raise SystemExit(f"unexpected chunk_old shape {chunk_old.shape}, expected ({H}, 32)")

    # 3. Inference at T_obs + offset.
    print(
        f"\n[sim] Loading obs @ episode={args.episode_idx} frame={args.frame_obs + offset} "
        f"(T_obs + offset)",
        flush=True,
    )
    obs_new, instruction_new = build_obs_recorded(data_dir, args.episode_idx, args.frame_obs + offset)
    if instruction_new != instruction:
        # Shouldn't happen on a single episode but defensive.
        print(f"[sim] WARN: instruction changed at T_recv: {instruction_new!r}")
    t0 = time.perf_counter()
    chunk_new = policy.infer_action_chunk(obs_new, instruction_new)
    print(f"[sim] chunk_new shape={chunk_new.shape} in {time.perf_counter() - t0:.2f} s", flush=True)

    # 4. Build three time-aligned trajectories (length H, aligned to T_obs).
    old_tail = chunk_old[offset:]            # [H - offset, 32]
    useful_new = chunk_new[: H - offset]     # [H - offset, 32]

    traj_old = chunk_old.astype(np.float32).copy()
    traj_hardcut = np.concatenate(
        [chunk_old[:offset].astype(np.float32), useful_new.astype(np.float32)], axis=0,
    )
    merged_segment = rtc_merge(old_tail, useful_new, args.merge_steps)
    traj_merged = np.concatenate(
        [chunk_old[:offset].astype(np.float32), merged_segment], axis=0,
    )

    overlap = min(old_tail.shape[0], useful_new.shape[0])
    blend_n = min(int(args.merge_steps), overlap)
    print(
        f"\n[sim] Stitch geometry: H={H} offset={offset} overlap={overlap} "
        f"blend_n={blend_n} merge_steps={args.merge_steps}",
        flush=True,
    )

    # 5. Optional GT chunk.
    gt_chunk: Optional[np.ndarray] = None
    if not args.no_gt:
        try:
            gt = load_gt_action_chunk(data_dir, args.episode_idx, args.frame_obs, H)
            if gt.shape[0] < H:
                print(f"[sim] GT chunk truncated: {gt.shape[0]} < {H}")
            gt_chunk = np.asarray(gt, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            print(f"[sim] GT chunk unavailable: {exc!r}")

    # 6. Metrics on the joint slice (model_joint action source consumes dims 20..31).
    print("\n[sim] Smoothness metrics (joint slice 20..31):")
    JOINT_SLICE = slice(20, 32)
    print(fmt_metrics("old_only", traj_old, JOINT_SLICE))
    print(fmt_metrics("hard_cut", traj_hardcut, JOINT_SLICE))
    print(fmt_metrics("merged", traj_merged, JOINT_SLICE))
    if gt_chunk is not None:
        print(fmt_metrics("ground_truth", gt_chunk, JOINT_SLICE))

    # Boundary-focused metrics: window around the stitch point.
    boundary_window = max(blend_n, 1) + 2
    s = max(0, offset - boundary_window)
    e = min(H, offset + boundary_window + blend_n)
    print(
        f"\n[sim] Boundary smoothness (rows {s}..{e}, joint slice):"
    )
    print(fmt_metrics("old_only", traj_old[s:e], JOINT_SLICE))
    print(fmt_metrics("hard_cut", traj_hardcut[s:e], JOINT_SLICE))
    print(fmt_metrics("merged", traj_merged[s:e], JOINT_SLICE))

    # 7. Save artifacts.
    npz_path = out_dir / "trajectories.npz"
    np.savez(
        npz_path,
        chunk_old=chunk_old,
        chunk_new=chunk_new,
        traj_old=traj_old,
        traj_hardcut=traj_hardcut,
        traj_merged=traj_merged,
        gt_chunk=gt_chunk if gt_chunk is not None else np.zeros((0, 32), dtype=np.float32),
        offset=np.int32(offset),
        merge_steps=np.int32(args.merge_steps),
        blend_n=np.int32(blend_n),
        action_horizon=np.int32(H),
        episode_idx=np.int32(args.episode_idx),
        frame_obs=np.int32(args.frame_obs),
    )
    print(f"\n[sim] Saved arrays → {npz_path}")

    summary = {
        "episode_idx": int(args.episode_idx),
        "frame_obs": int(args.frame_obs),
        "offset": int(offset),
        "merge_steps": int(args.merge_steps),
        "blend_n": int(blend_n),
        "action_horizon": int(H),
        "instruction": instruction,
        "boundary_max_delta_joints": {
            "old_only": float(step_deltas(traj_old[s:e], JOINT_SLICE).max() if (e - s) > 1 else 0.0),
            "hard_cut": float(step_deltas(traj_hardcut[s:e], JOINT_SLICE).max() if (e - s) > 1 else 0.0),
            "merged":   float(step_deltas(traj_merged[s:e], JOINT_SLICE).max() if (e - s) > 1 else 0.0),
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[sim] Saved summary → {summary_path}")

    # 8. Plot (optional).
    if not args.no_plot:
        dims = _parse_dim_range(args.plot_dims)
        title = (
            f"RTC stitch sim — ep {args.episode_idx} T_obs={args.frame_obs} "
            f"offset={offset} merge_steps={args.merge_steps} (blend_n={blend_n})"
        )
        plot_path = out_dir / "trajectories.png"
        ok = _maybe_plot(
            traj_old=traj_old,
            traj_hardcut=traj_hardcut,
            traj_merged=traj_merged,
            gt_chunk=gt_chunk,
            offset=offset,
            merge_steps=args.merge_steps,
            blend_n=blend_n,
            dims_plot=dims,
            out_path=plot_path,
            title=title,
        )
        if ok:
            print(f"[sim] Plot saved → {plot_path}")

    return 0


def _parse_dim_range(spec: str) -> range:
    """Parse '20-31' or '20-31,0-2' (only the FIRST range; for one figure)."""
    first = spec.split(",")[0].strip()
    if "-" in first:
        a, b = first.split("-")
        return range(int(a), int(b) + 1)
    return range(int(first), int(first) + 1)


if __name__ == "__main__":
    raise SystemExit(main())
