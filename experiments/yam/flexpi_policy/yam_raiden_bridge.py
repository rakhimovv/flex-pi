"""raiden ``ModelBridge`` wrapper for the YAM FlexPi deployment.

Direct analog of ``yam_openpi/deployment/openpi_bridge.py`` and
``maniflow_bridge.py``: a thin adapter that plugs ``YamFlexPiBridge``
(this directory's offline-friendly business-logic bridge) into raiden's
control loop. raiden owns hardware (ZED capture, motor control, FK on
proprio); FlexPi's bridge owns model-specific logic (per-cam resize,
intrinsics + depth delivery, normalize / denormalize, Yam32DRelativeAction
inversion, action chunking).

Two action sources, both selected from the 32D absolute action returned by
``YamFlexPiPolicy.infer_action_chunk`` (``yam_eef.STATE_LAYOUT``):

- ``"model_joint"`` (default, recommended starting path):
    ``predict()`` returns ``(14,)`` joint command in raiden's ``[R, L]`` order.
    Run with ``rd infer --action-type joint``.

- ``"eef_ik"``:
    ``predict()`` returns ``(20,)`` EE pose + grip layout for raiden's
    ``_ee_pose_to_joint_cmd`` to IK-solve. Run with
    ``rd infer --action-type ee_pose``. Includes the documented L↔R swap
    (see ``YamFlexPiBridge._abs32_to_command`` → ``_eef32_action_to_ee_pose20``).

Usage from CLI::

    # Joint control (no IK, simplest first run).
    rd infer \\
        --bridge experiments.yam.flexpi_policy.yam_raiden_bridge:YamFlexPiRaidenBridge \\
        --ckpt_path /path/to/runs/.../checkpoints/.../step_XXXXX.pt \\
        --action_hz 30 \\
        --action-type joint \\
        --bridge-kwargs prompt="put the plate to the rack" \\
        --bridge-kwargs replan_steps=8 \\
        --bridge-kwargs action_source=model_joint

    # EE pose + raiden IK.
    rd infer \\
        --bridge experiments.yam.flexpi_policy.yam_raiden_bridge:YamFlexPiRaidenBridge \\
        --ckpt_path /path/to/.pt \\
        --action_hz 30 \\
        --action-type ee_pose \\
        --bridge-kwargs action_source=eef_ik \\
        --bridge-kwargs prompt="..."

Or via this file's standalone ``main()`` (mirrors ``openpi_bridge.main()``):

    python -m experiments.yam.flexpi_policy.yam_raiden_bridge \\
        --ckpt_path /path/to/.pt \\
        --action_hz 30 \\
        --action_source model_joint \\
        --action_type joint \\
        --prompt "..." \\
        --replan_steps 8

Performance knobs forwarded via ``--bridge-kwargs`` (all optional; sane
defaults in ``YamFlexPiBridge``):
    - ``num_inference_steps=4``: cuts denoise from 10→4 steps.
    - ``replan_steps=N``: how many cmds to execute per inference. With
      ``action_horizon=32`` and ``action_hz=30`` you can go up to N=24.
    - ``device=cuda``, ``mixed_precision=bf16``.

Observation contract:
    raiden delivers a ``chiral.types.Observation`` with ``.cameras`` and
    ``.proprios``. Camera names are ``head / left_wrist / right_wrist``;
    ``YamFlexPiBridge`` already maps these to FlexPi-canonical
    ``cam_high / cam_left_wrist / cam_right_wrist``. The 32D state is
    assembled by ``YamFlexPiBridge`` via ``_raiden_to_eef32_state``
    (lazy-import of raiden's MuJoCo FK), so ``proprios`` only needs
    ``follower_r_joint_pos`` and ``follower_l_joint_pos`` from raiden's
    standard contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

# Path bootstrap so this file is importable without an installed wheel —
# matches deploy_policy.py / yam_bridge.py.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# raiden is required at module-import time when this file is actually USED.
# To allow `python -c "import experiments.yam.flexpi_policy.yam_raiden_bridge"`
# to succeed in environments without raiden installed (so ``ast.parse`` and
# similar tooling work), we degrade ``ModelBridge`` to a stub. ``load()``
# raises a clear error in that case.
try:
    from raiden.inference import ModelBridge as _ModelBridge
    _RAIDEN_AVAILABLE = True
except ImportError:
    _RAIDEN_AVAILABLE = False

    class _ModelBridge:  # type: ignore[no-redef]
        """Stub when raiden is not installed. Class is importable for inspection;
        ``load()`` will raise."""
        pass

from experiments.yam.flexpi_policy.yam_bridge import (  # noqa: E402
    YamFlexPiBridge,
    _RAIDEN_TO_FLEXPI_CAM,
)
from experiments.yam.flexpi_policy.deploy_policy import (  # noqa: E402
    _CAM_ORDER,
    _D,
    load_deploy_defaults,
)


def _parse_kwarg_bool(value: Any, default: bool) -> bool:
    """Parse a CLI-style bool from ``--bridge-kwargs`` (which arrive as strings).

    Verbatim from ``maniflow_bridge._parse_kwarg_bool`` so the parsing
    convention matches what raiden users already know.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    raise ValueError(f"cannot parse bool from {value!r}")


def _parse_kwarg_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "none", "null"):
        return None
    return int(s)


def _parse_kwarg_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "none", "null"):
        return None
    return float(s)


class YamFlexPiRaidenBridge(_ModelBridge):
    """raiden ``ModelBridge`` adapter for ``YamFlexPiBridge``.

    Construction is cheap (just stores knobs); ``load()`` does the heavy work
    (instantiating ``YamFlexPiBridge`` which loads the WAN backbone, T5,
    DINO, the trained checkpoint, and the processor). ``predict()`` returns
    one motor command per call; the underlying replan queue lives inside
    ``YamFlexPiBridge``.
    """

    def __init__(
        self,
        action_source: Literal["model_joint", "eef_ik"] = "model_joint",
        # Defaults from configs/real_yam.yaml; --bridge-kwargs still override.
        replan_steps: int = _D["replan_steps"],
        action_horizon: Optional[int] = _D["action_horizon"],
        num_inference_steps: int = _D["num_inference_steps"],
        prompt: str = "",
        verbose: bool = False,
        **kwargs,
    ) -> None:
        if action_source not in ("model_joint", "eef_ik"):
            raise ValueError(
                f"action_source must be 'model_joint' or 'eef_ik'; got {action_source!r}"
            )
        # Heavy work deferred to load(); store config only.
        self._bridge: Optional[YamFlexPiBridge] = None
        self._action_source = action_source
        self._replan_steps = int(replan_steps)
        self._action_horizon = action_horizon
        self._num_inference_steps = int(num_inference_steps)
        self._prompt = str(prompt)
        self._verbose = bool(verbose)
        # Per-step diagnostics state.
        self._step = 0
        self._first_obs_dumped = False
        self._t_infer_sum = 0.0
        self._n_fresh = 0

    # --- raiden ModelBridge interface --------------------------------------

    def load(self, ckpt_path: str, **kwargs) -> None:
        """Build the underlying ``YamFlexPiBridge`` and load the model.

        ``ckpt_path`` is raiden's ``--ckpt_path``. All other knobs flow
        through ``--bridge-kwargs key=value`` and are parsed here.

        Recognised ``--bridge-kwargs``:

        - ``prompt`` (str): task instruction. If omitted at load time, can
          also be set later via ``set_prompt()``.
        - ``action_source`` (``model_joint`` | ``eef_ik``): overrides ctor.
        - ``replan_steps`` (int): cmds executed per inference.
        - ``action_horizon`` (int): override the model's action horizon.
        - ``num_inference_steps`` (int): denoise step count (default 10;
          drop to 4 for ~2.5x speedup).
        - ``device`` (str): default "cuda".
        - ``mixed_precision`` (``no``|``fp16``|``bf16``): default bf16.
        - ``dataset_stats_path`` (str): optional override; auto-walked from
          ``ckpt_path`` if omitted.
        - ``intrinsics_json`` (str): optional fallback K source. The bridge
          prefers per-step ``cam.intrinsics`` from the raiden observation,
          but if those are missing the policy falls back to this.
        - ``seed`` (int | "none"): inference seed.
        - ``sigma_shift`` (float | "none"): denoise sigma shift override.
        - ``text_cfg_scale`` (float).
        - ``negative_prompt`` (str).
        - ``verbose`` (bool): print per-obs / per-step diagnostics.

        Raises:
            ImportError: if raiden is not installed at runtime — this
                wrapper has no purpose without it.
        """
        if not _RAIDEN_AVAILABLE:
            raise ImportError(
                "YamFlexPiRaidenBridge.load() requires the `raiden` package "
                "to be importable. Run inside the raiden venv (e.g. "
                "/home/<user>/projects/YAM_robot/.venv) or install raiden."
            )

        # --- override ctor knobs from --bridge-kwargs ---
        if "action_source" in kwargs:
            asrc = kwargs["action_source"]
            if asrc not in ("model_joint", "eef_ik"):
                raise ValueError(
                    f"action_source must be 'model_joint' or 'eef_ik'; got {asrc!r}"
                )
            self._action_source = asrc
        if "replan_steps" in kwargs:
            self._replan_steps = int(kwargs["replan_steps"])
        if "action_horizon" in kwargs:
            ah = _parse_kwarg_optional_int(kwargs["action_horizon"])
            self._action_horizon = ah
        if "num_inference_steps" in kwargs:
            self._num_inference_steps = int(kwargs["num_inference_steps"])
        if "prompt" in kwargs:
            self._prompt = str(kwargs["prompt"])
        if "verbose" in kwargs:
            self._verbose = _parse_kwarg_bool(kwargs["verbose"], self._verbose)

        # Fallbacks are configs/real_yam.yaml, so a knob the operator did not
        # pass reads the same value the factory would have used.
        device = str(kwargs.get("device", _D["device"]))
        mixed_precision = str(kwargs.get("mixed_precision", _D["mixed_precision"]))
        dataset_stats_path = kwargs.get("dataset_stats_path") or None
        intrinsics_json = kwargs.get("intrinsics_json") or None
        seed = _parse_kwarg_optional_int(kwargs.get("seed", _D["seed"]))
        sigma_shift = _parse_kwarg_optional_float(
            kwargs.get("sigma_shift", _D["sigma_shift"])
        )
        text_cfg_scale = float(kwargs.get("text_cfg_scale", _D["text_cfg_scale"]))
        negative_prompt = str(kwargs.get("negative_prompt", _D["negative_prompt"]))
        # Perf knobs.
        offload_text_encoder = _parse_kwarg_bool(
            kwargs.get("offload_text_encoder"), _D["offload_text_encoder"]
        )
        torch_compile = _parse_kwarg_bool(
            kwargs.get("torch_compile"), _D["torch_compile"]
        )
        torch_compile_mode = str(
            kwargs.get("torch_compile_mode", _D["torch_compile_mode"])
        )
        quantization_raw = kwargs.get("quantization", _D["quantization"])
        quantization = (
            None if quantization_raw is None or str(quantization_raw).strip().lower() in ("", "none", "null")
            else str(quantization_raw)
        )

        print(
            f"[yam-raiden-bridge] load: ckpt_path={ckpt_path}  "
            f"action_source={self._action_source}  replan_steps={self._replan_steps}  "
            f"num_inference_steps={self._num_inference_steps}  prompt={self._prompt!r}  "
            f"torch_compile={torch_compile}  offload_text_encoder={offload_text_encoder}"
        )

        self._bridge = YamFlexPiBridge(
            checkpoint_path=str(ckpt_path),
            dataset_stats_path=dataset_stats_path,
            intrinsics_json=intrinsics_json,
            device=device,
            mixed_precision=mixed_precision,
            action_horizon=self._action_horizon,
            num_inference_steps=self._num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            text_cfg_scale=text_cfg_scale,
            negative_prompt=negative_prompt,
            offload_text_encoder=offload_text_encoder,
            torch_compile=torch_compile,
            torch_compile_mode=torch_compile_mode,
            quantization=quantization,
            replan_steps=self._replan_steps,
            action_source=self._action_source,
            prompt=self._prompt,
            verbose=self._verbose,
        )
        self._step = 0
        self._first_obs_dumped = False
        self._t_infer_sum = 0.0
        self._n_fresh = 0

    def reset(self) -> None:
        """Called by raiden when the robot is homed between episodes."""
        if self._bridge is not None:
            self._bridge.reset()
        self._step = 0
        self._t_infer_sum = 0.0
        self._n_fresh = 0
        # Keep _first_obs_dumped True after the first episode — one debug
        # dump per process is enough.

    def predict(self, obs) -> np.ndarray:
        """Per-step entry. Returns ``(14,)`` or ``(20,)`` motor command."""
        if self._bridge is None:
            raise RuntimeError(
                "predict() called before load(). raiden's RaidenInferenceLoop "
                "should call load() during construction."
            )

        # First-step debug dump: mirrors openpi_bridge's ~/openpi_obs_debug/
        # dump pattern. Useful for visually verifying camera streams + state
        # before trusting the closed-loop run.
        if self._step == 0 and not self._first_obs_dumped:
            self._dump_first_obs(obs)
            self._first_obs_dumped = True

        result = self._bridge.act(obs)
        cmd = result["command"]
        if not result["from_queue"]:
            self._t_infer_sum += result["infer_s"]
            self._n_fresh += 1

        # Periodic heartbeat. Mirrors the rate openpi_bridge prints at
        # (every 50 steps).
        if self._step % 50 == 1:
            avg_infer_ms = (
                1000.0 * self._t_infer_sum / max(self._n_fresh, 1)
            )
            print(
                f"[yam-raiden-bridge] step={self._step:4d} "
                f"src={'queue' if result['from_queue'] else 'fresh':5s} "
                f"queue_len={result['queue_len']:2d} "
                f"infer_avg={avg_infer_ms:.1f} ms "
                f"cmd_range=[{float(cmd.min()):+.3f}, {float(cmd.max()):+.3f}]"
            )
        self._step += 1
        return cmd

    # --- helpers -----------------------------------------------------------

    def set_prompt(self, prompt: str) -> None:
        """Update the task instruction at runtime (e.g. between episodes)."""
        self._prompt = str(prompt)
        if self._bridge is not None:
            self._bridge.set_prompt(prompt)

    def _dump_first_obs(self, obs: Any) -> None:
        """Save the first raiden observation to ``~/yam_flexpi_obs_debug/``.

        Mirrors ``openpi_bridge._preprocess`` step-0 dump shape:
        - RGB → PNG (one per cam, named by raiden cam name)
        - Depth (if present) → 16-bit grayscale PNG in mm
        - Intrinsics, state_32 / proprio keys, prompt → meta.json
        """
        out_dir = Path(os.path.expanduser("~/yam_flexpi_obs_debug"))
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            import cv2  # raiden venv typically has cv2; soft-fail otherwise.
            have_cv2 = True
        except ImportError:
            have_cv2 = False

        meta: dict = {
            "step": 0,
            "action_source": self._action_source,
            "replan_steps": self._replan_steps,
            "prompt": self._prompt,
            "raiden_to_flexpi_cam": _RAIDEN_TO_FLEXPI_CAM,
            "flexpi_canonical_cam_order": list(_CAM_ORDER),
        }

        cameras = list(getattr(obs, "cameras", []) or [])
        for cam in cameras:
            name = getattr(cam, "name", None)
            if name is None:
                continue
            image = getattr(cam, "image", None)
            depth = getattr(cam, "depth", None)
            K = getattr(cam, "intrinsics", None)
            if image is not None and have_cv2:
                cv2.imwrite(
                    str(out_dir / f"rgb_{name}.png"),
                    cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
                )
            if depth is not None and have_cv2:
                d = np.asarray(depth)
                if d.dtype in (np.float32, np.float64):
                    d = (d * 1000.0).clip(0, 65535).astype(np.uint16)
                elif d.dtype != np.uint16:
                    d = d.astype(np.uint16)
                cv2.imwrite(str(out_dir / f"depth_{name}_mm.png"), d)
            if K is not None:
                meta[f"intrinsics_{name}"] = np.asarray(K, dtype=np.float32).tolist()

        proprios = dict(getattr(obs, "proprios", {}) or {})
        for k, v in proprios.items():
            meta[f"proprios_{k}"] = np.asarray(v, dtype=np.float32).tolist()

        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"[yam-raiden-bridge] dumped first observation to {out_dir}/")


# ---------------------------------------------------------------------------
# Standalone CLI entry — direct counterpart of openpi_bridge.main()
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    # Defaults come from configs/real_yam.yaml so `--help` shows the same
    # numbers the `rd infer --bridge` path uses.
    _d = load_deploy_defaults()
    p = argparse.ArgumentParser(
        description=(
            "Deploy YAM FlexPi via raiden's RaidenInferenceLoop. "
            "Mirrors yam_openpi/deployment/openpi_bridge.py's entry point."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt_path", required=True, help="FlexPi checkpoint (.pt).")
    p.add_argument("--action_hz", type=float, default=30.0,
                   help="Robot control frequency (must match dataset fps).")
    p.add_argument(
        "--action_source", default="model_joint",
        choices=["model_joint", "eef_ik"],
        help="model_joint = (14,) joint cmd; eef_ik = (20,) EE pose for raiden IK.",
    )
    p.add_argument(
        "--action_type", default="joint", choices=["joint", "ee_pose"],
        help="Forwarded to raiden. Must match action_source: "
             "model_joint+joint or eef_ik+ee_pose.",
    )
    p.add_argument("--replan_steps", type=int, default=_d["replan_steps"],
                   help="Cmds executed per inference (1..action_horizon).")
    p.add_argument("--num_inference_steps", type=int, default=_d["num_inference_steps"],
                   help="Denoise steps (drop to 4 for ~2.5x speedup).")
    p.add_argument("--action_horizon", type=int, default=_d["action_horizon"],
                   help="Override action horizon (default: from trained config).")
    p.add_argument("--prompt", default="", help="Task instruction.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--dataset_stats_path", default=None,
                   help="Override; auto-walked from --ckpt_path if omitted.")
    p.add_argument("--intrinsics_json", default=None,
                   help="Fallback intrinsics JSON when raiden cam.intrinsics is absent.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sigma_shift", type=float, default=None)
    p.add_argument("--text_cfg_scale", type=float, default=1.0)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--verbose", action="store_true")
    # Perf knobs (forwarded into YamFlexPiPolicy via --bridge-kwargs).
    p.add_argument("--offload_text_encoder", action="store_true",
                   help="Keep T5 on CPU to save ~10 GB VRAM.")
    p.add_argument("--torch_compile", action="store_true",
                   help="Compile the denoise step + capture CUDA graph (~3x speedup, "
                        "10-30 s warmup at startup).")
    p.add_argument("--torch_compile_mode", default="reduce-overhead",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--quantization", default=None,
                   help='None or "int8".')
    # raiden hardware args — forwarded to RaidenInferenceLoop kwargs.
    p.add_argument("--camera_config_file", default="./config/camera_config.json")
    p.add_argument("--calibration_file", default="./config/calibration_results.json")
    p.add_argument("--stereo_method", default="zed", choices=["zed", "ffs"])
    p.add_argument("--depth_mode", default="NEURAL_LIGHT")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    # Action-source / action-type compatibility guard. Same shape as
    # openpi_bridge / maniflow_bridge.
    if args.action_source == "model_joint" and args.action_type != "joint":
        sys.exit(
            "ERROR: --action_source model_joint requires --action_type joint."
        )
    if args.action_source == "eef_ik" and args.action_type != "ee_pose":
        sys.exit(
            "ERROR: --action_source eef_ik requires --action_type ee_pose."
        )

    if not _RAIDEN_AVAILABLE:
        sys.exit(
            "ERROR: raiden is not importable in this Python env. Run from "
            "the raiden venv (e.g. /home/<user>/projects/YAM_robot/.venv)."
        )

    from raiden.inference import RaidenInferenceLoop  # noqa: WPS433 (runtime import)

    bridge = YamFlexPiRaidenBridge(
        action_source=args.action_source,
        replan_steps=args.replan_steps,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        prompt=args.prompt,
        verbose=args.verbose,
    )

    bridge_kwargs = {
        "action_source": args.action_source,
        "replan_steps": args.replan_steps,
        "num_inference_steps": args.num_inference_steps,
        "prompt": args.prompt,
        "device": args.device,
        "mixed_precision": args.mixed_precision,
        "text_cfg_scale": args.text_cfg_scale,
        "negative_prompt": args.negative_prompt,
        "verbose": args.verbose,
        "offload_text_encoder": args.offload_text_encoder,
        "torch_compile": args.torch_compile,
        "torch_compile_mode": args.torch_compile_mode,
    }
    if args.action_horizon is not None:
        bridge_kwargs["action_horizon"] = args.action_horizon
    if args.dataset_stats_path is not None:
        bridge_kwargs["dataset_stats_path"] = args.dataset_stats_path
    if args.intrinsics_json is not None:
        bridge_kwargs["intrinsics_json"] = args.intrinsics_json
    if args.seed is not None:
        bridge_kwargs["seed"] = args.seed
    if args.sigma_shift is not None:
        bridge_kwargs["sigma_shift"] = args.sigma_shift
    if args.quantization is not None:
        bridge_kwargs["quantization"] = args.quantization

    loop = RaidenInferenceLoop(
        bridge=bridge,
        ckpt_path=args.ckpt_path,
        action_hz=args.action_hz,
        bridge_kwargs=bridge_kwargs,
        camera_config_file=args.camera_config_file,
        calibration_file=args.calibration_file,
        stereo_method=args.stereo_method,
        depth_mode=args.depth_mode,
        action_type=args.action_type,
    )
    loop.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
