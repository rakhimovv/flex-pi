"""Thin raiden bridge for YAM FlexPi that talks to a websocket policy server.

Mirrors ``yam_openpi/deployment/maniflow_bridge.py`` for the FlexPi
deployment. The in-process counterpart ``yam_raiden_bridge.py`` loads the
model directly inside the bridge process; this one delegates inference to a
separate server (``scripts/serve_yam_flexpi.py``), so:

  - Bridge starts in <2 s (no 110 s model load).
  - Bridge can be killed/restarted freely without touching the GPU.
  - Server keeps weights resident; survives bridge crashes.

Wire schema is OpenPI-flat (one key per per-cam stream); the server's
``YamFlexPiServerPolicy`` translates back to FlexPi-nested. See
``server_adapter.py``.

Two action sources, same as the in-process bridge (and same byte-identical
permutations as ``yam_openpi/deployment/openpi_bridge.py``):

  - ``model_joint`` (default) → 14D joint command in raiden's [R, L] order.
    Use ``rd infer --action-type joint``.
  - ``eef_ik`` → 20D EE pose for raiden's IK. Includes the mandatory L↔R
    swap (see ``yam_bridge._eef32_action_to_ee_pose20``).
    Use ``rd infer --action-type ee_pose``.

Usage (rd infer)::

    rd infer \\
        --bridge experiments.yam.flexpi_policy.yam_raiden_bridge_ws:YamFlexPiRaidenBridgeWS \\
        --ckpt_path unused \\
        --action_hz 30 --action-type joint \\
        --bridge-kwargs host=localhost \\
        --bridge-kwargs port=8000 \\
        --bridge-kwargs action_horizon=32 \\
        --bridge-kwargs action_source=model_joint \\
        --bridge-kwargs use_depth=true \\
        --bridge-kwargs prompt="..."

Standalone::

    python -m experiments.yam.flexpi_policy.yam_raiden_bridge_ws \\
        --host localhost --port 8000 \\
        --action_source model_joint --action_type joint \\
        --action_horizon 32 --prompt "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np

# Path bootstrap (matches yam_raiden_bridge.py:86-91).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Raiden ModelBridge is required at load() time but we let import succeed
# without raiden so this file can be parsed in the flexpi conda env too.
try:
    from raiden.inference import ModelBridge as _ModelBridge
    _RAIDEN_AVAILABLE = True
except ImportError:
    _RAIDEN_AVAILABLE = False

    class _ModelBridge:  # type: ignore[no-redef]
        """Stub when raiden is not installed."""
        pass

from experiments.yam.flexpi_policy._openpi_vendor import (  # noqa: E402
    action_chunk_broker,
    image_tools,
    websocket_client_policy as _ws,
)
from experiments.yam.flexpi_policy.deploy_policy import _D  # noqa: E402
from experiments.yam.flexpi_policy.yam_bridge import (  # noqa: E402
    _RAIDEN_TO_FLEXPI_CAM,
    _eef32_action_to_ee_pose20,
    _eef32_action_to_joint14_RL,
    _raiden_to_eef32_state,
)


LOG_PREFIX = "[yam-raiden-bridge-ws]"

# Hardcoded fallback if the server metadata is missing per_cam_hw — should
# match what the server reports (and what FlexPi's _PER_CAM_HW declares).
_DEFAULT_PER_CAM_HW: Dict[str, tuple] = {
    "cam_high":        (256, 320),
    "cam_left_wrist":  (224, 224),
    "cam_right_wrist": (224, 224),
}


def _parse_kwarg_bool(value: Any, default: bool) -> bool:
    """Mirror of ``maniflow_bridge._parse_kwarg_bool`` so the convention matches."""
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


def _resize_depth_and_scale_intrinsics(
    depth_mm: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    target_h: int, target_w: int,
):
    """Nearest-neighbor depth resize + per-axis intrinsics rescale.

    Inlined byte-identical to ``maniflow_bridge._resize_depth_and_scale_intrinsics``
    (yam_openpi/deployment/maniflow_bridge.py:624-633). Pure numpy, ~10 lines.
    """
    src_h, src_w = depth_mm.shape
    if (src_h, src_w) == (target_h, target_w):
        return depth_mm, fx, fy, cx, cy
    ys = np.linspace(0, src_h - 1, target_h).round().astype(np.int64)
    xs = np.linspace(0, src_w - 1, target_w).round().astype(np.int64)
    depth_out = depth_mm[ys[:, None], xs[None, :]]
    sx = (target_w - 1) / (src_w - 1) if src_w > 1 else 1.0
    sy = (target_h - 1) / (src_h - 1) if src_h > 1 else 1.0
    return depth_out, fx * sx, fy * sy, cx * sx, cy * sy


class YamFlexPiRaidenBridgeWS(_ModelBridge):
    """Thin raiden ``ModelBridge`` that delegates to a FlexPi policy server."""

    def __init__(
        self,
        action_source: Literal["model_joint", "eef_ik"] = "model_joint",
        action_horizon: int = 32,
        use_depth: bool = True,
        depth_max_mm: int = 65535,
        prompt: str = "",
        verbose: bool = False,
        use_rtc: bool = False,
        merge_steps: int = 4,
        replan_steps: int = _D["replan_steps"],
        **kwargs,
    ) -> None:
        if action_source not in ("model_joint", "eef_ik"):
            raise ValueError(
                f"action_source must be 'model_joint' or 'eef_ik'; got {action_source!r}"
            )
        # Heavy work (websocket connect) is deferred to load().
        self._action_source: Literal["model_joint", "eef_ik"] = action_source
        self._action_horizon = int(action_horizon)
        self._use_depth = bool(use_depth)
        self._depth_max_mm = int(depth_max_mm)
        self._prompt = str(prompt)
        self._verbose = bool(verbose)
        self._use_rtc = bool(use_rtc)
        self._merge_steps = int(merge_steps)
        self._replan_steps = int(replan_steps)

        # Flex runtime overrides — flipped via --bridge-kwargs at launch and
        # forwarded per-call as observation/present_* / observation/joint_*
        # wire keys. None = no override (model uses trained default).
        self._present_video: Optional[bool] = None
        self._present_dino: Optional[bool] = None
        self._present_pointmap: Optional[bool] = None
        self._joint_video: Optional[bool] = None
        self._joint_dino: Optional[bool] = None
        self._joint_pointmap: Optional[bool] = None

        # Broker is ActionChunkBroker by default; RTCStepBroker when use_rtc=True.
        self._broker: Optional[Any] = None
        self._per_cam_hw: Dict[str, tuple] = dict(_DEFAULT_PER_CAM_HW)
        self._server_metadata: Dict[str, Any] = {}

        # Diagnostics state.
        self._step = 0
        self._first_obs_dumped = False
        self._t_infer_sum = 0.0
        self._n_infer = 0

        # Observation recording (off by default). When ``record_observation_dir``
        # is set, ``predict()`` writes the head camera's frame to an MP4 at the
        # bridge's action_hz rate. Unlike the server's ``prediction.mp4``, this
        # is at full real-world rate (every ``predict()`` call, not every
        # broker WS round-trip). Writer is opened lazily on the first frame.
        self._record_observation_dir: Optional[str] = None
        self._record_observation_fps: int = 30
        self._obs_writer: Optional[Any] = None
        self._obs_writer_path: Optional[Path] = None

    # --- raiden ModelBridge interface --------------------------------------

    def load(self, ckpt_path: str, **kwargs) -> None:
        """Connect to the server. ``ckpt_path`` is ignored (server owns the model).

        Recognised ``--bridge-kwargs``:
            host, port, action_horizon, action_source, use_depth,
            depth_max_mm, prompt, verbose, api_key,
            use_rtc, merge_steps, replan_steps,
            present_video, present_dino, present_pointmap,
            joint_video, joint_dino, joint_pointmap
        """
        # Override ctor values from --bridge-kwargs string form.
        host = kwargs.get("host", "localhost")
        port = int(kwargs.get("port", 8000))
        api_key = kwargs.get("api_key") or None
        if "action_source" in kwargs:
            asrc = kwargs["action_source"]
            if asrc not in ("model_joint", "eef_ik"):
                raise ValueError(
                    f"action_source must be 'model_joint' or 'eef_ik'; got {asrc!r}"
                )
            self._action_source = asrc
        if "action_horizon" in kwargs:
            self._action_horizon = int(kwargs["action_horizon"])
        if "use_depth" in kwargs:
            self._use_depth = _parse_kwarg_bool(kwargs["use_depth"], self._use_depth)
        if "depth_max_mm" in kwargs:
            self._depth_max_mm = int(kwargs["depth_max_mm"])
        if "prompt" in kwargs:
            self._prompt = str(kwargs["prompt"] or "")
        if "verbose" in kwargs:
            self._verbose = _parse_kwarg_bool(kwargs["verbose"], self._verbose)
        if "use_rtc" in kwargs:
            self._use_rtc = _parse_kwarg_bool(kwargs["use_rtc"], self._use_rtc)
        if "merge_steps" in kwargs:
            self._merge_steps = int(kwargs["merge_steps"])
        if "replan_steps" in kwargs:
            self._replan_steps = int(kwargs["replan_steps"])
        # Flex runtime overrides: parsed once at launch; attached to every obs.
        for _name in (
            "present_video", "present_dino", "present_pointmap",
            "joint_video", "joint_dino", "joint_pointmap",
        ):
            if _name in kwargs:
                setattr(self, f"_{_name}", _parse_kwarg_bool(kwargs[_name], True))

        # Observation recording kwargs.
        if "record_observation_dir" in kwargs:
            self._record_observation_dir = str(kwargs["record_observation_dir"]) or None
        if "record_observation_fps" in kwargs:
            self._record_observation_fps = int(kwargs["record_observation_fps"])

        print(f"{LOG_PREFIX} Connecting to server at {host}:{port}")
        ws_policy = _ws.WebsocketClientPolicy(host=host, port=port, api_key=api_key)
        self._server_metadata = ws_policy.get_server_metadata()
        print(f"{LOG_PREFIX} server metadata: {self._server_metadata}")

        # Use the per_cam_hw the server advertises (falls back to default).
        meta_hw = self._server_metadata.get("per_cam_hw")
        if meta_hw:
            self._per_cam_hw = {cam: tuple(hw) for cam, hw in meta_hw.items()}

        # Clamp action_horizon to what the model can actually produce.
        srv_horizon = int(self._server_metadata.get("action_horizon", self._action_horizon))
        if self._action_horizon > srv_horizon:
            print(
                f"{LOG_PREFIX} clamping action_horizon {self._action_horizon} -> "
                f"server's {srv_horizon} (model can't emit more)."
            )
            self._action_horizon = srv_horizon


        if self._use_rtc:
            from experiments.yam.flexpi_policy.client_rtc_step_broker import RTCStepBroker
            action_dim = int(self._server_metadata.get("action_dim", 32))
            self._broker = RTCStepBroker(
                ws_policy=ws_policy,
                action_horizon=self._action_horizon,
                action_dim=action_dim,
                merge_steps=self._merge_steps,
                replan_steps=self._replan_steps,
            )
            broker_tag = (
                f"RTCStepBroker(merge_steps={self._merge_steps} "
                f"replan_steps={self._replan_steps})"
            )
        else:
            self._broker = action_chunk_broker.ActionChunkBroker(
                policy=ws_policy, action_horizon=self._action_horizon,
            )
            broker_tag = "ActionChunkBroker"
        print(
            f"{LOG_PREFIX} Ready (action_source={self._action_source} "
            f"action_horizon={self._action_horizon} use_depth={self._use_depth} "
            f"prompt={self._prompt!r}) broker={broker_tag}"
        )

    def reset(self) -> None:
        if self._broker is not None:
            self._broker.reset()
        self._step = 0
        self._t_infer_sum = 0.0
        self._n_infer = 0

    def predict(self, obs) -> np.ndarray:
        """Per-step: preprocess → broker.infer → action permutation."""
        if self._broker is None:
            raise RuntimeError(
                "predict() called before load(). raiden's RaidenInferenceLoop "
                "should call load() during construction."
            )

        # Snapshot the head camera at the bridge's full action rate (30 hz
        # with default --action_hz). Independent of the broker's chunk
        # cadence — every predict() writes one frame, so the file is a
        # real-time observation log, not a chunk-rate sampling.
        if self._record_observation_dir:
            self._record_observation_frame(obs)

        t0 = time.perf_counter()
        obs_dict = self._preprocess(obs)

        # First-step debug dump (mirrors maniflow_bridge._dump_first_obs).
        if self._step == 0 and not self._first_obs_dumped:
            self._dump_first_obs(obs_dict)
            self._first_obs_dumped = True

        # broker.infer returns one row (32D) per call. ActionChunkBroker
        # blocks ~190 ms when its cache is exhausted; RTCStepBroker runs
        # inference off-thread and (after cold start) never blocks here.
        is_chunk_broker = isinstance(self._broker, action_chunk_broker.ActionChunkBroker)
        if is_chunk_broker:
            new_chunk = self._broker._last_results is None  # noqa: SLF001
        else:
            new_chunk = False
        result = self._broker.infer(obs_dict)
        infer_ms = (time.perf_counter() - t0) * 1e3
        if new_chunk:
            self._t_infer_sum += infer_ms
            self._n_infer += 1

        action_32 = np.asarray(result["actions"], dtype=np.float32)
        if action_32.shape != (32,):
            raise ValueError(
                f"server returned action shape {action_32.shape}; expected (32,). "
                "Did the server send {'actions': chunk} with chunk[T, 32]?"
            )

        if self._action_source == "model_joint":
            cmd = _eef32_action_to_joint14_RL(action_32)
        else:
            cmd = _eef32_action_to_ee_pose20(action_32)

        if is_chunk_broker:
            if self._step % 50 == 1:
                avg_ms = self._t_infer_sum / max(self._n_infer, 1)
                tag = "fresh" if new_chunk else "queue"
                print(
                    f"{LOG_PREFIX} step={self._step:4d} src={tag:5s} "
                    f"infer_avg={avg_ms:.1f} ms "
                    f"cmd_range=[{float(cmd.min()):+.3f}, {float(cmd.max()):+.3f}]"
                )
        else:
            if self._step % max(self._action_horizon, 1) == 1:
                s = self._broker.stats
                print(
                    f"{LOG_PREFIX} step={self._step:4d} rtc "
                    f"submits={s['submits']:4d} merges={s['merges']:4d} "
                    f"jit_drained={s['merges_just_in_time']:3d} "
                    f"drain_recover={s['drain_recoveries']:3d} "
                    f"cold={s['cold_starts']:2d} stale={s['stale_drops']:2d} "
                    f"qlen={s['queue_len']:3d} "
                    f"last_offset={s['last_offset']:+3d} "
                    f"last_blend={s['last_blend_n']:2d} "
                    f"last_L={s['last_L']:3d} "
                    f"cmd_range=[{float(cmd.min()):+.3f}, {float(cmd.max()):+.3f}]"
                )
        self._step += 1
        return cmd

    # --- helpers -----------------------------------------------------------

    def set_prompt(self, prompt: str) -> None:
        """Update the task instruction between episodes."""
        self._prompt = str(prompt)

    # --- Observation video recording (bridge-side, action-rate) -------------

    def _record_observation_frame(self, obs: Any) -> None:
        """Append the head-camera frame to ``observation_<timestamp>.mp4``.

        Lazy-opens the writer on first call. Silent no-op if the head cam
        can't be found in this obs (the writer simply never opens). Errors
        are logged and the writer is dropped — never raises, so a recording
        failure can't take down the control loop.
        """
        head_image = None
        for cam in (getattr(obs, "cameras", []) or []):
            raiden_name = getattr(cam, "name", None)
            if raiden_name and _RAIDEN_TO_FLEXPI_CAM.get(raiden_name) == "cam_high":
                head_image = getattr(cam, "image", None)
                break
        if head_image is None:
            return

        img = np.asarray(head_image)
        if img.ndim != 3 or img.shape[-1] != 3:
            return
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        # libx264 requires even H/W — edge-pad if needed (same trick as the
        # server-side recorder).
        h, w = img.shape[:2]
        if h % 2 or w % 2:
            img = np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")

        if self._obs_writer is None:
            try:
                import atexit
                import imageio
                from datetime import datetime
                d = Path(self._record_observation_dir)
                d.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                self._obs_writer_path = d / f"observation_{stamp}.mp4"
                self._obs_writer = imageio.get_writer(
                    str(self._obs_writer_path),
                    fps=max(int(self._record_observation_fps), 1),
                    codec="libx264",
                    format="FFMPEG",
                    pixelformat="yuv420p",
                )
                atexit.register(self._close_observation_writer)
                print(
                    f"{LOG_PREFIX} recording observation video to "
                    f"{self._obs_writer_path} @ {self._record_observation_fps} fps"
                )
            except Exception as e:  # noqa: BLE001
                print(f"{LOG_PREFIX} obs writer open failed: {e!r}; recording disabled")
                self._obs_writer = None
                self._record_observation_dir = None
                return

        try:
            self._obs_writer.append_data(img)
        except Exception as e:  # noqa: BLE001
            print(f"{LOG_PREFIX} obs writer append failed: {e!r}; closing")
            self._close_observation_writer()

    def _close_observation_writer(self) -> None:
        if self._obs_writer is None:
            return
        try:
            self._obs_writer.close()
            print(f"{LOG_PREFIX} closed observation video: {self._obs_writer_path}")
        except Exception as e:  # noqa: BLE001
            print(f"{LOG_PREFIX} obs writer close failed: {e!r}")
        finally:
            self._obs_writer = None

    def _preprocess(self, obs: Any) -> Dict[str, Any]:
        """Build the OpenPI-flat obs dict from a raiden ``Observation``.

        Per cam:
          observation/image_<flexpi_cam>      uint8  [h, w, 3]   (resized RGB)
          observation/depth_<flexpi_cam>      uint16 [h, w]      (mm, resized) if use_depth
          observation/intrinsics_<flexpi_cam> float32 [4]        [fx,fy,cx,cy]  if use_depth
        Plus:
          observation/state   float32 [32]   yam_eef.STATE_LAYOUT (from raiden FK)
          prompt              str            optional task instruction
        """
        obs_dict: Dict[str, Any] = {}
        cameras = list(getattr(obs, "cameras", []) or [])
        proprios = dict(getattr(obs, "proprios", {}) or {})

        seen_cam_names: list = []
        for cam in cameras:
            raiden_name = getattr(cam, "name", None)
            seen_cam_names.append(raiden_name)
            flexpi_cam = _RAIDEN_TO_FLEXPI_CAM.get(raiden_name) if raiden_name else None
            if flexpi_cam is None:
                continue
            target_hw = self._per_cam_hw.get(flexpi_cam)
            if target_hw is None:
                continue
            h, w = int(target_hw[0]), int(target_hw[1])

            # RGB: raiden delivers uint8 (H,W,3) RGB; stretch-resize to (h,w).
            image = getattr(cam, "image", None)
            if image is None:
                raise ValueError(f"camera {raiden_name!r}: missing .image")
            rgb = image_tools.resize_stretch(np.asarray(image), h, w)
            obs_dict[f"observation/image_{flexpi_cam}"] = np.ascontiguousarray(rgb)

            if self._use_depth:
                depth = getattr(cam, "depth", None)
                K = getattr(cam, "intrinsics", None)
                if depth is None or K is None:
                    if self._step == 0:
                        print(
                            f"{LOG_PREFIX} WARN: use_depth=True but cam {raiden_name!r} "
                            f"has depth={depth is not None} K={K is not None}; "
                            "depth slot omitted."
                        )
                    continue
                d_arr = np.asarray(depth)
                if d_arr.dtype in (np.float32, np.float64):
                    depth_mm_src = (
                        d_arr.astype(np.float32) * 1000.0
                    ).clip(0, self._depth_max_mm).astype(np.uint16)
                elif d_arr.dtype == np.uint16:
                    depth_mm_src = d_arr
                else:
                    raise TypeError(
                        f"cam {raiden_name!r}: depth dtype {d_arr.dtype} unsupported. "
                        "Expected float32 metres (raiden) or uint16 mm."
                    )
                K_arr = np.asarray(K, dtype=np.float32)
                if K_arr.shape != (3, 3):
                    raise ValueError(
                        f"cam {raiden_name!r}: intrinsics shape {K_arr.shape} expected (3, 3)"
                    )
                fx, fy = float(K_arr[0, 0]), float(K_arr[1, 1])
                cx, cy = float(K_arr[0, 2]), float(K_arr[1, 2])
                depth_resized, fx, fy, cx, cy = _resize_depth_and_scale_intrinsics(
                    depth_mm_src, fx, fy, cx, cy, h, w,
                )
                obs_dict[f"observation/depth_{flexpi_cam}"] = np.ascontiguousarray(depth_resized)
                obs_dict[f"observation/intrinsics_{flexpi_cam}"] = np.array(
                    [fx, fy, cx, cy], dtype=np.float32,
                )

        # Completeness check: need all 3 cams.
        required = set(self._per_cam_hw)
        missing = required - {k.replace("observation/image_", "")
                              for k in obs_dict if k.startswith("observation/image_")}
        if missing:
            raise ValueError(
                f"{LOG_PREFIX} need all of {sorted(required)}; missing {sorted(missing)}. "
                f"raiden cameras seen: {seen_cam_names}. "
                f"Mapping in use: {_RAIDEN_TO_FLEXPI_CAM}"
            )

        # State: FK via raiden helpers (cached on first call inside _raiden_to_eef32_state).
        if "state_32" in proprios:
            state_32 = np.asarray(proprios["state_32"], dtype=np.float32)
        else:
            if "follower_r_joint_pos" not in proprios or "follower_l_joint_pos" not in proprios:
                raise ValueError(
                    "proprios must contain 'state_32' or both 'follower_r_joint_pos' "
                    f"and 'follower_l_joint_pos'. Got: {list(proprios)}"
                )
            r_pos = np.asarray(proprios["follower_r_joint_pos"], dtype=np.float32)
            l_pos = np.asarray(proprios["follower_l_joint_pos"], dtype=np.float32)
            state_32 = _raiden_to_eef32_state(r_pos, l_pos)
            # Rebase the relative-action anchor on the last COMMANDED (open-loop)
            # pose, not the measured (lagging) one — else the follower's gravity
            # droop is re-anchored on every chunk boundary and ratchets into a
            # periodic "sag". observation/state is BOTH the model proprio AND the
            # anchor the WS server inverts relative actions against
            # (server_adapter -> deploy_policy._denormalize_action, act_step +
            # RTC), so overriding it here fixes both. Override the EE base
            # [0:18] (eef_ik) and the joint base [20:32] (model_joint); grippers
            # [18:20] are absolute pass-through and stay measured. Commanded
            # joints are injected by raiden's inference loop as
            # follower_*_joint_cmd; no-op if absent or FLEXPI_EEF_BASE=measured.
            if os.environ.get("FLEXPI_EEF_BASE", "commanded") == "commanded":
                r_cmd = proprios.get("follower_r_joint_cmd")
                l_cmd = proprios.get("follower_l_joint_cmd")
                if r_cmd is not None and l_cmd is not None:
                    cmd_state = _raiden_to_eef32_state(
                        np.asarray(r_cmd, dtype=np.float32),
                        np.asarray(l_cmd, dtype=np.float32),
                    )
                    state_32[0:18] = cmd_state[0:18]
                    state_32[20:32] = cmd_state[20:32]
                    if self._step == 0:
                        print(
                            f"{LOG_PREFIX} eef base: EE [0:18] + joint [20:32] "
                            "anchored on COMMANDED joints (grippers measured); "
                            "set FLEXPI_EEF_BASE=measured to disable."
                        )
        if state_32.shape != (32,):
            raise ValueError(f"state_32 shape {state_32.shape} != (32,)")
        obs_dict["observation/state"] = state_32.astype(np.float32)

        if self._prompt:
            obs_dict["prompt"] = self._prompt

        # Flex runtime overrides — wire-encoded as np.bool_ scalars. Absent
        # keys mean "no override"; the server adapter / model fall back to
        # trained defaults.
        for _name in (
            "present_video", "present_dino", "present_pointmap",
            "joint_video", "joint_dino", "joint_pointmap",
        ):
            _val = getattr(self, f"_{_name}")
            if _val is not None:
                obs_dict[f"observation/{_name}"] = np.bool_(_val)

        return obs_dict

    def _dump_first_obs(self, obs_dict: Dict[str, Any]) -> None:
        """Save the first wire-shaped obs to ``~/yam_flexpi_ws_obs_debug/``.

        Mirrors maniflow_bridge._dump_first_obs (yam_openpi/deployment/maniflow_bridge.py:671).
        """
        out_dir = Path(os.path.expanduser("~/yam_flexpi_ws_obs_debug"))
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            import cv2
            have_cv2 = True
        except ImportError:
            have_cv2 = False

        meta: dict = {
            "step": 0,
            "action_source": self._action_source,
            "action_horizon": self._action_horizon,
            "use_depth": self._use_depth,
            "prompt": self._prompt,
            "per_cam_hw": {k: list(v) for k, v in self._per_cam_hw.items()},
            "server_metadata": self._server_metadata,
            "raiden_to_flexpi_cam": _RAIDEN_TO_FLEXPI_CAM,
        }
        for key, val in obs_dict.items():
            if key.startswith("observation/image_") and have_cv2:
                cam_name = key.replace("observation/image_", "")
                cv2.imwrite(
                    str(out_dir / f"rgb_{cam_name}.png"),
                    cv2.cvtColor(np.asarray(val), cv2.COLOR_RGB2BGR),
                )
            elif key.startswith("observation/depth_") and have_cv2:
                cam_name = key.replace("observation/depth_", "")
                cv2.imwrite(str(out_dir / f"depth_{cam_name}_mm.png"), np.asarray(val))
            elif key.startswith("observation/intrinsics_"):
                meta[key] = np.asarray(val).tolist()
            elif key == "observation/state":
                meta[key] = np.asarray(val).tolist()
            elif key == "prompt":
                meta[key] = val
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"{LOG_PREFIX} dumped first observation to {out_dir}/")


# ---------------------------------------------------------------------------
# Standalone CLI entry (direct counterpart of yam_raiden_bridge.main()).
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Deploy YAM FlexPi via a websocket policy server + thin raiden bridge. "
            "Mirrors yam_openpi/deployment/maniflow_bridge.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt_path", default="unused",
                   help="Unused (server owns the model). Required by raiden's CLI.")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--action_hz", type=float, default=30.0)
    p.add_argument("--action_source", default="model_joint",
                   choices=["model_joint", "eef_ik"])
    p.add_argument("--action_type", default="joint", choices=["joint", "ee_pose"])
    p.add_argument("--action_horizon", type=int, default=32,
                   help="Number of broker-cached actions per server inference.")
    p.add_argument("--use_depth", action="store_true", default=True,
                   help="Send depth+intrinsics to the server (required for the 0509 ckpt).")
    p.add_argument("--no_depth", dest="use_depth", action="store_false")
    p.add_argument("--prompt", default="", help="Task instruction.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--api_key", default=None)
    # raiden hardware args.
    p.add_argument("--camera_config_file", default="./config/camera_config.json")
    p.add_argument("--calibration_file", default="./config/calibration_results.json")
    p.add_argument("--stereo_method", default="zed", choices=["zed", "ffs"])
    p.add_argument("--depth_mode", default="NEURAL_LIGHT")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    # Action-source / action-type compatibility (mirrors yam_raiden_bridge.py:481-488).
    if args.action_source == "model_joint" and args.action_type != "joint":
        sys.exit("ERROR: --action_source model_joint requires --action_type joint.")
    if args.action_source == "eef_ik" and args.action_type != "ee_pose":
        sys.exit("ERROR: --action_source eef_ik requires --action_type ee_pose.")

    if not _RAIDEN_AVAILABLE:
        sys.exit(
            "ERROR: raiden is not importable in this Python env. Run from "
            "the raiden venv (e.g. /home/<user>/projects/YAM_robot/.venv)."
        )

    from raiden.inference import RaidenInferenceLoop  # noqa: WPS433

    bridge = YamFlexPiRaidenBridgeWS(
        action_source=args.action_source,
        action_horizon=args.action_horizon,
        use_depth=args.use_depth,
        prompt=args.prompt,
        verbose=args.verbose,
    )

    bridge_kwargs: Dict[str, Any] = {
        "host": args.host,
        "port": args.port,
        "action_horizon": args.action_horizon,
        "action_source": args.action_source,
        "use_depth": args.use_depth,
        "prompt": args.prompt,
        "verbose": args.verbose,
    }
    if args.api_key is not None:
        bridge_kwargs["api_key"] = args.api_key

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
