"""YAM real-world deployment bridge for FlexPi.

Adapts a runtime/raiden-style observation into the format ``YamFlexPiPolicy``
consumes, runs inference, maintains a replan queue, and emits one robot
command per ``act()`` call. Mirrors the role of
``yam_openpi/deployment/openpi_bridge.py``: a thin format adapter — the
policy class does all heavy lifting (per-cam resize, composite build, K
delivery, normalize/denormalize, Yam32DRelativeAction inversion, model
inference).

Two action sources, both selected from the 32D absolute action returned by
``YamFlexPiPolicy.infer_action_chunk`` (``yam_eef.STATE_LAYOUT``):

- ``"model_joint"`` (default, recommended starting path):
    Returns a ``(14,) float32`` joint command in raiden's ``[R, L]`` motor order
    taken directly from ``action_32[20:32]`` + grips. Bypasses IK entirely.
    Pair with ``rd infer --action-type joint``.

- ``"eef_ik"``:
    Returns a ``(20,) float32`` EE pose ``[l_xyz, r_xyz, l_rot6d, r_rot6d,
    l_grip, r_grip]`` with the L↔R swap documented in
    ``openpi_bridge._eef32_action_to_ee_pose20``. Pair with
    ``rd infer --action-type ee_pose``. raiden then runs IK on the bridge's
    20D output.

Camera-name mapping is the **only** translation the bridge owns: raiden's
``head / left_wrist / right_wrist`` → FlexPi canonical ``cam_high /
cam_left_wrist / cam_right_wrist``. The target keys are NOT hardcoded
model-side constants — they are imported as ``_CAM_ORDER`` from
``deploy_policy.py``.

Observation contract (raiden-style):
    ``obs.cameras`` : iterable of camera objects, each with
        ``.name``        - one of ``head / left_wrist / right_wrist``.
        ``.image``       - ``np.uint8 [H, W, 3]`` RGB.
        ``.depth``       - ``np.float32 [H, W]`` metres (raiden contract)
                            OR ``np.uint16 [H, W]`` mm (already converted).
                            None → camera contributes RGB only.
        ``.intrinsics``  - ``(3, 3)`` float K at depth's pixel grid, or None.
    ``obs.proprios`` : dict; bridge consumes one of two layouts:
        (a) ``proprios["state_32"]`` - precomputed ``(32,) float32`` per
            ``yam_eef.STATE_LAYOUT`` (offline tests, FK-computed upstream).
        (b) ``proprios["follower_r_joint_pos"]`` and
            ``proprios["follower_l_joint_pos"]`` - ``(7,) float32``. Bridge
            lazy-imports raiden's MuJoCo FK to fill EEF blocks. This path
            requires ``raiden`` installed.

Constraint reiterated for reviewers: nothing in this file hardcodes paths,
prompts, intrinsics resolution, action horizon, or normalizer paths. Those
all flow through CLI/constructor → ``build_policy_from_checkpoint`` →
``YamFlexPiPolicy``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Literal, Optional

import numpy as np

# Path bootstrap so this file is importable without an installed wheel —
# mirrors deploy_policy.py.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.yam.flexpi_policy.deploy_policy import (  # noqa: E402
    YamFlexPiPolicy,
    build_policy_from_checkpoint,
    _CAM_ORDER,
    _D,
)

logger = logging.getLogger(__name__)


# Raiden camera name → FlexPi canonical name. Source for the *raiden* side
# of this mapping is yam_openpi/deployment/openpi_bridge.py's ``_CAM_MAP``
# keys; the FlexPi-side targets are imported from deploy_policy.py to keep
# them config-derived rather than hardcoded model constants.
_RAIDEN_TO_FLEXPI_CAM: Dict[str, str] = {
    "head":        "cam_high",
    "left_wrist":  "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}


# ---------------------------------------------------------------------------
# Action-space mappings.
#
# DUPLICATED VERBATIM from yam_openpi/deployment/openpi_bridge.py — including
# the L↔R swap rationale in the eef_ik path's docstring. Do NOT modify these
# without first updating the source in openpi_bridge.py; the raiden side of
# the contract has been validated against this exact permutation.
# ---------------------------------------------------------------------------


def _eef32_action_to_joint14_RL(action_32: np.ndarray) -> np.ndarray:
    """Extract the model's 14D joint+grip output from a 32D EEF action.

    Source (yam_eef.STATE_LAYOUT):
      [18] l_grip  [19] r_grip  [20:26] l_joint  [26:32] r_joint

    Target (raiden 14D in [R, L] order — matches what RaidenInferenceLoop
    slices: action_14d[:7]→follower_r, action_14d[7:14]→follower_l):
      [0:6] r_arm  [6] r_grip  [7:13] l_arm  [13] l_grip

    No L↔R swap shenanigans here (unlike the ee_pose path): this is a direct
    permutation into raiden's documented motor-command convention.

    Note for FlexPi users: there is NO server-side AbsoluteActions transform
    in the FlexPi path. Yam32DRelativeAction was applied at *training* time
    on the action targets only (state stayed absolute end-to-end), so the
    policy's emitted ``action_32`` is already absolute after the
    Yam32DRelativeAction.backward step inside ``_denormalize_action``. The
    joint slice ``action_32[20:32]`` is therefore absolute joint targets in
    radians, identical in semantics to the openpi_bridge case.
    """
    if action_32.shape != (32,):
        raise ValueError(
            f"action_32 must be shape (32,); got {action_32.shape}. "
            "Did the model output a different action_dim?"
        )
    out = np.empty(14, dtype=np.float32)
    out[ 0: 6] = action_32[26:32]  # r_arm
    out[ 6   ] = action_32[19]     # r_grip
    out[ 7:13] = action_32[20:26]  # l_arm
    out[13   ] = action_32[18]     # l_grip
    return out


def _eef32_action_to_ee_pose20(action_32: np.ndarray) -> np.ndarray:
    """Reshuffle 32D EEF action into raiden's 20D ee_pose layout, with L↔R swap.

    Server has already applied output transforms (RelativeEEFActions backward
    + AbsoluteActions for the joint slice), so action_32 is in absolute world
    frame. We keep only the EEF blocks and grippers; the trailing joint dims
    [20:32] are ignored — raiden's _ee_pose_to_joint_cmd solves IK from the
    EE pose seeded on _last_joint_cmd.

    Source (yam_eef.STATE_LAYOUT):
      [ 0: 3] l_pos   [ 3: 9] l_rot6d   [ 9:12] r_pos   [12:18] r_rot6d
      [18] l_grip     [19] r_grip       [20:32] joints (dropped)

    Target (raiden _ee_pose_to_joint_cmd, _documented_ layout):
      [0:3] l_xyz   [3:6] r_xyz   [6:12] l_rot6d   [12:18] r_rot6d
      [18] l_grip   [19] r_grip

    L↔R swap rationale (raiden internal convention bug):
      raiden's `_ee_pose_to_joint_cmd` returns 14D as np.concatenate([cmd_l, cmd_r])
      i.e. [L, R] order, but `RaidenInferenceLoop.run()` and `_check_joint_delta`
      slice 14D as [R, L] (`action_14d[:7]` → follower_r, `[7:14]` → follower_l).
      Walking the data straight through swaps physical arms at the motor and
      trips _check_joint_delta on the first step (left/right home shoulders
      typically differ by >0.5 rad → > _max_joint_delta → estop).

      Easiest fix without touching raiden: feed the right arm's target into
      raiden's "left" slot and vice versa. Both `_kin_l`/`_kin_r` are
      instances of the SAME single-arm MJCF, so kin_l.ik(T) == kin_r.ik(T)
      — solving "the wrong arm's" Kinematics for a given target is fine.

      Cosmetic side effect: raiden's `[IK-L] ... [IK-R] ...` log labels
      will be swapped relative to the physical arms.

    FlexPi-side note (unlike the openpi server path, the policy here did NOT
    apply an AbsoluteActions transform server-side because the YAM 32D-rel
    training only converted the EEF blocks to body-frame relative, joints to
    scalar delta, and grippers passthrough. The Yam32DRelativeAction.backward
    inversion inside ``YamFlexPiPolicy._denormalize_action`` produces
    absolute EEF poses, so action_32[0:18] is in world-frame absolute units
    — same input shape this permutation expects).
    """
    if action_32.shape != (32,):
        raise ValueError(
            f"action_32 must be shape (32,); got {action_32.shape}."
        )
    out = np.empty(20, dtype=np.float32)
    # raiden "left" slot ← physical RIGHT
    out[ 0: 3] = action_32[ 9:12]   # raiden l_xyz   ← phys r_pos
    out[ 6:12] = action_32[12:18]   # raiden l_rot6d ← phys r_rot6d
    out[18]    = action_32[19]      # raiden l_grip  ← phys r_grip
    # raiden "right" slot ← physical LEFT
    out[ 3: 6] = action_32[ 0: 3]   # raiden r_xyz   ← phys l_pos
    out[12:18] = action_32[ 3: 9]   # raiden r_rot6d ← phys l_rot6d
    out[19]    = action_32[18]      # raiden r_grip  ← phys l_grip
    return out


def _raiden_to_eef32_state(
    r_joint_pos: np.ndarray,
    l_joint_pos: np.ndarray,
) -> np.ndarray:
    """Assemble raiden proprio into 32D state per yam_eef.STATE_LAYOUT.

    Uses raiden's MuJoCo FK helpers (same XML the dataset was built against)
    so the rot6d row convention matches training exactly. Lazy-imports
    ``raiden.policy_server`` — this code path only runs at real-robot time;
    offline tests pass ``proprios['state_32']`` directly and skip FK.

    Output layout (must match yam_eef.STATE_LAYOUT — duplicated here so the
    bridge does not need to import server-side openpi modules):

      [ 0: 3]  left_pos       (FK xyz)
      [ 3: 9]  left_rot6d     (FK R[:2, :3].flatten())
      [ 9:12]  right_pos
      [12:18]  right_rot6d
      [18]     left_grip
      [19]     right_grip
      [20:26]  left_joint  (6 arm joints)
      [26:32]  right_joint
    """
    try:
        from raiden.policy_server import (
            _fk_to_xyz_rot6d, _get_kin_l, _get_kin_r,
            _kin_l_lock, _kin_r_lock,
        )
    except ImportError as exc:
        raise ImportError(
            "Computing 32D state from raw joint positions requires the `raiden` "
            "package (for MuJoCo FK on the same XML training was built against). "
            "Either install raiden or pass observation.proprios['state_32'] directly."
        ) from exc

    with _kin_l_lock:
        l_xyz, l_rot6d = _fk_to_xyz_rot6d(_get_kin_l(), l_joint_pos[:6])
    with _kin_r_lock:
        r_xyz, r_rot6d = _fk_to_xyz_rot6d(_get_kin_r(), r_joint_pos[:6])

    state32 = np.empty(32, dtype=np.float32)
    state32[ 0: 3] = l_xyz
    state32[ 3: 9] = l_rot6d
    state32[ 9:12] = r_xyz
    state32[12:18] = r_rot6d
    state32[18]    = l_joint_pos[6]
    state32[19]    = r_joint_pos[6]
    state32[20:26] = l_joint_pos[:6]
    state32[26:32] = r_joint_pos[:6]
    return state32


# ---------------------------------------------------------------------------
# Bridge class
# ---------------------------------------------------------------------------


class YamFlexPiBridge:
    """Adapter: raiden-style obs → YamFlexPiPolicy → robot command."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        # All forwarded to build_policy_from_checkpoint. None ⇒ that side
        # auto-resolves (e.g., stats path walked from ckpt; intrinsics
        # supplied per-step in obs.intrinsics).
        dataset_stats_path: Optional[str] = None,
        intrinsics_json: Optional[str] = None,
        # Defaults come from configs/real_yam.yaml, the same dict the factory
        # uses -- so this layer forwarding them explicitly is a no-op rather
        # than a second, drifting copy of the table.
        device: str = _D["device"],
        mixed_precision: str = _D["mixed_precision"],
        action_horizon: Optional[int] = _D["action_horizon"],
        num_inference_steps: int = _D["num_inference_steps"],
        sigma_shift: Optional[float] = _D["sigma_shift"],
        seed: Optional[int] = _D["seed"],
        text_cfg_scale: float = _D["text_cfg_scale"],
        negative_prompt: str = _D["negative_prompt"],
        # Perf knobs (forwarded into YamFlexPiPolicy via the factory).
        offload_text_encoder: bool = _D["offload_text_encoder"],
        torch_compile: bool = _D["torch_compile"],
        torch_compile_mode: str = _D["torch_compile_mode"],
        quantization: Optional[str] = _D["quantization"],
        # Bridge-specific knob: consumed here, not forwarded.
        replan_steps: int = _D["replan_steps"],
        action_source: Literal["model_joint", "eef_ik"] = "model_joint",
        prompt: str = "",
        verbose: bool = False,
        # Pre-built policy injection (used by offline tests; lets callers reuse
        # a single load across many bridges or supply a mock).
        policy: Optional[YamFlexPiPolicy] = None,
    ) -> None:
        if action_source not in ("model_joint", "eef_ik"):
            raise ValueError(
                f"action_source must be 'model_joint' or 'eef_ik'; got {action_source!r}"
            )
        if policy is not None:
            self._policy: YamFlexPiPolicy = policy
        else:
            self._policy = build_policy_from_checkpoint(
                checkpoint_path=checkpoint_path,
                dataset_stats_path=dataset_stats_path,
                intrinsics_json=intrinsics_json,
                device=device,
                mixed_precision=mixed_precision,
                action_horizon=action_horizon,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                text_cfg_scale=text_cfg_scale,
                negative_prompt=negative_prompt,
                offload_text_encoder=offload_text_encoder,
                torch_compile=torch_compile,
                torch_compile_mode=torch_compile_mode,
                quantization=quantization,
            )

        # Clamp replan_steps to [1, policy.action_horizon]. The policy returns
        # a chunk of action_horizon commands per inference; the bridge queues
        # up to replan_steps of them and re-infers when the queue drains.
        self._replan_steps = int(max(1, min(int(replan_steps), self._policy.action_horizon)))
        self._action_source: Literal["model_joint", "eef_ik"] = action_source
        self._prompt = str(prompt)
        self._verbose = bool(verbose)

        # Action queue + diagnostic state.
        self._pending: Deque[np.ndarray] = deque()
        self._step = 0
        self._inference_count = 0
        self._infer_s_total = 0.0
        self._first_obs_logged = False

        print(
            f"[yam-bridge] ready  action_source={self._action_source}  "
            f"replan_steps={self._replan_steps}  "
            f"action_horizon={self._policy.action_horizon}  "
            f"prompt={self._prompt!r}"
        )
        if self._verbose:
            self._log_action_mapping_once()

    # --- API ----------------------------------------------------------------

    @property
    def policy(self) -> YamFlexPiPolicy:
        return self._policy

    @property
    def replan_steps(self) -> int:
        return self._replan_steps

    @property
    def action_source(self) -> str:
        return self._action_source

    def needs_fresh_observation(self) -> bool:
        """True iff the next ``act()`` call will trigger inference.

        Lets callers skip building a new observation when the queue is still
        serving cached commands — important when observation construction is
        expensive (e.g., file decode in the offline test, ZED capture on the
        real robot at full duty cycle).
        """
        return not self._pending

    def reset(self) -> None:
        """Clear the pending queue and step counters. Does not unload the model."""
        self._pending.clear()
        self._step = 0
        self._inference_count = 0
        self._infer_s_total = 0.0

    def set_prompt(self, prompt: str) -> None:
        self._prompt = str(prompt)

    def act(self, observation: Any) -> Dict[str, Any]:
        """Return one robot command. Runs inference iff the queue is empty.

        Returns a dict (NOT bare ndarray) carrying the command plus
        diagnostics:

            command          - np.float32 robot command (14D model_joint, 20D eef_ik).
            from_queue       - True iff this command came from the pending buffer
                               (no inference this call).
            queue_len        - remaining commands in the buffer AFTER this pop.
            step             - bridge step counter (monotonic across resets if not
                               .reset()-ed; resets only zero this).
            inference_count  - total inferences run since the last reset().
            infer_s          - wall-time of the inference that produced this
                               chunk, 0.0 if from_queue.
            action_source    - echoes the configured source for log parity.

        Callers in raiden integrations should access ``result["command"]``
        and feed it to the appropriate raiden action sink based on
        ``action_source``.
        """
        from_queue = bool(self._pending)
        infer_s = 0.0

        if not from_queue:
            obs_dict = self._preprocess(observation)
            t0 = time.perf_counter()
            action_chunk_abs = self._policy.infer_action_chunk(
                obs_dict, self._prompt, verbose=self._verbose,
            )
            infer_s = time.perf_counter() - t0
            self._infer_s_total += infer_s
            self._inference_count += 1
            self._enqueue_chunk(action_chunk_abs, infer_s=infer_s)

        cmd = self._pending.popleft()
        result = {
            "command": cmd,
            "from_queue": from_queue,
            "queue_len": len(self._pending),
            "step": self._step,
            "inference_count": self._inference_count,
            "infer_s": infer_s,
            "action_source": self._action_source,
        }
        if self._verbose:
            self._log_step(result)
        self._step += 1
        return result

    # --- Internals ----------------------------------------------------------

    def _enqueue_chunk(self, action_chunk_abs: np.ndarray, *, infer_s: float) -> None:
        """Convert ``action_chunk_abs`` (``[T, 32]``) per-step into robot commands and push."""
        if action_chunk_abs.ndim != 2 or action_chunk_abs.shape[-1] != 32:
            raise ValueError(
                f"Expected action chunk [T, 32]; got {action_chunk_abs.shape}"
            )
        if not np.isfinite(action_chunk_abs).all():
            n_nan = int(np.isnan(action_chunk_abs).sum())
            n_inf = int(np.isinf(action_chunk_abs).sum())
            print(
                f"[yam-bridge] WARN: action chunk contains non-finite values "
                f"(NaN={n_nan} Inf={n_inf}); commands will inherit them. "
                f"Likely cause: rot6d at the anchor state is degenerate."
            )
        n_exec = min(self._replan_steps, action_chunk_abs.shape[0])
        for k in range(n_exec):
            cmd = self._abs32_to_command(action_chunk_abs[k])
            self._pending.append(cmd)
        if self._verbose:
            cmd_first = self._pending[0]
            print(
                f"[yam-bridge] step={self._step:4d}  NEW CHUNK  "
                f"infer={infer_s*1000:.1f} ms  enqueued={n_exec}  "
                f"action_32[0]={np.array2string(action_chunk_abs[0], precision=3, suppress_small=True)}  "
                f"cmd[0]={np.array2string(cmd_first, precision=3, suppress_small=True)}"
            )

    def _abs32_to_command(self, action_32: np.ndarray) -> np.ndarray:
        if self._action_source == "model_joint":
            return _eef32_action_to_joint14_RL(action_32)
        return _eef32_action_to_ee_pose20(action_32)

    def _log_action_mapping_once(self) -> None:
        """Print exact slot-by-slot mapping in verbose mode (sanity for ops)."""
        if self._action_source == "model_joint":
            print("[yam-bridge] action mapping (model_joint, 14D raiden [R, L]):")
            print("  cmd[ 0: 6] = action_32[26:32]   # r_arm")
            print("  cmd[    6] = action_32[19]      # r_grip")
            print("  cmd[ 7:13] = action_32[20:26]   # l_arm")
            print("  cmd[   13] = action_32[18]      # l_grip")
        else:
            print("[yam-bridge] action mapping (eef_ik, 20D with L↔R swap):")
            print("  cmd[ 0: 3] = action_32[ 9:12]   # raiden l_xyz   ← phys r_pos")
            print("  cmd[ 3: 6] = action_32[ 0: 3]   # raiden r_xyz   ← phys l_pos")
            print("  cmd[ 6:12] = action_32[12:18]   # raiden l_rot6d ← phys r_rot6d")
            print("  cmd[12:18] = action_32[ 3: 9]   # raiden r_rot6d ← phys l_rot6d")
            print("  cmd[18]    = action_32[19]      # raiden l_grip  ← phys r_grip")
            print("  cmd[19]    = action_32[18]      # raiden r_grip  ← phys l_grip")

    def _log_step(self, result: Dict[str, Any]) -> None:
        cmd = result["command"]
        tag = "queue" if result["from_queue"] else "fresh"
        print(
            f"[yam-bridge] step={result['step']:4d}  src={tag:5s}  "
            f"queue_len={result['queue_len']:2d}  infer_s={result['infer_s']*1000:5.1f}  "
            f"cmd shape={cmd.shape} range=[{cmd.min():+.3f}, {cmd.max():+.3f}]"
        )

    def _preprocess(self, observation: Any) -> Dict[str, Any]:
        """Map raiden-style Observation → YamFlexPiPolicy observation dict.

        Accepts either:
          - an object with attributes ``.cameras`` (iterable) and
            ``.proprios`` (dict), per raiden's contract, OR
          - a plain ``dict`` with keys ``"cameras"`` and ``"proprios"`` (for
            tests that don't want to mock a class).

        Returns the dict YamFlexPiPolicy.infer_action_chunk consumes:
          ``{"rgb": ..., "depth": ..., "state_32": ..., "intrinsics": ...}``.
        """
        if isinstance(observation, dict):
            cameras = observation.get("cameras", [])
            proprios = observation.get("proprios", {})
        else:
            cameras = list(getattr(observation, "cameras", []) or [])
            proprios = dict(getattr(observation, "proprios", {}) or {})

        rgb: Dict[str, np.ndarray] = {}
        depth: Dict[str, np.ndarray] = {}
        intrinsics_by_cam: Dict[str, np.ndarray] = {}
        seen_raiden_names: list[str] = []

        for cam in cameras:
            raiden_name = getattr(cam, "name", None)
            seen_raiden_names.append(str(raiden_name))
            if raiden_name is None:
                continue
            flexpi_name = _RAIDEN_TO_FLEXPI_CAM.get(raiden_name)
            if flexpi_name is None:
                if self._verbose and not self._first_obs_logged:
                    print(
                        f"[yam-bridge] skipping camera {raiden_name!r}: "
                        f"not in raiden→FlexPi mapping {list(_RAIDEN_TO_FLEXPI_CAM)}"
                    )
                continue

            image = getattr(cam, "image", None)
            if image is None:
                raise ValueError(f"Camera {raiden_name!r}: missing .image attribute")
            rgb[flexpi_name] = np.asarray(image)

            d = getattr(cam, "depth", None)
            if d is not None:
                d_arr = np.asarray(d)
                depth[flexpi_name] = self._cast_depth_to_uint16_mm(d_arr, raiden_name)

            K = getattr(cam, "intrinsics", None)
            if K is not None:
                K_arr = np.asarray(K, dtype=np.float32)
                if K_arr.shape != (3, 3):
                    raise ValueError(
                        f"Camera {raiden_name!r}: intrinsics shape {K_arr.shape} "
                        "expected (3, 3)"
                    )
                intrinsics_by_cam[flexpi_name] = K_arr

        # Completeness checks. ``_CAM_ORDER`` comes from deploy_policy.py
        # (which sources it from the YAM training shape_meta); we don't
        # re-declare the canonical cam set here.
        missing_rgb = set(_CAM_ORDER) - set(rgb)
        if missing_rgb:
            raise ValueError(
                f"Bridge needs all of {_CAM_ORDER} (post raiden→FlexPi mapping); "
                f"missing {sorted(missing_rgb)}. raiden cameras seen: {seen_raiden_names}. "
                f"Mapping in use: {_RAIDEN_TO_FLEXPI_CAM}"
            )
        if depth:
            missing_depth = set(_CAM_ORDER) - set(depth)
            if missing_depth:
                raise ValueError(
                    f"Partial depth supplied: missing {sorted(missing_depth)}. "
                    "Either supply depth for all 3 cameras or omit depth entirely."
                )
        if intrinsics_by_cam:
            missing_K = set(_CAM_ORDER) - set(intrinsics_by_cam)
            if missing_K:
                raise ValueError(
                    f"Partial intrinsics supplied: missing {sorted(missing_K)}. "
                    "Either supply K for all 3 cameras or omit intrinsics entirely "
                    "(policy will fall back to its constructor-cached K)."
                )

        # State_32: prefer caller-provided, else FK from joint positions.
        if "state_32" in proprios:
            state_32 = np.asarray(proprios["state_32"], dtype=np.float32)
            state_source = "proprios['state_32']"
        else:
            if "follower_r_joint_pos" not in proprios or "follower_l_joint_pos" not in proprios:
                raise ValueError(
                    "proprios must contain either 'state_32' or BOTH "
                    "'follower_r_joint_pos' and 'follower_l_joint_pos'. "
                    f"Got keys: {list(proprios)}"
                )
            r_pos = np.asarray(proprios["follower_r_joint_pos"], dtype=np.float32)
            l_pos = np.asarray(proprios["follower_l_joint_pos"], dtype=np.float32)
            state_32 = _raiden_to_eef32_state(r_pos, l_pos)
            state_source = "FK via raiden.policy_server"

        if state_32.shape != (32,):
            raise ValueError(f"state_32 shape {state_32.shape} != (32,)")

        K_stack: Optional[np.ndarray] = None
        if intrinsics_by_cam:
            K_stack = np.stack([intrinsics_by_cam[c] for c in _CAM_ORDER], axis=0).astype(np.float32)

        depth_dict: Optional[Dict[str, np.ndarray]] = depth if depth else None

        if self._verbose and not self._first_obs_logged:
            self._log_first_obs(rgb, depth_dict, K_stack, state_32, state_source, seen_raiden_names)
            self._first_obs_logged = True

        return {
            "rgb": rgb,
            "depth": depth_dict,
            "state_32": state_32,
            "intrinsics": K_stack,
        }

    @staticmethod
    def _cast_depth_to_uint16_mm(d_arr: np.ndarray, raiden_name: str) -> np.ndarray:
        """Cast camera-supplied depth to the policy's uint16-mm contract.

        Two accepted input dtypes:
        - ``float32 / float64`` interpreted as **metres** (raiden's ZED-output
          contract; see ``maniflow_bridge._emit_depth``). Cast: ``d * 1000``
          clip to ``[0, 65535]`` and convert to ``uint16``.
        - ``uint16`` interpreted as **mm** already (offline tests that read
          straight from the dataset's FFV1 MKV stay byte-identical to disk).
        """
        if d_arr.dtype == np.uint16:
            return d_arr
        if d_arr.dtype in (np.float32, np.float64):
            return (d_arr.astype(np.float32) * 1000.0).clip(0, 65535).astype(np.uint16)
        raise TypeError(
            f"Camera {raiden_name!r}: depth dtype {d_arr.dtype} unsupported. "
            "Expected float32 metres (raiden contract) or uint16 mm."
        )

    def _log_first_obs(
        self,
        rgb: Dict[str, np.ndarray],
        depth: Optional[Dict[str, np.ndarray]],
        K_stack: Optional[np.ndarray],
        state_32: np.ndarray,
        state_source: str,
        seen_raiden_names: list[str],
    ) -> None:
        print("[yam-bridge] === first observation conversion ===")
        print(f"  raiden cams seen: {seen_raiden_names}")
        print(f"  raiden→flexpi mapping: {_RAIDEN_TO_FLEXPI_CAM}")
        print(f"  canonical cam order (from deploy_policy._CAM_ORDER): {_CAM_ORDER}")
        for cam in _CAM_ORDER:
            r = rgb[cam]
            line = f"  [{cam}] rgb shape={r.shape} dtype={r.dtype}"
            if depth is not None:
                d = depth[cam]
                line += f"  depth shape={d.shape} dtype={d.dtype} range=[{int(d.min())}, {int(d.max())}] mm"
            if K_stack is not None:
                k = K_stack[_CAM_ORDER.index(cam)]
                line += f"  K=fx{k[0,0]:.1f} fy{k[1,1]:.1f} cx{k[0,2]:.1f} cy{k[1,2]:.1f}"
            print(line)
        print(
            f"  state_32 shape={state_32.shape} range=[{state_32.min():+.4f}, {state_32.max():+.4f}]  "
            f"(source: {state_source})"
        )
        print(
            "  depth unit assumption: float32 metres (raiden) → uint16 mm (policy); "
            "uint16 inputs passed through unchanged"
        )
        if K_stack is None:
            print(
                "  intrinsics source: none in observation → policy uses constructor-cached "
                "K from --intrinsics-json (if any). If neither is provided, "
                "policy.infer_action_chunk will raise."
            )


# ---------------------------------------------------------------------------
# Offline bridge test CLI
# ---------------------------------------------------------------------------


class _MockCamera:
    """Minimal raiden-Camera stand-in for offline tests. Read-only attrs."""
    __slots__ = ("name", "image", "depth", "intrinsics")
    def __init__(self, name, image, depth=None, intrinsics=None):
        self.name = name; self.image = image; self.depth = depth; self.intrinsics = intrinsics


class _MockObservation:
    """Minimal raiden-Observation stand-in. ``.cameras`` + ``.proprios``."""
    __slots__ = ("cameras", "proprios")
    def __init__(self, cameras, proprios):
        self.cameras = cameras; self.proprios = proprios


def _build_mock_obs_from_recorded(
    data_dir: Path,
    episode_idx: int,
    frame_idx: int,
    fps: Optional[int] = None,
) -> tuple[_MockObservation, str]:
    """Reuse smoke_test's recorded helper and wrap it in raiden-style mocks.

    Crucially, this round-trips depth through float32 metres so the bridge's
    metres→mm conversion path is exercised (production depth from ZED is
    float32 metres).
    """
    from experiments.yam.flexpi_policy.smoke_test import build_obs_recorded

    obs_policy, instruction = build_obs_recorded(
        data_dir=data_dir,
        episode_idx=episode_idx,
        frame_idx=frame_idx,
        fps_hint=fps,
    )

    cameras = []
    for raiden_name, flexpi_name in _RAIDEN_TO_FLEXPI_CAM.items():
        image = obs_policy["rgb"][flexpi_name]
        depth_mm = obs_policy["depth"][flexpi_name]
        depth_metres = depth_mm.astype(np.float32) / 1000.0  # mm → metres for bridge to re-convert
        K = obs_policy["intrinsics"][_CAM_ORDER.index(flexpi_name)]  # (3, 3) float32
        cameras.append(_MockCamera(raiden_name, image=image, depth=depth_metres, intrinsics=K))

    proprios = {"state_32": obs_policy["state_32"]}  # offline path: skip FK
    return _MockObservation(cameras, proprios), instruction


def _build_offline_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Offline bridge test: build a raiden-style mock observation from a "
            "recorded LeRobot dataset, run N act() calls through the bridge, "
            "and validate queue / command shapes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="Path to trained YAM FlexPi checkpoint.")
    p.add_argument("--stats", default=None, help="Path to dataset_stats.json (auto-detect if None).")
    p.add_argument("--intrinsics-json", default=None,
                   help="Optional camera_intrinsics.json. Bridge passes through obs.intrinsics "
                        "when present, so this is only used as a fallback inside the policy.")
    p.add_argument("--data-dir", required=True, help="LeRobot dataset root (recorded obs source).")
    p.add_argument("--episode-idx", type=int, default=0)
    p.add_argument("--frame-idx", type=int, default=0, help="Frame to start at; subsequent steps advance this.")
    p.add_argument("--fps", type=int, default=None, help="FPS override for depth decode.")
    p.add_argument("--num-steps", type=int, default=16,
                   help="How many act() calls to make. ≥ 2*replan_steps to validate the second inference.")
    p.add_argument("--replan-steps", type=int, default=8)
    p.add_argument("--action-source", choices=["model_joint", "eef_ik"], default="model_joint")
    p.add_argument("--prompt", default="", help="Task instruction. Auto-read from tasks.jsonl if empty.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--action-horizon", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--advance-frames", action="store_true",
                   help="Advance frame_idx by 1 between act() calls. Default: hold the same frame for all calls.")
    return p


def _offline_main() -> int:
    args = _build_offline_parser().parse_args()
    ckpt = Path(args.ckpt).expanduser().resolve()
    if not ckpt.exists():
        print(f"ERROR: --ckpt not found: {ckpt}", file=sys.stderr); return 2
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        print(f"ERROR: --data-dir not found: {data_dir}", file=sys.stderr); return 2

    print(f"[offline-bridge-test] ckpt={ckpt}")
    print(f"[offline-bridge-test] data_dir={data_dir} ep={args.episode_idx} frame={args.frame_idx}")
    print(f"[offline-bridge-test] num_steps={args.num_steps} replan_steps={args.replan_steps} "
          f"action_source={args.action_source} advance_frames={args.advance_frames}")

    # Build bridge first so we can ask the policy how long the action chunk is.
    bridge = YamFlexPiBridge(
        checkpoint_path=str(ckpt),
        dataset_stats_path=args.stats,
        intrinsics_json=args.intrinsics_json,
        device=args.device,
        mixed_precision=args.mixed_precision,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        replan_steps=args.replan_steps,
        action_source=args.action_source,
        prompt=args.prompt,
        verbose=args.verbose,
    )

    expected_cmd_dim = 14 if args.action_source == "model_joint" else 20

    # Issue act() N times. Only rebuild the observation when the bridge will
    # actually use it (next call will trigger inference). Otherwise reuse the
    # last obs — the from-queue path inside ``bridge.act`` ignores observation
    # entirely, so re-decoding video files per step is wasted work AND piles
    # up fork+thread state until ffmpeg subprocesses deadlock.
    n_fresh = 0
    n_queue = 0
    last_cmd: Optional[np.ndarray] = None
    last_obs = None
    obs_builds = 0
    for step in range(args.num_steps):
        if last_obs is None or bridge.needs_fresh_observation():
            frame_idx = args.frame_idx + (step if args.advance_frames else 0)
            try:
                last_obs, instr = _build_mock_obs_from_recorded(
                    data_dir, args.episode_idx, frame_idx, args.fps,
                )
            except IndexError:
                print(f"[offline-bridge-test] hit end-of-episode at frame {frame_idx}; stopping.")
                break
            obs_builds += 1
            # First step: also set prompt from the dataset's tasks.jsonl if user didn't pass one.
            if step == 0 and not args.prompt:
                bridge.set_prompt(instr)
                print(f"[offline-bridge-test] prompt set from tasks.jsonl: {instr!r}")
        result = bridge.act(last_obs)
        cmd = result["command"]
        if cmd.shape != (expected_cmd_dim,):
            print(
                f"[offline-bridge-test] FAIL: cmd shape {cmd.shape} != ({expected_cmd_dim},) "
                f"for action_source={args.action_source}", file=sys.stderr,
            )
            return 1
        if not np.isfinite(cmd).all():
            print(
                f"[offline-bridge-test] FAIL: cmd contains non-finite values at step {step}: {cmd}",
                file=sys.stderr,
            )
            return 1
        n_fresh += (0 if result["from_queue"] else 1)
        n_queue += (1 if result["from_queue"] else 0)
        last_cmd = cmd

    print("\n[offline-bridge-test] === summary ===")
    print(f"  steps run:         {n_fresh + n_queue}")
    print(f"  fresh inferences:  {n_fresh}")
    print(f"  from-queue steps:  {n_queue}")
    print(f"  obs builds:        {obs_builds}")
    print(f"  cmd dim:           {expected_cmd_dim}")
    if last_cmd is not None:
        print(f"  last cmd:          {np.array2string(last_cmd, precision=3, suppress_small=True)}")
    expected_fresh = max(1, (n_fresh + n_queue + bridge.replan_steps - 1) // bridge.replan_steps)
    if n_fresh != expected_fresh:
        print(
            f"  WARN: expected ~{expected_fresh} fresh inferences for "
            f"{n_fresh + n_queue} steps at replan_steps={bridge.replan_steps}, got {n_fresh}"
        )
    print("[offline-bridge-test] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_offline_main())
