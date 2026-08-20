"""Per-frame YAM directory source reader (eef32, openpi-aligned).

Reads a single raw YAM episode laid out as:

    <ep_dir>/
        metadata.json
        rgb/{head,left_wrist,right_wrist}/0000000000.jpg, ...
        depth/{head,left_wrist,right_wrist}/0000000000.npz   (key='depth', uint16 mm)
        lowdim/{head,left_wrist,right_wrist}/0000000000.npz  (keys: joints, action, intrinsics, ...)

and emits a 32D eef32 state/action stream built from the **pre-computed 26D
``action`` field stored in lowdim/head/*.npz** (NOT by recomputing FK from
joints). Byte-equivalent to
``yam_openpi/yam_dataset_builder/convert_yam_data_to_lerobot.py:build_state32``,
so FlexPi-built and openpi-built LeRobot datasets carry identical state/action
up to float32 noise.

Why no FK here:
  ``rd convert`` (raiden) recorded ``lowdim.action`` at recording time using
  raiden's live FK (``Kinematics(combine_arm_and_gripper_xml(...), "grasp_site")``).
  That is the same FK raiden runs at deployment. Re-running FK in the builder
  against a potentially stale static XML risked silently drifting the
  training-time coordinate frame away from deploy. The fix is to read what is
  already on disk.

Output 32D layout (matches yam_openpi/yam_eef_util.STATE_LAYOUT exactly):

    [ left_pos(3)  left_rot6d(6)
      right_pos(3) right_rot6d(6)
      left_grip(1) right_grip(1)
      left_joint(6) right_joint(6) ]

rd-convert lowdim.action layout (26D):

    [ right_pos(3),  right_rot9(9), right_grip(1),
      left_pos(3),   left_rot9(9),  left_grip(1) ]

rot6d convention: first two ROWS of R, row-major flattened. Matches raiden's
``_rot9_to_rot6d`` and FlexPi's ``yam_eef.rot9_to_rot6d``.

Action convention (matches yam_openpi/convert_yam_data_to_lerobot.py):

    state[t]  = state32[t]
    action[t] = state32[t+1]
    drop the last raw frame so the parquet contains exactly N_raw - 1 rows.
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Vendored FlexPi yam_eef helpers (rot9_to_rot6d, STATE_DIM, STATE_LAYOUT,
# pose9d_to_mat). Math is ported verbatim from yam_openpi.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]          # scripts/yam_dataset_builder -> flex-pi
_FLEXPI_LEROBOT_ROOT = _PROJECT_ROOT / "src" / "flexpi" / "datasets" / "lerobot"
if str(_FLEXPI_LEROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLEXPI_LEROBOT_ROOT))
from utils import yam_eef  # noqa: E402


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@dataclass
class SourceLayout:
    """Configurable per-frame directory schema.

    Defaults match the YAM raw layout (place_lock_simple_raw,
    clean_up_table_V2_1, put_plate_to_rack_v4, etc.). Override via convert.py
    CLI flags only when converting datasets with non-standard naming.
    """
    rgb_subdir: str = "rgb"
    depth_subdir: str = "depth"
    lowdim_subdir: str = "lowdim"
    metadata_filename: str = "metadata.json"

    # Source camera dir name -> output LeRobot camera key. The default already
    # produces FlexPi canonical names — no rename pass required.
    camera_map: dict = field(default_factory=lambda: {
        "head": "cam_high",
        "left_wrist": "cam_left_wrist",
        "right_wrist": "cam_right_wrist",
    })

    # File extensions
    rgb_ext: str = ".jpg"
    depth_ext: str = ".npz"
    lowdim_ext: str = ".npz"

    # Keys inside the per-frame npz files
    depth_npz_key: str = "depth"
    joints_npz_key: str = "joints"          # (14,) float32 joint+grip stream
    action_npz_key: str = "action"          # (26,) float32 rd-convert FK output
    intrinsics_npz_key: str = "intrinsics"  # (4,) [fx, fy, cx, cy]

    # Output state/action dimensionality (locked at 32 — eef32 only).
    state_dim: int = yam_eef.STATE_DIM

    # Frame filename format: {idx:0{pad}d}{ext}
    frame_filename_pad: int = 10

    # Nested-key paths into metadata.json
    num_frames_path: tuple = ("num_frames",)
    fps_path: tuple = ("framerate",)
    prompt_path: tuple = ("language", "prompt")  # must resolve to list[str]


# Episode directory name pattern: "0000", "0001", ...
_EPISODE_RE = re.compile(r"^\d+$")


def discover_episodes(raw_dir: Path) -> list[Path]:
    eps = [p for p in raw_dir.iterdir() if p.is_dir() and _EPISODE_RE.match(p.name)]
    if not eps:
        raise FileNotFoundError(f"No episode subdirs (0000, 0001, ...) under {raw_dir}")
    return sorted(eps, key=lambda p: int(p.name))


def _walk(d: dict, path: tuple) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"metadata.json missing key path: {'.'.join(path)}")
        cur = cur[k]
    return cur


def _frame_path(subdir: Path, cam: str, idx: int, pad: int, ext: str) -> Path:
    return subdir / cam / f"{idx:0{pad}d}{ext}"


# ---------------------------------------------------------------------------
# 32D builder (openpi-aligned: read pre-computed action26, no FK)
# ---------------------------------------------------------------------------

def build_state32(joints14: np.ndarray, action26: np.ndarray) -> np.ndarray:
    """Assemble the 32D eef32 state vector from raw joints14 + rd-convert action26.

    Byte-equivalent to
    ``yam_openpi/yam_dataset_builder/convert_yam_data_to_lerobot.py:build_state32``.

    Inputs are batched ``(N, 14)`` and ``(N, 26)``; output is ``(N, 32)``.

    Slot map::

        0: 3   left_pos       <- action26[13:16]
        3: 9   left_rot6d     <- rot9_to_rot6d(action26[16:25])   (first two ROWS)
        9:12   right_pos      <- action26[ 0: 3]
        12:18  right_rot6d    <- rot9_to_rot6d(action26[ 3:12])
        18     left_gripper   <- action26[25]
        19     right_gripper  <- action26[12]
        20:26  left_joint     <- joints14[ 7:13]
        26:32  right_joint    <- joints14[ 0: 6]
    """
    if joints14.ndim != 2 or joints14.shape[1] != 14:
        raise ValueError(f"joints14 must be (N, 14), got {joints14.shape}")
    n = joints14.shape[0]
    if action26.shape != (n, 26):
        raise ValueError(f"action26 must be (N, 26) with N={n}; got {action26.shape}")

    out = np.empty((n, yam_eef.STATE_DIM), dtype=np.float32)
    out[:,  0: 3] = action26[:, 13:16]
    out[:,  3: 9] = yam_eef.rot9_to_rot6d(action26[:, 16:25])
    out[:,  9:12] = action26[:,  0: 3]
    out[:, 12:18] = yam_eef.rot9_to_rot6d(action26[:,  3:12])
    out[:, 18]    = action26[:, 25]
    out[:, 19]    = action26[:, 12]
    out[:, 20:26] = joints14[:,  7:13]
    out[:, 26:32] = joints14[:,  0: 6]
    return out


def validate_state32_first_episode(
    ep_dir: Path,
    layout: SourceLayout,
    n_check: int = 10,
) -> None:
    """One-shot SE(3) round-trip + orthonormality check on episode 0.

    Mirrors yam_openpi's ``validate_eef32_first_episode`` (minus the raiden FK
    cross-check — this builder no longer touches raiden at runtime). Confirms:
      - state32 reconstructs the SE(3) the rd-convert action26 encodes
      - every recovered R is orthonormal with det ≈ +1
      - rot6d row convention is correct (first two ROWS)

    Raises AssertionError on failure (the convert() caller aborts the build).
    """
    meta = json.loads((ep_dir / layout.metadata_filename).read_text())
    n_total = int(_walk(meta, layout.num_frames_path))
    n = min(n_total, n_check)
    lowdim_dir = ep_dir / layout.lowdim_subdir
    head_cam = next(iter(layout.camera_map.keys()))
    joints14 = np.empty((n, 14), dtype=np.float32)
    action26 = np.empty((n, 26), dtype=np.float32)
    for i in range(n):
        with np.load(
            _frame_path(lowdim_dir, head_cam, i, layout.frame_filename_pad, layout.lowdim_ext),
            allow_pickle=True,
        ) as z:
            joints14[i] = z[layout.joints_npz_key]
            action26[i] = z[layout.action_npz_key]
    state32 = build_state32(joints14, action26)

    for arm, state_slice, pos26, rot9_26 in [
        ("left",  slice(0, 9),  slice(13, 16), slice(16, 25)),
        ("right", slice(9, 18), slice( 0,  3), slice( 3, 12)),
    ]:
        T_from_state = yam_eef.pose9d_to_mat(state32[:, state_slice])  # (n, 4, 4)
        T_expected = np.tile(np.eye(4), (n, 1, 1))
        T_expected[:, :3, 3] = action26[:, pos26]
        T_expected[:, :3, :3] = action26[:, rot9_26].reshape(n, 3, 3)
        np.testing.assert_allclose(
            T_from_state[:, :3, 3], T_expected[:, :3, 3], atol=1e-5,
            err_msg=f"{arm} xyz mismatch (state32 vs rd-convert action26)",
        )
        np.testing.assert_allclose(
            T_from_state[:, :3, :3], T_expected[:, :3, :3], atol=1e-5,
            err_msg=f"{arm} rotation mismatch — likely a rot6d row/col convention bug",
        )
        for k in range(n):
            R = T_from_state[k, :3, :3]
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), f"{arm} frame {k}: R Rᵀ ≠ I"
            assert abs(np.linalg.det(R) - 1.0) < 1e-5, f"{arm} frame {k}: det(R) ≠ 1"


# ---------------------------------------------------------------------------
# Episode loader
# ---------------------------------------------------------------------------

def load_episode(ep_dir: Path, layout: SourceLayout) -> dict:
    """Load one raw YAM episode and emit a dict ready for convert._encode_episode.

    Returned dict (eef32 only)::

        {
            "state":     (N-1, 32) float32,  state32[:-1] built from joints14 + action26
            "action":    (N-1, 32) float32,  state32[1:]
            "rgb":       {output_cam: (N-1, H, W, 3) uint8 RGB},
            "depth":     {output_cam: (N-1, H, W)    uint16 mm},
            "intrinsic": {output_cam: (3, 3)         float32 — at SOURCE resolution},
            "prompts":   list[str],
            "fps":       int,
        }

    The last raw frame is dropped (yam_openpi convention). All streams are
    trimmed to the same N-1.
    """
    meta = json.loads((ep_dir / layout.metadata_filename).read_text())
    N_raw = int(_walk(meta, layout.num_frames_path))
    if N_raw < 2:
        raise ValueError(f"{ep_dir}: need >=2 raw frames for action shift; got {N_raw}")
    fps = int(_walk(meta, layout.fps_path))
    prompts = list(_walk(meta, layout.prompt_path))
    if not prompts or not all(isinstance(p, str) for p in prompts):
        raise ValueError(f"{ep_dir}: metadata prompt path must yield list[str], got {prompts!r}")

    rgb_dir = ep_dir / layout.rgb_subdir
    depth_dir = ep_dir / layout.depth_subdir
    lowdim_dir = ep_dir / layout.lowdim_subdir

    src_cams = list(layout.camera_map.keys())
    head_cam = src_cams[0]   # joints14 + action26 are identical across cameras

    # ---- joints14 + action26 from lowdim/head (threaded I/O) ----
    joints14 = np.empty((N_raw, 14), dtype=np.float32)
    action26 = np.empty((N_raw, 26), dtype=np.float32)

    def _load_lowdim_head(t: int) -> None:
        npz_path = _frame_path(lowdim_dir, head_cam, t, layout.frame_filename_pad, layout.lowdim_ext)
        with np.load(npz_path, allow_pickle=True) as z:
            j = z[layout.joints_npz_key]
            a = z[layout.action_npz_key]
        if j.shape != (14,):
            raise ValueError(f"{npz_path}: expected joints shape (14,), got {j.shape}")
        if a.shape != (26,):
            raise ValueError(f"{npz_path}: expected action shape (26,), got {a.shape}")
        joints14[t] = j.astype(np.float32, copy=False)
        action26[t] = a.astype(np.float32, copy=False)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_load_lowdim_head, range(N_raw)))

    # ---- 32D state (openpi build_state32); action shifted by one ----
    state32_full = build_state32(joints14, action26)
    state = state32_full[:-1]
    action = state32_full[1:]
    N = N_raw - 1

    # ---- RGB / depth / intrinsics (per camera) ----
    rgb: dict[str, np.ndarray] = {}
    depth: dict[str, np.ndarray] = {}
    intrinsic: dict[str, np.ndarray] = {}

    for src_cam, out_cam in layout.camera_map.items():
        # RGB. Source streams may have N_raw or N_raw + 1 files (the +1 covers
        # the next-state used for the dropped last action). We read exactly N
        # = N_raw - 1 here, matching state.shape[0]. JPG decode uses cv2
        # (libjpeg-turbo) and a thread pool — both release the GIL during
        # decode, so threading actually speeds it up.
        def _read_rgb(t: int, src_cam=src_cam) -> np.ndarray:
            jpg_path = _frame_path(rgb_dir, src_cam, t, layout.frame_filename_pad, layout.rgb_ext)
            bgr = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"{jpg_path}: cv2.imread returned None (missing or corrupt JPG)")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Read frame 0 to learn (H, W), then pre-allocate and fill the rest
        # from worker threads (avoids the np.stack copy).
        img0 = _read_rgb(0)
        H, W = img0.shape[:2]
        rgb_buf = np.empty((N, H, W, 3), dtype=np.uint8)
        rgb_buf[0] = img0
        if N > 1:
            def _fill_rgb(t: int) -> None:
                rgb_buf[t] = _read_rgb(t)
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(_fill_rgb, range(1, N)))
        rgb[out_cam] = rgb_buf

        # Depth. .npz is a zip wrapping one uint16 array; per-file zip parse
        # is the dominant cost so threading the loads matters here too.
        depth_frames = np.empty((N, H, W), dtype=np.uint16)

        def _fill_depth(t: int, src_cam=src_cam) -> None:
            npz_path = _frame_path(depth_dir, src_cam, t, layout.frame_filename_pad, layout.depth_ext)
            with np.load(npz_path, allow_pickle=True) as z:
                d = z[layout.depth_npz_key]
            if d.dtype != np.uint16:
                raise ValueError(f"{npz_path}: depth must be uint16 (mm), got {d.dtype}")
            if d.shape != (H, W):
                raise ValueError(f"{npz_path}: depth shape {d.shape} != RGB shape ({H},{W})")
            depth_frames[t] = d

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_fill_depth, range(N)))
        depth[out_cam] = depth_frames

        # Intrinsics (frame 0; assumed static within an episode).
        npz_path = _frame_path(lowdim_dir, src_cam, 0, layout.frame_filename_pad, layout.lowdim_ext)
        with np.load(npz_path, allow_pickle=True) as z:
            K_vec = z[layout.intrinsics_npz_key]
        if K_vec.shape != (4,):
            raise ValueError(f"{npz_path}: intrinsics must be (4,) [fx,fy,cx,cy], got {K_vec.shape}")
        fx, fy, cx, cy = (float(x) for x in K_vec)
        intrinsic[out_cam] = np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    return {
        "state": state,
        "action": action,
        "rgb": rgb,
        "depth": depth,
        "intrinsic": intrinsic,
        "prompts": prompts,
        "fps": fps,
    }
