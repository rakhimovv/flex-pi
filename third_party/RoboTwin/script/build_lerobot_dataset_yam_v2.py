#!/usr/bin/env python3
"""
build_lerobot_dataset_yam_v2.py
===============================
Convert `rd convert` processed YAM episodes into a LeRobot v2.0 dataset
(Parquet + pre-decoded frames) ready for training with the YAM × ManiFlow
pipeline.  No video encoding — frames are written directly.

Key improvements over v1:
  - FK-computed EEF poses (no more zero placeholders)
  - Index-based frame alignment (no timestamp unit mismatches)
  - Camera intrinsics read directly from per-frame lowdim NPZ files
  - RGB: visually-lossless H.264 (CRF 18, yuv420p) in videos/.
  - Depth: lossless dual-row H.264 (libx264rgb rgb24 CRF 0 all-intra) in videos/.
    Each (H, W) uint16 depth frame is packed as (2H, W, 3) uint8:
    top H rows = lo byte, bottom H rows = hi byte.  Decoded by
    _decode_lossless_depth_frame() in the starVLA sharded dataloader.

Input layout (from `rd convert`):
    <processed_root>/                          ← --processed-dir points here
        <task_name_a>/                         ← auto-detected task subdirectory
            0000/
                metadata.json
                rgb/{head,left_wrist,right_wrist}/<frame_idx:010d>.jpg
                depth/{head,left_wrist,right_wrist}/<frame_idx:010d>.npz
                lowdim/{head,left_wrist,right_wrist}/<frame_idx:010d>.npz
            0001/
            ...
        <task_name_b>/
            0000/
            ...

    Also supports pointing directly at a single task directory (backward compat):
    <task_dir>/
        0000/
            metadata.json
            ...
        0001/
        ...

Output layout:
    <output_dir>/
        data/chunk-000/
            episode_000000.parquet        ← 28D state + action per frame
        videos/chunk-000/
            observation.images.head_rgb/
                episode_000000.mp4         ← H.264 yuv420p CRF 18
            observation.images.head_depth/
                episode_000000.mp4         ← lossless H.264 libx264rgb rgb24 CRF 0, dual-row (2H,W,3)
            ...  (wrist_r, wrist_l same pattern)
        meta/
            info.json
            stats.json / stats_gr00t.json
            tasks.jsonl / episodes.jsonl
            modality.json
            {head,wrist_r,wrist_l}_camera_info.json

Usage:
    # Multi-task (processed_root contains task subdirs):
    python yam/build_lerobot_dataset_yam_v2.py \
        --processed-dir /path/to/YAM_robot/data/processed \
        --output-dir $YAM_DATA_ROOT/lerobot_yam_v2 \
        [--fps 30] [--rewrite]

    # Single-task (backward compat — point directly at a task dir):
    python yam/build_lerobot_dataset_yam_v2.py \
        --processed-dir /path/to/YAM_robot/data/processed/place_lock \
        --output-dir $YAM_DATA_ROOT/lerobot_yam_v2
"""

# ===========================================================================
# §0  Imports
# ===========================================================================
import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation
from tqdm import tqdm

# ===========================================================================
# §1  Constants
# ===========================================================================

FPS = 30

# 32D state/action vector layout (6D rotation, Zhou et al. CVPR 2019)
# ┌───────────────┬──────┬──────────────────────────────────────────────┐
# │ Field         │ Idx  │ Description                                  │
# ├───────────────┼──────┼──────────────────────────────────────────────┤
# │ left_pos      │ 0:3  │ left EEF position (x,y,z)  metres           │
# │ left_ori      │ 3:9  │ left EEF 6D rotation (first 2 cols of R)    │
# │ right_pos     │ 9:12 │ right EEF position (x,y,z) metres           │
# │ right_ori     │12:18 │ right EEF 6D rotation (first 2 cols of R)   │
# │ left_grip     │18:19 │ left gripper width (normalised [0,1])       │
# │ right_grip    │19:20 │ right gripper width (normalised [0,1])      │
# │ left_joint    │20:26 │ left arm joint angles, radians (6 DOF)      │
# │ right_joint   │26:32 │ right arm joint angles, radians (6 DOF)     │
# └───────────────┴──────┴──────────────────────────────────────────────┘
STATE_DIM = 32

# Camera name mapping: rd-convert name → lerobot name
# rd-convert uses: head, left_wrist, right_wrist
# lerobot uses:    head, wrist_l, wrist_r
CAMERAS_RD = ["head", "right_wrist", "left_wrist"]
CAM_RD_TO_LEROBOT = {
    "head": "head",
    "right_wrist": "wrist_r",
    "left_wrist": "wrist_l",
}

TASK_NAME_DEFAULT = "yam_teleoperation"

# ===========================================================================
# §2  Text / task registry
# ===========================================================================

class TextRegistry:
    """Maps task names to integer IDs and stores task instructions."""

    def __init__(self):
        self._tasks: Dict[int, str] = {}
        self._counter = 0

    def register(self, task_name: str) -> int:
        for tid, tname in self._tasks.items():
            if tname == task_name:
                return tid
        tid = self._counter
        self._tasks[tid] = task_name
        self._counter += 1
        return tid

    def items(self):
        return self._tasks.items()

    def to_jsonl(self, path: Path):
        with open(path, "w") as f:
            for tid, tname in sorted(self._tasks.items()):
                f.write(json.dumps({"task_index": tid, "task": tname}) + "\n")


# ===========================================================================
# §3  YAMAdapterV2 — rd-convert processed data → normalised 32D vectors
# ===========================================================================

def _batch_rot9_to_rot6d(rot9_flat: np.ndarray) -> np.ndarray:
    """Convert (N, 9) flattened rotation matrices to (N, 6) continuous 6D rotation.

    Extracts the first two columns of the 3×3 rotation matrix and flattens
    them into a 6D vector [col0(3), col1(3)].  This is the Zhou et al.
    CVPR 2019 representation, compatible with pytorch3d rotation_6d_to_matrix().
    """
    R = rot9_flat.reshape(-1, 3, 3)                      # (N, 3, 3)
    col0 = R[:, :, 0]                                    # (N, 3)
    col1 = R[:, :, 1]                                    # (N, 3)
    return np.concatenate([col0, col1], axis=-1).astype(np.float32)  # (N, 6)


class YAMAdapterV2:
    """
    Loads a single rd-convert processed episode and exposes per-frame
    state/action vectors plus image paths.

    Data alignment is purely index-based: frame i in lowdim corresponds
    to frame i in rgb and depth.  No timestamp matching is needed.

    rd-convert lowdim NPZ fields:
        joints  (14,): [right_arm(6), right_grip(1), left_arm(6), left_grip(1)]
        action  (26,): [right_pos(3), right_rot9(9), right_grip(1),
                        left_pos(3),  left_rot9(9),  left_grip(1)]
        intrinsics (4,): [fx, fy, cx, cy]
        extrinsics (4,4): cam2world homogeneous matrix
    """

    def __init__(self, episode_dir: Path):
        self.episode_dir = Path(episode_dir)
        self._metadata: Optional[dict] = None
        self._all_joints: Optional[np.ndarray] = None   # (T, 14)
        self._all_actions: Optional[np.ndarray] = None   # (T, 26)

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            with open(self.episode_dir / "metadata.json") as f:
                self._metadata = json.load(f)
        return self._metadata

    @property
    def num_frames(self) -> int:
        return self.metadata["num_frames"]

    def task_description(self) -> str:
        lang = self.metadata.get("language", {})
        prompts = lang.get("prompt", [])
        if prompts and prompts[0].strip():
            return prompts[0].strip()
        task = lang.get("task", "")
        if task.strip():
            return task.strip()
        return TASK_NAME_DEFAULT

    def _load_all_lowdim(self) -> None:
        """Bulk-load all lowdim NPZ files into two arrays.

        Uses threaded I/O to parallelize the ~T small-file reads.
        """
        if self._all_joints is not None:
            return
        T = self.num_frames
        lowdim_dir = self.episode_dir / "lowdim" / "head"
        all_joints = np.empty((T, 14), dtype=np.float32)
        all_actions = np.empty((T, 26), dtype=np.float32)

        def _load_one(i: int) -> None:
            ld = np.load(lowdim_dir / f"{i:010d}.npz", allow_pickle=True)
            all_joints[i] = ld["joints"]
            all_actions[i] = ld["action"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_load_one, range(T)))

        self._all_joints = all_joints
        self._all_actions = all_actions

    def _build_32d_batch(self, joints: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Convert (T, 14) joints + (T, 26) actions → (T, 32) state vectors.

        Rotation is stored as 6D (first two columns of R, Zhou et al. CVPR 2019).
        """
        T = len(joints)
        # Batch rotation conversion: right rot9 and left rot9 together
        right_rot9 = actions[:, 3:12]    # (T, 9)
        left_rot9 = actions[:, 16:25]    # (T, 9)
        all_rot9 = np.concatenate([right_rot9, left_rot9], axis=0)  # (2T, 9)
        all_rot6d = _batch_rot9_to_rot6d(all_rot9)                  # (2T, 6)
        right_rot6d = all_rot6d[:T]   # (T, 6)
        left_rot6d = all_rot6d[T:]    # (T, 6)

        out = np.empty((T, STATE_DIM), dtype=np.float32)
        out[:, 0:3]   = actions[:, 13:16]    # left_pos
        out[:, 3:9]   = left_rot6d           # left_ori (6D rotation)
        out[:, 9:12]  = actions[:, 0:3]      # right_pos
        out[:, 12:18] = right_rot6d          # right_ori (6D rotation)
        out[:, 18:19] = actions[:, 25:26]    # left_grip
        out[:, 19:20] = actions[:, 12:13]    # right_grip
        out[:, 20:26] = joints[:, 7:13]      # left_joint
        out[:, 26:32] = joints[:, 0:6]       # right_joint
        return out

    def state_vectors(self) -> np.ndarray:
        """Return (T, 32) state array at 30Hz.  Loads all lowdim once."""
        self._load_all_lowdim()
        return self._build_32d_batch(self._all_joints, self._all_actions)

    def action_vectors(self) -> np.ndarray:
        """Return (T, 32) action array at 30Hz.

        action[i] = state[i+1] (next-frame target).
        Last frame is repeated.
        """
        self._load_all_lowdim()
        # Shift joints and actions forward by 1 frame
        joints_shifted = np.concatenate([self._all_joints[1:], self._all_joints[-1:]], axis=0)
        actions_shifted = np.concatenate([self._all_actions[1:], self._all_actions[-1:]], axis=0)
        return self._build_32d_batch(joints_shifted, actions_shifted)

    def camera_intrinsics(self, rd_camera: str) -> dict:
        """Get camera intrinsics from frame 0 lowdim NPZ."""
        path = self.episode_dir / "lowdim" / rd_camera / "0000000000.npz"
        ld = np.load(path, allow_pickle=True)
        intr = ld["intrinsics"]  # (4,): [fx, fy, cx, cy]
        res = self.metadata["resolution"]  # [H, W]
        return {
            "fx": float(intr[0]),
            "fy": float(intr[1]),
            "cx": float(intr[2]),
            "cy": float(intr[3]),
            "depth_scale": 1000.0,  # mm → metres
            "width": res[1],
            "height": res[0],
            "model": "pinhole",
        }

    def rgb_dir(self, rd_camera: str) -> Path:
        return self.episode_dir / "rgb" / rd_camera

    def depth_dir(self, rd_camera: str) -> Path:
        return self.episode_dir / "depth" / rd_camera


# ===========================================================================
# §4  Frame output helpers
# ===========================================================================

def _copy_rgb_frames(
    src_dir: Path,
    dst_dir: Path,
    num_frames: int,
) -> None:
    """Copy RGB JPEGs from rd-convert to frames_root layout.

    rd-convert:  {src_dir}/{i:010d}.jpg
    frames_root: {dst_dir}/{i:06d}.jpg

    Uses os.link() (hardlink) for near-instant zero-copy on the same
    filesystem; falls back to shutil.copyfile() across filesystems.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        src = src_dir / f"{i:010d}.jpg"
        dst = dst_dir / f"{i:06d}.jpg"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)


def _encode_depth_video(
    depth_src_dir: Path,
    output_mp4: Path,
    num_frames: int,
    fps: int,
) -> None:
    """Encode depth NPZ frames as a dual-row lossless H.264 video.

    Each depth frame (H, W) uint16 mm is encoded as dual-row (2H, W, 3) uint8:
        top    H rows = lo byte  (depth_mm & 0xFF)
        bottom H rows = hi byte  (depth_mm >> 8)
    All 3 channels are identical (grayscale content stored as RGB so standard
    video tools can open the file).

    Codec: libx264rgb -pix_fmt rgb24 -crf 0  → truly lossless (no YUV conversion).
           libx264 with gbrp/crf=0 is NOT lossless — internal YUV conversion
           causes ±1 byte errors, giving max depth error ±257 mm.
    GOP  : -g 1 (all-intra) → every frame is a keyframe; decord can seek to
           any frame in O(1) with no partial-GOP decode penalty.
    """
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Read frame 0 to determine H, W
    first = np.load(str(depth_src_dir / f"{0:010d}.npz"))["depth"]
    H, W = first.shape

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{2 * H}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264rgb",   # RGB-mode H.264: no YUV conversion → truly lossless
        "-pix_fmt", "rgb24",    # keep packed RGB (libx264rgb requires rgb24 input)
        "-crf", "0",            # lossless quantizer
        "-g", "1",              # all-intra: every frame is independently seekable
        "-preset", "ultrafast", # fast encode (lossless quality is not affected)
        "-frames:v", str(num_frames),
        str(output_mp4),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(num_frames):
        depth_mm = np.load(str(depth_src_dir / f"{i:010d}.npz"))["depth"]  # (H, W) uint16
        lo = (depth_mm & 0xFF).astype(np.uint8)
        hi = (depth_mm >> 8).astype(np.uint8)
        dual = np.vstack([lo, hi])                     # (2H, W)
        frame = np.stack([dual, dual, dual], axis=-1)  # (2H, W, 3) uint8
        proc.stdin.write(frame.tobytes())
    # Do NOT call proc.stdin.close() before communicate() — Python 3.10's
    # _communicate() internally flushes stdin before closing it; calling
    # close() first raises "ValueError: flush of closed file".
    # communicate() handles the flush+close+wait sequence correctly.
    _, stderr_bytes = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed encoding {output_mp4}:\n{stderr_bytes.decode()}"
        )


# ===========================================================================
# §4b  RGB video encoding (for official GR00T loader compatibility)
# ===========================================================================

def _encode_rgb_video(frames_dir: Path, output_mp4: Path, num_frames: int, fps: int) -> None:
    """Encode pre-decoded JPEG frames into an MP4 file using ffmpeg.

    The output MP4 is stored at:
        videos/chunk-{i}/{original_key}/episode_{j}.mp4
    and is required by the official GR00T LeRobotEpisodeLoader, which uses
    torchcodec/decord to decode video frames at runtime.

    Depth cameras are encoded separately by _encode_depth_video() (dual-row
    lossless H.264 via libx264rgb rgb24 CRF 0 all-intra) — see process_episode().

    Args:
        frames_dir: Directory containing {i:06d}.jpg frames.
        output_mp4: Destination .mp4 path.
        num_frames: Number of frames to encode (safety cap for ffmpeg).
        fps: Frame rate for the output video.
    """
    import subprocess

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",           # visually lossless for robot RGB
        "-frames:v", str(num_frames),
        str(output_mp4),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed encoding {output_mp4}:\n{result.stderr}"
        )


# ===========================================================================
# §5  Per-episode processing
# ===========================================================================

def process_episode(
    episode_dir: Path,
    episode_idx: int,
    chunk_idx: int,
    output_dir: Path,
    text_registry: TextRegistry,
    fps: int,
) -> Dict[str, Any]:
    """Process one rd-convert episode → parquet rows + frames + depth videos."""
    adapter = YAMAdapterV2(episode_dir)
    T = adapter.num_frames
    res = adapter.metadata["resolution"]  # [H, W]
    H, W = res[0], res[1]

    states = adapter.state_vectors()    # (T, 32)
    actions = adapter.action_vectors()  # (T, 32)

    task_desc = adapter.task_description()
    task_id = text_registry.register(task_desc)

    ep_str = f"episode_{episode_idx:06d}"
    chunk_str = f"chunk-{chunk_idx:03d}"

    # ---- Parquet (columnar construction, no per-row dicts) ---------------
    import pandas as pd

    parquet_dir = output_dir / "data" / chunk_str
    parquet_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "episode_index": np.full(T, episode_idx, dtype=np.int64),
        "frame_index": np.arange(T, dtype=np.int64),
        "timestamp": np.arange(T, dtype=np.float32) / fps,
        "task_index": np.full(T, task_id, dtype=np.int64),
        "annotation.human.action.task_description": np.full(T, task_id, dtype=np.int64),
        "observation.state": states.tolist(),
        "action": actions.tolist(),
    })
    df.to_parquet(parquet_dir / f"{ep_str}.parquet", index=False)

    # ---- Copy/convert frames ------------------------------------------------
    frames_dir = output_dir / "frames"

    # RGB: hardlinks are near-instant, run inline (no need for thread pool)
    for rd_cam in CAMERAS_RD:
        lerobot_cam = CAM_RD_TO_LEROBOT[rd_cam]
        dst_dir = frames_dir / f"observation.images.{lerobot_cam}_rgb" / ep_str
        _copy_rgb_frames(adapter.rgb_dir(rd_cam), dst_dir, T)

    # ---- Encode RGB + depth as MP4 videos -----------------------------------
    # Both RGB and depth are stored in videos/ so decord can batch-load frames
    # efficiently with sorted indices (no per-frame file opens).
    videos_dir = output_dir / "videos"

    # RGB: visually-lossless H.264 (CRF 18, yuv420p) — same as before.
    for rd_cam in CAMERAS_RD:
        lerobot_cam = CAM_RD_TO_LEROBOT[rd_cam]
        original_key = f"observation.images.{lerobot_cam}_rgb"
        frames_src = frames_dir / f"observation.images.{lerobot_cam}_rgb" / ep_str
        mp4_dst = videos_dir / chunk_str / original_key / f"{ep_str}.mp4"
        _encode_rgb_video(frames_src, mp4_dst, T, fps)

    # Depth: lossless dual-row H.264 (libx264rgb rgb24 CRF 0 all-intra).
    # Encode 3 cameras in parallel — each spawns an ffmpeg subprocess, so
    # ThreadPoolExecutor overlaps I/O and encoding across cameras.
    def _enc_depth(rd_cam: str) -> None:
        lerobot_cam = CAM_RD_TO_LEROBOT[rd_cam]
        original_key = f"observation.images.{lerobot_cam}_depth"
        mp4_dst = videos_dir / chunk_str / original_key / f"{ep_str}.mp4"
        _encode_depth_video(adapter.depth_dir(rd_cam), mp4_dst, T, fps)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_enc_depth, rd_cam) for rd_cam in CAMERAS_RD]
        for f in futures:
            f.result()  # re-raise any encoding error

    return {
        "episode_index": episode_idx,
        "tasks": [task_desc],
        "length": T,
    }


# ===========================================================================
# §6  Global index rewriting
# ===========================================================================

def _rewrite_global_indices(output_dir: Path):
    """Add global frame index column to every parquet file."""
    import pandas as pd

    parquet_files = sorted((output_dir / "data").rglob("*.parquet"))
    global_idx = 0
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        df.insert(0, "index", range(global_idx, global_idx + len(df)))
        df.to_parquet(pf, index=False)
        global_idx += len(df)


# ===========================================================================
# §7  Metadata generation
# ===========================================================================

def generate_lerobot_metadata(
    output_dir: Path,
    episode_meta: List[Dict],
    text_registry: TextRegistry,
    first_adapter: YAMAdapterV2,
    fps: int,
):
    """Write all meta/*.json / *.jsonl files."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    res = first_adapter.metadata["resolution"]  # [H, W]
    H, W = res[0], res[1]

    # ---- episodes.jsonl --------------------------------------------------
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episode_meta:
            f.write(json.dumps(ep) + "\n")

    # ---- tasks.jsonl -----------------------------------------------------
    text_registry.to_jsonl(meta_dir / "tasks.jsonl")

    # ---- camera_info.json (one per camera) -------------------------------
    for rd_cam in CAMERAS_RD:
        lerobot_cam = CAM_RD_TO_LEROBOT[rd_cam]
        intrinsics = first_adapter.camera_intrinsics(rd_cam)
        with open(meta_dir / f"{lerobot_cam}_camera_info.json", "w") as f:
            json.dump(intrinsics, f, indent=2)

    # ---- modality.json ---------------------------------------------------
    # state/action: "start"/"end" are the required GR00T fields.
    # "original_key" is also used by LeRobotEpisodeLoader._extract_joint_groups()
    # and get_dataset_statistics() to look up the parquet column and stats key.
    # The extra fields (absolute, rotation_type, dtype) are custom metadata
    # ignored by the official loader but preserved for the starVLA pipeline.
    modality = {
        "state": {
            "left_pos":    {"start": 0,  "end": 3,  "original_key": "observation.state"},
            "left_ori":    {"start": 3,  "end": 9,  "original_key": "observation.state"},
            "right_pos":   {"start": 9,  "end": 12, "original_key": "observation.state"},
            "right_ori":   {"start": 12, "end": 18, "original_key": "observation.state"},
            "left_grip":   {"start": 18, "end": 19, "original_key": "observation.state"},
            "right_grip":  {"start": 19, "end": 20, "original_key": "observation.state"},
            "left_joint":  {"start": 20, "end": 26, "original_key": "observation.state"},
            "right_joint": {"start": 26, "end": 32, "original_key": "observation.state"},
        },
        "action": {
            "left_pos":    {"start": 0,  "end": 3,  "original_key": "action"},
            "left_ori":    {"start": 3,  "end": 9,  "original_key": "action"},
            "right_pos":   {"start": 9,  "end": 12, "original_key": "action"},
            "right_ori":   {"start": 12, "end": 18, "original_key": "action"},
            "left_grip":   {"start": 18, "end": 19, "original_key": "action"},
            "right_grip":  {"start": 19, "end": 20, "original_key": "action"},
            "left_joint":  {"start": 20, "end": 26, "original_key": "action"},
            "right_joint": {"start": 26, "end": 32, "original_key": "action"},
        },
        "video": {
            # RGB cameras: have MP4 files in videos/ — loadable by official GR00T loader.
            "head_rgb":      {"original_key": "observation.images.head_rgb"},
            "wrist_r_rgb":   {"original_key": "observation.images.wrist_r_rgb"},
            "wrist_l_rgb":   {"original_key": "observation.images.wrist_l_rgb"},
            # Depth cameras: dual-row lossless H.264 videos in videos/.
            # Frame shape (2H, W, 3) uint8; top H rows = lo byte, bottom H rows = hi byte.
            # Decoded by _decode_lossless_depth_frame() in the starVLA loader.
            "head_depth":    {"original_key": "observation.images.head_depth"},
            "wrist_r_depth": {"original_key": "observation.images.wrist_r_depth"},
            "wrist_l_depth": {"original_key": "observation.images.wrist_l_depth"},
        },
        # annotation: "original_key" is the parquet column the loader reads.
        # We use "task_index" (the standard LeRobot v2 column) so the official
        # loader can resolve task descriptions via tasks_map without needing the
        # separate annotation.human.action.task_description column.
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            },
        },
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    # ---- info.json -------------------------------------------------------
    total_frames = sum(ep["length"] for ep in episode_meta)

    features: dict = {
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
        "action":            {"dtype": "float32", "shape": [STATE_DIM]},
        "timestamp":         {"dtype": "float32", "shape": [1]},
        "episode_index":     {"dtype": "int64",   "shape": [1]},
        "frame_index":       {"dtype": "int64",   "shape": [1]},
        "index":             {"dtype": "int64",   "shape": [1]},
        "task_index":        {"dtype": "int64",   "shape": [1]},
        "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
    }
    # All 6 cameras (3 RGB + 3 depth) are stored as MP4 videos in videos/.
    # RGB  : libx264 yuv420p CRF 18 — visually lossless.
    # Depth: libx264rgb rgb24 CRF 0 all-intra — lossless dual-row uint8 encoding.
    #        Frame shape (2H, W, 3) uint8; decoded by _decode_lossless_depth_frame().
    rgb_video_feat = {
        "dtype": "video",
        "shape": [H, W, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": H,
            "video.width": W,
            "video.codec": "avc1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    depth_video_feat = {
        "dtype": "video",
        "shape": [2 * H, W, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": 2 * H,
            "video.width": W,
            "video.codec": "avc1",
            "video.pix_fmt": "rgb24",
            "video.is_depth_map": True,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
        "depth_format": "dual_row_uint8_mm",  # top H = lo byte, bottom H = hi byte
    }
    features["observation.images.head_rgb"]      = rgb_video_feat
    features["observation.images.wrist_r_rgb"]   = rgb_video_feat
    features["observation.images.wrist_l_rgb"]   = rgb_video_feat
    features["observation.images.head_depth"]    = depth_video_feat
    features["observation.images.wrist_r_depth"] = depth_video_feat
    features["observation.images.wrist_l_depth"] = depth_video_feat

    num_cameras = 6   # 3 RGB (yuv420p) + 3 depth (libx264rgb rgb24 lossless dual-row)
    total_chunks = max(1, (len(episode_meta) + 999) // 1000)
    info = {
        "codebase_version": "v2.2",
        "robot_type":       "yam_bimanual",
        "fps":              fps,
        "total_episodes":   len(episode_meta),
        "total_frames":     total_frames,
        "total_tasks":      len(text_registry._tasks),
        "total_chunks":     total_chunks,
        "total_videos":     len(episode_meta) * num_cameras,
        "chunks_size":      1000,
        "splits":           {"train": f"0:{len(episode_meta)}"},
        "data_path":   "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path":  "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "frames_path": "frames/{video_key}/episode_{episode_index:06d}/{frame_index:06d}",
        "features":    features,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # ---- stats.json  (32D q99/q01/mean/std/min/max) ---------------------
    import pandas as pd

    all_parquets = sorted((output_dir / "data").rglob("*.parquet"))
    all_states = []
    all_actions = []
    for pf in all_parquets:
        df = pd.read_parquet(pf)
        all_states.append(np.stack(df["observation.state"].values))
        all_actions.append(np.stack(df["action"].values))

    all_states = np.concatenate(all_states, axis=0).astype(np.float32)
    all_actions = np.concatenate(all_actions, axis=0).astype(np.float32)

    def _compute_stats(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(0).tolist(),
            "std":  arr.std(0).tolist(),
            "min":  arr.min(0).tolist(),
            "max":  arr.max(0).tolist(),
            "q01":  np.quantile(arr, 0.01, axis=0).tolist(),
            "q99":  np.quantile(arr, 0.99, axis=0).tolist(),
        }

    stats = {
        "observation.state": _compute_stats(all_states),
        "action":            _compute_stats(all_actions),
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    shutil.copy(meta_dir / "stats.json", meta_dir / "stats_gr00t.json")

    # ---- relative_stats.json  (cumulative deltas per action-horizon step) --
    # GR00T's LeRobotEpisodeLoader loads this file (if present) into
    # stats["relative_action"] for use with ActionRepresentation.RELATIVE.
    # Format: {joint_group: {stat: list[list[float]]}}  shape [horizon, dim]
    # We compute it from the already-loaded all_actions array.
    # action[t] = absolute EEF at t+1, so relative[t] = action[t] - action[t-1]
    # For a horizon H, cumulative relative actions are:
    #   rel_h[i] = action[i+h] - action[i]  for h in 1..H
    ACTION_HORIZON = 16
    modality_groups = {
        "left_pos":   (0, 3),
        "left_ori":   (3, 9),
        "right_pos":  (9, 12),
        "right_ori":  (12, 18),
        "left_grip":  (18, 19),
        "right_grip": (19, 20),
        "left_joint": (20, 26),
        "right_joint":(26, 32),
    }
    rel_stats: dict = {}
    for group_name, (s, e) in modality_groups.items():
        group_actions = all_actions[:, s:e]  # (N, dim)
        # For each horizon step h, collect all valid deltas action[i+h] - action[i]
        horizon_stats: dict[str, list] = {"mean": [], "std": [], "min": [], "max": [], "q01": [], "q99": []}
        for h in range(1, ACTION_HORIZON + 1):
            if len(group_actions) <= h:
                # Not enough frames; pad with zeros
                dim = e - s
                for k in horizon_stats:
                    horizon_stats[k].append([0.0] * dim)
                continue
            deltas = group_actions[h:] - group_actions[:-h]   # (N-h, dim)
            horizon_stats["mean"].append(deltas.mean(0).tolist())
            horizon_stats["std"].append(deltas.std(0).tolist())
            horizon_stats["min"].append(deltas.min(0).tolist())
            horizon_stats["max"].append(deltas.max(0).tolist())
            horizon_stats["q01"].append(np.quantile(deltas, 0.01, axis=0).tolist())
            horizon_stats["q99"].append(np.quantile(deltas, 0.99, axis=0).tolist())
        rel_stats[group_name] = horizon_stats

    with open(meta_dir / "relative_stats.json", "w") as f:
        json.dump(rel_stats, f, indent=2)

    print(f"Wrote metadata to {meta_dir}")
    print(f"  episodes:     {len(episode_meta)}")
    print(f"  total_frames: {total_frames}")


# ===========================================================================
# §8  Main entrypoint
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build a LeRobot v2.0 dataset from rd-convert processed YAM episodes."
    )
    parser.add_argument(
        "--processed-dir", required=True, type=Path,
        help=(
            "Root directory containing task subdirectories, each with episode "
            "subdirectories (e.g. .../processed/ with pick_lock/, place_box/ inside). "
            "Also supports pointing directly at a single task directory for "
            "backward compatibility (auto-detected)."
        ),
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Output directory for the LeRobot dataset.",
    )
    parser.add_argument("--fps", default=FPS, type=int, help=f"Frame rate (default {FPS}).")
    parser.add_argument(
        "--rewrite", action="store_true", default=False,
        help="Force reprocessing of all episodes, even if output already exists.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect: is processed_dir a single task dir or a root of task dirs?
    first_level = sorted(d for d in processed_dir.iterdir() if d.is_dir())
    if first_level and (first_level[0] / "metadata.json").exists():
        # Single-task mode: processed_dir IS the task dir (backward compat)
        task_dirs = [processed_dir]
    else:
        # Multi-task mode: processed_dir contains task subdirs
        task_dirs = first_level

    # Collect all episode dirs across tasks, ep_idx globally incremented
    all_episodes: List[Path] = []
    for task_dir in task_dirs:
        ep_dirs = sorted(
            d for d in task_dir.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        )
        all_episodes.extend(ep_dirs)

    if not all_episodes:
        raise RuntimeError(f"No episode directories found under {processed_dir}")
    print(f"Found {len(all_episodes)} episodes across {len(task_dirs)} task(s) in {processed_dir}")

    text_registry = TextRegistry()
    episode_meta: List[Dict] = []
    frames_per_chunk = 1000
    n_skipped = 0

    for ep_idx, ep_dir in enumerate(tqdm(all_episodes, desc="Processing episodes")):
        chunk_idx = ep_idx // frames_per_chunk
        ep_str = f"episode_{ep_idx:06d}"
        chunk_str = f"chunk-{chunk_idx:03d}"
        parquet_path = output_dir / "data" / chunk_str / f"{ep_str}.parquet"

        # Lightweight metadata read for task registration + skip check
        adapter = YAMAdapterV2(ep_dir)
        task_desc = adapter.task_description()
        text_registry.register(task_desc)

        if parquet_path.exists() and not args.rewrite:
            # Skip heavy processing, but still collect episode metadata
            n_skipped += 1
            episode_meta.append({
                "episode_index": ep_idx,
                "tasks": [task_desc],
                "length": adapter.num_frames,
            })
            continue

        try:
            meta = process_episode(
                episode_dir=ep_dir,
                episode_idx=ep_idx,
                chunk_idx=chunk_idx,
                output_dir=output_dir,
                text_registry=text_registry,
                fps=args.fps,
            )
            episode_meta.append(meta)
        except Exception as e:
            print(f"\n[WARN] Skipping episode {ep_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not episode_meta:
        raise RuntimeError("No episodes were successfully processed.")

    if n_skipped:
        print(f"Skipped {n_skipped} already-processed episodes (use --rewrite to force).")

    # Rewrite global frame indices
    _rewrite_global_indices(output_dir)

    # Always regenerate metadata + stats (covers incremental builds)
    first_adapter = YAMAdapterV2(all_episodes[0])
    generate_lerobot_metadata(
        output_dir=output_dir,
        episode_meta=episode_meta,
        text_registry=text_registry,
        first_adapter=first_adapter,
        fps=args.fps,
    )

    print(f"\nDataset written to: {output_dir}")
    print(f"  videos:       {output_dir / 'videos'}  (RGB yuv420p + depth libx264rgb lossless)")
    print(f"  frames:       {output_dir / 'frames'}  (RGB JPEGs for map-style loader only)")
    print("Next steps:")
    print("  1. Verify stats.json (non-zero EEF std confirms FK is working).")
    print("  2. Set --datasets.vla_data.data_root_dir to parent of output_dir.")
    print("  3. No frames_root needed — depth is now in videos/ alongside RGB.")


if __name__ == "__main__":
    main()
