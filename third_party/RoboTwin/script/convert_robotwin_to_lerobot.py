"""Convert new-RoboTwin HDF5 data into a LeRobot v2.1 dataset.

Produces a directory matching the layout of flex-pi/data/robotwin2.0:
  {out_dir}/
    meta/{info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl, camera_intrinsics.json}
    data/chunk-000/episode_000000.parquet, ...
    videos/chunk-000/observation.images.cam_high/episode_000000.mp4, ...
    videos/chunk-000/observation.depth_ffv1.cam_high/episode_000000.mkv, ...   (optional)
    videos/chunk-000/observation.depth_x264rgb.cam_high/episode_000000.mp4, ...  (optional)

Example:
  python script/convert_robotwin_to_lerobot.py \
      --raw-dir data/lift_pot/demo_randomized \
      --out-dir /tmp/lift_pot_lerobot \
      --depth-codec both \
      --episodes 0 1 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# The flex-pi-vendored lerobot, for the depth codec helpers. Derived from this
# file's location (third_party/RoboTwin/script/) so the tree can live anywhere.
_FLEXPI_LEROBOT_ROOT = Path(__file__).resolve().parents[3] / "src/flexpi/datasets/lerobot"
if str(_FLEXPI_LEROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLEXPI_LEROBOT_ROOT))
from lerobot.datasets.video_utils import (  # noqa: E402
    DEPTH_VIDEO_EXT,
    encode_depth_video_frames,
)


# --- Constants matching the reference robotwin2.0 schema ------------------------------------

MOTOR_NAMES = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]

# hdf5_camera_name -> lerobot feature-key-suffix
CAMERA_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
LEROBOT_CAMERAS = list(CAMERA_MAP.values())

CHUNK_SIZE = 1000
RGB_CODEC = "libsvtav1"   # ffmpeg encoder flag (produces AV1 streams)
RGB_CODEC_CANONICAL = "av1"  # what pyav / info.json reports; matches reference
RGB_PIX_FMT = "yuv420p"

# Map encoder flag -> canonical codec name as reported by pyav's stream.codec.canonical_name.
# The reference robotwin2.0 info.json stores the canonical form, not the encoder flag.
_CODEC_CANONICAL = {
    "libsvtav1": "av1",
    "libaom-av1": "av1",
    "libx264": "h264",
    "libx264rgb": "h264",
    "libx265": "hevc",
    "ffv1": "ffv1",
}


# --- Small helpers ---------------------------------------------------------------------------

def _ffmpeg_encode_rgb(frames: np.ndarray, out_path: Path, fps: int) -> None:
    """Encode (N, H, W, 3) uint8 RGB frames to libsvtav1 mp4 via ffmpeg stdin."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")
    assert frames.dtype == np.uint8 and frames.ndim == 4 and frames.shape[-1] == 3
    N, H, W, _ = frames.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(fps),
        "-i", "-",
        "-c:v", RGB_CODEC, "-pix_fmt", RGB_PIX_FMT, "-g", "2", "-crf", "30",
        str(out_path),
    ]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(),
                          capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg RGB encode failed (rc={proc.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {proc.stderr.decode(errors='replace')[:4000]}"
        )


def _load_episode_hdf5(ep_path: Path) -> dict:
    """Read one episode.hdf5 and return everything we need as numpy arrays.

    Reference robotwin2.0 convention (verified by diffing its episode_000000
    parquet): action[t] == state[t+1] exactly, and action[N-1] == state[N-1].
    I.e., the action signal is the commanded next-step joint target; the terminal
    frame replicates the final state. We follow this convention so a policy
    trained on our data sees the same target-semantics as models trained on the
    reference.
    """
    out: dict = {}
    with h5py.File(ep_path, "r") as f:
        vec = np.asarray(f["joint_action/vector"][:], dtype=np.float32)
        assert vec.shape[1] == 14, f"expected (N,14) joint_action.vector, got {vec.shape}"
        out["state"] = vec
        # action[t] = state[t+1], with action[N-1] replicating state[N-1].
        action = np.empty_like(vec)
        action[:-1] = vec[1:]
        action[-1] = vec[-1]
        out["action"] = action

        out["rgb"] = {}
        out["depth"] = {}
        out["intrinsic"] = {}
        for hdf5_cam, lerobot_cam in CAMERA_MAP.items():
            rgb_jpeg = f[f"observation/{hdf5_cam}/rgb"][:]
            frames = []
            # RoboTwin's pkl2hdf5.py encodes with cv2.imencode(".jpg", arr) where `arr`
            # is SAPIEN's RGB output. cv2.imencode treats its input as BGR, so the
            # stored JPEG bytes carry swapped luma/chroma math — but cv2.imdecode
            # swaps BACK on read, so the numpy array we get from cv2.imdecode is
            # already in the true RGB order. Applying an extra BGR2RGB here would
            # SWAP red and blue channels (verified empirically with a pure-red
            # synth test). Treat cv2.imdecode output as RGB as-is.
            for jpeg_bytes in rgb_jpeg:
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                rgb = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                frames.append(rgb)
            out["rgb"][lerobot_cam] = np.stack(frames, axis=0)  # (N, H, W, 3)

            depth = np.asarray(f[f"observation/{hdf5_cam}/depth"][:], dtype=np.uint16)
            out["depth"][lerobot_cam] = depth  # (N, H, W)

            # intrinsic_cv is (N, 3, 3); take frame 0 (verified constant per episode)
            K = np.asarray(f[f"observation/{hdf5_cam}/intrinsic_cv"][0], dtype=np.float32)
            out["intrinsic"][lerobot_cam] = K
    return out


def _build_tasks_registry(
    raw_dirs: list[Path], episode_ids_per_dir: list[list[int]]
) -> tuple[dict[str, int], list[dict[int, list[str]]]]:
    """Scan instructions/*.json across all raw_dirs, build a single global
    task -> task_index map. Returns that map plus a per-raw-dir list of
    {source_ep_idx: seen_list} dicts."""
    task_to_index: dict[str, int] = {}
    per_dir_seen: list[dict[int, list[str]]] = []
    for raw_dir, episode_ids in zip(raw_dirs, episode_ids_per_dir):
        per_ep: dict[int, list[str]] = {}
        for ep_idx in episode_ids:
            inst_path = raw_dir / "instructions" / f"episode{ep_idx}.json"
            if not inst_path.exists():
                raise FileNotFoundError(f"Missing instructions file: {inst_path}")
            with inst_path.open() as fh:
                inst = json.load(fh)
            seen = list(inst.get("seen", []))
            if not seen:
                raise RuntimeError(f"No 'seen' instructions in {inst_path}")
            per_ep[ep_idx] = seen
            for s in seen:
                if s not in task_to_index:
                    task_to_index[s] = len(task_to_index)
        per_dir_seen.append(per_ep)
    return task_to_index, per_dir_seen


# --- Per-episode worker (callable from ProcessPoolExecutor) ---------------------------------

@dataclass
class EpisodeJob:
    new_ep_idx: int
    src_ep_idx: int
    ep_path: str          # str for pickling
    task_name: str
    seen: list            # list[str]
    ep_task_indices: list # list[int] (indexes into global tasks.jsonl)
    frame_start: int
    out_dir: str
    depth_codec: str
    fps: int


def _encode_episode(job: EpisodeJob) -> dict:
    """Do all the filesystem work for one episode. Returns metadata dict.

    Runs in a worker process when parallelized. Writes:
      - data/chunk-XXX/episode_{N}.parquet
      - videos/.../episode_{N}.mp4 (rgb) + .mkv/.mp4 (depth)
    Returns:
      {
        "new_ep_idx": int,
        "src_ep_idx": int,
        "N": int,                   # frame count
        "seen": [str, ...],
        "state_stats": {...},
        "action_stats": {...},
        "H": int, "W": int,
        "intrinsic": {cam: 3x3-list},  # for camera_intrinsics.json
      }
    """
    out_dir = Path(job.out_dir)
    data = _load_episode_hdf5(Path(job.ep_path))
    N = data["state"].shape[0]
    H, W = data["depth"][LEROBOT_CAMERAS[0]].shape[1:]

    # Frame-level task_index: random sample from this episode's eligible indices.
    rng = np.random.default_rng(seed=job.new_ep_idx)
    ep_task_idx_arr = np.asarray(job.ep_task_indices, dtype=np.int64)
    per_frame_task_idx = rng.choice(ep_task_idx_arr, size=N)

    chunk_idx = job.new_ep_idx // CHUNK_SIZE
    parquet_path = out_dir / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{job.new_ep_idx:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame_index = np.arange(N, dtype=np.int64)
    timestamp = (frame_index / float(job.fps)).astype(np.float32)
    global_index = np.arange(job.frame_start, job.frame_start + N, dtype=np.int64)

    table = pa.table({
        "observation.state": pa.array(list(data["state"]), type=pa.list_(pa.float32(), 14)),
        "action":            pa.array(list(data["action"]), type=pa.list_(pa.float32(), 14)),
        "timestamp":      pa.array(timestamp, type=pa.float32()),
        "frame_index":    pa.array(frame_index, type=pa.int64()),
        "episode_index":  pa.array(np.full((N,), job.new_ep_idx, dtype=np.int64), type=pa.int64()),
        "index":          pa.array(global_index, type=pa.int64()),
        "task_index":     pa.array(per_frame_task_idx, type=pa.int64()),
    })
    pq.write_table(table, parquet_path)

    for cam in LEROBOT_CAMERAS:
        rgb_path = (out_dir / "videos" / f"chunk-{chunk_idx:03d}"
                    / f"observation.images.{cam}" / f"episode_{job.new_ep_idx:06d}.mp4")
        _ffmpeg_encode_rgb(data["rgb"][cam], rgb_path, fps=job.fps)

    codecs_to_emit = []
    if job.depth_codec in ("ffv1", "both"):
        codecs_to_emit.append(("ffv1", "depth_ffv1"))
    if job.depth_codec in ("x264rgb", "both"):
        codecs_to_emit.append(("libx264rgb", "depth_x264rgb"))
    for vcodec, feat_suffix in codecs_to_emit:
        for cam in LEROBOT_CAMERAS:
            ext = DEPTH_VIDEO_EXT[vcodec]
            depth_path = (out_dir / "videos" / f"chunk-{chunk_idx:03d}"
                          / f"observation.{feat_suffix}.{cam}" / f"episode_{job.new_ep_idx:06d}{ext}")
            encode_depth_video_frames(data["depth"][cam], depth_path, fps=job.fps,
                                      vcodec=vcodec, overwrite=True)

    def _stats(x: np.ndarray) -> dict:
        return {
            "min":   x.min(axis=0).astype(float).tolist(),
            "max":   x.max(axis=0).astype(float).tolist(),
            "mean":  x.mean(axis=0).astype(float).tolist(),
            "std":   x.std(axis=0).astype(float).tolist(),
            "count": [int(N)],
        }

    return {
        "new_ep_idx":   job.new_ep_idx,
        "src_ep_idx":   job.src_ep_idx,
        "task_name":    job.task_name,
        "N":            int(N),
        "seen":         job.seen,
        "state_stats":  _stats(data["state"]),
        "action_stats": _stats(data["action"]),
        "H":            int(H), "W": int(W),
        "intrinsic":    {cam: data["intrinsic"][cam].tolist() for cam in data["intrinsic"]},
    }


# --- Main conversion -------------------------------------------------------------------------

def convert(
    raw_dirs: list[Path] | Path,
    out_dir: Path,
    depth_codec: str,
    fps: int,
    episodes: list[int] | None,
    task_names: list[str] | str,
    overwrite: bool,
    workers: int = 1,
) -> None:
    # Normalize inputs to lists for uniform handling
    if isinstance(raw_dirs, Path):
        raw_dirs = [raw_dirs]
    if isinstance(task_names, str):
        task_names = [task_names] if task_names else [""] * len(raw_dirs)
    if len(task_names) != len(raw_dirs):
        raise ValueError(f"got {len(raw_dirs)} raw_dirs but {len(task_names)} task_names")

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass --overwrite to replace")
        shutil.rmtree(out_dir)
    (out_dir / "meta").mkdir(parents=True)

    # Discover episodes per raw_dir (paths + ids)
    per_dir_ep_paths: list[list[Path]] = []
    per_dir_ep_ids: list[list[int]] = []
    for raw_dir in raw_dirs:
        all_hdf5 = sorted(glob.glob(str(raw_dir / "data" / "episode*.hdf5")),
                          key=lambda p: int(Path(p).stem.replace("episode", "")))
        if not all_hdf5:
            raise FileNotFoundError(f"No episode HDF5 files under {raw_dir / 'data'}")
        ep_ids_available = [int(Path(p).stem.replace("episode", "")) for p in all_hdf5]
        if episodes is not None and len(raw_dirs) == 1:
            missing = [e for e in episodes if e not in ep_ids_available]
            if missing:
                raise ValueError(f"Requested episodes not found: {missing}")
            chosen = sorted(episodes)
            paths = [Path(all_hdf5[ep_ids_available.index(e)]) for e in chosen]
        else:
            chosen = ep_ids_available
            paths = [Path(p) for p in all_hdf5]
        per_dir_ep_ids.append(chosen)
        per_dir_ep_paths.append(paths)
    total_episodes = sum(len(ids) for ids in per_dir_ep_ids)

    # Global task registry (dedup across all raw_dirs)
    task_to_index, per_dir_seen = _build_tasks_registry(raw_dirs, per_dir_ep_ids)
    print(f"[tasks] Registered {len(task_to_index)} unique instructions "
          f"across {total_episodes} episodes from {len(raw_dirs)} source dirs")

    # Pre-scan: frame counts per episode (for global_index offsets without loading full frames).
    flat_eps: list[tuple[Path, int, Path, str, list]] = []  # (raw_dir, src_ep_idx, ep_path, task_name, seen)
    frame_counts: list[int] = []
    scan_pbar = tqdm(total=total_episodes, desc="scan frames", unit="ep")
    for raw_dir, task_name, ep_ids, ep_paths, ep_seen_map in zip(
            raw_dirs, task_names, per_dir_ep_ids, per_dir_ep_paths, per_dir_seen):
        for src_ep_idx, ep_path in zip(ep_ids, ep_paths):
            with h5py.File(ep_path, "r") as h:
                N = h["joint_action/vector"].shape[0]
            flat_eps.append((raw_dir, src_ep_idx, ep_path, task_name, ep_seen_map[src_ep_idx]))
            frame_counts.append(int(N))
            scan_pbar.update(1)
    scan_pbar.close()
    total_frames = sum(frame_counts)

    # Build per-episode jobs
    jobs: list[EpisodeJob] = []
    running = 0
    for new_ep_idx, ((raw_dir, src_ep_idx, ep_path, task_name, seen), N) in enumerate(
            zip(flat_eps, frame_counts)):
        ep_task_indices = [task_to_index[s] for s in seen]
        jobs.append(EpisodeJob(
            new_ep_idx=new_ep_idx,
            src_ep_idx=src_ep_idx,
            ep_path=str(ep_path),
            task_name=task_name,
            seen=list(seen),
            ep_task_indices=ep_task_indices,
            frame_start=running,
            out_dir=str(out_dir),
            depth_codec=depth_codec,
            fps=fps,
        ))
        running += N
    assert running == total_frames

    # Dispatch: serial or parallel
    print(f"[encode] {len(jobs)} episodes, {total_frames} frames, workers={workers}")
    results: list[dict] = []
    bar_kwargs = dict(total=len(jobs), desc=f"encode (w={workers})", unit="ep", dynamic_ncols=True)
    if workers <= 1:
        for j in tqdm(jobs, **bar_kwargs):
            r = _encode_episode(j)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex, tqdm(**bar_kwargs) as pbar:
            futures = {ex.submit(_encode_episode, j): j for j in jobs}
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    tqdm.write(f"[FAIL ep {j.new_ep_idx:06d}] {j.task_name} {Path(j.ep_path).name}: {e}")
                    raise
                results.append(r)
                pbar.update(1)
                pbar.set_postfix_str(f"ep {j.new_ep_idx:06d} {j.task_name}", refresh=False)
    # Sort back to new_ep_idx order for deterministic meta files
    results.sort(key=lambda r: r["new_ep_idx"])

    # Collect for meta writing
    episodes_records = [
        {"episode_index": r["new_ep_idx"], "tasks": r["seen"], "length": r["N"]}
        for r in results
    ]
    episodes_stats_records = [
        {"episode_index": r["new_ep_idx"],
         "stats": {"observation.state": r["state_stats"],
                   "action":            r["action_stats"]}}
        for r in results
    ]
    # Camera intrinsics: take from the first episode (all episodes share these in practice).
    first_r = results[0]
    first_ep_HW = (first_r["H"], first_r["W"])
    first_ep_intrinsics = {cam: np.asarray(arr) for cam, arr in first_r["intrinsic"].items()}

    # --- meta files ---------------------------------------------------------------------------
    tasks_path = out_dir / "meta" / "tasks.jsonl"
    with tasks_path.open("w") as fh:
        for task_str, idx in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            fh.write(json.dumps({"task_index": idx, "task": task_str}) + "\n")

    with (out_dir / "meta" / "episodes.jsonl").open("w") as fh:
        for rec in episodes_records:
            fh.write(json.dumps(rec) + "\n")

    with (out_dir / "meta" / "episodes_stats.jsonl").open("w") as fh:
        for rec in episodes_stats_records:
            fh.write(json.dumps(rec) + "\n")

    # camera_intrinsics.json
    H, W = first_ep_HW
    cam_intrinsics = {}
    for cam, K in first_ep_intrinsics.items():
        cam_intrinsics[cam] = {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "width": int(W), "height": int(H),
        }
    with (out_dir / "meta" / "camera_intrinsics.json").open("w") as fh:
        json.dump(cam_intrinsics, fh, indent=2)

    # info.json
    features: dict = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": [MOTOR_NAMES]},
        "action":            {"dtype": "float32", "shape": [14], "names": [MOTOR_NAMES]},
        "timestamp":     {"dtype": "float32", "shape": [1]},
        "frame_index":   {"dtype": "int64",   "shape": [1]},
        "episode_index": {"dtype": "int64",   "shape": [1]},
        "index":         {"dtype": "int64",   "shape": [1]},
        "task_index":    {"dtype": "int64",   "shape": [1]},
    }
    for cam in LEROBOT_CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [H, W, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": H, "video.width": W,
                "video.codec": RGB_CODEC_CANONICAL, "video.pix_fmt": RGB_PIX_FMT,
                "video.is_depth_map": False, "video.fps": fps,
                "video.channels": 3, "has_audio": False,
            },
        }
    for vcodec, feat_suffix in [("ffv1", "depth_ffv1"), ("libx264rgb", "depth_x264rgb")]:
        if (vcodec == "ffv1" and depth_codec in ("ffv1", "both")) or \
           (vcodec == "libx264rgb" and depth_codec in ("x264rgb", "both")):
            for cam in LEROBOT_CAMERAS:
                pix_fmt = "gray16le" if vcodec == "ffv1" else "rgb24"
                ch = 1 if vcodec == "ffv1" else 3
                ext = DEPTH_VIDEO_EXT[vcodec]  # ".mkv" for ffv1, ".mp4" for libx264rgb
                features[f"observation.{feat_suffix}.{cam}"] = {
                    # dtype = "depth_video" (not "video") so LeRobotDatasetMetadata.video_keys
                    # does NOT pick these up — the built-in loader would otherwise try to
                    # decode them via the single-template video_path (which hardcodes .mp4)
                    # and fail on our FFV1 .mkv files. Our own RobotVideoDataset reads
                    # the `depth_encoder` + `depth_ext` fields below to construct the right
                    # path and pick the right codec.
                    "dtype": "depth_video",
                    "shape": [H, W, ch],
                    "names": ["height", "width", "depth" if ch == 1 else "rgb_packed_u16"],
                    "info": {
                        "video.height": H, "video.width": W,
                        "video.codec": _CODEC_CANONICAL[vcodec], "video.pix_fmt": pix_fmt,
                        "video.is_depth_map": True, "video.fps": fps,
                        "video.channels": ch, "has_audio": False,
                        "depth_unit": "mm", "depth_dtype": "uint16",
                        "depth_encoder": vcodec,
                        "depth_ext": ext,
                        "depth_packing": "rgb_hi_lo_zero" if vcodec == "libx264rgb" else "gray16le",
                    },
                }

    # Only RGB features use the standard `.mp4` video_path template; depth features
    # store their extension individually in features[k]["info"]["depth_ext"].
    info = {
        "codebase_version": "v2.1",
        "robot_type": "aloha",  # match reference label
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "total_videos": total_episodes * sum(
            1 for k in features if k.startswith("observation.") and features[k]["dtype"] == "video"
        ),
        "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with (out_dir / "meta" / "info.json").open("w") as fh:
        json.dump(info, fh, indent=4)

    print(f"\n[done] wrote {total_episodes} episodes, {total_frames} frames to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw-dir", type=Path,
                     help="Path to one {RoboTwin/data/{task}/{config}} (contains data/, instructions/)")
    src.add_argument("--raw-dirs", type=Path, nargs="+",
                     help="Multiple raw dirs to merge into ONE combined LeRobot dataset. "
                          "Episodes and task_index numbering are global across all dirs.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--task-name", type=str, default="",
                    help="(single --raw-dir) human-readable task name")
    ap.add_argument("--task-names", type=str, nargs="+", default=None,
                    help="(multi --raw-dirs) one task name per raw dir, same order")
    ap.add_argument("--depth-codec", choices=["ffv1", "x264rgb", "both", "none"], default="both")
    ap.add_argument("--fps", type=int, default=50,
                    help=(
                        "Declared dataset fps. Default 50 to match the reference robotwin2.0 "
                        "dataset and the existing convert_aloha_data_to_lerobot_robotwin.py. "
                        "NOTE: native RoboTwin collection rate is 250/save_freq = 250/15 ≈ 16.67 Hz. "
                        "Setting fps=50 makes per-frame timestamps (frame_index/50) compress into "
                        "~1/3 of real time, but downstream models that key off frame_index are "
                        "unaffected. Pass --fps 17 for honest timestamps."
                    ))
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="Optional subset of source episode ids (default: all). "
                         "Only honored when --raw-dir (singular) is used.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel episode workers. Each worker runs its own ffmpeg "
                         "sequence (9 encodes per episode). ffmpeg itself auto-threads "
                         "across cores, so workers=4 on an 8-core box is often a sweet "
                         "spot; higher values may oversubscribe CPU. Default: 1 (serial).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.raw_dir is not None:
        raw_dirs = [args.raw_dir]
        task_names = [args.task_name]
    else:
        raw_dirs = list(args.raw_dirs)
        if args.task_names is None:
            task_names = [p.parent.name for p in raw_dirs]
        else:
            if len(args.task_names) != len(raw_dirs):
                ap.error(f"--task-names has {len(args.task_names)} entries "
                         f"but --raw-dirs has {len(raw_dirs)}")
            task_names = list(args.task_names)
        if args.episodes is not None:
            ap.error("--episodes is only supported with --raw-dir (single source)")

    convert(
        raw_dirs=raw_dirs,
        out_dir=args.out_dir,
        depth_codec=args.depth_codec,
        fps=args.fps,
        episodes=args.episodes,
        task_names=task_names,
        overwrite=args.overwrite,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
