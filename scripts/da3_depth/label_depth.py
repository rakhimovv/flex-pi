"""Add DA3 metric depth to an RGB-only LeRobot dataset (standalone side-car pass).

Input:  a LeRobot v2.1 dataset in the canonical Flex-π layout with RGB videos
        and ``meta/camera_intrinsics.json``, but NO depth streams.
Output: the SAME dataset, plus
          videos/chunk-{c:03d}/observation.depth_ffv1.<cam>/episode_{i:06d}.mkv
          meta/info.json  →  observation.depth_ffv1.<cam> features registered
Nothing else is touched — parquet, RGB, episodes.jsonl and tasks.jsonl are
read-only here. Re-running is safe (see Resume below).

The result is byte-compatible with what scripts/yam_dataset_builder/convert.py
writes when the capture DOES have depth, so ``RobotVideoDataset`` consumes both
without knowing which is which.

WHY THIS IS STANDALONE
    Zero Flex-π imports and zero pyav: the DA3 module is defined inline below
    and all video IO goes through ffmpeg/ffprobe subprocesses. So this runs in
    its own small conda env holding only torch + depth_anything_3, with nothing
    of this repo on PYTHONPATH — which is why it can pin torch 2.10 / numpy<2
    without disturbing the ``flexpi`` env.

THE TWO THINGS THAT MUST BE RIGHT
  1. K CONVENTION. DA3METRIC's raw output is not metric; it is scaled by
     ``focal / 300`` with ``focal = (fx + fy) / 2``. K therefore only sets a
     GLOBAL SCALE on the depth — a wrong K produces a plausible-looking depth
     map that is silently the wrong size in metres, with no error anywhere.
     We rescale K from the resolution declared in camera_intrinsics.json to the
     depth grid with the ENDPOINT convention ``s = (dst-1)/(src-1)``, which is
     what ``RobotVideoDataset._load_layout_intrinsics_for_dir`` re-applies at
     load time and what the linspace-endpoint depth resampler on both sides
     assumes. The plain ratio ``dst/src`` is a different convention and
     disagrees by ~0.1-0.5%.
  2. ASPECT RATIO. Feeding DA3 a stretched frame corrupts its geometry. The
     depth grid therefore defaults to each camera's OWN native RGB resolution
     (probed per camera, so a mixed-resolution rig is handled), which also makes
     the K rescale an exact identity when the intrinsics are stated at that same
     resolution. ``--depth-hw H W`` overrides for all cams and warns if it
     changes the aspect ratio.

Camera names are DISCOVERED from videos/*/observation.images.<cam>/ intersected
with the intrinsics keys — nothing about the 3-cam canonical set is hardcoded.

RESUME / SCALE
  A camera is skipped when its output .mkv already exists (encode is atomic:
  temp file + rename, so a killed run never leaves a half-written .mkv that
  looks done). One bad episode is logged to meta/depth_failures_shard<N>.txt
  and does not kill the run. ``--num-shards N --shard-id i`` splits episodes
  interleaved across N processes (one per GPU); run ``--patch-info-only`` ONCE
  after all shards finish — concurrent info.json writers would race.

MEMORY
  Decode → DA3 → encode is streamed frame-batch by frame-batch through ffmpeg
  pipes, so peak RAM is ``batch_size`` frames regardless of episode length or
  resolution.

Usage:
    # 0. look before you leap — validates cams, K, resolutions; no GPU, no writes
    python scripts/da3_depth/label_depth.py --dataset-dir ./data/<ds> --check

    # 1. label
    python scripts/da3_depth/label_depth.py \
        --dataset-dir ./data/<ds> \
        --da3-weights /path/to/DA3METRIC-LARGE

    # sharded across 4 GPUs, then finalize
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/da3_depth/label_depth.py \
        --dataset-dir ./data/<ds> --da3-weights ... --num-shards 4 --shard-id $i & done; wait
    python scripts/da3_depth/label_depth.py --dataset-dir ./data/<ds> --patch-info-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# DA3 metric-depth module, defined here rather than imported so this file
# depends only on depth_anything_3. Metric formula is the official DA3 one
# (README FAQ + alignment.py): metric_depth_m = net_output * focal / 300 with
# focal = (fx+fy)/2.
# ─────────────────────────────────────────────────────────────────────────────
_METRIC_SCALE_FACTOR = 300.0


class DA3DepthModule(nn.Module):
    PATCH_SIZE = 14
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, weights_dir: str):
        super().__init__()
        from depth_anything_3.api import DepthAnything3
        self.da3 = DepthAnything3.from_pretrained(weights_dir)
        self.da3.eval()
        for p in self.da3.parameters():
            p.requires_grad = False
        name = self.da3.model_name.upper()
        self.is_da3metric = "DA3METRIC" in name and "NESTED" not in name
        self.is_nested = "NESTED" in name
        self.register_buffer("img_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def _pad_to_patch(self, x):
        _, _, H, W = x.shape
        ph = (self.PATCH_SIZE - H % self.PATCH_SIZE) % self.PATCH_SIZE
        pw = (self.PATCH_SIZE - W % self.PATCH_SIZE) % self.PATCH_SIZE
        if ph or pw:
            x = nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, H, W

    @torch.no_grad()
    def forward(self, images: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) in [0,1] + (B,3,3) K -> (B,1,H,W) metric depth in metres."""
        x = (images - self.img_mean) / self.img_std
        x, H, W = self._pad_to_patch(x)
        # unsqueeze(1) => (B, N=1, 3, H, W): each frame is its own single-view
        # scene. Unsqueezing at dim 0 instead would make DA3 fuse the B frames
        # as one multi-view scene and collapse per-frame variation.
        raw = self.da3.forward(x.unsqueeze(1), export_feat_layers=[])
        depth = raw["depth"][:, :, :H, :W]
        if self.is_da3metric:
            focal = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2.0
            depth = depth * (focal.view(-1, 1, 1, 1) / _METRIC_SCALE_FACTOR)
        return depth


# ─────────────────────────────────────────────────────────────────────────────
# Dataset layout (canonical Flex-π / LeRobot v2.1)
# ─────────────────────────────────────────────────────────────────────────────
RGB_PREFIX = "observation.images."
DEPTH_FEAT = "observation.depth_ffv1"      # the codec the trainer defaults to
DEPTH_EXT = ".mkv"


def _rgb_dir(ds: Path, cam: str, chunk: int) -> Path:
    return ds / "videos" / f"chunk-{chunk:03d}" / f"{RGB_PREFIX}{cam}"


def _rgb_path(ds: Path, cam: str, ep: int, chunks_size: int) -> Path:
    return _rgb_dir(ds, cam, ep // chunks_size) / f"episode_{ep:06d}.mp4"


def _depth_path(ds: Path, cam: str, ep: int, chunks_size: int) -> Path:
    return (ds / "videos" / f"chunk-{ep // chunks_size:03d}"
            / f"{DEPTH_FEAT}.{cam}" / f"episode_{ep:06d}{DEPTH_EXT}")


def discover_cams(ds: Path, intr: dict, want: list[str] | None) -> list[str]:
    """Camera names = RGB video dirs on disk ∩ camera_intrinsics.json keys."""
    on_disk = sorted({
        d.name[len(RGB_PREFIX):]
        for d in ds.glob(f"videos/chunk-*/{RGB_PREFIX}*") if d.is_dir()
    })
    if not on_disk:
        raise SystemExit(f"no {RGB_PREFIX}* video dirs under {ds / 'videos'}")
    if want:
        missing = [c for c in want if c not in on_disk]
        if missing:
            raise SystemExit(f"--cams {missing} not on disk; available: {on_disk}")
        on_disk = list(want)
    no_k = [c for c in on_disk if c not in intr]
    if no_k:
        # DA3METRIC cannot produce metric depth without a focal length. Refuse
        # rather than invent one — a guessed K is a silent global scale error.
        raise SystemExit(
            f"cameras {no_k} have RGB but no entry in meta/camera_intrinsics.json "
            f"(has: {sorted(intr)}). DA3METRIC needs fx/fy; add them or pass "
            f"--cams to restrict to the cameras you do have K for."
        )
    unused = sorted(set(intr) - set(on_disk))
    if unused:
        print(f"[info] intrinsics list cameras with no RGB videos (ignored): {unused}")
    return on_disk


def list_episodes(ds: Path, cams: list[str]) -> list[int]:
    """Episode indices, from meta/episodes.jsonl when present, else from disk."""
    jl = ds / "meta" / "episodes.jsonl"
    if jl.exists():
        eps = sorted(json.loads(l)["episode_index"] for l in jl.read_text().splitlines() if l.strip())
        if eps:
            return eps
    return sorted({
        int(p.stem.split("_")[-1])
        for cam in cams for p in ds.glob(f"videos/chunk-*/{RGB_PREFIX}{cam}/episode_*.mp4")
    })


# ─────────────────────────────────────────────────────────────────────────────
# ffmpeg / ffprobe IO
# ─────────────────────────────────────────────────────────────────────────────
def probe_hw(path: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        text=True,
    ).strip().split("\n")[0]
    w, h = out.split(",")[:2]
    return int(h), int(w)


def _stderr_tail(fh, n: int = 2000) -> str:
    fh.seek(0)
    return fh.read().decode(errors="replace")[-n:]


def stream_rgb_to_depth(
    rgb_mp4: Path, out_mkv: Path, fps: int, K: np.ndarray, grid: tuple[int, int],
    module: DA3DepthModule, batch: int, device: str,
    max_depth_mm: int | None,
) -> int:
    """Decode → resize to `grid` → DA3 → encode, streamed. Returns frames written.

    `grid` is the (h, w) that K was rescaled to; RGB is resized to it BEFORE the
    DA3 forward and the depth is written at it. These have to move together —
    DA3METRIC scales its output by the focal in K, so a K for one grid applied
    to RGB at another is a silent global scale error in the output depth.
    Identity when grid == the RGB resolution (the default).

    Peak RAM is `batch` frames: ffmpeg decodes into a pipe we drain batch by
    batch, and each batch's depth is pushed straight into the ffv1 encoder's
    stdin. Encode target is a temp file renamed on success, so a killed run
    never leaves a truncated .mkv that the resume check would accept.
    """
    H, W = probe_hw(rgb_mp4)
    h_dst, w_dst = grid
    frame_bytes = H * W * 3
    out_mkv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mkv.with_name(out_mkv.stem + ".tmp" + DEPTH_EXT)
    K_t = torch.from_numpy(K).to(device)

    dec_err = tempfile.TemporaryFile()
    enc_err = tempfile.TemporaryFile()
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(rgb_mp4),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=dec_err,
    )
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{w_dst}x{h_dst}",
         "-framerate", str(fps), "-i", "pipe:0",
         # -level 3 -g 1: lossless all-intra, matches encode_depth_video_frames
         # so random-access decode at training time stays cheap.
         "-c:v", "ffv1", "-level", "3", "-g", "1", str(tmp)],
        stdin=subprocess.PIPE, stderr=enc_err,
    )
    n_written = 0
    try:
        while True:
            buf = dec.stdout.read(frame_bytes * batch)
            if not buf:
                break
            b = len(buf) // frame_bytes
            if b == 0:
                break  # trailing partial frame: ffmpeg truncated, treated as EOF
            arr = np.frombuffer(buf[: b * frame_bytes], dtype=np.uint8).reshape(b, H, W, 3)
            x = torch.from_numpy(arr.copy()).to(device).permute(0, 3, 1, 2).float() / 255.0
            if (H, W) != (h_dst, w_dst):
                # bilinear + antialias is DA3's own preprocessing kernel; a
                # plain resize would hand it a differently-filtered image than
                # it was trained on.
                x = torch.nn.functional.interpolate(
                    x, size=(h_dst, w_dst), mode="bilinear",
                    align_corners=False, antialias=True)
            depth_m = module(x, K_t.unsqueeze(0).expand(b, 3, 3).contiguous())
            depth_mm = (depth_m[:, 0] * 1000.0).round()
            if max_depth_mm is not None:
                # Beyond the cap = invalid; 0 is the sentinel the converter uses.
                depth_mm = torch.where(depth_mm > max_depth_mm,
                                       torch.zeros_like(depth_mm), depth_mm)
            depth_mm = depth_mm.clamp(0, 65535).to(torch.uint16).cpu().numpy()
            enc.stdin.write(np.ascontiguousarray(depth_mm, dtype="<u2").tobytes())
            n_written += b
        dec.stdout.close()
        if dec.wait() != 0:
            raise RuntimeError(f"ffmpeg decode failed on {rgb_mp4}:\n{_stderr_tail(dec_err)}")
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError(f"ffv1 encode failed for {out_mkv}:\n{_stderr_tail(enc_err)}")
        if n_written == 0:
            raise RuntimeError(f"decoded 0 frames from {rgb_mp4}")
        tmp.rename(out_mkv)          # atomic publish
        return n_written
    except BaseException:
        for p in (dec, enc):
            if p.poll() is None:
                p.kill()
        tmp.unlink(missing_ok=True)
        raise
    finally:
        dec_err.close()
        enc_err.close()


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsics
# ─────────────────────────────────────────────────────────────────────────────
def K_at_grid(entry: dict, h_dst: int, w_dst: int) -> np.ndarray:
    """(3,3) K endpoint-rescaled from the declared resolution to (h_dst, w_dst).

    Endpoint convention s = (dst-1)/(src-1) — identical to the builder's K
    rescale and to ``RobotVideoDataset._load_layout_intrinsics_for_dir``, and it
    pairs with the linspace-endpoint nearest depth resampler used on both sides.
    Identity when the grid equals the declared resolution.
    """
    fx, fy = float(entry["fx"]), float(entry["fy"])
    cx, cy = float(entry["cx"]), float(entry["cy"])
    src_w, src_h = float(entry["width"]), float(entry["height"])
    sx = (w_dst - 1) / (src_w - 1) if src_w > 1 else 1.0
    sy = (h_dst - 1) / (src_h - 1) if src_h > 1 else 1.0
    return np.array([[fx * sx, 0.0, cx * sx],
                     [0.0, fy * sy, cy * sy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def plan_grids(
    ds: Path, cams: list[str], intr: dict, eps: list[int], chunks_size: int,
    override_hw: tuple[int, int] | None,
) -> dict[str, tuple[int, int]]:
    """Per-cam depth grid + the consistency warnings that matter for geometry."""
    grids: dict[str, tuple[int, int]] = {}
    for cam in cams:
        probe_src = next(
            (p for ep in eps
             for p in [_rgb_path(ds, cam, ep, chunks_size)] if p.exists()),
            None,
        )
        if probe_src is None:
            raise SystemExit(f"no RGB episode found on disk for cam {cam!r}")
        rgb_hw = probe_hw(probe_src)
        grid = override_hw or rgb_hw
        e = intr[cam]
        k_hw = (int(e["height"]), int(e["width"]))
        if rgb_hw != k_hw:
            print(f"[warn] {cam}: RGB is {rgb_hw[0]}x{rgb_hw[1]} but "
                  f"camera_intrinsics.json declares {k_hw[0]}x{k_hw[1]}; K is "
                  f"endpoint-rescaled to the depth grid, but check that the "
                  f"intrinsics really belong to this RGB stream.")
        if abs(grid[1] / grid[0] - k_hw[1] / k_hw[0]) > 1e-3:
            print(f"[warn] {cam}: depth grid {grid[0]}x{grid[1]} has a different "
                  f"aspect ratio than the intrinsics {k_hw[0]}x{k_hw[1]} — DA3 "
                  f"geometry degrades on stretched frames.")
        grids[cam] = grid
    return grids


# ─────────────────────────────────────────────────────────────────────────────
# meta/info.json
# ─────────────────────────────────────────────────────────────────────────────
def patch_info_json(ds: Path, cams: list[str], chunks_size: int, eps: list[int]) -> None:
    """Register observation.depth_ffv1.<cam> features, sized from the real files.

    Shapes are probed off the written .mkv rather than assumed, so this is
    correct after a sharded run and for a mixed-resolution rig. dtype is
    "depth_video" (not "video") so LeRobot's own video loader skips these and
    RobotVideoDataset picks them up via depth_encoder/depth_ext.
    """
    info_path = ds / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    fps = int(info.get("fps", 30))
    feats = info.setdefault("features", {})
    for cam in cams:
        present = [p for ep in eps
                   for p in [_depth_path(ds, cam, ep, chunks_size)] if p.exists()]
        if not present:
            print(f"[warn] no depth files for {cam}; feature not registered")
            continue
        # Sample rather than probe all N — one ffprobe per episode is slow on a
        # large dataset, and a grid mismatch comes from a mis-flagged re-run
        # (e.g. part of the dataset labeled with --depth-hw), which shows up in
        # any sample. The loader itself is robust to this (it resizes to
        # shape_meta.depth and rescales K off camera_intrinsics.json, never off
        # the on-disk grid), but the shape we declare here would be a lie for
        # some episodes — worth saying out loud.
        sample = present[:: max(1, len(present) // 8)][:8] + present[-1:]
        grids = {probe_hw(p) for p in sample}
        h, w = probe_hw(present[0])
        if len(grids) > 1:
            print(f"[warn] {cam}: depth files have MIXED grids {sorted(grids)}; "
                  f"declaring {h}x{w} (from episode {present[0].stem}). Relabel "
                  f"the odd ones with a consistent --depth-hw.")
        feats[f"{DEPTH_FEAT}.{cam}"] = {
            "dtype": "depth_video",
            "shape": [h, w, 1],
            "names": ["height", "width", "depth"],
            "info": {
                "video.height": h, "video.width": w,
                "video.codec": "ffv1", "video.pix_fmt": "gray16le",
                "video.is_depth_map": True, "video.fps": fps,
                "video.channels": 1, "has_audio": False,
                "depth_unit": "mm", "depth_dtype": "uint16",
                "depth_encoder": "ffv1", "depth_ext": DEPTH_EXT,
                "depth_packing": "gray16le",
            },
        }
        print(f"[info] registered {DEPTH_FEAT}.{cam}  shape=[{h}, {w}, 1]")
    info_path.write_text(json.dumps(info, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add DA3 metric depth to an RGB-only LeRobot dataset.")
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--da3-weights", type=Path,
                    help="DA3METRIC weights dir, e.g. .../DA3METRIC-LARGE. "
                         "Required unless --check / --patch-info-only.")
    ap.add_argument("--cams", nargs="*", default=None,
                    help="Restrict to these cameras. Default: every RGB stream on disk.")
    ap.add_argument("--depth-hw", nargs=2, type=int, metavar=("H", "W"), default=None,
                    help="Force one depth grid for all cams. Default: each cam's "
                         "own native RGB resolution (K rescale becomes exact).")
    ap.add_argument("--episodes", nargs="*", type=int, default=None,
                    help="Only these episode indices (smoke tests).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-depth-mm", type=int, default=None,
                    help="Depth above this becomes 0 (the invalid sentinel). "
                         "Default: no cap, just the uint16 ceiling.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-label episodes that already have depth files.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--patch-info-only", action="store_true",
                    help="Only register the depth features in meta/info.json. "
                         "Run ONCE after all shards finish.")
    ap.add_argument("--check", action="store_true",
                    help="Validate the dataset and print the plan; no GPU, no writes.")
    args = ap.parse_args()

    ds = args.dataset_dir.resolve()
    info_path = ds / "meta" / "info.json"
    intr_path = ds / "meta" / "camera_intrinsics.json"
    if not info_path.exists():
        raise SystemExit(f"not a LeRobot dataset (no meta/info.json): {ds}")
    if not intr_path.exists():
        raise SystemExit(
            f"missing {intr_path}. DA3METRIC needs fx/fy per camera to produce "
            f"metric depth; there is no safe default.")
    info = json.loads(info_path.read_text())
    intr = json.loads(intr_path.read_text())
    fps = int(info.get("fps", 30))
    chunks_size = int(info.get("chunks_size", 1000))

    cams = discover_cams(ds, intr, args.cams)
    all_eps = list_episodes(ds, cams)
    if args.episodes is not None:
        missing = sorted(set(args.episodes) - set(all_eps))
        if missing:
            raise SystemExit(f"requested episodes not present: {missing}")
        all_eps = sorted(set(args.episodes))
    if not all_eps:
        raise SystemExit("no episodes found")

    # A dataset that already declares depth is either (a) not depth-free at all
    # — real sensor depth we must not clobber — or (b) a previous run of THIS
    # script that patched info.json before some episodes failed, and is now being
    # resumed. Both are handled by the per-camera file-existence skip below, so
    # this is a loud warning rather than a hard stop: without --overwrite no
    # existing depth file is ever touched.
    already = [k for k in info.get("features", {}) if k.startswith("observation.depth")]
    if already and not (args.patch_info_only or args.check):
        print(f"[warn] meta/info.json already declares depth features: {already}. "
              f"Existing depth files will be "
              f"{'OVERWRITTEN (--overwrite)' if args.overwrite else 'left untouched'}.")

    if args.patch_info_only:
        patch_info_json(ds, cams, chunks_size, all_eps)
        return

    grids = plan_grids(ds, cams, intr, all_eps, chunks_size, tuple(args.depth_hw) if args.depth_hw else None)
    Ks = {c: K_at_grid(intr[c], *grids[c]) for c in cams}

    print(f"[plan] {ds}")
    print(f"[plan] {len(all_eps)} episodes, fps={fps}, chunks_size={chunks_size}")
    for c in cams:
        h, w = grids[c]
        K = Ks[c]
        print(f"[plan] {c:>18}: depth grid {h}x{w}  K@grid fx={K[0,0]:.2f} "
              f"fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}  "
              f"focal={(K[0,0]+K[1,1])/2:.2f}")
    if args.check:
        done = sum(1 for ep in all_eps for c in cams
                   if _depth_path(ds, c, ep, chunks_size).exists())
        print(f"[check] {done}/{len(all_eps) * len(cams)} (episode, cam) pairs "
              f"already labeled. OK — nothing written.")
        return

    if args.da3_weights is None:
        raise SystemExit("--da3-weights is required (unless --check / --patch-info-only)")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"--shard-id must be in [0, {args.num_shards})")
    # Interleaved so every shard gets a uniform mix of short and long episodes.
    ep_idxs = all_eps[args.shard_id::args.num_shards]
    tag = f"shard {args.shard_id}/{args.num_shards}"
    print(f"[{tag}] {len(ep_idxs)}/{len(all_eps)} episodes", flush=True)

    module = DA3DepthModule(str(args.da3_weights)).to(args.device).eval()
    print(f"[{tag}] DA3 loaded: is_da3metric={module.is_da3metric} "
          f"is_nested={module.is_nested}", flush=True)
    if not module.is_da3metric and not module.is_nested:
        print("[warn] weights are neither DA3METRIC nor DA3NESTED — output is "
              "relative depth, NOT metric millimetres.", flush=True)

    fail_log = ds / "meta" / f"depth_failures_shard{args.shard_id}.txt"
    done = skipped = failed = 0
    for n, ep in enumerate(ep_idxs):
        try:
            wrote_any = False
            for cam in cams:
                out = _depth_path(ds, cam, ep, chunks_size)
                if out.exists() and not args.overwrite:
                    continue
                src = _rgb_path(ds, cam, ep, chunks_size)
                if not src.exists():
                    raise FileNotFoundError(src)
                n_f = stream_rgb_to_depth(
                    src, out, fps, Ks[cam], grids[cam], module,
                    args.batch_size, args.device, args.max_depth_mm)
                wrote_any = True
                if n == 0:
                    print(f"[{tag}] ep{ep} {cam}: {n_f} frames -> {out.name}", flush=True)
            done += wrote_any
            skipped += (not wrote_any)
        except Exception as e:      # one bad episode must not kill a long run
            failed += 1
            with open(fail_log, "a") as fh:
                fh.write(f"ep{ep}\t{type(e).__name__}: {e}\n")
            print(f"[FAIL {tag}] ep{ep}: {type(e).__name__}: {e}", flush=True)
        if n % 50 == 0:
            print(f"[{tag}] {n}/{len(ep_idxs)} (done={done} skip={skipped} "
                  f"fail={failed})", flush=True)

    print(f"[{tag}] FINISHED: done={done} skipped={skipped} failed={failed}"
          f"{' — see ' + str(fail_log) if failed else ''}", flush=True)
    if args.num_shards == 1:
        patch_info_json(ds, cams, chunks_size, all_eps)
    else:
        print("[info] sharded run — after ALL shards, finalize once with "
              "--patch-info-only", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
