"""Streaming recorder for the predicted observation rollout at deployment.

Mirrors the training-time eval-video layout 1:1
(``trainer._run_eval_and_log`` row captions at trainer.py:856-866). The saved
video stacks **six rows** top → bottom, each row is two 384×320 tiles
side-by-side:

  1. RGB             : pred | vae_recon
  2. DINO            : pred_pca | gt_pca
  3. Pointmap XYZ    : pred | vae_recon                      (build_pointmap_row_vae)
  4. Pointmap depth  : pred | vae_recon         (turbo)      (build_pointmap_depth_row_vae)
  5. Pointmap VAE-PCA: pred_pca | gt_pca
  6. RGB VAE-PCA     : pred_pca | gt_pca

The reference tile (vae_recon / gt_pca) for each chunk comes from the
chunk's own present input, frozen across the chunk's predicted frame count.

**PCA color stability**: the three PCA rows (2, 5, 6) would otherwise
fit a fresh PCA basis + per-frame min/max every chunk, so the same
feature maps to different colors across chunks. To fix this, the
recorder buffers the first ``pca_warmup_chunks`` chunks (default 3),
fits ONE PCA basis + ONE global min/max per modality over those chunks,
then renders every chunk (the buffered ones and all later ones) with
those fixed stats. Colors are then consistent across the whole session.
The cost is a small bounded buffer (~3 chunks) and the first few frames
are written once the warmup window closes.

The real-world observation stream at full action rate is recorded
*bridge-side* (``yam_raiden_bridge_ws`` ``record_observation_dir``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Panel width all rows are resized to before vertical stacking.
_PANEL_W = 640


# ----------------------------------------------------------------------
# Frame helpers
# ----------------------------------------------------------------------

def _to_even(frame: np.ndarray) -> np.ndarray:
    """libx264 requires even H/W. Edge-pad if needed."""
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _resize_row_to_width(row_np: np.ndarray, target_w: int) -> np.ndarray:
    """Resize ``[F, H, W, 3]`` uint8 to ``target_w`` along W, bilinear, keep aspect."""
    if row_np.shape[2] == target_w:
        return row_np
    src_h, src_w = row_np.shape[1:3]
    target_h = max(int(round(src_h * target_w / src_w)), 1)
    out = np.empty((row_np.shape[0], target_h, target_w, 3), dtype=row_np.dtype)
    for t in range(row_np.shape[0]):
        out[t] = np.array(
            Image.fromarray(row_np[t]).resize((target_w, target_h), Image.BILINEAR)
        )
    return out


def _pil_frames_to_tensor01(frames: List[Image.Image]) -> torch.Tensor:
    """List[PIL RGB] → ``[3, F, H, W]`` float32 in ``[0, 1]``."""
    arrs = [np.asarray(f.convert("RGB")) for f in frames]
    arr = np.stack(arrs, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()


def _expand_f(t: torch.Tensor, target_f: int) -> torch.Tensor:
    """Broadcast a length-1 frame axis (dim=1) to ``target_f``."""
    if t.shape[1] == target_f:
        return t
    if t.shape[1] != 1:
        idx = torch.from_numpy(
            np.clip(np.arange(target_f) * t.shape[1] // target_f, 0, t.shape[1] - 1)
        )
        return t.index_select(1, idx)
    return t.expand(-1, target_f, *([-1] * (t.ndim - 2))).contiguous()


def _row_2col_uint8(
    pred: torch.Tensor, vae_recon: torch.Tensor,
) -> np.ndarray:
    """Stitch ``[pred | vae_recon]`` along W per frame → ``[F, H, 2W, 3]`` uint8."""
    if pred.shape != vae_recon.shape:
        raise ValueError(
            f"_row_2col_uint8: shape mismatch  pred={tuple(pred.shape)}  "
            f"recon={tuple(vae_recon.shape)}"
        )
    stitched = torch.cat([pred, vae_recon], dim=-1).clamp(0.0, 1.0)
    arr = (stitched * 255.0).to(torch.uint8).cpu().numpy()
    return np.transpose(arr, (1, 2, 3, 0))


def _match_f(row_np: np.ndarray, target_f: int) -> np.ndarray:
    """Hold-last / subsample ``[F, H, W, 3]`` along F to match ``target_f``."""
    F = row_np.shape[0]
    if F == target_f:
        return row_np
    if F == 0:
        raise ValueError("_match_f: zero-frame row")
    indices = np.clip(np.arange(target_f) * F // target_f, 0, F - 1)
    return row_np[indices]


# ----------------------------------------------------------------------
# PCA stats (fixed over the warmup window)
# ----------------------------------------------------------------------

def _cpu(t: Optional[torch.Tensor]):
    if t is None:
        return None
    return t.detach().to("cpu")


def _flatten_feat(t: torch.Tensor) -> np.ndarray:
    """Flatten a latent/feature tensor to ``[M, C]`` numpy for PCA fitting.

    Accepts ``[B,C,F,N,1]`` (DINO tokens) or ``[B,C,F,H,W]`` (VAE latents),
    or their batch-dropped forms. ``M`` = F·N or F·H·W.
    """
    if t.ndim == 5:
        t = t[0]  # drop batch → [C,F,N,1] or [C,F,H,W]
    if t.ndim == 4 and t.shape[-1] == 1:  # [C,F,N,1]
        t = t[..., 0]                      # [C,F,N]
    if t.ndim == 3:                        # [C,F,N]
        C, F, N = t.shape
        return t.permute(1, 2, 0).reshape(F * N, C).float().cpu().numpy()
    C, F, H, W = t.shape                   # [C,F,H,W]
    return t.permute(1, 2, 3, 0).reshape(F * H * W, C).float().cpu().numpy()


def _fit_pca_stats(feat_list: List[np.ndarray]) -> Optional[dict]:
    """Fit a fixed PCA basis + global min/max from a list of ``[M_i, C]`` arrays.

    Returns ``{"basis": (mean[1,C], vt[3,C]), "range": (pca_min[3], pca_max[3],
    norm_min, norm_max)}`` — the format the patched ``visualize_vae_latent`` /
    ``_compute_dino_pca_and_norm`` accept via their ``pca_basis`` + ``value_range``
    args. Returns None if no features.
    """
    feat_list = [f for f in feat_list if f is not None and f.size > 0]
    if not feat_list:
        return None
    allf = np.concatenate(feat_list, axis=0).astype(np.float32)  # [sumM, C]
    mean = allf.mean(axis=0, keepdims=True)
    _, _, vt_full = np.linalg.svd(allf - mean, full_matrices=False)
    vt = vt_full[:3]
    proj = (allf - mean) @ vt.T            # [sumM, 3]
    pca_min = proj.min(axis=0)             # [3]
    pca_max = proj.max(axis=0)
    norm = np.sqrt(np.mean(allf * allf, axis=-1))
    return {
        "basis": (mean, vt),
        "range": (pca_min, pca_max, float(norm.min()), float(norm.max())),
    }


# ----------------------------------------------------------------------
# Recorder
# ----------------------------------------------------------------------

class PredictionRecorder:
    """Writes one ``prediction.mp4`` per session with color-stable PCA rows."""

    def __init__(
        self,
        session_dir: Path,
        fps: int = 8,
        frames_per_chunk: int = 0,
        pca_warmup_chunks: int = 3,
    ) -> None:
        self._session_dir = Path(session_dir)
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._mp4_path = self._session_dir / "prediction.mp4"
        self._writer = imageio.get_writer(
            str(self._mp4_path),
            fps=max(int(fps), 1),
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv444p",
        )
        self._closed = False
        self._frames_per_chunk = int(frames_per_chunk)
        self._warmup_chunks = max(1, int(pca_warmup_chunks))

        # Warmup state: buffer raw per-chunk data until we've seen
        # ``_warmup_chunks`` of them, then fit fixed PCA stats and render.
        self._warmup_buf: List[Dict[str, Any]] = []
        self._stats_ready = False
        self._dino_stats: Optional[dict] = None
        self._ptvae_stats: Optional[dict] = None
        self._rgbvae_stats: Optional[dict] = None
        self._model: Any = None  # captured for close()-time flush

        self._n_chunks = 0
        self._n_frames = 0
        logger.info(
            "[PredictionRecorder] opened %s  (fps=%d, frames_per_chunk=%s, "
            "pca_warmup_chunks=%d)",
            self._mp4_path, fps,
            self._frames_per_chunk if self._frames_per_chunk > 0 else "all",
            self._warmup_chunks,
        )

    @property
    def path(self) -> Path:
        return self._mp4_path

    @property
    def n_chunks(self) -> int:
        return self._n_chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def append_chunk(
        self,
        model,
        pred: Dict[str, Optional[torch.Tensor]],
        present: Dict[str, Any],
    ) -> None:
        if self._closed:
            raise RuntimeError("PredictionRecorder.append_chunk after close()")
        if pred.get("video_latents") is None:
            logger.warning(
                "[PredictionRecorder] video_latents=None (no-video regime); "
                "skipping chunk."
            )
            return

        self._model = model
        cd = self._stash(pred, present)

        if self._stats_ready:
            self._render_and_write(model, cd)
            return

        # Warmup: buffer until we can fit stable PCA stats, then drain.
        self._warmup_buf.append(cd)
        if len(self._warmup_buf) >= self._warmup_chunks:
            self._finalize_stats()
            for buffered in self._warmup_buf:
                self._render_and_write(model, buffered)
            self._warmup_buf.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Session ended before the warmup window filled — fit from whatever
        # we have so the short session still gets consistent colors.
        if not self._stats_ready and self._warmup_buf and self._model is not None:
            try:
                self._finalize_stats()
                for buffered in self._warmup_buf:
                    self._render_and_write(self._model, buffered)
            except Exception as e:  # noqa: BLE001
                logger.warning("[PredictionRecorder] warmup flush failed: %r", e)
            self._warmup_buf.clear()
        try:
            self._writer.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("[PredictionRecorder] writer close failed: %r", e)
        logger.info(
            "[PredictionRecorder] closed %s  chunks=%d  frames=%d",
            self._mp4_path, self._n_chunks, self._n_frames,
        )

    def __enter__(self) -> "PredictionRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Buffering + stats
    # ------------------------------------------------------------------

    @staticmethod
    def _stash(
        pred: Dict[str, Optional[torch.Tensor]],
        present: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Copy the raw tensors needed to render this chunk later (CPU)."""
        return {
            "video_lat": _cpu(pred.get("video_latents")),
            "dino_lat": _cpu(pred.get("dino_latents")),
            "pt_lat": _cpu(pred.get("pointmap_latents")),
            "input_image": _cpu(present.get("input_image")),
            "per_cam": {k: _cpu(v) for k, v in (present.get("per_cam") or {}).items()},
            "per_cam_depth": {
                k: _cpu(v) for k, v in (present.get("per_cam_depth") or {}).items()
            },
            "K": _cpu(present.get("camera_intrinsics")),
        }

    def _finalize_stats(self) -> None:
        """Fit fixed PCA stats per modality from the buffered warmup chunks."""
        dino_feats = [
            _flatten_feat(cd["dino_lat"]) for cd in self._warmup_buf
            if cd["dino_lat"] is not None
        ]
        pt_feats = [
            _flatten_feat(cd["pt_lat"]) for cd in self._warmup_buf
            if cd["pt_lat"] is not None
        ]
        rgb_feats = [
            _flatten_feat(cd["video_lat"]) for cd in self._warmup_buf
            if cd["video_lat"] is not None
        ]
        self._dino_stats = _fit_pca_stats(dino_feats)
        self._ptvae_stats = _fit_pca_stats(pt_feats)
        self._rgbvae_stats = _fit_pca_stats(rgb_feats)
        self._stats_ready = True
        logger.info(
            "[PredictionRecorder] PCA stats fixed from %d warmup chunk(s) "
            "(dino=%s ptvae=%s rgbvae=%s)",
            len(self._warmup_buf),
            self._dino_stats is not None,
            self._ptvae_stats is not None,
            self._rgbvae_stats is not None,
        )

    # ------------------------------------------------------------------
    # Render one chunk (all 6 rows) using the fixed stats
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _render_and_write(self, model, cd: Dict[str, Any]) -> None:
        device = model.device
        dtype = model.torch_dtype
        target_hw = tuple(getattr(getattr(model, "_layout", None), "composite_hw", (384, 320)))

        # === Row 1: RGB pred | vae_recon ===
        pred_video_lat = cd["video_lat"].to(device=device, dtype=dtype)
        pred_rgb = _pil_frames_to_tensor01(model._decode_latents(pred_video_lat, tiled=False))
        F_dec = pred_rgb.shape[1]

        input_image = cd["input_image"].to(device=device, dtype=dtype)
        rgb_lat_dev = model._encode_video_latents(input_image.unsqueeze(2), tiled=False)
        vae_recon_rgb_1f = _pil_frames_to_tensor01(
            model._decode_latents(rgb_lat_dev, tiled=False)[:1]
        )
        rgb_row = _row_2col_uint8(pred_rgb, _expand_f(vae_recon_rgb_1f, F_dec))
        rows: List[np.ndarray] = [rgb_row]

        # === Row 2: DINO PCA (fixed stats) ===
        if cd["dino_lat"] is not None and self._dino_stats is not None:
            try:
                rows.append(self._render_dino_row(model, cd, F_dec))
            except Exception as e:  # noqa: BLE001
                logger.warning("[PredictionRecorder] DINO row failed: %r", e)

        # === Rows 3, 4, 5: Pointmap ===
        if cd["pt_lat"] is not None:
            try:
                xyz_row, depth_row, ptvae_row = self._render_pointmap_rows(
                    model, cd, target_hw,
                )
                rows.append(xyz_row)
                rows.append(depth_row)
                if ptvae_row is not None:
                    rows.append(ptvae_row)
            except Exception as e:  # noqa: BLE001
                logger.warning("[PredictionRecorder] pointmap rows failed: %r", e)

        # === Row 6: RGB VAE-PCA (fixed stats) ===
        if self._rgbvae_stats is not None:
            try:
                gt_rgb_lat = rgb_lat_dev[0].detach().cpu()             # [C,1,H,W]
                F_lat = pred_video_lat.shape[2]
                gt_rgb_lat_exp = gt_rgb_lat.expand(-1, F_lat, -1, -1).contiguous()
                rows.append(self._render_vae_pca_row(
                    gt_rgb_lat_exp, pred_video_lat[0].detach().cpu(),
                    target_hw, self._rgbvae_stats,
                ))
            except Exception as e:  # noqa: BLE001
                logger.warning("[PredictionRecorder] RGB VAE-PCA row failed: %r", e)

        rows = [_resize_row_to_width(r, _PANEL_W) for r in rows]
        rows = [_match_f(r, F_dec) for r in rows]

        keep = self._frames_per_chunk if self._frames_per_chunk > 0 else F_dec
        keep = max(1, min(keep, F_dec))
        for k in range(keep):
            stacked = np.concatenate([row[k] for row in rows], axis=0)
            self._writer.append_data(_to_even(stacked))

        self._n_chunks += 1
        self._n_frames += keep

    # ------------------------------------------------------------------
    # PCA row renderers (call the patched leaf helpers with fixed stats)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _render_dino_row(self, model, cd: Dict[str, Any], target_f: int) -> np.ndarray:
        from flexpi.vis import (
            _arrange_layout,
            _compute_dino_pca_and_norm,
        )
        from flexpi.composite_layouts import get_layout

        pred_dino = cd["dino_lat"]               # [B,C,F,N,1]
        b, c, f_d, n, _ = pred_dino.shape

        per_cam_5d = {
            k: (v.unsqueeze(2) if v.ndim == 4 else v)
            for k, v in (cd.get("per_cam") or {}).items()
        }
        gt_dino_5d = model.dino_encoder.encode_video(
            video=None, per_cam={k: v.to(model.device) for k, v in per_cam_5d.items()},
            concat_mode="tshape_robotwin_384x320_uniform", **model._dino_encode_kwargs(),
            temporal_stride=model.dino_temporal_stride, first_frame_only=True,
        )
        gt_dino_1f = gt_dino_5d[0].cpu()         # [C,1,N,1]
        gt_features = gt_dino_1f.index_select(1, torch.zeros(f_d, dtype=torch.long))
        pred_features = pred_dino[0]             # [C,F,N,1]

        cam_patches = [tuple(p) for p in model.dino_cam_patches]
        basis = self._dino_stats["basis"]
        vrange = self._dino_stats["range"]
        gt_pca, _gt_norm, _ = _compute_dino_pca_and_norm(
            gt_features, cam_patches, pca_basis=basis, value_range=vrange,
        )
        pred_pca, _, _ = _compute_dino_pca_and_norm(
            pred_features, cam_patches, pca_basis=basis, value_range=vrange,
        )
        layout = get_layout("tshape_robotwin_384x320_uniform")
        gt_pca_t = _arrange_layout(gt_pca, is_rgb=True, layout=layout)
        pred_pca_t = _arrange_layout(pred_pca, is_rgb=True, layout=layout)
        return np.concatenate([pred_pca_t, gt_pca_t], axis=2)

    @torch.no_grad()
    def _render_pointmap_rows(
        self, model, cd: Dict[str, Any], target_hw: tuple,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        from flexpi.vis import (
            build_pointmap_depth_row_vae,
            build_pointmap_row_vae,
        )

        device = model.device
        dtype = model.torch_dtype

        K = cd.get("K")
        if K is None:
            raise RuntimeError("pointmap row needs camera_intrinsics in present obs")
        K_t = K.to(device=device, dtype=torch.float32)
        per_cam_depth = {k: v.to(device=device) for k, v in (cd.get("per_cam_depth") or {}).items()}

        pt_composite = model.pointmap_encoder.encode_composite(
            per_cam_depth=per_cam_depth, camera_intrinsics=K_t,
            concat_mode="tshape_robotwin_384x320_uniform", first_frame_only=True,
        )  # [1,3,1,384,320] in [-1,1]
        pt_lat_dev = model._encode_video_latents(pt_composite, tiled=False)
        vae_recon_pt_1f = _pil_frames_to_tensor01(
            model._decode_latents(pt_lat_dev, tiled=False)[:1]
        )

        pred_pt_lat = cd["pt_lat"].to(device=device, dtype=dtype)
        pred_pt = _pil_frames_to_tensor01(model._decode_latents(pred_pt_lat, tiled=False))
        F_pt = pred_pt.shape[1]
        vae_recon_pt = _expand_f(vae_recon_pt_1f, F_pt)

        # Rows 3, 4: XYZ + depth (no PCA — already color-stable).
        xyz_row = build_pointmap_row_vae(pred=pred_pt, vae_recon=vae_recon_pt)
        pt_min = model.pointmap_encoder.pt_min
        pt_max = model.pointmap_encoder.pt_max
        z_min = float(pt_min[0, 2, 0, 0].item())
        z_max = float(pt_max[0, 2, 0, 0].item())
        depth_vis_mode = getattr(model, "pointmap_depth_vis_mode", "turbo")
        depth_row = build_pointmap_depth_row_vae(
            pred=pred_pt, vae_recon=vae_recon_pt,
            z_min=z_min, z_max=z_max, vis_mode=depth_vis_mode,
        )

        # Row 5: pointmap VAE-PCA (fixed stats).
        ptvae_row = None
        if self._ptvae_stats is not None:
            gt_pt_lat = pt_lat_dev[0].detach().cpu()
            F_lat = pred_pt_lat.shape[2]
            gt_pt_lat_exp = gt_pt_lat.expand(-1, F_lat, -1, -1).contiguous()
            ptvae_row = self._render_vae_pca_row(
                gt_pt_lat_exp, pred_pt_lat[0].detach().cpu(),
                target_hw, self._ptvae_stats,
            )
        return xyz_row, depth_row, ptvae_row

    @staticmethod
    def _render_vae_pca_row(
        gt_latent: torch.Tensor,
        pred_latent: torch.Tensor,
        target_hw: tuple,
        stats: dict,
    ) -> np.ndarray:
        """2-col VAE-latent PCA row with the fixed (basis, range) stats."""
        from flexpi.vis import visualize_vae_latent

        th, tw = target_hw
        basis = stats["basis"]
        vrange = stats["range"]
        gt_pca, _gt_norm, _ = visualize_vae_latent(
            gt_latent, target_h=th, target_w=tw, pca_basis=basis, value_range=vrange,
        )
        pred_pca, _, _ = visualize_vae_latent(
            pred_latent, target_h=th, target_w=tw, pca_basis=basis, value_range=vrange,
        )
        return np.concatenate([pred_pca, gt_pca], axis=2)


# ----------------------------------------------------------------------
# Session-dir helper
# ----------------------------------------------------------------------

def make_session_dir(base_dir: str | os.PathLike, remote_addr: Optional[str] = None) -> Path:
    """Create a timestamped session subdir under ``base_dir`` and return its Path."""
    import re
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = ""
    if remote_addr:
        cleaned = re.sub(r"[^A-Za-z0-9.]+", "_", str(remote_addr)).strip("_")
        if cleaned:
            suffix = "_" + cleaned
    out = Path(base_dir) / f"{stamp}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    return out
