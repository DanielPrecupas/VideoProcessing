# src/combine.py
"""
Frame combiner / compositor (NO post-fx).

- subject frames (required)
- optional background frames (bg plate)
- optional overlay frames (must have alpha), pose-attached (generic: wings/hats/swords/etc)

Now supports MULTIPLE overlays.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Tuple

import cv2
import numpy as np
import mediapipe as mp

from src.compose_utils import alpha_composite, rotate_and_scale_rgba, fit_image
from src.rig import PoseRig, ATTACH_POINTS


@dataclass
class OverlayConfig:
    frames_dir: str
    layer: Literal["behind", "front"] = "behind"

    subject_on: int = 0
    subject_off: Optional[int] = None

    attach: str = "chest"

    width_multiplier: float = 1.5
    offset_x: int = 0
    offset_y: int = 0

    pivot_x: float = 0.5
    pivot_y: float = 0.5

    rotation: Literal["none", "shoulders"] = "none"
    loop: bool = True


@dataclass
class CombineConfig:
    subject_dir: str
    output_dir: str

    bg_dir: Optional[str] = None

    # Back-compat (old single overlay)
    overlay: Optional[OverlayConfig] = None

    # New multi-overlay
    overlays: List[OverlayConfig] = field(default_factory=list)

    bg_fit: Literal["stretch", "cover", "contain"] = "cover"

    use_segmentation: bool = True
    seg_blur_sigma: float = 2.5

    smoothing: float = 0.6
    reuse_last_on_miss: bool = True


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def read_bgra_required(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        raise RuntimeError(f"Overlay frame has no alpha (expected BGRA PNG): {path}")
    return img


def soft_subject_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    return np.clip(mask, 0.0, 1.0)[..., None]


def _apply_overlay(out_bgr: np.ndarray, subject_bgr: np.ndarray, idx: int, ov_cfg: OverlayConfig, ov_files: List[str], rig: PoseRig, smoothing: float, reuse_last_on_miss: bool) -> np.ndarray:
    if not ov_files or ov_cfg.attach == "none":
        return out_bgr
    
    on = int(ov_cfg.subject_on or 0)
    off = ov_cfg.subject_off
    if idx < on:
        return out_bgr
    if off is not None and idx >= int(off):
        return out_bgr

    # ---- overlay frame selection (loop vs play-once) ----
    local_i = idx - on
    if local_i < 0:
        return out_bgr
    if ov_cfg.loop:
        frame_i = local_i % len(ov_files)
    else:
        if local_i >= len(ov_files):
            return out_bgr  # finished playing -> disappear
        frame_i = local_i

    rot_mode = ov_cfg.rotation if ov_cfg.rotation in ("none", "shoulders") else "none"
    rig_state = rig.estimate(
        frame_bgr=subject_bgr,
        attach=ov_cfg.attach,
        rotation=("shoulders" if rot_mode == "shoulders" else "none"),
        smoothing=smoothing,
        reuse_last_on_miss=reuse_last_on_miss,
    )
    if rig_state is None:
        return out_bgr

    ov = read_bgra_required(ov_files[frame_i])
    target_w = rig_state.ref_dist * float(ov_cfg.width_multiplier)
    angle = rig_state.angle_deg if ov_cfg.rotation == "shoulders" else 0.0
    ov = rotate_and_scale_rgba(ov, angle, target_w)

    px = float(getattr(ov_cfg, "pivot_x", 0.5))
    py = float(getattr(ov_cfg, "pivot_y", 0.5))
    x = int(round(rig_state.attach_xy[0] - ov.shape[1] * px + ov_cfg.offset_x))
    y = int(round(rig_state.attach_xy[1] - ov.shape[0] * py + ov_cfg.offset_y))
    return alpha_composite(out_bgr, ov, x, y)


def combine_frames(cfg: CombineConfig) -> None:
    ensure_dir(cfg.output_dir)

    subj_files = list_images(cfg.subject_dir)
    if not subj_files:
        raise RuntimeError(f"No subject frames found in: {cfg.subject_dir}")

    bg_files = list_images(cfg.bg_dir) if cfg.bg_dir else []

    # Build overlay list (new) with back-compat for old cfg.overlay
    overlays: List[OverlayConfig] = []
    if cfg.overlays:
        overlays.extend(cfg.overlays)
    if cfg.overlay is not None:
        overlays.append(cfg.overlay)

    # Validate attach names
    for ov in overlays:
        if ov.attach not in ATTACH_POINTS:
            raise ValueError(f"overlay.attach='{ov.attach}' not supported. Use one of: {ATTACH_POINTS}")

    # Load overlay file lists once
    ov_file_lists: List[List[str]] = [list_images(ov.frames_dir) for ov in overlays]

    # IMPORTANT: one rig per overlay (so smoothing doesn't mix attach points)
    rigs: List[PoseRig] = [PoseRig() for _ in overlays]

    # segmentation
    seg = None
    if cfg.use_segmentation:
        mp_selfie = mp.solutions.selfie_segmentation
        seg = mp_selfie.SelfieSegmentation(model_selection=1)

    try:
        for idx, subj_path in enumerate(subj_files):
            subject = cv2.imread(subj_path, cv2.IMREAD_COLOR)
            if subject is None:
                continue

            h, w = subject.shape[:2]

            # ---- Base plate ----
            if bg_files:
                bg = cv2.imread(bg_files[idx % len(bg_files)], cv2.IMREAD_COLOR)
                out = fit_image(bg, w, h, cfg.bg_fit) if bg is not None else subject.copy()
            else:
                out = subject.copy()

            # ---- All overlays behind ----
            for ov_cfg, ov_files, rig in zip(overlays, ov_file_lists, rigs):
                if ov_cfg.layer == "behind":
                    out = _apply_overlay(out, subject, idx, ov_cfg, ov_files, rig, cfg.smoothing, cfg.reuse_last_on_miss)

            # ---- Subject occlusion ----
            if seg is not None:
                rgb = cv2.cvtColor(subject, cv2.COLOR_BGR2RGB)
                s = seg.process(rgb)
                if s.segmentation_mask is not None:
                    m = soft_subject_mask(s.segmentation_mask, float(cfg.seg_blur_sigma))
                    out = (out.astype(np.float32) * (1.0 - m) + subject.astype(np.float32) * m).astype(np.uint8)

            # ---- All overlays front ----
            for ov_cfg, ov_files, rig in zip(overlays, ov_file_lists, rigs):
                if ov_cfg.layer == "front":
                    out = _apply_overlay(out, subject, idx, ov_cfg, ov_files, rig, cfg.smoothing, cfg.reuse_last_on_miss)

            out_path = os.path.join(cfg.output_dir, os.path.basename(subj_path))
            cv2.imwrite(out_path, out)

    finally:
        for r in rigs:
            r.close()
        if seg is not None:
            seg.close()
