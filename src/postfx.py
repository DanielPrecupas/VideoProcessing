"""
Post-FX stage (separate from compositing).

Drop-in compatible with your existing run.py usage, but adds:
- posterize (N levels) + optional ordered dithering
- optional halftone-ish "screen" look (coarse luma sampling)
- caption overlay (multiline + boxed + placement + jitter)
- zero-touch overrides via:
    work/caption.txt   (caption text)
    work/postfx.json   (any PostFXConfig fields)

No MediaPipe here. Pure image processing.
"""

from __future__ import annotations

import os
import glob
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np


# ----------------------------
# Config
# ----------------------------
@dataclass
class PostFXConfig:
    enable: bool = True

    # ---- Core look mode ----
    # "two_tone"   = classic ink/paper threshold (your old default)
    # "posterize"  = N-level posterize (more Forgive Me Father-ish)
    mode: str = "two_tone"

    # Two-tone monochrome palette (BGR)
    ink_bgr: Tuple[int, int, int] = (10, 10, 10)
    paper_bgr: Tuple[int, int, int] = (245, 245, 245)

    # Pre-contrast (LAB L channel)
    clahe_clip: float = 2.0
    clahe_tile: int = 8

    # two_tone thresholding
    threshold_mode: str = "otsu"     # "otsu" or "fixed"
    fixed_thresh: int = 128          # used if threshold_mode="fixed"

    # posterize
    posterize_levels: int = 4        # 2..10 recommended
    posterize_gamma: float = 1.0     # >1 darkens mids, <1 lifts
    posterize_smooth: float = 0.0    # 0..1 (tiny blur before quantize)

    # Ordered dithering (helps posterize look more "printy")
    # "none" | "bayer4" | "bayer8"
    dither: str = "none"
    dither_strength: float = 0.75    # 0..1 typical

    # Optional "halftone-ish" screen: sample luma in blocks then upscale
    halftone: bool = False
    halftone_cell: int = 6           # 4..12 typical

    # Cleanup (helps speckle/holes)
    median_ksize: int = 0            # 0 disables; 3 for mild cleanup
    close_iters: int = 0             # 0 disables; 1 can fill small holes
    open_iters: int = 0              # 0 disables; 1 can remove tiny noise

    # Optional outline (very fine)
    outline_enable: bool = False
    outline_strength: float = 0.25   # 0..1
    outline_thresh: int = 35         # higher => fewer edges
    outline_thickness: int = 1       # 1 = very fine

    # Grain + vignette
    grain_amount: float = 0.0        # 0..0.06 typical
    vignette: float = 0.0            # 0..0.4 typical

    # Sequence-level time posterize (frame hold)
    time_posterize: bool = False
    source_fps: float = 30.0
    target_fps: float = 10.0

    # ---- Caption overlay ----
    caption_enable: bool = False
    caption_text: str = ""           # if empty, will try work/caption.txt or env POSTFX_CAPTION
    caption_start: int = 0           # frame index inclusive
    caption_end: Optional[int] = None  # frame index exclusive

    # placement: "top_left", "top_center", "top_right",
    #            "bottom_left", "bottom_center", "bottom_right",
    #            "center"
    caption_pos: str = "bottom_center"
    caption_margin: int = 24
    caption_font: int = cv2.FONT_HERSHEY_SIMPLEX
    caption_scale: float = 1.0
    caption_thickness: int = 2
    caption_color_bgr: Tuple[int, int, int] = (245, 245, 245)

    caption_box: bool = True
    caption_box_color_bgr: Tuple[int, int, int] = (10, 10, 10)
    caption_box_alpha: float = 0.65
    caption_box_pad: int = 10

    caption_jitter_px: int = 0       # 0 disables; subtle 1..3 looks "alive"

    # IO
    out_ext: str = "png"             # png recommended


# ----------------------------
# IO helpers
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

def clear_dir(p: str) -> None:
    if not os.path.isdir(p):
        return
    for name in os.listdir(p):
        fp = os.path.join(p, name)
        try:
            if os.path.isfile(fp) or os.path.islink(fp):
                os.remove(fp)
        except Exception:
            pass


def _try_load_overrides(cfg: PostFXConfig, in_dir: str) -> None:
    """
    Allow zero-touch tweaking without editing run.py:
      - work/postfx.json (parent of in_dir)
      - env POSTFX_JSON (path)
    """
    candidates: List[str] = []
    parent = os.path.dirname(os.path.normpath(in_dir))
    candidates.append(os.path.join(parent, "postfx.json"))

    env_json = os.environ.get("POSTFX_JSON", "").strip()
    if env_json:
        candidates.insert(0, env_json)

    path = next((p for p in candidates if p and os.path.isfile(p)), None)
    if not path:
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    if not isinstance(data, dict):
        return

    # Apply only known fields
    for k, v in data.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass


def _resolve_caption_text(cfg: PostFXConfig, in_dir: str) -> str:
    """
    If cfg.caption_text is empty, try:
      - env POSTFX_CAPTION
      - work/caption.txt (parent of in_dir)
    """
    if cfg.caption_text and cfg.caption_text.strip():
        return cfg.caption_text

    env_cap = os.environ.get("POSTFX_CAPTION", "").strip()
    if env_cap:
        return env_cap

    parent = os.path.dirname(os.path.normpath(in_dir))
    cap_path = os.path.join(parent, "caption.txt")
    if os.path.isfile(cap_path):
        try:
            with open(cap_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""

    return ""


# ----------------------------
# Core image ops
# ----------------------------
def apply_vignette(img_bgr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r2 = x * x + y * y
    mask = 1.0 - float(strength) * np.clip(r2, 0.0, 1.0)
    out = img_bgr.astype(np.float32) * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

def add_grain(img_bgr: np.ndarray, amount: float, seed: Optional[int] = None) -> np.ndarray:
    if amount <= 0:
        return img_bgr
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 255.0 * float(amount), size=img_bgr.shape).astype(np.float32)
    out = img_bgr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def _clahe_luma(img_bgr: np.ndarray, cfg: PostFXConfig) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    clahe = cv2.createCLAHE(
        clipLimit=float(cfg.clahe_clip),
        tileGridSize=(int(cfg.clahe_tile), int(cfg.clahe_tile)),
    )
    L2 = clahe.apply(L)
    return L2

def _cleanup_bw(bw: np.ndarray, cfg: PostFXConfig) -> np.ndarray:
    if cfg.median_ksize and cfg.median_ksize >= 3 and cfg.median_ksize % 2 == 1:
        bw = cv2.medianBlur(bw, int(cfg.median_ksize))

    if cfg.close_iters > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=int(cfg.close_iters))
    if cfg.open_iters > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=int(cfg.open_iters))
    return bw

def _bayer(n: int) -> np.ndarray:
    # Classic Bayer matrices normalized to [0,1)
    if n == 4:
        m = np.array([
            [0,  8,  2, 10],
            [12, 4, 14,  6],
            [3, 11,  1,  9],
            [15, 7, 13,  5],
        ], dtype=np.float32)
        return (m + 0.5) / 16.0
    if n == 8:
        m4 = _bayer(4)
        # Expand 4->8 (simple recursive construction)
        m = np.block([
            [4*m4,     4*m4 + 2],
            [4*m4 + 3, 4*m4 + 1],
        ])
        return (m + 0.5) / 64.0
    raise ValueError("bayer size must be 4 or 8")

def _ordered_dither(lum_8u: np.ndarray, cfg: PostFXConfig) -> np.ndarray:
    d = (cfg.dither or "none").lower()
    if d == "none":
        return lum_8u

    size = 4 if d == "bayer4" else 8
    mat = _bayer(size)  # [0,1)
    h, w = lum_8u.shape[:2]

    # Tile to image size
    tiled = np.tile(mat, (h // size + 1, w // size + 1))[:h, :w]
    # Convert to offset in [-0.5..0.5] and scale
    offset = (tiled - 0.5) * (255.0 * float(cfg.dither_strength))
    out = lum_8u.astype(np.float32) + offset.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)

def _halftoneish(lum_8u: np.ndarray, cell: int) -> np.ndarray:
    cell = max(2, int(cell))
    h, w = lum_8u.shape[:2]
    sh = max(1, h // cell)
    sw = max(1, w // cell)
    small = cv2.resize(lum_8u, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def two_tone_mono(img_bgr: np.ndarray, cfg: PostFXConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      toned_bgr: mapped to ink/paper colors
      lum_8u:    luminance used for optional outline
    """
    lum = _clahe_luma(img_bgr, cfg)

    if (cfg.dither or "none").lower() != "none":
        lum = _ordered_dither(lum, cfg)

    if cfg.halftone:
        lum = _halftoneish(lum, cfg.halftone_cell)

    if cfg.threshold_mode.lower() == "fixed":
        t = int(np.clip(cfg.fixed_thresh, 0, 255))
        _, bw = cv2.threshold(lum, t, 255, cv2.THRESH_BINARY)
    else:
        _, bw = cv2.threshold(lum, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bw = _cleanup_bw(bw, cfg)

    out = np.zeros_like(img_bgr, dtype=np.uint8)
    out[bw == 0] = cfg.ink_bgr
    out[bw == 255] = cfg.paper_bgr
    return out, lum

def posterize(img_bgr: np.ndarray, cfg: PostFXConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Posterize via L channel quantization, then map to ink/paper-ish palette
    by remapping quantized levels between ink and paper.
    """
    lum = _clahe_luma(img_bgr, cfg)

    if cfg.posterize_smooth > 0:
        sigma = float(cfg.posterize_smooth) * 2.5
        lum = cv2.GaussianBlur(lum, (0, 0), sigma)

    if cfg.posterize_gamma and abs(float(cfg.posterize_gamma) - 1.0) > 1e-3:
        x = lum.astype(np.float32) / 255.0
        x = np.power(np.clip(x, 0, 1), float(cfg.posterize_gamma))
        lum = np.clip(x * 255.0, 0, 255).astype(np.uint8)

    if (cfg.dither or "none").lower() != "none":
        lum = _ordered_dither(lum, cfg)

    if cfg.halftone:
        lum = _halftoneish(lum, cfg.halftone_cell)

    levels = int(np.clip(cfg.posterize_levels, 2, 10))
    # quantize to [0..levels-1]
    q = np.floor((lum.astype(np.float32) / 255.0) * levels).astype(np.int32)
    q = np.clip(q, 0, levels - 1)

    # map each level to a linear ramp between ink and paper
    t = (q.astype(np.float32) / float(levels - 1))[..., None]  # 0..1
    ink = np.array(cfg.ink_bgr, dtype=np.float32)[None, None, :]
    paper = np.array(cfg.paper_bgr, dtype=np.float32)[None, None, :]
    out = ink * (1.0 - t) + paper * t
    return np.clip(out, 0, 255).astype(np.uint8), lum

def apply_outline(toned_bgr: np.ndarray, lum_8u: np.ndarray, cfg: PostFXConfig) -> np.ndarray:
    if not cfg.outline_enable or cfg.outline_strength <= 0 or cfg.outline_thickness <= 0:
        return toned_bgr

    grad = cv2.morphologyEx(lum_8u, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edges = cv2.threshold(grad, int(cfg.outline_thresh), 255, cv2.THRESH_BINARY)[1]

    k = 2 * int(cfg.outline_thickness) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    edges = cv2.dilate(edges, kernel, iterations=1)

    edge_mask = (edges.astype(np.float32) / 255.0)[..., None]
    out = toned_bgr.astype(np.float32) * (1.0 - float(cfg.outline_strength) * edge_mask)
    return np.clip(out, 0, 255).astype(np.uint8)

def _draw_caption(img_bgr: np.ndarray, cfg: PostFXConfig, text: str, idx: int) -> np.ndarray:
    if not text.strip():
        return img_bgr

    if idx < int(cfg.caption_start):
        return img_bgr
    if cfg.caption_end is not None and idx >= int(cfg.caption_end):
        return img_bgr

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return img_bgr

    h, w = img_bgr.shape[:2]
    margin = int(cfg.caption_margin)

    # Measure block
    sizes = [cv2.getTextSize(ln, cfg.caption_font, float(cfg.caption_scale), int(cfg.caption_thickness))[0] for ln in lines]
    line_h = max(s[1] for s in sizes) + 8
    block_w = max(s[0] for s in sizes)
    block_h = line_h * len(lines)

    pad = int(cfg.caption_box_pad) if cfg.caption_box else 0
    box_w = block_w + 2 * pad
    box_h = block_h + 2 * pad

    # Anchor
    pos = (cfg.caption_pos or "bottom_center").lower()
    if pos == "top_left":
        x0, y0 = margin, margin
    elif pos == "top_center":
        x0, y0 = (w - box_w) // 2, margin
    elif pos == "top_right":
        x0, y0 = w - box_w - margin, margin
    elif pos == "bottom_left":
        x0, y0 = margin, h - box_h - margin
    elif pos == "bottom_right":
        x0, y0 = w - box_w - margin, h - box_h - margin
    elif pos == "center":
        x0, y0 = (w - box_w) // 2, (h - box_h) // 2
    else:  # bottom_center
        x0, y0 = (w - box_w) // 2, h - box_h - margin

    # Jitter
    j = int(cfg.caption_jitter_px)
    if j > 0:
        rng = np.random.default_rng(10_000 + idx)
        x0 += int(rng.integers(-j, j + 1))
        y0 += int(rng.integers(-j, j + 1))

    # Clamp
    x0 = int(np.clip(x0, 0, max(0, w - box_w)))
    y0 = int(np.clip(y0, 0, max(0, h - box_h)))

    out = img_bgr.copy()

    # Box (alpha blended)
    if cfg.caption_box and cfg.caption_box_alpha > 0:
        alpha = float(np.clip(cfg.caption_box_alpha, 0.0, 1.0))
        bx1, by1 = x0 + box_w, y0 + box_h
        overlay = out.copy()
        cv2.rectangle(
            overlay,
            (x0, y0),
            (bx1, by1),
            tuple(int(v) for v in cfg.caption_box_color_bgr),
            thickness=-1,
        )
        out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)

    # Text lines
    tx = x0 + pad
    ty = y0 + pad + line_h - 10
    for ln in lines:
        cv2.putText(
            out,
            ln,
            (tx, ty),
            cfg.caption_font,
            float(cfg.caption_scale),
            tuple(int(v) for v in cfg.caption_color_bgr),
            int(cfg.caption_thickness),
            lineType=cv2.LINE_AA,
        )
        ty += line_h

    return out

def postfx_frame(img_bgr: np.ndarray, cfg: PostFXConfig, seed: Optional[int] = None) -> np.ndarray:
    if not cfg.enable:
        out = img_bgr
        # caption can still be used even with enable=False if you want
        if cfg.caption_enable:
            out = _draw_caption(out, cfg, cfg.caption_text, seed or 0)
        return out

    mode = (cfg.mode or "two_tone").lower()
    if mode == "posterize":
        toned, lum = posterize(img_bgr, cfg)
    else:
        toned, lum = two_tone_mono(img_bgr, cfg)

    toned = apply_outline(toned, lum, cfg)
    toned = apply_vignette(toned, cfg.vignette)
    toned = add_grain(toned, cfg.grain_amount, seed=seed)

    if cfg.caption_enable:
        toned = _draw_caption(toned, cfg, cfg.caption_text, seed or 0)

    return toned


# ----------------------------
# Folder processor
# ----------------------------
def apply_postfx_folder(
    in_dir: str,
    out_dir: str,
    cfg: PostFXConfig,
    *,
    clear_out_dir: bool = True,
) -> int:
    """
    Reads frames from in_dir, writes processed frames to out_dir.
    Returns number of frames written.

    If cfg.time_posterize is True, it will hold frames to achieve target_fps look.
    """

    # Allow overrides without touching run.py
    _try_load_overrides(cfg, in_dir)

    # Resolve caption text (caption.txt / env)
    if cfg.caption_enable:
        cfg.caption_text = _resolve_caption_text(cfg, in_dir)

    ensure_dir(out_dir)
    if clear_out_dir:
        clear_dir(out_dir)

    files = list_images(in_dir)
    if not files:
        raise RuntimeError(f"No frames found in: {in_dir}")

    hold = 1
    if cfg.time_posterize:
        if cfg.target_fps <= 0 or cfg.source_fps <= 0:
            raise ValueError("source_fps and target_fps must be > 0 when time_posterize is enabled")
        hold = max(1, int(round(float(cfg.source_fps) / float(cfg.target_fps))))

    last_stylized: Optional[np.ndarray] = None
    written = 0

    for idx, path in enumerate(files):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        do_update = (idx % hold == 0) or (last_stylized is None)

        if do_update:
            last_stylized = postfx_frame(img, cfg, seed=idx)
        else:
            # hold stylized frame (temporal stepping)
            if cfg.caption_enable:
                # caption can jitter per-frame; re-draw on held base if you want it "alive"
                last_stylized = _draw_caption(last_stylized, cfg, cfg.caption_text, idx) if last_stylized is not None else None

        out = last_stylized if last_stylized is not None else img

        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{base}.{cfg.out_ext}")
        cv2.imwrite(out_path, out)
        written += 1

    return written
