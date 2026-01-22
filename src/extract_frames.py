# src/extract_frames.py
"""
Video -> posterized frames extractor with:
- timestamp-based sampling (robust to variable FPS)
- percent-based cropping
- optional chroma-key alpha (none | auto | green/blue/black/white | #RRGGBB)
- optional trimming to non-transparent bounds
- optional resize

Designed to be imported by run.py, but can also be executed directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import argparse


# ----------------------------
# Config
# ----------------------------
@dataclass
class ExtractFramesConfig:
    target_fps: float = 10.0
    start_sec: float = 0.0
    end_sec: Optional[float] = None

    # NEW: stop after writing this many frames (None = no limit)
    max_frames: Optional[int] = None

    # crop percents (0..100)
    crop_top: float = 0.0
    crop_left: float = 0.0
    crop_right: float = 0.0
    crop_bottom: float = 0.0

    # resize (optional)
    resize_w: Optional[int] = None
    resize_h: Optional[int] = None

    # output naming
    prefix: str = "frame_"
    ext: str = "png"  # png or jpg

    # background keying
    bg: str = "none"  # none|auto|green|blue|black|white|#RRGGBB
    key_thr: float = 22.0
    key_soft: float = 35.0
    border_pct: float = 3.0

    # trim to alpha bbox
    trim: bool = False
    trim_pad: int = 0

    # overwrite folder contents
    clear_out_dir: bool = True


# ----------------------------
# Small utilities
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def clear_dir(p: str) -> None:
    if not os.path.isdir(p):
        return
    for name in os.listdir(p):
        fp = os.path.join(p, name)
        try:
            if os.path.isfile(fp) or os.path.islink(fp):
                os.remove(fp)
            elif os.path.isdir(fp):
                # shallow delete
                for root, dirs, files in os.walk(fp, topdown=False):
                    for f in files:
                        os.remove(os.path.join(root, f))
                    for d in dirs:
                        os.rmdir(os.path.join(root, d))
                os.rmdir(fp)
        except Exception:
            # best-effort cleanup
            pass


def percent_crop(frame: np.ndarray, *, top=0.0, left=0.0, right=0.0, bottom=0.0) -> np.ndarray:
    """
    Crop a frame by percentages of original size.
    Example: left=50 keeps right half.
    """
    h, w = frame.shape[:2]

    top_px = int(round(h * (top / 100.0)))
    bottom_px = int(round(h * (bottom / 100.0)))
    left_px = int(round(w * (left / 100.0)))
    right_px = int(round(w * (right / 100.0)))

    y1 = max(0, top_px)
    y2 = max(y1 + 1, h - bottom_px)
    x1 = max(0, left_px)
    x2 = max(x1 + 1, w - right_px)

    return frame[y1:y2, x1:x2]


def parse_bg_spec(bg: str) -> tuple[str, Optional[Tuple[int, int, int]]]:
    """
    Returns one of:
      ("none", None)
      ("auto", None)
      ("color", (b,g,r))
    """
    bg = (bg or "none").strip().lower()
    if bg in ("none", "off", "false", "0"):
        return ("none", None)
    if bg == "auto":
        return ("auto", None)
    if bg in ("green", "blue", "black", "white"):
        named = {
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
        }
        return ("color", named[bg])

    # hex: #RRGGBB or RRGGBB
    s = bg[1:] if bg.startswith("#") else bg
    s = s.lower()
    if len(s) == 6 and all(c in "0123456789abcdef" for c in s):
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return ("color", (b, g, r))

    raise ValueError(f"Unrecognized bg spec: {bg} (use none|auto|green|blue|black|white|#RRGGBB)")


def detect_key_color_from_border_bgr(frame_bgr: np.ndarray, border_pct: float = 3.0) -> Tuple[int, int, int]:
    """
    Heuristic: sample pixels from the border and pick the dominant color (median in Lab space).
    Works well for solid-color matte backgrounds.
    """
    h, w = frame_bgr.shape[:2]
    b = max(1, int(round(min(h, w) * (border_pct / 100.0))))

    top = frame_bgr[:b, :, :]
    bottom = frame_bgr[h - b : h, :, :]
    left = frame_bgr[:, :b, :]
    right = frame_bgr[:, w - b : w, :]

    samples = np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    )

    samples_bgr = samples.reshape(-1, 1, 3).astype(np.uint8)
    samples_lab = cv2.cvtColor(samples_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    med_lab = np.median(samples_lab, axis=0).astype(np.uint8).reshape(1, 1, 3)
    med_bgr = cv2.cvtColor(med_lab, cv2.COLOR_LAB2BGR).reshape(3).tolist()

    return (int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2]))


def alpha_from_key_lab(frame_bgr: np.ndarray, key_bgr: Tuple[int, int, int], thr: float, soft: float) -> np.ndarray:
    """
    Create alpha based on distance from key color in Lab space.
      dist <= thr       -> alpha=0
      dist >= thr+soft  -> alpha=255
      ramp in-between
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    key_patch = np.uint8([[list(key_bgr)]])
    key_lab = cv2.cvtColor(key_patch, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0, :]

    dist = np.linalg.norm(lab - key_lab[None, None, :], axis=2)

    a = (dist - float(thr)) / max(1e-6, float(soft))
    a = np.clip(a, 0.0, 1.0)
    return (a * 255.0).astype(np.uint8)


def trim_to_alpha(img_bgra: np.ndarray, alpha_thresh: int = 5, pad: int = 0) -> np.ndarray:
    """
    Crop image to bounding box of alpha > alpha_thresh.
    """
    a = img_bgra[:, :, 3]
    ys, xs = np.where(a > alpha_thresh)
    if len(xs) == 0:
        return img_bgra

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    x0 = max(0, int(x0) - pad)
    y0 = max(0, int(y0) - pad)
    x1 = min(img_bgra.shape[1] - 1, int(x1) + pad)
    y1 = min(img_bgra.shape[0] - 1, int(y1) + pad)

    return img_bgra[y0 : y1 + 1, x0 : x1 + 1]


# ----------------------------
# Main API
# ----------------------------
def extract_frames(video_path: str, out_dir: str, cfg: ExtractFramesConfig) -> int:
    """
    Extract posterized-time frames from video into out_dir.
    Returns number of frames saved.
    """
    if cfg.target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    for v in (cfg.crop_top, cfg.crop_left, cfg.crop_right, cfg.crop_bottom):
        if v < 0 or v >= 100:
            raise ValueError("Crop percentages must be in [0, 100)")

    bg_mode, bg_color = parse_bg_spec(cfg.bg)

    # If we produce alpha, force png
    ext = cfg.ext.lower()
    if bg_mode != "none" and ext != "png":
        ext = "png"

    ensure_dir(out_dir)
    if cfg.clear_out_dir:
        clear_dir(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, cfg.start_sec) * 1000.0)

    interval_ms = 1000.0 / float(cfg.target_fps)
    next_t_ms = max(0.0, cfg.start_sec) * 1000.0

    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if cfg.end_sec is not None and t_ms > cfg.end_sec * 1000.0:
            break

        # Save when we pass the next target timestamp
        if t_ms + 0.5 >= next_t_ms:
            # Crop
            if cfg.crop_top or cfg.crop_left or cfg.crop_right or cfg.crop_bottom:
                frame = percent_crop(
                    frame,
                    top=cfg.crop_top,
                    left=cfg.crop_left,
                    right=cfg.crop_right,
                    bottom=cfg.crop_bottom,
                )

            # Key -> BGRA
            if bg_mode != "none":
                key_bgr = (
                    detect_key_color_from_border_bgr(frame, border_pct=cfg.border_pct)
                    if bg_mode == "auto"
                    else bg_color
                )
                alpha = alpha_from_key_lab(frame, key_bgr, thr=cfg.key_thr, soft=cfg.key_soft)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                frame[:, :, 3] = alpha

                if cfg.trim:
                    frame = trim_to_alpha(frame, alpha_thresh=5, pad=int(cfg.trim_pad))

            # Resize
            if cfg.resize_w is not None or cfg.resize_h is not None:
                h, w = frame.shape[:2]
                if cfg.resize_w is not None and cfg.resize_h is not None:
                    rw, rh = cfg.resize_w, cfg.resize_h
                elif cfg.resize_w is not None:
                    rw = cfg.resize_w
                    rh = int(round(h * (cfg.resize_w / w)))
                else:
                    rh = cfg.resize_h
                    rw = int(round(w * (cfg.resize_h / h)))

                interp = cv2.INTER_AREA if (rw < w or rh < h) else cv2.INTER_LINEAR
                frame = cv2.resize(frame, (rw, rh), interpolation=interp)

            out_path = os.path.join(out_dir, f"{cfg.prefix}{saved:06d}.{ext}")
            cv2.imwrite(out_path, frame)

            saved += 1
            if cfg.max_frames is not None and saved >= int(cfg.max_frames):
                break
            next_t_ms += interval_ms

    cap.release()
    return saved


# ----------------------------
# CLI wrapper (optional)
# ----------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract posterized-time frames from a video with percent cropping and optional chroma-key alpha."
    )
    ap.add_argument("video", help="Input video path")
    ap.add_argument("out_dir", help="Output directory for frames")

    ap.add_argument("--fps", type=float, default=10.0, help="Target FPS (default: 10)")
    ap.add_argument("--start", type=float, default=0.0, help="Start time seconds")
    ap.add_argument("--end", type=float, default=None, help="End time seconds")

    ap.add_argument("--top", type=float, default=0.0, help="Crop percent from top")
    ap.add_argument("--left", type=float, default=0.0, help="Crop percent from left")
    ap.add_argument("--right", type=float, default=0.0, help="Crop percent from right")
    ap.add_argument("--bottom", type=float, default=0.0, help="Crop percent from bottom")

    ap.add_argument("--resize_w", type=int, default=None, help="Resize width")
    ap.add_argument("--resize_h", type=int, default=None, help="Resize height")

    ap.add_argument("--prefix", type=str, default="frame_", help="Filename prefix")
    ap.add_argument("--ext", type=str, default="png", choices=["png", "jpg"], help="Output format")

    ap.add_argument(
        "--bg",
        type=str,
        default="none",
        help="Background key: none|auto|green|blue|black|white|#RRGGBB (enables alpha; forces png)",
    )
    ap.add_argument("--key_thr", type=float, default=22.0, help="Key threshold")
    ap.add_argument("--key_soft", type=float, default=35.0, help="Key feather softness")
    ap.add_argument("--border_pct", type=float, default=3.0, help="Border percent used for --bg auto")

    ap.add_argument("--trim", action="store_true", help="Trim to non-transparent bounding box")
    ap.add_argument("--trim_pad", type=int, default=0, help="Padding (px) when trimming")

    ap.add_argument("--no_clear", action="store_true", help="Do not clear output directory before writing")

    return ap


def main() -> None:
    ap = _build_arg_parser()
    args = ap.parse_args()

    cfg = ExtractFramesConfig(
        target_fps=args.fps,
        start_sec=args.start,
        end_sec=args.end,
        crop_top=args.top,
        crop_left=args.left,
        crop_right=args.right,
        crop_bottom=args.bottom,
        resize_w=args.resize_w,
        resize_h=args.resize_h,
        prefix=args.prefix,
        ext=args.ext,
        bg=args.bg,
        key_thr=args.key_thr,
        key_soft=args.key_soft,
        border_pct=args.border_pct,
        trim=args.trim,
        trim_pad=args.trim_pad,
        clear_out_dir=(not args.no_clear),
    )

    n = extract_frames(args.video, args.out_dir, cfg)
    print(f"Done. Saved {n} frames to: {args.out_dir}")


if __name__ == "__main__":
    main()
