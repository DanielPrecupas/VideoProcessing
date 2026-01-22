import os
import cv2
import argparse
import numpy as np

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def percent_crop(frame, *, top=0.0, left=0.0, right=0.0, bottom=0.0):
    """
    Crop a frame by percentages.
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

def parse_bg_spec(bg: str):
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
    s = bg
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 6 and all(c in "0123456789abcdef" for c in s):
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return ("color", (b, g, r))

    raise ValueError(f"Unrecognized --bg value: {bg} (use none|auto|green|blue|black|white|#RRGGBB)")

def detect_key_color_from_border_bgr(frame_bgr: np.ndarray, border_pct: float = 3.0) -> tuple[int, int, int]:
    """
    Heuristic: sample pixels from the border and pick the dominant color (median in Lab space).
    Works well for solid-color matte backgrounds.
    """
    h, w = frame_bgr.shape[:2]
    b = max(1, int(round(min(h, w) * (border_pct / 100.0))))

    top = frame_bgr[:b, :, :]
    bottom = frame_bgr[h-b:h, :, :]
    left = frame_bgr[:, :b, :]
    right = frame_bgr[:, w-b:w, :]

    samples = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3),
    ], axis=0)

    # Convert samples to Lab and take median for robustness
    samples_bgr = samples.reshape(-1, 1, 3).astype(np.uint8)
    samples_lab = cv2.cvtColor(samples_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    med_lab = np.median(samples_lab, axis=0).astype(np.uint8).reshape(1, 1, 3)
    med_bgr = cv2.cvtColor(med_lab, cv2.COLOR_LAB2BGR).reshape(3).tolist()

    return (int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2]))

def alpha_from_key_lab(frame_bgr: np.ndarray, key_bgr: tuple[int, int, int], thr: float, soft: float) -> np.ndarray:
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
    alpha = (a * 255.0).astype(np.uint8)
    return alpha

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

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(img_bgra.shape[1] - 1, x1 + pad)
    y1 = min(img_bgra.shape[0] - 1, y1 + pad)

    return img_bgra[y0:y1+1, x0:x1+1]

def extract_posterized_frames(
    video_path: str,
    out_dir: str,
    target_fps: float = 10.0,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    resize_w: int | None = None,
    resize_h: int | None = None,
    crop_top: float = 0.0,
    crop_left: float = 0.0,
    crop_right: float = 0.0,
    crop_bottom: float = 0.0,
    prefix: str = "frame_",
    ext: str = "png",
    bg: str = "none",
    key_thr: float = 22.0,
    key_soft: float = 35.0,
    border_pct: float = 3.0,
    trim: bool = False,
    trim_pad: int = 0,
):
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    for v in (crop_top, crop_left, crop_right, crop_bottom):
        if v < 0 or v >= 100:
            raise ValueError("Crop percentages must be in [0, 100)")

    bg_mode, bg_color = parse_bg_spec(bg)

    # If we're producing alpha, force PNG
    if bg_mode != "none" and ext.lower() != "png":
        print("[info] --bg enabled -> forcing PNG output (JPEG cannot store alpha).")
        ext = "png"

    ensure_dir(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_sec) * 1000.0)

    interval_ms = 1000.0 / float(target_fps)
    next_t_ms = max(0.0, start_sec) * 1000.0

    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if end_sec is not None and t_ms > end_sec * 1000.0:
            break

        if t_ms + 0.5 >= next_t_ms:
            # --- Crop first ---
            if crop_top or crop_left or crop_right or crop_bottom:
                frame = percent_crop(
                    frame,
                    top=crop_top,
                    left=crop_left,
                    right=crop_right,
                    bottom=crop_bottom,
                )

            # --- Background key -> BGRA ---
            if bg_mode != "none":
                if bg_mode == "auto":
                    key_bgr = detect_key_color_from_border_bgr(frame, border_pct=border_pct)
                else:
                    key_bgr = bg_color

                alpha = alpha_from_key_lab(frame, key_bgr, thr=key_thr, soft=key_soft)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                frame[:, :, 3] = alpha

                if trim:
                    frame = trim_to_alpha(frame, alpha_thresh=5, pad=int(trim_pad))

            # --- Resize ---
            if resize_w is not None or resize_h is not None:
                h, w = frame.shape[:2]
                if resize_w is not None and resize_h is not None:
                    rw, rh = resize_w, resize_h
                elif resize_w is not None:
                    rw = resize_w
                    rh = int(round(h * (resize_w / w)))
                else:
                    rh = resize_h
                    rw = int(round(w * (resize_h / h)))

                interp = cv2.INTER_AREA if (rw < w or rh < h) else cv2.INTER_LINEAR
                frame = cv2.resize(frame, (rw, rh), interpolation=interp)

            out_path = os.path.join(out_dir, f"{prefix}{saved:06d}.{ext}")
            cv2.imwrite(out_path, frame)

            saved += 1
            next_t_ms += interval_ms

            if saved % 100 == 0:
                print(f"Saved {saved} frames... ({t_ms/1000.0:.2f}s)")

    cap.release()
    print(f"Done. Saved {saved} frames to: {out_dir}")

def main():
    ap = argparse.ArgumentParser(
        description="Extract frames from a video with time posterization, percent cropping, and optional chroma-key alpha."
    )
    ap.add_argument("video", help="Path to input video file")
    ap.add_argument(
        "--dest",
        choices=["input_frames", "wings_frames"],
        default="input_frames",
        help="Destination folder name (default: input_frames)",
    )
    ap.add_argument("--out", default=None, help="Output directory (overrides --dest)")
    ap.add_argument("--fps", type=float, default=10.0, help="Target FPS (default: 10)")
    ap.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    ap.add_argument("--end", type=float, default=None, help="End time in seconds")

    # Cropping (percent)
    ap.add_argument("--top", type=float, default=0.0, help="Crop percent from top")
    ap.add_argument("--left", type=float, default=0.0, help="Crop percent from left")
    ap.add_argument("--right", type=float, default=0.0, help="Crop percent from right")
    ap.add_argument("--bottom", type=float, default=0.0, help="Crop percent from bottom")

    # Resize
    ap.add_argument("--resize_w", type=int, default=None, help="Resize width")
    ap.add_argument("--resize_h", type=int, default=None, help="Resize height")

    ap.add_argument("--prefix", type=str, default="frame_", help="Filename prefix")
    ap.add_argument("--ext", type=str, default="png", choices=["png", "jpg"], help="Output format (png recommended)")

    # Background keying
    ap.add_argument(
        "--bg",
        type=str,
        default="none",
        help="Background key: none|auto|green|blue|black|white|#RRGGBB (enables alpha; forces png)",
    )
    ap.add_argument("--key_thr", type=float, default=22.0, help="Key threshold (lower = more aggressive transparency)")
    ap.add_argument("--key_soft", type=float, default=35.0, help="Key feather softness (higher = smoother edges)")
    ap.add_argument("--border_pct", type=float, default=3.0, help="Border percent used for --bg auto detection")
    ap.add_argument("--trim", action="store_true", help="Trim to non-transparent bounding box (alpha > 5)")
    ap.add_argument("--trim_pad", type=int, default=0, help="Padding (pixels) when trimming")

    args = ap.parse_args()
    out_dir = args.out if args.out is not None else args.dest

    extract_posterized_frames(
        video_path=args.video,
        out_dir=out_dir,
        target_fps=args.fps,
        start_sec=args.start,
        end_sec=args.end,
        resize_w=args.resize_w,
        resize_h=args.resize_h,
        crop_top=args.top,
        crop_left=args.left,
        crop_right=args.right,
        crop_bottom=args.bottom,
        prefix=args.prefix,
        ext=args.ext,
        bg=args.bg,
        key_thr=args.key_thr,
        key_soft=args.key_soft,
        border_pct=args.border_pct,
        trim=args.trim,
        trim_pad=args.trim_pad,
    )

if __name__ == "__main__":
    main()
