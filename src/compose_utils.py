# src/compose_utils.py
from __future__ import annotations

import cv2
import numpy as np


def alpha_composite(bg_bgr: np.ndarray, fg_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    h_bg, w_bg = bg_bgr.shape[:2]
    h_fg, w_fg = fg_rgba.shape[:2]

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w_fg, w_bg)
    y2 = min(y + h_fg, h_bg)

    if x1 >= x2 or y1 >= y2:
        return bg_bgr

    fg = fg_rgba[y1 - y : y2 - y, x1 - x : x2 - x]
    bg = bg_bgr[y1:y2, x1:x2]

    fg_rgb = fg[:, :, :3].astype(np.float32)
    fg_a = fg[:, :, 3:4].astype(np.float32) / 255.0
    bg_rgb = bg.astype(np.float32)

    out = fg_rgb * fg_a + bg_rgb * (1.0 - fg_a)
    bg_bgr[y1:y2, x1:x2] = out.astype(np.uint8)
    return bg_bgr


def rotate_and_scale_rgba(img: np.ndarray, angle_deg: float, target_width_px: float) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= 0:
        return img

    scale = float(target_width_px) / float(w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if abs(angle_deg) < 1e-3:
        return img

    h2, w2 = img.shape[:2]
    center = (w2 / 2.0, h2 / 2.0)
    M = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    bw = int(h2 * sin + w2 * cos)
    bh = int(h2 * cos + w2 * sin)

    M[0, 2] += (bw / 2.0) - center[0]
    M[1, 2] += (bh / 2.0) - center[1]

    return cv2.warpAffine(
        img,
        M,
        (bw, bh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def fit_image(bg: np.ndarray, target_w: int, target_h: int, mode: str) -> np.ndarray:
    """
    Fit bg into (target_w, target_h) without distortion unless mode='stretch'.

    modes:
      - stretch: resize directly (distorts)
      - cover: scale to fill, then center-crop
      - contain: scale to fit, then letterbox
    """
    h, w = bg.shape[:2]

    if mode == "stretch":
        return cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    scale_w = target_w / w
    scale_h = target_h / h

    if mode == "cover":
        scale = max(scale_w, scale_h)
    elif mode == "contain":
        scale = min(scale_w, scale_h)
    else:
        raise ValueError(f"Unknown bg_fit mode: {mode}")

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if mode == "cover":
        x0 = max(0, (new_w - target_w) // 2)
        y0 = max(0, (new_h - target_h) // 2)
        return resized[y0 : y0 + target_h, x0 : x0 + target_w]

    # contain -> letterbox
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas
