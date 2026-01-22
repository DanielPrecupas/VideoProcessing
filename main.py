import os
import glob
import cv2
import numpy as np
import mediapipe as mp

# ----------------------------
# CONFIG
# ----------------------------
INPUT_DIR = "input_frames"
WINGS_DIR = "wings_frames"
OUTPUT_DIR = "output_frames"

# Where wings attach relative to torso:
# 0.0 = mid-shoulders, 1.0 = mid-hips. ~0.20–0.35 usually good.
TORSO_ATTACH_T = 0.25

# Wings placement tweaks (pixels in output frame space)
WING_OFFSET_X = 0
WING_OFFSET_Y = 0

# How wide should wings be relative to shoulder width?
WING_WIDTH_MULTIPLIER = 4

# Smooth pose to reduce jitter (0=no smoothing, 0.8=heavy)
SMOOTHING = 0.6

# If pose not detected: reuse last transform or skip wings
REUSE_LAST_ON_MISS = True

# --- Wings keying (remove black background) ---
CHROMA_KEY_BGR = (0, 0, 0)   # black
KEY_THR = 22                # lower => more aggressive key
KEY_SOFT = 35               # feather range (higher => smoother edges)

# --- Subject occlusion (wings behind person) ---
USE_SEGMENTATION = True
SEG_MODEL_SELECTION = 1
SEG_BLUR_SIGMA = 2.5

# --- Wing rotation mode ---
# "none": do not rotate wings at all (recommended if wings frames already animate)
# "shoulders": follow shoulder line angle
# "torso_clamped": rotate by torso lean, clamped
WING_ROTATION_MODE = "none"
TORSO_CLAMP_DEG = 8.0  # used in torso_clamped

# ----------------------------
# POST / "BRUTALIST" LOOK
# ----------------------------
ENABLE_POST = True

# Time posterize (frame holds)
SOURCE_FPS = 10
TARGET_FPS = 10  # 8–12 feels animated

# Optional motion-based "impact" holds
ENABLE_IMPACT_HOLDS = True
IMPACT_MOTION_THRESH = 12.0
IMPACT_HOLD_EXTRA = 1

# Brutalist tone mapping
TONE_LEVELS = 3          # 2, 3, or 4
CLAHE_CLIP = 2.0
CLAHE_TILE = 8

# Palette (BGR). Use mono tones by default (safe, brutal)
# If you want an accent palette, change these.
PALETTE_2 = [(18, 18, 18), (235, 235, 235)]
PALETTE_3 = [(18, 18, 18), (95, 95, 95), (235, 235, 235)]
PALETTE_4 = [(18, 18, 18), (70, 70, 70), (150, 150, 150), (235, 235, 235)]

EDGE_THICKNESS = 0
EDGE_STRENGTH = 0.25
EDGE_THRESH = 35

# Grain + vignette (matte film)
GRAIN_AMOUNT = 0.025     # 0..0.06
VIGNETTE = 0.15          # 0..0.4
GATE_WEAVE_PX = 0        # 0..2 (keep 0 for "realistic brutal")

# ----------------------------
# Helpers
# ----------------------------
def list_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def lerp(a, b, t):
    return a * (1.0 - t) + b * t

def add_alpha_by_chroma_key(img_bgr, key_bgr=(0, 0, 0), thr=22, soft=35):
    """
    Create RGBA by making pixels near key_bgr transparent.
    Feathered edge via distance ramp.
    """
    bgr16 = img_bgr.astype(np.int16)
    key16 = np.array(key_bgr, dtype=np.int16)
    dist = np.linalg.norm(bgr16 - key16, axis=2).astype(np.float32)

    a = (dist - float(thr)) / max(1.0, float(soft))
    a = np.clip(a, 0.0, 1.0)
    alpha = (a * 255.0).astype(np.uint8)
    return np.dstack([img_bgr, alpha])

def read_rgba_wings(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return img
    return add_alpha_by_chroma_key(img, CHROMA_KEY_BGR, KEY_THR, KEY_SOFT)

def alpha_composite(bg_bgr, fg_rgba, x, y):
    h_bg, w_bg = bg_bgr.shape[:2]
    h_fg, w_fg = fg_rgba.shape[:2]

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w_fg, w_bg)
    y2 = min(y + h_fg, h_bg)
    if x1 >= x2 or y1 >= y2:
        return bg_bgr

    fg_crop = fg_rgba[y1 - y : y2 - y, x1 - x : x2 - x]
    bg_crop = bg_bgr[y1:y2, x1:x2]

    fg_rgb = fg_crop[:, :, :3].astype(np.float32)
    fg_a = (fg_crop[:, :, 3:4].astype(np.float32)) / 255.0
    bg_rgb = bg_crop.astype(np.float32)

    out = fg_rgb * fg_a + bg_rgb * (1.0 - fg_a)
    bg_bgr[y1:y2, x1:x2] = out.astype(np.uint8)
    return bg_bgr

def rotate_and_scale_rgba(img_rgba, angle_degrees, target_width_px):
    h, w = img_rgba.shape[:2]
    if w <= 0:
        return img_rgba

    scale = float(target_width_px) / float(w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    scaled = cv2.resize(img_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if abs(angle_degrees) < 1e-3:
        return scaled

    h2, w2 = scaled.shape[:2]
    center = (w2 / 2.0, h2 / 2.0)
    M = cv2.getRotationMatrix2D(center, float(angle_degrees), 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w_bb = int((h2 * sin) + (w2 * cos))
    new_h_bb = int((h2 * cos) + (w2 * sin))

    M[0, 2] += (new_w_bb / 2.0) - center[0]
    M[1, 2] += (new_h_bb / 2.0) - center[1]

    rotated = cv2.warpAffine(
        scaled,
        M,
        (new_w_bb, new_h_bb),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated

def soft_subject_mask(segmentation_mask, sigma=2.5):
    m = segmentation_mask
    if sigma and sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    m = np.clip(m, 0.0, 1.0)
    return m[..., None].astype(np.float32)

# ----------------------------
# Brutalist Post-FX
# ----------------------------
def apply_vignette(img_bgr, strength=0.15):
    if strength <= 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r2 = x * x + y * y
    mask = 1.0 - strength * np.clip(r2, 0.0, 1.0)
    mask = mask[..., None]
    out = img_bgr.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)

def add_grain(img_bgr, amount=0.025, seed=None):
    if amount <= 0:
        return img_bgr
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 255.0 * amount, size=img_bgr.shape).astype(np.float32)
    out = img_bgr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def tone_map_brutalist(img_bgr, levels=2):
    """
    True 2-tone monochrome:
      - CLAHE on luminance
      - Otsu threshold (or use a fixed threshold if you prefer)
      - map to PALETTE_2
    Returns (toned_bgr, lum_8u_used_for_edges)
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=float(CLAHE_CLIP), tileGridSize=(int(CLAHE_TILE), int(CLAHE_TILE)))
    L2 = clahe.apply(L)

    # Otsu gives stable 2-tone without hand tuning per scene
    _, bw = cv2.threshold(L2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    out = np.zeros_like(img_bgr, dtype=np.uint8)
    out[bw == 0] = PALETTE_2[0]
    out[bw == 255] = PALETTE_2[1]

    return out, L2


def ink_edges(toned_bgr, lum_8u):
    """
    Optional ultra-thin silhouette/feature edge.
    If EDGE_THICKNESS == 0 or EDGE_STRENGTH == 0 -> returns toned_bgr unchanged.
    """
    if EDGE_THICKNESS <= 0 or EDGE_STRENGTH <= 0:
        return toned_bgr

    grad = cv2.morphologyEx(lum_8u, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edges = cv2.threshold(grad, int(EDGE_THRESH), 255, cv2.THRESH_BINARY)[1]

    # 1px-ish outline (thickness=1 dilates a touch)
    if EDGE_THICKNESS > 0:
        k = 2 * int(EDGE_THICKNESS) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges = cv2.dilate(edges, kernel, iterations=1)

    edge_mask = (edges.astype(np.float32) / 255.0)[..., None]
    out = toned_bgr.astype(np.float32) * (1.0 - float(EDGE_STRENGTH) * edge_mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def brutalist_postfx(img_bgr, seed=None):
    """
    Brutalist, realistic OpenCV look:
      - tone map to limited palette
      - ink edges (black)
      - matte grain + vignette
      - optional gate weave (kept off by default for 'realistic')
    """
    h, w = img_bgr.shape[:2]

    # optional gate weave
    if GATE_WEAVE_PX > 0:
        rng = np.random.default_rng(seed)
        dx = int(rng.integers(-GATE_WEAVE_PX, GATE_WEAVE_PX + 1))
        dy = int(rng.integers(-GATE_WEAVE_PX, GATE_WEAVE_PX + 1))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img_bgr = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    toned, lum = tone_map_brutalist(img_bgr, levels=int(TONE_LEVELS))
    inked = ink_edges(toned, lum)
    inked = apply_vignette(inked, strength=float(VIGNETTE))
    inked = add_grain(inked, amount=float(GRAIN_AMOUNT), seed=seed)
    return inked

# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dir(OUTPUT_DIR)

    input_files = list_images(INPUT_DIR)
    wing_files = list_images(WINGS_DIR)

    if not input_files:
        raise RuntimeError(f"No input images found in {INPUT_DIR}")
    if not wing_files:
        raise RuntimeError(f"No wing images found in {WINGS_DIR}")

    wings = [read_rgba_wings(p) for p in wing_files]

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    seg = None
    if USE_SEGMENTATION:
        mp_selfie = mp.solutions.selfie_segmentation
        seg = mp_selfie.SelfieSegmentation(model_selection=SEG_MODEL_SELECTION)

    last_pose = None  # (attach_xy, angle_deg, target_w)

    # Animation state (for time posterize / holds)
    last_stylized = None
    hold_countdown = 0
    hold = max(1, int(round(SOURCE_FPS / float(TARGET_FPS))))

    for idx, path in enumerate(input_files):
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Skipping unreadable frame: {path}")
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Pose ---
        res_pose = pose.process(rgb)
        have_pose = res_pose.pose_landmarks is not None

        if have_pose:
            lm = res_pose.pose_landmarks.landmark

            def pt(name):
                i = getattr(mp_pose.PoseLandmark, name).value
                return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

            L_sh = pt("LEFT_SHOULDER")
            R_sh = pt("RIGHT_SHOULDER")
            L_hip = pt("LEFT_HIP")
            R_hip = pt("RIGHT_HIP")

            mid_sh = (L_sh + R_sh) * 0.5
            mid_hip = (L_hip + R_hip) * 0.5
            attach = lerp(mid_sh, mid_hip, TORSO_ATTACH_T)

            # Shoulder width for scaling
            shoulder_vec = (R_sh - L_sh)
            shoulder_dist = float(np.linalg.norm(shoulder_vec))
            target_w = max(10.0, shoulder_dist * WING_WIDTH_MULTIPLIER)

            # rotation choice
            if WING_ROTATION_MODE == "shoulders":
                angle = float(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])))
            elif WING_ROTATION_MODE == "torso_clamped":
                torso_vec = (mid_hip - mid_sh)
                # angle relative to vertical (lean left/right)
                # atan2(dx, dy) gives tilt; clamp
                tilt = float(np.degrees(np.arctan2(torso_vec[0], torso_vec[1] + 1e-6)))
                angle = float(np.clip(tilt, -TORSO_CLAMP_DEG, TORSO_CLAMP_DEG))
            else:
                angle = 0.0  # "none"

            # smoothing
            if last_pose is None:
                sm_attach, sm_angle, sm_target_w = attach, angle, target_w
            else:
                prev_attach, prev_angle, prev_target_w = last_pose
                sm_attach = lerp(prev_attach, attach, 1.0 - SMOOTHING)
                sm_angle = lerp(prev_angle, angle, 1.0 - SMOOTHING)
                sm_target_w = lerp(prev_target_w, target_w, 1.0 - SMOOTHING)

            last_pose = (sm_attach, sm_angle, sm_target_w)

        elif REUSE_LAST_ON_MISS and last_pose is not None:
            sm_attach, sm_angle, sm_target_w = last_pose
        else:
            out_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
            cv2.imwrite(out_path, frame)
            continue

        # --- Segmentation mask (subject) ---
        subj_mask = None
        if seg is not None:
            res_seg = seg.process(rgb)
            if res_seg.segmentation_mask is not None:
                subj_mask = soft_subject_mask(res_seg.segmentation_mask, SEG_BLUR_SIGMA)

        # --- Choose wings frame ---
        wing = wings[idx % len(wings)]

        # Rotate/scale wings (rotation fixed by mode above)
        wing_rs = rotate_and_scale_rgba(wing, sm_angle, sm_target_w)

        wh, ww = wing_rs.shape[:2]
        top_left_x = int(round(sm_attach[0] - ww / 2.0 + WING_OFFSET_X))
        top_left_y = int(round(sm_attach[1] - wh / 2.0 + WING_OFFSET_Y))

        # Composite wings first
        out = frame.copy()
        out = alpha_composite(out, wing_rs, top_left_x, top_left_y)

        # Put subject back on top (wings behind body)
        if subj_mask is not None:
            out_f = out.astype(np.float32)
            frame_f = frame.astype(np.float32)
            out = (out_f * (1.0 - subj_mask) + frame_f * subj_mask).astype(np.uint8)

        # ----------------------------
        # POST + TIME POSTERIZE / HOLDS
        # ----------------------------
        if not ENABLE_POST:
            out_to_write = out
        else:
            # motion score for impact holds (pose-based, stable)
            motion_score = 0.0
            if have_pose and last_pose is not None:
                # compare current smoothed pose with last saved pose estimate
                prev_attach, prev_angle, prev_target_w = last_pose
                motion_score = 0.25 * abs(sm_angle - prev_angle)  # angle is mostly 0 now
                motion_score += 0.02 * abs(sm_target_w - prev_target_w)

            # update decision: on cadence, or if countdown finished, or if first frame
            do_update = (idx % hold == 0) or (hold_countdown <= 0) or (last_stylized is None)

            # trigger hold burst on motion spikes
            if ENABLE_IMPACT_HOLDS and motion_score > IMPACT_MOTION_THRESH:
                hold_countdown = hold + IMPACT_HOLD_EXTRA
            else:
                hold_countdown = max(hold_countdown - 1, 0)

            if do_update:
                stylized = brutalist_postfx(out, seed=idx)
                last_stylized = stylized.copy()

            out_to_write = last_stylized if last_stylized is not None else out

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
        cv2.imwrite(out_path, out_to_write)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(input_files)}")

    pose.close()
    if seg is not None:
        seg.close()

    print("Done. Output in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
