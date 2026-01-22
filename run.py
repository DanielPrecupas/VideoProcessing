# run.py
from __future__ import annotations

import os
import shutil
import argparse
import urllib.request
from typing import Optional, List, Dict, Any

from src.download import is_url, download_to_mp4
from src.extract_frames import ExtractFramesConfig, extract_frames
from src.combine import CombineConfig, OverlayConfig, combine_frames
from src.postfx import PostFXConfig, apply_postfx_folder
from src.rig import ATTACH_POINTS

ROOT = os.path.dirname(os.path.abspath(__file__))

MEDIA_DIR = os.path.join(ROOT, "media")
WORK_DIR = os.path.join(ROOT, "work")

SUBJECT_MP4 = os.path.join(MEDIA_DIR, "subject.mp4")
BG_MP4 = os.path.join(MEDIA_DIR, "bg.mp4")
FG_MP4 = os.path.join(MEDIA_DIR, "fg.mp4")

INPUT_FRAMES = os.path.join(WORK_DIR, "input_frames")
BG_FRAMES = os.path.join(WORK_DIR, "bg_frames")
FG_FRAMES = os.path.join(WORK_DIR, "fg_frames")

OUT_RAW = os.path.join(WORK_DIR, "output_frames_raw")
OUT_FINAL = os.path.join(WORK_DIR, "output_frames")

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def wipe_dir(p: str) -> None:
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    os.makedirs(p, exist_ok=True)

def next_numbered_mp4(out_dir: str, base: str = "output", digits: int = 4) -> str:
    os.makedirs(out_dir, exist_ok=True)
    i = 0
    while True:
        path = os.path.join(out_dir, f"{base}_{i:0{digits}d}.mp4")
        if not os.path.exists(path):
            return path
        i += 1


def resolve_numbered_out_path(requested_path: str) -> str:
    """
    If the user leaves default work/output.mp4 (or work/output_XXXX.mp4), auto-pick next numbered output.
    If they pass a custom filename/path, we respect it.
    """
    # Normalize
    req = os.path.normpath(requested_path)

    work_dir_norm = os.path.normpath(WORK_DIR)
    default_legacy = os.path.normpath(os.path.join(WORK_DIR, "output.mp4"))

    # If user didn't change the default legacy name, use numbered.
    if req == default_legacy:
        return next_numbered_mp4(WORK_DIR, base="output", digits=4)

    # If they explicitly set something like work/output_0007.mp4, keep it.
    # Otherwise, keep their custom path.
    return requested_path



def resolve_source_to_mp4(
    arg_value: Optional[str],
    default_path: str,
    label: str,
    *,
    allow_cached_default: bool,
) -> Optional[str]:
    """
    - URL -> download to default_path (overwrite), return default_path
    - local file -> return it
    - omitted -> return default_path ONLY if allow_cached_default and file exists; else None
    """
    ensure_dir(MEDIA_DIR)

    if arg_value is None:
        if allow_cached_default and os.path.isfile(default_path):
            return default_path
        return None

    if is_url(arg_value):
        print(f"[download] {label}: {arg_value}")
        download_to_mp4(arg_value, default_path, overwrite=True)
        return default_path

    if not os.path.isfile(arg_value):
        raise FileNotFoundError(f"{label} file not found: {arg_value}")
    return arg_value


def add_extract_args(ap: argparse.ArgumentParser, prefix: str, *, defaults: dict):
    """
    Adds flags like:
      --subject_fps, --subject_start, --subject_left, --subject_bg, ...
    """
    p = prefix

    ap.add_argument(f"--{p}_fps", type=float, default=defaults.get("fps", 10.0))
    ap.add_argument(f"--{p}_start", type=float, default=defaults.get("start", 0.0))
    ap.add_argument(f"--{p}_end", type=float, default=defaults.get("end", None))

    ap.add_argument(f"--{p}_top", type=float, default=defaults.get("top", 0.0))
    ap.add_argument(f"--{p}_left", type=float, default=defaults.get("left", 0.0))
    ap.add_argument(f"--{p}_right", type=float, default=defaults.get("right", 0.0))
    ap.add_argument(f"--{p}_bottom", type=float, default=defaults.get("bottom", 0.0))

    ap.add_argument(f"--{p}_resize_w", type=int, default=defaults.get("resize_w", None))
    ap.add_argument(f"--{p}_resize_h", type=int, default=defaults.get("resize_h", None))

    # keying (optional; mostly for bg/fg)
    ap.add_argument(
        f"--{p}_bg",
        type=str,
        default=defaults.get("bg", "none"),
        help=f"{p} key: none|auto|green|blue|black|white|#RRGGBB",
    )
    ap.add_argument(f"--{p}_key_thr", type=float, default=defaults.get("key_thr", 22.0))
    ap.add_argument(f"--{p}_key_soft", type=float, default=defaults.get("key_soft", 35.0))
    ap.add_argument(f"--{p}_border_pct", type=float, default=defaults.get("border_pct", 3.0))
    ap.add_argument(f"--{p}_trim", action="store_true", default=defaults.get("trim", False))
    ap.add_argument(f"--{p}_trim_pad", type=int, default=defaults.get("trim_pad", 0))


def build_extract_cfg(args, prefix: str) -> ExtractFramesConfig:
    p = prefix
    return ExtractFramesConfig(
        target_fps=getattr(args, f"{p}_fps"),
        start_sec=getattr(args, f"{p}_start"),
        end_sec=getattr(args, f"{p}_end"),
        crop_top=getattr(args, f"{p}_top"),
        crop_left=getattr(args, f"{p}_left"),
        crop_right=getattr(args, f"{p}_right"),
        crop_bottom=getattr(args, f"{p}_bottom"),
        resize_w=getattr(args, f"{p}_resize_w"),
        resize_h=getattr(args, f"{p}_resize_h"),
        prefix="frame_",
        ext="png",
        bg=getattr(args, f"{p}_bg"),
        key_thr=getattr(args, f"{p}_key_thr"),
        key_soft=getattr(args, f"{p}_key_soft"),
        border_pct=getattr(args, f"{p}_border_pct"),
        trim=getattr(args, f"{p}_trim"),
        trim_pad=getattr(args, f"{p}_trim_pad"),
        clear_out_dir=True,
    )


def hex_to_bgr(h: str):
    s = h.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError(f"Expected #RRGGBB, got: {h}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def frames_to_mp4(frames_dir: str, out_path: str, fps: float) -> None:
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("OpenCV (cv2) is required to render video. Install opencv-python.") from e

    if not os.path.isdir(frames_dir):
        raise RuntimeError(f"frames_dir not found: {frames_dir}")

    frames = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not frames:
        raise RuntimeError(f"No frames found in: {frames_dir}")

    first = cv2.imread(os.path.join(frames_dir, frames[0]))
    if first is None:
        raise RuntimeError("Failed to read first frame")

    h, w = first.shape[:2]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")

    for name in frames:
        path = os.path.join(frames_dir, name)
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        if img.shape[1] != w or img.shape[0] != h:
            raise RuntimeError(
                f"Frame size mismatch at {name}: expected {(w,h)}, got {(img.shape[1], img.shape[0])}"
            )
        writer.write(img)

    writer.release()
    print(f"[video] wrote {len(frames)} frames -> {out_path} @ {fps} fps")


def _is_image(s: str) -> bool:
    return s.lower().endswith(IMG_EXTS)


def _download_file(url: str, out_path: str) -> str:
    ensure_dir(os.path.dirname(out_path) or ".")
    urllib.request.urlretrieve(url, out_path)
    return out_path


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "y", "on")


def parse_overlay_spec(spec: str) -> Dict[str, Any]:
    """
    Format:
      SRC|k=v|k=v|...
    Example:
      "wings.mp4|layer=behind|attach=shoulders_mid|scale=5|pivot=0.5,0.65|fps=10|start=6|bg=auto|trim=1|pad=8"
      "mask.png|layer=front|attach=head|scale=1.2|pivot=0.5,0.5|offset_y=-10"
    """
    parts = [p.strip() for p in spec.split("|") if p.strip()]
    if not parts:
        raise ValueError("Empty --overlay spec")

    src = parts[0]
    kv: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" not in p:
            raise ValueError(f"Bad overlay token (expected k=v): {p}")
        k, v = p.split("=", 1)
        kv[k.strip().lower()] = v.strip()

    out: Dict[str, Any] = {
        "src": src,
        "layer": kv.get("layer", "behind"),
        # subject-timeline visibility (frame indices in SUBJECT frames)
        "on_sec": float(kv.get("on", kv.get("on_sec", "0"))),
        "off_sec": None if kv.get("off", "").lower() in ("", "none") else float(kv["off"]),
        "loop": _parse_bool(kv.get("loop", "1")),
        "attach": kv.get("attach", "chest"),
        "scale": float(kv.get("scale", "1.5")),
        "offset_x": int(kv.get("offset_x", "0")),
        "offset_y": int(kv.get("offset_y", "0")),
        "rotation": kv.get("rotation", "none"),
        "pivot_x": float(kv.get("pivot_x", "0.5")),
        "pivot_y": float(kv.get("pivot_y", "0.5")),
        # extraction options for video overlays
        "fps": float(kv.get("fps", "10")),
        "start": float(kv.get("start", "0")),
        "end": None if kv.get("end", "").lower() in ("", "none") else float(kv["end"]),
        "bg": kv.get("bg", "auto"),
        "trim": _parse_bool(kv.get("trim", "0")),
        "pad": int(kv.get("pad", kv.get("trim_pad", "0"))),
        "crop_left": float(kv.get("left", "0")),
        "crop_right": float(kv.get("right", "0")),
        "crop_top": float(kv.get("top", "0")),
        "crop_bottom": float(kv.get("bottom", "0"))
    }

    if "pivot" in kv:
        px, py = kv["pivot"].split(",", 1)
        out["pivot_x"] = float(px)
        out["pivot_y"] = float(py)

    # validate early
    if out["layer"] not in ("behind", "front"):
        raise ValueError(f"overlay layer must be behind|front, got: {out['layer']}")
    if out["attach"] not in ATTACH_POINTS:
        raise ValueError(f"overlay attach must be one of {ATTACH_POINTS}, got: {out['attach']}")
    if not (0.0 <= out["pivot_x"] <= 1.0 and 0.0 <= out["pivot_y"] <= 1.0):
        raise ValueError(f"overlay pivot must be in 0..1, got: {out['pivot_x']},{out['pivot_y']}")

    return out


def main():
    ap = argparse.ArgumentParser(description="Video -> frames -> combine -> postfx pipeline")

    # Sources
    ap.add_argument("--subject", type=str, default=None, help="URL or path. If omitted, uses media/subject.mp4")
    ap.add_argument("--bg", type=str, default=None, help="URL or path. If omitted, bg is NONE by default")
    ap.add_argument("--fg", type=str, default=None, help="URL or path. If omitted, fg is NONE by default")

    # Repeatable multi-overlay spec
    ap.add_argument(
        "--overlay",
        action="append",
        default=[],
        help='Repeatable. Format: "URL|layer=behind|attach=chest|scale=4.0|fps=10|start=0|end=none|bg=green|trim=1|pad=8|left=0|right=0|top=0|bottom=0|pivot=0.5,0.5|rotation=shoulders|offset_x=0|offset_y=0|on=0|off=none|loop=1"'
    )

    # Explicitly opt into cached bg/fg when args omitted
    ap.add_argument("--use_cached_bg", action="store_true", help="If --bg is omitted, use media/bg.mp4 if present")
    ap.add_argument("--use_cached_fg", action="store_true", help="If --fg is omitted, use media/fg.mp4 if present")

    # Render mp4
    ap.add_argument("--render_video", action="store_true", help="Render OUT_FINAL frames into an mp4")
    ap.add_argument(
        "--out_video",
        type=str,
        default=None,
        help="Output mp4 path. If omitted, auto-generates output_0000.mp4, output_0001.mp4, ... in work/",
    )
    ap.add_argument("--out_video_fps", type=float, default=None, help="MP4 FPS (defaults to --subject_fps)")

    # Per-source extraction flags
    add_extract_args(ap, "subject", defaults=dict(fps=10.0, start=0.0, bg="none"))
    add_extract_args(ap, "bg", defaults=dict(fps=10.0, start=0.0, bg="none"))
    add_extract_args(ap, "fg", defaults=dict(fps=10.0, start=0.0, bg="auto"))  # fg defaults to auto key

    # Combine options (single overlay legacy)
    ap.add_argument("--fg_layer", choices=["behind", "front"], default="behind")
    ap.add_argument("--attach", choices=ATTACH_POINTS, default="chest")
    ap.add_argument("--fg_scale", type=float, default=3.0)
    ap.add_argument("--fg_offset_x", type=int, default=0)
    ap.add_argument("--fg_offset_y", type=int, default=0)
    ap.add_argument("--fg_pivot_x", type=float, default=0.5, help="Overlay pivot X (0..1). 0=left, 1=right")
    ap.add_argument("--fg_pivot_y", type=float, default=0.5, help="Overlay pivot Y (0..1). 0=top, 1=bottom")
    ap.add_argument("--fg_rotation", choices=["none", "shoulders"], default="none")

    # Subject occlusion
    ap.add_argument("--seg", action="store_true", help="Enable subject segmentation occlusion")
    ap.add_argument("--no_seg", action="store_true", help="Disable subject segmentation occlusion")
    ap.add_argument("--seg_blur", type=float, default=2.5)

    ap.add_argument("--smoothing", type=float, default=0.6)
    ap.add_argument("--no_reuse_last", action="store_true")

    # Background fit mode
    ap.add_argument(
        "--bg_fit",
        choices=["stretch", "cover", "contain"],
        default="cover",
        help="How to fit bg frames to subject size. stretch=warp, cover=no warp (crop), contain=no warp (letterbox).",
    )

    # PostFX
    ap.add_argument("--postfx", action="store_true")
    ap.add_argument("--ink", type=str, default="#0A0A0A")
    ap.add_argument("--paper", type=str, default="#F5F5F5")
    ap.add_argument("--outline", action="store_true")
    ap.add_argument("--outline_strength", type=float, default=0.25)
    ap.add_argument("--outline_thresh", type=int, default=35)
    ap.add_argument("--grain", type=float, default=0.0)
    ap.add_argument("--vignette", type=float, default=0.0)

    ap.add_argument("--time_posterize", action="store_true")
    ap.add_argument("--source_fps", type=float, default=30.0)
    ap.add_argument("--target_fps", type=float, default=10.0)

    args = ap.parse_args()

    # Resolve sources (download if URL)
    subject_path = resolve_source_to_mp4(args.subject, SUBJECT_MP4, "subject", allow_cached_default=True)
    if subject_path is None:
        raise RuntimeError("No subject video provided and media/subject.mp4 not found.")

    bg_path = resolve_source_to_mp4(args.bg, BG_MP4, "bg", allow_cached_default=bool(args.use_cached_bg))
    fg_path = resolve_source_to_mp4(args.fg, FG_MP4, "fg", allow_cached_default=bool(args.use_cached_fg))

    # Rebuild work dirs
    ensure_dir(WORK_DIR)
    wipe_dir(INPUT_FRAMES)
    wipe_dir(BG_FRAMES)
    wipe_dir(FG_FRAMES)
    wipe_dir(OUT_RAW)
    wipe_dir(OUT_FINAL)

    # Also wipe per-overlay dirs (if any existed from previous runs)
    for d in os.listdir(WORK_DIR):
        if d.startswith("ov_") and d.endswith("_frames"):
            wipe_dir(os.path.join(WORK_DIR, d))

    # Extract subject
    subj_cfg = build_extract_cfg(args, "subject")
    n_subj = extract_frames(subject_path, INPUT_FRAMES, subj_cfg)
    print(f"[extract] subject: {n_subj} frames -> {INPUT_FRAMES}")

    # Extract bg (optional)
    n_bg = 0
    if bg_path is not None:
        bg_cfg = build_extract_cfg(args, "bg")
        bg_cfg.max_frames = n_subj
        n_bg = extract_frames(bg_path, BG_FRAMES, bg_cfg)
        print(f"[extract] bg: {n_bg} frames -> {BG_FRAMES}")
    else:
        print("[extract] bg: none")

    # Multi overlays
    overlays: List[OverlayConfig] = []
    if args.overlay:
        for i, spec in enumerate(args.overlay):
            ov = parse_overlay_spec(spec)
            ov_dir = os.path.join(WORK_DIR, f"ov_{i:02d}_frames")
            wipe_dir(ov_dir)

            src = ov["src"]

            # resolve source (download if URL)
            if is_url(src):
                if _is_image(src):
                    local_img = os.path.join(MEDIA_DIR, f"ov_{i:02d}.png")
                    print(f"[download] overlay{i}: {src}")
                    src_path = _download_file(src, local_img)
                else:
                    local_mp4 = os.path.join(MEDIA_DIR, f"ov_{i:02d}.mp4")
                    print(f"[download] overlay{i}: {src}")
                    download_to_mp4(src, local_mp4, overwrite=True)
                    src_path = local_mp4
            else:
                if not os.path.isfile(src):
                    raise FileNotFoundError(f"overlay source not found: {src}")
                src_path = src

            # extract/copy into ov_dir
            if _is_image(src_path):
                dst = os.path.join(ov_dir, "frame_000000.png")
                shutil.copy2(src_path, dst)
                print(f"[extract] overlay{i}: still -> {ov_dir}")
            else:
                cfg = ExtractFramesConfig(
                    max_frames = n_subj,
                    target_fps=ov["fps"],
                    start_sec=ov["start"],
                    end_sec=ov["end"],
                    crop_top=ov["crop_top"],
                    crop_left=ov["crop_left"],
                    crop_right=ov["crop_right"],
                    crop_bottom=ov["crop_bottom"],
                    resize_w=None, resize_h=None,
                    prefix="frame_", ext="png",
                    bg=ov["bg"],
                    key_thr=22.0, key_soft=35.0, border_pct=3.0,
                    trim=ov["trim"],
                    trim_pad=ov["pad"],
                    clear_out_dir=True,
                )
                n = extract_frames(src_path, ov_dir, cfg)
                print(f"[extract] overlay{i}: {n} frames -> {ov_dir}")
            
            subj_fps = float(args.subject_fps)            
            on_f = int(round(float(ov.get("on_sec", 0.0)) * subj_fps))
            off_f = None
            if ov.get("off_sec", None) is not None:
                off_f = int(round(float(ov["off_sec"]) * subj_fps))

            overlays.append(
                OverlayConfig(
                    frames_dir=ov_dir,
                    layer=ov["layer"],
                    subject_on=on_f,
                    subject_off=off_f,
                    loop=ov["loop"],
                    attach=ov["attach"],
                    width_multiplier=ov["scale"],
                    offset_x=ov["offset_x"],
                    offset_y=ov["offset_y"],
                    pivot_x=ov["pivot_x"],
                    pivot_y=ov["pivot_y"],
                    rotation=ov["rotation"],
                )
            )

    # Extract fg (legacy single overlay) ONLY if no --overlay used
    n_fg = 0
    overlay_cfg = None
    if not args.overlay:
        if fg_path is not None:
            fg_cfg = build_extract_cfg(args, "fg")
            fg_cfg.max_frames = n_subj
            n_fg = extract_frames(fg_path, FG_FRAMES, fg_cfg)
            print(f"[extract] fg: {n_fg} frames -> {FG_FRAMES}")
        else:
            print("[extract] fg: none")

        if n_fg > 0:
            overlay_cfg = OverlayConfig(
                frames_dir=FG_FRAMES,
                layer=args.fg_layer,
                attach=args.attach,
                width_multiplier=args.fg_scale,
                offset_x=args.fg_offset_x,
                offset_y=args.fg_offset_y,
                pivot_x=args.fg_pivot_x,
                pivot_y=args.fg_pivot_y,
                rotation=args.fg_rotation,
            )
    else:
        # keep logs clear
        print(f"[extract] fg: skipped (using {len(overlays)} overlays)")

    # Combine
    use_seg = True
    if args.no_seg:
        use_seg = False
    elif args.seg:
        use_seg = True

    comb_cfg = CombineConfig(
        subject_dir=INPUT_FRAMES,
        bg_dir=BG_FRAMES if n_bg > 0 else None,
        bg_fit=args.bg_fit,
        overlay=overlay_cfg,          # legacy path (combine.py can append it)
        output_dir=OUT_RAW,
        use_segmentation=use_seg,
        seg_blur_sigma=args.seg_blur,
        smoothing=args.smoothing,
        reuse_last_on_miss=(not args.no_reuse_last),
    )
    setattr(comb_cfg, "bg_fit", args.bg_fit)

    # NEW multi-overlay list
    setattr(comb_cfg, "overlays", overlays)

    combine_frames(comb_cfg)
    print(f"[combine] wrote raw composites -> {OUT_RAW}")

    # PostFX
    if args.postfx:
        fx_cfg = PostFXConfig(
            enable=True,
            ink_bgr=hex_to_bgr(args.ink),
            paper_bgr=hex_to_bgr(args.paper),
            outline_enable=args.outline,
            outline_strength=args.outline_strength,
            outline_thresh=args.outline_thresh,
            outline_thickness=1,
            grain_amount=args.grain,
            vignette=args.vignette,
            time_posterize=args.time_posterize,
            source_fps=args.source_fps,
            target_fps=args.target_fps,
            out_ext="png",
        )
        n_out = apply_postfx_folder(OUT_RAW, OUT_FINAL, fx_cfg, clear_out_dir=True)
        print(f"[postfx] wrote {n_out} frames -> {OUT_FINAL}")
    else:
        for name in os.listdir(OUT_RAW):
            shutil.copy2(os.path.join(OUT_RAW, name), os.path.join(OUT_FINAL, name))
        print(f"[postfx] disabled; copied raw -> {OUT_FINAL}")

    out_path = None

    if args.render_video:
        fps = args.out_video_fps if args.out_video_fps is not None else float(args.subject_fps)

        if args.out_video is None:
            # DEFAULT BEHAVIOR: always auto-number
            out_path = next_numbered_mp4(WORK_DIR, base="output", digits=4)
        else:
            # User explicitly provided a path → respect it
            out_path = args.out_video

        frames_to_mp4(OUT_FINAL, out_path, fps)
        print(f"[video] final output path: {out_path}")

    print("\nDone.")
    print(f"Final frames: {OUT_FINAL}")
    if args.render_video:
        print(f"Final video:  {out_path}")


if __name__ == "__main__":
    main()
