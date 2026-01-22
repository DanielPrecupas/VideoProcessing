from __future__ import annotations

import os
import sys
import subprocess
from typing import List


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.path.join(ROOT, "work")


def scene_to_path(scene_number: int, *, base: str = "output", digits: int = 4) -> str:
    """
    User inputs:
      1  -> output_0000.mp4
      13 -> output_0012.mp4
    """
    idx = scene_number - 1
    if idx < 0:
        raise ValueError("Scene numbers must be >= 1")
    return os.path.join(WORK_DIR, f"{base}_{idx:0{digits}d}.mp4")


def next_numbered_mp4(out_dir: str, base: str, digits: int = 4) -> str:
    os.makedirs(out_dir, exist_ok=True)
    i = 0
    while True:
        path = os.path.join(out_dir, f"{base}_{i:0{digits}d}.mp4")
        if not os.path.exists(path):
            return path
        i += 1


def which_ffmpeg() -> str:
    # On Windows, "where"; on mac/linux, "which"
    cmd = ["where", "ffmpeg"] if os.name == "nt" else ["which", "ffmpeg"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.splitlines()[0].strip()
    except Exception:
        pass
    return ""


def write_concat_list(paths: List[str], list_path: str) -> None:
    """
    ffmpeg concat demuxer needs:
      file 'C:\path\to\clip.mp4'
    """
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            ap = os.path.abspath(p).replace("\\", "/")
            f.write(f"file '{ap}'\n")

def normalize_clip(inp: str, outp: str, ffmpeg: str) -> None:
    """
    Re-encode to a "clean" H.264 mp4 with regenerated timestamps.
    This fixes concat issues caused by OpenCV mp4v outputs / weird PTS.
    """
    cmd = [
        ffmpeg,
        "-y",
        "-fflags", "+genpts",
        "-i", inp,
        "-vf", "setsar=1",
        "-an",  # no audio (your scene clips have none)
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        outp,
    ]
    print("[ffmpeg:norm] " + " ".join(cmd))
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0 or not os.path.isfile(outp):
        raise RuntimeError(f"ffmpeg normalize failed for: {inp}")



def concat_movie(inputs: List[str], out_path: str) -> None:
    ffmpeg = which_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg and restart your terminal.")

    if len(inputs) == 1:
        # Just re-encode to a clean output (no audio)
        cmd = [
            ffmpeg, "-y",
            "-i", inputs[0],
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-vf", "setsar=1",
            "-an",
            out_path,
        ]
        print("\n[ffmpeg] " + " ".join(cmd))
        if subprocess.run(cmd, text=True).returncode != 0:
            raise RuntimeError("ffmpeg render failed.")
        return

    # --- Get target W/H from first clip (avoid scale2ref) ---
    try:
        import cv2
    except Exception:
        cv2 = None

    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for editor concat sizing in this mode.")

    best_area = -1
    W = H = 0

    for p in inputs:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        area = w * h
        if w > 0 and h > 0 and area > best_area:
            best_area = area
            W, H = w, h

    if W <= 0 or H <= 0:
        raise RuntimeError("Could not read width/height from any input clip.")


    # --- Build ffmpeg command with concat FILTER (not demuxer) ---
    cmd = [ffmpeg, "-y"]
    for p in inputs:
        cmd += ["-i", p]

    # For each input i:
    # - normalize SAR
    # - scale down to fit WxH (no crop), then pad to WxH
    # - force yuv420p
    chains = []
    labels = []
    for i in range(len(inputs)):
        v = (
            f"[{i}:v]"
            f"setpts=PTS-STARTPTS,"
            f"scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},"
            f"format=yuv420p,"
            f"setsar=1"
            f"[v{i}]"
        )
        chains.append(v)
        labels.append(f"[v{i}]")

    filtergraph = ";".join(chains) + ";" + "".join(labels) + f"concat=n={len(inputs)}:v=1:a=0[v]"

    cmd += [
        "-filter_complex", filtergraph,
        "-map", "[v]",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        out_path,
    ]

    print("\n[ffmpeg] " + " ".join(cmd))
    if subprocess.run(cmd, text=True).returncode != 0:
        raise RuntimeError("ffmpeg concat failed.")


def main() -> int:
    os.makedirs(WORK_DIR, exist_ok=True)

    print("Scene editor")
    print("Type scene numbers in order. Examples:")
    print("  1  -> work/output_0000.mp4")
    print("  13 -> work/output_0012.mp4")
    print("Type 0 to finish and render the movie.\n")

    chosen: List[str] = []

    while True:
        try:
            s = input(f"Scene {len(chosen)+1} number (0 to finish): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1

        if not s:
            continue

        if s == "0":
            break

        if not s.isdigit():
            print("Please enter a number (e.g. 1, 13, 0).")
            continue

        n = int(s)
        if n <= 0:
            print("Scene numbers must be >= 1 (or 0 to finish).")
            continue

        path = scene_to_path(n)
        if not os.path.isfile(path):
            print(f"Not found: {path}")
            continue

        chosen.append(path)
        print(f"Added: {os.path.basename(path)}")

    if not chosen:
        print("No scenes chosen. Nothing to do.")
        return 0

    out_path = next_numbered_mp4(WORK_DIR, base="movie", digits=4)
    print("\nScenes selected:")
    for p in chosen:
        print(" - " + os.path.basename(p))

    print(f"\nRendering movie -> {out_path}")
    concat_movie(chosen, out_path)
    print(f"Done: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
