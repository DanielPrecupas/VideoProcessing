import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames_dir",
        default="work/output_frames",
        help="Directory containing rendered frames"
    )
    parser.add_argument(
        "--out",
        default="output.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Output video FPS"
    )
    args = parser.parse_args()

    frames_dir = args.frames_dir
    out_path = args.out
    fps = args.fps

    frames = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}")

    first = cv2.imread(os.path.join(frames_dir, frames[0]))
    if first is None:
        raise RuntimeError("Failed to read first frame")

    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for name in frames:
        path = os.path.join(frames_dir, name)
        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"Failed to read frame {path}")
        writer.write(frame)

    writer.release()
    print(f"[video] wrote {len(frames)} frames -> {out_path} @ {fps} fps")

if __name__ == "__main__":
    main()
