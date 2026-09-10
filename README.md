# VideoProcessing

A small video compositing pipeline built with OpenCV and MediaPipe. It takes a video of a person, tracks their body pose frame by frame, and lets you attach overlays (wings, props, masks, anything with transparency) that follow their movement. You can also drop in a background and a foreground layer, and run the whole thing through a "brutalist" post effect pass: posterized tones, ink edges, grain, and vignette, for a stylized comic or motion-graphic look.

## Backstory

This started as a college project for a computer vision course. I got a bit carried away with it after the semester ended and kept building on it for fun, adding the overlay system, the pose-driven attachment points, the stylized post-processing, and a small GUI so I didn't have to fight the command line every time. What you see here is the personal, post-college version of that assignment.

## What it does

1. Takes a subject video (a person walking, dancing, whatever) and optionally a background video and one or more overlay assets.
2. Extracts frames and runs pose detection (via MediaPipe) on the subject so it knows where the shoulders, chest, hips, hands, and so on are in every frame.
3. Attaches your overlay to a chosen body point (shoulders, chest, back, hands, etc), scales and rotates it to follow the body, and composites it either behind or in front of the subject.
4. Optionally applies subject segmentation so overlays can sit "behind" the person realistically.
5. Optionally runs a stylized post effect pass: limited color palette, ink outlines, grain, vignette, and time posterization for a stop-motion feel.
6. Renders the final frames back into an mp4.

There's also a standalone `main.py` script, an earlier and simpler version of the pipeline that does the wings-on-a-person effect in one file, useful for reading if you want to understand the core idea before diving into the full pipeline in `run.py` and `src/`.

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) installed and available on your PATH (used for video merging when downloading sources)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) if you want to pull source videos directly from a URL instead of a local file

Install the Python dependencies:

```bash
pip install opencv-python numpy mediapipe PySide6 yt-dlp
```

## Usage

### Basic run

Drop your subject video at `media/subject.mp4`, then run:

```bash
python run.py --render_video
```

This extracts frames, runs pose tracking, composites everything, and writes the final video to `work/output_0000.mp4` (it auto-numbers so you never overwrite a previous run).

### Adding a background

```bash
python run.py --bg path/to/background.mp4 --bg_fit cover --render_video
```

### Adding an overlay (like wings) that follows the body

```bash
python run.py --overlay "media/wings.mp4|layer=behind|attach=shoulders_mid|scale=4|bg=auto|trim=1" --render_video
```

The overlay spec is a small pipe-separated mini language: `source|key=value|key=value|...`. Useful keys:

- `layer`: `behind` or `front` relative to the subject
- `attach`: where on the body to anchor it, for example `chest`, `shoulders_mid`, `head`, `left_hand`, `back`
- `scale`: size relative to shoulder width
- `bg`: how to key out the overlay's background, `none`, `auto`, `green`, `black`, or a hex color like `#00FF00`
- `on` / `off`: seconds into the subject video when the overlay should appear/disappear
- `rotation`: `none` or `shoulders` (follow shoulder tilt)

You can pass `--overlay` more than once to stack several assets.

### Sourcing videos from a URL

Any `--subject`, `--bg`, `--fg`, or overlay source can be a URL instead of a local path. It gets downloaded with yt-dlp and cached under `media/`:

```bash
python run.py --subject "https://example.com/clip.mp4" --render_video
```

### Stylized post effects

```bash
python run.py --postfx --outline --grain 0.03 --vignette 0.2 --time_posterize --render_video
```

### Using the GUI

If you'd rather not remember flags, there's a PySide6 GUI that wraps the same pipeline:

```bash
python gui.py
```

It lets you pick your subject/background/overlay files, tweak the same options through form fields, and kicks off `run.py` for you in the background.

## Project layout

```
main.py              standalone single-file version of the effect (wings only, good starting point to read)
run.py                full CLI pipeline: extract -> combine -> postfx -> render
gui.py                PySide6 GUI wrapper around run.py
preview_qt.py         preview panel used by the GUI
extract_posterized_frames.py   standalone frame posterization utility
src/
  download.py         yt-dlp wrapper for pulling video from a URL
  extract_frames.py   video -> frame extraction, cropping, resizing, keying
  combine.py           pose tracking + overlay compositing
  compose_utils.py     shared compositing helpers
  postfx.py             the "brutalist" stylization pass
  rig.py                body attach point definitions
  editor.py             misc editing helpers
game/                 a small side experiment, unrelated to the video pipeline
```

## Notes

Generated output, extracted frames, and source media are not tracked in this repo (see `.gitignore`). You'll need to supply your own subject video and any overlay assets, and everything under `work/` gets rebuilt fresh each run.
