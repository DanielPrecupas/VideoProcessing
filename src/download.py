# src/download.py
"""
URL -> MP4 downloader (yt-dlp wrapper).

Usage model for run.py:
- If a CLI arg looks like a URL (http/https), call download_to_mp4(url, "media/subject.mp4")
- Otherwise treat it as a local file path.

Requirements:
- yt-dlp installed (pip install yt-dlp)
- ffmpeg installed and available on PATH (recommended; needed for best merge quality)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def is_url(s: str) -> bool:
    return bool(s) and bool(_URL_RE.match(s.strip()))


@dataclass
class DownloadResult:
    out_path: str
    used_ffmpeg: bool
    command: list[str]


class DownloadError(RuntimeError):
    pass


def _which_or_raise(exe: str) -> str:
    p = shutil.which(exe)
    if not p:
        raise DownloadError(
            f"Required executable not found on PATH: {exe}\n"
            f"Install it (or add it to PATH) and try again."
        )
    return p


def download_to_mp4(
    url: str,
    out_path: str,
    *,
    overwrite: bool = True,
    no_playlist: bool = True,
    cookies: Optional[str] = None,
    user_agent: Optional[str] = None,
    quiet: bool = False,
) -> DownloadResult:
    """
    Download a URL to a single MP4 file at out_path.

    - Overwrites out_path if overwrite=True
    - Forces merge output format MP4
    - Uses bestvideo+bestaudio when possible, else best

    Raises DownloadError on failure.
    """
    if not is_url(url):
        raise DownloadError(f"Not a URL: {url}")

    yt_dlp = _which_or_raise("yt-dlp")

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # If user passes e.g. media/subject.mp4, we want EXACTLY that path.
    # yt-dlp will write exactly -o out_path, and --merge-output-format mp4 ensures mp4 container.
    cmd = [
        yt_dlp,
        "-f", "bv*+ba/b",                 # best video+audio, fallback best
        "--merge-output-format", "mp4",   # force mp4 container
        "-o", out_path,                   # exact output path
    ]

    if overwrite:
        cmd.append("--force-overwrites")

    if no_playlist:
        cmd.append("--no-playlist")

    # Optional extras
    if cookies:
        cmd += ["--cookies", cookies]
    if user_agent:
        cmd += ["--user-agent", user_agent]

    # Logging
    if quiet:
        cmd += ["--quiet", "--no-warnings"]
    else:
        cmd += ["--newline"]

    cmd.append(url)

    # ffmpeg presence check (yt-dlp can still download without it, but merges may be limited)
    used_ffmpeg = bool(shutil.which("ffmpeg"))

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE if quiet else None,
            stderr=subprocess.PIPE if quiet else None,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise DownloadError(str(e)) from e

    if proc.returncode != 0:
        msg = "yt-dlp failed"
        if quiet:
            msg += f"\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        raise DownloadError(msg)

    if not os.path.isfile(out_path):
        # Sometimes yt-dlp may decide a different extension; we are forcing mp4,
        # but this is a sanity check.
        raise DownloadError(f"Download finished but output file not found: {out_path}")

    return DownloadResult(out_path=out_path, used_ffmpeg=used_ffmpeg, command=cmd)


# ----------------------------
# Optional CLI for manual testing
# ----------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Download a URL to an MP4 using yt-dlp.")
    ap.add_argument("url", help="Video URL (http/https)")
    ap.add_argument("--out", required=True, help="Output path, e.g. media/subject.mp4")
    ap.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing file")
    ap.add_argument("--cookies", default=None, help="Path to cookies.txt (optional)")
    ap.add_argument("--ua", default=None, help="User-Agent string (optional)")
    ap.add_argument("--quiet", action="store_true", help="Quiet mode (captures stdout/stderr)")

    args = ap.parse_args()

    res = download_to_mp4(
        args.url,
        args.out,
        overwrite=not args.no_overwrite,
        cookies=args.cookies,
        user_agent=args.ua,
        quiet=args.quiet,
    )
    print(f"Saved: {res.out_path}")
    print(f"ffmpeg detected: {res.used_ffmpeg}")


if __name__ == "__main__":
    main()
