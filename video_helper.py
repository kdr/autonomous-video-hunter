"""
Autonomous Video Hunter - Video Helper Utilities

This module provides lightweight utilities for video download and frame extraction.
It handles remote video retrieval and generates uniformly spaced or 1-FPS thumbnail
sequences for analysis workflows.

Key Capabilities:
- HTTP/HTTPS video download with streaming
- Uniform frame extraction (evenly spaced samples)
- 1-FPS frame extraction for dense analysis
- FFmpeg integration for video processing
- Configurable output resolution and quality

Author: @kdr
"""

import os
import uuid
import math
import pathlib
import subprocess
from typing import List, Dict
import requests


class VideoHelper:
    """
    Lightweight utility for:
      • Downloading videos to a local media directory
      • Extracting a fixed number of uniformly spaced thumbnails

    Parameters
    ----------
    media_dir : str | pathlib.Path
        Directory in which all downloaded videos and thumbnails are stored.
        It is created on-demand if it doesn’t already exist.
    """

    def __init__(self, media_dir: str | pathlib.Path = 'media') -> None:
        self.media_dir = pathlib.Path(media_dir).expanduser().resolve()
        self.media_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def download_from_url(self, url: str) -> pathlib.Path:
        """
        Download a remote video (HTTP/HTTPS) and save it as MP4.

        Returns
        -------
        pathlib.Path
            Absolute path of the saved video.
        """
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError("Only HTTP(S) URLs are supported")

        filename = f"{uuid.uuid4()}.mp4"
        out_path = self.media_dir / filename

        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(out_path, "wb") as fp:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    fp.write(chunk)

        return str(out_path.resolve())

    def extract_frames_uniform(
        self,
        video_path: str | pathlib.Path,
        num_frames: int = 3,
        max_width: int = 1024,
    ) -> List[Dict[str, str]]:
        """
        Extract `num_frames` evenly spaced JPEG thumbnails.

        Each output is scaled so the width ≤ `max_width`
        while preserving aspect ratio.  Thumbnails are written to:

            <media_dir>/<uuid>.jpg

        Returns
        -------
        list[dict]
            Each dict contains:
              • 'thumbnail_path' : absolute path of the saved JPEG
              • 'timestamp'      : timestamp (s) from which it was captured
        """
        video = pathlib.Path(video_path)
        if not video.exists():
            raise FileNotFoundError(video)

        duration = self._probe_duration(video)
        if duration == 0:
            raise RuntimeError("Could not determine video duration")

        # Evenly spaced timestamps (avoid first/last frame exactly)
        step = duration / (num_frames + 1)
        timestamps = [step * (i + 1) for i in range(num_frames)]

        results = []
        for ts in timestamps:
            thumb_name = f"{uuid.uuid4()}.jpg"
            thumb_path = self.media_dir / thumb_name

            # -ss seek **before** -i for speed; -frames:v 1 grabs one frame.
            # scale filter keeps aspect ratio, limits width.
            cmd = [
                "ffmpeg",
                "-loglevel", "error",
                "-ss", f"{ts}",
                "-i", str(video),
                "-frames:v", "1",
                "-vf", f"scale='min(iw,{max_width})':-2",
                "-q:v", "2",
                str(thumb_path),
            ]
            subprocess.run(cmd, check=True)

            results.append({
                "thumbnail_path": str(thumb_path.resolve()),
                "timestamp": round(ts, 3),
            })

        return results
    
    # ------------------------------------------------------------------ #
    # 1-FPS thumbnail extraction                                         #
    # ------------------------------------------------------------------ #
    def extract_frames_1fps(
        self,
        video_path: str | pathlib.Path,
        max_width: int = 1024,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> List[Dict[str, str]]:
        """
        Grab thumbnails at exactly 1 frame per second.

        Parameters
        ----------
        video_path : str | pathlib.Path
            Local path to the video.
        max_width : int, default 1024
            Largest allowed width for the JPEGs.
        start_time, end_time : float | None
            Optional trim window (seconds).  If omitted, the whole clip
            is processed.

        Returns
        -------
        list[dict]  Each dict has:
          • 'thumbnail_path' : absolute path of saved JPEG
          • 'timestamp'      : timestamp (s) captured from
        """
        video = pathlib.Path(video_path)
        if not video.exists():
            raise FileNotFoundError(video)

        duration = self._probe_duration(video)
        if duration == 0:
            raise RuntimeError("Could not determine video duration")

        t0 = max(0, start_time or 0)
        t1 = min(duration, end_time) if end_time is not None else duration
        if t0 >= t1:
            raise ValueError("Invalid start/end bounds")

        # Use a temp pattern, then rename → UUID so we know exact timestamps.
        temp_pattern = self.media_dir / f"{uuid.uuid4()}-%06d.jpg"
        vf = f"fps=1,scale='min(iw,{max_width})':-2"

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-ss", str(t0),
            "-to", str(t1),
            "-i", str(video),
            "-vf", vf,
            "-q:v", "2",
            str(temp_pattern),
        ]
        subprocess.run(cmd, check=True)

        # Gather frames and record timestamps.
        # ffmpeg numbers frames starting at 1; timestamp = t0 + (idx-1) seconds
        results: List[Dict[str, str]] = []
        for jpg in sorted(self.media_dir.glob(f"{temp_pattern.stem[:-7]}-*.jpg")):
            # Extract frame index from "...-000123.jpg"
            idx = int(jpg.stem.split("-")[-1])
            ts = t0 + (idx - 1)  # seconds
            # Rename to a clean UUID to avoid leaking order info
            new_name = self.media_dir / f"{uuid.uuid4()}.jpg"
            jpg.rename(new_name)

            results.append({
                "thumbnail_path": str(new_name.resolve()),
                "timestamp": round(ts, 3),
            })

        return results



    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _probe_duration(self, video: pathlib.Path) -> float:
        """Return video duration (seconds) using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ]
        try:
            output = subprocess.check_output(cmd, text=True).strip()
            return float(output)
        except Exception:
            return 0.0

    def get_duration(self, video_path: str | pathlib.Path) -> float:
        """
        Return the duration of a video in **seconds**.

        Parameters
        ----------
        video_path : str | pathlib.Path
            Path to the video file (local).

        Returns
        -------
        float
            Duration in seconds (may be fractional).
        """
        return self._probe_duration(pathlib.Path(video_path))

# ----------------------- Usage example ----------------------- #
# helper = VideoHelper("./media")
# video_file = helper.download_from_url("https://example.com/demo.mp4")
# thumbs = helper.extract_frames_uniform(video_file, num_frames=5)
# print(thumbs)
