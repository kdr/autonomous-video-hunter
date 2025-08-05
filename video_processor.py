"""
Autonomous Video Hunter - Video Processing Pipeline

This module provides end-to-end video processing capabilities, combining video helper
utilities with Cloudglue AI analysis. It handles the complete pipeline from video
input to structured analysis results stored in JSONL format.

Processing Pipeline:
1. Video normalization and frame extraction
2. Cloudglue multimodal analysis (transcription, scene description, entity extraction)
3. Duration calculation and metadata extraction
4. Structured result caching in JSONL database
5. Duplicate processing prevention

Author: @kdr
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from video_helper import VideoHelper
from video_understander import VideoUnderstander


class VideoProcessor:
    """
    One-shot “upload ➜ summarize ➜ cache” pipeline built on
    `VideoHelper` and `VideoUnderstander`.

    Parameters
    ----------
    media_dir : str | Path, optional
        Where original media and extracted thumbnails are stored.
        Defaults to "./media".
    db_path : str | Path, optional
        JSONL file that stores one summary dict per processed video.
        Defaults to "<media_dir>/db.jsonl".
    video_understander : VideoUnderstander, optional
        Pre-configured CloudGlue wrapper.  If omitted, you must supply
        `api_key` so the processor can create its own.
    api_key : str, optional
        CloudGlue API key used only when `video_understander` is None.
    """

    def __init__(
        self,
        media_dir: str | Path = "media",
        db_path: str | Path | None = None,
        *,
        video_understander: Optional[VideoUnderstander] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.media_dir = Path(media_dir).expanduser().resolve()
        self.media_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = (
            Path(db_path).expanduser().resolve()
            if db_path is not None
            else self.media_dir / "db.jsonl"
        )
        # Touch the DB file so later `open(..., 'a')` never fails
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.touch(exist_ok=True)

        # Helper & CloudGlue abstraction
        self._vh = VideoHelper(media_dir=self.media_dir)
        if video_understander is not None:
            self._vu = video_understander
        else:
            if api_key is None:
                api_key = os.getenv("CLOUDGLUE_API_KEY")
            if not api_key:
                raise ValueError(
                    "CloudGlue API key missing.  Pass `api_key=` or set "
                    "env var CLOUDGLUE_API_KEY."
                )
            self._vu = VideoUnderstander(api_key=api_key)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def get_processed_output(self, input_video_path: str | Path) -> Dict[str, Any] | None:
        """
        Look up `input_video_path` in the JSONL cache.  Returns the stored
        summary dict on hit, otherwise None.
        """
        norm_path = str(self._normalize_path(input_video_path))
        if not self.db_path.exists():
            return None

        with open(self.db_path, "r", encoding="utf-8") as fp:
            for line in fp:
                try:
                    record = json.loads(line)
                    if record.get("local_video_path") == norm_path:
                        return record
                except json.JSONDecodeError:
                    # skip corrupted line
                    continue
        return None

    def process(self, input_video_path: str | Path, *, save_to_db: bool = False) -> Dict[str, Any]:
        """
        End-to-end processing pipeline.

        1. Checks cache → returns cached result if present.
        2. Extracts 8 uniformly spaced thumbnails to `media_dir`.
        3. Runs CloudGlue describe+extract via `VideoUnderstander`.
        4. Computes duration (seconds).
        5. On success, optionally appends the result as a JSON line to
           `db.jsonl`.

        Returns the consolidated summary dict.
        """
        cached = self.get_processed_output(input_video_path)
        if cached is not None:
            return cached  # short-circuit

        video_path = self._normalize_path(input_video_path)

        # Step-1: thumbnails
        local_frames: List[Dict[str, str]] = self._vh.extract_frames_uniform(
            video_path, num_frames=8
        )

        # Step-2: CloudGlue summary
        summary = self._vu.general_summary(video_path)

        # Step-3: duration
        duration_sec = self._vh.get_duration(video_path)

        # Consolidated record
        record: Dict[str, Any] = {
            "local_video_path": str(video_path),
            "cloudglue_uri": summary.get("cloudglue_uri"),
            "local_frames": local_frames,
            "description": summary.get("description"),
            "has_logo": summary.get("has_logo"),
            "logos": summary.get("logos"),
            "has_face": summary.get("has_face"),
            "has_speech": summary.get("has_speech"),
            "is_outdoors": summary.get("is_outdoors"),
            "has_text_on_screen": summary.get("has_text_on_screen"),
            "duration_seconds": duration_sec,  # handy extra field
        }

        if save_to_db:
            with open(self.db_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_path(path: str | Path) -> Path:
        """Return an absolute, expanded, resolved Path object."""
        return Path(path).expanduser().resolve()
