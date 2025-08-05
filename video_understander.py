"""
Autonomous Video Hunter - Cloudglue Integration Wrapper

This module provides a high-level interface to Cloudglue's multimodal video analysis
capabilities. It handles video upload, parallel processing of transcription and
entity extraction, and structured result formatting for agent consumption.

Cloudglue Capabilities Used:
- Video transcription with scene text extraction
- Visual scene description and summarization
- Entity extraction (faces, logos, text, environmental features)
- Parallel processing for optimal performance

Author: @kdr
"""

from __future__ import annotations
import concurrent.futures as _cf
from pathlib import Path
from typing import Dict, Any

from cloudglue import CloudGlue


class VideoUnderstander:
    """
    High-level wrapper around CloudGlue for single-shot “upload ➜ describe ➜ extract”.

    Parameters
    ----------
    api_key : str
        Your CloudGlue API key.

    Example
    -------
    >>> vu = VideoUnderstander(api_key="cg_live_…")
    >>> summary = vu.general_summary("/path/to/video.mp4")
    >>> summary["description"][:500]  # markdown description
    """

    def __init__(self, api_key: str) -> None:
        self._client = CloudGlue(api_key=api_key)

    # --------------------------------------------------------------------- #
    # Low-level helpers
    # --------------------------------------------------------------------- #
    def upload_file(self, file_path: str | Path):
        """Upload a local file and return the `File` object."""
        file_path = Path(file_path).expanduser()
        return self._client.files.upload(
            file_path=str(file_path),
            wait_until_finish=True
        )

    # --------------------------------------------------------------------- #
    # One-shot high-level call
    # --------------------------------------------------------------------- #
    def general_summary(self, file_path: str | Path) -> Dict[str, Any]:
        """
        1. Uploads *file_path* to CloudGlue.
        2. Runs a describe/transcribe job **and** an extraction job in parallel.
        3. Returns a merged summary dictionary.

        Keys in returned dict
        ---------------------
        description : str   – markdown document
        has_logo    : bool
        logos       : list[str]
        has_face    : bool
        has_speech  : bool
        is_outdoors : bool
        has_text_on_screen : bool
        cloudglue_uri : str
        """
        file_obj = self.upload_file(file_path)

        # configuration copied from your notebook settings
        transcribe_cfg = dict(
            url=file_obj.uri,
            enable_summary=True,
            enable_speech=True,
            enable_scene_text=True,
            enable_visual_scene_description=True,
        )

        extract_prompt = (
            "Extract whether the video contains:\n"
            "- a visually identifiable human face\n"
            "- spoken narration\n"
            "- is outdoors\n"
            "- has text on screen\n"
            "- has logos\n"
            "- visible logo name (provide if known)"
        )
        extract_schema = {
            "has_face": False,
            "has_speech": True,
            "is_outdoors": False,
            "has_text_on_screen": False,
            "logos": ["name of logo if known"],
            "has_logo": True,
        }

        # run in parallel threads because SDK is blocking / network-bound
        with _cf.ThreadPoolExecutor(max_workers=2) as ex:
            f_describe = ex.submit(self._client.transcribe.run, **transcribe_cfg)
            f_extract = ex.submit(
                self._client.extract.run,
                url=file_obj.uri,
                prompt=extract_prompt,
                schema=extract_schema,
                enable_video_level_entities=True,
                enable_segment_level_entities=False,
            )
            describe_job = f_describe.result()
            extract_job = f_extract.result()

        # grab markdown summary
        markdown_doc = self._client.transcribe.get(
            describe_job.job_id, response_format="markdown"
        ).data.content

        ent = extract_job.data.entities
        return {
            "description": markdown_doc,
            "has_logo": ent.get("has_logo"),
            "logos": ent.get("logos", []),
            "has_face": ent.get("has_face"),
            "has_speech": ent.get("has_speech"),
            "is_outdoors": ent.get("is_outdoors"),
            "has_text_on_screen": ent.get("has_text_on_screen"),
            "cloudglue_uri": file_obj.uri
        }
