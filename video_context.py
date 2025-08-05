"""
Autonomous Video Hunter - Video Context Store

This module provides a lightweight, in-memory context database for storing and retrieving
processed video analysis results. It manages video summaries, extracted features, and
investigation memories in JSONL format for efficient agent consumption.

Key Features:
- JSONL-based persistent storage
- In-memory caching for fast access
- Video summary and memory management
- Pagination support for large collections
- Configurable local frame inclusion

Author: @kdr
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional


class VideoContextStore:
    """
    Lightweight, in-memory context DB for LLM / agent consumption.

    Internal layout
    ---------------
    {
        "<abs_video_path>": {
            "summary": { … VideoProcessor record … },
            "memories": [("name", "value"), ...]   # newest first
        },
        ...
    }
    """

    # ------------------------------------------------------------------ #
    # Construction / Loading                                             #
    # ------------------------------------------------------------------ #

    def __init__(self, db_jsonl_path: str | Path | None = None) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._db_path: Optional[Path] = None  # remembered for reload()

        if db_jsonl_path is not None:
            self._db_path = Path(db_jsonl_path).expanduser().resolve()
            self._load_from_jsonl(self._db_path)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def get_current_context(
        self,
        local_video_path: str | Path,
        *,
        include_local_frames: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Return video summary + memories, or None if unseen.

        Parameters
        ----------
        include_local_frames : bool, default False
            If False the `local_frames` field is stripped to save prompt space.
        """
        key = self._normalize_path(local_video_path)
        entry = self._store.get(key)
        if entry is None:
            return None

        summary_copy = dict(entry["summary"])  # avoid mutating cache
        if not include_local_frames:
            summary_copy.pop("local_frames", None)

        return {
            "video_path": key,
            "summary": summary_copy,
            "memories": [{"name": n, "value": v} for n, v in entry["memories"]],
        }

    def add_memory(self, local_video_path: str | Path, memory_name: str, memory_value: str) -> None:
        """Prepend a new (name, value) memory for the given video."""
        key = self._normalize_path(local_video_path)
        if key not in self._store:
            self._store[key] = {"summary": {}, "memories": []}
        self._store[key]["memories"].insert(0, (memory_name, memory_value))

    def get_collection_context(
        self,
        *,
        start: int = 0,
        limit: int = 10,
        include_local_frames: bool = False,
    ) -> Dict[str, Any]:
        """
        Paginated, alphabetically sorted view of ALL video contexts.

        Parameters
        ----------
        start, limit : pagination controls
        include_local_frames : bool, default False
            Strip or include the potentially large `local_frames` list.
        """
        keys_sorted = sorted(self._store.keys())
        total = len(keys_sorted)
        slice_ = keys_sorted[start : start + limit]

        contexts = []
        for k in slice_:
            summary_copy = dict(self._store[k]["summary"])
            if not include_local_frames:
                summary_copy.pop("local_frames", None)

            contexts.append(
                {
                    "video_path": k,
                    "summary": summary_copy,
                    "memories": [
                        {"name": n, "value": v} for n, v in self._store[k]["memories"]
                    ],
                }
            )

        return {"start": start, "limit": limit, "total": total, "video_contexts": contexts}

    def reload(self, db_jsonl_path: str | Path | None = None) -> None:
        """
        Refresh the in-memory store from disk.

        If `db_jsonl_path` is omitted, reloads from the path used during
        construction / last reload.
        """
        path = (
            Path(db_jsonl_path).expanduser().resolve()
            if db_jsonl_path is not None
            else self._db_path
        )
        if path is None:
            raise ValueError("No DB path available; provide `db_jsonl_path`.")

        self._db_path = path
        self._store.clear()
        self._load_from_jsonl(path)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _load_from_jsonl(self, path: Path) -> None:
        """Populate the store from a VideoProcessor JSONL file."""
        if not path.exists():
            raise FileNotFoundError(f"JSONL DB not found: {path}")

        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                try:
                    record = json.loads(line)
                    key = self._normalize_path(record["local_video_path"])
                    self._store[key] = {"summary": record, "memories": []}
                except (json.JSONDecodeError, KeyError):
                    continue  # skip malformed rows silently

    @staticmethod
    def _normalize_path(p: str | Path) -> str:
        """Return a canonical absolute path string."""
        return str(Path(p).expanduser().resolve())
