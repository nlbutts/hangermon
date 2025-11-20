"""Helpers for managing recorded clips."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ClipRecord:
    path: Path
    timestamp: datetime
    duration: float
    frames: int
    confidence: float

    def to_dict(self, relative_path: str | None = None) -> dict:
        payload = {
            "path": str(self.path),
            "filename": self.path.name,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration,
            "frames": self.frames,
            "confidence": self.confidence,
        }
        if relative_path:
            payload["relative_path"] = relative_path
        return payload


def load_metadata(meta_path: Path) -> ClipRecord:
    with open(meta_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    video_path = meta_path.with_suffix("")
    return ClipRecord(
        path=video_path,
        timestamp=datetime.fromisoformat(payload["timestamp"]),
        duration=payload.get("duration", 0.0),
        frames=payload.get("frames", 0),
        confidence=payload.get("confidence", 0.0),
    )


def list_clips(base_dir: Path, limit: int = 50) -> List[ClipRecord]:
    entries: List[ClipRecord] = []
    if not base_dir.exists():
        return entries
    candidates = sorted(base_dir.rglob("*.json"), reverse=True)
    for meta in candidates[:limit]:
        try:
            entries.append(load_metadata(meta))
        except Exception:  # pragma: no cover - just skip corrupt files
            LOGGER.warning("Unable to read clip metadata: %s", meta, exc_info=True)
    return entries


def prune_old(base_dir: Path, retention_days: int) -> int:
    if retention_days <= 0 or not base_dir.exists():
        return 0
    removed = 0
    cutoff = datetime.now() - timedelta(days=retention_days)
    for meta in base_dir.rglob("*.json"):
        try:
            clip = load_metadata(meta)
        except Exception:
            meta.unlink(missing_ok=True)
            continue
        if clip.timestamp < cutoff:
            try:
                meta.unlink(missing_ok=True)
                clip.path.unlink(missing_ok=True)
                removed += 1
            except OSError:
                LOGGER.warning("Failed to delete %s", clip.path, exc_info=True)
    return removed


__all__ = ["ClipRecord", "list_clips", "prune_old"]
