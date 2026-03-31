"""Video recording utilities."""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, Tuple
import shutil

from ..config import RecordingSettings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ClipInfo:
    path: Path
    started_at: float
    duration: float
    frame_count: int
    last_confidence: float

    def to_metadata(self) -> dict:
        return {
            "path": self.path.name,
            "timestamp": datetime.fromtimestamp(self.started_at).isoformat(),
            "duration": round(self.duration, 2),
            "frames": self.frame_count,
            "confidence": round(self.last_confidence, 3),
        }


class EventManager:
    """Manages event directories of BMPs for the background stitcher."""

    def __init__(self, cfg: RecordingSettings, fps: int) -> None:
        self._cfg = cfg
        self._fps = max(fps, 1)
        self._pre_buffer: Deque[Path] = deque(maxlen=int(self._fps * cfg.pre_buffer_seconds))
        self._is_detecting = False
        self._last_detect_ts = 0.0
        self._last_confidence = 0.0
        self._grace = cfg.grace_period_seconds
        self._active_event_dir: Optional[Path] = None
        self._active_event_start = 0.0
        self._frame_count = 0
        self._last_completed: Optional[dict] = None

    def update(self, filepath: Path, detected: bool, confidence: float) -> None:
        now = time.time()
        if detected:
            self._last_detect_ts = now
            self._last_confidence = confidence
            if not self._is_detecting:
                self._start_event(now)
            self._hardlink(filepath)
        else:
            if self._is_detecting:
                if now - self._last_detect_ts > self._grace:
                    self._stop_event()
                else:
                    self._hardlink(filepath)
            else:
                self._pre_buffer.append(filepath)

    def force_stop(self) -> None:
        if self._is_detecting:
            self._stop_event()

    def consume_last_clip(self) -> Optional[dict]:
        payload, self._last_completed = self._last_completed, None
        return payload

    # internal -------------------------------------------------------------
    def _start_event(self, ts: float) -> None:
        self._is_detecting = True
        self._active_event_start = ts
        self._frame_count = 0
        event_id = datetime.fromtimestamp(ts).strftime("event-%Y%m%d_%H%M%S")
        
        events_root = Path("/tmp/hangermon_events")
        events_root.mkdir(parents=True, exist_ok=True)
        self._active_event_dir = events_root / event_id
        self._active_event_dir.mkdir(parents=True, exist_ok=True)
        
        # Dump pre-buffer into the new event dir
        for past_file in list(self._pre_buffer):
            self._hardlink(past_file)
        self._pre_buffer.clear()
        LOGGER.info("Started detection event: %s", self._active_event_dir)

    def _hardlink(self, filepath: Path) -> None:
        if not self._active_event_dir or not filepath or not filepath.exists():
            return
        target = self._active_event_dir / filepath.name
        try:
            target.hardlink_to(filepath)
        except OSError:
            shutil.copy2(filepath, target)
        self._frame_count += 1

    def _stop_event(self) -> None:
        self._is_detecting = False
        duration = time.time() - self._active_event_start
        if self._active_event_dir:
            meta = {
                "fps": self._fps,
                "duration": round(duration, 2),
                "frames": self._frame_count,
                "confidence": round(self._last_confidence, 3),
                "timestamp": datetime.fromtimestamp(self._active_event_start).isoformat(),
                "path": self._active_event_dir.name
            }
            # Write ready trigger file for stitcher
            meta_path = self._active_event_dir / "ready.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
                
            self._last_completed = meta | {"relative_path": str(self._active_event_dir.name)}
            LOGGER.info("Stopped detection event: %s (%d frames)", self._active_event_dir, self._frame_count)
        self._active_event_dir = None


# For backwards compatibility with service imports during refactor
ClipRecorder = EventManager
__all__ = ["ClipRecorder", "EventManager"]
