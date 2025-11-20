"""Video recording utilities."""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np

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


class ClipRecorder:
    """Creates MP4 clips whenever detections are active."""

    def __init__(self, cfg: RecordingSettings, fps: int) -> None:
        self._cfg = cfg
        self._fps = max(fps, 1)
        self._buffer: Deque[np.ndarray] = deque(maxlen=int(self._fps * cfg.pre_buffer_seconds))
        self._writer: Optional[cv2.VideoWriter] = None
        self._current_clip: Optional[ClipInfo] = None
        self._last_completed: Optional[dict] = None
        self._last_detection_ts: float = 0.0
        self._last_confidence: float = 0.0
        self._grace = cfg.grace_period_seconds
        self._fourcc = cv2.VideoWriter_fourcc(*cfg.codec)
        self._cfg.base_dir.mkdir(parents=True, exist_ok=True)

    def update(self, frame: np.ndarray, detected: bool, confidence: float) -> None:
        now = time.time()
        if detected:
            self._last_detection_ts = now
            self._last_confidence = confidence
            if self._writer is None:
                self._start_clip(frame)
                for buffered_frame in list(self._buffer):
                    self._write(buffered_frame)
            self._write(frame)
        else:
            self._buffer.append(frame.copy())
            if self._writer and now - self._last_detection_ts > self._grace:
                self._stop_clip()

    def force_stop(self) -> None:
        if self._writer:
            self._stop_clip()

    def consume_last_clip(self) -> Optional[dict]:
        payload, self._last_completed = self._last_completed, None
        return payload

    # internal -------------------------------------------------------------
    def _start_clip(self, frame: np.ndarray) -> None:
        ts = datetime.now()
        rel_dir = Path(ts.strftime("%Y/%m/%d"))
        target_dir = self._cfg.base_dir / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = ts.strftime("hanger-%H%M%S") + self._cfg.extension
        path = target_dir / filename
        height, width = frame.shape[:2]
        self._writer = cv2.VideoWriter(str(path), self._fourcc, self._fps, (width, height))
        self._current_clip = ClipInfo(path=path, started_at=time.time(), duration=0, frame_count=0, last_confidence=0.0)
        LOGGER.info("Started recording %s", path)

    def _write(self, frame: np.ndarray) -> None:
        if not self._writer or not self._current_clip:
            return
        self._writer.write(frame)
        self._current_clip.frame_count += 1
        self._current_clip.duration = self._current_clip.frame_count / self._fps
        self._current_clip.last_confidence = self._last_confidence

    def _stop_clip(self) -> None:
        if not self._writer or not self._current_clip:
            return
        self._writer.release()
        self._writer = None
        meta_path = self._current_clip.path.with_suffix(self._cfg.extension + ".json")
        meta = self._current_clip.to_metadata()
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        clip_rel = self._current_clip.path.relative_to(self._cfg.base_dir)
        self._last_completed = meta | {"relative_path": str(clip_rel)}
        LOGGER.info("Stopped recording %s", self._current_clip.path)
        self._current_clip = None


__all__ = ["ClipRecorder", "ClipInfo"]
