"""Long-running monitoring service that glues everything together."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional
from pathlib import Path

import cv2
import numpy as np

from .camera.streamer import CameraStreamer
from .config import Settings, settings
from .detection import DetectionResult, YoloDetector
from .recording.writer import ClipRecorder
from .recording.stitcher import BackgroundStitcher
from .storage import catalog

LOGGER = logging.getLogger(__name__)


class MonitorService:
    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or settings
        self._detector = YoloDetector(self._cfg.yolo)
        self._camera = CameraStreamer(self._cfg.camera)
        self._recorder = ClipRecorder(self._cfg.recording, fps=self._cfg.camera.fps)
        self._stitcher = BackgroundStitcher(self._cfg.recording, self._cfg.camera)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._status_lock = threading.Lock()
        self._status: Dict[str, object] = {
            "human_present": False,
            "confidence": 0.0,
            "fps": 0.0,
            "recording_state": "standby",
            "last_clip": None,
            "last_updated": None,
        }
        self._fps_window: Deque[float] = deque(maxlen=30)
        self._latest_jpeg: Optional[bytes] = None
        self._last_detections = []
        self._target_labels = set(self._cfg.yolo.target_labels) or {"person"}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        LOGGER.info("Starting monitor service")
        catalog.prune_old(self._cfg.recording.base_dir, self._cfg.recording.retention_days)
        self._camera.start()
        self._stitcher.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="monitor-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._camera.stop()
        self._recorder.force_stop()
        self._stitcher.stop()

    def latest_frame_bytes(self) -> Optional[bytes]:
        return self._latest_jpeg

    def status_snapshot(self) -> Dict[str, object]:
        with self._status_lock:
            return dict(self._status)

    # Internal -------------------------------------------------------------
    def _run(self) -> None:
        last_inference = 0.0
        interval = self._cfg.yolo.inference_interval_seconds

        for frame in self._camera.frames():
            if self._stop.is_set():
                break
                
            frame_small = cv2.resize(frame.image, (self._cfg.camera.resize_width, self._cfg.camera.resize_height))
            self._latest_jpeg = self._to_jpeg(frame_small)

            now = time.time()
            if now - last_inference >= interval:
                detection = self._detector.detect(frame_small, None, None)
                last_inference = now
                self._last_detections = detection.detections
                self._handle_detection(frame.timestamp, detection, frame.cir_buf_file)
            else:
                # Maintain recording event state continuously
                self._fps_window.append(now)
                
                current_detected = bool(self._status.get("human_present", False))
                conf = float(self._status.get("confidence", 0.0))
                self._recorder.update(frame.cir_buf_file, current_detected, conf)
                
                rec_state = "recording" if self._recorder._is_detecting else "standby"
                self._status_update({"recording_state": rec_state})
                
                clip_meta = self._recorder.consume_last_clip()
                if clip_meta:
                    self._status_update({"last_clip": clip_meta})

    def _handle_detection(self, timestamp: float, detection: DetectionResult, cir_buf_file: Optional[Path]) -> None:
        human_conf = 0.0
        if detection.detections:
            human_conf = max(
                (
                    det.confidence
                    for det in detection.detections
                    if det.label in self._target_labels
                ),
                default=0.0,
            )
        self._recorder.update(cir_buf_file, detection.human_present, human_conf)
        clip_meta = self._recorder.consume_last_clip()
        self._fps_window.append(time.time())
        fps = self._compute_fps()
        self._status_update(
            {
                "human_present": detection.human_present,
                "confidence": round(human_conf, 3),
                "fps": fps,
                "last_updated": timestamp,
                "last_clip": clip_meta or self._status.get("last_clip"),
                "recording_state": "recording" if self._recorder._is_detecting else "standby",
            }
        )

    def _status_update(self, payload: Dict[str, object]) -> None:
        with self._status_lock:
            self._status.update(payload)

    def _compute_fps(self) -> float:
        now = time.time()
        while self._fps_window and now - self._fps_window[0] > 2.0:
            self._fps_window.popleft()
        if len(self._fps_window) < 2:
            return 0.0
        elapsed = self._fps_window[-1] - self._fps_window[0]
        frames = len(self._fps_window)
        return round((frames - 1) / elapsed, 2) if elapsed > 0 else 0.0

    @staticmethod
    def _to_jpeg(frame: np.ndarray) -> Optional[bytes]:
        success, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if success else None


__all__ = ["MonitorService"]
