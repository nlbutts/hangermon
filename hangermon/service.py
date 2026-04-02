"""Long-running monitoring service that glues everything together."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Optional

import cv2
import numpy as np

from .camera.streamer import CameraStreamer
from .config import Settings, settings
from .detection import DetectionResult, YoloDetector
from .sensehat import sensehat
from .storage import catalog

LOGGER = logging.getLogger(__name__)


class MonitorService:
    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or settings
        self._detector = YoloDetector(self._cfg.yolo)
        self._camera = CameraStreamer(self._cfg.camera)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._status_lock = threading.Lock()
        self._status: Dict[str, object] = {
            "human_present": False,
            "confidence": 0.0,
            "fps": 0.0,
            "recording_state": "monitoring",
            "temperature_c": 0.0,
            "temperature_f": 32.0,
            "humidity": 0.0,
            "led_intensity": 0,
            "last_clip": None,
            "last_updated": None,
        }
        self._fps_window: Deque[float] = deque(maxlen=30)
        self._latest_jpeg: Optional[bytes] = None
        self._target_labels = set(self._cfg.yolo.target_labels) or {"person"}
        self._last_save_time: float = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        LOGGER.info("Starting monitor service")
        catalog.prune_old(self._cfg.recording.base_dir, self._cfg.recording.retention_days)
        self._camera.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="monitor-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._camera.stop()
        sensehat.shutdown()

    def latest_frame_bytes(self) -> Optional[bytes]:
        return self._latest_jpeg

    def status_snapshot(self) -> Dict[str, object]:
        with self._status_lock:
            return dict(self._status)

    # Internal -------------------------------------------------------------
    def _run(self) -> None:
        """Main monitoring loop.

        Each frame (~1fps):
        1. JPEG-encoded for the web UI
        2. Sent to the YOLO server for inference
        3. On confirmed detection → record a minimum_clip-second clip → cooldown → resume
        """
        for frame in self._camera.frames():
            if self._stop.is_set():
                break

            # JPEG for web streaming (always update, even during cooldown)
            self._latest_jpeg = self._to_jpeg(frame.image)

            # Sense HAT sensors
            sensor_data = sensehat.read_sensors()
            self._status_update(sensor_data | {"led_intensity": sensehat.get_led_intensity()})

            # FPS tracking
            self._fps_window.append(time.time())

            # Cooldown: skip YOLO inference until the pre-trigger buffer refills
            cooldown = self._cfg.camera.pre_trigger_time + 5  # extra margin
            if time.time() - self._last_save_time < cooldown:
                fps = self._compute_fps()
                self._status_update({
                    "human_present": False,
                    "fps": fps,
                    "recording_state": "cooldown",
                    "last_updated": frame.timestamp,
                })
                continue

            # YOLO inference on every frame
            detection = self._detector.detect(frame.image, None, None)
            self._handle_detection(frame.timestamp, detection, frame.image)

    def _handle_detection(self, timestamp: float, detection: DetectionResult, frame: np.ndarray) -> None:
        human_conf = 0.0
        if detection.detections:
            human_conf = max(
                (det.confidence for det in detection.detections if det.label in self._target_labels),
                default=0.0,
            )

        fps = self._compute_fps()

        if detection.human_present:
            min_clip = self._cfg.camera.minimum_clip
            pre_trigger = self._cfg.camera.pre_trigger_time
            LOGGER.info(
                "Human confirmed (conf=%.2f), recording %ds clip (%ds pre-trigger + %ds live)...",
                human_conf, min_clip, pre_trigger, min_clip - pre_trigger,
            )

            self._status_update({
                "human_present": True,
                "confidence": round(human_conf, 3),
                "fps": fps,
                "recording_state": "saving",
                "last_updated": timestamp,
            })

            # Launch recording in background so the monitor loop keeps running
            threading.Thread(
                target=self._background_record,
                args=(timestamp, human_conf, frame),
                name="recorder-bg",
                daemon=True,
            ).start()

            # Reset detector so consecutive count starts fresh
            self._detector.reset()
        else:
            self._status_update({
                "human_present": False,
                "confidence": round(human_conf, 3),
                "fps": fps,
                "recording_state": "monitoring",
                "last_updated": timestamp,
            })

    def _background_record(self, timestamp: float, confidence: float, triggering_frame: np.ndarray) -> None:
        """Saves a clip, metadata, and thumbnail in the background."""
        try:
            clip_path = self._camera.save_clip(self._cfg.recording.base_dir)
            self._last_save_time = time.time()

            if clip_path:
                LOGGER.info("Clip saved: %s", clip_path)
                
                # Save thumbnail
                thumb_path = clip_path.with_suffix(".thumb.jpg")
                cv2.imwrite(str(thumb_path), triggering_frame)

                # Write metadata JSON
                meta = {
                    "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                    "duration": self._cfg.camera.minimum_clip,
                    "frames": 0,
                    "confidence": round(confidence, 3),
                    "thumbnail": thumb_path.name,
                }
                meta_path = clip_path.with_suffix(clip_path.suffix + ".json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                self._status_update({
                    "recording_state": "cooldown",
                    "last_clip": {
                        "path": clip_path.name,
                        "timestamp": timestamp,
                        "relative_path": str(clip_path.relative_to(self._cfg.recording.base_dir)),
                        "thumbnail": thumb_path.name,
                    },
                })
            else:
                self._status_update({"recording_state": "cooldown"})
        except Exception:
            LOGGER.error("Background recording failed", exc_info=True)
            self._status_update({"recording_state": "monitoring"})


    def _status_update(self, payload: Dict[str, object]) -> None:
        with self._status_lock:
            self._status.update(payload)

    def _compute_fps(self) -> float:
        now = time.time()
        while self._fps_window and now - self._fps_window[0] > 10.0:
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
