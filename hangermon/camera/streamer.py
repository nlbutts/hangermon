"""Camera capture utilities."""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Generator, Optional

import cv2
import numpy as np

try:  # Picamera2 is optional and only available on Raspberry Pi OS
    from picamera2 import Picamera2  # type: ignore
except ImportError:  # pragma: no cover - not available in CI
    Picamera2 = None  # type: ignore

from ..config import CameraSettings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Frame:
    image: np.ndarray
    timestamp: float
    metadata: Optional[dict] = None


class CameraStreamer:
    """Continuously capture frames from Pi Camera or OpenCV device."""

    def __init__(self, config: CameraSettings, synthetic: bool = False) -> None:
        self._cfg = config
        self._synthetic = synthetic
        self._queue: "queue.Queue[Frame]" = queue.Queue(maxsize=self._cfg.queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._capture = None
        self._latest: Optional[Frame] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        LOGGER.info("Starting camera streamer (device=%s synthetic=%s)", self._cfg.device, self._synthetic)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:  # pragma: no cover - defensive
                LOGGER.warning("Failed to release camera", exc_info=True)
        self._capture = None

    def latest(self) -> Optional[Frame]:
        return self._latest

    def frames(self) -> Generator[Frame, None, None]:
        while not self._stop.is_set():
            try:
                yield self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

    # Internal helpers -----------------------------------------------------
    def _loop(self) -> None:
        if self._synthetic:
            self._synthetic_loop()
            return
        if self._cfg.use_picamera2 and Picamera2:
            self._picamera_loop()
            return
        self._opencv_loop()

    def _opencv_loop(self) -> None:
        self._capture = cv2.VideoCapture(self._cfg.device)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        self._capture.set(cv2.CAP_PROP_FPS, self._cfg.fps)
        while not self._stop.is_set():
            grabbed, frame = self._capture.read()
            if not grabbed:
                LOGGER.warning("Camera frame grab failed; retrying")
                time.sleep(0.05)
                continue
            self._publish(frame, None)

    def _picamera_loop(self) -> None:  # pragma: no cover - requires hardware
        assert Picamera2 is not None
        camera = Picamera2()
        config = camera.create_video_configuration(main={"size": (self._cfg.width, self._cfg.height)})
        camera.configure(config)
        camera.start()
        try:
            while not self._stop.is_set():
                request = camera.capture_request()
                try:
                    frame = request.make_array("main")
                    metadata = request.get_metadata()
                finally:
                    request.release()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self._publish(frame_bgr, metadata)
        finally:
            camera.stop()

    def _synthetic_loop(self) -> None:
        interval = 1.0 / max(self._cfg.fps, 1)
        t = 0
        while not self._stop.is_set():
            frame = np.zeros((self._cfg.height, self._cfg.width, 3), dtype=np.uint8)
            cv2.putText(frame, f"synthetic {t:04d}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            synthetic_meta = {
                "imx500": {
                    "results": []
                }
            }
            self._publish(frame, synthetic_meta)
            time.sleep(interval)
            t += 1

    def _publish(self, frame: np.ndarray, metadata: Optional[dict]) -> None:
        stamped = Frame(image=frame, timestamp=time.time(), metadata=metadata)
        self._latest = stamped
        try:
            self._queue.put_nowait(stamped)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(stamped)


__all__ = ["CameraStreamer", "Frame"]
