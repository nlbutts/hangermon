"""Camera capture utilities."""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

from picamera2 import Picamera2  # type: ignore

from ..config import CameraSettings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Frame:
    image: np.ndarray
    timestamp: float
    metadata: Optional[dict] = None
    picamera: Optional[object] = None
    cir_buf_file: Optional[Path] = None


class CircularBufferWriter:
    """Writes frames to disk as BMPs in a background thread, maintaining a rolling window."""

    def __init__(self, cfg: CameraSettings) -> None:
        self._cfg = cfg
        self._dir = Path(cfg.circular_buffer_dir)
        self._duration = cfg.circular_buffer_duration_sec
        self._history: deque[tuple[float, Path]] = deque()
        self._queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=30)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        for old in self._dir.glob("*.bmp"):
            old.unlink(missing_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="cir-buf-writer", daemon=True)
        self._thread.start()
        LOGGER.info("Circular buffer writer started -> %s", self._dir)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def submit(self, frame: np.ndarray, timestamp: float) -> Path:
        """Submit a frame and return the expected filename (may not be written yet)."""
        filename = self._dir / f"{timestamp:.6f}.bmp"
        try:
            self._queue.put_nowait((frame, timestamp))
        except queue.Full:
            LOGGER.warning("Circular buffer write queue full, dropping frame")
        return filename

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame, timestamp = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            filename = self._dir / f"{timestamp:.6f}.bmp"
            cv2.imwrite(str(filename), frame)
            self._history.append((timestamp, filename))

            cutoff = timestamp - self._duration
            while self._history and self._history[0][0] < cutoff:
                _, old_file = self._history.popleft()
                old_file.unlink(missing_ok=True)


class CameraStreamer:
    """Continuously capture frames from Pi Camera."""

    def __init__(self, config: CameraSettings, synthetic: bool = False) -> None:
        self._cfg = config
        self._synthetic = synthetic
        self._queue: "queue.Queue[Frame]" = queue.Queue(maxsize=self._cfg.queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._latest: Optional[Frame] = None
        self._buf_writer = CircularBufferWriter(config)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        LOGGER.info("Starting camera streamer (device=%s synthetic=%s)", self._cfg.device, self._synthetic)
        self._buf_writer.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._buf_writer.stop()

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
        self._picamera_loop()

    def _picamera_loop(self) -> None:  # pragma: no cover - requires hardware
        camera = Picamera2()
        config = camera.create_video_configuration(main={"size": (self._cfg.width, self._cfg.height)})
        camera.configure(config)
        camera.start()
        LOGGER.info("Picamera2 started at %dx%d", self._cfg.width, self._cfg.height)
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
            camera.close()
            LOGGER.info("Picamera2 stopped and closed")

    def _synthetic_loop(self) -> None:
        interval = 1.0 / max(self._cfg.fps, 1)
        t = 0
        while not self._stop.is_set():
            frame = np.zeros((self._cfg.height, self._cfg.width, 3), dtype=np.uint8)
            cv2.putText(frame, f"synthetic {t:04d}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            self._publish(frame, None)
            time.sleep(interval)
            t += 1

    def _publish(self, frame: np.ndarray, metadata: Optional[dict]) -> None:
        timestamp = time.time()
        buf_file = self._buf_writer.submit(frame, timestamp)
        stamped = Frame(image=frame, timestamp=timestamp, metadata=metadata, cir_buf_file=buf_file)
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
