"""Camera capture with H.264 circular buffer and lores still output."""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

from picamera2 import Picamera2  # type: ignore
from picamera2.encoders import H264Encoder  # type: ignore
from picamera2.outputs import CircularOutput  # type: ignore

from ..config import CameraSettings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Frame:
    """A 640x480 still captured once per second."""
    image: np.ndarray
    timestamp: float


class CameraStreamer:
    """Captures 1920x1080 H.264 to in-memory circular buffer.

    The circular buffer holds `pre_trigger_time` seconds of H.264 data.
    On save_clip(), the buffer is flushed (pre-trigger footage) and the
    encoder continues recording live for the remaining time to reach
    `minimum_clip` seconds total.

    Yields 640x480 stills at ~1fps for the web UI and YOLO inference.
    """

    def __init__(self, config: CameraSettings, synthetic: bool = False) -> None:
        self._cfg = config
        self._synthetic = synthetic
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=self._cfg.queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._latest: Optional[Frame] = None
        self._picam2: Optional[Picamera2] = None
        self._encoder: Optional[H264Encoder] = None
        self._circ_output: Optional[CircularOutput] = None
        self._save_lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        LOGGER.info("Starting camera streamer")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def latest(self) -> Optional[Frame]:
        return self._latest

    def frames(self) -> Generator[Frame, None, None]:
        """Yields ~1fps frames. Blocks until a frame is available."""
        while not self._stop.is_set():
            try:
                yield self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

    def save_clip(self, output_dir: Path) -> Optional[Path]:
        """Save a clip of minimum_clip seconds total.

        The clip consists of:
          - pre_trigger_time seconds from the circular buffer (pre-trigger)
          - (minimum_clip - pre_trigger_time) seconds of live recording

        Blocks for the live recording duration. Returns the MP4 path on
        success, or None on failure.
        """
        if self._circ_output is None:
            LOGGER.warning("No circular output available for saving")
            return None

        with self._save_lock:
            ts = datetime.now()
            rel_dir = ts.strftime("%Y/%m/%d")
            target_dir = output_dir / rel_dir
            target_dir.mkdir(parents=True, exist_ok=True)

            h264_path = target_dir / ts.strftime("hanger-%H%M%S.h264")
            mp4_path = h264_path.with_suffix(".mp4")

            pre_trigger = self._cfg.pre_trigger_time
            live_duration = max(self._cfg.minimum_clip - pre_trigger, 5)

            try:
                # Start file output: dumps the pre-trigger buffer AND
                # continues recording live frames to the same file.
                self._circ_output.fileoutput = str(h264_path)
                self._circ_output.start()
                LOGGER.info(
                    "Recording clip: %ds pre-trigger + %ds live = %ds total → %s",
                    pre_trigger, live_duration, pre_trigger + live_duration, h264_path.name,
                )

                # Wait for the live portion (camera keeps feeding frames
                # to both the circular buffer and the file during this time)
                self._stop.wait(timeout=live_duration)

                self._circ_output.stop()
                self._circ_output.fileoutput = None
                LOGGER.info("Clip recording complete: %s", h264_path)

                # Remux H.264 → MP4 (container copy, no re-encode, nearly instant)
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(h264_path), "-c", "copy", str(mp4_path)],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    h264_path.unlink(missing_ok=True)
                    LOGGER.info("Remuxed to MP4: %s", mp4_path)
                    return mp4_path
                else:
                    LOGGER.error("FFmpeg remux failed (exit %d): %s", result.returncode, result.stderr)
                    return h264_path  # still usable as raw H.264

            except Exception:
                LOGGER.error("Failed to save clip", exc_info=True)
                try:
                    self._circ_output.stop()
                except Exception:
                    pass
                self._circ_output.fileoutput = None
                return None

    # Internal helpers --------------------------------------------------
    def _loop(self) -> None:
        if self._synthetic:
            self._synthetic_loop()
            return
        self._picamera_loop()

    def _picamera_loop(self) -> None:  # pragma: no cover - requires hardware
        self._picam2 = Picamera2()

        video_config = self._picam2.create_video_configuration(
            main={"size": (self._cfg.width, self._cfg.height)},
            lores={"size": (self._cfg.resize_width, self._cfg.resize_height)},
            encode="main",
        )
        self._picam2.configure(video_config)

        # H.264 hardware encoder → circular ring buffer sized to pre_trigger_time
        self._encoder = H264Encoder(bitrate=self._cfg.h264_bitrate)
        buf_frames = self._cfg.fps * self._cfg.pre_trigger_time
        self._circ_output = CircularOutput(buffersize=buf_frames)

        self._picam2.start_recording(self._encoder, self._circ_output)
        LOGGER.info(
            "Picamera2 H.264 recording started: main=%dx%d@%dfps, lores=%dx%d, "
            "pre_trigger=%ds (%d frames), minimum_clip=%ds, bitrate=%d",
            self._cfg.width, self._cfg.height, self._cfg.fps,
            self._cfg.resize_width, self._cfg.resize_height,
            self._cfg.pre_trigger_time, buf_frames,
            self._cfg.minimum_clip,
            self._cfg.h264_bitrate,
        )

        try:
            while not self._stop.is_set():
                # Grab a lores still (non-blocking, doesn't interfere with encoder)
                request = self._picam2.capture_request()
                try:
                    lores = request.make_array("lores")
                finally:
                    request.release()

                lores_bgr = cv2.cvtColor(lores, cv2.COLOR_YUV420p2RGB)
                self._publish(lores_bgr)

                # Wait ~1 second before next still capture
                self._stop.wait(timeout=1.0)
        finally:
            self._picam2.stop_recording()
            self._picam2.close()
            self._picam2 = None
            self._circ_output = None
            LOGGER.info("Picamera2 stopped and closed")

    def _synthetic_loop(self) -> None:
        t = 0
        while not self._stop.is_set():
            frame = np.zeros((self._cfg.resize_height, self._cfg.resize_width, 3), dtype=np.uint8)
            cv2.putText(frame, f"synthetic {t:04d}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self._publish(frame)
            self._stop.wait(timeout=1.0)
            t += 1

    def _publish(self, frame: np.ndarray) -> None:
        stamped = Frame(image=frame, timestamp=time.time())
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
