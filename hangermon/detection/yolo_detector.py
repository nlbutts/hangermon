"""YOLO detector using ZeroMQ communication with yolo_server.py."""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import zmq

from ..config import YoloSettings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: Sequence[float]


@dataclass(slots=True)
class DetectionResult:
    detections: List[Detection]
    annotated_frame: np.ndarray
    inference_time_ms: float
    human_present: bool


class YoloDetector:
    """Consumes YOLO detections via ZeroMQ server."""

    def __init__(self, cfg: YoloSettings) -> None:
        self._cfg = cfg
        self._target_labels = set(cfg.target_labels) or {"person"}
        self._zmq_context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._connection_failed = False
        self._consecutive_detections = 0
        self._init_connection()

    def _init_connection(self) -> None:
        """Initialize ZeroMQ connection with retry logic."""
        max_retries = 10
        base_delay = 1.0
        max_delay = 30.0

        for attempt in range(max_retries):
            try:
                self._zmq_context = zmq.Context()
                self._socket = self._zmq_context.socket(zmq.REQ)
                self._socket.setsockopt(zmq.RCVTIMEO, 5000)
                self._socket.setsockopt(zmq.SNDTIMEO, 5000)

                address = f"tcp://localhost:{self._cfg.server_port}"
                self._socket.connect(address)

                self._connection_failed = False
                LOGGER.info("YOLO detector connected to server at %s", address)
                return

            except zmq.ZMQError as e:
                delay = min(base_delay * (2**attempt), max_delay)
                LOGGER.warning(
                    "YOLO server connection attempt %d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    e,
                    delay,
                )
                time.sleep(delay)

        self._connection_failed = True
        LOGGER.error("Failed to connect to YOLO server after %d attempts", max_retries)

    def _send_request(self, image_bytes: bytes, width: int, height: int) -> dict:
        """Send frame to YOLO server and return response."""
        header = struct.pack("<II", width, height)
        self._socket.send_multipart([header, image_bytes])
        response_bytes = self._socket.recv()
        return __import__("json").loads(response_bytes.decode("utf-8"))

    def detect(
        self,
        frame: np.ndarray,
        metadata: Optional[dict] = None,
        picamera: Optional[object] = None,
    ) -> DetectionResult:
        """Run YOLO detection on frame."""
        if self._connection_failed:
            self._init_connection()
            if self._connection_failed:
                return DetectionResult(
                    detections=[],
                    annotated_frame=frame.copy(),
                    inference_time_ms=0.0,
                    human_present=False,
                )

        height, width = frame.shape[:2]
        # Frame arrives as BGR from cv2; YOLO server expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_bytes = frame_rgb.tobytes()

        try:
            start_time = time.perf_counter()
            response = self._send_request(image_bytes, width, height)
            inference_time_ms = (time.perf_counter() - start_time) * 1000

            if not response.get("success"):
                LOGGER.error("YOLO server returned error: %s", response.get("error"))
                return DetectionResult(
                    detections=[],
                    annotated_frame=frame.copy(),
                    inference_time_ms=inference_time_ms,
                    human_present=False,
                )

            detections = self._parse_detections(response, width, height)
            annotated = self._draw_detections(frame.copy(), detections)
            human_present = any(det.label in self._target_labels for det in detections)

            if human_present:
                self._consecutive_detections += 1
            else:
                self._consecutive_detections = 0

            human_present = (
                self._consecutive_detections >= self._cfg.detections_required
            )

            return DetectionResult(
                detections=detections,
                annotated_frame=annotated,
                inference_time_ms=inference_time_ms,
                human_present=human_present,
            )

        except zmq.ZMQError as e:
            LOGGER.error("ZeroMQ error during detection: %s", e)
            self._connection_failed = True
            return DetectionResult(
                detections=[],
                annotated_frame=frame.copy(),
                inference_time_ms=0.0,
                human_present=False,
            )

    def _parse_detections(
        self, response: dict, width: int, height: int
    ) -> List[Detection]:
        """Parse YOLO server response into Detection objects."""
        detections = []
        for det in response.get("detections", []):
            confidence = det.get("confidence", 0.0)
            if confidence < self._cfg.confidence_threshold:
                continue

            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            label = det.get("class_name", "object")
            detections.append(
                Detection(
                    label=label, confidence=confidence, bbox_xyxy=[x1, y1, x2, y2]
                )
            )

        return detections

    def _draw_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """Draw detections on frame."""
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox_xyxy)
            color = (0, 204, 102) if det.label == "person" else (0, 128, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{det.label}:{det.confidence:.2f}"
            cv2.putText(
                frame,
                label_text,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return frame



    def reset(self) -> None:
        """Reset detection state. Called after saving a clip."""
        self._consecutive_detections = 0
        LOGGER.info("YOLO detector reset (consecutive detections cleared)")

    def stop(self) -> None:
        """Stop detector and cleanup."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass

        if self._zmq_context:
            try:
                self._zmq_context.term()
            except Exception:
                pass

        LOGGER.info("YOLO detector stopped")
