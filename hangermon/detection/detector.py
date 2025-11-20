"""IMX500 metadata-based detection utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
from picamera2.devices.imx500 import IMX500

from ..config import DetectionSettings

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


class Imx500Detector:
    """Consumes IMX500 inference metadata emitted by Picamera2."""

    def __init__(self, cfg: DetectionSettings) -> None:
        self._cfg = cfg
        self._metadata_path = [chunk for chunk in cfg.metadata_path.split(".") if chunk]
        self._target_labels = set(cfg.target_labels) or {"person"}

        # This must be called before instantiation of Picamera2
        imx500 = IMX500('/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk')

    def detect(self, frame: np.ndarray, metadata: Optional[dict]) -> DetectionResult:
        width = frame.shape[1]
        height = frame.shape[0]
        detections = self._extract_detections(metadata, width, height)
        annotated = frame.copy()
        if self._cfg.overlay:
            for det in detections:
                self._draw_box(annotated, det)
        human_present = any(det.label in self._target_labels for det in detections)
        latency = self._latency_from_metadata(metadata)
        return DetectionResult(detections, annotated, latency, human_present)

    def _extract_detections(self, metadata: Optional[dict], width: int, height: int) -> List[Detection]:
        entries = self._resolve_metadata(metadata)
        if entries is None:
            entries = self._guess_entries(metadata)
            if entries is None:
                if metadata is not None:
                    LOGGER.debug("IMX500 metadata path %s missing in payload", ".".join(self._metadata_path))
                return []
        detections: List[Detection] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label = entry.get(self._cfg.label_field)
            if not label:
                continue
            confidence = self._coerce_float(entry.get(self._cfg.score_field))
            if confidence < self._cfg.min_confidence:
                continue
            bbox = self._normalize_bbox(entry.get(self._cfg.bbox_field), width, height)
            if not bbox:
                continue
            detections.append(Detection(label=str(label), confidence=confidence, bbox_xyxy=bbox))
        return detections

    def _resolve_metadata(self, metadata: Optional[dict]) -> Any:
        node: Any = metadata or {}
        for key in self._metadata_path:
            if isinstance(node, dict):
                node = node.get(key)
            else:
                return None
        return node

    def _normalize_bbox(self, bbox: Any, width: int, height: int) -> Optional[Sequence[float]]:
        if isinstance(bbox, str):
            tokens = [token.strip() for token in bbox.replace(";", ",").split(",") if token.strip()]
            bbox = tokens
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        fmt = self._cfg.bbox_format.lower()
        values = [self._coerce_float(v) for v in bbox]
        if self._cfg.bbox_normalized:
            values[0] *= width
            values[1] *= height
            values[2] *= width
            values[3] *= height
        if fmt == "xywh":
            x, y, w, h = values
            x1, y1 = x, y
            x2, y2 = x + w, y + h
        else:  # assume xyxy
            x1, y1, x2, y2 = values
        return [self._clip(x1, 0, width), self._clip(y1, 0, height), self._clip(x2, 0, width), self._clip(y2, 0, height)]

    def _latency_from_metadata(self, metadata: Optional[dict]) -> float:
        field = self._cfg.latency_field
        if not field or not metadata:
            return 0.0
        node: Any = metadata
        for key in field.split("."):
            if isinstance(node, dict):
                node = node.get(key)
            else:
                return 0.0
        return self._coerce_float(node)

    def _guess_entries(self, metadata: Any) -> Optional[List[dict]]:
        if metadata is None:
            return None
        if isinstance(metadata, list):
            if metadata and all(isinstance(item, dict) for item in metadata):
                sample = metadata[0]
                if isinstance(sample, dict) and self._cfg.label_field in sample and self._cfg.score_field in sample:
                    return metadata
            for item in metadata:
                found = self._guess_entries(item)
                if found:
                    return found
        if isinstance(metadata, dict):
            for value in metadata.values():
                found = self._guess_entries(value)
                if found:
                    return found
        return None

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _draw_box(frame: np.ndarray, detection: Detection) -> None:
        import cv2

        x1, y1, x2, y2 = map(int, detection.bbox_xyxy)
        color = (0, 204, 102) if detection.label == "person" else (0, 128, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.label}:{detection.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


__all__ = ["Detection", "DetectionResult", "Imx500Detector"]
