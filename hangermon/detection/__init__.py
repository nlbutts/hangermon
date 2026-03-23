"""Detection helpers."""
from __future__ import annotations

from .detector import Detection, DetectionResult
from .yolov8_detector import Yolov8Detector
from ..config import Settings


def make_detector(cfg: Settings) -> Yolov8Detector:
    """Factory: return the YOLOv8 detector."""
    return Yolov8Detector(cfg.yolov8)


__all__ = ["Detection", "DetectionResult", "Yolov8Detector", "make_detector"]
