"""Detection base types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import numpy as np


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: Sequence[float]


@dataclass(slots=True)
class DetectionResult:
    detections: List[Detection]
    annotated_frame: np.ndarray
    latency_ms: float
    human_present: bool
