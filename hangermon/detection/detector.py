"""IMX500 metadata/tensor driven detector."""
from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

try:  # Optional dependency – only available on Raspberry Pi with Picamera2
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
    from picamera2.devices.imx500.postprocess import scale_boxes
except ImportError:  # pragma: no cover - keeps unit tests working off-device
    IMX500 = None  # type: ignore
    NetworkIntrinsics = None  # type: ignore
    postprocess_nanodet_detection = None  # type: ignore
    scale_boxes = None  # type: ignore

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
    """Consumes either high-level metadata or raw IMX500 tensors."""

    def __init__(self, cfg: DetectionSettings) -> None:
        self._cfg = cfg
        self._metadata_path = [chunk for chunk in cfg.metadata_path.split(".") if chunk]
        self._target_labels = set(cfg.target_labels) or {"person"}
        self._imx500 = None
        self._intrinsics = None
        self._labels: Sequence[str] = tuple(cfg.target_labels) or ("person",)
        self._iou = cfg.iou
        self._max_detections = max(cfg.max_detections, 1)
        self._init_imx500()

    def detect(
        self,
        frame: np.ndarray,
        metadata: Optional[dict],
        picamera: Optional[object] = None,
    ) -> DetectionResult:
        width = frame.shape[1]
        height = frame.shape[0]
        detections = self._extract_detections(metadata, width, height)
        if not detections:
            detections = self._parse_imx500_outputs(metadata, picamera, width, height)
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

    # ------------------------------------------------------------------
    # IMX500 tensor parsing path (reference script adaptation)
    def _init_imx500(self) -> None:
        if IMX500 is None:
            LOGGER.info("picamera2 IMX500 helpers unavailable; relying on metadata path only")
            return
        model_path = self._cfg.model_path
        if not model_path:
            return
        try:
            self._imx500 = IMX500(model_path)
        except Exception as exc:  # pragma: no cover - requires hardware/files
            LOGGER.warning("Unable to load IMX500 model %s: %s", model_path, exc)
            self._imx500 = None
            return
        intrinsics = getattr(self._imx500, "network_intrinsics", None)
        if intrinsics is None and NetworkIntrinsics is not None:
            intrinsics = NetworkIntrinsics()
        if intrinsics is None:
            LOGGER.warning("IMX500 network intrinsics not available")
            return
        if hasattr(intrinsics, "update_with_defaults"):
            intrinsics.update_with_defaults()
        if self._cfg.postprocess:
            intrinsics.postprocess = self._cfg.postprocess
        if hasattr(intrinsics, "bbox_normalization"):
            intrinsics.bbox_normalization = self._cfg.bbox_normalized
        if hasattr(intrinsics, "bbox_order") and self._cfg.bbox_format.lower() == "xyxy":
            intrinsics.bbox_order = "xy"
        if hasattr(intrinsics, "ignore_dash_labels"):
            intrinsics.ignore_dash_labels = self._cfg.ignore_dash_labels
        if not getattr(intrinsics, "labels", None) and self._cfg.labels_path:
            self._load_labels_from_path(intrinsics)
        labels = getattr(intrinsics, "labels", None)
        if not labels:
            labels = list(self._cfg.target_labels) or ["person"]
        if getattr(intrinsics, "ignore_dash_labels", False):
            labels = [label for label in labels if label and label != "-"]
        intrinsics.labels = labels
        self._intrinsics = intrinsics
        self._labels = tuple(labels)

    def _load_labels_from_path(self, intrinsics: Any) -> None:
        path = Path(self._cfg.labels_path or "")
        if not path.exists():
            LOGGER.warning("IMX500 labels file not found: %s", path)
            return
        with open(path, "r", encoding="utf-8") as handle:
            labels = [line.strip() for line in handle.readlines() if line.strip()]
        intrinsics.labels = labels

    def _parse_imx500_outputs(
        self,
        metadata: Optional[dict],
        picamera: Optional[object],
        width: int,
        height: int,
    ) -> List[Detection]:
        if metadata is None or self._imx500 is None:
            return []
        try:
            np_outputs = self._imx500.get_outputs(metadata, add_batch=True)
        except Exception:
            LOGGER.debug("Unable to retrieve IMX500 outputs", exc_info=True)
            return []
        if np_outputs is None:
            return []
        try:
            input_w, input_h = self._imx500.get_input_size()
        except Exception:
            input_w, input_h = width, height
        intr = self._intrinsics
        postprocess = getattr(intr, "postprocess", None) if intr else None
        if postprocess == "nanodet" and postprocess_nanodet_detection is not None:
            dets = postprocess_nanodet_detection(
                outputs=np_outputs[0],
                conf=self._cfg.min_confidence,
                iou_thres=self._iou,
                max_out_dets=self._max_detections,
            )
            if not dets:
                return []
            boxes, scores, classes = dets[0]
            if scale_boxes is not None:
                boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes = np_outputs[0]
            scores = np_outputs[1]
            classes = np_outputs[2]
        boxes_arr = self._reshape_boxes(boxes)
        scores_arr = np.asarray(scores).reshape(-1)
        classes_arr = np.asarray(classes).reshape(-1)
        if intr and getattr(intr, "bbox_normalization", False):
            boxes_arr = boxes_arr / float(max(input_h, 1))
        if intr and getattr(intr, "bbox_order", "yx") == "xy":
            boxes_arr = boxes_arr[:, [1, 0, 3, 2]]
        detections: List[Detection] = []
        for box, score, cls in zip(boxes_arr, scores_arr, classes_arr):
            score_f = float(score)
            if score_f < self._cfg.min_confidence:
                continue
            coords = self._convert_imx500_coords(
                box,
                metadata,
                picamera,
                width,
                height,
                input_w,
                input_h,
            )
            if not coords:
                continue
            label = self._label_from_class(cls)
            detections.append(Detection(label=label, confidence=score_f, bbox_xyxy=coords))
            if len(detections) >= self._max_detections:
                break
        return detections

    @staticmethod
    def _reshape_boxes(boxes: Any) -> np.ndarray:
        arr = np.asarray(boxes)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[0] == 4 and arr.shape[1] != 4:
            arr = arr.T
        arr = arr.reshape(-1, 4)
        return arr.astype(float)

    def _convert_imx500_coords(
        self,
        box: Sequence[float],
        metadata: Optional[dict],
        picamera: Optional[object],
        frame_width: int,
        frame_height: int,
        input_w: int,
        input_h: int,
    ) -> Optional[Sequence[float]]:
        if self._imx500 is not None and picamera is not None:
            try:
                x, y, w, h = self._imx500.convert_inference_coords(box, metadata, picamera)  # type: ignore[attr-defined]
                return [
                    self._clip(x, 0, frame_width),
                    self._clip(y, 0, frame_height),
                    self._clip(x + w, 0, frame_width),
                    self._clip(y + h, 0, frame_height),
                ]
            except Exception:
                LOGGER.debug("convert_inference_coords failed; using scaled coords", exc_info=True)
        arr = np.asarray(box, dtype=float).flatten()
        if arr.size != 4:
            return None
        y0, x0, y1, x1 = arr
        normalized = bool(self._intrinsics and getattr(self._intrinsics, "bbox_normalization", False))
        if normalized:
            x0 *= frame_width
            x1 *= frame_width
            y0 *= frame_height
            y1 *= frame_height
        else:
            scale_x = frame_width / float(max(input_w, 1))
            scale_y = frame_height / float(max(input_h, 1))
            x0 *= scale_x
            x1 *= scale_x
            y0 *= scale_y
            y1 *= scale_y
        return [
            self._clip(x0, 0, frame_width),
            self._clip(y0, 0, frame_height),
            self._clip(x1, 0, frame_width),
            self._clip(y1, 0, frame_height),
        ]

    def _label_from_class(self, cls_index: Any) -> str:
        try:
            idx = int(cls_index)
        except (TypeError, ValueError):
            return str(cls_index)
        if 0 <= idx < len(self._labels):
            return self._labels[idx]
        return str(idx)

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
