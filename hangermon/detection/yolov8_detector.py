"""YOLOv8 TFLite CPU detector for the Pi Camera v3 pipeline."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import Yolov8Settings
from .detector import Detection, DetectionResult

LOGGER = logging.getLogger(__name__)

# COCO class names (80 classes). Index 0 is "person".
COCO_CLASSES: Tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


def _load_tflite_interpreter(model_path: str):
    """Load TFLite interpreter; returns None if runtime unavailable."""
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
        LOGGER.info("Loaded TFLite model from %s", model_path)
        return interp
    except ImportError:
        try:
            # Fallback: full TensorFlow (slower but usually pip-installable)
            import tensorflow as tf  # type: ignore
            interp = tf.lite.Interpreter(model_path=model_path)
            interp.allocate_tensors()
            LOGGER.info("Loaded TFLite model via TensorFlow from %s", model_path)
            return interp
        except ImportError:
            LOGGER.error(
                "Neither tflite_runtime nor tensorflow is installed. "
                "Install tflite-runtime to use the YOLOv8 detector."
            )
            return None
    except Exception as exc:
        LOGGER.error("Failed to load TFLite model %s: %s", model_path, exc)
        return None


class Yolov8Detector:
    """
    CPU-side YOLOv8n detector via TFLite.

    Expected model output tensor shape: [1, 84, 8400]
      - axis 1 rows 0-3: cx, cy, w, h (normalised 0-1)
      - axis 1 rows 4-83: per-class scores (80 COCO classes)
    """

    def __init__(self, cfg: Yolov8Settings) -> None:
        self._cfg = cfg
        self._target_labels = set(cfg.target_labels) or {"person"}
        self._interp = None
        model_path = Path(cfg.model_path)
        if model_path.exists():
            self._interp = _load_tflite_interpreter(str(model_path))
        else:
            LOGGER.warning(
                "YOLOv8 model not found at %s. "
                "Download: https://github.com/ultralytics/assets/releases",
                model_path,
            )

    # ------------------------------------------------------------------
    # Public API (same signature as Imx500Detector.detect)
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        metadata: Optional[dict] = None,  # unused for this detector
        picamera: Optional[object] = None,  # unused
    ) -> DetectionResult:
        h, w = frame.shape[:2]
        if self._interp is None:
            # No model: return empty result
            return DetectionResult([], frame.copy(), 0.0, False)

        t0 = time.perf_counter()
        detections = self._run_inference(frame, w, h)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        annotated = frame.copy()
        if self._cfg.overlay:
            for det in detections:
                self._draw_box(annotated, det)

        human_present = any(d.label in self._target_labels for d in detections)
        return DetectionResult(detections, annotated, elapsed_ms, human_present)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_inference(self, frame: np.ndarray, frame_w: int, frame_h: int) -> List[Detection]:
        input_size = self._cfg.input_size

        interp = self._interp
        in_details = interp.get_input_details()[0]
        out_details = interp.get_output_details()[0]

        # 1. Prepare Input
        blob = cv2.resize(frame, (input_size, input_size))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)

        # Handle quantization if input is not float32
        if in_details["dtype"] != np.float32:
            scale, zero_point = in_details.get("quantization", (0.0, 0))
            if scale == 0:
                scale = 1.0
            blob = (blob / scale + zero_point).astype(in_details["dtype"])
        else:
            blob = blob.astype(np.float32) / 255.0

        blob = np.expand_dims(blob, axis=0)
        interp.set_tensor(in_details["index"], blob)
        interp.invoke()

        # 2. Get Output
        raw = interp.get_tensor(out_details["index"])
        
        # Handle quantization for output if necessary
        if out_details["dtype"] != np.float32:
            scale, zero_point = out_details.get("quantization", (0.0, 0))
            if scale != 0:
                raw = (raw.astype(np.float32) - zero_point) * scale

        return self._postprocess(raw, frame_w, frame_h, input_size)

    def _postprocess(
        self,
        raw: np.ndarray,
        frame_w: int,
        frame_h: int,
        input_size: int,
    ) -> List[Detection]:
        # raw shape: [1, 84, 8400]
        preds = raw[0].T  # [8400, 84]

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        class_scores = preds[:, 4:]          # [8400, 80]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Filter by confidence
        mask = confidences >= self._cfg.min_confidence
        if not mask.any():
            return []

        cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Convert normalised cx/cy/w/h → pixel x1,y1,x2,y2
        scale_x = frame_w / input_size
        scale_y = frame_h / input_size
        x1 = (cx - bw / 2) * scale_x
        y1 = (cy - bh / 2) * scale_y
        x2 = (cx + bw / 2) * scale_x
        y2 = (cy + bh / 2) * scale_y

        # NMS via OpenCV
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        scores_list = confidences.tolist()

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores_list,
            self._cfg.min_confidence,
            self._cfg.iou,
        )
        if isinstance(indices, tuple) or len(indices) == 0:
            return []

        indices = np.array(indices).flatten()
        detections: List[Detection] = []
        for idx in indices:
            cls_id = int(class_ids[idx])
            label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
            conf = float(confidences[idx])
            bbox: Sequence[float] = [
                float(np.clip(x1[idx], 0, frame_w)),
                float(np.clip(y1[idx], 0, frame_h)),
                float(np.clip(x2[idx], 0, frame_w)),
                float(np.clip(y2[idx], 0, frame_h)),
            ]
            detections.append(Detection(label=label, confidence=conf, bbox_xyxy=bbox))

        return detections

    @staticmethod
    def _draw_box(frame: np.ndarray, detection: Detection) -> None:
        x1, y1, x2, y2 = map(int, detection.bbox_xyxy)
        color = (0, 204, 102) if detection.label == "person" else (0, 128, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.label}:{detection.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


__all__ = ["Yolov8Detector"]
