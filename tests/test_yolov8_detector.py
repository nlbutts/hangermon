"""Tests for the YOLOv8 TFLite detector (mocked TFLite runtime)."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hangermon.config import Yolov8Settings


def _make_fake_output(cx=320.0, cy=240.0, bw=100.0, bh=80.0, cls_id=0, score=0.85):
    """
    Build a synthetic YOLOv8 output tensor [1, 84, 8400].
    One detection at given position with class 0 (person) and given score.
    """
    n_anchors = 8400
    tensor = np.zeros((1, 84, n_anchors), dtype=np.float32)
    # cx, cy, w, h in anchor 0
    tensor[0, 0, 0] = cx
    tensor[0, 1, 0] = cy
    tensor[0, 2, 0] = bw
    tensor[0, 3, 0] = bh
    # class score at class_id
    tensor[0, 4 + cls_id, 0] = score
    return tensor


def _load_detector_with_mock(model_path: str = "/fake/yolov8n.tflite"):
    """Patch file existence and tflite, return a Yolov8Detector."""
    # Build a minimal fake tflite_runtime module
    tflite_mod = types.ModuleType("tflite_runtime")
    interp_mod = types.ModuleType("tflite_runtime.interpreter")
    fake_interp = MagicMock()
    interp_mod.Interpreter = MagicMock(return_value=fake_interp)
    tflite_mod.interpreter = interp_mod
    sys.modules.setdefault("tflite_runtime", tflite_mod)
    sys.modules.setdefault("tflite_runtime.interpreter", interp_mod)

    cfg = Yolov8Settings(
        model_path=model_path,
        input_size=640,
        min_confidence=0.5,
        iou=0.45,
        target_labels=("person",),
    )

    with patch("pathlib.Path.exists", return_value=True):
        from hangermon.detection.yolov8_detector import Yolov8Detector
        detector = Yolov8Detector(cfg)

    # Replace the real interpreter with our mock
    detector._interp = fake_interp
    return detector, fake_interp


def test_yolov8_returns_empty_when_no_model():
    from hangermon.detection.yolov8_detector import Yolov8Detector

    cfg = Yolov8Settings(model_path="/nonexistent/model.tflite")
    detector = Yolov8Detector(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert result.human_present is False
    assert result.detections == []


def test_yolov8_detects_person():
    detector, fake_interp = _load_detector_with_mock()
    frame_h, frame_w = 480, 640

    # Configure mock to return our synthetic tensor
    output_tensor = _make_fake_output(cx=320.0, cy=240.0, bw=100.0, bh=80.0, cls_id=0, score=0.9)
    fake_interp.get_input_details.return_value = [{"index": 0, "dtype": np.float32}]
    fake_interp.get_output_details.return_value = [{"index": 1, "dtype": np.float32}]
    fake_interp.get_tensor.side_effect = lambda idx: output_tensor if idx == 1 else None

    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    result = detector.detect(frame)

    # Person should be detected
    assert result.human_present is True
    assert len(result.detections) >= 1
    det = result.detections[0]
    assert det.label == "person"
    assert det.confidence >= 0.5


def test_yolov8_skips_low_confidence():
    detector, fake_interp = _load_detector_with_mock()

    # Score below threshold (0.5)
    output_tensor = _make_fake_output(score=0.2)
    fake_interp.get_input_details.return_value = [{"index": 0, "dtype": np.float32}]
    fake_interp.get_output_details.return_value = [{"index": 1, "dtype": np.float32}]
    fake_interp.get_tensor.side_effect = lambda idx: output_tensor if idx == 1 else None

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)

    assert result.human_present is False
    assert result.detections == []


def test_make_detector_factory():
    """Factory should return Yolov8Detector."""
    from hangermon.detection import make_detector
    from hangermon.detection.yolov8_detector import Yolov8Detector
    from hangermon.config import load_settings

    cfg = load_settings("/nonexistent/config.yaml")
    detector = make_detector(cfg)
    assert isinstance(detector, Yolov8Detector)


def test_make_detector_factory_yolov8(monkeypatch):
    """Factory should return Yolov8Detector for picamera3 sensor."""
    from hangermon.detection import make_detector
    from hangermon.detection.yolov8_detector import Yolov8Detector
    from hangermon.config import load_settings

    # Ensure model_path doesn't exist so constructor doesn't error on load
    cfg = load_settings("/nonexistent/config.yaml")
    cfg.sensor = "picamera3"
    detector = make_detector(cfg)
    assert isinstance(detector, Yolov8Detector)
def test_yolov8_int8_quantization():
    """Test that int8 quantization parameters are applied."""
    detector, fake_interp = _load_detector_with_mock()
    frame_h, frame_w = 480, 640

    # Synthetic int8 output
    # scale = 0.1, zero_point = 10
    # True value 0.9 = (int_val - 10) * 0.1 => int_val = 19
    int_output = np.zeros((1, 84, 8400), dtype=np.int8)
    int_output[0, 4, 0] = 19
    
    fake_interp.get_input_details.return_value = [{"index": 0, "dtype": np.float32}]
    fake_interp.get_output_details.return_value = [{
        "index": 1, 
        "dtype": np.int8, 
        "quantization": (0.1, 10)
    }]
    fake_interp.get_tensor.side_effect = lambda idx: int_output if idx == 1 else None

    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    result = detector.detect(frame)

    assert result.human_present is True
    assert result.detections[0].confidence == pytest.approx(0.9)
