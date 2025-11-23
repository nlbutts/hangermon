import numpy as np

from hangermon.config import DetectionSettings
from hangermon.detection.detector import Imx500Detector


def test_imx500_detector_uses_metadata():
    cfg = DetectionSettings(
        metadata_path="imx500.results",
        target_labels=("person",),
        min_confidence=0.1,
        bbox_format="xywh",
        bbox_normalized=True,
        overlay=False,
    )
    detector = Imx500Detector(cfg)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    metadata = {
        "imx500": {
            "results": [
                {
                    "label": "person",
                    "score": 0.92,
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                }
            ]
        }
    }

    result = detector.detect(frame, metadata, None)

    assert result.human_present is True
    assert len(result.detections) == 1
    det = result.detections[0]
    # bbox: x1=20, y1=20, x2=80, y2=60 in pixel space
    assert det.label == "person"
    assert det.confidence == 0.92
    assert det.bbox_xyxy[0] == 20.0
    assert det.bbox_xyxy[1] == 20.0
    assert det.bbox_xyxy[2] == 80.0
    assert det.bbox_xyxy[3] == 60.0


def test_imx500_detector_falls_back_to_nested_results():
    cfg = DetectionSettings(
        metadata_path="missing.path",
        target_labels=("person",),
        min_confidence=0.1,
        bbox_format="xyxy",
        bbox_normalized=False,
        overlay=False,
    )
    detector = Imx500Detector(cfg)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    metadata = {
        "outer": {
            "inner": [
                {
                    "label": "person",
                    "score": 0.8,
                    "bbox": [5, 5, 25, 30],
                }
            ]
        }
    }

    result = detector.detect(frame, metadata, None)

    assert result.human_present is True
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.bbox_xyxy == [5.0, 5.0, 25.0, 30.0]


def test_imx500_detector_draws_boxes_for_people_without_overlay_flag():
    cfg = DetectionSettings(
        metadata_path="imx500.results",
        target_labels=("person",),
        min_confidence=0.1,
        bbox_format="xywh",
        bbox_normalized=True,
        overlay=False,
    )
    detector = Imx500Detector(cfg)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    original_frame = frame.copy()
    metadata = {
        "imx500": {
            "results": [
                {
                    "label": "person",
                    "score": 0.95,
                    "bbox": [0.25, 0.25, 0.3, 0.4],
                }
            ]
        }
    }

    result = detector.detect(frame, metadata, None)

    assert np.any(result.annotated_frame != original_frame)
