"""Central configuration for Hanger Monitor."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env-var helpers
# ---------------------------------------------------------------------------

def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CameraSettings:
    sensor: str = "picamera3"
    device: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 15
    queue_size: int = 10


@dataclass
class Yolov8Settings:
    """Pi Camera v3 / YOLOv8 TFLite settings."""
    model_path: str = "models/yolov8n_int8.tflite"
    input_size: int = 640
    min_confidence: float = 0.45
    iou: float = 0.45
    target_labels: Tuple[str, ...] = ("person",)
    overlay: bool = True


@dataclass
class RecordingSettings:
    base_dir: Path = Path("videos")
    codec: str = "mp4v"
    extension: str = ".mp4"
    pre_buffer_seconds: float = 1.0
    grace_period_seconds: float = 2.5
    retention_days: int = 14


@dataclass
class WebSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


@dataclass
class Settings:
    sensor: str = "picamera3"
    camera: CameraSettings = field(default_factory=CameraSettings)
    yolov8: Yolov8Settings = field(default_factory=Yolov8Settings)
    recording: RecordingSettings = field(default_factory=RecordingSettings)
    web: WebSettings = field(default_factory=WebSettings)

    @property
    def video_root(self) -> Path:
        self.recording.base_dir.mkdir(parents=True, exist_ok=True)
        return self.recording.base_dir


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _tuple_of_strings(value: object) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    if isinstance(value, str):
        return tuple(s.strip() for s in value.split(",") if s.strip())
    return ()


def load_settings(yaml_path: str | Path = "config.yaml") -> Settings:
    data = _load_yaml(Path(yaml_path))

    sensor = os.getenv("SENSOR", data.get("sensor", "picamera3"))

    cam_d = data.get("camera", {})
    camera = CameraSettings(
        sensor=sensor,
        device=int(os.getenv("CAMERA_DEVICE", cam_d.get("device", 0))),
        width=int(os.getenv("CAMERA_WIDTH", cam_d.get("width", 1280))),
        height=int(os.getenv("CAMERA_HEIGHT", cam_d.get("height", 720))),
        fps=int(os.getenv("CAMERA_FPS", cam_d.get("fps", 15))),
        queue_size=int(os.getenv("FRAME_QUEUE_SIZE", cam_d.get("queue_size", 10))),
    )

    yolo_d = data.get("yolov8", {})
    yolo_labels_raw = os.getenv("YOLO_TARGET_LABELS")
    yolo_labels: Tuple[str, ...] = (
        _tuple_of_strings(yolo_labels_raw) if yolo_labels_raw else _tuple_of_strings(yolo_d.get("target_labels", ["person"]))
    )
    yolov8 = Yolov8Settings(
        model_path=os.getenv("YOLO_MODEL_PATH", yolo_d.get("model_path", "models/yolov8n_int8.tflite")),
        input_size=int(os.getenv("YOLO_INPUT_SIZE", yolo_d.get("input_size", 640))),
        min_confidence=float(os.getenv("YOLO_MIN_CONFIDENCE", yolo_d.get("min_confidence", 0.45))),
        iou=float(os.getenv("YOLO_IOU", yolo_d.get("iou", 0.45))),
        target_labels=yolo_labels,
        overlay=_env_bool("YOLO_DRAW_OVERLAY", yolo_d.get("overlay", True)),
    )

    rec_d = data.get("recording", {})
    recording = RecordingSettings(
        base_dir=Path(os.getenv("VIDEO_DIR", rec_d.get("video_dir", "videos"))),
        codec=os.getenv("VIDEO_CODEC", rec_d.get("codec", "mp4v")),
        extension=os.getenv("VIDEO_EXTENSION", rec_d.get("extension", ".mp4")),
        pre_buffer_seconds=float(os.getenv("VIDEO_PREBUFFER", rec_d.get("pre_buffer_seconds", 1.0))),
        grace_period_seconds=float(os.getenv("VIDEO_GRACE_PERIOD", rec_d.get("grace_period_seconds", 2.5))),
        retention_days=int(os.getenv("VIDEO_RETENTION_DAYS", rec_d.get("retention_days", 14))),
    )

    web_d = data.get("web", {})
    web = WebSettings(
        host=os.getenv("SERVER_HOST", web_d.get("host", "0.0.0.0")),
        port=int(os.getenv("SERVER_PORT", web_d.get("port", 8000))),
        debug=_env_bool("SERVER_DEBUG", web_d.get("debug", False)),
    )

    return Settings(
        sensor=sensor,
        camera=camera,
        yolov8=yolov8,
        recording=recording,
        web=web,
    )


settings: Settings = load_settings()
