"""Central configuration for Hanger Monitor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import logging
import os
import yaml

LOGGER = logging.getLogger(__name__)


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(key: str, default: str) -> Tuple[str, ...]:
    raw = os.getenv(key, default)
    items = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not items:
        items = [chunk.strip() for chunk in default.split(",") if chunk.strip()]
    return tuple(items)


def _load_yaml_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        LOGGER.warning("Failed to load YAML config: %s", e)
        return {}


@dataclass(slots=True)
class CameraSettings:
    device: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    use_picamera2: bool = True
    queue_size: int = 5
    circular_buffer_duration_sec: int = 60
    resize_width: int = 640
    resize_height: int = 480
    h264_bitrate: int = 5000000
    minimum_clip: int = 30
    pre_trigger_time: int = 15


@dataclass(slots=True)
class YoloSettings:
    model_path: str = "yolo_det_rpi/yolov8n.onnx"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    target_labels: Tuple[str, ...] = ("person",)
    detections_required: int = 2
    inference_interval_seconds: float = 1.0
    server_port: int = 5555


@dataclass(slots=True)
class RecordingSettings:
    base_dir: Path = Path("videos")
    extension: str = ".mp4"
    retention_days: int = 14


@dataclass(slots=True)
class WebSettings:
    host: str = _env("SERVER_HOST", "0.0.0.0")
    port: int = _env_int("SERVER_PORT", 8000)
    debug: bool = _env_bool("SERVER_DEBUG", False)


@dataclass(slots=True)
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    yolo: YoloSettings = field(default_factory=YoloSettings)
    recording: RecordingSettings = field(default_factory=RecordingSettings)
    web: WebSettings = field(default_factory=WebSettings)
    version: str = "1.0.0"

    @property
    def video_root(self) -> Path:
        self.recording.base_dir.mkdir(parents=True, exist_ok=True)
        return self.recording.base_dir



def load_config(config_path: Optional[Path] = None) -> Settings:
    """Load configuration from YAML file or use defaults."""
    if config_path is None:
        config_path = Path("config.yaml")

    yaml_config = _load_yaml_config(config_path)

    yolo_config = yaml_config.get("yolo", {})
    yolo = YoloSettings(
        model_path=yolo_config.get("model_path", "yolo_det_rpi/yolov8n.onnx"),
        confidence_threshold=yolo_config.get("confidence_threshold", 0.5),
        nms_threshold=yolo_config.get("nms_threshold", 0.45),
        target_labels=tuple(yolo_config.get("target_labels", ["person"])),
        detections_required=yolo_config.get("detections_required", 2),
        inference_interval_seconds=yolo_config.get("inference_interval_seconds", 1.0),
        server_port=yolo_config.get("server_port", 5555),
    )

    recording_config = yaml_config.get("recording", {})
    recording = RecordingSettings(
        base_dir=Path(recording_config.get("video_dir", "videos")),
        extension=recording_config.get("video_extension", ".mp4"),
        retention_days=recording_config.get("video_retention_days", 14),
    )

    camera_config = yaml_config.get("camera", {})
    camera = CameraSettings(
        device=camera_config.get("device", 0),
        width=camera_config.get("width", 1920),
        height=camera_config.get("height", 1080),
        fps=camera_config.get("fps", 30),
        use_picamera2=camera_config.get("use_picamera2", True),
        queue_size=camera_config.get("queue_size", 5),
        circular_buffer_duration_sec=camera_config.get("circular_buffer_duration_sec", 60),
        resize_width=camera_config.get("resize_width", 640),
        resize_height=camera_config.get("resize_height", 480),
        h264_bitrate=camera_config.get("h264_bitrate", 5000000),
        minimum_clip=camera_config.get("minimum_clip", 30),
        pre_trigger_time=camera_config.get("pre_trigger_time", 15),
    )

    web_config = yaml_config.get("web", {})
    web = WebSettings(
        host=web_config.get("host", "0.0.0.0"),
        port=web_config.get("port", 8000),
        debug=web_config.get("debug", False),
    )

    return Settings(
        camera=camera, 
        yolo=yolo, 
        recording=recording, 
        web=web,
        version=yaml_config.get("version", "1.0.0")
    )



settings = load_config()
