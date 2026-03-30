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
    device: int = _env_int("CAMERA_DEVICE", 0)
    width: int = _env_int("CAMERA_WIDTH", 1280)
    height: int = _env_int("CAMERA_HEIGHT", 720)
    fps: int = _env_int("CAMERA_FPS", 15)
    use_picamera2: bool = _env_bool("USE_PICAMERA2", True)
    queue_size: int = _env_int("FRAME_QUEUE_SIZE", 10)


@dataclass(slots=True)
class DetectionSettings:
    metadata_path: str = _env("IMX500_METADATA_PATH", "imx500.results")
    label_field: str = _env("IMX500_LABEL_FIELD", "label")
    score_field: str = _env("IMX500_SCORE_FIELD", "score")
    bbox_field: str = _env("IMX500_BBOX_FIELD", "bbox")
    bbox_format: str = _env("IMX500_BBOX_FORMAT", "xywh")
    bbox_normalized: bool = _env_bool("IMX500_BBOX_NORMALIZED", True)
    min_confidence: float = _env_float("IMX500_MIN_CONFIDENCE", 0.45)
    target_labels: Tuple[str, ...] = _env_csv("IMX500_TARGET_LABELS", "person")
    overlay: bool = _env_bool("IMX500_DRAW_OVERLAY", True)
    latency_field: Optional[str] = _env("IMX500_LATENCY_FIELD", "") or None
    model_path: str = _env(
        "IMX500_MODEL_PATH",
        "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
    )
    iou: float = _env_float("IMX500_IOU", 0.65)
    max_detections: int = _env_int("IMX500_MAX_DETECTIONS", 10)
    postprocess: str = _env("IMX500_POSTPROCESS", "")
    labels_path: Optional[str] = _env("IMX500_LABELS_PATH", "") or None
    ignore_dash_labels: bool = _env_bool("IMX500_IGNORE_DASH_LABELS", True)


@dataclass(slots=True)
class YoloSettings:
    model_path: str = "yolo_det_rpi/yolov8n.onnx"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    target_labels: Tuple[str, ...] = ("person",)
    detections_required: int = 3
    inference_interval_seconds: float = 0.5
    server_port: int = 5555


@dataclass(slots=True)
class RecordingSettings:
    base_dir: Path = Path(_env("VIDEO_DIR", "videos"))
    codec: str = _env("VIDEO_CODEC", "mp4v")
    extension: str = _env("VIDEO_EXTENSION", ".mp4")
    pre_buffer_seconds: float = _env_float("VIDEO_PREBUFFER", 1.0)
    grace_period_seconds: float = _env_float("VIDEO_GRACE_PERIOD", 2.5)
    retention_days: int = _env_int("VIDEO_RETENTION_DAYS", 14)


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
        detections_required=yolo_config.get("detections_required", 3),
        inference_interval_seconds=yolo_config.get("inference_interval_seconds", 0.5),
        server_port=yolo_config.get("server_port", 5555),
    )

    recording_config = yaml_config.get("recording", {})
    recording = RecordingSettings(
        base_dir=Path(recording_config.get("video_dir", "videos")),
        codec=recording_config.get("video_codec", "mp4v"),
        extension=recording_config.get("video_extension", ".mp4"),
        pre_buffer_seconds=recording_config.get("video_prebuffer", 1.0),
        grace_period_seconds=recording_config.get("video_grace_period", 2.5),
        retention_days=recording_config.get("video_retention_days", 14),
    )

    camera_config = yaml_config.get("camera", {})
    camera = CameraSettings(
        device=camera_config.get("device", 0),
        width=camera_config.get("width", 1280),
        height=camera_config.get("height", 720),
        fps=camera_config.get("fps", 15),
        use_picamera2=camera_config.get("use_picamera2", True),
        queue_size=camera_config.get("queue_size", 10),
    )

    web_config = yaml_config.get("web", {})
    web = WebSettings(
        host=web_config.get("host", "0.0.0.0"),
        port=web_config.get("port", 8000),
        debug=web_config.get("debug", False),
    )

    return Settings(camera=camera, yolo=yolo, recording=recording, web=web)


settings = load_config()
