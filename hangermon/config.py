"""Central configuration for Hanger Monitor."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import os


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
    port: int = _env_int("SERVER_PORT", 8080)
    debug: bool = _env_bool("SERVER_DEBUG", False)


@dataclass(slots=True)
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    recording: RecordingSettings = field(default_factory=RecordingSettings)
    web: WebSettings = field(default_factory=WebSettings)

    @property
    def video_root(self) -> Path:
        self.recording.base_dir.mkdir(parents=True, exist_ok=True)
        return self.recording.base_dir


settings = Settings()
