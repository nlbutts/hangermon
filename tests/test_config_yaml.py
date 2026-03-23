"""Tests for the YAML-based configuration loader."""
import os
import textwrap
from pathlib import Path

import pytest

from hangermon.config import load_settings


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def test_load_yaml_imx500_defaults(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        """\
        sensor: imx500
        camera:
          width: 1920
          height: 1080
          fps: 30
        """,
    )
    cfg = load_settings(cfg_path)
    assert cfg.sensor == "imx500"
    assert cfg.camera.width == 1920
    assert cfg.camera.height == 1080
    assert cfg.camera.fps == 30


def test_load_yaml_picamera3_sensor(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        """\
        sensor: picamera3
        yolov8:
          model_path: /tmp/yolov8n.tflite
          min_confidence: 0.3
          target_labels:
            - person
            - cat
        """,
    )
    cfg = load_settings(cfg_path)
    assert cfg.sensor == "picamera3"
    assert cfg.yolov8.model_path == "/tmp/yolov8n.tflite"
    assert cfg.yolov8.min_confidence == pytest.approx(0.3)
    assert "person" in cfg.yolov8.target_labels
    assert "cat" in cfg.yolov8.target_labels


def test_env_var_overrides_yaml(tmp_path, monkeypatch):
    cfg_path = _write_yaml(
        tmp_path,
        """\
        sensor: imx500
        camera:
          fps: 10
        """,
    )
    monkeypatch.setenv("CAMERA_FPS", "25")
    monkeypatch.setenv("SENSOR", "picamera3")
    cfg = load_settings(cfg_path)
    assert cfg.camera.fps == 25
    assert cfg.sensor == "picamera3"


def test_missing_yaml_uses_defaults():
    cfg = load_settings("/nonexistent/path/config.yaml")
    # Should not raise; should return sane defaults
    assert cfg.sensor == "picamera3"
    assert cfg.camera.width == 1280


def test_recording_settings(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        """\
        recording:
          video_dir: /tmp/vids
          retention_days: 7
        """,
    )
    cfg = load_settings(cfg_path)
    assert str(cfg.recording.base_dir) == "/tmp/vids"
    assert cfg.recording.retention_days == 7


def test_web_settings(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        """\
        web:
          port: 9000
          host: 127.0.0.1
        """,
    )
    cfg = load_settings(cfg_path)
    assert cfg.web.port == 9000
    assert cfg.web.host == "127.0.0.1"
