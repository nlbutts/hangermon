"""Background stitcher to convert BMP events to MP4."""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import shutil
from datetime import datetime
from pathlib import Path

from ..config import RecordingSettings, CameraSettings

LOGGER = logging.getLogger(__name__)

class BackgroundStitcher:
    """Monitors event directories and uses ffmpeg to stitch BMPs into MP4s."""

    def __init__(self, rec_cfg: RecordingSettings, cam_cfg: CameraSettings) -> None:
        self._cfg = rec_cfg
        self._cam_cfg = cam_cfg
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._events_dir = Path("/tmp/hangermon_events")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._events_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="bg-stitcher", daemon=True)
        self._thread.start()
        LOGGER.info("Background stitcher started.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        LOGGER.info("Background stitcher stopped.")

    def _loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(2.0)
            if not self._events_dir.exists():
                continue
                
            for event_dir in self._events_dir.iterdir():
                if self._stop.is_set():
                    break
                if not event_dir.is_dir():
                    continue
                    
                ready_file = event_dir / "ready.json"
                if ready_file.exists():
                    try:
                        self._process_event(event_dir, ready_file)
                    except Exception as e:
                        LOGGER.error("Failed to stitch event %s: %s", event_dir, e, exc_info=True)
                        ready_file.rename(event_dir / "failed.json")

    def _process_event(self, event_dir: Path, ready_file: Path) -> None:
        with open(ready_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        fps = meta.get("fps", self._cam_cfg.fps)
        ts_iso = meta.get("timestamp", datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(ts_iso)
        except Exception:
            dt = datetime.now()
            
        out_root = self._cfg.base_dir / dt.strftime("%Y/%m/%d")
        out_root.mkdir(parents=True, exist_ok=True)
        
        base_name = dt.strftime("hanger-%H%M%S")
        out_full = (out_root / f"{base_name}_full").with_suffix(self._cfg.extension)
        out_small = (out_root / f"{base_name}_small").with_suffix(self._cfg.extension)
        
        # Sort and natively rename to sequence to bypass missing glob support
        bmp_files = sorted(event_dir.glob("*.bmp"))
        if not bmp_files:
            LOGGER.warning("No BMP files found in event directory %s. Skipping.", event_dir)
            shutil.rmtree(event_dir)
            return
            
        for i, bmp_path in enumerate(bmp_files):
            new_path = event_dir / f"{i:05d}.bmp"
            bmp_path.rename(new_path)
        
        cmd_full = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", "%05d.bmp",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_full.absolute())
        ]
        
        LOGGER.info("Stitching full res MP4 for event %s -> %s", event_dir.name, out_full)
        result = subprocess.run(cmd_full, cwd=event_dir, capture_output=True, text=True)
        if result.returncode != 0:
            LOGGER.error("FFmpeg full-res failed: %s", result.stderr)
            result.check_returncode()
        
        cmd_small = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", "%05d.bmp",
            "-vf", f"scale={self._cam_cfg.resize_width}:{self._cam_cfg.resize_height}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_small.absolute())
        ]
        
        LOGGER.info("Stitching low res MP4 for event %s -> %s", event_dir.name, out_small)
        result = subprocess.run(cmd_small, cwd=event_dir, capture_output=True, text=True)
        if result.returncode != 0:
            LOGGER.error("FFmpeg small-res failed: %s", result.stderr)
            result.check_returncode()
        
        meta_out = out_root / f"{base_name}_small.json"
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            
        LOGGER.info("Deleting event directory %s", event_dir)
        shutil.rmtree(event_dir)

__all__ = ["BackgroundStitcher"]
