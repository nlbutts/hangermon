"""Flask entrypoint for Hanger Monitor."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Generator

from flask import Flask, Response, abort, jsonify, render_template, send_file

from hangermon.config import settings
from hangermon.service import MonitorService
from hangermon.storage import catalog

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

monitor = MonitorService(settings)


def create_app() -> Flask:
    app = Flask(__name__)
    monitor.start()

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/api/status")
    def api_status():
        return jsonify(monitor.status_snapshot())

    @app.route("/api/clips")
    def api_clips():
        records = []
        base = settings.recording.base_dir.resolve()
        for clip in catalog.list_clips(base):
            try:
                relative = clip.path.resolve().relative_to(base)
            except ValueError:
                relative = clip.path.name
            records.append(clip.to_dict(str(relative)))
        return jsonify({"clips": records})

    @app.route("/api/clips/<path:relative>")
    def api_clip(relative: str):
        root = settings.recording.base_dir.resolve()
        target = (root / relative).resolve()
        if not str(target).startswith(str(root)) or not target.exists():
            abort(404)
        return send_file(target)

    @app.route("/stream.mjpg")
    def stream() -> Response:
        return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def _mjpeg_generator() -> Generator[bytes, None, None]:
    boundary = b"--frame"
    while True:
        frame = monitor.latest_frame_bytes()
        if frame is None:
            time.sleep(0.1)
            continue
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)


if os.getenv("HANGERMON_SKIP_FLASK_INIT"):
    app = Flask(__name__)
else:
    app = create_app()

if __name__ == "__main__":  # pragma: no cover
    app.run(host=settings.web.host, port=settings.web.port, debug=settings.web.debug)
