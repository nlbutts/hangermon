"""Flask entrypoint for Hanger Monitor."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Generator

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from hangermon.config import settings
from hangermon.sensehat import sensehat
from hangermon.storage import catalog

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logging.getLogger("werkzeug").setLevel(logging.INFO)

# Lazy singleton: only created once, on first access
_monitor = None


def _get_monitor():
    global _monitor
    if _monitor is None:
        from hangermon.service import MonitorService
        _monitor = MonitorService(settings)
    return _monitor


def create_app() -> Flask:
    app = Flask(__name__)

    # Only start the monitor in the actual worker process, not the reloader parent
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        _get_monitor().start()

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/api/status")
    def api_status():
        return jsonify(_get_monitor().status_snapshot())

    @app.route("/api/led", methods=["POST"])
    def api_set_led():
        data = request.get_json(silent=True) or {}
        intensity = int(data.get("intensity", 0))
        sensehat.set_led_intensity(intensity)
        return jsonify({"intensity": sensehat.get_led_intensity()})

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
        return send_file(target, as_attachment=True, download_name=target.name)

    @app.route("/stream.mjpg")
    def stream() -> Response:
        return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def _mjpeg_generator() -> Generator[bytes, None, None]:
    boundary = b"--frame"
    monitor = _get_monitor()
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
    # Never use the reloader with hardware camera - it forks the process and
    # the child tries to re-open the camera which is already held by the parent.
    app.run(host=settings.web.host, port=settings.web.port,
            debug=settings.web.debug, use_reloader=False)
