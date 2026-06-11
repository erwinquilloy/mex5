"""Preview-only dashboard: serves the serve_dashboard HTML with stubbed APIs.

Skips camera + driver init entirely so you can eyeball the layout without
hardware. The 3 camera tiles render a static "preview mode" placeholder;
buttons hit stubbed endpoints that report 'preview mode' rather than
actually moving the arm or launching motion_server.

    python -m benchmarks.scripts.preview_dashboard --port 8081

No env vars required. Runs anywhere Flask + numpy + opencv are installed.
"""
from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify

from benchmarks.benchmark import droid_tasks
from benchmarks.scripts.serve_dashboard import _INDEX_HTML

log = logging.getLogger("dashboard-preview")


def _placeholder_jpeg(label: str, width: int = 640, height: int = 480) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = 30
    img[:, :, 1] = 30
    img[:, :, 2] = 40
    cv2.putText(img, "PREVIEW MODE", (24, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 200, 220), 2, cv2.LINE_AA)
    cv2.putText(img, label, (24, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(img, "no camera attached", (24, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 140), 1, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return jpg.tobytes() if ok else b""


def make_app() -> Flask:
    app = Flask(__name__)
    placeholders = {
        "ext": _placeholder_jpeg("external 1"),
        "ext2": _placeholder_jpeg("external 2"),
        "wrist": _placeholder_jpeg("wrist RGB"),
    }

    def _stream(jpg: bytes):
        def gen():
            while True:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(1.0)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def index():
        # Force MJPEG so the HTML doesn't try to negotiate WebRTC.
        return _INDEX_HTML.replace("__WEBRTC_ENABLED__", "false")

    @app.route("/stream/ext")
    def stream_ext(): return _stream(placeholders["ext"])

    @app.route("/stream/ext2")
    def stream_ext2(): return _stream(placeholders["ext2"])

    @app.route("/stream/wrist_rgb")
    def stream_wrist(): return _stream(placeholders["wrist"])

    @app.route("/api/status")
    def api_status():
        return jsonify({
            "busy": False, "what": None,
            "last": {"ok": True, "info": "preview mode (no hardware)"},
            "transport": "rest",
            "progress": {"chunk": 0, "max": 0},
            "cam_offsets": {"dx": 0.0, "dy": 0.0, "dz": 0.0},
            "motion_server": {
                "configured": False, "bin_path": None,
                "running": False, "pid": None, "log_path": None,
            },
        })

    @app.route("/api/tasks")
    def api_tasks():
        return jsonify({"tasks": [
            {"task_id": t.task_id, "instruction": t.instruction,
             "paper_success_rate": t.paper_success_rate,
             "max_chunks": t.max_chunks, "trials": t.trials}
            for t in droid_tasks.all_tasks()
        ]})

    @app.route("/api/offsets", methods=["GET", "POST"])
    def api_offsets():
        return jsonify({"ok": True, "dx": 0.0, "dy": 0.0, "dz": 0.0,
                        "info": "preview mode (no driver)"})

    @app.route("/api/preferences")
    def api_preferences():
        return jsonify({
            "cam_offsets": {"dx": 0.0, "dy": 0.0, "dz": 0.0},
            "hold_min_dist_m": 0.08,
            "grasp_retry_limit": 3,
            "settings_path": "(preview mode — no file)",
        })

    def _preview(*_a, **_kw):
        return jsonify({"ok": True, "info": "preview mode (no hardware)"})

    app.add_url_rule("/api/home", view_func=_preview, methods=["POST"])
    app.add_url_rule("/api/task", view_func=_preview, methods=["POST"])
    app.add_url_rule("/api/stop", view_func=_preview, methods=["POST"])
    app.add_url_rule("/api/motion_server/<action>", view_func=_preview, methods=["POST"])
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log.info("dashboard preview on http://%s:%d  (no hardware)", args.host, args.port)
    make_app().run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
