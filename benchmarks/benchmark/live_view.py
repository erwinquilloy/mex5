"""In-process MJPEG server for live viewing during a benchmark run.

The benchmark grabs frames for inference. We stash a copy here and serve them
as MJPEG so a browser can watch without opening the cameras a second time
(which fails since v4l2/pyrealsense2 hold them exclusively).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response

_latest_ext: Optional[np.ndarray] = None
_latest_wrist: Optional[np.ndarray] = None
_lock = threading.Lock()


def update(external: np.ndarray, wrist: np.ndarray) -> None:
    global _latest_ext, _latest_wrist
    with _lock:
        _latest_ext = external.copy()
        _latest_wrist = wrist.copy()


def _encode(frame_rgb: np.ndarray) -> Optional[bytes]:
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    return jpg.tobytes() if ok else None


def _mjpeg(which: str):
    while True:
        with _lock:
            f = _latest_ext if which == "ext" else _latest_wrist
        if f is None:
            time.sleep(0.1)
            continue
        jpg = _encode(f)
        if jpg is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(1 / 15)


def _make_app() -> Flask:
    app = Flask(__name__)

    @app.route("/ext")
    def ext():
        return Response(_mjpeg("ext"),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/wrist")
    def wrist():
        return Response(_mjpeg("wrist"),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def home():
        return ("<h3>MolmoAct2-DROID benchmark - live cams</h3>"
                "<div style='display:flex;gap:1em;font-family:sans-serif'>"
                "<div><div>external (tripod)</div><img src='/ext' width='640'></div>"
                "<div><div>wrist (D455)</div><img src='/wrist' width='640'></div>"
                "</div>")

    return app


def start(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the MJPEG server in a daemon thread. Idempotent."""
    if getattr(start, "_started", False):
        return
    start._started = True
    app = _make_app()
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True, use_reloader=False),
        daemon=True,
    )
    t.start()
