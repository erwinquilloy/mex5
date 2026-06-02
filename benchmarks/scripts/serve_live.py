"""Standalone MJPEG dashboard for the MolmoAct2-DROID benchmark.

Reads the latest captured frames from /dev/shm (written by the benchmark via
`live_view.update`) and serves them as MJPEG. Decoupled from the benchmark
process so it can't starve the RT control threads.

Usage:
    python -m benchmarks.scripts.serve_live --port 8080
Then open http://<workstation-ip>:8080/ in a browser.
"""
from __future__ import annotations

import argparse
import time

from flask import Flask, Response

from benchmarks.benchmark.live_view import EXT_PATH, WRIST_PATH


def _read(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


def _mjpeg(path: str, fps: float):
    period = 1.0 / fps
    while True:
        b = _read(path)
        if b:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b + b"\r\n")
        time.sleep(period)


def make_app(fps: float = 10.0) -> Flask:
    app = Flask(__name__)

    @app.route("/ext")
    def ext():
        return Response(_mjpeg(EXT_PATH, fps),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/wrist")
    def wrist():
        return Response(_mjpeg(WRIST_PATH, fps),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def home():
        return ("<h3>MolmoAct2-DROID - live cams</h3>"
                "<div style='display:flex;gap:1em;font-family:sans-serif'>"
                "<div><div>external (tripod)</div><img src='/ext' width='640'></div>"
                "<div><div>wrist (D455)</div><img src='/wrist' width='640'></div>"
                "</div>")

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=10.0,
                    help="dashboard refresh rate (frames in tmpfs update at the benchmark's step rate)")
    args = ap.parse_args()
    app = make_app(fps=args.fps)
    app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
