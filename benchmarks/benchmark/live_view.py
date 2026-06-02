"""Frame stash for live viewing — writes to tmpfs, served by a separate process.

The previous design ran Flask MJPEG in the benchmark process and contended
with libfranka's RT threads, causing hangs. This refactor reduces the
in-process cost to one JPEG encode + atomic file write per inference step.
A standalone server (`benchmarks/scripts/serve_live.py`) reads from these
paths and serves the MJPEG dashboard.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

EXT_PATH = "/dev/shm/mex5_ext.jpg"
WRIST_PATH = "/dev/shm/mex5_wrist.jpg"


def _atomic_write_jpeg(path: str, frame_rgb: np.ndarray) -> None:
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(jpg.tobytes())
    os.replace(tmp, path)


def update(external: np.ndarray, wrist: np.ndarray) -> None:
    _atomic_write_jpeg(EXT_PATH, external)
    _atomic_write_jpeg(WRIST_PATH, wrist)


def start(*_args, **_kwargs) -> None:
    """No-op kept for backwards compatibility; live viewing is now external."""
    return None
