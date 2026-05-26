"""Camera source: RealSense D457 with a file-replay fallback for dev/CI."""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Optional, Protocol

from PIL import Image


class Camera(Protocol):
    def grab(self) -> Image.Image: ...
    def close(self) -> None: ...


class RealsenseCamera:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        import pyrealsense2 as rs  # imported lazily so dev/CI without the SDK still works
        self._rs = rs
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self._pipeline.start(cfg)
        for _ in range(5):                        # auto-exposure warmup
            self._pipeline.wait_for_frames()

    def grab(self) -> Image.Image:
        import numpy as np
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("no color frame")
        arr = np.asanyarray(color.get_data())
        return Image.fromarray(arr, mode="RGB")

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


class FileCamera:
    """Cycle through PNG/JPGs in a directory. Useful for repeatable dry-runs."""

    def __init__(self, folder: str | Path):
        self._files = sorted(Path(folder).glob("*.png")) + sorted(Path(folder).glob("*.jpg"))
        if not self._files:
            raise FileNotFoundError(f"no images in {folder}")
        self._idx = 0

    def grab(self) -> Image.Image:
        path = self._files[self._idx % len(self._files)]
        self._idx += 1
        return Image.open(path).convert("RGB")

    def close(self) -> None:
        pass


def make_camera() -> Camera:
    """Pick RealSense if available, else fall back to FRANKA_BENCH_IMAGES dir."""
    folder = os.environ.get("FRANKA_BENCH_IMAGES")
    if folder:
        return FileCamera(folder)
    return RealsenseCamera()


def image_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
