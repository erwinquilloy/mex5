"""Dashboard-owned camera capture: RealSense color + external webcam.

The benchmark's DualCamera is owned by the benchmark process. The dashboard
needs to keep running without a benchmark in flight, so it owns its own
RealSense pipeline.

A background thread continuously grabs frames and stashes JPEGs in memory
under a lock. MJPEG endpoints just yield the latest bytes; they never block
on hardware.

Cameras can only be opened by one process, so the dashboard and the CLI
benchmark cannot run at the same time.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

_VALID_ROT = (0, 90, 180, 270)


def _rotate(img: np.ndarray, deg: int) -> np.ndarray:
    if deg == 0:
        return img
    if deg == 90:
        return np.ascontiguousarray(np.rot90(img, k=1))
    if deg == 180:
        return np.ascontiguousarray(np.rot90(img, k=2))
    if deg == 270:
        return np.ascontiguousarray(np.rot90(img, k=3))
    raise ValueError(f"rotation must be one of {_VALID_ROT}, got {deg}")


def _apply_flips(img: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
    if flip_h:
        img = np.ascontiguousarray(img[:, ::-1, :])
    if flip_v:
        img = np.ascontiguousarray(img[::-1, :, :])
    return img


@dataclass
class _Latest:
    ext_jpg: Optional[bytes] = None
    wrist_rgb_jpg: Optional[bytes] = None
    ext_rgb: Optional[np.ndarray] = None      # raw arrays for inference
    wrist_rgb: Optional[np.ndarray] = None
    t_grab_ms: float = 0.0
    timestamp: float = 0.0


def _encode_jpeg(rgb: np.ndarray, quality: int = 80) -> Optional[bytes]:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return jpg.tobytes() if ok else None


class DashboardCamera:
    """Owns RealSense (color) + external webcam. Background-threaded."""

    def __init__(
        self,
        wrist_serial: Optional[str] = None,
        external_webcam_index: Optional[int] = None,
        width: int = 256,
        height: int = 256,
        fps: int = 30,
        external_rotation_deg: int = 0,
        external_flip_h: bool = False,
        external_flip_v: bool = False,
        wrist_rotation_deg: int = 0,
        wrist_flip_h: bool = False,
        wrist_flip_v: bool = False,
    ):
        import pyrealsense2 as rs

        if external_rotation_deg not in _VALID_ROT:
            raise ValueError(f"external_rotation_deg must be one of {_VALID_ROT}")
        if wrist_rotation_deg not in _VALID_ROT:
            raise ValueError(f"wrist_rotation_deg must be one of {_VALID_ROT}")
        self._ext_rot = int(external_rotation_deg)
        self._ext_flip_h = bool(external_flip_h)
        self._ext_flip_v = bool(external_flip_v)
        self._wrist_rot = int(wrist_rotation_deg)
        self._wrist_flip_h = bool(wrist_flip_h)
        self._wrist_flip_v = bool(wrist_flip_v)
        self._rs = rs
        self._pipe = rs.pipeline()
        cfg = rs.config()
        if wrist_serial:
            cfg.enable_device(wrist_serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self._pipe.start(cfg)
        for _ in range(5):
            self._pipe.wait_for_frames()

        self._ext: Optional[cv2.VideoCapture] = None
        if external_webcam_index is not None:
            cap = cv2.VideoCapture(external_webcam_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not cap.isOpened():
                raise RuntimeError(f"webcam {external_webcam_index} not open")
            for _ in range(5):
                cap.read()
            self._ext = cap

        self._lock = threading.Lock()
        self._latest = _Latest()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                frames = self._pipe.wait_for_frames()
                color = frames.get_color_frame()
                if color is None:
                    continue
                wrist_rgb = np.ascontiguousarray(np.asanyarray(color.get_data()))
            except Exception:
                time.sleep(0.05)
                continue

            ext_rgb: Optional[np.ndarray] = None
            if self._ext is not None:
                try:
                    ok, bgr = self._ext.read()
                    if ok:
                        ext_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    ext_rgb = None
            if ext_rgb is None:
                # duplicate mode (degraded but keeps the panel alive)
                ext_rgb = wrist_rgb.copy()
            # Match DualCamera's transforms so the dashboard shows (and feeds
            # to the model) the same image the CLI runner would.
            if self._ext_rot:
                ext_rgb = _rotate(ext_rgb, self._ext_rot)
            if self._ext_flip_h or self._ext_flip_v:
                ext_rgb = _apply_flips(ext_rgb, self._ext_flip_h, self._ext_flip_v)
            if self._wrist_rot:
                wrist_rgb = _rotate(wrist_rgb, self._wrist_rot)
            if self._wrist_flip_h or self._wrist_flip_v:
                wrist_rgb = _apply_flips(wrist_rgb, self._wrist_flip_h, self._wrist_flip_v)

            ext_jpg = _encode_jpeg(ext_rgb)
            wrist_jpg = _encode_jpeg(wrist_rgb)

            with self._lock:
                self._latest = _Latest(
                    ext_jpg=ext_jpg,
                    wrist_rgb_jpg=wrist_jpg,
                    ext_rgb=ext_rgb,
                    wrist_rgb=wrist_rgb,
                    t_grab_ms=(time.perf_counter() - t0) * 1000.0,
                    timestamp=time.time(),
                )

    # ----- accessors used by the Flask app -----

    def latest_jpegs(self) -> tuple[Optional[bytes], Optional[bytes]]:
        with self._lock:
            return self._latest.ext_jpg, self._latest.wrist_rgb_jpg

    def latest_rgb_pair(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """For inference: (external, wrist) RGB + camera-grab latency ms."""
        with self._lock:
            return self._latest.ext_rgb, self._latest.wrist_rgb, self._latest.t_grab_ms

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._pipe.stop()
        except Exception:
            pass
        if self._ext is not None:
            try:
                self._ext.release()
            except Exception:
                pass


def from_env() -> DashboardCamera:
    return DashboardCamera(
        wrist_serial=os.environ.get("FRANKA_BENCH_WRIST_SERIAL") or None,
        external_webcam_index=(
            int(os.environ["FRANKA_BENCH_EXT_INDEX"])
            if "FRANKA_BENCH_EXT_INDEX" in os.environ else None
        ),
        width=int(os.environ.get("FRANKA_BENCH_CAM_W", "256")),
        height=int(os.environ.get("FRANKA_BENCH_CAM_H", "256")),
        fps=int(os.environ.get("FRANKA_BENCH_CAM_FPS", "30")),
        external_rotation_deg=int(os.environ.get("FRANKA_BENCH_EXT_ROT_DEG", "0")),
        external_flip_h=os.environ.get("FRANKA_BENCH_EXT_FLIP_H", "0") not in ("0", "", "false", "False"),
        external_flip_v=os.environ.get("FRANKA_BENCH_EXT_FLIP_V", "0") not in ("0", "", "false", "False"),
        wrist_rotation_deg=int(os.environ.get("FRANKA_BENCH_WRIST_ROT_DEG", "0")),
        wrist_flip_h=os.environ.get("FRANKA_BENCH_WRIST_FLIP_H", "0") not in ("0", "", "false", "False"),
        wrist_flip_v=os.environ.get("FRANKA_BENCH_WRIST_FLIP_V", "0") not in ("0", "", "false", "False"),
    )
