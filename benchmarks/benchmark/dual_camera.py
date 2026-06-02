"""Two-camera capture: wrist RealSense + external USB webcam.

MolmoAct2-DROID expects (external_cam, wrist_cam). Your rig only has a wrist
D457, so this module also supports running with the wrist image duplicated as
external (DUPLICATE mode), at the cost of expected accuracy.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frames:
    external: np.ndarray   # (H, W, 3) uint8 RGB
    wrist:    np.ndarray   # (H, W, 3) uint8 RGB
    t_grab_ms: float


class _RealsenseWrist:
    def __init__(self, serial: Optional[str], width: int, height: int, fps: int):
        import pyrealsense2 as rs
        self._rs = rs
        self._pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self._pipe.start(cfg)
        for _ in range(5):
            self._pipe.wait_for_frames()

    def grab(self) -> np.ndarray:
        frames = self._pipe.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("RealSense: no color frame")
        return np.ascontiguousarray(np.asanyarray(color.get_data()))   # already RGB

    def close(self) -> None:
        try: self._pipe.stop()
        except Exception: pass


class _Webcam:
    """UVC / v4l2 webcam via OpenCV."""
    def __init__(self, index: int, width: int, height: int):
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(f"webcam {index} not open")
        for _ in range(5):
            self._cap.read()

    def grab(self) -> np.ndarray:
        ok, frame_bgr = self._cap.read()
        if not ok:
            raise RuntimeError("webcam: read failed")
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        try: self._cap.release()
        except Exception: pass


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


class DualCamera:
    """external + wrist capture with three configurable modes."""

    def __init__(
        self,
        wrist_serial: Optional[str] = None,
        external_webcam_index: Optional[int] = None,
        external_static_image: Optional[str] = None,
        width: int = 256,
        height: int = 256,
        fps: int = 30,
        external_rotation_deg: int = 0,
        external_flip_h: bool = False,
        external_flip_v: bool = False,
    ):
        if external_rotation_deg not in _VALID_ROT:
            raise ValueError(f"external_rotation_deg must be one of {_VALID_ROT}")
        self._ext_rot = external_rotation_deg
        self._ext_flip_h = bool(external_flip_h)
        self._ext_flip_v = bool(external_flip_v)
        self._wrist = _RealsenseWrist(wrist_serial, width, height, fps)
        self._external = None
        self._static_ext = None
        if external_webcam_index is not None:
            self._external = _Webcam(external_webcam_index, width, height)
        elif external_static_image is not None:
            from PIL import Image
            self._static_ext = np.asarray(
                Image.open(external_static_image).convert("RGB").resize((width, height))
            )
        # else: duplicate-wrist mode (degraded)

    def grab(self) -> Frames:
        t0 = time.perf_counter()
        wrist = self._wrist.grab()
        if self._external is not None:
            ext = self._external.grab()
        elif self._static_ext is not None:
            ext = self._static_ext
        else:
            ext = wrist.copy()         # DUPLICATE mode
        if self._ext_rot:
            ext = _rotate(ext, self._ext_rot)
        if self._ext_flip_h or self._ext_flip_v:
            ext = _apply_flips(ext, self._ext_flip_h, self._ext_flip_v)
        return Frames(external=ext, wrist=wrist, t_grab_ms=(time.perf_counter() - t0) * 1000.0)

    def close(self) -> None:
        self._wrist.close()
        if self._external is not None:
            self._external.close()


def from_env() -> DualCamera:
    """Build from FRANKA_BENCH_* env vars (set in the launcher script)."""
    return DualCamera(
        wrist_serial=os.environ.get("FRANKA_BENCH_WRIST_SERIAL") or None,
        external_webcam_index=(
            int(os.environ["FRANKA_BENCH_EXT_INDEX"])
            if "FRANKA_BENCH_EXT_INDEX" in os.environ else None
        ),
        external_static_image=os.environ.get("FRANKA_BENCH_EXT_STATIC") or None,
        width=int(os.environ.get("FRANKA_BENCH_CAM_W", "256")),
        height=int(os.environ.get("FRANKA_BENCH_CAM_H", "256")),
        external_rotation_deg=int(os.environ.get("FRANKA_BENCH_EXT_ROT_DEG", "0")),
        external_flip_h=os.environ.get("FRANKA_BENCH_EXT_FLIP_H", "0") not in ("0", "", "false", "False"),
        external_flip_v=os.environ.get("FRANKA_BENCH_EXT_FLIP_V", "0") not in ("0", "", "false", "False"),
    )
