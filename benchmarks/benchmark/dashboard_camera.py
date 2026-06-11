"""Dashboard-owned camera capture: RealSense color + up to two external webcams.

The benchmark's DualCamera is owned by the benchmark process. The dashboard
needs to keep running without a benchmark in flight, so it owns its own
RealSense pipeline.

A background thread continuously grabs frames and stashes JPEGs in memory
under a lock. MJPEG endpoints just yield the latest bytes; they never block
on hardware.

When two external webcams are configured the inference-side accessor returns
them concatenated horizontally (matching DROID training convention), while
the dashboard tiles render them as three separate streams.

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
    # JPEGs for the three MJPEG tiles.
    ext1_jpg: Optional[bytes] = None
    ext2_jpg: Optional[bytes] = None
    wrist_jpg: Optional[bytes] = None
    # Per-camera RGB ndarrays, kept for WebRTC + screenshot use.
    ext1_rgb: Optional[np.ndarray] = None
    ext2_rgb: Optional[np.ndarray] = None
    wrist_rgb: Optional[np.ndarray] = None
    # Inference-time external: ext1 or [ext1|ext2] horizontal concat.
    ext_for_infer: Optional[np.ndarray] = None
    t_grab_ms: float = 0.0
    timestamp: float = 0.0


def _encode_jpeg(rgb: np.ndarray, quality: int = 80) -> Optional[bytes]:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return jpg.tobytes() if ok else None


class DashboardCamera:
    """Owns RealSense (color) + up to two external webcams. Background-threaded."""

    def __init__(
        self,
        wrist_serial: Optional[str] = None,
        external_webcam_index: Optional[int] = None,
        external_webcam_index2: Optional[int] = None,
        # D45x has no 256x256 color mode; default to a supported tuple.
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        external_rotation_deg: int = 0,
        external_flip_h: bool = False,
        external_flip_v: bool = False,
        external2_rotation_deg: int = 0,
        external2_flip_h: bool = False,
        external2_flip_v: bool = False,
        wrist_rotation_deg: int = 0,
        wrist_flip_h: bool = False,
        wrist_flip_v: bool = False,
    ):
        for name, val in (("external_rotation_deg", external_rotation_deg),
                          ("external2_rotation_deg", external2_rotation_deg),
                          ("wrist_rotation_deg", wrist_rotation_deg)):
            if val not in _VALID_ROT:
                raise ValueError(f"{name} must be one of {_VALID_ROT}")
        self._ext_rot = int(external_rotation_deg)
        self._ext_flip_h = bool(external_flip_h)
        self._ext_flip_v = bool(external_flip_v)
        self._ext2_rot = int(external2_rotation_deg)
        self._ext2_flip_h = bool(external2_flip_h)
        self._ext2_flip_v = bool(external2_flip_v)
        self._wrist_rot = int(wrist_rotation_deg)
        self._wrist_flip_h = bool(wrist_flip_h)
        self._wrist_flip_v = bool(wrist_flip_v)

        # Remembered so restart_with_resolution() can rebuild the same setup.
        self._wrist_serial = wrist_serial
        self._ext_index = external_webcam_index
        self._ext_index2 = external_webcam_index2
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)

        self._lock = threading.Lock()
        self._latest = _Latest()
        self._restart_lock = threading.Lock()
        self._stop = threading.Event()
        self._open_pipelines()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ----- hardware open/close -----

    def _open_pipelines(self) -> None:
        import pyrealsense2 as rs
        self._rs = rs
        self._pipe = rs.pipeline()
        cfg = rs.config()
        if self._wrist_serial:
            cfg.enable_device(self._wrist_serial)
        cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)
        self._pipe.start(cfg)
        for _ in range(5):
            self._pipe.wait_for_frames()

        self._ext: Optional[cv2.VideoCapture] = None
        self._ext2: Optional[cv2.VideoCapture] = None
        if self._ext_index is not None:
            cap = cv2.VideoCapture(self._ext_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            if not cap.isOpened():
                raise RuntimeError(f"webcam {self._ext_index} not open")
            for _ in range(5):
                cap.read()
            self._ext = cap
        if self._ext_index2 is not None:
            cap2 = cv2.VideoCapture(self._ext_index2)
            cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            if not cap2.isOpened():
                raise RuntimeError(f"webcam {self._ext_index2} not open")
            for _ in range(5):
                cap2.read()
            self._ext2 = cap2

    def _close_pipelines(self) -> None:
        try:
            self._pipe.stop()
        except Exception:
            pass
        for cap in (self._ext, self._ext2):
            if cap is not None:
                try: cap.release()
                except Exception: pass
        self._ext = None
        self._ext2 = None

    def restart_with_resolution(self, width: int, height: int, fps: Optional[int] = None) -> None:
        """Tear down + rebuild the camera pipelines at a new resolution.

        Used by the dashboard's resolution dropdown. Holds an internal lock so
        the capture loop sees a consistent set of handles."""
        with self._restart_lock:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._close_pipelines()
            self._width = int(width)
            self._height = int(height)
            if fps is not None:
                self._fps = int(fps)
            self._stop.clear()
            self._open_pipelines()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    # ----- capture loop -----

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

            ext1_rgb: Optional[np.ndarray] = None
            ext2_rgb: Optional[np.ndarray] = None
            if self._ext is not None:
                try:
                    ok, bgr = self._ext.read()
                    if ok:
                        ext1_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    ext1_rgb = None
            if self._ext2 is not None:
                try:
                    ok, bgr = self._ext2.read()
                    if ok:
                        ext2_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    ext2_rgb = None

            if ext1_rgb is None:
                # duplicate mode (degraded but keeps the panel alive)
                ext1_rgb = wrist_rgb.copy()
            if self._ext_rot:
                ext1_rgb = _rotate(ext1_rgb, self._ext_rot)
            if self._ext_flip_h or self._ext_flip_v:
                ext1_rgb = _apply_flips(ext1_rgb, self._ext_flip_h, self._ext_flip_v)
            if ext2_rgb is not None:
                if self._ext2_rot:
                    ext2_rgb = _rotate(ext2_rgb, self._ext2_rot)
                if self._ext2_flip_h or self._ext2_flip_v:
                    ext2_rgb = _apply_flips(ext2_rgb, self._ext2_flip_h, self._ext2_flip_v)

            # NOTE: wrist rotation/flip intentionally disabled on the
            # dashboard for now (env vars FRANKA_BENCH_WRIST_ROT_DEG /
            # _FLIP_H / _FLIP_V are still read into self._wrist_rot /
            # _flip_*, but not applied here). Empirically the rotation
            # appears to make the under-gripper view harder for the
            # MolmoAct2-DROID policy to grasp from -- re-enable once
            # we know what orientation the policy actually wants.

            # Inference-side external: concat horizontally if both cams are
            # available, matching DROID training convention.
            if ext2_rgb is not None and ext1_rgb.shape == ext2_rgb.shape:
                ext_for_infer = np.concatenate([ext1_rgb, ext2_rgb], axis=1)
            else:
                ext_for_infer = ext1_rgb

            ext1_jpg = _encode_jpeg(ext1_rgb)
            ext2_jpg = _encode_jpeg(ext2_rgb) if ext2_rgb is not None else None
            wrist_jpg = _encode_jpeg(wrist_rgb)

            with self._lock:
                self._latest = _Latest(
                    ext1_jpg=ext1_jpg,
                    ext2_jpg=ext2_jpg,
                    wrist_jpg=wrist_jpg,
                    ext1_rgb=ext1_rgb,
                    ext2_rgb=ext2_rgb,
                    wrist_rgb=wrist_rgb,
                    ext_for_infer=ext_for_infer,
                    t_grab_ms=(time.perf_counter() - t0) * 1000.0,
                    timestamp=time.time(),
                )

    # ----- accessors used by the Flask app -----

    def latest_jpegs(self) -> tuple[Optional[bytes], Optional[bytes], Optional[bytes]]:
        """(ext1_jpg, ext2_jpg, wrist_jpg). ext2_jpg is None if no 2nd webcam."""
        with self._lock:
            return self._latest.ext1_jpg, self._latest.ext2_jpg, self._latest.wrist_jpg

    def latest_rgbs(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """(ext1_rgb, ext2_rgb, wrist_rgb) ndarrays for WebRTC framing."""
        with self._lock:
            return self._latest.ext1_rgb, self._latest.ext2_rgb, self._latest.wrist_rgb

    def latest_rgb_pair(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """For inference: (external, wrist) RGB + camera-grab latency ms.

        external is the horizontal concat of ext1+ext2 when both are
        configured, matching DROID's training convention."""
        with self._lock:
            return self._latest.ext_for_infer, self._latest.wrist_rgb, self._latest.t_grab_ms

    def resolution(self) -> tuple[int, int, int]:
        return self._width, self._height, self._fps

    def has_second_external(self) -> bool:
        return self._ext_index2 is not None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._close_pipelines()


def from_env() -> DashboardCamera:
    return DashboardCamera(
        wrist_serial=os.environ.get("FRANKA_BENCH_WRIST_SERIAL") or None,
        external_webcam_index=(
            int(os.environ["FRANKA_BENCH_EXT_INDEX"])
            if "FRANKA_BENCH_EXT_INDEX" in os.environ else None
        ),
        external_webcam_index2=(
            int(os.environ["FRANKA_BENCH_EXT_INDEX2"])
            if "FRANKA_BENCH_EXT_INDEX2" in os.environ else None
        ),
        # D457/D455 have no 256x256 mode — see dual_camera.from_env note.
        width=int(os.environ.get("FRANKA_BENCH_CAM_W", "640")),
        height=int(os.environ.get("FRANKA_BENCH_CAM_H", "480")),
        fps=int(os.environ.get("FRANKA_BENCH_CAM_FPS", "30")),
        external_rotation_deg=int(os.environ.get("FRANKA_BENCH_EXT_ROT_DEG", "0")),
        external_flip_h=os.environ.get("FRANKA_BENCH_EXT_FLIP_H", "0") not in ("0", "", "false", "False"),
        external_flip_v=os.environ.get("FRANKA_BENCH_EXT_FLIP_V", "0") not in ("0", "", "false", "False"),
        external2_rotation_deg=int(os.environ.get("FRANKA_BENCH_EXT2_ROT_DEG", "0")),
        external2_flip_h=os.environ.get("FRANKA_BENCH_EXT2_FLIP_H", "0") not in ("0", "", "false", "False"),
        external2_flip_v=os.environ.get("FRANKA_BENCH_EXT2_FLIP_V", "0") not in ("0", "", "false", "False"),
        wrist_rotation_deg=int(os.environ.get("FRANKA_BENCH_WRIST_ROT_DEG", "0")),
        wrist_flip_h=os.environ.get("FRANKA_BENCH_WRIST_FLIP_H", "0") not in ("0", "", "false", "False"),
        wrist_flip_v=os.environ.get("FRANKA_BENCH_WRIST_FLIP_V", "0") not in ("0", "", "false", "False"),
    )
