"""Interactive web dashboard for the MolmoAct2-DROID rig.

Shows three live camera streams (2 external + 1 wrist RealSense RGB), a
DROID task dropdown, runtime controls for camera resolution and wrist-cam
XYZ offsets, a Home button, a Run Benchmark button, and a motion_server
launcher (its initialize() runs goHome, so restarting it doubles as an arm
reset).

Run on the workstation that has the cameras + motion_server:
    pip install flask aiortc av     # aiortc/av only for WebRTC streaming
    export FRANKA_BENCH_EXT_INDEX=0
    export FRANKA_BENCH_EXT_INDEX2=1
    export FRANKA_REST_HOST=192.168.2.1
    python -m benchmarks.scripts.serve_dashboard --port 8080
Then open http://<workstation-ip>:8080/.

Video defaults to WebRTC (smoother / lower latency than MJPEG); falls back
to MJPEG when aiortc/av aren't installed, or pass --no-webrtc to force it.

FCI is no longer dashboard-supported; the CLI runner still has it. Only
the REST transport (motion_server) is available here.

Cannot run alongside `run_droid_benchmark.py` -- both want the cameras.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, Response, jsonify, request

from benchmarks.benchmark.dashboard_camera import DashboardCamera, from_env as camera_from_env
from benchmarks.benchmark.driver_errors import CollisionAborted
from benchmarks.benchmark import droid_tasks
from benchmarks.benchmark.droid_runner import (
    _enforce_hold_until_target,
    _enforce_hold_until_transported,
)
from benchmarks.benchmark.molmoact_droid_client import DroidClient
from benchmarks.benchmark.transport import make_driver

log = logging.getLogger("dashboard")


# Resolution presets exposed via the UI dropdown. All are color modes that
# D45x RealSense + typical USB webcams accept at 30 fps.
_RESOLUTIONS = [
    (320, 240, 30),
    (424, 240, 30),
    (640, 480, 30),
    (848, 480, 30),
    (1280, 720, 30),
]


def _build_webrtc_broadcaster():
    """Returns a WebRTCBroadcaster, or None if aiortc/av aren't installed.
    Dashboard then silently falls back to MJPEG."""
    try:
        from benchmarks.benchmark.webrtc_broadcaster import WebRTCBroadcaster
        return WebRTCBroadcaster()
    except Exception as e:
        log.warning("WebRTC disabled (falling back to MJPEG): %s", e)
        return None


class _ActionLock:
    """One-action-at-a-time guard with a small status surface for the UI."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False
        self._what: Optional[str] = None
        self._last: dict = {"ok": True, "info": "idle"}

    def acquire(self, what: str) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._what = what
            return True

    def release(self, result: dict) -> None:
        with self._lock:
            self._busy = False
            self._what = None
            self._last = result

    def snapshot(self) -> dict:
        with self._lock:
            return {"busy": self._busy, "what": self._what, "last": self._last}


class MotionServerLauncher:
    """Subprocess wrapper around franka/cpp/build/motion_server.

    Starting motion_server runs its initialize() which calls goHome(), so
    "restart motion_server" doubles as an arm-reset.
    """
    def __init__(self, bin_path: Optional[Path], log_path: Optional[Path] = None):
        self.bin_path = Path(bin_path) if bin_path else None
        self.log_path = log_path
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def configured(self) -> bool:
        return self.bin_path is not None and self.bin_path.exists()

    def status(self) -> dict:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            return {
                "configured": self.configured(),
                "bin_path": str(self.bin_path) if self.bin_path else None,
                "running": running,
                "pid": (self._proc.pid if running else None),
                "log_path": str(self.log_path) if self.log_path else None,
            }

    def start(self) -> dict:
        with self._lock:
            if not self.configured():
                return {"ok": False, "error": f"motion_server binary not found at {self.bin_path!s}"}
            if self._proc is not None and self._proc.poll() is None:
                return {"ok": False, "error": "motion_server already running",
                        "pid": self._proc.pid}
            log_f = None
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                log_f = open(self.log_path, "ab", buffering=0)
            try:
                self._proc = subprocess.Popen(
                    [str(self.bin_path)],
                    cwd=str(self.bin_path.parent),
                    stdout=log_f or subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            except Exception as e:
                return {"ok": False, "error": f"spawn failed: {e}"}
            return {"ok": True, "pid": self._proc.pid,
                    "log_path": str(self.log_path) if self.log_path else None}

    def stop(self, timeout_s: float = 5.0) -> dict:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = None
                return {"ok": True, "info": "not running"}
            try:
                self._proc.terminate()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
                except Exception:
                    pass
            self._proc = None
            return {"ok": True, "info": "stopped"}

    def restart(self) -> dict:
        stop_result = self.stop()
        if not stop_result.get("ok", False):
            return stop_result
        # motion_server itself takes a beat to release FCI before re-acquiring.
        time.sleep(0.5)
        return self.start()


def _subsample_rows(actions: np.ndarray, max_rows: int) -> np.ndarray:
    """Evenly thin ``actions`` to at most ``max_rows`` rows, always keeping the
    final waypoint. ``max_rows <= 0`` disables thinning (returns all rows)."""
    n = len(actions)
    if max_rows <= 0 or n <= max_rows:
        return actions
    idx = np.unique(np.linspace(0, n - 1, max_rows).round().astype(int))
    return actions[idx]


class DashboardState:
    def __init__(
        self,
        molmoact_url: str,
        rest_step_time_s: float,
        exec_rows: int,
        grasp_commit_grip_frac: float,
        fine_refinement_travel_rad: float,
        approach_max_rows: int = 4,
        max_chunks: int = 30,
        motion_server: Optional[MotionServerLauncher] = None,
    ):
        self.molmoact_url = molmoact_url
        self.client = DroidClient(molmoact_url)
        self.camera = camera_from_env()
        self.transport = "rest"
        self.rest_step_time_s = float(rest_step_time_s)
        self.exec_rows = int(exec_rows)
        self.approach_max_rows = int(approach_max_rows)
        self.grasp_commit_grip_frac = float(grasp_commit_grip_frac)
        self.fine_refinement_travel_rad = float(fine_refinement_travel_rad)
        self.max_chunks = int(max_chunks)
        # Mirrors the CLI runner's read of the same env var so the dashboard
        # behaves the same way for top-down grasp tasks.
        self.lock_gripper_down = (
            os.environ.get("FRANKA_BENCH_LOCK_GRIPPER_DOWN", "0")
            not in ("0", "", "false", "False")
        )
        self.driver = make_driver(self.transport, step_time_s=self.rest_step_time_s)
        self.motion_server = motion_server or MotionServerLauncher(None)
        self.action_lock = _ActionLock()
        self._driver_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._progress_lock = threading.Lock()
        self._progress = {"chunk": 0, "max": 0}

    def request_stop(self) -> None:
        self._stop_event.set()

    def _set_progress(self, chunk: int, max_chunks: int) -> None:
        with self._progress_lock:
            self._progress = {"chunk": chunk, "max": max_chunks}

    def progress(self) -> dict:
        with self._progress_lock:
            return dict(self._progress)

    def do_home(self) -> dict:
        with self._driver_lock:
            self.driver.home()
        return {"ok": True, "info": "homed via rest"}

    def do_task(self, instruction: str, max_chunks: Optional[int] = None,
                task_id: Optional[str] = None,
                hold_min_dist_m: float = 0.08) -> dict:
        """Run a full trial loop: capture → infer → exec, repeated up to
        max_chunks (or self.max_chunks if None) or until /api/stop is hit.

        When ``task_id`` resolves to a DroidTask that has ``target_zone_xy``
        defined, the policy's gripper-open commands are suppressed for any
        row whose commanded TCP XY lies outside that box while an object
        is detected in the jaws -- mirrors the CLI runner's
        --hold-until-target behavior so the arm can't drop the object
        mid-transport just because MolmoAct2 emitted a premature release."""
        self._stop_event.clear()
        cap = int(max_chunks) if max_chunks is not None else self.max_chunks
        target_zone = None
        if task_id:
            try:
                target_zone = droid_tasks.by_id(task_id).target_zone_xy
            except KeyError:
                target_zone = None
        t0 = time.perf_counter()
        chunks_done = 0
        last_pred_rows = 0
        last_exec_rows = 0
        total_server_ms = 0.0
        total_rtt_ms = 0.0
        total_suppressed = 0
        grasp_xy: Optional[tuple[float, float]] = None
        stopped_by = "max_chunks"
        for chunk_i in range(cap):
            if self._stop_event.is_set():
                stopped_by = "stop_requested"
                break
            self._set_progress(chunk_i + 1, cap)
            ext_rgb, wrist_rgb, _ = self.camera.latest_rgb_pair()
            if ext_rgb is None or wrist_rgb is None:
                self._set_progress(0, 0)
                return {"ok": False, "error": "cameras not ready yet",
                        "chunks_done": chunks_done}
            with self._driver_lock:
                state8 = self.driver.state_vec8()
            pred = self.client.act(
                external_cam=ext_rgb,
                wrist_cam=wrist_rgb,
                instruction=instruction,
                state=state8,
                num_steps=10,
            )
            rows = self._select_rows(pred.actions)
            # Copy once if either guard is active; both helpers mutate in place.
            if target_zone is not None or hold_min_dist_m > 0.0:
                rows = np.asarray(rows, dtype=np.float64).copy()
            if target_zone is not None:
                n_supp = _enforce_hold_until_target(
                    rows, float(state8[7]), target_zone,
                )
                if n_supp > 0:
                    total_suppressed += n_supp
                    log.warning(
                        "hold-until-target: kept gripper closed on %d/%d row(s) "
                        "(%s chunk %d) — policy tried to release outside target zone",
                        n_supp, len(rows), task_id, chunk_i + 1,
                    )
            elif hold_min_dist_m > 0.0:
                # No fixed zone — fall back to the "must transport ≥ min_dist"
                # heuristic. grasp_xy persists across chunks of this trial.
                n_supp, grasp_xy = _enforce_hold_until_transported(
                    rows, np.asarray(state8, dtype=np.float64),
                    grasp_xy, hold_min_dist_m,
                )
                if n_supp > 0:
                    total_suppressed += n_supp
                    log.warning(
                        "hold-until-transported: kept gripper closed on %d/%d "
                        "row(s) (%s chunk %d) — policy tried to release within "
                        "%.3f m of pickup",
                        n_supp, len(rows), task_id or "custom", chunk_i + 1,
                        hold_min_dist_m,
                    )
            try:
                with self._driver_lock:
                    self.driver.send_chunk(
                        rows,
                        step_dt_s=self.rest_step_time_s,
                        lock_gripper_down=self.lock_gripper_down,
                        stop_check=self._stop_event.is_set,
                    )
            except CollisionAborted as ca:
                # Libfranka's reflex aborted the move (most commonly: gripper
                # contacted the table). Stop the trial loop, send the arm
                # home, and surface a clean status so the user can re-setup.
                self._set_progress(0, 0)
                home_error: Optional[str] = None
                try:
                    with self._driver_lock:
                        self.driver.home()
                except Exception as he:
                    home_error = str(he)
                return {
                    "ok": False,
                    "error": "collision detected; trial aborted",
                    "stopped_by": "collision",
                    "collision_detail": str(ca)[:300],
                    "homed": home_error is None,
                    "home_error": home_error,
                    "chunks_done": chunks_done,
                }
            # If Stop was clicked during this chunk, send_chunk returned early.
            # Don't count it as a completed chunk and don't run any further
            # chunks -- fall through to the post-loop home() below.
            if self._stop_event.is_set():
                stopped_by = "stop_requested"
                break
            chunks_done += 1
            last_pred_rows = int(len(pred.actions))
            last_exec_rows = int(len(rows))
            total_server_ms += float(pred.server_dt_ms)
            total_rtt_ms += float(pred.rtt_ms)
        self._set_progress(0, 0)
        # Stop overrides the task: cancel and return the arm to home.
        homed_after_stop = None
        home_error: Optional[str] = None
        if stopped_by == "stop_requested":
            try:
                with self._driver_lock:
                    self.driver.home()
                homed_after_stop = True
            except Exception as he:
                homed_after_stop = False
                home_error = str(he)
        return {
            "ok": True,
            "info": f"trial done via rest ({stopped_by})",
            "stopped_by": stopped_by,
            "chunks_done": chunks_done,
            "chunks_budget": cap,
            "wallclock_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "infer_server_ms_total": round(total_server_ms, 1),
            "infer_rtt_ms_total": round(total_rtt_ms, 1),
            "last_rows_received": last_pred_rows,
            "last_rows_executed": last_exec_rows,
            "hold_until_target": target_zone is not None,
            "hold_min_dist_m": hold_min_dist_m if target_zone is None else None,
            "gripper_open_suppressed": total_suppressed,
            "homed_after_stop": homed_after_stop,
            "home_error": home_error,
        }

    def _select_rows(self, actions: np.ndarray) -> np.ndarray:
        # Adaptive row selection (cf. droid_runner.run_trial), plus an extra
        # approach-subsample step the CLI runner doesn't have.
        gripper = actions[:, 7] >= 0.5
        grasp_committed = gripper.sum() >= self.grasp_commit_grip_frac * len(gripper)
        if len(actions) > 1:
            deltas = np.diff(actions[:, :7], axis=0)
            total_travel = float(np.sum(np.linalg.norm(deltas, axis=1)))
        else:
            total_travel = 0.0
        is_fine = total_travel < self.fine_refinement_travel_rad
        if grasp_committed:
            # Never thin out the grasp chunk — every waypoint matters here.
            return actions
        if is_fine and self.exec_rows > 0:
            return actions[:max(1, self.exec_rows)]
        # Approach (large free-space travel): each row is a separate
        # accel-to-zero/decel-to-zero move, so executing all ~10 waypoints is
        # the dominant cost. Evenly subsample to at most approach_max_rows
        # (always keeping the terminal waypoint) to cut the stop-go count.
        # The driver's velocity cap + per-substep angular ceiling still bound
        # the larger inter-waypoint jumps. 0 disables (run every row).
        return _subsample_rows(actions, self.approach_max_rows)

    # ----- runtime knobs (offsets + resolution) -----

    def get_cam_offsets(self) -> tuple[float, float, float]:
        getter = getattr(self.driver, "get_cam_offsets", None)
        if getter is None:
            return 0.0, 0.0, 0.0
        return getter()

    def set_cam_offsets(self, dx: float, dy: float, dz: float) -> dict:
        setter = getattr(self.driver, "set_cam_offsets", None)
        if setter is None:
            return {"ok": False, "error": "driver does not support runtime cam offsets"}
        with self._driver_lock:
            setter(dx, dy, dz)
        return {"ok": True, "dx": dx, "dy": dy, "dz": dz}

    def gripper_status(self) -> dict:
        """Best-effort gripper read for the status panel. Skips the driver
        call when a higher-level action holds the lock so the status poll
        doesn't stall waiting for a trial to finish."""
        if self.action_lock.snapshot()["busy"]:
            return {"state": "busy", "width_m": None}
        try:
            with self._driver_lock:
                s = self.driver.get_state()
        except Exception as e:
            return {"state": "error", "width_m": None, "error": str(e)[:120]}
        w = float(s.gripper_width)
        # Thresholds match droid_runner._is_holding / _HOLDING_WIDTH_*.
        if w >= 0.075:
            label = "open"
        elif w > 0.005:
            label = "holding"
        else:
            label = "closed"
        return {"state": label, "width_m": round(w, 4)}

    def set_resolution(self, width: int, height: int, fps: int) -> dict:
        try:
            self.camera.restart_with_resolution(width, height, fps)
        except Exception as e:
            return {"ok": False, "error": f"restart failed: {e}"}
        w, h, f = self.camera.resolution()
        return {"ok": True, "width": w, "height": h, "fps": f}

    def close(self) -> None:
        try: self.camera.close()
        except Exception: pass
        try: self.driver.close()
        except Exception: pass


# ---------- HTML (inline; one page, vanilla JS) ----------

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>MolmoAct2-DROID dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: ui-sans-serif, system-ui, sans-serif; background:#111; color:#eee;
         margin:0; padding:1.5rem; }
  h1 { font-size:1.2rem; margin:0 0 1rem; }
  .cams { display:grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap:1rem; }
  .tile { background:#1c1c1c; border:1px solid #2a2a2a; border-radius:8px; padding:0.6rem; }
  .tile h2 { font-size:0.85rem; margin:0 0 0.4rem; color:#9ad; font-weight:600;
             display:flex; align-items:center; justify-content:space-between; }
  .tile img, .tile video { width:100%; height:auto; background:#000; border-radius:4px; display:block; }
  .panel { margin-top:1.2rem; background:#1c1c1c; border:1px solid #2a2a2a; border-radius:8px;
           padding:1rem; display:grid; gap:0.8rem; }
  .panel h3 { margin:0; font-size:0.85rem; color:#9ad; font-weight:600;
              text-transform:uppercase; letter-spacing:0.04em; }
  .row { display:flex; gap:0.6rem; align-items:center; flex-wrap:wrap; }
  label { font-size:0.85rem; color:#9ad; }
  select, input, button, textarea {
    background:#222; color:#eee; border:1px solid #333; border-radius:6px;
    padding:0.45rem 0.7rem; font-size:0.9rem;
  }
  input[type=number] { width:7em; }
  button { cursor:pointer; }
  button:hover:enabled { background:#2c3e50; }
  button:disabled { opacity:0.55; cursor:default; }
  button.primary { background:#2c5b3e; border-color:#3d7b54; }
  button.primary:hover:enabled { background:#377050; }
  button.danger { background:#5b2c2c; border-color:#7b3d3d; }
  button.danger:hover:enabled { background:#703737; }
  #task-instruction { color:#bbb; font-size:0.85rem; min-height:1.2em; font-style:italic; }
  #status { font-size:0.85rem; color:#bbb; white-space:pre-wrap; min-height:1.2em; }
  #ms-status { font-size:0.85rem; color:#bbb; }
  .grip-pill { display:inline-flex; align-items:center; gap:0.4rem;
               background:#222; border:1px solid #333; border-radius:999px;
               padding:0.25rem 0.7rem; font-size:0.85rem; font-weight:600;
               letter-spacing:0.04em; text-transform:uppercase; }
  .grip-dot { width:0.7rem; height:0.7rem; border-radius:50%;
              background:#666; box-shadow:0 0 0.4rem currentColor; }
  .grip-open    .grip-dot { background:#9ad; color:#9ad; }
  .grip-holding .grip-dot { background:#7ad07a; color:#7ad07a; }
  .grip-closed  .grip-dot { background:#e07a7a; color:#e07a7a; }
  .grip-busy    .grip-dot { background:#e0c47a; color:#e0c47a; }
  .grip-error   .grip-dot { background:#888; color:#888; }
  .ok { color:#7ad07a; }
  .err { color:#e07a7a; }
  .busy { color:#e0c47a; }
  .sep { width:1px; height:24px; background:#333; margin:0 0.3rem; }
</style>
</head>
<body>
<h1>
  MolmoAct2-DROID dashboard
  <span id="gripper-pill" class="grip-pill grip-error" style="margin-left:1rem;">
    <span class="grip-dot"></span>
    <span id="gripper-label">gripper —</span>
  </span>
</h1>

<div class="cams">
  <div class="tile">
    <h2><span>external 1</span><span id="tag-ext" style="color:#777">[?]</span></h2>
    <div id="cam-ext"></div>
  </div>
  <div class="tile">
    <h2><span>external 2</span><span id="tag-ext2" style="color:#777">[?]</span></h2>
    <div id="cam-ext2"></div>
  </div>
  <div class="tile">
    <h2><span>wrist RGB (D45x)</span><span id="tag-wrist" style="color:#777">[?]</span></h2>
    <div id="cam-wrist"></div>
  </div>
</div>

<div class="panel">
  <h3>task</h3>
  <div class="row">
    <label for="task-select">task:</label>
    <select id="task-select"></select>
    <button id="btn-home">home</button>
    <button id="btn-run" class="primary">run benchmark</button>
    <button id="btn-stop" class="danger" disabled>stop</button>
  </div>
  <div id="task-instruction">(select a task)</div>
</div>

<div class="panel">
  <h3>camera resolution</h3>
  <div class="row">
    <label for="res-select">preset:</label>
    <select id="res-select"></select>
    <button id="btn-apply-res">apply</button>
    <span id="res-current" style="color:#bbb; font-size:0.85rem;"></span>
  </div>
</div>

<div class="panel">
  <h3>hold-until-transported guard</h3>
  <div class="row">
    <label for="hold-min-dist">min transport (m):</label>
    <input id="hold-min-dist" type="number" step="0.01" min="0" value="0.08">
    <span style="color:#bbb; font-size:0.8rem;">
      0 = disable. Suppresses gripper-open commands within this radius of
      the pickup point. Ignored when the task has a fixed target_zone_xy.
    </span>
  </div>
</div>

<div class="panel">
  <h3>wrist-cam XYZ offset (metres, base frame)</h3>
  <div class="row">
    <label>DX</label><input id="off-dx" type="number" step="0.005" value="0">
    <label>DY</label><input id="off-dy" type="number" step="0.005" value="0">
    <label>DZ</label><input id="off-dz" type="number" step="0.005" value="0">
    <button id="btn-apply-off">apply</button>
    <span id="off-current" style="color:#bbb; font-size:0.85rem;"></span>
  </div>
</div>

<div class="panel">
  <h3>motion_server (reset-home via restart)</h3>
  <div class="row">
    <button id="btn-ms-start">start</button>
    <button id="btn-ms-restart">restart (re-homes)</button>
    <button id="btn-ms-stop" class="danger">stop</button>
    <span id="ms-status">…</span>
  </div>
</div>

<div class="panel">
  <h3>status</h3>
  <div id="status">loading...</div>
</div>

<script>
const $ = s => document.querySelector(s);
const status = $("#status");
const WEBRTC_ENABLED = __WEBRTC_ENABLED__;
const CAMS = ["ext", "ext2", "wrist"];
const STREAM_PATH = { ext: "/stream/ext", ext2: "/stream/ext2", wrist: "/stream/wrist_rgb" };
const SLOT = { ext: ["cam-ext", "tag-ext"], ext2: ["cam-ext2", "tag-ext2"], wrist: ["cam-wrist", "tag-wrist"] };

function mountMjpeg(slotId, tagId, src) {
  const el = document.getElementById(slotId);
  el.innerHTML = `<img src="${src}">`;
  const tag = document.getElementById(tagId);
  if (tag) { tag.textContent = "[mjpeg]"; tag.style.color = "#777"; }
}

// aiortc is non-trickle: it ignores ICE candidates sent after the offer, so
// we must let the browser finish gathering host candidates and POST the
// completed SDP. Skipping this leaves the PC unable to connect — the <video>
// mounts but never receives media (frozen/blank panel).
function waitForIceGathering(pc) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const check = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", check);
    // Safety net: some browsers never fire 'complete' for host-only candidates.
    setTimeout(resolve, 2000);
  });
}

async function mountWebRTC(slotId, tagId, cam) {
  const el = document.getElementById(slotId);
  el.innerHTML = '<video autoplay playsinline muted></video>';
  const video = el.querySelector("video");
  const pc = new RTCPeerConnection();
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.ontrack = (ev) => { if (ev.track.kind === "video") video.srcObject = ev.streams[0]; };
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await waitForIceGathering(pc);
  const r = await fetch(`/offer/${cam}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({sdp: pc.localDescription.sdp, type: pc.localDescription.type}),
  });
  if (!r.ok) { pc.close(); throw new Error(`offer ${r.status}`); }
  const ans = await r.json();
  await pc.setRemoteDescription(ans);

  // Media watchdog. Signaling (SDP exchange) can succeed while ICE/media
  // never actually connects — common when the browser is on a different
  // host than the server and WebRTC's UDP path can't traverse. Without this
  // the <video> mounts empty and never falls back, leaving a blank panel.
  // If no frame arrives shortly, tear down and throw so the caller falls
  // back to MJPEG, which rides the same HTTP path the page already loaded on.
  await new Promise((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener("loadeddata", onData);
      pc.removeEventListener("connectionstatechange", onState);
    };
    const onData = () => { if (!settled) { settled = true; cleanup(); resolve(); } };
    const onState = () => {
      if (!settled && ["failed", "disconnected", "closed"].includes(pc.connectionState)) {
        settled = true; cleanup(); pc.close(); reject(new Error(`pc ${pc.connectionState}`));
      }
    };
    const timer = setTimeout(() => {
      if (!settled) { settled = true; cleanup(); pc.close(); reject(new Error("no media within 5s")); }
    }, 5000);
    video.addEventListener("loadeddata", onData);
    pc.addEventListener("connectionstatechange", onState);
    if (video.readyState >= 2) onData();  // already has data
  });

  const tag = document.getElementById(tagId);
  if (tag) { tag.textContent = "[webrtc]"; tag.style.color = "#7ad07a"; }
}

async function mountOne(cam) {
  const [slotId, tagId] = SLOT[cam];
  if (WEBRTC_ENABLED) {
    try { await mountWebRTC(slotId, tagId, cam); return; }
    catch (e) { console.warn(`webrtc ${cam} failed, falling back to mjpeg:`, e); }
  }
  mountMjpeg(slotId, tagId, STREAM_PATH[cam] + `?t=${Date.now()}`);
}

async function mountCameras() {
  // Mount in parallel so one camera's WebRTC watchdog timeout doesn't delay
  // the others' fallback.
  await Promise.all(CAMS.map(mountOne));
}

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const j = await r.json();
    const cls = j.busy ? "busy" : (j.last && j.last.ok === false ? "err" : "ok");
    const prog = j.progress || {chunk: 0, max: 0};
    const progStr = prog.max > 0 ? ` (chunk ${prog.chunk}/${prog.max})` : "";
    const what = j.busy ? `[busy: ${j.what}${progStr}]` : "idle";
    const last = j.last ? JSON.stringify(j.last) : "";
    status.className = cls;
    status.textContent = `${what}\\nlast: ${last}`;
    $("#btn-home").disabled = j.busy;
    $("#btn-run").disabled = j.busy;
    $("#btn-stop").disabled = !j.busy;
    if (j.resolution) {
      $("#res-current").textContent = `current: ${j.resolution.width}x${j.resolution.height}@${j.resolution.fps}`;
    }
    if (j.cam_offsets) {
      const o = j.cam_offsets;
      $("#off-current").textContent = `current: DX=${o.dx.toFixed(3)} DY=${o.dy.toFixed(3)} DZ=${o.dz.toFixed(3)}`;
    }
    if (j.gripper) {
      const g = j.gripper;
      const pill = $("#gripper-pill");
      pill.classList.remove("grip-open", "grip-holding", "grip-closed", "grip-busy", "grip-error");
      pill.classList.add(`grip-${g.state}`);
      const w = (g.width_m === null || g.width_m === undefined) ? "—" : `${(g.width_m*1000).toFixed(1)} mm`;
      $("#gripper-label").textContent = `gripper ${g.state} (${w})`;
    }
    if (j.motion_server) {
      const ms = j.motion_server;
      const lbl = ms.configured
        ? (ms.running ? `running (pid ${ms.pid})` : "stopped")
        : "not configured";
      $("#ms-status").textContent = `motion_server: ${lbl}`;
      $("#ms-status").className = ms.running ? "ok" : (ms.configured ? "" : "err");
      $("#btn-ms-start").disabled = !ms.configured || ms.running;
      $("#btn-ms-restart").disabled = !ms.configured;
      $("#btn-ms-stop").disabled = !ms.running;
    }
  } catch (e) {
    status.className = "err";
    status.textContent = "status error: " + e;
  }
}

async function loadTasks() {
  const r = await fetch("/api/tasks");
  const j = await r.json();
  const sel = $("#task-select");
  sel.innerHTML = "";
  for (const t of j.tasks) {
    const opt = document.createElement("option");
    opt.value = t.task_id;
    opt.textContent = t.task_id;
    opt.dataset.instruction = t.instruction;
    opt.dataset.maxChunks = t.max_chunks;
    sel.appendChild(opt);
  }
  if (j.tasks.length) updateInstruction();
  sel.addEventListener("change", updateInstruction);
}

function updateInstruction() {
  const sel = $("#task-select");
  const opt = sel.options[sel.selectedIndex];
  $("#task-instruction").textContent = opt ? `instruction: "${opt.dataset.instruction}"` : "";
}

async function loadResolutions() {
  const r = await fetch("/api/resolutions");
  const j = await r.json();
  const sel = $("#res-select");
  sel.innerHTML = "";
  for (const p of j.presets) {
    const opt = document.createElement("option");
    opt.value = `${p.width}x${p.height}x${p.fps}`;
    opt.textContent = `${p.width}x${p.height}@${p.fps}`;
    sel.appendChild(opt);
  }
  const cur = `${j.current.width}x${j.current.height}x${j.current.fps}`;
  for (const o of sel.options) if (o.value === cur) o.selected = true;
}

async function loadOffsets() {
  const r = await fetch("/api/offsets");
  const j = await r.json();
  $("#off-dx").value = j.dx;
  $("#off-dy").value = j.dy;
  $("#off-dz").value = j.dz;
}

$("#btn-home").addEventListener("click", async () => {
  $("#btn-home").disabled = true;
  const r = await fetch("/api/home", { method: "POST" });
  status.textContent = JSON.stringify(await r.json());
  await refreshStatus();
});

$("#btn-run").addEventListener("click", async () => {
  const sel = $("#task-select");
  const opt = sel.options[sel.selectedIndex];
  if (!opt) { status.textContent = "no task selected"; return; }
  $("#btn-run").disabled = true;
  const r = await fetch("/api/task", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      task_id: opt.value,
      instruction: opt.dataset.instruction,
      max_chunks: Number(opt.dataset.maxChunks),
      hold_min_dist_m: Number($("#hold-min-dist").value),
    }),
  });
  status.textContent = JSON.stringify(await r.json());
  await refreshStatus();
});

$("#btn-stop").addEventListener("click", async () => {
  $("#btn-stop").disabled = true;
  await fetch("/api/stop", { method: "POST" });
  await refreshStatus();
});

$("#btn-apply-res").addEventListener("click", async () => {
  const [w, h, f] = $("#res-select").value.split("x").map(Number);
  $("#btn-apply-res").disabled = true;
  const r = await fetch("/api/resolution", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ width: w, height: h, fps: f }),
  });
  status.textContent = JSON.stringify(await r.json());
  $("#btn-apply-res").disabled = false;
  await mountCameras();
  await refreshStatus();
});

$("#btn-apply-off").addEventListener("click", async () => {
  const dx = Number($("#off-dx").value);
  const dy = Number($("#off-dy").value);
  const dz = Number($("#off-dz").value);
  const r = await fetch("/api/offsets", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ dx, dy, dz }),
  });
  status.textContent = JSON.stringify(await r.json());
  await refreshStatus();
});

async function msPost(action) {
  const r = await fetch(`/api/motion_server/${action}`, { method: "POST" });
  status.textContent = JSON.stringify(await r.json());
  await refreshStatus();
}
$("#btn-ms-start").addEventListener("click", () => msPost("start"));
$("#btn-ms-restart").addEventListener("click", () => msPost("restart"));
$("#btn-ms-stop").addEventListener("click", () => msPost("stop"));

(async () => {
  // Bring the controls up FIRST and never block them on camera negotiation:
  // WebRTC setup + the MJPEG-fallback watchdog can take several seconds, and
  // the task dropdown / Run button / status poller must be usable immediately.
  await loadTasks();
  await loadResolutions();
  await loadOffsets();
  setInterval(refreshStatus, 1500);
  refreshStatus();
  mountCameras();  // fire-and-forget; streams appear when ready
})();
</script>
</body>
</html>
"""


# ---------- Flask app ----------

def make_app(state: DashboardState, fps: float = 30.0, webrtc=None) -> Flask:
    app = Flask(__name__)
    period = 1.0 / max(fps, 1.0)
    app.config["WEBRTC_ENABLED"] = webrtc is not None

    def _stream(getter):
        def gen():
            while True:
                b = getter()
                if b:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b + b"\r\n"
                time.sleep(period)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def index():
        flag = "true" if app.config["WEBRTC_ENABLED"] else "false"
        return _INDEX_HTML.replace("__WEBRTC_ENABLED__", flag)

    @app.route("/stream/ext")
    def stream_ext():
        return _stream(lambda: state.camera.latest_jpegs()[0])

    @app.route("/stream/ext2")
    def stream_ext2():
        return _stream(lambda: state.camera.latest_jpegs()[1])

    @app.route("/stream/wrist_rgb")
    def stream_wrist_rgb():
        return _stream(lambda: state.camera.latest_jpegs()[2])

    @app.route("/offer/<cam>", methods=["POST"])
    def offer(cam: str):
        if webrtc is None:
            return jsonify({"ok": False, "error": "webrtc disabled on server"}), 503
        if cam not in ("ext", "ext2", "wrist"):
            return jsonify({"ok": False, "error": f"unknown cam {cam!r}"}), 400
        body = request.get_json(silent=True) or {}
        sdp = body.get("sdp"); type_ = body.get("type")
        if not sdp or not type_:
            return jsonify({"ok": False, "error": "sdp and type required"}), 400
        if cam == "ext2" and not state.camera.has_second_external():
            return jsonify({"ok": False, "error": "no 2nd external configured"}), 400
        idx = {"ext": 0, "ext2": 1, "wrist": 2}[cam]
        getter = lambda: state.camera.latest_rgbs()[idx]
        try:
            ans = webrtc.handle_offer(sdp, type_, getter)
        except Exception as e:
            log.exception("WebRTC offer failed")
            return jsonify({"ok": False, "error": f"offer failed: {e}"}), 500
        return jsonify(ans)

    @app.route("/api/status")
    def api_status():
        snap = state.action_lock.snapshot()
        snap["transport"] = "rest"
        snap["molmoact_url"] = state.molmoact_url
        snap["progress"] = state.progress()
        w, h, f = state.camera.resolution()
        snap["resolution"] = {"width": w, "height": h, "fps": f}
        dx, dy, dz = state.get_cam_offsets()
        snap["cam_offsets"] = {"dx": dx, "dy": dy, "dz": dz}
        snap["motion_server"] = state.motion_server.status()
        snap["gripper"] = state.gripper_status()
        return jsonify(snap)

    @app.route("/api/tasks")
    def api_tasks():
        out = []
        for t in droid_tasks.all_tasks():
            out.append({
                "task_id": t.task_id,
                "instruction": t.instruction,
                "paper_success_rate": t.paper_success_rate,
                "max_chunks": t.max_chunks,
                "trials": t.trials,
                "target_zone_xy": t.target_zone_xy,
                "hold_until_target": t.target_zone_xy is not None,
            })
        return jsonify({"tasks": out})

    @app.route("/api/resolutions")
    def api_resolutions():
        w, h, f = state.camera.resolution()
        return jsonify({
            "presets": [{"width": p[0], "height": p[1], "fps": p[2]} for p in _RESOLUTIONS],
            "current": {"width": w, "height": h, "fps": f},
        })

    @app.route("/api/resolution", methods=["POST"])
    def api_resolution():
        body = request.get_json(silent=True) or {}
        try:
            width = int(body["width"]); height = int(body["height"]); fps = int(body["fps"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"ok": False, "error": "width, height, fps required (ints)"}), 400
        if not state.action_lock.acquire("resolution"):
            return jsonify({"ok": False, "error": "another action is in progress"}), 409
        try:
            result = state.set_resolution(width, height, fps)
        finally:
            state.action_lock.release(result)
        return jsonify(result)

    @app.route("/api/offsets", methods=["GET", "POST"])
    def api_offsets():
        if request.method == "GET":
            dx, dy, dz = state.get_cam_offsets()
            return jsonify({"dx": dx, "dy": dy, "dz": dz})
        body = request.get_json(silent=True) or {}
        try:
            dx = float(body["dx"]); dy = float(body["dy"]); dz = float(body["dz"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"ok": False, "error": "dx, dy, dz required (floats)"}), 400
        return jsonify(state.set_cam_offsets(dx, dy, dz))

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        # No action_lock: the running trial is what holds the lock; this
        # endpoint must remain callable while the loop is mid-flight so it
        # can signal an early exit on the next iteration boundary.
        state.request_stop()
        return jsonify({"ok": True, "info": "stop requested"})

    @app.route("/api/home", methods=["POST"])
    def api_home():
        if not state.action_lock.acquire("home"):
            return jsonify({"ok": False, "error": "another action is in progress"}), 409
        try:
            try:
                result = state.do_home()
            except Exception as e:
                result = {"ok": False, "error": f"home failed: {e}"}
        finally:
            state.action_lock.release(result)
        return jsonify(result)

    @app.route("/api/task", methods=["POST"])
    def api_task():
        body = request.get_json(silent=True) or {}
        instruction = (body.get("instruction") or "").strip()
        task_id = body.get("task_id")
        max_chunks = body.get("max_chunks")
        try:
            hold_min_dist_m = float(body.get("hold_min_dist_m", 0.08))
        except (TypeError, ValueError):
            hold_min_dist_m = 0.08
        if not instruction:
            return jsonify({"ok": False, "error": "instruction is required"}), 400
        if not state.action_lock.acquire(f"task:{task_id or 'custom'}"):
            return jsonify({"ok": False, "error": "another action is in progress"}), 409
        try:
            try:
                result = state.do_task(instruction, max_chunks=max_chunks,
                                       task_id=task_id,
                                       hold_min_dist_m=hold_min_dist_m)
                result["instruction"] = instruction
                if task_id:
                    result["task_id"] = task_id
            except Exception as e:
                result = {"ok": False, "error": f"task failed: {e}", "instruction": instruction}
        finally:
            state.action_lock.release(result)
        return jsonify(result)

    @app.route("/api/motion_server/<action>", methods=["POST"])
    def api_motion_server(action: str):
        if action == "start":
            return jsonify(state.motion_server.start())
        if action == "stop":
            return jsonify(state.motion_server.stop())
        if action == "restart":
            return jsonify(state.motion_server.restart())
        if action == "status":
            return jsonify(state.motion_server.status())
        return jsonify({"ok": False, "error": f"unknown action {action!r}"}), 400

    return app


def _resolve_motion_server_bin(arg: Optional[str]) -> Optional[Path]:
    if arg:
        return Path(arg).expanduser().resolve()
    # Best-effort default: <repo-root>/franka/cpp/build/motion_server
    here = Path(__file__).resolve()
    for p in here.parents:
        candidate = p / "franka" / "cpp" / "build" / "motion_server"
        if candidate.exists():
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--molmoact-url", default="http://localhost:8000")
    ap.add_argument("--rest-step-time-s", type=float, default=2.5)
    ap.add_argument("--exec-rows", type=int, default=3,
                    help="rows to run for a FINE-refinement chunk (small travel).")
    ap.add_argument("--approach-exec-rows", type=int, default=4,
                    help="max rows to run for an APPROACH chunk (large free-space "
                         "travel); evenly subsampled, terminal waypoint kept. "
                         "Lower = fewer stop-go moves = faster approach. 0 = run all rows.")
    ap.add_argument("--grasp-commit-grip-frac", type=float, default=0.5)
    ap.add_argument("--fine-refinement-travel-rad", type=float, default=0.2)
    ap.add_argument("--max-chunks", type=int, default=30,
                    help="safety cap on action chunks per dashboard trial "
                         "(used when the selected task doesn't carry its own).")
    ap.add_argument("--mjpeg-fps", type=float, default=30.0)
    ap.add_argument("--no-webrtc", action="store_true",
                    help="disable WebRTC streaming and force MJPEG.")
    ap.add_argument("--motion-server-bin", default=None,
                    help="path to the motion_server binary. Defaults to "
                         "<repo>/franka/cpp/build/motion_server if found.")
    ap.add_argument("--motion-server-log", default=None,
                    help="path to write motion_server stdout/stderr.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # FCI used to be valid here; the dashboard no longer supports it. The CLI
    # runner still does. Reject early so a stale FRANKA_HOST in the env can't
    # silently flip us off the supported REST path.
    if not os.environ.get("FRANKA_REST_HOST"):
        log.error(
            "FRANKA_REST_HOST is not set. The dashboard only supports the REST "
            "transport (motion_server). Export FRANKA_REST_HOST=<motion_server IP> "
            "and re-run. (FCI/MCP are no longer dashboard-supported; use the CLI runner.)"
        )
        return 2
    log.info("transport: rest")

    bin_path = _resolve_motion_server_bin(args.motion_server_bin)
    log_path = Path(args.motion_server_log).expanduser().resolve() if args.motion_server_log else None
    motion_server = MotionServerLauncher(bin_path, log_path=log_path)
    if motion_server.configured():
        log.info("motion_server binary: %s", bin_path)
    else:
        log.warning("motion_server binary not found; the UI controls will be disabled.")

    state = DashboardState(
        molmoact_url=args.molmoact_url,
        rest_step_time_s=args.rest_step_time_s,
        exec_rows=args.exec_rows,
        grasp_commit_grip_frac=args.grasp_commit_grip_frac,
        fine_refinement_travel_rad=args.fine_refinement_travel_rad,
        approach_max_rows=args.approach_exec_rows,
        max_chunks=args.max_chunks,
        motion_server=motion_server,
    )

    webrtc = None if args.no_webrtc else _build_webrtc_broadcaster()
    if webrtc is not None:
        log.info("WebRTC enabled")
    else:
        log.info("WebRTC disabled; using MJPEG")

    app = make_app(state, fps=args.mjpeg_fps, webrtc=webrtc)
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        if webrtc is not None:
            try: webrtc.close()
            except Exception: pass
        try: motion_server.stop()
        except Exception: pass
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
