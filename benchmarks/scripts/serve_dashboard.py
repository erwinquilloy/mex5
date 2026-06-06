"""Interactive web dashboard for the MolmoAct2-DROID rig.

Shows live external webcam + wrist RealSense RGB streams, a "Home" button,
and a task-instruction input that runs one inference->exec cycle per click.
Auto-detects the robot transport (fci / rest) from env vars; a UI dropdown
lets you switch without restarting. MCP is no longer supported on the
dashboard -- use the CLI runner if you need MCP.

Run on the workstation that has the cameras:
    pip install flask aiortc av    # aiortc/av only needed for WebRTC streaming
    export FRANKA_BENCH_EXT_INDEX=0
    export FRANKA_MCP_URL=...   # or FRANKA_REST_HOST / FRANKA_HOST
    python -m benchmarks.scripts.serve_dashboard --port 8080
Then open http://<workstation-ip>:8080/.

Video streaming defaults to WebRTC (smoother / lower latency than MJPEG).
Falls back to MJPEG automatically when aiortc/av aren't installed, or pass
--no-webrtc to force MJPEG.

Cannot run alongside `run_droid_benchmark.py` -- both want the cameras.
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from typing import Optional

import numpy as np
from flask import Flask, Response, jsonify, request

from benchmarks.benchmark.dashboard_camera import DashboardCamera, from_env as camera_from_env
from benchmarks.benchmark.driver_errors import CollisionAborted
from benchmarks.benchmark.molmoact_droid_client import DroidClient
from benchmarks.benchmark.transport import autodetect_transport, make_driver

log = logging.getLogger("dashboard")


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


class DashboardState:
    def __init__(
        self,
        molmoact_url: str,
        transport: str,
        rest_step_time_s: float,
        exec_rows: int,
        grasp_commit_grip_frac: float,
        fine_refinement_travel_rad: float,
        max_chunks: int = 30,
    ):
        self.molmoact_url = molmoact_url
        self.client = DroidClient(molmoact_url)
        self.camera = camera_from_env()
        self.transport = transport
        self.rest_step_time_s = float(rest_step_time_s)
        self.exec_rows = int(exec_rows)
        self.grasp_commit_grip_frac = float(grasp_commit_grip_frac)
        self.fine_refinement_travel_rad = float(fine_refinement_travel_rad)
        self.max_chunks = int(max_chunks)
        # Mirrors the CLI runner's read of the same env var so the dashboard
        # behaves the same way for top-down grasp tasks.
        self.lock_gripper_down = (
            os.environ.get("FRANKA_BENCH_LOCK_GRIPPER_DOWN", "0")
            not in ("0", "", "false", "False")
        )
        self.driver = make_driver(transport, step_time_s=self.rest_step_time_s)
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

    def switch_transport(self, new_transport: str) -> dict:
        with self._driver_lock:
            try:
                new_driver = make_driver(new_transport, step_time_s=self.rest_step_time_s)
            except Exception as e:
                return {"ok": False, "error": f"could not build {new_transport!r}: {e}"}
            try:
                self.driver.close()
            except Exception:
                pass
            self.driver = new_driver
            self.transport = new_transport
            return {"ok": True, "transport": new_transport}

    def step_dt_for_send_chunk(self) -> float:
        # FCI: 0.1 s substep ramp. REST/MCP: per-row REST move duration.
        return 0.1 if self.transport == "fci" else self.rest_step_time_s

    def do_home(self) -> dict:
        with self._driver_lock:
            self.driver.home()
        return {"ok": True, "info": f"homed via {self.transport}"}

    def do_task(self, instruction: str) -> dict:
        """Run a full trial loop: capture → infer → exec, repeated up to
        self.max_chunks times or until a /api/stop is received. Mirrors
        droid_runner.run_trial's per-chunk loop, minus the success prompt."""
        self._stop_event.clear()
        t0 = time.perf_counter()
        chunks_done = 0
        last_pred_rows = 0
        last_exec_rows = 0
        total_server_ms = 0.0
        total_rtt_ms = 0.0
        stopped_by = "max_chunks"
        for chunk_i in range(self.max_chunks):
            if self._stop_event.is_set():
                stopped_by = "stop_requested"
                break
            self._set_progress(chunk_i + 1, self.max_chunks)
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
            try:
                with self._driver_lock:
                    self.driver.send_chunk(
                        rows,
                        step_dt_s=self.step_dt_for_send_chunk(),
                        lock_gripper_down=self.lock_gripper_down,
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
            chunks_done += 1
            last_pred_rows = int(len(pred.actions))
            last_exec_rows = int(len(rows))
            total_server_ms += float(pred.server_dt_ms)
            total_rtt_ms += float(pred.rtt_ms)
        self._set_progress(0, 0)
        return {
            "ok": True,
            "info": f"trial done via {self.transport} ({stopped_by})",
            "stopped_by": stopped_by,
            "chunks_done": chunks_done,
            "chunks_budget": self.max_chunks,
            "wallclock_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "infer_server_ms_total": round(total_server_ms, 1),
            "infer_rtt_ms_total": round(total_rtt_ms, 1),
            "last_rows_received": last_pred_rows,
            "last_rows_executed": last_exec_rows,
        }

    def _select_rows(self, actions: np.ndarray) -> np.ndarray:
        # Same adaptive logic as droid_runner.run_trial.
        gripper = actions[:, 7] >= 0.5
        grasp_committed = gripper.sum() >= self.grasp_commit_grip_frac * len(gripper)
        if len(actions) > 1:
            deltas = np.diff(actions[:, :7], axis=0)
            total_travel = float(np.sum(np.linalg.norm(deltas, axis=1)))
        else:
            total_travel = 0.0
        is_fine = total_travel < self.fine_refinement_travel_rad
        if grasp_committed:
            return actions
        if is_fine and self.exec_rows > 0:
            return actions[:max(1, self.exec_rows)]
        return actions

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
  .grid { display:grid; grid-template-columns: repeat(2, minmax(300px, 1fr)); gap:1rem; }
  .tile { background:#1c1c1c; border:1px solid #2a2a2a; border-radius:8px; padding:0.6rem; }
  .tile h2 { font-size:0.85rem; margin:0 0 0.4rem; color:#9ad; font-weight:600; }
  .tile img, .tile video { width:100%; height:auto; background:#000; border-radius:4px; display:block; }
  .controls { margin-top:1.2rem; background:#1c1c1c; border:1px solid #2a2a2a; border-radius:8px;
              padding:1rem; display:grid; gap:0.8rem; }
  .row { display:flex; gap:0.6rem; align-items:center; flex-wrap:wrap; }
  label { font-size:0.85rem; color:#9ad; }
  select, input[type=text], button {
    background:#222; color:#eee; border:1px solid #333; border-radius:6px;
    padding:0.45rem 0.7rem; font-size:0.9rem;
  }
  input[type=text] { flex:1 1 320px; min-width:260px; }
  button { cursor:pointer; }
  button:hover:enabled { background:#2c3e50; }
  button:disabled { opacity:0.55; cursor:default; }
  #status { font-size:0.85rem; color:#bbb; white-space:pre-wrap; min-height:1.2em; }
  .ok { color:#7ad07a; }
  .err { color:#e07a7a; }
  .busy { color:#e0c47a; }
  code { background:#222; padding:0 0.3rem; border-radius:3px; }
</style>
</head>
<body>
<h1>MolmoAct2-DROID dashboard</h1>

<div class="grid">
  <div class="tile">
    <h2>external (tripod) <span id="tag-ext" style="color:#777">[?]</span></h2>
    <div id="cam-ext"></div>
  </div>
  <div class="tile">
    <h2>wrist RGB (D457) <span id="tag-wrist" style="color:#777">[?]</span></h2>
    <div id="cam-wrist"></div>
  </div>
</div>

<div class="controls">
  <div class="row">
    <label for="transport">transport:</label>
    <select id="transport">
      <option value="fci">fci (panda_py)</option>
      <option value="rest">rest (motion_server)</option>
    </select>
    <button id="btn-home">home</button>
    <span id="status">loading...</span>
  </div>
  <div class="row">
    <label for="task">instruction:</label>
    <input id="task" type="text" placeholder='e.g. "Put the apple on the plate."' />
    <button id="btn-run">run trial</button>
    <button id="btn-stop" disabled>stop</button>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const status = $("#status");
const WEBRTC_ENABLED = __WEBRTC_ENABLED__;

function mountMjpeg(slotId, tagId, src) {
  const el = document.getElementById(slotId);
  el.innerHTML = `<img src="${src}">`;
  const tag = document.getElementById(tagId);
  if (tag) { tag.textContent = "[mjpeg]"; tag.style.color = "#777"; }
}

async function mountWebRTC(slotId, tagId, cam) {
  const el = document.getElementById(slotId);
  el.innerHTML = '<video autoplay playsinline muted></video>';
  const video = el.querySelector("video");
  const pc = new RTCPeerConnection();
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.ontrack = (ev) => {
    if (ev.track.kind === "video") {
      video.srcObject = ev.streams[0];
    }
  };
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  const r = await fetch(`/offer/${cam}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({sdp: pc.localDescription.sdp, type: pc.localDescription.type}),
  });
  if (!r.ok) throw new Error(`offer ${r.status}`);
  const ans = await r.json();
  await pc.setRemoteDescription(ans);
  const tag = document.getElementById(tagId);
  if (tag) { tag.textContent = "[webrtc]"; tag.style.color = "#7ad07a"; }
}

(async () => {
  const mjpegSrc = { ext: "/stream/ext", wrist: "/stream/wrist_rgb" };
  const slot = { ext: ["cam-ext", "tag-ext"], wrist: ["cam-wrist", "tag-wrist"] };
  for (const cam of ["ext", "wrist"]) {
    const [slotId, tagId] = slot[cam];
    if (WEBRTC_ENABLED) {
      try {
        await mountWebRTC(slotId, tagId, cam);
        continue;
      } catch (e) {
        console.warn(`webrtc ${cam} failed, falling back to mjpeg:`, e);
      }
    }
    mountMjpeg(slotId, tagId, mjpegSrc[cam]);
  }
})();

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const j = await r.json();
    $("#transport").value = j.transport;
    const cls = j.busy ? "busy" : (j.last && j.last.ok === false ? "err" : "ok");
    const prog = j.progress || {chunk: 0, max: 0};
    const progStr = prog.max > 0 ? ` (chunk ${prog.chunk}/${prog.max})` : "";
    const what = j.busy ? `[busy: ${j.what}${progStr}]` : "idle";
    const last = j.last ? JSON.stringify(j.last) : "";
    status.className = cls;
    status.textContent = `${what}\\nlast: ${last}`;
    $("#btn-home").disabled = j.busy;
    $("#btn-run").disabled = j.busy;
    // Stop is only meaningful while a trial is running.
    $("#btn-stop").disabled = !j.busy;
  } catch (e) {
    status.className = "err";
    status.textContent = "status error: " + e;
  }
}

$("#btn-home").addEventListener("click", async () => {
  $("#btn-home").disabled = true;
  const r = await fetch("/api/home", { method: "POST" });
  const j = await r.json();
  status.textContent = JSON.stringify(j);
  await refreshStatus();
});

$("#btn-run").addEventListener("click", async () => {
  const instruction = $("#task").value.trim();
  if (!instruction) { status.textContent = "type an instruction first"; return; }
  $("#btn-run").disabled = true;
  const r = await fetch("/api/task", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ instruction })
  });
  const j = await r.json();
  status.textContent = JSON.stringify(j);
  await refreshStatus();
});

$("#btn-stop").addEventListener("click", async () => {
  $("#btn-stop").disabled = true;
  await fetch("/api/stop", { method: "POST" });
  // The trial loop checks the flag between chunks, so the next chunk
  // boundary will exit. Status refresh re-enables the button if needed.
  await refreshStatus();
});

$("#transport").addEventListener("change", async (ev) => {
  const r = await fetch("/api/transport", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ transport: ev.target.value })
  });
  const j = await r.json();
  status.textContent = JSON.stringify(j);
  await refreshStatus();
});

setInterval(refreshStatus, 1500);
refreshStatus();
</script>
</body>
</html>
"""


# ---------- Flask app ----------

def make_app(state: DashboardState, fps: float = 10.0,
             webrtc=None) -> Flask:
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

    @app.route("/stream/wrist_rgb")
    def stream_wrist_rgb():
        return _stream(lambda: state.camera.latest_jpegs()[1])

    @app.route("/offer/<cam>", methods=["POST"])
    def offer(cam: str):
        if webrtc is None:
            return jsonify({"ok": False, "error": "webrtc disabled on server"}), 503
        if cam not in ("ext", "wrist"):
            return jsonify({"ok": False, "error": f"unknown cam {cam!r}"}), 400
        body = request.get_json(silent=True) or {}
        sdp = body.get("sdp"); type_ = body.get("type")
        if not sdp or not type_:
            return jsonify({"ok": False, "error": "sdp and type required"}), 400
        idx = 0 if cam == "ext" else 1
        getter = lambda: state.camera.latest_rgb_pair()[idx]
        try:
            ans = webrtc.handle_offer(sdp, type_, getter)
        except Exception as e:
            log.exception("WebRTC offer failed")
            return jsonify({"ok": False, "error": f"offer failed: {e}"}), 500
        return jsonify(ans)

    @app.route("/api/status")
    def api_status():
        snap = state.action_lock.snapshot()
        snap["transport"] = state.transport
        snap["molmoact_url"] = state.molmoact_url
        snap["progress"] = state.progress()
        return jsonify(snap)

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
        if not instruction:
            return jsonify({"ok": False, "error": "instruction is required"}), 400
        if not state.action_lock.acquire("task"):
            return jsonify({"ok": False, "error": "another action is in progress"}), 409
        try:
            try:
                result = state.do_task(instruction)
                result["instruction"] = instruction
            except Exception as e:
                result = {"ok": False, "error": f"task failed: {e}", "instruction": instruction}
        finally:
            state.action_lock.release(result)
        return jsonify(result)

    @app.route("/api/transport", methods=["POST"])
    def api_transport():
        body = request.get_json(silent=True) or {}
        new = (body.get("transport") or "").strip()
        if new not in ("fci", "rest"):
            return jsonify({"ok": False, "error": "transport must be fci/rest "
                                                 "(mcp is no longer dashboard-supported)"}), 400
        if not state.action_lock.acquire(f"switch->{new}"):
            return jsonify({"ok": False, "error": "another action is in progress"}), 409
        try:
            result = state.switch_transport(new)
        finally:
            state.action_lock.release(result)
        return jsonify(result)

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--molmoact-url", default="http://localhost:8000")
    ap.add_argument("--transport", choices=["fci", "rest"], default=None,
                    help="override env-var autodetect. mcp is no longer "
                         "supported on the dashboard; use rest instead.")
    ap.add_argument("--rest-step-time-s", type=float, default=2.5)
    ap.add_argument("--exec-rows", type=int, default=3)
    ap.add_argument("--grasp-commit-grip-frac", type=float, default=0.5)
    ap.add_argument("--fine-refinement-travel-rad", type=float, default=0.2)
    ap.add_argument("--max-chunks", type=int, default=30,
                    help="safety cap on action chunks per dashboard trial. "
                         "Matches the CLI runner's --max-chunks default (also "
                         "the per-task max_chunks in droid_tasks.py).")
    ap.add_argument("--mjpeg-fps", type=float, default=10.0)
    ap.add_argument("--no-webrtc", action="store_true",
                    help="disable WebRTC streaming and force MJPEG even if "
                         "aiortc is installed (useful when troubleshooting).")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    transport = args.transport or autodetect_transport()
    if transport == "mcp":
        # autodetect picked mcp from FRANKA_MCP_URL in the env, but the
        # dashboard no longer supports the MCP transport. Surface a clear
        # error rather than letting it fail later in the dropdown / driver.
        log.error("MCP transport is no longer supported on the dashboard. "
                  "Either unset FRANKA_MCP_URL or pass --transport rest/fci. "
                  "(The CLI runner still supports mcp.)")
        return 2
    log.info("transport: %s", transport)

    state = DashboardState(
        molmoact_url=args.molmoact_url,
        transport=transport,
        rest_step_time_s=args.rest_step_time_s,
        exec_rows=args.exec_rows,
        grasp_commit_grip_frac=args.grasp_commit_grip_frac,
        fine_refinement_travel_rad=args.fine_refinement_travel_rad,
        max_chunks=args.max_chunks,
    )

    webrtc = None if args.no_webrtc else _build_webrtc_broadcaster()
    if webrtc is not None:
        log.info("WebRTC enabled (pip install aiortc av if you want to disable, "
                 "or use --no-webrtc)")
    else:
        log.info("WebRTC disabled; using MJPEG")

    app = make_app(state, fps=args.mjpeg_fps, webrtc=webrtc)
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        if webrtc is not None:
            try: webrtc.close()
            except Exception: pass
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
