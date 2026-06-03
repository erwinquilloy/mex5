"""Interactive web dashboard for the MolmoAct2-DROID rig.

Shows live RealSense RGB + depth + external webcam streams, a "Home" button,
and a task-instruction input that runs one inference->exec cycle per click.
Auto-detects the robot transport (fci / rest / mcp) from env vars; a UI
dropdown lets you switch without restarting.

Run on the workstation that has the cameras:
    pip install flask
    export FRANKA_BENCH_EXT_INDEX=0
    export FRANKA_MCP_URL=...   # or FRANKA_REST_HOST / FRANKA_HOST
    python -m benchmarks.scripts.serve_dashboard --port 8080
Then open http://<workstation-ip>:8080/.

Cannot run alongside `run_droid_benchmark.py` -- both want the cameras.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Optional

import numpy as np
from flask import Flask, Response, jsonify, request

from benchmarks.benchmark.dashboard_camera import DashboardCamera, from_env as camera_from_env
from benchmarks.benchmark.molmoact_droid_client import DroidClient
from benchmarks.benchmark.transport import autodetect_transport, make_driver

log = logging.getLogger("dashboard")


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
    ):
        self.molmoact_url = molmoact_url
        self.client = DroidClient(molmoact_url)
        self.camera = camera_from_env()
        self.transport = transport
        self.rest_step_time_s = float(rest_step_time_s)
        self.exec_rows = int(exec_rows)
        self.grasp_commit_grip_frac = float(grasp_commit_grip_frac)
        self.fine_refinement_travel_rad = float(fine_refinement_travel_rad)
        self.driver = make_driver(transport, step_time_s=self.rest_step_time_s)
        self.action_lock = _ActionLock()
        self._driver_lock = threading.Lock()

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
        ext_rgb, wrist_rgb, t_grab_ms = self.camera.latest_rgb_pair()
        if ext_rgb is None or wrist_rgb is None:
            return {"ok": False, "error": "cameras not ready yet"}
        with self._driver_lock:
            state8 = self.driver.state_vec8()
        t0 = time.perf_counter()
        pred = self.client.act(
            external_cam=ext_rgb,
            wrist_cam=wrist_rgb,
            instruction=instruction,
            state=state8,
            num_steps=10,
        )
        rows = self._select_rows(pred.actions)
        with self._driver_lock:
            self.driver.send_chunk(rows, step_dt_s=self.step_dt_for_send_chunk())
        return {
            "ok": True,
            "info": f"executed {len(rows)} rows via {self.transport}",
            "camera_ms": round(t_grab_ms, 1),
            "infer_server_ms": round(pred.server_dt_ms, 1),
            "infer_rtt_ms": round(pred.rtt_ms, 1),
            "wallclock_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "rows_received": int(len(pred.actions)),
            "rows_executed": int(len(rows)),
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
  .grid { display:grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap:1rem; }
  .tile { background:#1c1c1c; border:1px solid #2a2a2a; border-radius:8px; padding:0.6rem; }
  .tile h2 { font-size:0.85rem; margin:0 0 0.4rem; color:#9ad; font-weight:600; }
  .tile img { width:100%; height:auto; background:#000; border-radius:4px; display:block; }
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
  <div class="tile"><h2>external (tripod)</h2><img src="/stream/ext"></div>
  <div class="tile"><h2>wrist RGB (D457)</h2><img src="/stream/wrist_rgb"></div>
  <div class="tile"><h2>wrist depth (jet, &le;<span id="depthmax">2.0</span> m)</h2><img src="/stream/wrist_depth"></div>
</div>

<div class="controls">
  <div class="row">
    <label for="transport">transport:</label>
    <select id="transport">
      <option value="fci">fci (panda_py)</option>
      <option value="rest">rest (motion_server)</option>
      <option value="mcp">mcp (fastmcp)</option>
    </select>
    <button id="btn-home">home</button>
    <span id="status">loading...</span>
  </div>
  <div class="row">
    <label for="task">instruction:</label>
    <input id="task" type="text" placeholder='e.g. "pick up the apple and put it on the plate"' />
    <button id="btn-run">run one chunk</button>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const status = $("#status");

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const j = await r.json();
    $("#transport").value = j.transport;
    const cls = j.busy ? "busy" : (j.last && j.last.ok === false ? "err" : "ok");
    const what = j.busy ? `[busy: ${j.what}]` : "idle";
    const last = j.last ? JSON.stringify(j.last) : "";
    status.className = cls;
    status.textContent = `${what}\\nlast: ${last}`;
    $("#btn-home").disabled = j.busy;
    $("#btn-run").disabled = j.busy;
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

def make_app(state: DashboardState, fps: float = 10.0) -> Flask:
    app = Flask(__name__)
    period = 1.0 / max(fps, 1.0)

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
        return _INDEX_HTML

    @app.route("/stream/ext")
    def stream_ext():
        return _stream(lambda: state.camera.latest_jpegs()[0])

    @app.route("/stream/wrist_rgb")
    def stream_wrist_rgb():
        return _stream(lambda: state.camera.latest_jpegs()[1])

    @app.route("/stream/wrist_depth")
    def stream_wrist_depth():
        return _stream(lambda: state.camera.latest_jpegs()[2])

    @app.route("/api/status")
    def api_status():
        snap = state.action_lock.snapshot()
        snap["transport"] = state.transport
        snap["molmoact_url"] = state.molmoact_url
        return jsonify(snap)

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
        if new not in ("fci", "rest", "mcp"):
            return jsonify({"ok": False, "error": "transport must be fci/rest/mcp"}), 400
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
    ap.add_argument("--transport", choices=["fci", "rest", "mcp"], default=None,
                    help="override env-var autodetect")
    ap.add_argument("--rest-step-time-s", type=float, default=2.5)
    ap.add_argument("--exec-rows", type=int, default=3)
    ap.add_argument("--grasp-commit-grip-frac", type=float, default=0.5)
    ap.add_argument("--fine-refinement-travel-rad", type=float, default=0.2)
    ap.add_argument("--mjpeg-fps", type=float, default=10.0)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    transport = args.transport or autodetect_transport()
    log.info("transport: %s", transport)

    state = DashboardState(
        molmoact_url=args.molmoact_url,
        transport=transport,
        rest_step_time_s=args.rest_step_time_s,
        exec_rows=args.exec_rows,
        grasp_commit_grip_frac=args.grasp_commit_grip_frac,
        fine_refinement_travel_rad=args.fine_refinement_travel_rad,
    )

    app = make_app(state, fps=args.mjpeg_fps)
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
