# MolmoAct2 on Franka Emika — Benchmark

End-to-end harness for evaluating **MolmoAct2** on a Franka Research robot:

| Path | Model | Where it runs | Auto-scored? | Purpose |
|---|---|---|---|---|
| **DROID real-robot** (`scripts/run_droid_benchmark.py`) | `allenai/MolmoAct2-DROID` | upstream `host_server_droid.py` on HPC + tunnel + workstation client | no (human grader) | Table 6 reproduction of arXiv:2605.02881 (the course exercise target) |

## Topology (DROID path)

```
RealSense D457 (wrist)    ──┐
USB webcam left  (ext 1)  ──┤
USB webcam right (ext 2)  ──┴► workstation ──► ssh -L 8000 ──► HPC A100
                               │                                  │
                               │                   uv run host_server_droid.py
                               │                                  │
                               ▼                                  │
                   one of three transports ──► Franka  ◄──────────┘  (actions[N,8])
                   ┌────────────────────────────────────────────────┐
                   │ fci  : panda_py.JointPosition (direct libfranka) │
                   │ rest : motion_server (cartesian xyz+ZYX deltas)  │
                   │ mcp  : fastmcp ──► motion_server (same as rest)  │
                   └────────────────────────────────────────────────┘
```

Ext 1 and ext 2 are stacked side-by-side into a single `(H, 2W, 3)` image before
being sent to the model (matches the DROID training convention for multi-camera rigs).

Action contract (from the model card): `actions[N, 8]` = `[q1..q7, gripper]`,
**absolute joint targets in radians + gripper command**. Gripper ≥ 0.5 ⇒ close.

## Robot transports

The runner can drive the Franka three ways. Pick one with `--transport`:

| Flag | Path | Pros | Cons |
|---|---|---|---|
| `fci` *(default)* | `panda_py` → libfranka directly. Joint-position streaming with substep ramping. | Native to DROID's action space (joint targets). Fastest. Full per-step velocity control. | Needs `panda_py`/libfranka on the workstation. Reflex tuning matters. |
| `rest` | POST `moveToCartesian` per row to `franka/cpp/motion_server`. We FK the model's `q[7]` locally and send `(xyz, ΔZYX-deg)`. | One interface, server-side safety. No FCI on the workstation. | Each row blocks ~`--rest-step-time-s` (default 2.5 s). Reprojects through cartesian. |
| `mcp` | Same as `rest` but routed through `franka/python/mcp_server.py` (fastmcp). | Same arm callable by any MCP client / agent. | One extra hop. Needs `fastmcp` installed. |

The REST and MCP paths share `FrankaRestDriver`; MCP is a thin subclass that
swaps the transport call (`benchmark/franka_mcp_driver.py`). Both rely on the
patched `motion_server.cpp` that adds `readJointState` (returns `[q0..q6,
gripper_width]`) and lowers the per-move tf floor from 5.0 s → 0.5 s.

### Rebuilding motion_server (one-time, REST/MCP only)

On the box that runs motion_server (typically the Linux host with libfranka):

```bash
cd franka/cpp
mkdir -p build && cd build
cmake .. && make
./motion_server     # binds 0.0.0.0:34568
```

If you didn't pull our patch, `--transport rest` will error on the first
`readJointState` call.

### Starting the MCP server (MCP only)

```bash
cd franka/python
pip install fastmcp
python3 mcp_server.py   # default: streamable-http on 0.0.0.0:8085/franka
```

Then set `FRANKA_MCP_URL=http://<that host>:8085/franka` (or pass `--mcp-url`).

## Reproducibility caveats vs. Table 6

- **Dual external cameras**: Table 6's protocol randomizes the external camera pose
  each trial. We mount two USB webcams at the back of the robot arm, both facing
  45° inward toward the work area (`FRANKA_BENCH_EXT_INDEX` + `FRANKA_BENCH_EXT_INDEX2`);
  they are stacked side-by-side before inference. If no external cam is set,
  `dual_camera.py` falls back to duplicating the wrist image (substantially OOD
  for the model — expect a large gap from the paper numbers).
- **OOD objects**: bring objects that the model has not seen in DROID training.
  Common household items work; avoid anything that looks like the demo videos.
- **Controller**: paper uses the DROID NUC + polymetis stack; we use `panda_py`
  directly. Same action space, different timing characteristics.

## Tasks (Table 6 of arXiv:2605.02881)

Five tasks × 15 trials each. The `task_id` column is what `--tasks` takes;
pass multiple ids to filter (e.g. `--tasks apple_on_plate knife_in_box`).
Defaults to all five if omitted. Operator grades success after each trial.

| task_id | instruction | paper | scene setup |
|---|---|---:|---|
| `apple_on_plate` | "Put the apple on the plate." | 100.0% | Real apple + empty plate within reach. Use OOD objects. |
| `pipette_in_tray` | "Put the pipette in the tray." | 86.7% | Pipette on table, tray on the side. OOD pipette color/shape preferred. |
| `red_cube_in_tape_roll` | "Put the red cube inside the tape roll." | 93.3% | Red cube ~2–3 cm; roll of tape lying flat. |
| `knife_in_box` | "Put the knife in the box." | 93.3% | Plastic / safe knife on table, open box within reach. |
| `objects_in_bowl` | "Put the objects in the bowl." | 62.0% | Multiple small objects scattered; success only when ALL visible target objects are in the bowl. |

Per the Table 6 protocol, the external camera pose is **re-randomized
between trials**, and the runner prompts you to do so between trials.
The `instruction` text above is the exact string sent to MolmoAct2 —
don't paraphrase, it has to match the training distribution.

Canonical Table 6 source: see `benchmarks/benchmark/droid_tasks.py`
(`TASKS` list) — that's where success rates and instructions live; the
runner pulls from it and prints the side-by-side at the end.

## Setup

### 1. HPC (model server)

See `benchmarks/hpc/README.md`. TL;DR on the HPC node:
```bash
git clone https://github.com/allenai/molmoact2.git ~/molmoact2 && cd ~/molmoact2
uv sync && uv run hf download allenai/MolmoAct2-DROID
uv run python examples/droid/host_server_droid.py --host 0.0.0.0 --port 8000 --dtype bfloat16
```

### 2. Workstation (controls Franka, runs the eval client)

```bash
pip install -r benchmarks/requirements.txt
ssh -N -L 8000:localhost:8000 erwin.quilloy@ai-n002.hpc.coe.upd.edu.ph &
curl http://localhost:8000/act    # confirm: {"status": "ok", "repo_id": "...", ...}
```

Workspace prep:
- Wrist D457 mounted, USB-C / FAKRA switch set per `franka/README.md`.
- Both external USB webcams plugged in. The RealSense also registers as
  `/dev/video*`, so a bare `ls /dev/video*` is misleading. Use
  `v4l2-ctl --list-devices` and pick the two indices that are *not* under
  "Intel RealSense" (on `airscan4`: index `2` = left cam, index `0` = right cam;
  no flip needed — raw output is spatially correct).
- Franka in white/unlocked state, FCI activated (see `franka/python/basic.py`).

### 3. Run

Common env vars (cameras + model server):

```bash
# Two external webcams at the back of the arm, 45° inward — stacked left→right.
export FRANKA_BENCH_EXT_INDEX=2          # left physical cam  (airscan4: /dev/video2)
export FRANKA_BENCH_EXT_INDEX2=0         # right physical cam (airscan4: /dev/video0)
# No per-camera flip needed: raw output is spatially correct.
export FRANKA_BENCH_CAM_W=640            # RealSense D455 has no 256x256 mode
export FRANKA_BENCH_CAM_H=480
export FRANKA_BENCH_REST_CAM_DZ_M=-0.09  # downward offset: drive toward table at grasp; Z floor 0.016 stops it (REST/MCP)
export FRANKA_BENCH_FCI_CAM_DZ_M=-0.05   # same idea for FCI (no Z floor there yet)
# optional: FRANKA_BENCH_WRIST_SERIAL=<D457 serial>  to pin the wrist cam
```

#### Switching between FCI and REST

`motion_server` and `panda_py` cannot share FCI — the protocol allows exactly
one client. Switching transports is a kill-the-other-side ritual:

- **FCI → REST.** Stop any benchmark first
  (`pgrep -af run_droid_benchmark` should be empty), then launch
  `motion_server` in its own terminal (`cd franka/cpp/build && ./motion_server`,
  wait for `Listening at: http://0.0.0.0:34568`). Now run the benchmark with
  `--transport rest`.
- **REST → FCI.** Stop motion_server (Ctrl-C its terminal, or
  `pkill -f motion_server`), confirm with `pgrep -af motion_server` (empty),
  then run the benchmark without `--transport` (FCI is the default).

If you forget and both clients try FCI, you'll see
`libfranka: Connection timeout` from whichever starts second.

#### FCI (default — direct libfranka)

```bash
export FRANKA_HOST=192.168.2.100         # robot's FCI IP

# full Table 6 (5 tasks x 15 trials, ~hours)
python -m benchmarks.scripts.run_droid_benchmark

# single task, one trial -- smoke test the whole loop end-to-end
python -m benchmarks.scripts.run_droid_benchmark --tasks apple_on_plate --trials 1
```

One-liner (smoke test):
```bash
FRANKA_HOST=192.168.2.100 python -m benchmarks.scripts.run_droid_benchmark --tasks apple_on_plate --trials 1
```

#### REST (motion_server)

In one terminal, start motion_server:

```bash
cd franka/cpp/build && ./motion_server
# wait for "Listening at: http://0.0.0.0:34568"; leave it running
```

In another terminal, run the benchmark:

```bash
export FRANKA_REST_HOST=192.168.2.1      # the box running motion_server, NOT the robot
python -m benchmarks.scripts.run_droid_benchmark \
    --transport rest --rest-step-time-s 2.0 \
    --tasks apple_on_plate --trials 1
```

`--rest-step-time-s` is the time motion_server interpolates each commanded
EE pose over. Lower = faster but more reflex risk: at `0.5 s` a large
per-step wrist roll (common on chunk 1) trips
`cartesian_motion_generator_velocity_discontinuity` and motion_server
returns 400. Start at `2.0`, then dial down toward `1.0` once you trust
the loop. Quick health check before launching:

```bash
curl -m 3 -X POST http://192.168.2.1:34568/api/floats \
     -H 'Content-Type: application/json' \
     -d '{"readJointState":[]}'
# expect JSON with 7+ joint angles
```

One-liner:
```bash
FRANKA_REST_HOST=192.168.2.1 python -m benchmarks.scripts.run_droid_benchmark --transport rest --rest-step-time-s 2.0 --tasks apple_on_plate --trials 1
```

#### MCP (fastmcp → motion_server)

```bash
export FRANKA_MCP_URL=http://<mcp-host>:8085/franka
python -m benchmarks.scripts.run_droid_benchmark \
    --transport mcp --rest-step-time-s 2.5 \
    --tasks apple_on_plate --trials 1
```

One-liner:
```bash
FRANKA_MCP_URL=http://<mcp-host>:8085/franka python -m benchmarks.scripts.run_droid_benchmark --transport mcp --rest-step-time-s 2.5 --tasks apple_on_plate --trials 1
```

`FRANKA_HOST` is **only** the FCI path; REST/MCP each have their own variable
(no fallback) so you can't accidentally point one transport at the other's host.

#### Compare runs across transports

After running the same `--tasks/--trials` under multiple transports, load all
the result files into a side-by-side table:

```bash
python -m benchmarks.scripts.compare_runs benchmarks/results/*.json
# or pick specific files:
python -m benchmarks.scripts.compare_runs \
    benchmarks/results/droid-fci-run.json \
    benchmarks/results/droid-rest-run.json \
    benchmarks/results/droid-mcp-run.json \
    --label fci --label rest --label mcp
```

One-liner (labelled):
```bash
python -m benchmarks.scripts.compare_runs benchmarks/results/droid-fci-run.json benchmarks/results/droid-rest-run.json benchmarks/results/droid-mcp-run.json --label fci --label rest --label mcp
```

Prints overall success, per-task success (cell shows `pct (n_trials)`), and
mean/p50/p95 for `infer_server_ms` / `e2e_ms` / `motion_rest_ms` per column.
The motion_rest jump from FCI to REST/MCP is where the transport overhead lives.

## Dashboard (interactive web UI)

Single-page Flask app for hands-on benchmarking: live camera tiles, a
DROID task dropdown, runtime knobs for resolution and wrist-cam XYZ
offsets, a launcher for `motion_server`, a live gripper state pill, and
two action buttons (Home + Run Benchmark) with a Stop that aborts the
in-flight chunk and re-homes. Replaces the old CLI prompts for
operator-in-the-loop work.

The dashboard is **REST-only** — it talks to `motion_server` exclusively.
FCI and MCP were removed from the dashboard; the CLI runner
(`run_droid_benchmark.py`) still supports both.

### What you get

- **Three live camera tiles:** 2 external webcams + wrist RealSense RGB.
  The two externals are stacked horizontally before being fed to
  MolmoAct2 (DROID training convention). Streaming defaults to
  **WebRTC** when `aiortc` + `av` are installed, with automatic fallback
  to **MJPEG** otherwise. Per-tile tag shows `[webrtc]` or `[mjpeg]`.
- **Gripper status pill** in the page header (open / holding / closed /
  busy / error) updated from the latest `readJointState`. Polled every
  1.5 s and skipped while a trial holds the driver lock.
- **Task dropdown:** populated from `droid_tasks.TASKS` (the Table 6
  list). Each task carries its own `max_chunks` and instruction; the
  Run button just executes whichever task is selected.
- **Home button** → `driver.home()` (cartesian-interpolated to FK'd
  home).
- **Run benchmark button** → one full trial of the selected task
  (capture → infer → exec, up to the task's `max_chunks`).
- **Stop button** → cancels mid-chunk (~1–2 s worst-case latency) and
  auto-homes the arm.
- **motion_server launcher panel:** start / stop / restart the
  `motion_server` binary as a subprocess. Restart = re-runs
  `initialize()`, which calls `goHome` — so it doubles as an arm reset
  beyond `driver.home()`'s reach.
- **Resolution dropdown:** swap between 320×240, 424×240, 640×480,
  848×480, 1280×720 at 30 fps without restarting the dashboard. Tears
  down + rebuilds the camera pipelines under a lock.
- **Wrist-cam XYZ offset inputs:** set DX / DY / DZ live; pushed through
  `FrankaRestDriver.set_cam_offsets()`. Useful while tuning the
  cam-offset for terminal-row alignment.
- **Hold-until-transported guard:** since no Table 6 task carries a
  fixed `target_zone_xy`, the dashboard falls back to a distance
  heuristic — once the policy first grasps an object, no policy-emitted
  open is honoured until the TCP has moved ≥ `hold_min_dist_m` (default
  0.08 m) from the pickup XY. Configurable per trial from the panel; set
  to 0 to disable. If a task ever gets a `target_zone_xy` defined, that
  static box takes precedence.
- **Autonomous grasp retry:** if the previous chunk commanded close but
  the gripper width is now < 5 mm (jaws closed-empty), the dashboard
  treats it as a missed grasp: force-opens the jaws, drops the chunk
  the policy planned from the (now stale) closed-empty state, and
  re-prompts MolmoAct2 on the next iteration with fresh cameras + open
  jaws so the model can re-plan a corrected approach from vision. The
  same code path catches mid-transport slips: if you were holding the
  object and it falls out, the next chunk's empty-jaw reading triggers
  the same retry. Budgeted by `grasp_retry_limit` (default 3, exposed
  in the dashboard panel + sent on each `/api/task` POST). On a
  successful grasp the budget resets so a later slip + re-grasp gets
  its own allowance. On budget exhaustion the trial bails with
  `stopped_by="grasp_retry_budget_exhausted"`. The result payload
  reports `grasp_fail_count`, `grasp_retries_used`, `grasp_retry_limit`,
  and `ever_held` so each failure mode is distinguishable from the
  payload alone.
- **Persisted settings:** the four runtime knobs (XYZ offsets,
  resolution, hold-min-dist, retry budget) are saved to
  `~/.cache/mex5_dashboard_settings.json` (override with
  `--settings-file`) on every apply / Run click and reloaded at startup.
  Tune once, restart freely, your inputs come back with the values you
  left. Reset by deleting the file.

### Prerequisites

The dashboard owns the cameras and talks to `motion_server`, so:

- [ ] `host_server_droid.py` running on the HPC at port 8000
- [ ] SSH tunnel up: `ssh -N -L 8000:localhost:8000 <hpc>` (test with
      `curl http://localhost:8000/act`)
- [ ] Cameras connected: RealSense D45x wrist + two USB webcams
- [ ] CLI runner stopped: `pkill -f run_droid_benchmark`
- [ ] `FRANKA_REST_HOST` exported (the dashboard refuses to start
      without it now)
- [ ] `motion_server` binary built (`cd franka/cpp/build && cmake .. && make`)
      — you can start it from the dashboard's launcher panel, no need
      to spawn it manually
- [ ] *(Optional)* `pip install aiortc av` for WebRTC streaming

### Launching

Pick indices for your two external webcams (see `v4l2-ctl --list-devices`),
then export and run:

```bash
cd ~/erwin/mex5
export FRANKA_REST_HOST=192.168.2.1            # motion_server host, NOT the robot FCI IP
export FRANKA_BENCH_EXT_INDEX=0                # first external webcam (capture node)
export FRANKA_BENCH_EXT_INDEX2=2               # second external webcam
# optional: export FRANKA_BENCH_WRIST_SERIAL=<D45x serial>
# optional: per-camera rotation if needed: FRANKA_BENCH_EXT_ROT_DEG / _EXT2_ROT_DEG ∈ {0,90,180,270}

python -m benchmarks.scripts.serve_dashboard --port 8080 \
    --molmoact-url http://localhost:8000 \
    --rest-step-time-s 2.5
```

Then open `http://<workstation-ip>:8080/` in a browser. The default
`--motion-server-bin` searches for `<repo>/franka/cpp/build/motion_server`;
override with `--motion-server-bin /path/to/binary` if it's elsewhere.

### Recommended workflow

The end-to-end flow for a single benchmark trial:

1. **Open the dashboard** at `http://<workstation-ip>:8080/`. Confirm
   all three tiles are live and the tag reads `[webrtc]` (or
   `[mjpeg]` if WebRTC isn't installed).
2. **Start motion_server.** From the *motion_server* panel, click
   **start**. The pill should flip from "stopped" to "running (pid …)".
   This also runs `initialize()` → `goHome`, so the arm should jog to
   home on first start.
3. **Tune cameras (optional).** Pick a resolution from the dropdown if
   the default 640×480 isn't what you want, and click **apply**. The
   browser will reconnect the streams automatically.
4. **Set wrist-cam offsets (optional).** Type DX / DY / DZ in the
   *wrist-cam XYZ offset* panel and click **apply**. These are pushed
   through to `FrankaRestDriver.set_cam_offsets()` — they take effect
   on the next chunk.
5. **Set the hold-until-transported radius.** Default 0.08 m. Lower
   it (e.g. 0.04 m) if your target tray is right next to pickup; 0
   disables the guard entirely.
6. **Pick a task** from the dropdown (`apple_on_plate`,
   `pipette_in_tray`, etc.). The selected task's instruction shows
   underneath.
7. **Home the arm** (Home button) and place the scene per the task's
   `setup_notes`.
8. **Click Run benchmark.** The dashboard loops
   capture → MolmoAct2 → execute, up to the task's `max_chunks`.
   Status panel shows live chunk progress, gripper state, and any
   suppression warnings.
9. **Use Stop** at any time. The current chunk aborts within ~1–2 s
   and the arm auto-homes.

When you're done: Ctrl-C the dashboard. The launcher will gracefully
stop the motion_server subprocess on exit.

### Useful flags

| Flag | Default | What it does |
|---|---|---|
| `--port` | `8080` | dashboard HTTP port |
| `--host` | `0.0.0.0` | bind address |
| `--molmoact-url` | `http://localhost:8000` | MolmoAct2-DROID server |
| `--rest-step-time-s` | `2.5` | per-row REST move time (slow zone) |
| `--exec-rows` | `3` | rows of each chunk to execute when the policy is in fine-refinement mode |
| `--max-chunks` | `30` | safety cap when the selected task has no `max_chunks` of its own |
| `--mjpeg-fps` | `30.0` | MJPEG fallback refresh rate (ignored when WebRTC is active) |
| `--no-webrtc` | off | force MJPEG even if `aiortc` is installed |
| `--motion-server-bin` | autodetect | path to the motion_server binary; defaults to `<repo>/franka/cpp/build/motion_server` |
| `--motion-server-log` | unset | write motion_server stdout+stderr to this file |
| `--settings-file` | `~/.cache/mex5_dashboard_settings.json` | JSON file persisting the runtime knobs across restarts |

### REST endpoints (for debugging / scripting)

| Path | Method | Use |
|---|---|---|
| `/api/status` | GET | live snapshot (busy, progress, resolution, offsets, gripper, motion_server) |
| `/api/tasks` | GET | enumerate the Table 6 tasks + their `target_zone_xy` |
| `/api/resolutions` | GET | available presets + current |
| `/api/resolution` | POST `{width, height, fps}` | swap presets |
| `/api/offsets` | GET / POST `{dx, dy, dz}` | read or write wrist-cam offsets (persisted on POST) |
| `/api/preferences` | GET | persisted trial knobs (hold-min-dist, retry budget) for UI repopulation |
| `/api/home` | POST | drive to home |
| `/api/task` | POST `{task_id, instruction, max_chunks, hold_min_dist_m, grasp_retry_limit}` | run one trial (persists hold-min-dist + retry budget) |
| `/api/stop` | POST | signal abort; trial loop interrupts current chunk + homes |
| `/api/motion_server/{start,stop,restart,status}` | POST/GET | manage the subprocess |

### Preview-only mode (no hardware)

If you just want to eyeball the UI on a machine without cameras or
`motion_server`:

```bash
python -m benchmarks.scripts.preview_dashboard --port 8081
```

The three tiles show placeholder JPEGs ("PREVIEW MODE — no camera
attached"), the task dropdown still loads from `droid_tasks.py`, and
every button POST returns a no-op. Useful for demoing or styling work.

### Dashboard vs CLI runner

| Need | Use |
|---|---|
| Hands-on "type an instruction and watch the arm try it" | dashboard |
| Live camera streams + per-trial knobs (offsets, resolution, hold-dist) | dashboard |
| Launch/restart motion_server from a UI | dashboard |
| Reproduce Table 6 with N trials, per-task scoring, results JSON | CLI runner |
| Auto-home + operator-graded success between trials | CLI runner |
| FCI or MCP transports | CLI runner |
| `--hold-until-target` with a fixed `target_zone_xy` | CLI runner |

They share `FrankaRestDriver`, `DroidClient`, `droid_runner._enforce_*`
helpers, so model + transport behavior is identical — the dashboard is
an interactive shell around the same core.

#### Mutual exclusion — only one can run at a time

The dashboard and the CLI runner can **never** run simultaneously:

- **RealSense.** `pyrealsense2` lets one process open the wrist cam at
  a time. Both `DashboardCamera` and `DualCamera` call
  `pipeline.start()` on the same serial; whichever starts second
  errors with `xioctl(VIDIOC_S_FMT) failed, errno=16` or
  `Device or resource busy`.
- **motion_server.** Holds FCI exclusively. If a CLI runner started
  with `--transport fci` is already up, motion_server can't acquire
  FCI.

Symptoms when you forget:

```
# trying to start the CLI while the dashboard is up
RuntimeError: Couldn't resolve requests
# or
xioctl(VIDIOC_S_FMT) failed, errno=16 Last Error: Device or resource busy
```

Switching between the two cleanly:

```bash
# dashboard → CLI
# Ctrl-C the dashboard; pgrep -af serve_dashboard should be empty.
# If RealSense stays busy, unplug + replug the USB cable.
python -m benchmarks.scripts.run_droid_benchmark --tasks apple_on_plate ...

# CLI → dashboard
pkill -f run_droid_benchmark
python -m benchmarks.scripts.serve_dashboard --port 8080
```

#### Mutual exclusion — only one can run at a time

The dashboard and the CLI runner can **never** run simultaneously,
regardless of transport:

- **RealSense.** `pyrealsense2` lets one process open the wrist cam at a
  time. Both `DashboardCamera` and `DualCamera` call `pipeline.start()`
  on the same serial; whichever starts second errors with a libuvc /
  `Device or resource busy` style message.
- **FCI.** `panda_py` and `motion_server` both hold FCI exclusively, so
  even if you somehow avoided the camera clash you'd get
  `Connection actively refused remote peer` from libfranka on the
  second client.

Symptoms when you forget:

```
# trying to start the CLI while the dashboard is up
RuntimeError: Couldn't resolve requests
# or
RuntimeError: libfranka: Connection error: Connection actively refused...
```

Switching between the two cleanly:

```bash
# dashboard -> CLI
# (Ctrl-C the dashboard terminal, wait a beat for the cams to release)
python -m benchmarks.scripts.run_droid_benchmark --tasks apple_on_plate ...

# CLI -> dashboard
pkill -f run_droid_benchmark        # or Ctrl-C the runner
python -m benchmarks.scripts.serve_dashboard --port 8080
```

If you want the CLI's results-JSON-and-grading workflow **plus** a live
browser view, run the CLI as usual and start `serve_live` (read-only
camera viewer) alongside — that one connects to the dashboard-less live
endpoints without claiming the cameras itself.

## End-to-end walkthrough (Linux workstation, fresh shell)

Step-by-step recipe to bring everything up and run all three transports back
to back, then compare. Use this as a checklist when you sit down at the rig.

Do these in order from a fresh login. Steps 0–4 are common to every
transport; then pick **one** of 5a/5b/5c.

### 0. cd into the repo and pull latest

```bash
cd ~/erwin/mex5                     # path may differ per box; adjust below
git pull --ff-only origin main
```

### 1. Activate the venv

Workstation deps (panda-py, RealSense SDK, json-numpy) live in the project
venv. Outside it the system Python doesn't even have `python` aliased to
`python3`, let alone the right libraries.

```bash
source ~/erwin/mex5/.venv/bin/activate
# pinocchio's .so must be on the loader path for panda-py kinematics:
export LD_LIBRARY_PATH=/opt/openrobots/lib
```

Your prompt should now start with `(.venv)`. Quick sanity check:
```bash
which python
python -c "import panda_py, pyrealsense2; print('ok')"
```

If `panda_py` is missing, or its `.so` links against `libfranka 0.9.x`
(the PyPI wheel), this rig is an FR3 (server v9) and you need the source
build against libfranka 0.15. Recipe is in your auto-memory
(`panda_py_fr3_install.md`); short form:
```bash
CMAKE_PREFIX_PATH=/usr/local:/opt/openrobots \
CMAKE_ARGS='-DLIBFRANKA_VER=0x000f00 -DCMAKE_CXX_FLAGS=-I/opt/openrobots/include' \
  pip install --no-binary :all: \
  "git+https://github.com/JeanElsner/panda-py.git@jean/chore/update-upstream"
```

### 2. SSH tunnel to the HPC model server

The benchmark client hits `http://localhost:8000/act`, which the tunnel
forwards to `host_server_droid.py` on the HPC. `-f` lets ssh prompt for
your password in the foreground, then forks to the background (avoids the
`Stopped (tty input)` you get from `ssh ... &` with backgrounded auth).

```bash
# kill any stale tunnel from a previous session
pkill -f "ssh -N -L 8000:localhost:8000" 2>/dev/null

ssh -f -N -L 8000:localhost:8000 erwin.quilloy@ai-n002.hpc.coe.upd.edu.ph

# verify the model server replies
curl -m 3 http://localhost:8000/act
# expect: {"status": "ok", "repo_id": "...", ...}
```

If `curl` connection-refuses, the tunnel is up but `host_server_droid.py`
isn't running on the HPC — SSH in and check the tmux/screen session
(see `benchmarks/hpc/README.md`).

### 3. Robot prep (Desk)

In a browser → `https://192.168.2.100/desk`:
- Joints **unlocked** (LEDs white, not blue).
- **FCI active** (top-of-page mode chip, not "Programming"/"Execution").
- Clear any red reflex/error banners.

The runner will auto-home at the start of each trial; you don't need to
home manually.

### 4. Camera env vars (all transports)

```bash
# Two external webcams (both at the back of the arm, 45° inward).
# Stacked left→right into a single (H, 2W, 3) image before inference.
export FRANKA_BENCH_EXT_INDEX=2            # left physical cam  (airscan4: /dev/video2)
export FRANKA_BENCH_EXT_INDEX2=0           # right physical cam (airscan4: /dev/video0)
# No per-camera flip needed: raw output is already spatially correct.
export FRANKA_BENCH_CAM_W=640              # RealSense D455 has no 256x256 mode
export FRANKA_BENCH_CAM_H=480
# Per-camera rotation if needed: FRANKA_BENCH_EXT_ROT_DEG / FRANKA_BENCH_EXT2_ROT_DEG ∈ {0,90,180,270}

# Required on the lab rig (airscan4): wrist RealSense mounted UNDER the gripper.
# No X offset. 5 cm downward Z offset corrects the cam→TCP vertical gap so
# the model's grasp terminal pose lands on the object rather than above it.
# These DX/DZ values fire on the grasp chunk's terminal row only (the
# default OFFSET_MODE=grasp_terminal). The 'always' mode is documented
# below as an opt-in for whole-trajectory perception bias; not recommended
# on FCI (orientation drift from per-row IK branch flips) and only on REST
# with the two-phase fast time DISABLED.
export FRANKA_BENCH_REST_CAM_DZ_M=-0.09    # downward offset: descend toward table at grasp; clamped at Z floor 0.016 m (REST/MCP)
export FRANKA_BENCH_FCI_CAM_DZ_M=-0.05     # FCI has no Z floor yet — keep small (terminal-pose only)

# Wrist-cam reorientation (added after the under-gripper relocation, since the
# camera body sits in a different orientation than the DROID-canonical mount).
# Find values by launching the dashboard, grasping something, and walking 0 →
# 90 → 180 → 270 until the gripper jaws sit at the bottom of the wrist preview
# and world-down = image-down. Then mirror that into the runner via these:
# export FRANKA_BENCH_WRIST_ROT_DEG=0          # 0 / 90 / 180 / 270
# export FRANKA_BENCH_WRIST_FLIP_H=0           # set to 1 if cam was mounted mirrored
# export FRANKA_BENCH_WRIST_FLIP_V=0

# optional: FRANKA_BENCH_WRIST_SERIAL=<D457 serial> to pin the wrist cam
# optional: FRANKA_BENCH_LOCK_GRIPPER_DOWN=1
#   FCI only — IK-solve each row with the TCP Z-axis pinned to vertical so
#   the gripper can't tilt mid-trajectory. Helps top-down grasps (apple,
#   cube, knife, objects_in_bowl). Skip for pipette_in_tray (long tool;
#   model may genuinely want a tilted approach). Also skip if the table is
#   low — vertical IK can become unreachable on the FR3 and rows get
#   silently dropped (arm freezes hovering above the object).

# optional REST/MCP two-phase approach speed — race through free space,
# slow only inside the precision zone near the table. Both vars must be
# set; either unset leaves single-speed behavior (--rest-step-time-s
# applies to every row).
# export FRANKA_BENCH_REST_FAST_STEP_TIME_S=2.0   # per-row time when above zone
# export FRANKA_BENCH_REST_SLOW_ZONE_Z_M=0.20     # TCP Z (m) at/below which slow time applies
#   Pick SLOW_ZONE_Z_M ≈ table_height + 0.20. On airscan4 the table is
#   roughly at base Z = 0, so 0.20 means "slow for the last 20 cm of
#   descent". Tune by watching where the gripper transitions.
```

Confirm the external indices with `v4l2-ctl --list-devices` and pick the two
**not** under "Intel RealSense". On `airscan4`: `EXT_INDEX=2` (left physical cam),
`EXT_INDEX2=0` (right physical cam). Both need `FLIP_H=1` — their drivers produce
a horizontally mirrored image.

#### Cam-offset gating modes (advanced)

The DX/DZ values above are applied per the env var
`FRANKA_BENCH_REST_CAM_OFFSET_MODE` / `FRANKA_BENCH_FCI_CAM_OFFSET_MODE`.
Default is `grasp_terminal`; the other two are opt-in for specific
calibration scenarios. **Don't change this unless the default's behavior
isn't matching what you observe.**

| Mode | When the DX/DZ shift fires | Use when | Gotchas |
|---|---|---|---|
| `grasp_terminal` *(default)* | Last row of chunks whose terminal action commands gripper-close. | The model commits to a grasp on its own, you just need to fine-tune the close-pose alignment (small geometric wrist-cam→TCP offset). | If the model never produces a grasp chunk (it hovers/aborts), the offset never fires. Symptom: shift values change behavior at all only when the gripper closes. |
| `every_terminal` | Last row of every chunk. | The model behaves like above but you want each chunk-boundary pose corrected too (e.g. when stretching small offsets across long approach trajectories). | Slightly warps the approach path; the policy "thinks" it ended row N at A but the robot ended at A + offset, so subsequent inference is on slightly different state. |
| `always` | Every row of every chunk — a constant translation in the base frame applied to all commanded TCP poses. | Whole-trajectory perception bias correction (model's spatial estimate is consistently off by a few cm in X/Y/Z across the entire trial). Use when the model never even reaches grasp on its own and bias is roughly constant across workspace positions. | **FCI:** per-row FK→shift→IK can pick a different IK branch, causing gripper orientation to drift across the trajectory. Verify orientation on the wrist cam stream. **REST:** safe, but don't combine with very short `FAST_STEP_TIME_S` (≤ 1.0 s) — first-chunk-from-home with 1.0 s tf + always shift trips `cartesian_motion_generator_joint_acceleration_discontinuity`. The README's recommended `FAST_STEP_TIME_S=2.0` is the safe floor; if you need faster, either disable two-phase entirely or test small. |

To opt into one of the non-default modes (REST/MCP example):
```bash
export FRANKA_BENCH_REST_CAM_OFFSET_MODE=always   # or every_terminal
export FRANKA_BENCH_REST_CAM_DX_M=-0.06           # whole-trajectory shift values
export FRANKA_BENCH_REST_CAM_DZ_M=-0.08
```
Mirror with `FRANKA_BENCH_FCI_CAM_OFFSET_MODE` for FCI runs.

#### Workspace XYZ safety clip (always on)

The REST driver hard-clamps every commanded TCP target X/Y/Z into the
airscan4 safe box before sending. X/Y catch bad cam offsets,
out-of-distribution policy actions, or an accidental `OFFSET_MODE=always`.
The Z floor lets the grasp descent use a deliberately large downward cam
offset (`FRANKA_BENCH_REST_CAM_DZ_M=-0.09`) to drive the gripper down
way to the table without pushing through it — the descent stops at
`_LAB_Z_MIN`.

| Axis | Range (base-frame metres) |
|---|---|
| X | `[0.0, 0.57]` |
| Y | `[-0.4, +0.4]` |
| Z | `[0.016, ∞)` |

On the grasp terminal row (gripper closing + cam offset applied), the
driver descends to the clamped floor with the jaws **open**, then closes
once at table height — so the gripper can't shut mid-air and bulldoze the
object on the way down. See `send_chunk` in `franka_rest_driver.py`.

If a row gets clamped, the driver prints one line per row, e.g.:

```
[FrankaRestDriver] clamped TCP target XYZ (+0.420, -0.020, -0.180) -> (+0.420, -0.020, +0.016) m to lab box X[0.00,0.57] Y[-0.40,+0.40] Z[>= 0.016]
```

Values are hardcoded in `franka_rest_driver.py`
(`_LAB_X_MIN/MAX`, `_LAB_Y_MIN/MAX`, `_LAB_Z_MIN`). If you're running on a
different rig, edit them there — there's intentionally no env var so they
can't be disabled by accident. (The X/Y clamp is also in `panda_driver.py`
for the FCI path; the Z floor + descend-then-close is REST-only.)

#### `--hold-until-target` (debug override)

Opt-in safety net for premature gripper release during a task. When
enabled, the runner inspects every predicted action chunk: if the
Franka gripper width is in `(0.005, 0.075)` m (i.e. the jaws stopped on
an object) **and** a row would command gripper-open with a target XY
*outside* the task's `target_zone_xy` box, that row's gripper bit is
forced to "close". Other rows pass through untouched.

> **This biases the benchmark.** Zero-shot Table 6 numbers must be
> generated *without* this flag. Use it for debugging, demos, or
> sanity-checking the rest of the loop — not paper-replication runs.

To use:

1. Define the target box in `benchmarks/benchmark/droid_tasks.py` per
   task you want to guard:
   ```python
   DroidTask(
       task_id="apple_on_plate",
       ...
       target_zone_xy=(0.40, 0.50, -0.10, 0.10),  # (xmin, xmax, ymin, ymax) m
   ),
   ```
   Without `target_zone_xy` set, passing `--hold-until-target` for that
   task errors at trial start rather than guessing.

2. Pass the flag:
   ```bash
   python -m benchmarks.scripts.run_droid_benchmark \
       --transport rest --rest-step-time-s 2.5 \
       --tasks apple_on_plate --trials 1 \
       --hold-until-target
   ```

Each chunk that hits the guard logs a one-liner:
```
WARNING bench.droid hold-until-target: kept gripper closed on 3/8 row(s) (apple_on_plate/#0) — policy tried to release outside target zone
```

Use that log to judge how often the policy *would* have released
prematurely — it's the honest read on whether the loop is working.

### 5. Pick a transport and run

> **Always launch the benchmark from the repo root** (`~/erwin/mex5`).
> `python -m benchmarks.scripts.run_droid_benchmark` resolves the
> `benchmarks` package from your current directory — running it from
> `franka/cpp/build/` (where motion_server lives) errors with
> `ModuleNotFoundError: No module named 'benchmarks'`. Easiest pattern is
> **one terminal per process**: motion_server stays in `franka/cpp/build/`,
> the benchmark client stays in `~/erwin/mex5`.

#### 5a. FCI (default — direct libfranka)

Cleanup ritual — FCI is single-client, so anything else holding the robot
must die first:
```bash
# motion_server holds FCI exclusively if it's up
pkill -f motion_server 2>/dev/null
# previous run_droid_benchmark / dashboard / panda_py session
pkill -f run_droid_benchmark 2>/dev/null
pkill -f serve_dashboard 2>/dev/null
# confirm nothing's left
pgrep -af motion_server; pgrep -af run_droid_benchmark; pgrep -af serve_dashboard
# (each line should return empty)
```

Then run (from the repo root):
```bash
cd ~/erwin/mex5
export FRANKA_HOST=192.168.2.100           # robot FCI IP
# (FCI activation is done in Desk per step 3, not via env vars — the
# runner never reads FRANKA_USER/FRANKA_PASS.)

python -m benchmarks.scripts.run_droid_benchmark \
    --transport fci --tasks apple_on_plate --trials 1
```

> **FR3 only:** `panda_py` real-time control needs the **PREEMPT_RT
> kernel**. Check with `uname -r` — must show `…-rt…`. Stock kernels throw
> `RealtimeException`. If you're on a stock kernel, either reboot into the
> RT entry or use the REST transport instead.

**Tuning for accuracy + smoothness.** Defaults trade speed for accuracy:
chunks execute fully open-loop unless they're tagged "fine refinement"
(default threshold ~11° total joint travel), and per-row time is 100 ms.
On the lab rig that produces noticeable overshoot at chunk boundaries and
abrupt decel where the controller stops between chunks. Recommended FCI
settings:

```bash
python -m benchmarks.scripts.run_droid_benchmark \
    --transport fci --tasks apple_on_plate --trials 1 \
    --exec-rows 2 --fine-refinement-travel-rad 5.0 \
    --chunk-step-dt 0.2 --max-chunks 120
```

What each one does:

| Flag | Default | Recommended | Effect |
|---|---:|---:|---|
| `--exec-rows` | `3` | `2` | Number of rows to run before re-perceiving + re-inferring. Lower = more closed-loop, less overshoot. |
| `--fine-refinement-travel-rad` | `0.2` | `5.0` | Below this much total joint travel (rad), the chunk is treated as fine refinement and `--exec-rows` applies. Default only triggers on tiny refinement chunks; raise it so almost every non-grasp chunk runs receding-horizon. |
| `--chunk-step-dt` | `0.1` | `0.2` | Commanded duration per action row (sec). Doubling halves average joint velocity, which makes the substep ramp longer and softens chunk-end decel. |
| `--max-chunks` | task default (30) | `120` | Per-trial safety cap on action chunks. With `--exec-rows 2`, each chunk advances 4× less than the default 8 rows, so you need ~4× more chunks before the policy actually finishes the task. Without this, trials hit the cap mid-grasp and the operator gets prompted for success on an unfinished motion. |
| `--grasp-commit-grip-frac` | `0.5` | `0.5` | When ≥ this fraction of a chunk's rows command gripper-close, run the whole chunk uninterrupted (don't break a grasp mid-flight). Leave alone. |

Trade-off: ~3–4× more inference calls per trial → a few extra seconds end-to-end,
which is acceptable for accuracy. If chunk-boundary decel is still too sharp
after `--chunk-step-dt 0.2`, the deeper knobs (`max_joint_vel_rad_s`,
`substep_dt_s` inside `panda_driver.send_chunk`) are currently hardcoded —
exposing them as flags is a small follow-up.

#### 5b. REST (motion_server)

Cleanup ritual — motion_server's `initialize()` opens FCI itself, so any
prior FCI client (an earlier `run_droid_benchmark --transport fci`, or
the dashboard) must die first or `motion_server` will refuse with
`libfranka: Connection timeout`:
```bash
pkill -f run_droid_benchmark 2>/dev/null
pkill -f serve_dashboard 2>/dev/null
# and any stale motion_server itself
pkill -f motion_server 2>/dev/null
# confirm
pgrep -af motion_server; pgrep -af run_droid_benchmark; pgrep -af serve_dashboard
```

In a **separate terminal** (on the motion_server host — typically the same
box) start motion_server:
```bash
cd ~/erwin/mex5/franka/cpp
mkdir -p build && cd build && cmake .. && make
./motion_server                            # binds 0.0.0.0:34568; leave running
```

If `initialize()` fails with `cartesian_reflex` (common when the arm is in
pack pose), get the arm out of pack first — easiest is the **Unpack**
action in Desk; alternatively hand-guide it into a roughly extended pose
or run `python -c "import os, panda_py; panda_py.Panda(os.environ['FRANKA_HOST']).move_to_start()"`
in a separate (venv-activated) terminal, then re-run motion_server.

Back in the benchmark terminal (must be at the repo root, not in
`franka/cpp/build/` where you started motion_server):
```bash
cd ~/erwin/mex5
unset FRANKA_HOST                          # avoid transport confusion
export FRANKA_REST_HOST=192.168.2.1        # motion_server box, NOT the robot

# Two-phase approach speed is now ON by default (FAST=2.0 s, ZONE=0.20 m):
# race through free space at 2.0 s/row, then drop to the slow
# --rest-step-time-s the moment the commanded TCP Z is at or below 20 cm.
# These exports are only needed to override the defaults; set FAST equal to
# --rest-step-time-s (or push ZONE above the workspace) to fall back to
# single-speed. NOTE: the EE linear-velocity cap also defaults to 0.4 m/s
# (FRANKA_BENCH_REST_MAX_LIN_M_S) — that cap governs the long approach legs
# and is the dominant speedup lever; lower it if the traverse feels fast.
export FRANKA_BENCH_REST_FAST_STEP_TIME_S=2.0
export FRANKA_BENCH_REST_SLOW_ZONE_Z_M=0.20

python -m benchmarks.scripts.run_droid_benchmark \
    --transport rest --rest-step-time-s 2.5 \
    --tasks apple_on_plate --trials 1
```

**Tuning the two-phase loop** (REST/MCP only):

| Env var | Recommended | Effect |
|---|---:|---|
| `FRANKA_BENCH_REST_FAST_STEP_TIME_S` | `2.0` | Per-row motion time when target TCP Z is above the zone. Lower = faster traverse through free space, but very short values (≤ 1.0 s) on first-chunk-from-home moves trip `cartesian_motion_generator_joint_acceleration_discontinuity`. Server enforces a floor of 0.5 s. |
| `FRANKA_BENCH_REST_SLOW_ZONE_Z_M` | `0.20` | TCP Z (m, base frame) at/below which the slow `--rest-step-time-s` kicks in. Roughly "table height + 0.20". On airscan4 the table is at base Z ≈ 0, so 0.20 ≈ "slow for the last 20 cm of descent". |
| `--rest-step-time-s` | `2.5` | Per-row motion time inside the slow zone (and the only time used when either env var above is unset). Default 2.5 s. |

If you don't know where Z = 0 lands on your rig, read the current TCP Z
once after homing (with motion_server **stopped**, since it holds FCI):
```bash
python -c "import os, panda_py; print(panda_py.Panda(os.environ['FRANKA_HOST']).get_state().O_T_EE[14])"
```
Subtract roughly the table-top measurement from there. If the slow→fast
transition happens too late and you feel a jolt mid-descent, raise
`SLOW_ZONE_Z_M` to `0.25` or `0.30` to start slowing higher.

The REST driver also imports `panda_py` (for FK only — no FCI, no RT
kernel needed). If the FCI version-mismatch error blocked you earlier, REST
still works once `panda_py` is installed.

#### 5c. MCP (fastmcp → motion_server)

Needs **both** servers running. In two extra terminals:
```bash
# terminal A: motion_server (same as 5b)
cd ~/erwin/mex5/franka/cpp/build && ./motion_server

# terminal B: mcp_server
pip install fastmcp                        # one-time, on both boxes
cd ~/erwin/mex5/franka/python && python3 mcp_server.py
```

Benchmark terminal (at the repo root):
```bash
cd ~/erwin/mex5
export FRANKA_MCP_URL=http://<mcp host>:8085/franka

python -m benchmarks.scripts.run_droid_benchmark \
    --transport mcp --rest-step-time-s 2.5 \
    --tasks apple_on_plate --trials 1
```

### 6. (Optional) Watch the cameras while the benchmark runs

The runner doesn't open any windows — it just owns the cameras and
writes the latest captured frames to `/dev/shm` after every grab. A
companion script (`serve_live.py`) reads those files and serves them as
MJPEG over HTTP, so you can watch what the model is seeing from a
browser without touching the benchmark's RT control threads.

In a **separate terminal** (any time — before, during, or after the
benchmark starts):
```bash
cd ~/erwin/mex5
python -m benchmarks.scripts.serve_live --port 8080
```

Then open `http://<workstation-ip>:8080/` in a browser. Two MJPEG tiles:
external webcam + wrist RealSense (RGB). The page refreshes at ~10 fps;
if a tile stays blank, the benchmark hasn't grabbed a frame yet for that
camera (or the camera index/serial is wrong — see step 4).

This is also what the start-of-run log line is pointing you at:
```
INFO bench.droid live cam view: run python -m benchmarks.scripts.serve_live
in another shell, then open http://<workstation-ip>:8080/
```

`serve_live` is read-only; it can't drive the robot or change behavior.
For the interactive "type-an-instruction-and-run" experience, see the
separate **Dashboard** section above (`serve_dashboard.py`).

### 7. Compare

```bash
ls -lt benchmarks/results/ | head -5
python -m benchmarks.scripts.compare_runs benchmarks/results/*.json
```

The three latest files are FCI / REST / MCP; each column header tags its
transport (`fci:...`, `rest:...`, `mcp:...`). Motion overhead shows up in
`motion_rest_ms`.

### Common bites

- **Wrong host for REST/MCP.** `192.168.2.100` is FCI; `192.168.2.1` is what
  the repo's existing clients use for motion_server. The driver no longer
  falls back from `FRANKA_HOST` — wrong env → loud error with a hint.
- **Both endpoints up at once.** motion_server holds FCI exclusively; FCI
  runs will fail until you Ctrl-C motion_server.
- **`readJointState` empty / 8-vec missing.** You didn't rebuild
  motion_server from the patched source. The Python driver raises a clear
  "did you rebuild motion_server?" error pointing here.
- **MCP `readJointState` empty even after rebuild.** Means your installed
  `fastmcp` version returns a `call_tool` result shape `_unwrap` in
  `franka_mcp_driver.py` doesn't recognize. Run `franka/python/mcp_client.py`
  manually and paste the `response` repr — one extra branch fixes it.
- **REST per-row time clamps.** Server enforces `tf ∈ [0.5, ∞)`; below 0.5 s
  is silently floored. Above 5 s is fine.
- **Dashboard tile says `[mjpeg]` even after installing aiortc.** Two usual
  causes: (1) `aiortc` / `av` aren't visible to the venv the dashboard is
  running in — re-run `pip install aiortc av` inside the same venv and
  check the startup log for `WebRTC enabled`. (2) The browser couldn't
  reach the workstation directly (e.g. across a NAT) — WebRTC needs a
  routable path; on a flat LAN this is automatic, off-LAN you need an SSH
  tunnel or a STUN server. The page already falls back to MJPEG so the
  preview keeps working; check the browser console for `webrtc ... failed`
  to confirm.

The runner prompts you per-trial to set up the scene + randomize the external
camera, then asks for the success grade after each trial. Results stream to
`benchmarks/results/<run_id>.json` after every trial (crash-safe). At the end
it prints a side-by-side with the paper:

```
===== Table 6 comparison (MolmoAct2-DROID) =====
task                            paper     ours    n
apple_on_plate                  100.0%    93.3%   15
pipette_in_tray                  86.7%    66.7%   15
...
```

## Manual homing

The runner auto-homes at the start of every trial, so you don't normally need
these. Use them between sessions or when the arm is parked somewhere awkward
after a debug run.

**FCI** (motion_server must be stopped):

```bash
FRANKA_HOST=192.168.2.100 python -c "from benchmarks.benchmark.panda_driver import PandaDriver; PandaDriver().home()"
```

**REST** — either restart motion_server (its `initialize()` calls `goHome`,
joint-space, exactly the upstream behavior):

```bash
# Ctrl-C the running motion_server, then re-run it:
cd franka/cpp/build && ./motion_server
```

…or POST via the driver (faster; cartesian-interpolated to FK'd home):

```bash
FRANKA_REST_HOST=192.168.2.1 python -c "from benchmarks.benchmark.franka_rest_driver import FrankaRestDriver; FrankaRestDriver().home()"
```

**MCP** — same two options. Restart motion_server (MCP fans through it), or:

```bash
FRANKA_MCP_URL=http://<mcp host>:8085/franka python -c "from benchmarks.benchmark.franka_mcp_driver import FrankaMcpDriver; FrankaMcpDriver().home()"
```

## Layout

```
benchmarks/
  README.md                            this file
  requirements.txt                     workstation deps
  hpc/README.md                        HPC server setup (upstream host_server_droid.py)
  benchmark/
    dual_camera.py                     wrist (RealSense color) + one or two external (UVC) cams; dual cams stacked side-by-side
    dashboard_camera.py                wrist RealSense (color) + webcam, background-threaded
    panda_driver.py                    --transport=fci : panda_py joint-position streaming
    franka_rest_driver.py              --transport=rest: FK + motion_server REST
    franka_mcp_driver.py               --transport=mcp : fastmcp wrapper over the REST driver
    transport.py                       autodetect_transport() + make_driver(transport, ...)
    molmoact_droid_client.py           POST /act with json_numpy
    droid_tasks.py                     Table 6 task suite (5 tasks, paper success rates)
    droid_runner.py                    capture -> infer -> exec loop (transport-agnostic)
    metrics.py                         per-step record + p50/p95/p99 summaries
    molmoact_client.py                 (legacy /predict client; unused)
    franka_client.py                   delta-cartesian REST client (used by legacy runner.py)
    camera.py                          (legacy single-camera; unused for DROID path)
    tasks.py                           (legacy spatial/goal task suite for the legacy runner)
    runner.py                          (legacy real-robot runner using motion_server)
  scripts/
    run_droid_benchmark.py             PRIMARY: Table 6 zero-shot eval (--transport {fci,rest,mcp})
    serve_dashboard.py                 interactive web UI: streams + Home + task input
    serve_live.py                      passive MJPEG viewer for the CLI runner (reads /dev/shm)
    compare_runs.py                    side-by-side table across N result JSONs
    run_benchmark.py                   legacy
  results/                             per-run JSON dumps
```

## What the legacy files are

The first version of this harness assumed a custom inference server we'd write
(`benchmark/molmoact_client.py`) and a cartesian REST action interface
(`benchmark/franka_client.py`). After reading the MolmoAct2 model card and
Allen AI's repo, we pivoted to:
- upstream `host_server_droid.py` for serving (correct schema, json_numpy)
- `panda_py` for action exec (DROID uses joint targets, not cartesian)

The legacy modules are retained because they document the API of the
underlying C++ motion server and `franka_client.py` is still useful as a
thin REST client for ad-hoc moves.

The REST/MCP transports added later go through the same `motion_server` but
bridge the joint-vs-cartesian gap by FK-ing each model row locally before
sending the cartesian setpoint, so DROID's joint-target chunks can ride the
professor's REST interface without giving up the model's wrist orientation.
