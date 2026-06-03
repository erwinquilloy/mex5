# MolmoAct2 on Franka Emika — Benchmark

End-to-end harness for evaluating **MolmoAct2** on a Franka Research robot:

| Path | Model | Where it runs | Auto-scored? | Purpose |
|---|---|---|---|---|
| **DROID real-robot** (`scripts/run_droid_benchmark.py`) | `allenai/MolmoAct2-DROID` | upstream `host_server_droid.py` on HPC + tunnel + workstation client | no (human grader) | Table 6 reproduction of arXiv:2605.02881 (the course exercise target) |

## Topology (DROID path)

```
RealSense D457 (wrist) ──┐
USB webcam (external) ───┴► workstation ──► ssh -L 8000 ──► HPC A100
                              │                                 │
                              │                  uv run host_server_droid.py
                              │                                 │
                              ▼                                 │
                  one of three transports ──► Franka  ◄─────────┘  (actions[N,8])
                  ┌────────────────────────────────────────────────┐
                  │ fci  : panda_py.JointPosition (direct libfranka) │
                  │ rest : motion_server (cartesian xyz+ZYX deltas)  │
                  │ mcp  : fastmcp ──► motion_server (same as rest)  │
                  └────────────────────────────────────────────────┘
```

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

- **Single external camera**: Table 6's protocol randomizes the external camera pose
  each trial. We use whatever you mount as `FRANKA_BENCH_EXT_INDEX`. If no
  external is set, `dual_camera.py` falls back to duplicating the wrist image
  (substantially OOD for the model — expect a large gap from the paper numbers).
- **OOD objects**: bring objects that the model has not seen in DROID training.
  Common household items work; avoid anything that looks like the demo videos.
- **Controller**: paper uses the DROID NUC + polymetis stack; we use `panda_py`
  directly. Same action space, different timing characteristics.

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
- External USB webcam plugged in. Find its index with `ls /dev/video*` (use the
  number after `video`, e.g. `/dev/video0` → `0`).
- Franka in white/unlocked state, FCI activated (see `franka/python/basic.py`).

### 3. Run

Common env vars (cameras + model server):

```bash
export FRANKA_BENCH_EXT_INDEX=0          # /dev/video0
# optional: FRANKA_BENCH_WRIST_SERIAL=<D457 serial>  to pin the wrist cam
```

#### FCI (default — direct libfranka)

```bash
export FRANKA_HOST=192.168.1.131         # robot's FCI IP
export FRANKA_USER=...
export FRANKA_PASS=...

# full Table 6 (5 tasks x 15 trials, ~hours)
python -m benchmarks.scripts.run_droid_benchmark

# single task, one trial -- smoke test the whole loop end-to-end
python -m benchmarks.scripts.run_droid_benchmark --tasks apple_on_plate --trials 1
```

One-liner (smoke test):
```bash
FRANKA_HOST=192.168.1.131 FRANKA_USER=... FRANKA_PASS=... python -m benchmarks.scripts.run_droid_benchmark --tasks apple_on_plate --trials 1
```

#### REST (motion_server)

```bash
export FRANKA_REST_HOST=192.168.2.1      # the box running motion_server, NOT the robot
python -m benchmarks.scripts.run_droid_benchmark \
    --transport rest --rest-step-time-s 2.5 \
    --tasks apple_on_plate --trials 1
```

One-liner:
```bash
FRANKA_REST_HOST=192.168.2.1 python -m benchmarks.scripts.run_droid_benchmark --transport rest --rest-step-time-s 2.5 --tasks apple_on_plate --trials 1
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

For ad-hoc driving — type an instruction, see what the model does, watch the
streams — there's a Flask dashboard that replaces the CLI prompts:

- Three live MJPEG tiles: external webcam, wrist RealSense RGB, wrist
  RealSense **depth** (COLORMAP_JET, capped at 2 m by default).
- **Home** button → calls `driver.home()` on whichever transport is active.
- **Instruction input + Run** button → one inference→exec cycle per click
  (capture → MolmoAct2 → execute the returned chunk). Click again to keep
  going. No success grading, no result-file writes — use the CLI runner for
  that.
- Transport **dropdown**: auto-detected at startup from env vars (precedence
  `FRANKA_MCP_URL > FRANKA_REST_HOST > FRANKA_HOST`); the dropdown lets you
  switch without restarting (the underlying constraints still apply —
  motion_server must be stopped to switch to FCI, running to switch to
  REST/MCP).

### Launching the dashboard

The dashboard owns the cameras, so **stop `run_droid_benchmark.py` first** —
both can't open the RealSense at the same time. It also needs the same
upstream pieces as the CLI runner (model server + tunnel, plus motion_server
/ mcp_server for the REST and MCP transports respectively).

**Prerequisites checklist** (only what your chosen transport needs):

- [ ] `host_server_droid.py` running on the HPC, port 8000
- [ ] SSH tunnel up: `ssh -N -L 8000:localhost:8000 <hpc>` (and `curl http://localhost:8000/act` returns ok)
- [ ] Cameras connected (RealSense D457 wrist + USB webcam external)
- [ ] CLI runner stopped (`pkill -f run_droid_benchmark` if you're not sure)
- [ ] **FCI only:** robot in white/unlocked state, FCI activated, motion_server **stopped**
- [ ] **REST only:** motion_server running (see the rebuild + launch steps below)
- [ ] **MCP only:** motion_server running, **and** mcp_server.py running

Common env vars (cameras):

```bash
cd ~/mex5
export FRANKA_BENCH_EXT_INDEX=0
# optional: export FRANKA_BENCH_WRIST_SERIAL=<D457 serial>
# optional: export FRANKA_BENCH_DEPTH_MAX_M=2.0          # depth colormap range

# External-cam orientation. Canonical DROID is camera behind+above the robot;
# if your tripod is in front looking back at the robot, robot +Y appears as
# image-left, which the model reads as the opposite axis. Mirror it back:
# export FRANKA_BENCH_EXT_FLIP_H=1
# (X/forward-back is still OOD vs canonical -- no 2D transform fixes that.)
# Also available: FRANKA_BENCH_EXT_FLIP_V, FRANKA_BENCH_EXT_ROT_DEG ∈ {0,90,180,270}.
```

Then pick one of the three launches.

#### Launch (FCI)

```bash
export FRANKA_HOST=192.168.1.131
export FRANKA_USER=...
export FRANKA_PASS=...
python -m benchmarks.scripts.serve_dashboard \
    --port 8080 --molmoact-url http://localhost:8000
```

One-liner:
```bash
FRANKA_BENCH_EXT_INDEX=0 FRANKA_HOST=192.168.1.131 FRANKA_USER=... FRANKA_PASS=... python -m benchmarks.scripts.serve_dashboard --port 8080 --molmoact-url http://localhost:8000
```

#### Launch (REST)

```bash
# in another terminal on the motion_server host:
cd ~/mex5/franka/cpp/build && ./motion_server

# in the dashboard terminal:
export FRANKA_REST_HOST=192.168.2.1
python -m benchmarks.scripts.serve_dashboard \
    --port 8080 --molmoact-url http://localhost:8000 \
    --rest-step-time-s 2.5
```

One-liner:
```bash
FRANKA_BENCH_EXT_INDEX=0 FRANKA_REST_HOST=192.168.2.1 python -m benchmarks.scripts.serve_dashboard --port 8080 --molmoact-url http://localhost:8000 --rest-step-time-s 2.5
```

#### Launch (MCP)

```bash
# terminal A (motion_server host):
cd ~/mex5/franka/cpp/build && ./motion_server

# terminal B (mcp_server host):
cd ~/mex5/franka/python && python3 mcp_server.py

# terminal C (dashboard):
export FRANKA_MCP_URL=http://<mcp host>:8085/franka
python -m benchmarks.scripts.serve_dashboard \
    --port 8080 --molmoact-url http://localhost:8000 \
    --rest-step-time-s 2.5
```

One-liner:
```bash
FRANKA_BENCH_EXT_INDEX=0 FRANKA_MCP_URL=http://<mcp host>:8085/franka python -m benchmarks.scripts.serve_dashboard --port 8080 --molmoact-url http://localhost:8000 --rest-step-time-s 2.5
```

#### Using the dashboard

1. Open `http://<workstation-ip>:8080/` in a browser.
2. Confirm all three video tiles are live (external, wrist RGB, wrist
   depth). If the depth tile stays black, the RealSense depth stream
   isn't starting — check `FRANKA_BENCH_WRIST_SERIAL` and that the
   D457 USB-C switch is set per `franka/README.md`.
3. The status line at the bottom shows the auto-detected transport. To
   switch, pick a different option in the dropdown (the underlying
   constraints in the prerequisites still apply — switching to FCI while
   motion_server is running will fail loudly).
4. Click **home** to reset the arm.
5. Type an instruction (e.g. *"pick up the apple and put it on the plate"*),
   click **run one chunk**. The dashboard captures, runs inference, and
   executes the returned action chunk. Click again to keep going.

Stop the dashboard with Ctrl-C in its terminal; it closes the driver and
releases the cameras.

#### Useful flags

| Flag | Default | What it does |
|---|---|---|
| `--port` | `8080` | dashboard HTTP port |
| `--molmoact-url` | `http://localhost:8000` | MolmoAct2-DROID server |
| `--transport {fci,rest,mcp}` | autodetect | override env-var selection |
| `--rest-step-time-s` | `2.5` | per-row REST move time (REST/MCP only) |
| `--exec-rows` | `3` | rows of each chunk to run when the policy is in fine-refinement mode |
| `--mjpeg-fps` | `10.0` | dashboard refresh rate |

### Dashboard vs CLI runner

| Need | Use |
|---|---|
| Quick "type and try" with one instruction | dashboard |
| Watch depth in real time | dashboard (CLI runner doesn't stream depth) |
| Reproduce Table 6 with N trials, per-task scoring, results JSON | CLI runner |
| Auto-home between trials with operator-graded success | CLI runner |

The two share the same drivers (`PandaDriver` / `FrankaRestDriver` /
`FrankaMcpDriver`) and the same `DroidClient`, so model+transport behavior is
identical — the dashboard is just an interactive shell around them.

## End-to-end walkthrough (Linux workstation, fresh shell)

Step-by-step recipe to bring everything up and run all three transports back
to back, then compare. Use this as a checklist when you sit down at the rig.

### 0. Pull latest

```bash
cd ~/mex5
git pull --ff-only origin main      # should land at 3c6f40e or later
```

### 1. Rebuild motion_server (REST/MCP only — skip if FCI-only)

On the host that runs motion_server (the one with libfranka). If it's a
different box from the workstation, SSH in first.

```bash
cd ~/mex5/franka/cpp
mkdir -p build && cd build
cmake ..
make
./motion_server     # binds 0.0.0.0:34568, prompts before homing
```

One-liner:
```bash
cd ~/mex5/franka/cpp && mkdir -p build && cd build && cmake .. && make && ./motion_server
```

Note the IP this box has on the lab network — that's `FRANKA_REST_HOST` below.

### 2. Install fastmcp (MCP only)

On the workstation (benchmark client) **and** on the host where you'll run
`mcp_server.py`:

```bash
pip install fastmcp
```

### 3. Model server + tunnel (all transports)

```bash
# HPC side (tmux / screen):
cd ~/molmoact2 && uv run python examples/droid/host_server_droid.py \
    --host 0.0.0.0 --port 8000 --dtype bfloat16

# workstation:
ssh -N -L 8000:localhost:8000 erwin.quilloy@ai-n002.hpc.coe.upd.edu.ph &
curl http://localhost:8000/act      # expect {"status": "ok", "repo_id": "...", ...}
```

### 4. Smoke-test each transport (one task, one trial each)

Common env vars first:

```bash
cd ~/mex5
# activate whichever conda/venv has the workstation deps installed
export FRANKA_BENCH_EXT_INDEX=0
# optional: export FRANKA_BENCH_WRIST_SERIAL=<D457 serial>
```

**4a. FCI** — motion_server must be **stopped** (it would hold FCI exclusively):

```bash
# Ctrl-C motion_server in its terminal first
export FRANKA_HOST=192.168.1.131            # robot FCI IP
export FRANKA_USER=...
export FRANKA_PASS=...
python -m benchmarks.scripts.run_droid_benchmark \
    --transport fci --tasks apple_on_plate --trials 1
```

One-liner:
```bash
FRANKA_HOST=192.168.1.131 FRANKA_USER=... FRANKA_PASS=... python -m benchmarks.scripts.run_droid_benchmark --transport fci --tasks apple_on_plate --trials 1
```

**4b. REST** — start motion_server (step 1) in another terminal first:

```bash
unset FRANKA_HOST                           # avoid confusion
export FRANKA_REST_HOST=<motion_server IP>  # repo default in clients: 192.168.2.1
python -m benchmarks.scripts.run_droid_benchmark \
    --transport rest --rest-step-time-s 2.5 \
    --tasks apple_on_plate --trials 1
```

One-liner:
```bash
unset FRANKA_HOST; FRANKA_REST_HOST=<motion_server IP> python -m benchmarks.scripts.run_droid_benchmark --transport rest --rest-step-time-s 2.5 --tasks apple_on_plate --trials 1
```

**4c. MCP** — motion_server **and** mcp_server both running:

```bash
# third terminal:
cd ~/mex5/franka/python
python3 mcp_server.py                       # 0.0.0.0:8085/franka

# back in the benchmark terminal:
export FRANKA_MCP_URL=http://<mcp host>:8085/franka
python -m benchmarks.scripts.run_droid_benchmark \
    --transport mcp --rest-step-time-s 2.5 \
    --tasks apple_on_plate --trials 1
```

One-liners:
```bash
# third terminal (mcp_server):
cd ~/mex5/franka/python && python3 mcp_server.py
# benchmark terminal:
FRANKA_MCP_URL=http://<mcp host>:8085/franka python -m benchmarks.scripts.run_droid_benchmark --transport mcp --rest-step-time-s 2.5 --tasks apple_on_plate --trials 1
```

### 5. Compare

```bash
ls -lt benchmarks/results/ | head -5
python -m benchmarks.scripts.compare_runs benchmarks/results/*.json
```

The three latest files are FCI / REST / MCP; each column header tags its
transport (`fci:...`, `rest:...`, `mcp:...`). Motion overhead shows up in
`motion_rest_ms`.

### Common bites

- **Wrong host for REST/MCP.** `192.168.1.131` is FCI; `192.168.2.1` is what
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
FRANKA_HOST=192.168.1.131 python -c "from benchmarks.benchmark.panda_driver import PandaDriver; PandaDriver().home()"
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
    dual_camera.py                     wrist (RealSense color) + external (UVC) capture
    dashboard_camera.py                wrist RealSense (color + depth) + webcam, background-threaded
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
    serve_live.py                      legacy passive MJPEG viewer (reads tmpfs)
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
