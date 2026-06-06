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
- External USB webcam plugged in. The RealSense also registers as
  `/dev/video*`, so a bare `ls /dev/video*` is misleading. Use
  `v4l2-ctl --list-devices` and pick the index that's *not* under
  "Intel RealSense" (on `airscan4` this is currently index `6`).
- Franka in white/unlocked state, FCI activated (see `franka/python/basic.py`).

### 3. Run

Common env vars (cameras + model server):

```bash
export FRANKA_BENCH_EXT_INDEX=6          # USB webcam (NOT the RealSense node)
# RealSense D455 has no 256x256 mode; use a native resolution
export FRANKA_BENCH_CAM_W=640
export FRANKA_BENCH_CAM_H=480
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
export FRANKA_USER=...
export FRANKA_PASS=...

# full Table 6 (5 tasks x 15 trials, ~hours)
python -m benchmarks.scripts.run_droid_benchmark

# single task, one trial -- smoke test the whole loop end-to-end
python -m benchmarks.scripts.run_droid_benchmark --tasks apple_on_plate --trials 1
```

One-liner (smoke test):
```bash
FRANKA_HOST=192.168.2.100 FRANKA_USER=... FRANKA_PASS=... python -m benchmarks.scripts.run_droid_benchmark --tasks apple_on_plate --trials 1
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

For ad-hoc driving — type an instruction, see what the model does, watch the
streams — there's a Flask dashboard that replaces the CLI prompts:

- Two live MJPEG tiles: external webcam, wrist RealSense RGB. (The
  model is RGB-only; depth was removed since it was display-only and
  unused by inference.)
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
export FRANKA_HOST=192.168.2.100
export FRANKA_USER=...
export FRANKA_PASS=...
python -m benchmarks.scripts.serve_dashboard \
    --port 8080 --molmoact-url http://localhost:8000
```

One-liner:
```bash
FRANKA_BENCH_EXT_INDEX=0 FRANKA_HOST=192.168.2.100 FRANKA_USER=... FRANKA_PASS=... python -m benchmarks.scripts.serve_dashboard --port 8080 --molmoact-url http://localhost:8000
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
2. Confirm both video tiles are live (external, wrist RGB). If wrist RGB
   stays black, check `FRANKA_BENCH_WRIST_SERIAL` and that the D457
   USB-C switch is set per `franka/README.md`.
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
| Live camera streams in a browser | dashboard |
| Reproduce Table 6 with N trials, per-task scoring, results JSON | CLI runner |
| Auto-home between trials with operator-graded success | CLI runner |

The two share the same drivers (`PandaDriver` / `FrankaRestDriver` /
`FrankaMcpDriver`) and the same `DroidClient`, so model+transport behavior is
identical — the dashboard is just an interactive shell around them.

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
export FRANKA_BENCH_EXT_INDEX=6            # USB webcam (NOT the RealSense /dev/video* node)
export FRANKA_BENCH_CAM_W=640              # RealSense D455 has no 256x256 mode
export FRANKA_BENCH_CAM_H=480

# Required on the lab rig (airscan4): the wrist RealSense sits ~8 cm forward
# of the TCP and ~5 cm above the grasp plane, and the external tripod faces
# the robot (DROID canonical is behind+above, so robot +Y maps to image-left
# — flip to compensate). DZ is negative because the code adds the offset to
# the commanded terminal Z, and we need TCP to descend further to grasp.
export FRANKA_BENCH_REST_CAM_DX_M=0.08     # wrist-cam → TCP X offset (REST/MCP)
export FRANKA_BENCH_REST_CAM_DZ_M=-0.05    # wrist-cam → TCP Z offset (REST/MCP)
export FRANKA_BENCH_FCI_CAM_DX_M=0.08      # same for FCI (terminal-pose only)
export FRANKA_BENCH_FCI_CAM_DZ_M=-0.05     # same for FCI
export FRANKA_BENCH_EXT_FLIP_H=1           # mirror external view back to canonical

# optional: FRANKA_BENCH_WRIST_SERIAL=<D457 serial> to pin the wrist cam
# optional: FRANKA_BENCH_EXT_FLIP_V=1 / FRANKA_BENCH_EXT_ROT_DEG ∈ {0,90,180,270}
```

Confirm the external index with `v4l2-ctl --list-devices` and pick the one
**not** under "Intel RealSense".

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
export FRANKA_USER=<real desk username>
export FRANKA_PASS=<real desk password>

python -m benchmarks.scripts.run_droid_benchmark \
    --transport fci --tasks apple_on_plate --trials 1
```

> **FR3 only:** `panda_py` real-time control needs the **PREEMPT_RT
> kernel**. Check with `uname -r` — must show `…-rt…`. Stock kernels throw
> `RealtimeException`. If you're on a stock kernel, either reboot into the
> RT entry or use the REST transport instead.

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

python -m benchmarks.scripts.run_droid_benchmark \
    --transport rest --rest-step-time-s 2.5 \
    --tasks apple_on_plate --trials 1
```

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

### 6. Compare

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
    dual_camera.py                     wrist (RealSense color) + external (UVC) capture
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
