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
python -m benchmarks.scripts.serve_dashboard \
    --port 8080 --molmoact-url http://localhost:8000
```

One-liner:
```bash
FRANKA_BENCH_EXT_INDEX=0 FRANKA_HOST=192.168.2.100 python -m benchmarks.scripts.serve_dashboard --port 8080 --molmoact-url http://localhost:8000
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
# These DX/DZ values fire on the grasp chunk's terminal row only (the
# default OFFSET_MODE=grasp_terminal). The 'always' mode is documented
# below as an opt-in for whole-trajectory perception bias; not recommended
# on FCI (orientation drift from per-row IK branch flips) and only on REST
# with the two-phase fast time DISABLED.
export FRANKA_BENCH_REST_CAM_DX_M=0.08     # wrist-cam → TCP X offset (REST/MCP)
export FRANKA_BENCH_REST_CAM_DZ_M=-0.05    # wrist-cam → TCP Z offset (REST/MCP)
export FRANKA_BENCH_FCI_CAM_DX_M=0.08      # same for FCI (terminal-pose only)
export FRANKA_BENCH_FCI_CAM_DZ_M=-0.05     # same for FCI
export FRANKA_BENCH_EXT_FLIP_H=1           # mirror external view back to canonical

# optional: FRANKA_BENCH_WRIST_SERIAL=<D457 serial> to pin the wrist cam
# optional: FRANKA_BENCH_EXT_FLIP_V=1 / FRANKA_BENCH_EXT_ROT_DEG ∈ {0,90,180,270}
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

Confirm the external index with `v4l2-ctl --list-devices` and pick the one
**not** under "Intel RealSense".

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

# Two-phase approach speed (recommended on this rig): race through free
# space at 2.0 s/row, then drop to the slow --rest-step-time-s the moment
# the commanded TCP Z is at or below 20 cm. Comment out either var to fall
# back to single-speed.
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
