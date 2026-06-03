# MolmoAct2 on Franka Emika — Benchmark

End-to-end harness for evaluating **MolmoAct2** on a Franka Research robot.
Two evaluation paths share a common metrics schema (`benchmark/metrics.py`):

| Path | Model | Where it runs | Auto-scored? | Purpose |
|---|---|---|---|---|
| **DROID real-robot** (`scripts/run_droid_benchmark.py`) | `allenai/MolmoAct2-DROID` | upstream `host_server_droid.py` on HPC + tunnel + workstation client | no (human grader) | **Primary**: Table 6 reproduction of arXiv:2605.02881 (the course exercise target) |
| **LIBERO sim** (`scripts/run_libero_benchmark.py`) | `allenai/MolmoAct2-LIBERO` | one Python process on the HPC GPU (in-process, no HTTP) | yes (`env.check_success()`) | Smoke test / debug — fast suite-wide success numbers without a robot |

Current focus: **DROID real-robot** for the Table 6 zero-shot replication. The
LIBERO sim path remains useful for fast smoke testing of the model + harness.

## Sim path (LIBERO + MolmoAct2-LIBERO)

```
HPC A100 (one Python process)
  ├── allenai/MolmoAct2-LIBERO  (loaded in-process via transformers)
  └── robosuite/MuJoCo Franka   (LIBERO env, agentview + wrist cameras)
        ↑                          │
        └── env.step(action[7]) ◄──┘  predict_action -> actions[N,7]
```

### HPC setup

```bash
# clone + LIBERO + sim deps + model deps -- all in one env
cd ~
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
git clone https://github.com/erwinquilloy/mex5.git

# use the upstream MolmoAct2 venv (uv-managed, has the right transformers pin)
# OR: a fresh conda env -- avoid mixing with the broken `cached_path` env from earlier
conda create -n molmoact2-libero python=3.10 -y && conda activate molmoact2-libero
pip install -U "transformers>=4.46" "huggingface_hub>=0.24" "accelerate>=0.34" torch
pip install -r ~/mex5/benchmarks/requirements.txt
pip install -r ~/mex5/benchmarks/requirements_sim.txt
pip install -e ~/LIBERO

export MUJOCO_GL=egl
export HF_HOME=$HOME/hf-cache
mkdir -p $HF_HOME
```

### Run on the HPC GPU

```bash
conda activate molmoact2-libero        # required -- (base) is missing the deps
cd ~/mex5
export CUDA_VISIBLE_DEVICES=3          # pick a free GPU on the node (nvidia-smi to check)

# list tasks in a suite (no model load -- fast)
python -m benchmarks.scripts.run_libero_benchmark --list libero_spatial

# smoke test: one task, one trial -- validates model + env wire end-to-end
python -m benchmarks.scripts.run_libero_benchmark \
    --suite libero_spatial --tasks 0 --trials 1

# full libero_spatial suite (10 tasks x 5 trials)
python -m benchmarks.scripts.run_libero_benchmark --suite libero_spatial --trials 5

# all four "main" LIBERO suites, 5 trials each
for s in libero_spatial libero_object libero_goal libero_10; do
  python -m benchmarks.scripts.run_libero_benchmark --suite $s --trials 5
done
```

Common stumbles:
- `ModuleNotFoundError: No module named 'benchmarks'` → you're not in `~/mex5`, or you forgot `conda activate molmoact2-libero`.
- Crash inside `LiberoEnv.reset()` about `task_id=` kwarg, or `np.asarray(...)` on a CUDA tensor → pull latest; both fixed in `54533a3`.

Results stream into `benchmarks/results/<run_id>.json` after every trial. The
final stdout block prints overall success rate, per-task success rate, and
inference / e2e latency p50/p95/p99 — directly comparable to the DROID real-robot
numbers later.

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

#### REST (motion_server)

```bash
export FRANKA_REST_HOST=192.168.2.1      # the box running motion_server, NOT the robot
python -m benchmarks.scripts.run_droid_benchmark \
    --transport rest --rest-step-time-s 2.5 \
    --tasks apple_on_plate --trials 1
```

#### MCP (fastmcp → motion_server)

```bash
export FRANKA_MCP_URL=http://<mcp-host>:8085/franka
python -m benchmarks.scripts.run_droid_benchmark \
    --transport mcp --rest-step-time-s 2.5 \
    --tasks apple_on_plate --trials 1
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

Prints overall success, per-task success (cell shows `pct (n_trials)`), and
mean/p50/p95 for `infer_server_ms` / `e2e_ms` / `motion_rest_ms` per column.
The motion_rest jump from FCI to REST/MCP is where the transport overhead lives.

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
conda activate molmoact2-libero     # or whichever env has the workstation deps
cd ~/mex5
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

**4b. REST** — start motion_server (step 1) in another terminal first:

```bash
unset FRANKA_HOST                           # avoid confusion
export FRANKA_REST_HOST=<motion_server IP>  # repo default in clients: 192.168.2.1
python -m benchmarks.scripts.run_droid_benchmark \
    --transport rest --rest-step-time-s 2.5 \
    --tasks apple_on_plate --trials 1
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

## Layout

```
benchmarks/
  README.md                            this file
  requirements.txt                     workstation deps
  hpc/README.md                        HPC server setup (upstream host_server_droid.py)
  benchmark/
    dual_camera.py                     wrist (RealSense) + external (UVC) capture
    panda_driver.py                    --transport=fci : panda_py joint-position streaming
    franka_rest_driver.py              --transport=rest: FK + motion_server REST
    franka_mcp_driver.py               --transport=mcp : fastmcp wrapper over the REST driver
    molmoact_droid_client.py           POST /act with json_numpy
    droid_tasks.py                     Table 6 task suite (5 tasks, paper success rates)
    droid_runner.py                    capture -> infer -> exec loop (transport-agnostic)
    metrics.py                         per-step record + p50/p95/p99 summaries
    libero_env.py                      LIBERO sim env (smoke-test path)
    sim_runner.py                      LIBERO loop
    molmoact_client.py                 (legacy /predict client; LIBERO path only)
    franka_client.py                   delta-cartesian REST client (used by legacy runner.py)
    camera.py                          (legacy single-camera; unused for DROID path)
    tasks.py                           (legacy LIBERO-style real-robot tasks)
    runner.py                          (legacy real-robot runner using motion_server)
  scripts/
    run_droid_benchmark.py             PRIMARY: Table 6 zero-shot eval (--transport {fci,rest,mcp})
    run_libero_benchmark.py            sim smoke test
    compare_runs.py                    side-by-side table across N result JSONs
    run_benchmark.py                   legacy
  results/                             per-run JSON dumps
```

## What the legacy files are

The first version of this harness assumed a custom inference server we'd write
(`benchmark/molmoact_client.py`, `serve_molmoact2.py`) and a cartesian REST
action interface (`benchmark/franka_client.py`). After reading the MolmoAct2
model card and Allen AI's repo, we pivoted to:
- upstream `host_server_droid.py` for serving (correct schema, json_numpy)
- `panda_py` for action exec (DROID uses joint targets, not cartesian)

The legacy modules are retained because the LIBERO sim path may still use them
in future work, and they document the API of the underlying C++ motion server.

The REST/MCP transports added later go through the same `motion_server` but
bridge the joint-vs-cartesian gap by FK-ing each model row locally before
sending the cartesian setpoint, so DROID's joint-target chunks can ride the
professor's REST interface without giving up the model's wrist orientation.
