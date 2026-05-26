# MolmoAct2 on Franka Emika — Benchmark

End-to-end harness for evaluating **MolmoAct2** on a Franka Research robot.
Two evaluation paths share a common metrics schema (`benchmark/metrics.py`):

| Path | Model | Where it runs | Auto-scored? | Purpose |
|---|---|---|---|---|
| **LIBERO sim** (`scripts/run_libero_benchmark.py`) | `allenai/MolmoAct2-LIBERO` | one Python process on the HPC GPU (in-process, no HTTP) | yes (`env.check_success()`) | **Primary**: get suite-wide success numbers fast, no robot |
| **DROID real-robot** (`scripts/run_droid_benchmark.py`) | `allenai/MolmoAct2-DROID` | upstream `host_server_droid.py` on HPC + tunnel + workstation client | no (human grader) | Table 6 reproduction once sim is validated |

Current focus: **LIBERO sim first**, real robot later.

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
cd ~/mex5

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
                       panda_py ──► libfranka ──► Franka  ◄─────┘  (actions[N,8])
```

Action contract (from the model card): `actions[N, 8]` = `[q1..q7, gripper]`,
**absolute joint targets in radians + gripper command**. Gripper ≥ 0.5 ⇒ close.

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

```bash
export FRANKA_HOST=192.168.1.131
export FRANKA_USER=...
export FRANKA_PASS=...
export FRANKA_BENCH_EXT_INDEX=0          # /dev/video0
# optional: FRANKA_BENCH_WRIST_SERIAL=<D457 serial>  to pin the wrist cam

# full Table 6 (5 tasks x 15 trials, ~hours)
python -m benchmarks.scripts.run_droid_benchmark

# single task, one trial -- smoke test the whole loop end-to-end
python -m benchmarks.scripts.run_droid_benchmark --tasks apple_on_plate --trials 1
```

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
    panda_driver.py                    panda_py wrapper: state -> 8-vec, send action chunk
    molmoact_droid_client.py           POST /act with json_numpy
    droid_tasks.py                     Table 6 task suite (5 tasks, paper success rates)
    droid_runner.py                    capture -> infer -> joint exec loop
    metrics.py                         per-step record + p50/p95/p99 summaries
    libero_env.py                      LIBERO sim env (smoke-test path)
    sim_runner.py                      LIBERO loop
    molmoact_client.py                 (legacy /predict client; LIBERO path only)
    franka_client.py                   (legacy REST cartesian client; unused for DROID path)
    camera.py                          (legacy single-camera; unused for DROID path)
    tasks.py                           (legacy LIBERO-style real-robot tasks)
    runner.py                          (legacy real-robot runner using motion_server)
  scripts/
    run_droid_benchmark.py             PRIMARY: Table 6 zero-shot eval
    run_libero_benchmark.py            sim smoke test
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
