# MolmoAct2 on Franka Emika — Benchmark

End-to-end harness for evaluating **MolmoAct2** (served on an A100 HPC node)
driving a **Franka Research** robot through the REST motion server from
[`roatienza/autonomous-robots/franka`](https://github.com/roatienza/autonomous-robots/tree/main/franka).

## Metrics collected

- **Task success rate** — LIBERO-style subset (4 spatial + 2 goal tasks), human-scored per trial.
- **Inference latency** — GPU/processor time reported by the server (`infer_server_ms`) and full HTTP RTT (`infer_rtt_ms`).
- **End-to-end control latency** — camera grab → MolmoAct2 → Franka REST return (`e2e_ms`).
- Per-step record of camera/infer/motion timings dumped as JSON in `results/`.

## Topology

```
RealSense D457 ──► workstation ──► ssh -L 8000 ──► HPC A100
                       │                              │
                       │                              serve_molmoact2.py
                       │                              (FastAPI + transformers)
                       ▼
              franka motion_server (192.168.2.1:34568)
                       │
                       ▼
                 Franka Emika
```

## Setup

**On the HPC node (`ai-n002.hpc.coe.upd.edu.ph`):**

```bash
conda create -n molmoact2 python=3.10 -y && conda activate molmoact2
pip install -r benchmarks/hpc/requirements_hpc.txt
# pick the actual HF id of MolmoAct2 (defaults to allenai/MolmoAct-2-7B)
MOLMOACT2_MODEL=allenai/MolmoAct-2-7B python benchmarks/hpc/serve_molmoact2.py --port 8000
```

**On the workstation:**

```bash
pip install -r benchmarks/requirements.txt
# tunnel the server
bash benchmarks/hpc/launch_server.sh tunnel
# in another shell -- start the Franka motion server per franka/README.md, then:
python -m benchmarks.scripts.run_benchmark --trials 5
```

## Action contract

MolmoAct2 returns a 7-vector per step: `[dx, dy, dz, droll, dpitch, dyaw, grip]`
in **meters / degrees**, where `grip > 0.5` ⇒ close. The Franka REST API
already accepts cartesian targets + delta-degree rotations (see
`franka/README.md`), so `FrankaClient.apply_delta` adds the position deltas to
the current pose, forwards rotation deltas verbatim, and toggles the gripper
before commanding the motion.

If the upstream model uses a different convention, override
`FrankaClient.apply_delta` rather than changing the network code.

## Files

| Path | Purpose |
|---|---|
| `hpc/serve_molmoact2.py` | FastAPI predict server (runs on A100) |
| `hpc/launch_server.sh` | SSH tunnel + remote launch helpers |
| `benchmark/franka_client.py` | REST wrapper with timing hooks |
| `benchmark/molmoact_client.py` | HTTP client → MolmoAct2 server |
| `benchmark/camera.py` | RealSense D457 + file-replay fallback |
| `benchmark/libero_env.py` | LIBERO sim env wrapper (robosuite/MuJoCo) |
| `benchmark/tasks.py` | Real-robot task definitions (LIBERO-style) |
| `benchmark/metrics.py` | Latency timers, record dataclasses, summary |
| `benchmark/runner.py` | Real-robot capture→infer→execute loop |
| `benchmark/sim_runner.py` | LIBERO sim infer→step loop |
| `scripts/run_benchmark.py` | CLI: real-robot run |
| `scripts/run_libero_benchmark.py` | CLI: LIBERO sim run |
| `results/` | Per-run JSON dumps |

## Sim path (LIBERO)

LIBERO ships task suites (`libero_spatial`, `libero_object`, `libero_goal`,
`libero_10`, `libero_90`) with a Franka Panda in robosuite/MuJoCo, language
instructions per task, and built-in `check_success()` — so we get automated
scoring without the real robot or a human grader.

### Install

```bash
# 1. base benchmark deps
pip install -r benchmarks/requirements.txt

# 2. sim deps + LIBERO from source (the package isn't on PyPI)
pip install -r benchmarks/requirements_sim.txt
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/LIBERO
pip install -e ~/LIBERO
# LIBERO downloads BDDL/task assets on first env construction;
# headless mujoco needs MUJOCO_GL=egl (Linux) or osmesa.
export MUJOCO_GL=egl
```

### Run

```bash
# list tasks in a suite
python -m benchmarks.scripts.run_libero_benchmark --list libero_spatial

# full libero_spatial suite (10 tasks), 5 trials each, MolmoAct2 over tunnel
python -m benchmarks.scripts.run_libero_benchmark \
    --suite libero_spatial --trials 5 \
    --molmoact-url http://localhost:8000

# single task, larger action chunk
python -m benchmarks.scripts.run_libero_benchmark \
    --suite libero_goal --tasks 0 --trials 3 --n-action-chunk 4
```

### Action scaling note

LIBERO's OSC_POSE controller expects actions in `[-1, 1]`. MolmoAct2 may emit
raw meters/degrees + binary grip. Tune `--action-scale` so the first six dims
land in that range; the gripper is remapped automatically (`>0.5` ⇒ close).
If the model is already trained on LIBERO/OXE-normalized actions, leave
`--action-scale 1.0`.

## Dry-run (no robot, no GPU)

```bash
# point at any folder of jpegs; success prompt suppressed
FRANKA_BENCH_IMAGES=./fixtures \
  python -m benchmarks.scripts.run_benchmark \
    --tasks spatial_red_left --trials 1 \
    --molmoact-url http://localhost:8000 \
    --no-interactive
```

This still requires a reachable MolmoAct2 endpoint (mock it with a FastAPI
stub that returns zeros if you just want to exercise the loop).

## Tunables worth sweeping

- `--n-action-chunk` — actions returned per inference; trades fewer GPU calls for more open-loop drift.
- `--step-time` — commanded motion duration per cartesian step (REST `t` field).
- Image resolution in `camera.RealsenseCamera(...)` — affects both inference latency and policy quality.
