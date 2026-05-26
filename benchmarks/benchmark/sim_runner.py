"""LIBERO sim loop: MolmoAct2 -> action -> env.step -> auto success check."""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

import numpy as np

from .libero_env import LiberoEnv, SUITES, list_suite_tasks
from .metrics import RunRecord, StepRecord, Stopwatch, TrialRecord
from .molmoact_client import MolmoActClient

log = logging.getLogger("bench.sim")


def _scale_action(a, action_scale: float, clip: float) -> np.ndarray:
    """LIBERO/robosuite OSC_POSE action lives in [-1, 1]; MolmoAct2 may emit
    raw meters/degrees + gripper {0,1}. We scale the first 6 dims and remap
    the gripper. Tune `action_scale` per model. Set to 1.0 if model already
    outputs in [-1, 1]."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if a.shape[0] != 7:
        raise ValueError(f"expected 7-dim action, got {a.shape}")
    out = np.empty(7, dtype=np.float32)
    out[:6] = np.clip(a[:6] * action_scale, -clip, clip)
    out[6] = 1.0 if a[6] > 0.5 else -1.0   # robosuite: +1=close, -1=open
    return out


def run_trial(
    env: LiberoEnv,
    trial: int,
    init_index: int,
    molmo: MolmoActClient,
    max_steps: int,
    n_action_chunk: int,
    action_scale: float,
    action_clip: float,
    seed: Optional[int] = None,
) -> TrialRecord:
    molmo.reset()
    img = env.reset(init_index=init_index, seed=seed)
    rec = TrialRecord(task_id=env.spec.task_id, trial=trial, success=False, n_steps=0, wallclock_s=0.0)
    t_start = time.perf_counter()
    cam_sw, step_sw = Stopwatch(), Stopwatch()
    success = False

    for _ in range(max_steps):
        e2e_t0 = time.perf_counter()
        with cam_sw():
            _ = img  # already in hand from previous step/reset; "grab" is just the reference
        pred = molmo.predict(img, env.spec.instruction, state=env.proprio(),
                             n_actions=n_action_chunk)
        for a in pred.actions:
            a_env = _scale_action(a, action_scale, action_clip)
            with step_sw():
                img, _reward, done, _info = env.step(a_env)
            success = env.success()
            e2e_ms = (time.perf_counter() - e2e_t0) * 1000.0
            rec.steps.append(StepRecord(
                step=len(rec.steps),
                camera_ms=cam_sw.ms,
                infer_server_ms=pred.server_dt_ms,
                infer_rtt_ms=pred.rtt_ms,
                motion_rest_ms=step_sw.ms,    # sim "rest" time = env.step wallclock
                motion_cmd_ms=0.0,            # no commanded duration in sim
                e2e_ms=e2e_ms,
                action=list(map(float, a)),
                instruction=env.spec.instruction,
            ))
            if success or done:
                break
            e2e_t0 = time.perf_counter()
        if success or done:
            break

    rec.n_steps = len(rec.steps)
    rec.wallclock_s = time.perf_counter() - t_start
    rec.success = bool(success)
    return rec


def run_libero_benchmark(
    suite: str = "libero_spatial",
    task_indices: Optional[list[int]] = None,
    trials_per_task: int = 5,
    molmoact_url: str = "http://localhost:8000",
    n_action_chunk: int = 1,
    max_steps: int = 300,
    action_scale: float = 1.0,
    action_clip: float = 1.0,
    camera_height: int = 256,
    camera_width: int = 256,
    seed: int = 0,
    results_dir: str = "benchmarks/results",
) -> RunRecord:
    if suite not in SUITES:
        raise ValueError(f"suite must be one of {SUITES}")

    molmo = MolmoActClient(molmoact_url)
    health = molmo.health()
    log.info("molmoact health: %s", health)

    if task_indices is None:
        task_indices = [i for i, _ in list_suite_tasks(suite)]

    run = RunRecord(
        run_id=f"libero-{suite}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        started_at=time.time(),
        model_id=health.get("model_id", "?"),
        franka_endpoint=f"sim:{suite}",
        molmoact_endpoint=molmoact_url,
    )

    for task_index in task_indices:
        env = LiberoEnv(suite=suite, task_index=task_index,
                        camera_height=camera_height, camera_width=camera_width)
        try:
            print(f"\n=== {env.spec.task_id} ===\ninstruction: {env.spec.instruction!r}")
            for trial in range(trials_per_task):
                # Cycle through LIBERO's pre-canned init states for variation.
                init_idx = trial % env.spec.n_init_states
                tr = run_trial(
                    env, trial, init_idx, molmo,
                    max_steps=max_steps,
                    n_action_chunk=n_action_chunk,
                    action_scale=action_scale,
                    action_clip=action_clip,
                    seed=seed + trial,
                )
                run.trials.append(tr)
                run.dump(results_dir)
                log.info("trial done: %s/#%d success=%s steps=%d",
                         tr.task_id, tr.trial, tr.success, tr.n_steps)
        finally:
            env.close()
    return run
