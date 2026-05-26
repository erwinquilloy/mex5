"""LIBERO sim loop with in-process MolmoAct2-LIBERO."""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from .libero_env import LiberoEnv, LiberoObs, SUITES, list_suite_tasks
from .metrics import RunRecord, StepRecord, Stopwatch, TrialRecord
from .molmoact_libero_local import MolmoActLiberoLocal

log = logging.getLogger("bench.sim")


def run_trial(
    env: LiberoEnv,
    trial: int,
    init_index: int,
    molmo: MolmoActLiberoLocal,
    max_steps: int,
    num_steps: int,
    seed: Optional[int] = None,
) -> TrialRecord:
    molmo.reset()
    obs: LiberoObs = env.reset(init_index=init_index, seed=seed)
    rec = TrialRecord(task_id=env.spec.task_id, trial=trial, success=False, n_steps=0, wallclock_s=0.0)
    t_start = time.perf_counter()
    step_sw = Stopwatch()
    success = False
    done = False

    while len(rec.steps) < max_steps and not (success or done):
        e2e_t0 = time.perf_counter()
        pred = molmo.act(
            agentview=obs.agentview,
            wrist=obs.wrist,
            instruction=env.spec.instruction,
            state8=obs.state8,
            num_steps=num_steps,
        )
        # Execute every action in the returned chunk before re-querying the model.
        for a in pred.actions:
            with step_sw():
                obs, _reward, done, _info = env.step(a)
            success = env.success()
            e2e_ms = (time.perf_counter() - e2e_t0) * 1000.0
            rec.steps.append(StepRecord(
                step=len(rec.steps),
                camera_ms=0.0,          # in-sim render is bundled into env.step
                infer_server_ms=pred.server_dt_ms,
                infer_rtt_ms=pred.rtt_ms,
                motion_rest_ms=step_sw.ms,
                motion_cmd_ms=0.0,
                e2e_ms=e2e_ms,
                action=[float(x) for x in a],
                instruction=env.spec.instruction,
            ))
            if success or done or len(rec.steps) >= max_steps:
                break
            e2e_t0 = time.perf_counter()

    rec.n_steps = len(rec.steps)
    rec.wallclock_s = time.perf_counter() - t_start
    rec.success = bool(success)
    return rec


def run_libero_benchmark(
    suite: str = "libero_spatial",
    task_indices: Optional[list[int]] = None,
    trials_per_task: int = 5,
    model_id: str = "allenai/MolmoAct2-LIBERO",
    dtype: str = "bf16",
    num_steps: int = 10,
    max_steps: int = 300,
    camera_height: int = 256,
    camera_width: int = 256,
    enable_cuda_graph: bool = True,
    seed: int = 0,
    results_dir: str = "benchmarks/results",
) -> RunRecord:
    if suite not in SUITES:
        raise ValueError(f"suite must be one of {SUITES}")

    log.info("loading model %s (dtype=%s)", model_id, dtype)
    molmo = MolmoActLiberoLocal(model_id=model_id, dtype=dtype, enable_cuda_graph=enable_cuda_graph)
    log.info("model ready: %s", molmo.health())

    if task_indices is None:
        task_indices = [i for i, _ in list_suite_tasks(suite)]

    run = RunRecord(
        run_id=f"libero-{suite}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        started_at=time.time(),
        model_id=model_id,
        franka_endpoint=f"sim:{suite}",
        molmoact_endpoint="in-process",
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
                    num_steps=num_steps,
                    seed=seed + trial,
                )
                run.trials.append(tr)
                run.dump(results_dir)
                log.info("trial done: %s/#%d success=%s steps=%d wallclock=%.1fs",
                         tr.task_id, tr.trial, tr.success, tr.n_steps, tr.wallclock_s)
        finally:
            env.close()
    return run
