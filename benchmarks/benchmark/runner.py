"""Capture -> MolmoAct2 -> Franka REST loop with per-step latency logging."""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from .camera import Camera, make_camera
from .franka_client import FrankaClient
from .metrics import RunRecord, StepRecord, Stopwatch, TrialRecord
from .molmoact_client import MolmoActClient
from .tasks import Task, all_tasks, by_id

log = logging.getLogger("bench")


def _ask_success(task: Task, trial: int) -> tuple[bool, str]:
    print(f"\n[{task.task_id}] trial {trial}: did MolmoAct2 complete the task? [y/n] ", end="", flush=True)
    ans = input().strip().lower()
    notes = ""
    if ans not in ("y", "yes"):
        print("  notes (optional, one line): ", end="", flush=True)
        notes = input().strip()
        return False, notes
    return True, notes


def run_trial(
    task: Task,
    trial: int,
    camera: Camera,
    molmo: MolmoActClient,
    franka: FrankaClient,
    n_action_chunk: int = 1,
    step_time_s: float = 1.0,
    interactive_success: bool = True,
) -> TrialRecord:
    molmo.reset()
    franka.move_cartesian(*task.home_xyz, t_sec=3.0)
    franka.open_gripper()
    rec = TrialRecord(task_id=task.task_id, trial=trial, success=False, n_steps=0, wallclock_s=0.0)
    cur = list(task.home_xyz)
    t_start = time.perf_counter()
    cam_sw, motion_sw = Stopwatch(), Stopwatch()

    for step in range(task.max_steps):
        e2e_t0 = time.perf_counter()
        with cam_sw():
            img = camera.grab()
        pred = molmo.predict(img, task.instruction, n_actions=n_action_chunk)
        # Execute every action in the returned chunk before re-querying the model.
        for a in pred.actions:
            with motion_sw():
                fr = franka.apply_delta(cur, a, t_sec=step_time_s)
            cur[0] += a[0]; cur[1] += a[1]; cur[2] += a[2]
            e2e_ms = (time.perf_counter() - e2e_t0) * 1000.0
            rec.steps.append(StepRecord(
                step=len(rec.steps),
                camera_ms=cam_sw.ms,
                infer_server_ms=pred.server_dt_ms,
                infer_rtt_ms=pred.rtt_ms,
                motion_rest_ms=motion_sw.ms,
                motion_cmd_ms=fr.motion_ms,
                e2e_ms=e2e_ms,
                action=list(a),
                instruction=task.instruction,
            ))
            e2e_t0 = time.perf_counter()  # next step's e2e starts here

    rec.n_steps = len(rec.steps)
    rec.wallclock_s = time.perf_counter() - t_start
    if interactive_success:
        rec.success, rec.notes = _ask_success(task, trial)
    return rec


def run_benchmark(
    task_ids: Optional[list[str]] = None,
    trials_per_task: int = 5,
    franka_ip: str = "192.168.2.1",
    franka_port: int = 34568,
    molmoact_url: str = "http://localhost:8000",
    n_action_chunk: int = 1,
    step_time_s: float = 1.0,
    results_dir: str = "benchmarks/results",
    interactive_success: bool = True,
) -> RunRecord:
    camera = make_camera()
    molmo = MolmoActClient(molmoact_url)
    franka = FrankaClient(franka_ip, franka_port)
    health = molmo.health()
    log.info("molmoact health: %s", health)

    tasks = [by_id(t) for t in task_ids] if task_ids else all_tasks()
    run = RunRecord(
        run_id=f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        started_at=time.time(),
        model_id=health.get("model_id", "?"),
        franka_endpoint=f"{franka_ip}:{franka_port}",
        molmoact_endpoint=molmoact_url,
    )
    try:
        for task in tasks:
            print(f"\n=== {task.task_id} ({task.suite}) ===")
            print(f"setup: {task.setup_notes}")
            print(f"instruction: {task.instruction!r}")
            for trial in range(trials_per_task):
                tr = run_trial(task, trial, camera, molmo, franka,
                               n_action_chunk=n_action_chunk,
                               step_time_s=step_time_s,
                               interactive_success=interactive_success)
                run.trials.append(tr)
                out = run.dump(results_dir)  # persist after every trial
                log.info("trial done: %s/#%d success=%s steps=%d (dumped %s)",
                         task.task_id, trial, tr.success, tr.n_steps, out)
    finally:
        camera.close()
    return run
