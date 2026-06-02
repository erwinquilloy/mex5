"""Real-Franka eval loop for MolmoAct2-DROID, mirroring Table 6 protocol."""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from . import live_view
from .dual_camera import DualCamera, Frames, from_env as camera_from_env
from .droid_tasks import DroidTask, all_tasks, by_id
from .metrics import RunRecord, StepRecord, Stopwatch, TrialRecord
from .molmoact_droid_client import DroidClient
from .panda_driver import PandaDriver

log = logging.getLogger("bench.droid")


def _prompt_setup(task: DroidTask, trial: int) -> bool:
    print("\n" + "=" * 70)
    print(f"[{task.task_id}] trial {trial + 1}/{task.trials}")
    print(f"  instruction: {task.instruction!r}")
    print(f"  scene setup: {task.setup_notes}")
    print("Place the scene, randomize the external camera pose,")
    print("then press ENTER to begin (or 'skip' to skip this trial).")
    ans = input("> ").strip().lower()
    return ans != "skip"


def _prompt_success(task: DroidTask, trial: int) -> tuple[bool, str]:
    print(f"\n[{task.task_id}] trial {trial + 1}: success? [y/n/a(bort-run)]")
    ans = input("> ").strip().lower()
    if ans in ("a", "abort"):
        raise KeyboardInterrupt("operator aborted run")
    if ans in ("y", "yes"):
        return True, ""
    print("  notes (one line, optional): ", end="", flush=True)
    return False, input().strip()


def run_trial(
    task: DroidTask,
    trial: int,
    camera: DualCamera,
    client: DroidClient,
    panda: PandaDriver,
    num_steps: int,
    chunk_step_dt_s: float,
) -> TrialRecord:
    rec = TrialRecord(task_id=task.task_id, trial=trial, success=False, n_steps=0, wallclock_s=0.0)
    panda.home()
    if not _prompt_setup(task, trial):
        rec.notes = "skipped at setup"
        return rec

    t_start = time.perf_counter()
    chunk_sw, exec_sw = Stopwatch(), Stopwatch()

    for chunk_i in range(task.max_chunks):
        e2e_t0 = time.perf_counter()
        with chunk_sw():
            frames: Frames = camera.grab()
        live_view.update(frames.external, frames.wrist)
        state8 = panda.state_vec8()
        pred = client.act(
            external_cam=frames.external,
            wrist_cam=frames.wrist,
            instruction=task.instruction,
            state=state8,
            num_steps=num_steps,
        )
        with exec_sw():
            panda.send_chunk(pred.actions, step_dt_s=chunk_step_dt_s)
        e2e_ms = (time.perf_counter() - e2e_t0) * 1000.0
        rec.steps.append(StepRecord(
            step=len(rec.steps),
            camera_ms=frames.t_grab_ms,
            infer_server_ms=pred.server_dt_ms,
            infer_rtt_ms=pred.rtt_ms,
            motion_rest_ms=exec_sw.ms,
            motion_cmd_ms=chunk_step_dt_s * 1000.0 * len(pred.actions),
            e2e_ms=e2e_ms,
            action=pred.actions[-1].astype(float).tolist(),   # log just the last cmd
            instruction=task.instruction,
        ))

    rec.n_steps = len(rec.steps)
    rec.wallclock_s = time.perf_counter() - t_start
    rec.success, rec.notes = _prompt_success(task, trial)
    return rec


def run_droid_benchmark(
    task_ids: Optional[list[str]] = None,
    trials_per_task: Optional[int] = None,
    molmoact_url: str = "http://localhost:8000",
    franka_host: Optional[str] = None,
    num_steps: int = 10,
    chunk_step_dt_s: float = 0.1,
    results_dir: str = "benchmarks/results",
) -> RunRecord:
    client = DroidClient(molmoact_url)
    health = client.health()
    log.info("server health: %s", health)
    log.info("live cam view: run `python -m benchmarks.scripts.serve_live` in another shell, then open http://<workstation-ip>:8080/")

    panda = PandaDriver(hostname=franka_host)
    camera = camera_from_env()

    tasks = [by_id(t) for t in task_ids] if task_ids else all_tasks()
    run = RunRecord(
        run_id=f"droid-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        started_at=time.time(),
        model_id=health.get("repo_id", "?"),
        franka_endpoint=(franka_host or "<env:FRANKA_HOST>"),
        molmoact_endpoint=molmoact_url,
    )

    try:
        for task in tasks:
            n_trials = trials_per_task if trials_per_task is not None else task.trials
            for trial in range(n_trials):
                try:
                    tr = run_trial(task, trial, camera, client, panda,
                                   num_steps=num_steps,
                                   chunk_step_dt_s=chunk_step_dt_s)
                except KeyboardInterrupt:
                    log.warning("aborted by operator at %s/#%d", task.task_id, trial)
                    raise
                run.trials.append(tr)
                run.dump(results_dir)
                log.info("trial done: %s/#%d success=%s steps=%d",
                         tr.task_id, tr.trial, tr.success, tr.n_steps)
    finally:
        try: camera.close()
        except Exception: pass
        try: panda.close()
        except Exception: pass
    return run
