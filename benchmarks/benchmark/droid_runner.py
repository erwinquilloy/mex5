"""Real-Franka eval loop for MolmoAct2-DROID, mirroring Table 6 protocol."""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Optional

import numpy as np

from . import live_view
from .dual_camera import DualCamera, Frames, from_env as camera_from_env
from .droid_tasks import DroidTask, all_tasks, by_id
from .metrics import RunRecord, StepRecord, Stopwatch, TrialRecord
from .molmoact_droid_client import DroidClient
from .transport import make_driver

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
    panda,  # PandaDriver | FrankaRestDriver — same surface
    num_steps: int,
    chunk_step_dt_s: float,
    exec_rows: int = 3,
    grasp_commit_grip_frac: float = 0.5,
    fine_refinement_travel_rad: float = 0.2,
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
        # Adaptive execution. Default = trust the model and run the full
        # chunk in one open-loop burst (less stop-and-go, faster overall).
        # Only switch to short receding-horizon bursts when the chunk is
        # clearly a fine-refinement (small total joint travel) - that's the
        # case where the policy is "twitching toward target near the object"
        # and benefits from fresh visual after each small correction.
        #
        # Always commit the full chunk when the gripper is closing (grasp
        # phase); interrupting that for inference latency causes failures.
        gripper_signal = pred.actions[:, 7] >= 0.5
        grasp_committed = (gripper_signal.sum() >= grasp_commit_grip_frac * len(gripper_signal))
        if len(pred.actions) > 1:
            deltas = np.diff(pred.actions[:, :7], axis=0)
            total_travel_rad = float(np.sum(np.linalg.norm(deltas, axis=1)))
        else:
            total_travel_rad = 0.0
        is_fine_refinement = total_travel_rad < fine_refinement_travel_rad
        if grasp_committed:
            rows_to_run = pred.actions
        elif is_fine_refinement and exec_rows > 0:
            rows_to_run = pred.actions[:max(1, exec_rows)]
        else:
            rows_to_run = pred.actions
        lock_down = os.environ.get("FRANKA_BENCH_LOCK_GRIPPER_DOWN", "0") not in ("0", "", "false", "False")
        with exec_sw():
            panda.send_chunk(
                rows_to_run,
                step_dt_s=chunk_step_dt_s,
                lock_gripper_down=lock_down,
            )
        e2e_ms = (time.perf_counter() - e2e_t0) * 1000.0
        rec.steps.append(StepRecord(
            step=len(rec.steps),
            camera_ms=frames.t_grab_ms,
            infer_server_ms=pred.server_dt_ms,
            infer_rtt_ms=pred.rtt_ms,
            motion_rest_ms=exec_sw.ms,
            motion_cmd_ms=chunk_step_dt_s * 1000.0 * len(rows_to_run),
            e2e_ms=e2e_ms,
            action=rows_to_run[-1].astype(float).tolist(),   # log just the last cmd actually executed
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
    exec_rows: int = 3,
    grasp_commit_grip_frac: float = 0.5,
    fine_refinement_travel_rad: float = 0.2,
    transport: str = "fci",
    rest_host: Optional[str] = None,
    rest_port: int = 34568,
    rest_step_time_s: float = 2.5,
    mcp_url: Optional[str] = None,
) -> RunRecord:
    client = DroidClient(molmoact_url)
    health = client.health()
    log.info("server health: %s", health)
    log.info("live cam view: run `python -m benchmarks.scripts.serve_live` in another shell, then open http://<workstation-ip>:8080/")

    panda = make_driver(
        transport,
        fci_host=franka_host,
        rest_host=rest_host,
        rest_port=rest_port,
        mcp_url=mcp_url,
        step_time_s=rest_step_time_s,
    )
    if transport in ("rest", "mcp"):
        # On REST/MCP paths chunk_step_dt_s carries the per-row REST move
        # duration, not the FCI substep ramp interval.
        chunk_step_dt_s = rest_step_time_s
    camera = camera_from_env()

    tasks = [by_id(t) for t in task_ids] if task_ids else all_tasks()
    run = RunRecord(
        run_id=f"droid-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        started_at=time.time(),
        model_id=health.get("repo_id", "?"),
        franka_endpoint=f"{transport}:{franka_host or rest_host or mcp_url or '<env>'}",
        molmoact_endpoint=molmoact_url,
    )

    try:
        for task in tasks:
            n_trials = trials_per_task if trials_per_task is not None else task.trials
            for trial in range(n_trials):
                try:
                    tr = run_trial(task, trial, camera, client, panda,
                                   num_steps=num_steps,
                                   chunk_step_dt_s=chunk_step_dt_s,
                                   exec_rows=exec_rows,
                                   grasp_commit_grip_frac=grasp_commit_grip_frac,
                                   fine_refinement_travel_rad=fine_refinement_travel_rad)
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
