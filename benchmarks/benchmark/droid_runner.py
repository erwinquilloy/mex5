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


# Franka gripper width thresholds. Fully open ≈ 0.08 m, fully closed ≈ 0.0 m.
# Anything in between means the jaws stopped on an object — i.e. holding.
_HOLDING_WIDTH_MIN_M = 0.005
_HOLDING_WIDTH_MAX_M = 0.075


def _is_holding(width: float) -> bool:
    return _HOLDING_WIDTH_MIN_M < width < _HOLDING_WIDTH_MAX_M


def _row_tcp_xy(q_row: np.ndarray) -> tuple[float, float]:
    import panda_py
    T = panda_py.fk(np.asarray(q_row[:7], dtype=np.float64))
    return float(T[0, 3]), float(T[1, 3])


def _enforce_hold_until_target(
    actions: np.ndarray,
    width: float,
    target_zone: tuple[float, float, float, float],
    grip_threshold: float = 0.5,
) -> int:
    """Suppress policy-commanded gripper-opens while holding an object outside
    the target zone. Mutates ``actions[:, 7]`` in place; returns the number of
    rows that got overridden. No-op when the gripper isn't holding anything."""
    if not _is_holding(width):
        return 0
    xmin, xmax, ymin, ymax = target_zone
    suppressed = 0
    for i in range(len(actions)):
        if actions[i, 7] < grip_threshold:
            x, y = _row_tcp_xy(actions[i])
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                actions[i, 7] = 1.0
                suppressed += 1
    return suppressed


def _enforce_hold_until_transported(
    actions: np.ndarray,
    state8: np.ndarray,
    grasp_xy: Optional[tuple[float, float]],
    min_dist_m: float,
    grip_threshold: float = 0.5,
) -> tuple[int, Optional[tuple[float, float]]]:
    """Heuristic alternative to _enforce_hold_until_target for setups where the
    target destination is not a fixed XY box (e.g. a plate moved around between
    trials).

    Idea: capture the TCP XY at the first chunk where we observe the gripper
    holding an object, treat that as the pickup point, and suppress any later
    gripper-open whose commanded TCP XY is within ``min_dist_m`` of it. So the
    arm must physically transport the object at least ``min_dist_m`` (Euclidean
    in base-frame XY) before a policy-commanded release is honoured.

    Mutates ``actions[:, 7]`` in place. Returns
    ``(n_rows_suppressed, updated_grasp_xy)`` -- the caller persists
    ``grasp_xy`` across chunks of one trial and resets it when the gripper is
    no longer holding (so the next pickup gets its own reference point).

    ``min_dist_m <= 0`` disables the guard.
    """
    if min_dist_m <= 0.0:
        return 0, grasp_xy
    width = float(state8[7])
    if not _is_holding(width):
        # No object in the jaws -- reset so the next grasp re-anchors.
        return 0, None
    if grasp_xy is None:
        grasp_xy = _row_tcp_xy(state8)
    gx, gy = grasp_xy
    r2 = float(min_dist_m) ** 2
    suppressed = 0
    for i in range(len(actions)):
        if actions[i, 7] < grip_threshold:
            x, y = _row_tcp_xy(actions[i])
            if (x - gx) ** 2 + (y - gy) ** 2 < r2:
                actions[i, 7] = 1.0
                suppressed += 1
    return suppressed, grasp_xy


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
    max_chunks: Optional[int] = None,
    hold_until_target: bool = False,
) -> TrialRecord:
    if hold_until_target and task.target_zone_xy is None:
        raise ValueError(
            f"--hold-until-target was set but task {task.task_id!r} has no "
            "target_zone_xy defined in droid_tasks.py. Either define the box "
            "(xmin, xmax, ymin, ymax in metres, base frame) or run without "
            "the flag."
        )
    rec = TrialRecord(task_id=task.task_id, trial=trial, success=False, n_steps=0, wallclock_s=0.0)
    panda.home()
    if not _prompt_setup(task, trial):
        rec.notes = "skipped at setup"
        return rec

    t_start = time.perf_counter()
    chunk_sw, exec_sw = Stopwatch(), Stopwatch()

    chunk_budget = max_chunks if max_chunks is not None else task.max_chunks
    for chunk_i in range(chunk_budget):
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
        if hold_until_target:
            # Copy so we don't pollute pred.actions (the model's raw output, which
            # later logging and analysis may want to see verbatim).
            rows_to_run = np.asarray(rows_to_run, dtype=np.float64).copy()
            n_suppressed = _enforce_hold_until_target(
                rows_to_run, float(state8[7]), task.target_zone_xy,  # type: ignore[arg-type]
            )
            if n_suppressed > 0:
                log.warning(
                    "hold-until-target: kept gripper closed on %d/%d row(s) "
                    "(%s/#%d) — policy tried to release outside target zone",
                    n_suppressed, len(rows_to_run), task.task_id, trial,
                )
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
    max_chunks: Optional[int] = None,
    transport: str = "fci",
    rest_host: Optional[str] = None,
    rest_port: int = 34568,
    rest_step_time_s: float = 2.5,
    mcp_url: Optional[str] = None,
    hold_until_target: bool = False,
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
                                   fine_refinement_travel_rad=fine_refinement_travel_rad,
                                   max_chunks=max_chunks,
                                   hold_until_target=hold_until_target)
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
