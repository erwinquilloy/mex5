"""Pure helpers + config for receding-horizon, transition-aware execution.

This is the flag-gated alternative to the dashboard's chunk-streaming path. Per
inference the executor runs only ONE clipped waypoint -- the pose just before
the next gripper transition, or the chunk tail -- then re-perceives, so the
closed loop self-corrects instead of streaming a whole trajectory open-loop.

The functions here are deliberately pure (no robot/IO) so they're unit-testable
without hardware. The loop that drives them lives in
DashboardState.do_task_receding (serve_dashboard.py). Logic ported from a
colleague's control_loop.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

GRASP_SUCCESS = "Object grasped successfully."


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


@dataclass
class RecedingConfig:
    """Knobs for the receding-horizon executor. Defaults match the colleague's
    (Deo's) working control_loop config; each overridable via FRANKA_BENCH_RH_*.

    Strategy (Deo): per inference pick one waypoint -- the gripper-transition row
    when the chunk grasps/releases, else the farthest approach waypoint within
    ``max_step_delta`` of the current pose -- then move the FULL distance there
    via moveToJointPose with a speed-limited ``tf``. No delta-clipping, no
    separate cartesian descent: a grasp closes at the policy's grasp pose, with
    grasp-on-contact catching the object if the joint move reaches it early."""
    exec_index: int = -1                 # cap on lookahead waypoint (Python idx)
    max_step_delta: float = 0.20         # rad budget for approach-waypoint pick
    exec_time_s: float = 1.0             # tf for moveToJointPose (server floor 1.0)
    max_joint_speed: float = 0.6         # rad/s; lengthen tf for big deltas. 0=off
    gripper_close_threshold: float = 0.5  # model gripper >= this => close
    gripper_open_threshold: float = 0.5   # < this => open (single 0.5 like Deo)
    grasp_width_m: float = 0.01          # squeeze target for closeGripper
    # Treat a reflex during the (full) move to the grasp pose as "reached the
    # object" and close on it, rather than aborting. (Deo's grasp-on-contact.)
    grasp_on_contact: bool = True
    # Optional Z-only pre-grasp descent (Deo's pregrasp). The policy's grasp pose
    # can sit a few cm high; after reaching it, lower straight down so the fingers
    # straddle the object's body. grasp-on-contact stops the descent at the real
    # grasp height; pregrasp_z_min floors it so a thin object can't be driven
    # through the table. ON here because this rig's policy aims high (Deo keeps it
    # off -- his policy doesn't need it). Skipped if the joint move already
    # contacted the object.
    pregrasp_enabled: bool = True
    pregrasp_z_offset: float = -0.06     # max downward descent (m); contact stops it early
    pregrasp_z_min: float = 0.02         # absolute TCP z floor (m) for the descent
    pregrasp_time_s: float = 0.5
    # Fixed lateral (base-frame XY) correction applied at the grasp pose before
    # descending. The closed loop can't remove a SYSTEMATIC policy XY bias (it
    # only averages jitter), so dial this to the consistent miss: if the gripper
    # lands the same distance/direction off the object every attempt, set
    # (dx, dy) = -(that miss). 0,0 = trust the policy's XY (Deo's default).
    grasp_xy_offset_m: tuple = (0.0, 0.0)
    grasp_xy_time_s: float = 2.0         # tf for the lateral align (slow: it's a
                                         # several-cm move, not the small Z descent)
    # After a failed grasp (jaws closed empty / object slipped), lift straight up
    # by this much before re-attempting, so the next approach starts from a clean
    # vantage and re-perceives from above instead of jabbing back down at the same
    # low pose. 0 disables.
    grasp_fail_retreat_z_m: float = 0.10
    max_collision_recoveries: int = 8
    home_q: tuple = field(default_factory=lambda: (
        0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4))

    @classmethod
    def from_env(cls) -> "RecedingConfig":
        c = cls()
        c.exec_index = _env_int("FRANKA_BENCH_RH_EXEC_INDEX", c.exec_index)
        c.max_step_delta = _env_float(
            "FRANKA_BENCH_RH_MAX_STEP_DELTA", c.max_step_delta)
        c.exec_time_s = _env_float("FRANKA_BENCH_RH_EXEC_TIME_S", c.exec_time_s)
        c.max_joint_speed = _env_float(
            "FRANKA_BENCH_RH_MAX_JOINT_SPEED", c.max_joint_speed)
        c.gripper_close_threshold = _env_float(
            "FRANKA_BENCH_RH_GRIPPER_CLOSE_THRESHOLD", c.gripper_close_threshold)
        c.gripper_open_threshold = _env_float(
            "FRANKA_BENCH_RH_GRIPPER_OPEN_THRESHOLD", c.gripper_open_threshold)
        c.grasp_width_m = _env_float(
            "FRANKA_BENCH_RH_GRASP_WIDTH_M", c.grasp_width_m)
        c.max_collision_recoveries = _env_int(
            "FRANKA_BENCH_RH_MAX_COLLISION_RECOVERIES", c.max_collision_recoveries)
        goc = os.environ.get("FRANKA_BENCH_RH_GRASP_ON_CONTACT")
        if goc is not None:
            c.grasp_on_contact = goc.strip().lower() in ("1", "true", "yes", "on")
        pe = os.environ.get("FRANKA_BENCH_RH_PREGRASP_ENABLED")
        if pe is not None:
            c.pregrasp_enabled = pe.strip().lower() in ("1", "true", "yes", "on")
        c.pregrasp_z_offset = _env_float(
            "FRANKA_BENCH_RH_PREGRASP_Z_OFFSET", c.pregrasp_z_offset)
        c.pregrasp_z_min = _env_float(
            "FRANKA_BENCH_RH_PREGRASP_Z_MIN", c.pregrasp_z_min)
        c.pregrasp_time_s = _env_float(
            "FRANKA_BENCH_RH_PREGRASP_TIME_S", c.pregrasp_time_s)
        c.grasp_xy_time_s = _env_float(
            "FRANKA_BENCH_RH_GRASP_XY_TIME_S", c.grasp_xy_time_s)
        c.grasp_fail_retreat_z_m = _env_float(
            "FRANKA_BENCH_RH_GRASP_FAIL_RETREAT_Z_M", c.grasp_fail_retreat_z_m)
        xy = os.environ.get("FRANKA_BENCH_RH_GRASP_XY_OFFSET")
        if xy:
            parts = [float(p) for p in xy.split(",")]
            if len(parts) != 2:
                raise ValueError(
                    "FRANKA_BENCH_RH_GRASP_XY_OFFSET must be 'dx,dy' (metres)")
            c.grasp_xy_offset_m = tuple(parts)
        return c


def find_gripper_transition(
    actions: np.ndarray,
    currently_closed: bool,
    close_threshold: float,
    open_threshold: float,
) -> tuple[str | None, int | None]:
    """First predicted gripper transition in a chunk. If the gripper is
    currently closed we look for the first row commanding open (<= open_thr);
    otherwise the first row commanding close (>= close_thr). Returns
    (intent, row_index) or (None, None) when the chunk holds the current
    state throughout."""
    gripper = np.asarray(actions[:, 7], dtype=np.float64)
    if currently_closed:
        matches = np.flatnonzero(gripper <= open_threshold)
        intent = "open"
    else:
        matches = np.flatnonzero(gripper >= close_threshold)
        intent = "close"
    if matches.size == 0:
        return None, None
    return intent, int(matches[0])


def select_target_index(actions: np.ndarray, transition_index: int | None) -> int:
    """Chunk tail for pure motion, or the gripper-transition row itself when the
    chunk grasps/releases -- so we actuate the gripper AT the pose the policy
    chose for it, not one waypoint short. (Deo targets the transition row t;
    targeting t-1 left the hand mid-approach, off the object's XY.)"""
    if transition_index is None:
        return actions.shape[0] - 1
    return transition_index


def clip_blocking_target(
    current_q: np.ndarray,
    target_q: np.ndarray,
    max_dq: float,
) -> np.ndarray:
    """Clamp the per-joint delta to +/- max_dq so a single blocking command
    never lunges -- the receding loop re-perceives after each small step."""
    current = np.asarray(current_q, dtype=np.float64)
    target = np.asarray(target_q, dtype=np.float64)
    return current + np.clip(target - current, -max_dq, max_dq)


def validate_actions(actions: np.ndarray) -> np.ndarray:
    """Validate one MolmoAct2-DROID action chunk before robot execution."""
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] < 8:
        raise ValueError("actions must have shape (T, >=8) with T > 0.")
    if not np.all(np.isfinite(actions[:, :8])):
        raise ValueError("actions contain non-finite values.")
    return actions


def select_waypoint_index_within_budget(
    actions: np.ndarray,
    current_q7: np.ndarray,
    max_step_delta: float,
    max_index: int = -1,
) -> int:
    """Farthest chunk waypoint whose max per-joint delta from ``current_q7`` is
    within ``max_step_delta`` (rad), capped at ``max_index``. Proximity-aware
    lookahead (Deo): bounds per-step motion so a near-object correction can't
    overshoot, while a long reach still advances ~max_step_delta per step.
    ``max_step_delta`` <= 0 disables budgeting and returns the cap. Ported from
    Deo's transforms.py."""
    actions = np.asarray(actions, dtype=np.float64)
    n = actions.shape[0]
    cur = np.asarray(current_q7, dtype=np.float64)
    cap = max_index % n if max_index < 0 else min(max_index, n - 1)
    if max_step_delta is None or max_step_delta <= 0:
        return cap
    chosen = 0
    for i in range(cap + 1):
        d = float(np.max(np.abs(actions[i, :7] - cur)))
        if d <= max_step_delta:
            chosen = i
        else:
            break  # deltas grow monotonically along a reach; stop at first overrun
    return chosen


def speed_limited_tf(
    target_q7: np.ndarray,
    current_q7: np.ndarray | None,
    base_tf: float,
    max_joint_speed: float,
) -> float:
    """Lengthen ``base_tf`` so no joint's cubic peak velocity exceeds the cap.
    Fast, large-delta joint moves are what trip the Franka reflex; for a cubic
    profile the peak velocity is ~1.875*delta/tf, so this keeps the worst joint
    under ``max_joint_speed`` rad/s, never shorter than ``base_tf``. ``0``
    disables. Ported from Deo's control_loop._speed_limited_tf."""
    if not max_joint_speed or current_q7 is None:
        return base_tf
    d = float(np.max(np.abs(
        np.asarray(target_q7, dtype=np.float64)
        - np.asarray(current_q7, dtype=np.float64))))
    return max(base_tf, 1.875 * d / max_joint_speed)
