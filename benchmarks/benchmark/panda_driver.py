"""panda_py-based driver for MolmoAct2-DROID's absolute joint-pose actions.

The DROID checkpoint outputs `actions[N, 8]` where:
  actions[:, :7] = absolute joint positions (radians)
  actions[:, 7]  = gripper command (continuous; we treat >=0.5 as close)

We bypass the C++ motion_server REST API because that endpoint only speaks
cartesian deltas. panda_py talks to libfranka directly and is what
franka/python/basic.py already uses on this rig.

Tunable via env vars (all optional):
    FRANKA_BENCH_FCI_CAM_DX_M           wrist-cam → TCP X offset in base
                                        frame. Set to the camera's forward
                                        offset from the gripper (e.g. 0.08).
                                        Default 0.
    FRANKA_BENCH_FCI_CAM_DZ_M           same idea for Z. Default 0.
    FRANKA_BENCH_FCI_CAM_OFFSET_MODE    when the DX/DZ shift fires:
                                        grasp_terminal (default) — last row
                                        of the grasp chunk only (gripper
                                        closes); every_terminal — last row
                                        of every chunk; always — every row
                                        of every chunk (constant TCP frame
                                        shift; use for whole-trajectory
                                        perception bias).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .driver_errors import CollisionAborted

# panda_py constants
_HOME_Q = np.array([0., -np.pi/4, 0., -3*np.pi/4, 0., np.pi/2, np.pi/4], dtype=np.float64)
_GRIPPER_MAX_M = 0.08

# airscan4 lab rig workspace safe box (base-frame metres). Cam-offset-shifted
# poses are clamped into this box before re-IK so the cam shift can't push the
# gripper past the rig's safe reach. Z is intentionally not clipped.
_LAB_X_MIN, _LAB_X_MAX = 0.0, 0.57
_LAB_Y_MIN, _LAB_Y_MAX = -0.4, 0.4


@dataclass
class DriverState:
    q: np.ndarray            # (7,) joint positions, radians
    gripper_width: float     # meters
    timestamp: float


class PandaDriver:
    def __init__(
        self,
        hostname: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        activate_fci: bool = False,
    ):
        import panda_py
        from panda_py import libfranka

        self._panda_py = panda_py
        host = hostname or os.environ["FRANKA_HOST"]
        if username and password:
            desk = panda_py.Desk(host, username, password)
            if activate_fci:
                desk.activate_fci()
        self._panda = panda_py.Panda(host)
        self._gripper = libfranka.Gripper(host)
        try:
            self._gripper.homing()
        except Exception:
            pass
        # Match libfranka examples' setDefaultBehavior (common.cpp from
        # roatienza/autonomous-robots and Franka's own examples). Looser
        # collision thresholds reduce spurious reflex stops on contact, and
        # the joint/Cartesian impedance values are the canonical defaults
        # for examples that stream setpoints.
        try:
            self._panda.get_robot().setCollisionBehavior(
                [20.0] * 7, [20.0] * 7,
                [10.0] * 7, [10.0] * 7,
                [20.0] * 6, [20.0] * 6,
                [10.0] * 6, [10.0] * 6,
            )
        except Exception:
            pass
        try:
            self._panda.get_robot().setJointImpedance([3000, 3000, 3000, 2500, 2500, 2000, 2000])
        except Exception:
            pass
        try:
            self._panda.get_robot().setCartesianImpedance([3000, 3000, 3000, 300, 300, 300])
        except Exception:
            pass

        # Wrist-cam → TCP offsets in the robot base frame. Applied per
        # FRANKA_BENCH_FCI_CAM_OFFSET_MODE, mirroring FrankaRestDriver:
        #   grasp_terminal (default) — last row of grasp chunks only (FK→shift→IK)
        #   every_terminal           — last row of every chunk (FK→shift→IK)
        #   always                   — every row of every chunk (FK→shift→IK each)
        # See module docstring + README.
        self._cam_dx_m = float(os.environ.get("FRANKA_BENCH_FCI_CAM_DX_M", "0.0"))
        self._cam_dz_m = float(os.environ.get("FRANKA_BENCH_FCI_CAM_DZ_M", "0.0"))
        _mode = os.environ.get("FRANKA_BENCH_FCI_CAM_OFFSET_MODE", "grasp_terminal").strip().lower()
        if _mode not in ("grasp_terminal", "every_terminal", "always"):
            raise ValueError(
                f"FRANKA_BENCH_FCI_CAM_OFFSET_MODE={_mode!r}; expected one of "
                "'grasp_terminal', 'every_terminal', 'always'."
            )
        self._cam_offset_mode = _mode

    # ----- state -----

    def get_state(self) -> DriverState:
        q = np.asarray(self._panda.get_state().q, dtype=np.float32)
        try:
            width = float(self._gripper.read_once().width)
        except Exception:
            width = _GRIPPER_MAX_M
        return DriverState(q=q, gripper_width=width, timestamp=time.time())

    def state_vec8(self) -> np.ndarray:
        s = self.get_state()
        return np.concatenate([s.q, [s.gripper_width]], dtype=np.float32)

    # ----- action chunk execution -----

    def _lock_orientation_downward(
        self,
        actions: np.ndarray,
        max_joint_jump_rad: float = 0.5,
    ) -> np.ndarray:
        """Replace each commanded EE rotation with downward-facing, keeping yaw.

        IK is seeded with the previous accepted row's joint solution so
        the solver picks a continuous branch (without a seed it can flip
        wrist / elbow configurations between consecutive rows, producing
        big joint jumps that trip the reflex). Rows whose IK result jumps
        more than ``max_joint_jump_rad`` per joint from the seed are
        skipped and the seed is held - prevents rare bad IK solutions
        from being executed.
        """
        try:
            import panda_py
        except Exception:
            return actions
        out = actions.copy()
        try:
            seed = np.asarray(self._panda.get_state().q, dtype=np.float64)
        except Exception:
            seed = np.asarray(actions[0, :7], dtype=np.float64)
        for i in range(len(actions)):
            q = actions[i, :7]
            try:
                pose = panda_py.fk(q)
                R = pose[:3, :3]
                ee_x_world = R[:, 0]
                yaw_dir = np.array([ee_x_world[0], ee_x_world[1], 0.0], dtype=np.float64)
                n = np.linalg.norm(yaw_dir)
                if n < 1e-6:
                    yaw_dir = np.array([1.0, 0.0, 0.0])
                else:
                    yaw_dir = yaw_dir / n
                new_x = yaw_dir
                new_z = np.array([0.0, 0.0, -1.0])
                new_y = np.cross(new_z, new_x)
                new_y = new_y / np.linalg.norm(new_y)
                R_new = np.column_stack([new_x, new_y, new_z])
                pose[:3, :3] = R_new
                try:
                    q_new = panda_py.ik(pose, seed)
                except TypeError:
                    q_new = panda_py.ik(pose)
                if q_new is None or np.any(np.isnan(q_new)):
                    continue
                if np.max(np.abs(q_new - seed)) > max_joint_jump_rad:
                    continue
                out[i, :7] = q_new
                seed = q_new
            except Exception:
                pass
        return out

    def _apply_cam_offset_to_terminal(
        self,
        actions: np.ndarray,
        grip_threshold: float = 0.5,
        max_joint_jump_rad: float = 0.5,
    ) -> np.ndarray:
        """Shift commanded TCP X/Z by self._cam_d{x,z}_m via FK→shift→IK.

        Which rows get shifted is set by self._cam_offset_mode:
          'grasp_terminal' — terminal row of grasp chunks only (gripper closes)
          'every_terminal' — terminal row of every chunk
          'always'         — every row of every chunk (constant TCP frame shift
                             across the whole trajectory)

        For each shifted row: FK the joints, shift TCP X/Z, IK back seeded
        with the previous accepted row's joints so the solver stays on the
        same branch. Falls back to the original joints if IK fails or jumps
        more than ``max_joint_jump_rad`` per joint.
        """
        if self._cam_dx_m == 0.0 and self._cam_dz_m == 0.0:
            return actions
        if len(actions) == 0:
            return actions
        if self._cam_offset_mode == "grasp_terminal" and float(actions[-1, 7]) < grip_threshold:
            return actions
        try:
            import panda_py
        except Exception:
            return actions

        if self._cam_offset_mode == "always":
            indices = list(range(len(actions)))
        else:
            indices = [len(actions) - 1]  # both *_terminal modes shift only the last row

        out = actions.copy()
        # Seed for the first shifted row's IK: either the previous (unshifted) row's
        # joints or current state when N==1 / shifting from index 0.
        if indices[0] == 0:
            try:
                seed = np.asarray(self._panda.get_state().q, dtype=np.float64)
            except Exception:
                seed = np.asarray(actions[0, :7], dtype=np.float64)
        else:
            seed = np.asarray(actions[indices[0] - 1, :7], dtype=np.float64)

        for idx in indices:
            try:
                q_row = np.asarray(actions[idx, :7], dtype=np.float64)
                pose = panda_py.fk(q_row)
                pose[0, 3] += self._cam_dx_m
                pose[2, 3] += self._cam_dz_m
                x_pre, y_pre = float(pose[0, 3]), float(pose[1, 3])
                pose[0, 3] = min(_LAB_X_MAX, max(_LAB_X_MIN, pose[0, 3]))
                pose[1, 3] = min(_LAB_Y_MAX, max(_LAB_Y_MIN, pose[1, 3]))
                if pose[0, 3] != x_pre or pose[1, 3] != y_pre:
                    print(
                        f"[PandaDriver] clamped cam-offset TCP target XY "
                        f"({x_pre:+.3f}, {y_pre:+.3f}) -> "
                        f"({float(pose[0,3]):+.3f}, {float(pose[1,3]):+.3f}) m to lab box "
                        f"X[{_LAB_X_MIN:.2f},{_LAB_X_MAX:.2f}] Y[{_LAB_Y_MIN:+.2f},{_LAB_Y_MAX:+.2f}]"
                    )
                try:
                    q_new = panda_py.ik(pose, seed)
                except TypeError:
                    q_new = panda_py.ik(pose)
                if q_new is None or np.any(np.isnan(q_new)):
                    continue
                if np.max(np.abs(q_new - seed)) > max_joint_jump_rad:
                    continue
                out[idx, :7] = q_new
                seed = q_new
            except Exception:
                continue
        return out

    def send_chunk(
        self,
        actions: np.ndarray,
        step_dt_s: float = 0.1,
        grip_threshold: float = 0.5,
        substep_dt_s: float = 0.01,
        max_joint_vel_rad_s: float = 0.5,
        lock_gripper_down: bool = False,
    ) -> None:
        """Stream an (N, 8) action chunk through panda_py's JointPosition controller.

        Between consecutive chunk rows we linearly interpolate the setpoint at
        ``substep_dt_s`` resolution so the controller sees a ramp instead of a
        step. The number of substeps per row is the larger of the nominal
        ``step_dt_s / substep_dt_s`` and what's needed to keep peak joint
        velocity under ``max_joint_vel_rad_s``. That way short moves still
        finish in ``step_dt_s`` but the first big move out of home no longer
        traverses a wide angle in 100 ms and trips the reflex.

        Gripper toggles happen between substep loops since the gripper RPCs
        aren't safe to invoke inside the streaming loop.
        """
        import panda_py.controllers as pc

        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"expected (N, 8), got {actions.shape}")
        if len(actions) == 0:
            return

        if lock_gripper_down:
            actions = self._lock_orientation_downward(actions)
        actions = self._apply_cam_offset_to_terminal(actions, grip_threshold=grip_threshold)

        nominal_sub = max(1, int(round(step_dt_s / max(substep_dt_s, 1e-3))))

        ctrl = pc.JointPosition()
        self._panda.start_controller(ctrl)
        # libfranka reflex (contact, velocity/accel discontinuity) raises out
        # of ctrl.set_control or the controller lifecycle. Translate to
        # CollisionAborted so the consumer loop can abort+home cleanly.
        # _REFLEX_HINTS captures the substrings libfranka surfaces; the broad
        # set avoids needing to depend on a specific panda_py exception class.
        _REFLEX_HINTS = ("reflex", "collision", "discontinuity",
                         "cartesian_motion_generator", "joint_motion_generator")
        try:
            prev_q = np.asarray(self._panda.get_state().q, dtype=np.float64)
            last_grip: Optional[bool] = None
            for row in actions:
                q_target = row[:7]
                close = bool(row[7] >= grip_threshold)
                if last_grip is None or close != last_grip:
                    if close:
                        self._gripper.grasp(0.0, 0.1, 60, epsilon_inner=0.04, epsilon_outer=0.04)
                    else:
                        self._gripper.move(_GRIPPER_MAX_M, 0.1)
                    last_grip = close
                # stretch the ramp so peak per-joint velocity stays bounded
                delta = q_target - prev_q
                vel_floor_sub = int(np.ceil(
                    np.max(np.abs(delta)) / (max_joint_vel_rad_s * substep_dt_s)
                ))
                n_sub = max(nominal_sub, vel_floor_sub, 1)
                for k in range(1, n_sub + 1):
                    alpha = k / n_sub
                    q_interp = prev_q + alpha * delta
                    ctrl.set_control(q_interp)
                    time.sleep(substep_dt_s)
                prev_q = q_target
        except Exception as e:
            msg = str(e).lower()
            if any(h in msg for h in _REFLEX_HINTS):
                raise CollisionAborted(f"FCI control reflex: {e}") from e
            raise
        finally:
            try:
                self._panda.stop_controller()
            except Exception:
                pass

    # ----- lifecycle -----

    def home(self) -> None:
        # After a reflex, libfranka requires automaticErrorRecovery before
        # the next motion is accepted. Safe to call when not in error state.
        # Try both panda_py spellings; one of them is always present.
        for attempt in (
            lambda: self._panda.recover(),
            lambda: self._panda.get_robot().automaticErrorRecovery(),
        ):
            try:
                attempt()
                break
            except Exception:
                continue
        self._panda.move_to_joint_position(_HOME_Q, speed_factor=0.3)
        try:
            self._gripper.move(_GRIPPER_MAX_M, 0.1)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._panda.move_to_joint_position(_HOME_Q, speed_factor=0.3)
        except Exception:
            pass
