"""panda_py-based driver for MolmoAct2-DROID's absolute joint-pose actions.

The DROID checkpoint outputs `actions[N, 8]` where:
  actions[:, :7] = absolute joint positions (radians)
  actions[:, 7]  = gripper command (continuous; we treat >=0.5 as close)

We bypass the C++ motion_server REST API because that endpoint only speaks
cartesian deltas. panda_py talks to libfranka directly and is what
franka/python/basic.py already uses on this rig.

Tunable via env vars (all optional):
    FRANKA_BENCH_FCI_CAM_DX_M   wrist-cam → TCP X offset in base frame, applied
                                ONLY to the terminal row of each chunk. FK the
                                row's joints → shift TCP X → IK back with the
                                previous row as seed. Set to the camera's
                                forward offset from the gripper (e.g. 0.08).
                                Default 0.
    FRANKA_BENCH_FCI_CAM_DZ_M   same idea for Z (cam above TCP). Default 0.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# panda_py constants
_HOME_Q = np.array([0., -np.pi/4, 0., -3*np.pi/4, 0., np.pi/2, np.pi/4], dtype=np.float64)
_GRIPPER_MAX_M = 0.08


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

        # Wrist-cam → TCP offsets in the robot base frame. Applied only to the
        # last row of each chunk, mirroring FrankaRestDriver. See module docstring.
        self._cam_dx_m = float(os.environ.get("FRANKA_BENCH_FCI_CAM_DX_M", "0.0"))
        self._cam_dz_m = float(os.environ.get("FRANKA_BENCH_FCI_CAM_DZ_M", "0.0"))

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
        max_joint_jump_rad: float = 0.5,
    ) -> np.ndarray:
        """Shift only the last row's TCP X/Z by self._cam_d{x,z}_m via FK→shift→IK.

        Seeded with the previous row's joints (or current state if N==1) so the
        IK solution stays on the same branch and the substep interpolator can
        ramp smoothly into it. Falls back to the original row if IK fails or
        jumps more than ``max_joint_jump_rad`` per joint.
        """
        if self._cam_dx_m == 0.0 and self._cam_dz_m == 0.0:
            return actions
        try:
            import panda_py
        except Exception:
            return actions
        out = actions.copy()
        try:
            if len(actions) >= 2:
                seed = np.asarray(actions[-2, :7], dtype=np.float64)
            else:
                seed = np.asarray(self._panda.get_state().q, dtype=np.float64)
            q_last = np.asarray(actions[-1, :7], dtype=np.float64)
            pose = panda_py.fk(q_last)
            pose[0, 3] += self._cam_dx_m
            pose[2, 3] += self._cam_dz_m
            try:
                q_new = panda_py.ik(pose, seed)
            except TypeError:
                q_new = panda_py.ik(pose)
            if q_new is None or np.any(np.isnan(q_new)):
                return out
            if np.max(np.abs(q_new - seed)) > max_joint_jump_rad:
                return out
            out[-1, :7] = q_new
        except Exception:
            pass
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
        actions = self._apply_cam_offset_to_terminal(actions)

        nominal_sub = max(1, int(round(step_dt_s / max(substep_dt_s, 1e-3))))

        ctrl = pc.JointPosition()
        self._panda.start_controller(ctrl)
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
        finally:
            try:
                self._panda.stop_controller()
            except Exception:
                pass

    # ----- lifecycle -----

    def home(self) -> None:
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
