"""panda_py-based driver for MolmoAct2-DROID's absolute joint-pose actions.

The DROID checkpoint outputs `actions[N, 8]` where:
  actions[:, :7] = absolute joint positions (radians)
  actions[:, 7]  = gripper command (continuous; we treat >=0.5 as close)

We bypass the C++ motion_server REST API because that endpoint only speaks
cartesian deltas. panda_py talks to libfranka directly and is what
franka/python/basic.py already uses on this rig.
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

    def _apply_camera_z_offset(self, actions: np.ndarray, z_offset_m: float) -> np.ndarray:
        """Lower (or raise) each commanded EE Z by ``z_offset_m`` via FK->shift->IK.

        Compensates for the wrist camera being mounted above the gripper:
        the model targets put the *camera* on the object, but we want the
        *gripper* on the object. Sign convention: ``z_offset_m`` is the amount
        to subtract from the commanded EE Z (so a positive value lowers the
        gripper relative to where the camera would be). IK failures fall
        through to the original target unchanged.
        """
        if abs(z_offset_m) < 1e-6:
            return actions
        try:
            import panda_py
        except Exception:
            return actions
        out = actions.copy()
        for i in range(len(actions)):
            q = actions[i, :7]
            try:
                pose = panda_py.fk(q)
                pose[2, 3] -= z_offset_m
                q_new = panda_py.ik(pose)
                if q_new is not None and not np.any(np.isnan(q_new)):
                    out[i, :7] = q_new
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
        wrist_cam_z_offset_m: float = 0.0,
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

        if wrist_cam_z_offset_m:
            actions = self._apply_camera_z_offset(actions, wrist_cam_z_offset_m)

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
