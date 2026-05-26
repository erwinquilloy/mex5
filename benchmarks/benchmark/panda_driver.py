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

    def send_chunk(
        self,
        actions: np.ndarray,
        step_dt_s: float = 0.1,
        grip_threshold: float = 0.5,
    ) -> None:
        """Replay an (N, 8) action chunk.

        Each row commands `move_to_joint_position` toward `actions[i, :7]` with
        a short motion budget. Gripper toggles only on transitions to avoid
        spamming the gripper driver.
        """
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"expected (N, 8), got {actions.shape}")
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
            # blocking move; speed_factor caps duration roughly at step_dt_s.
            self._panda.move_to_joint_position(q_target, speed_factor=min(1.0, 0.5))
            # If panda_py returned faster than step_dt_s (because the move was tiny),
            # honor the requested chunk rhythm. If slower, just continue immediately.
            time.sleep(0)

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
