"""REST-API driver for MolmoAct2-DROID, talking to franka/cpp/motion_server.

Drop-in alternative to PandaDriver: same public surface (home, state_vec8,
send_chunk, close), but every robot interaction goes through the C++
motion_server REST endpoints instead of opening libfranka directly.

The server speaks cartesian xyz + ZYX-Euler deltas in degrees; the DROID
policy emits absolute joint positions. We bridge by FK-ing both the current
and target joint vectors locally with panda_py, then sending the Euler delta.
panda_py.fk is a pure-kinematics call (no FCI), so it coexists fine with the
server's exclusive FCI session.

Requires the readJointState endpoint added to motion_server (returns
[q0..q6, gripper_width]).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import requests

_HOME_Q = np.array([0., -np.pi/4, 0., -3*np.pi/4, 0., np.pi/2, np.pi/4], dtype=np.float64)
_GRIPPER_MAX_M = 0.08
_DEFAULT_REST_STEP_TIME_S = 2.5
_DEFAULT_HOME_TIME_S = 3.0


@dataclass
class DriverState:
    q: np.ndarray            # (7,) joint positions, radians
    gripper_width: float     # meters
    timestamp: float


def _zyx_euler_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """Port of motion_server's getRotationAngles (ZYX: Rz(α) Ry(β) Rx(γ))."""
    beta = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    cos_beta = math.cos(beta)
    if abs(cos_beta) < 1e-5:
        if beta > 0:
            beta = math.pi / 2
            alpha = 0.0
            gamma = math.atan2(R[0, 1], R[1, 1])
        else:
            beta = -math.pi / 2
            alpha = 0.0
            gamma = -math.atan2(R[0, 1], R[1, 1])
    else:
        alpha = math.atan2(R[1, 0], R[0, 0])
        gamma = math.atan2(R[2, 1], R[2, 2])
    return alpha, beta, gamma


class FrankaRestDriver:
    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 34568,
        timeout_s: float = 60.0,
        step_time_s: float = _DEFAULT_REST_STEP_TIME_S,
    ):
        host = host or os.environ.get("FRANKA_REST_HOST")
        if not host:
            raise RuntimeError(
                "FrankaRestDriver needs the motion_server host (not the robot's FCI IP). "
                "Pass --rest-host or set FRANKA_REST_HOST. Repo defaults: 192.168.2.1."
            )
        self._url = f"http://{host}:{port}/api/floats"
        self._timeout_s = timeout_s
        self._session = requests.Session()
        self._step_time_s = float(step_time_s)
        self._last_grip: Optional[bool] = None

        try:
            import panda_py  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "FrankaRestDriver needs panda_py for local FK. "
                "Install panda-py on this machine (kinematics only, no FCI)."
            ) from e

    # ----- REST plumbing -----

    def _post(self, command: str, params: Sequence[float]) -> dict:
        r = self._session.post(
            self._url,
            json={command: [float(x) for x in params]},
            timeout=self._timeout_s,
        )
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {}

    # ----- state -----

    def get_state(self) -> DriverState:
        body = self._post("readJointState", [])
        arr = body.get("readJointState", [])
        if len(arr) < 8:
            raise RuntimeError(
                f"readJointState returned {arr!r}; did you rebuild motion_server "
                "with the readJointState endpoint?"
            )
        q = np.asarray(arr[:7], dtype=np.float32)
        width = float(arr[7])
        return DriverState(q=q, gripper_width=width, timestamp=time.time())

    def state_vec8(self) -> np.ndarray:
        s = self.get_state()
        return np.concatenate([s.q, [s.gripper_width]], dtype=np.float32)

    # ----- action chunk execution -----

    def _row_to_cartesian(
        self,
        q_cur: np.ndarray,
        q_target: np.ndarray,
        lock_down: bool,
    ) -> tuple[float, float, float, float, float, float]:
        import panda_py
        T_cur = panda_py.fk(np.asarray(q_cur, dtype=np.float64))
        T_tgt = panda_py.fk(np.asarray(q_target, dtype=np.float64))
        x, y, z = (float(T_tgt[0, 3]), float(T_tgt[1, 3]), float(T_tgt[2, 3]))
        if lock_down:
            return x, y, z, 0.0, 0.0, 0.0
        a_c, b_c, g_c = _zyx_euler_from_R(T_cur[:3, :3])
        a_t, b_t, g_t = _zyx_euler_from_R(T_tgt[:3, :3])
        d_alpha = math.degrees(a_t - a_c)
        d_beta = math.degrees(b_t - b_c)
        d_gamma = math.degrees(g_t - g_c)
        d_alpha = max(-90.0, min(90.0, d_alpha))
        d_beta = max(-90.0, min(90.0, d_beta))
        d_gamma = max(-90.0, min(90.0, d_gamma))
        return x, y, z, d_alpha, d_beta, d_gamma

    def _move_to_q(
        self,
        q_target: np.ndarray,
        t_sec: float,
        lock_down: bool,
    ) -> None:
        cur = self.get_state()
        x, y, z, da, db, dg = self._row_to_cartesian(cur.q.astype(np.float64), q_target, lock_down)
        self._post("moveToCartesian", [x, y, z, t_sec, da, db, dg])

    def _set_gripper(self, close: bool) -> None:
        if self._last_grip is not None and close == self._last_grip:
            return
        if close:
            self._post("closeGripper", [0.0])
        else:
            self._post("openGripper", [_GRIPPER_MAX_M])
        self._last_grip = close

    def send_chunk(
        self,
        actions: np.ndarray,
        step_dt_s: Optional[float] = None,
        grip_threshold: float = 0.5,
        lock_gripper_down: bool = False,
        # Accepted for PandaDriver-API compatibility; unused on REST path.
        substep_dt_s: float = 0.01,
        max_joint_vel_rad_s: float = 0.5,
    ) -> None:
        """Stream a (N, 8) action chunk through moveToCartesian + openGripper/closeGripper.

        Each row of `actions` is (q[0..6], grip). We FK each row locally to
        produce the server's xyz + ZYX-Euler-delta-degrees input. Gripper
        toggles are issued only when the binary state changes between rows.

        `step_dt_s` is the per-row commanded motion time. PandaDriver uses
        ~0.1 s for substep ramping; on the REST path it's the full move
        duration, so callers should pass the step-time (e.g. 2.5 s) explicitly.
        """
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"expected (N, 8), got {actions.shape}")
        if len(actions) == 0:
            return
        t_sec = float(step_dt_s if step_dt_s is not None else self._step_time_s)
        for row in actions:
            q_target = row[:7]
            close = bool(row[7] >= grip_threshold)
            self._set_gripper(close)
            self._move_to_q(q_target, t_sec=t_sec, lock_down=lock_gripper_down)

    # ----- lifecycle -----

    def home(self) -> None:
        self._move_to_q(_HOME_Q, t_sec=_DEFAULT_HOME_TIME_S, lock_down=False)
        try:
            self._post("openGripper", [_GRIPPER_MAX_M])
            self._last_grip = False
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.home()
        except Exception:
            pass
