"""REST-API driver for MolmoAct2-DROID, talking to franka/cpp/motion_server.

Drop-in alternative to PandaDriver: same public surface (home, state_vec8,
send_chunk, close), but every robot interaction goes through the C++
motion_server REST endpoints in `franka/cpp/motion_server.cpp`:

    POST /api/floats  {"moveToCartesian":  [xf, yf, zf, tf, da, db, dg]}
    POST /api/floats  {"closeGripper":     [width_m]}
    POST /api/floats  {"openGripper":      [speed_m_s]}
    POST /api/floats  {"readState":        []}      -> [x,y,z, alpha_deg, beta_deg, gamma_deg]
    POST /api/floats  {"readJointState":   []}      -> [q0..q6, gripper_width]

Server semantics that drive this design:
- `moveToCartesian` takes ABSOLUTE position (x, y, z) but DELTA orientation
  (dα, dβ, dγ in degrees) added to whatever orientation the EE is at when
  the move starts. The server clips each axis delta to ±90° silently — any
  bigger request is dropped to 0°.
- Motion time `tf` is clipped to a minimum of 0.5 s server-side.
- Cartesian velocity/acceleration discontinuities trigger a libfranka reflex
  that aborts the move; motion_server returns HTTP 400 with a
  "collision_recovery:" prefix.

The DROID policy emits ABSOLUTE joint positions. We bridge by FK-ing both
the current and target joint vectors locally with panda_py, computing the
shortest-arc Euler delta, then subdividing it across as many sequential
moveToCartesian POSTs as needed to:
  (a) keep each per-call axis delta below the server's ±90° clip, AND
  (b) keep angular/linear velocity below safe thresholds so the reflex
      doesn't fire.

panda_py.fk is a pure-kinematics call (no FCI), so it coexists fine with
the server's exclusive FCI session.

Tunable via env vars (all optional):
    FRANKA_BENCH_REST_MAX_DELTA_DEG     per-substep ceiling, default 45°
                                        (server hard cap is 90°; we leave headroom)
    FRANKA_BENCH_REST_MAX_OMEGA_DEG_S   per-axis max angular vel, default 45°/s
    FRANKA_BENCH_REST_MAX_LIN_M_S       EE linear speed cap, default 0.25 m/s
    FRANKA_BENCH_REST_MIN_STEP_TIME_S   minimum per-call tf, default 0.5 s
    FRANKA_BENCH_REST_CAM_DX_M          wrist-cam → TCP X offset in base frame,
                                        added ONLY to the last row of each
                                        action chunk (the terminal pose where
                                        alignment matters). Set to the camera's
                                        forward offset from the gripper (e.g.
                                        0.08 if the RealSense sits +8 cm along
                                        +x of the TCP). Default 0.
    FRANKA_BENCH_REST_CAM_DZ_M          same idea for Z (cam above TCP), also
                                        terminal-only. Default 0.
    FRANKA_BENCH_REST_CAM_OFFSET_MODE   when the DX/DZ shift fires:
                                        grasp_terminal (default) — last row of
                                        the grasp chunk only;
                                        every_terminal — last row of every
                                        chunk;
                                        always — every row of every chunk
                                        (use for whole-trajectory perception
                                        bias, not small geometric offsets).
    FRANKA_BENCH_REST_FAST_STEP_TIME_S  per-row motion time used when the
                                        commanded TCP Z is at/above
                                        FRANKA_BENCH_REST_SLOW_ZONE_Z_M. Free
                                        space → race through; only the default
                                        --rest-step-time-s is used near the
                                        table. Both env vars must be set to
                                        enable the two-phase behavior.
    FRANKA_BENCH_REST_SLOW_ZONE_Z_M     TCP Z (m, base frame) at and below
                                        which the slow time applies. Roughly
                                        "table height + 20 cm".
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

# Server-side hard limits, copied here so we don't post things the server will
# silently drop.
_SERVER_MAX_DELTA_DEG = 90.0
_SERVER_MIN_STEP_TIME_S = 0.5

# airscan4 lab rig workspace safe box (base-frame metres). Every commanded
# TCP X/Y gets clamped into this box before being sent to motion_server, so a
# bad cam-offset or out-of-distribution policy output can't drive the gripper
# past the rig's safe reach. Z is intentionally not clipped — the slow-zone
# logic and table contact handle Z.
_LAB_X_MIN, _LAB_X_MAX = 0.0, 0.57
_LAB_Y_MIN, _LAB_Y_MAX = -0.4, 0.4


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


def _wrap_deg(d: float) -> float:
    """Wrap a degrees delta into (-180, 180] so we take the shortest arc."""
    return (d + 180.0) % 360.0 - 180.0


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

        # Subdivision guardrails — see module docstring.
        self._max_delta_per_step_deg = min(
            _SERVER_MAX_DELTA_DEG,
            float(os.environ.get("FRANKA_BENCH_REST_MAX_DELTA_DEG", "45.0")),
        )
        self._max_omega_deg_s = float(
            os.environ.get("FRANKA_BENCH_REST_MAX_OMEGA_DEG_S", "45.0")
        )
        self._max_lin_m_s = float(
            os.environ.get("FRANKA_BENCH_REST_MAX_LIN_M_S", "0.25")
        )
        self._min_step_time_s = max(
            _SERVER_MIN_STEP_TIME_S,
            float(os.environ.get("FRANKA_BENCH_REST_MIN_STEP_TIME_S", str(_SERVER_MIN_STEP_TIME_S))),
        )

        # Wrist-cam → TCP offsets in the robot base frame. The policy reasons
        # about the scene through the camera, but commands the TCP; if the cam
        # is mounted forward/above the gripper, every target needs to be shifted
        # by the same offset so the gripper lands where the cam sees the object.
        self._cam_dx_m = float(os.environ.get("FRANKA_BENCH_REST_CAM_DX_M", "0.0"))
        self._cam_dz_m = float(os.environ.get("FRANKA_BENCH_REST_CAM_DZ_M", "0.0"))
        # When the cam offset fires:
        #   grasp_terminal (default) — last row of the grasp chunk only
        #   every_terminal           — last row of every chunk
        #   always                   — every row of every chunk
        # See README "Tuning the cam offset" for which to pick.
        _mode = os.environ.get("FRANKA_BENCH_REST_CAM_OFFSET_MODE", "grasp_terminal").strip().lower()
        if _mode not in ("grasp_terminal", "every_terminal", "always"):
            raise ValueError(
                f"FRANKA_BENCH_REST_CAM_OFFSET_MODE={_mode!r}; expected one of "
                "'grasp_terminal', 'every_terminal', 'always'."
            )
        self._cam_offset_mode = _mode

        # Two-phase approach speed. When BOTH env vars are set, rows whose FK'd
        # target TCP Z is at or above SLOW_ZONE_Z_M use FAST_STEP_TIME_S
        # instead of the default step_time_s. Idea: race the gripper through
        # free space, slow down only inside the "approach zone" near the table
        # where precision matters. Leave either var unset to keep the original
        # single-speed behavior.
        _fast = os.environ.get("FRANKA_BENCH_REST_FAST_STEP_TIME_S")
        _zone = os.environ.get("FRANKA_BENCH_REST_SLOW_ZONE_Z_M")
        self._fast_step_time_s: Optional[float] = float(_fast) if _fast else None
        self._slow_zone_z_m: Optional[float] = float(_zone) if _zone else None

        try:
            import panda_py  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "FrankaRestDriver needs panda_py for local FK. "
                "Install panda-py on this machine (kinematics only, no FCI)."
            ) from e

    # ----- REST plumbing -----

    def _post(self, command: str, params: Sequence[float]) -> dict:
        params_f = [float(x) for x in params]
        r = self._session.post(
            self._url,
            json={command: params_f},
            timeout=self._timeout_s,
        )
        if not r.ok:
            raise RuntimeError(
                f"motion_server {r.status_code} on {command}({params_f}): {r.text[:500]}"
            )
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

    # ----- planning -----

    def _plan_substeps(
        self,
        x_c: float, y_c: float, z_c: float,
        x_t: float, y_t: float, z_t: float,
        a_c: float, b_c: float, g_c: float,
        a_t: float, b_t: float, g_t: float,
        t_sec: float,
        lock_down: bool,
    ) -> list[tuple[float, float, float, float, float, float, float]]:
        """Return a list of (xf, yf, zf, tf, da, db, dg) substeps.

        - xyz interpolated linearly from current → target across n substeps.
        - Euler deltas wrapped to shortest arc and split equally across n.
        - n chosen so each substep's angular delta is ≤ max_delta_per_step_deg.
        - Total time stretched if needed to keep angular and linear velocity
          under the configured caps; per-substep tf is at least the server
          minimum.
        """
        # angular deltas (degrees, shortest arc)
        if lock_down:
            da_total = db_total = dg_total = 0.0
        else:
            da_total = _wrap_deg(math.degrees(a_t - a_c))
            db_total = _wrap_deg(math.degrees(b_t - b_c))
            dg_total = _wrap_deg(math.degrees(g_t - g_c))

        max_abs_angle = max(abs(da_total), abs(db_total), abs(dg_total))
        lin_dist = math.sqrt((x_t - x_c) ** 2 + (y_t - y_c) ** 2 + (z_t - z_c) ** 2)

        # Number of substeps so each axis stays within server's per-call ceiling.
        if max_abs_angle <= 0.0:
            n = 1
        else:
            n = max(1, math.ceil(max_abs_angle / self._max_delta_per_step_deg))

        # Velocity guardrails: stretch total time if the requested t_sec
        # would exceed the configured caps. This trades speed for safety.
        t_required_ang = max_abs_angle / self._max_omega_deg_s if self._max_omega_deg_s > 0 else 0.0
        t_required_lin = lin_dist / self._max_lin_m_s if self._max_lin_m_s > 0 else 0.0
        t_total = max(t_sec, t_required_ang, t_required_lin)

        # Per-substep time, clamped to the server's minimum.
        t_each = max(self._min_step_time_s, t_total / n)

        # Per-substep deltas.
        da = da_total / n
        db = db_total / n
        dg = dg_total / n

        substeps = []
        for k in range(1, n + 1):
            frac = k / n
            xf = x_c + (x_t - x_c) * frac
            yf = y_c + (y_t - y_c) * frac
            zf = z_c + (z_t - z_c) * frac
            substeps.append((xf, yf, zf, t_each, da, db, dg))
        return substeps

    # ----- action chunk execution -----

    def _move_to_q(
        self,
        q_target: np.ndarray,
        t_sec: float,
        lock_down: bool,
        apply_cam_offset: bool = False,
    ) -> None:
        import panda_py
        cur = self.get_state()
        T_cur = panda_py.fk(cur.q.astype(np.float64))
        T_tgt = panda_py.fk(np.asarray(q_target, dtype=np.float64))

        x_c, y_c, z_c = float(T_cur[0, 3]), float(T_cur[1, 3]), float(T_cur[2, 3])
        x_t, y_t, z_t = float(T_tgt[0, 3]), float(T_tgt[1, 3]), float(T_tgt[2, 3])
        if apply_cam_offset:
            x_t += self._cam_dx_m
            z_t += self._cam_dz_m
        x_clipped = min(_LAB_X_MAX, max(_LAB_X_MIN, x_t))
        y_clipped = min(_LAB_Y_MAX, max(_LAB_Y_MIN, y_t))
        if x_clipped != x_t or y_clipped != y_t:
            print(
                f"[FrankaRestDriver] clamped TCP target XY ({x_t:+.3f}, {y_t:+.3f}) "
                f"-> ({x_clipped:+.3f}, {y_clipped:+.3f}) m to lab box "
                f"X[{_LAB_X_MIN:.2f},{_LAB_X_MAX:.2f}] Y[{_LAB_Y_MIN:+.2f},{_LAB_Y_MAX:+.2f}]"
            )
        x_t, y_t = x_clipped, y_clipped
        a_c, b_c, g_c = _zyx_euler_from_R(T_cur[:3, :3])
        a_t, b_t, g_t = _zyx_euler_from_R(T_tgt[:3, :3])

        substeps = self._plan_substeps(
            x_c, y_c, z_c, x_t, y_t, z_t,
            a_c, b_c, g_c, a_t, b_t, g_t,
            t_sec=t_sec, lock_down=lock_down,
        )
        for xf, yf, zf, tf, da, db, dg in substeps:
            self._post("moveToCartesian", [xf, yf, zf, tf, da, db, dg])

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
        """Stream a (N, 8) action chunk through moveToCartesian + open/closeGripper.

        Each row of `actions` is (q[0..6], grip). We FK each row locally to
        produce the server's xyz + ZYX-Euler-delta-degrees input, subdividing
        any row whose orientation delta would exceed the server's ±90° per-axis
        clip or the configured angular/linear velocity caps.
        """
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"expected (N, 8), got {actions.shape}")
        if len(actions) == 0:
            return
        slow_t_sec = float(step_dt_s if step_dt_s is not None else self._step_time_s)
        last_idx = len(actions) - 1
        # Cam-offset gating depends on the configured mode. See __init__ for
        # the mode semantics. is_grasp_chunk is only consulted in
        # 'grasp_terminal' mode but compute it once either way; cheap.
        is_grasp_chunk = bool(actions[last_idx, 7] >= grip_threshold)
        two_phase = self._fast_step_time_s is not None and self._slow_zone_z_m is not None
        if two_phase:
            import panda_py
        for i, row in enumerate(actions):
            q_target = row[:7]
            close = bool(row[7] >= grip_threshold)
            self._set_gripper(close)
            t_sec = slow_t_sec
            if two_phase:
                try:
                    z_target = float(panda_py.fk(np.asarray(q_target, dtype=np.float64))[2, 3])
                    if z_target >= self._slow_zone_z_m:  # type: ignore[operator]
                        t_sec = self._fast_step_time_s  # type: ignore[assignment]
                except Exception:
                    pass
            if self._cam_offset_mode == "always":
                cam_offset = True
            elif self._cam_offset_mode == "every_terminal":
                cam_offset = (i == last_idx)
            else:  # grasp_terminal
                cam_offset = (i == last_idx and is_grasp_chunk)
            self._move_to_q(
                q_target,
                t_sec=t_sec,
                lock_down=lock_gripper_down,
                apply_cam_offset=cam_offset,
            )

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
