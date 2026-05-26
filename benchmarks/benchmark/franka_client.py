"""Thin wrapper around the motion_server REST API with timing hooks."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import requests


@dataclass
class FrankaResult:
    ok: bool
    status: int
    body: dict
    rest_ms: float           # POST round-trip
    motion_ms: float         # commanded motion duration (echoed back)


class FrankaClient:
    """Speak to the C++ motion_server at <ip>:<port>/api/floats."""

    def __init__(self, ip: str = "192.168.2.1", port: int = 34568, timeout_s: float = 30.0):
        self.url = f"http://{ip}:{port}/api/floats"
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def _post(self, command: str, params: Sequence[float]) -> FrankaResult:
        t0 = time.perf_counter()
        r = self._session.post(self.url, json={command: list(params)}, timeout=self.timeout_s)
        dt = (time.perf_counter() - t0) * 1000.0
        body: dict = {}
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text[:256]}
        commanded_time = float(params[3]) if command == "moveToCartesian" and len(params) >= 4 else 0.0
        return FrankaResult(ok=r.ok, status=r.status_code, body=body, rest_ms=dt, motion_ms=commanded_time * 1000.0)

    # ----- high-level helpers -----

    def move_cartesian(
        self,
        x: float, y: float, z: float,
        t_sec: float = 5.0,
        dyaw_deg: float = 0.0, dpitch_deg: float = 0.0, droll_deg: float = 0.0,
    ) -> FrankaResult:
        return self._post("moveToCartesian", [x, y, z, t_sec, dyaw_deg, dpitch_deg, droll_deg])

    def open_gripper(self, width: float = 0.08) -> FrankaResult:
        return self._post("openGripper", [width])

    def close_gripper(self, width: float = 0.0) -> FrankaResult:
        return self._post("closeGripper", [width])

    def apply_delta(
        self,
        current_xyz: Sequence[float],
        delta_xyz_rpy_grip: Sequence[float],
        t_sec: float = 1.0,
        grip_threshold: float = 0.5,
    ) -> FrankaResult:
        """Apply a MolmoAct-style 7-vector [dx,dy,dz,droll,dpitch,dyaw,grip] as one motion step.

        - Cartesian delta is added to current_xyz; rotations are sent as degrees-delta
          (the motion_server already speaks delta-degrees per the README).
        - grip > grip_threshold => close, else open. Gripper command is fire-and-forget
          before the cartesian move so they sequence cleanly.
        """
        dx, dy, dz, droll, dpitch, dyaw, grip = list(delta_xyz_rpy_grip)
        if grip > grip_threshold:
            self.close_gripper()
        else:
            self.open_gripper()
        x, y, z = current_xyz
        return self.move_cartesian(
            x + dx, y + dy, z + dz, t_sec,
            dyaw_deg=dyaw, dpitch_deg=dpitch, droll_deg=droll,
        )
