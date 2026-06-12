"""Manual jog + TCP read-out helper for cam-offset calibration.

Talks to a running ``motion_server`` over the same REST transport the dashboard
uses (FrankaRestDriver -> http://$FRANKA_REST_HOST:34568/api/floats). It does NOT
own the cameras or the policy, so it can run alongside the dashboard -- but do
NOT use it while the dashboard is executing a trial, or the two will fight over
the arm. Stop/idle the dashboard first; keep motion_server running.

Subcommands
-----------
  read                 print the live TCP pose (XYZ m, RPY deg) + gripper width
  goto X Y Z           move the TCP to an absolute base-frame coordinate,
                       keeping the current orientation fixed
  open | close         open / close the gripper

Cam-offset calibration workflow
-------------------------------
  1. Start motion_server (dashboard launcher panel is fine). Idle the dashboard.
  2. Find a safe spot:    python -m benchmarks.scripts.calib_jog read
  3. Hover above target:  python -m benchmarks.scripts.calib_jog goto 0.45 0.0 0.10
  4. Descend to grasp Z:  python -m benchmarks.scripts.calib_jog goto 0.45 0.0 0.016
     -> place the object centered in the open jaws. Now (0.45, 0.0, 0.016) is
        your KNOWN object location.
  5. In the dashboard, zero DX/DY/DZ (apply), Home, and Run the grasp.
  6. Read the policy's intended grasp from the dashboard console line:
        [cam-calib] grasp-terminal target: policy=(x, y, z) ...
     delta = policy - known_object_xyz. Dial DX/DY/DZ = -delta into the panel.
  7. Re-run to confirm; fine-tune once.

Safety: X/Y are clamped to the rig's lab box and Z to the table floor (same
limits the driver enforces). ``goto`` prints the planned move and asks for
confirmation unless ``--yes`` is passed.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from benchmarks.benchmark.franka_rest_driver import (
    FrankaRestDriver,
    _GRIPPER_MAX_M,
    _LAB_X_MAX,
    _LAB_X_MIN,
    _LAB_Y_MAX,
    _LAB_Y_MIN,
    _LAB_Z_MIN,
    _zyx_euler_from_R,
)


def _tcp_pose(driver: FrankaRestDriver):
    """Return (x, y, z, roll, pitch, yaw, gripper_width_m) for the current state.
    XYZ in metres, angles in radians (ZYX Euler), via local FK of the joints."""
    import panda_py

    st = driver.get_state()
    T = panda_py.fk(st.q.astype(np.float64))
    x, y, z = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
    a, b, g = _zyx_euler_from_R(T[:3, :3])
    return x, y, z, a, b, g, st.gripper_width


def _print_pose(driver: FrankaRestDriver, prefix: str = "") -> None:
    x, y, z, a, b, g, w = _tcp_pose(driver)
    print(
        f"{prefix}TCP xyz=({x:+.4f}, {y:+.4f}, {z:+.4f}) m  "
        f"rpy=({math.degrees(a):+.1f}, {math.degrees(b):+.1f}, {math.degrees(g):+.1f}) deg  "
        f"gripper={w * 1000:.1f} mm"
    )


def _goto(driver: FrankaRestDriver, x: float, y: float, z: float,
          t_sec: float, assume_yes: bool) -> int:
    # Clamp into the same lab box the driver enforces, so we never command the
    # arm somewhere motion_server would reject (or somewhere unsafe).
    xc = min(_LAB_X_MAX, max(_LAB_X_MIN, x))
    yc = min(_LAB_Y_MAX, max(_LAB_Y_MIN, y))
    zc = max(_LAB_Z_MIN, z)
    if (xc, yc, zc) != (x, y, z):
        print(
            f"[calib_jog] target ({x:+.4f}, {y:+.4f}, {z:+.4f}) clamped to lab box "
            f"-> ({xc:+.4f}, {yc:+.4f}, {zc:+.4f}) m "
            f"[X({_LAB_X_MIN:.2f},{_LAB_X_MAX:.2f}) Y({_LAB_Y_MIN:+.2f},{_LAB_Y_MAX:+.2f}) "
            f"Z>={_LAB_Z_MIN:.3f}]"
        )

    cx, cy, cz, ca, cb, cg, _ = _tcp_pose(driver)
    print(f"[calib_jog] current TCP: ({cx:+.4f}, {cy:+.4f}, {cz:+.4f}) m")
    print(f"[calib_jog] move to:     ({xc:+.4f}, {yc:+.4f}, {zc:+.4f}) m "
          f"over {t_sec:.1f}s, orientation locked")
    if not assume_yes:
        try:
            if input("[calib_jog] WARNING: this will MOVE the robot. Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("[calib_jog] aborted.")
                return 1
        except EOFError:
            print("[calib_jog] no tty for confirmation; pass --yes to run non-interactively.")
            return 1

    # lock_down=True zeroes the orientation deltas, so we translate only and the
    # wrist stays at its current pose. Euler args are irrelevant under lock_down.
    substeps = driver._plan_substeps(
        cx, cy, cz, xc, yc, zc,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        t_sec=t_sec, lock_down=True,
    )
    for xf, yf, zf, tf, da, db, dg in substeps:
        driver._post("moveToCartesian", [xf, yf, zf, tf, da, db, dg])
    _print_pose(driver, prefix="[calib_jog] arrived: ")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("FRANKA_REST_HOST"),
                   help="motion_server host (default: $FRANKA_REST_HOST)")
    p.add_argument("--port", type=int, default=34568, help="motion_server REST port")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("read", help="print the live TCP pose + gripper width")

    g = sub.add_parser("goto", help="move TCP to an absolute base-frame XYZ")
    g.add_argument("x", type=float)
    g.add_argument("y", type=float)
    g.add_argument("z", type=float)
    g.add_argument("--t", type=float, default=4.0, help="move duration seconds (default 4)")
    g.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    sub.add_parser("open", help="open the gripper")
    sub.add_parser("close", help="close the gripper")

    args = p.parse_args(argv)
    if not args.host:
        p.error("no motion_server host: pass --host or export FRANKA_REST_HOST")

    driver = FrankaRestDriver(host=args.host, port=args.port)

    if args.cmd == "read":
        _print_pose(driver)
        return 0
    if args.cmd == "goto":
        return _goto(driver, args.x, args.y, args.z, args.t, args.yes)
    if args.cmd == "open":
        driver._post("openGripper", [_GRIPPER_MAX_M])
        print("[calib_jog] gripper opened.")
        return 0
    if args.cmd == "close":
        driver._post("closeGripper", [0.0])
        print("[calib_jog] gripper closed.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
