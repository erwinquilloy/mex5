"""Shared driver exceptions.

Kept in its own module so both panda_driver and franka_rest_driver can raise
it without creating an import cycle through transport.py.
"""
from __future__ import annotations


class CollisionAborted(RuntimeError):
    """Raised when libfranka's reflex aborted the commanded motion.

    Covers contact-driven reflexes ("the gripper touched the table") and
    also velocity/acceleration-discontinuity reflexes — the common factor
    is that the in-flight motion stopped unexpectedly and a normal retry
    will fail until the robot's error state is cleared. Consumers should
    catch this, recover/home the robot, and surface a clean status.
    """
