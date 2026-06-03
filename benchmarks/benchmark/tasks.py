"""Legacy task suite for the cartesian-delta runner (benchmarks/runner.py).

Two task families on physical hardware:
  - spatial : same objects, different spatial relations
  - goal    : same objects/positions, different goals

Each Task carries the natural-language instruction passed to MolmoAct2, plus
a `verify` callback the human operator answers via the CLI (success/fail).

Object/landmark coordinates are placeholders -- tune to your tabletop layout
(meters, robot base frame) before running.

The primary DROID Table 6 suite lives in droid_tasks.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

# Tabletop home pose (close to the safe start above the workspace center).
HOME_XYZ: tuple[float, float, float] = (0.40, 0.00, 0.30)
MAX_STEPS = 40                  # per trial


@dataclass
class Task:
    task_id: str
    suite: str                  # "spatial" | "goal"
    instruction: str            # the language prompt sent to MolmoAct2
    setup_notes: str            # human-readable scene setup
    home_xyz: tuple[float, float, float] = HOME_XYZ
    max_steps: int = MAX_STEPS


# --- Spatial (same objects, different spatial relations) -----------------
SPATIAL: list[Task] = [
    Task("spatial_red_left",  "spatial",
         "pick up the red block on the left of the table",
         "Place red block ~15cm left of center, blue block on the right."),
    Task("spatial_red_right", "spatial",
         "pick up the red block on the right of the table",
         "Swap red/blue positions from the previous trial."),
    Task("spatial_red_front", "spatial",
         "pick up the red block closest to the robot",
         "Place red block near front edge of workspace, blue block at back."),
    Task("spatial_red_back",  "spatial",
         "pick up the red block farthest from the robot",
         "Mirror of spatial_red_front."),
]

# --- Goal (same scene, different goals) ----------------------------------
GOAL: list[Task] = [
    Task("goal_block_in_bin", "goal",
         "put the red block in the white bin",
         "Red block center, white bin on the right."),
    Task("goal_block_on_plate", "goal",
         "place the red block on the blue plate",
         "Red block center, blue plate on the left."),
]


def all_tasks() -> list[Task]:
    return SPATIAL + GOAL


def by_id(task_id: str) -> Task:
    for t in all_tasks():
        if t.task_id == task_id:
            return t
    raise KeyError(task_id)
