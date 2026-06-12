"""MolmoAct2 paper Table 6 task suite (real-world DROID embodiment eval).

Source: Table 6 of arXiv:2605.02881. Each task is evaluated over 15 trials with
randomly initialized camera poses and unseen objects in OOD scenes. Reference
success rates for MolmoAct2-DROID are recorded here as the replication target.

NOTE: Faithful replication requires a separately mounted external camera; this
rig has only a wrist camera. Numbers will deviate from the paper. See
`benchmarks/README.md` for the camera caveat.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DroidTask:
    task_id: str
    instruction: str            # exact language passed to MolmoAct2
    setup_notes: str            # human-readable scene setup for the operator
    paper_success_rate: float   # MolmoAct2-DROID reference % from Table 6
    trials: int = 15
    max_chunks: int = 30        # safety cap on action chunks per trial
    # Base-frame XY box of the placement target (xmin, xmax, ymin, ymax in metres).
    # Used ONLY when run_droid_benchmark is invoked with --hold-until-target, in
    # which case the runner suppresses any policy-commanded gripper-open whose
    # commanded TCP XY falls outside this box (while an object is detected in
    # the jaws). Leave None for tasks where you haven't measured the target
    # location yet; with --hold-until-target the runner will error rather than
    # guess.
    target_zone_xy: Optional[tuple[float, float, float, float]] = None


# Reference numbers: MolmoAct2-DROID column of Table 6.
TASKS: list[DroidTask] = [
    DroidTask(
        task_id="apple_on_plate",
        instruction="Put the apple on the plate.",
        setup_notes=(
            "Place a real apple in the workspace and an empty plate within reach. "
            "Use objects not seen during training. Randomize the external camera "
            "pose vs. its position at any earlier trial."
        ),
        paper_success_rate=100.0,
    ),
    DroidTask(
        task_id="pipette_in_tray",
        instruction="Put the marker in the tray.",
        setup_notes=(
            "Marker on the table (stand-in for the pipette), tray on the side. "
            "OOD marker color/shape preferred. Re-randomize cam pose."
        ),
        paper_success_rate=86.7,
    ),
    DroidTask(
        task_id="red_cube_in_tape_roll",
        instruction="Put the red cube inside the tape roll.",
        setup_notes=(
            "Red cube ~2-3cm; a roll of tape lying flat. Re-randomize cam pose."
        ),
        paper_success_rate=93.3,
    ),
    DroidTask(
        task_id="knife_in_box",
        instruction="Put the knife in the box.",
        setup_notes=(
            "Plastic / safe knife on the table, open box within reach. "
            "Re-randomize cam pose."
        ),
        paper_success_rate=93.3,
    ),
    DroidTask(
        task_id="objects_in_bowl",
        instruction="Put the objects in the bowl.",
        setup_notes=(
            "Multiple small objects scattered around a bowl. Counted as success "
            "only when all visible target objects are in the bowl. "
            "Re-randomize cam pose."
        ),
        paper_success_rate=62.0,
    ),
]


def by_id(task_id: str) -> DroidTask:
    for t in TASKS:
        if t.task_id == task_id:
            return t
    raise KeyError(task_id)


def all_tasks() -> list[DroidTask]:
    return list(TASKS)
