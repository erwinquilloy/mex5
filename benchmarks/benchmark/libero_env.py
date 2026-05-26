"""LIBERO sim env wrapper.

LIBERO ships task suites (libero_spatial, libero_object, libero_goal, libero_10,
libero_90) backed by robosuite/MuJoCo with a Franka Panda. Each task carries an
initial-state set and a built-in `check_success()`, so we get automated success
scoring without a human in the loop.

This wrapper:
  - resolves a task by (suite, task_index) and returns its language instruction
  - constructs an OffScreenRenderEnv (RGB only, no GUI)
  - exposes reset(init_idx), step(action), render() -> PIL.Image, success()

Install (see README): `pip install -e .` of LIBERO with robosuite + mujoco.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


@dataclass
class LiberoTaskSpec:
    suite: str
    task_index: int
    task_id: str            # "<suite>/<task_index>"
    instruction: str        # language goal
    bddl_file: str
    n_init_states: int


class LiberoEnv:
    """Single-task sim env, RGB-only obs, 7-DoF delta-pose + grip action."""

    def __init__(
        self,
        suite: str,
        task_index: int,
        camera_height: int = 256,
        camera_width: int = 256,
        camera_name: str = "agentview",
    ):
        if suite not in SUITES:
            raise ValueError(f"suite must be one of {SUITES}, got {suite!r}")
        # Imported lazily so the real-robot path doesn't require LIBERO/MuJoCo.
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite_cls = benchmark.get_benchmark_dict()[suite]
        self._suite = suite_cls()
        if task_index >= self._suite.n_tasks:
            raise IndexError(f"{suite} has {self._suite.n_tasks} tasks, got {task_index}")

        task = self._suite.get_task(task_index)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = self._suite.get_task_init_states(task_id=task_index)

        self.spec = LiberoTaskSpec(
            suite=suite,
            task_index=task_index,
            task_id=f"{suite}/{task_index}",
            instruction=task.language,
            bddl_file=bddl,
            n_init_states=len(init_states),
        )
        self._init_states = init_states
        self._camera = camera_name
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=camera_height,
            camera_widths=camera_width,
            camera_names=[camera_name],
        )
        self._last_obs: dict | None = None

    # ----- lifecycle -----

    def reset(self, init_index: int = 0, seed: Optional[int] = None) -> Image.Image:
        if seed is not None:
            self._env.seed(seed)
        self._env.reset()
        self._env.set_init_state(self._init_states[init_index % self.spec.n_init_states])
        # robosuite needs a few no-op steps for the controller to settle.
        for _ in range(5):
            obs, _, _, _ = self._env.step(np.zeros(7, dtype=np.float32))
        self._last_obs = obs
        return self.render()

    def step(self, action) -> tuple[Image.Image, float, bool, dict]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 7:
            raise ValueError(f"action must be 7-dim, got {a.shape}")
        obs, reward, done, info = self._env.step(a)
        self._last_obs = obs
        return self.render(), float(reward), bool(done), dict(info)

    def render(self) -> Image.Image:
        key = f"{self._camera}_image"
        img = self._last_obs[key]
        # robosuite returns frames upside-down relative to the OpenGL convention.
        return Image.fromarray(np.ascontiguousarray(img[::-1]), mode="RGB")

    def success(self) -> bool:
        try:
            return bool(self._env.check_success())
        except Exception:
            return False

    def proprio(self) -> Optional[list[float]]:
        if self._last_obs is None:
            return None
        # Try common robosuite key names; return None if unavailable.
        for k in ("robot0_eef_pos", "robot0_proprio-state"):
            if k in self._last_obs:
                return np.asarray(self._last_obs[k], dtype=np.float32).tolist()
        return None

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass


def list_suite_tasks(suite: str) -> list[tuple[int, str]]:
    """Return [(task_index, language_instruction), ...] for a LIBERO suite."""
    from libero.libero import benchmark
    s = benchmark.get_benchmark_dict()[suite]()
    return [(i, s.get_task(i).language) for i in range(s.n_tasks)]
