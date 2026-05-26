"""LIBERO sim env wrapper, tuned to MolmoAct2-LIBERO's input contract.

MolmoAct2-LIBERO expects:
  - images: [agentview_rgb, wrist_rgb]            (PIL or HxWx3 uint8)
  - state:  float32(8,) = [eef_xyz(3), eef_axis_angle(3), gripper_qpos(2)]
  - actions returned: LIBERO-scale (already denormalized) -- pass straight to env.step

LIBERO ships task suites (libero_spatial, libero_object, libero_goal,
libero_10, libero_90) backed by robosuite/MuJoCo with a Franka Panda. Each
task carries an initial-state set and a built-in `check_success()`, so we get
automated success scoring without a human in the loop.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
AGENT_CAM = "agentview"
WRIST_CAM = "robot0_eye_in_hand"


@dataclass
class LiberoTaskSpec:
    suite: str
    task_index: int
    task_id: str            # "<suite>/<task_index>"
    instruction: str        # language goal
    bddl_file: str
    n_init_states: int


@dataclass
class LiberoObs:
    agentview: np.ndarray   # (H, W, 3) uint8 RGB
    wrist:     np.ndarray   # (H, W, 3) uint8 RGB
    state8:    np.ndarray   # (8,) float32, MolmoAct2-LIBERO state convention


def _quat_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    """robosuite returns quaternions in (x, y, z, w) order. Return a 3-vector
    whose direction is the rotation axis and magnitude is the rotation angle."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    x, y, z, w = q
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return np.array([x / s, y / s, z / s], dtype=np.float32) * float(angle)


class LiberoEnv:
    """Dual-camera LIBERO env (agentview + wrist) for MolmoAct2-LIBERO."""

    def __init__(
        self,
        suite: str,
        task_index: int,
        camera_height: int = 256,
        camera_width: int = 256,
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
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=camera_height,
            camera_widths=camera_width,
            camera_names=[AGENT_CAM, WRIST_CAM],
        )
        self._last_obs: dict | None = None

    # ----- lifecycle -----

    def reset(self, init_index: int = 0, seed: Optional[int] = None) -> LiberoObs:
        if seed is not None:
            self._env.seed(seed)
        self._env.reset()
        self._env.set_init_state(self._init_states[init_index % self.spec.n_init_states])
        # robosuite needs a few no-op steps for the controller to settle.
        for _ in range(5):
            obs, _, _, _ = self._env.step(np.zeros(7, dtype=np.float32))
        self._last_obs = obs
        return self.observe()

    def step(self, action) -> tuple[LiberoObs, float, bool, dict]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 7:
            raise ValueError(f"action must be 7-dim, got {a.shape}")
        obs, reward, done, info = self._env.step(a)
        self._last_obs = obs
        return self.observe(), float(reward), bool(done), dict(info)

    def observe(self) -> LiberoObs:
        obs = self._last_obs or {}
        # robosuite returns frames upside-down relative to the OpenGL convention.
        agent = np.ascontiguousarray(obs[f"{AGENT_CAM}_image"][::-1])
        wrist = np.ascontiguousarray(obs[f"{WRIST_CAM}_image"][::-1])
        eef_pos = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
        eef_quat = np.asarray(obs.get("robot0_eef_quat", np.array([0, 0, 0, 1])), dtype=np.float32)
        grip_qpos = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)
        state8 = np.concatenate(
            [eef_pos, _quat_to_axis_angle(eef_quat), grip_qpos]
        ).astype(np.float32)
        if state8.shape[0] != 8:
            raise RuntimeError(f"state8 wrong shape: {state8.shape}")
        return LiberoObs(agentview=agent, wrist=wrist, state8=state8)

    def success(self) -> bool:
        try:
            return bool(self._env.check_success())
        except Exception:
            return False

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
