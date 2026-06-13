"""Unit tests for the pure receding-horizon helpers (no hardware)."""
import numpy as np

from benchmarks.benchmark.receding import (
    RecedingConfig,
    clip_blocking_target,
    find_gripper_transition,
    select_target_index,
    validate_actions,
)


def _chunk(grip_seq):
    """Build an (T, 8) chunk: arbitrary joints + the given gripper column."""
    grip = np.asarray(grip_seq, dtype=np.float64)
    joints = np.tile(np.arange(7, dtype=np.float64), (len(grip), 1))
    return np.concatenate([joints, grip[:, None]], axis=1)


# ----- find_gripper_transition -----

def test_transition_close_when_open():
    # Gripper currently open; first row >= close_thr (0.6) is the transition.
    actions = _chunk([0.0, 0.1, 0.7, 0.9])
    intent, idx = find_gripper_transition(actions, currently_closed=False,
                                          close_threshold=0.6, open_threshold=0.4)
    assert intent == "close"
    assert idx == 2


def test_transition_open_when_closed():
    # Gripper currently closed; first row <= open_thr (0.4) is the transition.
    actions = _chunk([1.0, 0.9, 0.3, 0.1])
    intent, idx = find_gripper_transition(actions, currently_closed=True,
                                          close_threshold=0.6, open_threshold=0.4)
    assert intent == "open"
    assert idx == 2


def test_no_transition_returns_none():
    # Open gripper, chunk never commands a close -> no transition.
    actions = _chunk([0.0, 0.1, 0.2, 0.3])
    intent, idx = find_gripper_transition(actions, currently_closed=False,
                                          close_threshold=0.6, open_threshold=0.4)
    assert intent is None and idx is None


def test_transition_at_row_zero():
    actions = _chunk([0.8, 0.9])
    intent, idx = find_gripper_transition(actions, currently_closed=False,
                                          close_threshold=0.6, open_threshold=0.4)
    assert intent == "close" and idx == 0


# ----- select_target_index -----

def test_select_target_no_transition_uses_tail():
    actions = _chunk([0.0, 0.0, 0.0])
    assert select_target_index(actions, None) == 2


def test_select_target_is_transition_row():
    # Target the grasp row itself (Deo), not one waypoint short.
    actions = _chunk([0.0, 0.0, 0.7])
    assert select_target_index(actions, 2) == 2


def test_select_target_transition_at_zero():
    actions = _chunk([0.8, 0.9])
    assert select_target_index(actions, 0) == 0


# ----- clip_blocking_target -----

def test_clip_limits_large_delta():
    current = np.zeros(7)
    target = np.full(7, 1.0)  # 1.0 rad jump, exceeds max_dq
    out = clip_blocking_target(current, target, max_dq=0.2)
    assert np.allclose(out, 0.2)


def test_clip_passes_small_delta():
    current = np.zeros(7)
    target = np.full(7, 0.05)
    out = clip_blocking_target(current, target, max_dq=0.2)
    assert np.allclose(out, 0.05)


def test_clip_mixed_signs():
    current = np.array([0.0, 1.0])
    target = np.array([0.5, 0.0])  # +0.5 clipped to +0.2; -1.0 clipped to -0.2
    out = clip_blocking_target(current, target, max_dq=0.2)
    assert np.allclose(out, [0.2, 0.8])


# ----- validate_actions -----

def test_validate_rejects_bad_shape():
    import pytest
    with pytest.raises(ValueError):
        validate_actions(np.zeros((0, 8)))
    with pytest.raises(ValueError):
        validate_actions(np.zeros((3, 7)))


def test_validate_rejects_non_finite():
    import pytest
    bad = _chunk([0.0, 0.0])
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_actions(bad)


# ----- config -----

def test_config_env_override(monkeypatch):
    monkeypatch.setenv("FRANKA_BENCH_RH_MAX_STEP_DELTA", "0.05")
    monkeypatch.setenv("FRANKA_BENCH_RH_EXEC_TIME_S", "2.0")
    monkeypatch.setenv("FRANKA_BENCH_RH_GRASP_ON_CONTACT", "0")
    c = RecedingConfig.from_env()
    assert c.max_step_delta == 0.05
    assert c.exec_time_s == 2.0
    assert c.grasp_on_contact is False
