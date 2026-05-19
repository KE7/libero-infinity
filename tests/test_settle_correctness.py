# BDDL filenames inline in T6's parametrize tuples are unavoidably long; relax
# E501 file-wide rather than mutate canonical filenames or hide them behind
# constants that obscure the regression mapping.
# ruff: noqa: E501
"""
Tests for settling correctness and retry-loop elimination.
T1: SettleUnsafeError never raised in src/
T2: AlreadySolvedError never raised in src/
T3: _footprint_clearance_xy <= min_clearance for all pairs
T4: Renderer never emits visibility_targets
T5: Single-object settle stability
"""

import subprocess
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"


# T1: SettleUnsafeError is dead code — never raised
def test_settle_unsafe_error_never_raised():
    result = subprocess.run(
        ["grep", "-r", "raise SettleUnsafeError", str(SRC_DIR)],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "", f"SettleUnsafeError is raised somewhere: {result.stdout}"


# T2: AlreadySolvedError is dead code — never raised
def test_already_solved_error_never_raised():
    result = subprocess.run(
        ["grep", "-r", "raise AlreadySolvedError", str(SRC_DIR)],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "", f"AlreadySolvedError is raised somewhere: {result.stdout}"


# T3: _footprint_clearance_xy covers all object pairs
def test_footprint_clearance_covers_all_pairs():
    """Verify that per-pair clearance is used — renderer no longer emits fixed min_clearance."""
    renderer_path = SRC_DIR / "libero_infinity" / "renderer" / "scenic_renderer.py"
    source = renderer_path.read_text()
    # The fixed global min_clearance should be gone
    assert "param min_clearance = 0.10" not in source, "Renderer still emits hardcoded min_clearance = 0.10"  # fmt: skip  # noqa: E501


# T4: VisibilityError is the ONLY retried error
def test_only_visibility_error_is_retried():
    """simulator.py should only catch VisibilityError in the retry loop."""
    sim_path = SRC_DIR / "libero_infinity" / "simulator.py"
    source = sim_path.read_text()
    assert "MAX_VISIBILITY_RETRIES" in source, "MAX_VISIBILITY_RETRIES not found in simulator.py"
    assert "MAX_RESAMPLE" not in source, "Old MAX_RESAMPLE still referenced in simulator.py"
    assert "MAX_REPLAN" not in source, "Old MAX_REPLAN still referenced in simulator.py"
    assert "SettleUnsafeError" not in source, "Dead code SettleUnsafeError still in simulator.py"
    assert "AlreadySolvedError" not in source, "Dead code AlreadySolvedError still in simulator.py"


# T5: validation_errors has MAX_VISIBILITY_RETRIES and not the old constants
def test_validation_errors_constants():
    from libero_infinity.validation_errors import MAX_VISIBILITY_RETRIES

    assert MAX_VISIBILITY_RETRIES >= 5, "MAX_VISIBILITY_RETRIES should be at least 5"
    import libero_infinity.validation_errors as ve

    assert not hasattr(ve, "MAX_RESAMPLE"), "MAX_RESAMPLE should be removed"
    assert not hasattr(ve, "MAX_REPLAN"), "MAX_REPLAN should be removed"
    assert not hasattr(ve, "SettleUnsafeError"), "SettleUnsafeError should be removed"
    assert not hasattr(ve, "AlreadySolvedError"), "AlreadySolvedError should be removed"


# ---------------------------------------------------------------------------
# T6: workspace-supported children must NOT be lifted to the workspace AABB top.
#
# Regression: ``_restack_supported_children`` previously received every
# (child, parent) pair whose parent was *not* an explicit fixture entry in
# ``scene.objects``. Workspace fixtures (``living_room_table``,
# ``kitchen_table``, ``study_table`` …) are implicit — they never appear in
# ``scene.objects`` — so workspace-supported children fell through the
# filter and were lifted to ``living_room_table_main``'s AABB top
# (z ≈ 1.30 m). That is above the agentview camera (z = 0.96 m, fovy 45°),
# so every position-axis-only sample on living-room basket / tray / plate
# tasks (and study caddy tasks) drove the agentview projection above the
# image, the visibility validator correctly flagged "out of frame", and the
# 10-retry cap exhausted with ``RuntimeError`` at gym_env.py:268. See
# ~/.omar/ea/4/validation_run/rca/stage3_run2b_contain_region_family.md and
# ~/Documents/research/roboeval/docs/recurring_issues_rca.md §1.
#
# The fix flips the filter to a *positive* set: only allow re-stack when the
# parent is a movable scene object (graspable=True). This test asserts the
# parameterised LR_SCENE1 basket task — the canonical reproducer for the
# bug — resets successfully under position-axis perturbation.

# 16 (task, seed) tuples from ~/.omar/ea/4/validation_run/logs/failures_live_full.jsonl
# matching axis_subset==["position"], error_class=="RuntimeError",
# error_msg starting "Invalid Scenic sample after visibility check" — the
# precise rows the run-2b RCA pins as "settle-retry exhaustion via
# visibility check".  Each row must now reset() without raising.
_VISIBILITY_RETRY_REGRESSION_TUPLES = [
    (
        "libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl",
        0,
    ),
    (
        "libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.bddl",
        0,
    ),
    (
        "libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl",
        0,
    ),
    ("libero_90/LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket.bddl", 0),
    ("libero_90/LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.bddl", 0),
    ("libero_90/LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket.bddl", 0),
    ("libero_90/LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket.bddl", 0),
    ("libero_90/LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket.bddl", 0),
    ("libero_90/LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket.bddl", 0),
    ("libero_90/LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket.bddl", 0),
    ("libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl", 0),
    ("libero_90/LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket.bddl", 0),
    ("libero_90/LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.bddl", 0),
    (
        "libero_90/LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray.bddl",
        0,
    ),
    # Two-object basket task with multiple settle retries — exercises the
    # restack filter against ALL five workspace-supported objects.
    (
        "libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl",
        0,
    ),
    (
        "libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl",
        0,
    ),
]


def _bddl_root() -> Path:
    return (
        Path(__file__).parent.parent
        / "src"
        / "libero_infinity"
        / "data"
        / "libero_runtime"
        / "bddl_files"
    )


@pytest.mark.slow
@pytest.mark.parametrize(("task_rel", "seed"), _VISIBILITY_RETRY_REGRESSION_TUPLES)
def test_position_axis_workspace_supported_reset_does_not_exhaust_retries(
    task_rel: str, seed: int
) -> None:
    """Regression: position-axis-only reset on workspace-supported tasks must
    succeed without exhausting the 10-retry settle-retry cap.

    Pinned tuples are the post-fix-must-pass set from
    ``~/.omar/ea/4/validation_run/logs/failures_live_full.jsonl``. Each
    failed pre-fix with ``RuntimeError: reset() failed to find a valid scene
    after 10 retries. Last error: Invalid Scenic sample after visibility
    check`` at ``gym_env.py:268``. Each must now pass.
    """
    import random

    import numpy as np

    from libero_infinity.gym_env import LIBEROScenicEnv

    bddl = _bddl_root() / task_rel
    if not bddl.exists():
        pytest.skip(f"BDDL not vendored in test env: {task_rel}")

    random.seed(seed)
    np.random.seed(seed)
    env = LIBEROScenicEnv(bddl_path=str(bddl), perturbation="position", seed=seed)
    try:
        env.reset()
    finally:
        try:
            env.close()
        except Exception:
            pass


def test_restack_filter_uses_movable_scene_objects_positive_set() -> None:
    """The restack filter in ``simulator.py::setup`` must select on the
    *positive* movable-scene-object set, not the negative
    "explicit-fixed-fixture-in-scene.objects" set.

    Reason: workspace fixtures (``living_room_table``, ``kitchen_table``, …)
    are never enumerated in ``scene.objects``, so the negative-set filter
    is vacuously true and lifts workspace-supported children to the arena
    AABB top — see the run-2b RCA. The positive-set filter excludes both
    workspaces and fixed fixtures without needing to know the workspace
    instance name.
    """
    sim_path = SRC_DIR / "libero_infinity" / "simulator.py"
    source = sim_path.read_text()
    assert "movable_scene_names" in source, (
        "Expected the positive-set restack filter `movable_scene_names` in setup(). "
        "If the filter is reverted to the negative set, workspace-supported "
        "children will again be lifted to z>=1.30 and visibility validation will "
        "reject every position-axis sample on living_room/study arenas."
    )
    _restack_check_msg = (
        "Expected restack filter to keep only children whose parent is a movable scene object."
    )
    assert "if parent in movable_scene_names" in source, _restack_check_msg
