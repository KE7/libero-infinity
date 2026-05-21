"""Unit tests for the G4-A identity adapter (``_scene_view.SceneView``).

These tests verify that ``identity.py`` reports correct booleans when handed
Scenic-shaped scenes (only ``.objects`` and ``.params``), not the richer IR
the test fixtures in ``test_invariants_identity.py`` use.

We build "Scenic-shaped" scenes via :class:`_FakeScenicScene` — a stand-in for
``scenic.core.scenarios.Scene`` that exposes the exact attribute surface the
real Scenic Scene exposes (``.objects`` tuple + ``.params`` mapping), and
populate ``.objects`` with stand-ins for ``LIBEROObject`` / ``LIBEROFixture``
that match the renderer's emitted properties (``libero_name``, ``position``,
``asset_class``).

Both positive (identity preserved on inactive axes) and negative (axis ACTIVE
on perturbed → corresponding field differs) cases are covered.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from libero_infinity.validation.invariants import g4_identity_hook
from libero_infinity.validation.invariants._scene_view import (
    SceneView,
    wrap_scene,
)
from libero_infinity.validation.invariants.identity import (
    AXES,
    assert_articulation_unchanged,
    assert_background_unchanged,
    assert_camera_unchanged,
    assert_distractor_unchanged,
    assert_lighting_unchanged,
    assert_object_unchanged,
    assert_position_unchanged,
    assert_robot_unchanged,
    assert_texture_unchanged,
)

# ---------------------------------------------------------------------------
# Scenic-shaped fixtures
# ---------------------------------------------------------------------------


class _LIBEROObject:
    """Stand-in for a sampled Scenic ``LIBEROObject`` (or fixture / distractor)."""

    def __init__(self, libero_name, position=None, asset_class=None, is_fixture=False):
        self.libero_name = libero_name
        self.position = position
        self.asset_class = asset_class
        # Spoof Scenic's class-name surface so ``_scenic_class_name`` returns
        # the right tag for fixtures vs regular objects.
        self.__class__.__name__ = "LIBEROFixture" if is_fixture else "LIBEROObject"


class _LIBEROFixture(_LIBEROObject):
    def __init__(self, libero_name, position=None, asset_class=None):
        super().__init__(libero_name, position, asset_class, is_fixture=True)


class _FakeScenicScene:
    """Mirrors ``scenic.core.scenarios.Scene``: only ``.objects`` and ``.params``."""

    def __init__(self, objects=(), params=None):
        self.objects = tuple(objects)
        self.params = dict(params or {})


def _baseline_scenic_scene():
    """No-axes baseline: canonical positions, no per-axis params emitted."""
    return _FakeScenicScene(
        objects=[
            _LIBEROFixture("kitchen_table", position=(0.0, 0.0, 0.8)),
            _LIBEROFixture("wooden_cabinet_1", position=(0.5, 0.2, 0.9)),
            _LIBEROObject("bowl", position=(0.1, 0.2, 0.85)),
            _LIBEROObject("cup", position=(0.3, 0.15, 0.85)),
        ],
        params={
            "task": "demo_task",
            "active_axes": "",
        },
    )


# ---------------------------------------------------------------------------
# Adapter — basic shape
# ---------------------------------------------------------------------------


def test_wrap_scene_is_idempotent_on_rich_scene():
    rich = SimpleNamespace(objects=[], fixtures=[], lights=[], cameras=[])
    assert wrap_scene(rich) is rich


def test_wrap_scene_wraps_scenic_scene():
    s = _baseline_scenic_scene()
    view = wrap_scene(s)
    assert isinstance(view, SceneView)
    # objects/fixtures/distractors are now partitioned
    assert {o.name for o in view.objects} == {"bowl", "cup"}
    assert {f.name for f in view.fixtures} == {"kitchen_table", "wooden_cabinet_1"}
    assert view.distractors == ()
    # Param-derived axes return stable canonical sentinels when inactive.
    assert view.lights == ()
    assert view.cameras == ()
    assert view.robot.name == "__inactive__"
    assert view.background.name == "__inactive__"


def test_distractor_partitioning():
    s = _FakeScenicScene(
        objects=[
            _LIBEROObject("bowl", position=(0, 0, 0)),
            _LIBEROObject("distractor_0", position=(1, 1, 1), asset_class="apple"),
            _LIBEROObject("distractor_1", position=(2, 2, 2), asset_class="book"),
        ],
        params={"n_distractors": 2},
    )
    view = wrap_scene(s)
    assert {o.name for o in view.objects} == {"bowl"}
    assert {d.name for d in view.distractors} == {"distractor_0", "distractor_1"}


# ---------------------------------------------------------------------------
# Per-axis positive: identity preserved on every inactive axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", AXES)
def test_inactive_axis_identity_passes_on_scenic_scene(axis):
    """When ``axis`` is NOT in active_axes on either side, identity holds."""
    baseline = _baseline_scenic_scene()
    # Perturbed mirrors baseline exactly (no axis active) — represents the
    # post-fix behaviour: inactive on both → vacuously identical.
    perturbed = _baseline_scenic_scene()
    result = g4_identity_hook(baseline, perturbed, active_axes=[])
    assert result[axis] is True, f"axis={axis} regressed: {result}"


def test_inactive_axes_pass_when_only_one_axis_active():
    """Baseline is no-axes; perturbed activates one axis at a time; every
    OTHER axis must still report identity == True."""
    baseline = _baseline_scenic_scene()
    # Perturbed activates 'object' axis (different asset_class chosen).
    perturbed = _FakeScenicScene(
        objects=[
            _LIBEROFixture("kitchen_table", position=(0.0, 0.0, 0.8)),
            _LIBEROFixture("wooden_cabinet_1", position=(0.5, 0.2, 0.9)),
            _LIBEROObject("bowl", position=(0.1, 0.2, 0.85), asset_class="bowl_v2"),
            _LIBEROObject("cup", position=(0.3, 0.15, 0.85)),
        ],
        params={"task": "demo_task", "active_axes": "object", "chosen_asset": "bowl_v2"},
    )
    result = g4_identity_hook(baseline, perturbed, active_axes=["object"])
    # 'object' is active → omitted from result.
    assert "object" not in result
    # Every OTHER axis identity should pass.
    for ax in AXES:
        if ax == "object":
            continue
        assert result[ax] is True, f"axis={ax} should pass when only 'object' active: {result}"


# ---------------------------------------------------------------------------
# Per-axis negative: when axis is active and the corresponding field changes,
# direct (non-hook) assertions correctly report False.
#
# Note: the hook contract OMITS active axes from its dict (identity does not
# constrain them). The negative tests verify the underlying per-axis
# assertion still flags drift when invoked directly.
# ---------------------------------------------------------------------------


def test_position_drift_detected():
    baseline = _baseline_scenic_scene()
    perturbed = _FakeScenicScene(
        objects=[
            _LIBEROFixture("kitchen_table", position=(0.0, 0.0, 0.8)),
            _LIBEROFixture("wooden_cabinet_1", position=(0.5, 0.2, 0.9)),
            _LIBEROObject("bowl", position=(0.12, 0.2, 0.85)),  # x moved 0.02m
            _LIBEROObject("cup", position=(0.3, 0.15, 0.85)),
        ],
        params={"active_axes": "position"},
    )
    result = assert_position_unchanged(wrap_scene(baseline), wrap_scene(perturbed))
    assert result.passed is False
    assert "object_positions" in result.delta


def test_object_class_drift_detected():
    baseline = _baseline_scenic_scene()
    perturbed = _baseline_scenic_scene()
    # Mutate the bowl's asset_class
    bowl = next(o for o in perturbed.objects if o.libero_name == "bowl")
    bowl.asset_class = "bowl_v9_different"
    result = assert_object_unchanged(wrap_scene(baseline), wrap_scene(perturbed))
    assert result.passed is False


def test_articulation_drift_detected():
    baseline = _baseline_scenic_scene()
    perturbed = _FakeScenicScene(
        objects=list(baseline.objects),
        params={
            **baseline.params,
            "active_axes": "articulation",
            "articulation_wooden_cabinet_1": 0.25,
            "articulation_wooden_cabinet_1_state": "open",
        },
    )
    result = assert_articulation_unchanged(wrap_scene(baseline), wrap_scene(perturbed))
    assert result.passed is False
    assert "fixture_joint_states" in result.delta


def test_robot_drift_detected():
    baseline = _baseline_scenic_scene()
    perturbed = _FakeScenicScene(
        objects=list(baseline.objects),
        params={
            **baseline.params,
            "active_axes": "robot",
            "robot_model": "panda",
            "robot_init_qpos": [0.1, -0.3, 0.0, -2.0, 0.0, 1.7, 0.7],
            "robot_init_radius": 0.05,
        },
    )
    result = assert_robot_unchanged(wrap_scene(baseline), wrap_scene(perturbed))
    assert result.passed is False


def test_camera_drift_detected():
    baseline = _baseline_scenic_scene()
    perturbed = _FakeScenicScene(
        objects=list(baseline.objects),
        params={
            **baseline.params,
            "active_axes": "camera",
            "cam_azimuth": 35.0,
            "cam_elevation": 10.0,
            "cam_distance": 1.2,
        },
    )
    result = assert_camera_unchanged(wrap_scene(baseline), wrap_scene(perturbed))
    assert result.passed is False
    assert "cameras" in result.delta


def test_lighting_drift_detected():
    baseline = _baseline_scenic_scene()
    perturbed = _FakeScenicScene(
        objects=list(baseline.objects),
        params={
            **baseline.params,
            "active_axes": "lighting",
            "light_intensity": 0.7,
            "light_x_offset": 0.1,
            "light_y_offset": 0.0,
            "light_z_offset": 0.0,
            "ambient_level": 0.3,
        },
    )
    result = assert_lighting_unchanged(wrap_scene(baseline), wrap_scene(perturbed))
    assert result.passed is False


def test_background_drift_detected():
    baseline = _baseline_scenic_scene()
    perturbed = _FakeScenicScene(
        objects=list(baseline.objects),
        params={
            **baseline.params,
            "active_axes": "background",
            "wall_texture": "marble_grey",
            "floor_texture": "tile_blue",
        },
    )
    result = assert_background_unchanged(wrap_scene(baseline), wrap_scene(perturbed))
    assert result.passed is False


def test_texture_drift_detected_via_global_param():
    """Texture in the renderer is a single global ``table_texture`` param —
    when it's set on perturbed but not on baseline, every object's projected
    material differs."""
    baseline = _baseline_scenic_scene()
    perturbed = _FakeScenicScene(
        objects=list(baseline.objects),
        params={**baseline.params, "active_axes": "texture", "table_texture": "wood_dark"},
    )
    result = assert_texture_unchanged(wrap_scene(baseline), wrap_scene(perturbed))
    assert result.passed is False
    assert "materials" in result.delta


def test_distractor_drift_detected():
    baseline = _baseline_scenic_scene()
    perturbed = _FakeScenicScene(
        objects=[
            *baseline.objects,
            _LIBEROObject("distractor_0", position=(1, 1, 1), asset_class="apple"),
        ],
        params={**baseline.params, "active_axes": "distractor", "n_distractors": 1},
    )
    result = assert_distractor_unchanged(wrap_scene(baseline), wrap_scene(perturbed))
    assert result.passed is False
    assert "distractors" in result.delta


# ---------------------------------------------------------------------------
# Regression for the RCA signature: bare Scenic scene used to report every
# inactive axis as False. After the fix the hook returns True everywhere.
# ---------------------------------------------------------------------------


def test_rca_regression_inactive_axes_no_longer_false():
    """Reproduces the exact failure signature documented at
    ``~/.omar/ea/4/validation_run/rca/stage3_g4_identity_adapter_gap.md`` —
    a Scenic Scene with only ``.objects`` and ``.params`` was reporting
    every inactive axis as False because the legacy readers saw
    ``<missing>`` on both sides. Post-adapter this returns all True."""
    baseline = _baseline_scenic_scene()
    perturbed = _FakeScenicScene(
        objects=list(baseline.objects),
        # Match the campaign axis_subset = ["position", "object"]
        params={**baseline.params, "active_axes": "object,position"},
    )
    result = g4_identity_hook(baseline, perturbed, active_axes=["position", "object"])
    # Active axes omitted; everything else identity-True.
    for ax in AXES:
        if ax in ("position", "object"):
            assert ax not in result
        else:
            assert result[ax] is True, f"axis={ax} still false post-fix: {result}"
