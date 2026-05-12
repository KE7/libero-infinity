"""Unit tests for G4 family A — identity invariants.

For each of the 9 canonical axes we construct a minimal baseline scene and
a deliberately one-bit-perturbed scene, then assert:

  * the identity assertion PASSES on (baseline, baseline)
  * the identity assertion FAILS on (baseline, perturbed)
  * ``assert_all_identities`` omits the active axis and runs the rest
  * ``g4_identity_hook`` returns ``{axis: passed}`` for inactive axes only

No live MuJoCo env is required — scenes are built from ``SimpleNamespace``.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from libero_infinity.validation.invariants.identity import (
    AXES,
    AssertionResult,
    IDENTITY_ASSERTIONS,
    assert_all_identities,
    assert_articulation_unchanged,
    assert_background_unchanged,
    assert_camera_unchanged,
    assert_distractor_unchanged,
    assert_lighting_unchanged,
    assert_object_unchanged,
    assert_position_unchanged,
    assert_robot_unchanged,
    assert_texture_unchanged,
    g4_identity_hook,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ns(**kw):
    return SimpleNamespace(**kw)


def _baseline_scene():
    """A small but complete scene covering every axis-field group."""
    return _ns(
        objects=[
            _ns(name="bowl", class_id="bowl_v1",
                position=(0.1, 0.2, 0.3), material="ceramic_white"),
            _ns(name="cup", class_id="cup_v2",
                position=(0.4, 0.5, 0.6), material="porcelain_blue"),
        ],
        fixtures=[
            _ns(name="drawer", joint_states={"slide": 0.0}, material="oak"),
            _ns(name="stove", joint_states={"knob": 0.0}, material="metal_brushed"),
        ],
        distractors=[
            _ns(name="apple"),
            _ns(name="book"),
        ],
        lights=[
            _ns(name="key", position=(1.0, 0.0, 2.0), intensity=1.0),
            _ns(name="fill", position=(-1.0, 0.5, 2.0), intensity=0.5),
        ],
        cameras=[
            _ns(name="agentview", position=(0.0, -1.0, 1.5),
                rotation=(0.0, 0.0, 0.0, 1.0)),
        ],
        robot=_ns(name="panda",
                  init_qpos=(0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.7)),
        background="kitchen_v1",
    )


# ---------------------------------------------------------------------------
# Per-axis: pass on identity, fail on one-bit perturbation
# ---------------------------------------------------------------------------


def test_registry_covers_canonical_axes():
    assert set(IDENTITY_ASSERTIONS.keys()) == set(AXES)
    assert len(AXES) == 9


def test_position_pass_and_fail():
    b = _baseline_scene()
    assert assert_position_unchanged(b, _baseline_scene()).passed
    p = _baseline_scene()
    p.objects[0].position = (0.1, 0.2, 0.3 + 1e-6)  # 1µm bump — too big for 1e-9
    r = assert_position_unchanged(b, p)
    assert not r.passed and "object_positions" in r.delta


def test_articulation_pass_and_fail():
    b = _baseline_scene()
    assert assert_articulation_unchanged(b, _baseline_scene()).passed
    p = _baseline_scene()
    p.fixtures[0].joint_states["slide"] = 0.01
    r = assert_articulation_unchanged(b, p)
    assert not r.passed


def test_object_class_pass_and_fail():
    b = _baseline_scene()
    assert assert_object_unchanged(b, _baseline_scene()).passed
    p = _baseline_scene()
    p.objects[1].class_id = "cup_v3"   # one-bit asset swap
    r = assert_object_unchanged(b, p)
    assert not r.passed
    assert "object_classes" in r.delta


def test_robot_pass_and_fail():
    b = _baseline_scene()
    assert assert_robot_unchanged(b, _baseline_scene()).passed
    p = _baseline_scene()
    p.robot.init_qpos = (0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.6)
    r = assert_robot_unchanged(b, p)
    assert not r.passed


def test_texture_pass_and_fail():
    b = _baseline_scene()
    assert assert_texture_unchanged(b, _baseline_scene()).passed
    p = _baseline_scene()
    p.fixtures[0].material = "oak_dark"
    r = assert_texture_unchanged(b, p)
    assert not r.passed


def test_lighting_pass_and_fail():
    b = _baseline_scene()
    assert assert_lighting_unchanged(b, _baseline_scene()).passed
    p = _baseline_scene()
    p.lights[0].intensity = 1.001
    r = assert_lighting_unchanged(b, p)
    assert not r.passed
    # Position perturbation also detected.
    p2 = _baseline_scene()
    p2.lights[0].position = (1.0 + 1e-3, 0.0, 2.0)
    assert not assert_lighting_unchanged(b, p2).passed


def test_camera_pass_and_fail():
    b = _baseline_scene()
    assert assert_camera_unchanged(b, _baseline_scene()).passed
    p = _baseline_scene()
    p.cameras[0].position = (0.0, -1.0 + 1e-3, 1.5)
    assert not assert_camera_unchanged(b, p).passed


def test_distractor_pass_and_fail():
    b = _baseline_scene()
    assert assert_distractor_unchanged(b, _baseline_scene()).passed
    p = _baseline_scene()
    p.distractors.append(_ns(name="banana"))
    assert not assert_distractor_unchanged(b, p).passed
    p2 = _baseline_scene()
    p2.distractors.pop()
    assert not assert_distractor_unchanged(b, p2).passed


def test_background_pass_and_fail():
    b = _baseline_scene()
    assert assert_background_unchanged(b, _baseline_scene()).passed
    p = _baseline_scene()
    p.background = "kitchen_v2"
    assert not assert_background_unchanged(b, p).passed


# ---------------------------------------------------------------------------
# Tolerance policy
# ---------------------------------------------------------------------------


def test_numeric_abs_tol_is_strict_1e_9():
    b = _baseline_scene()
    p = _baseline_scene()
    # Within 1e-9 → passes (identity tolerance).
    p.objects[0].position = (0.1 + 5e-10, 0.2, 0.3)
    assert assert_position_unchanged(b, p).passed
    # 1e-8 perturbation must fail — identity is not "approximately equal".
    p2 = _baseline_scene()
    p2.objects[0].position = (0.1 + 1e-8, 0.2, 0.3)
    assert not assert_position_unchanged(b, p2).passed


def test_categorical_field_is_exact_equality():
    b = _baseline_scene()
    p = _baseline_scene()
    p.background = "kitchen_v1 "  # trailing space — must fail (exact match)
    assert not assert_background_unchanged(b, p).passed


# ---------------------------------------------------------------------------
# Composite / hook
# ---------------------------------------------------------------------------


def test_assert_all_identities_omits_active_axes():
    b = _baseline_scene()
    p = _baseline_scene()
    results = assert_all_identities(b, p, active_axes=["position", "lighting"])
    names = {r.name for r in results}
    assert "identity:position" not in names
    assert "identity:lighting" not in names
    # 9 - 2 inactive axes covered.
    assert len(results) == 7
    assert all(isinstance(r, AssertionResult) for r in results)
    assert all(r.passed for r in results)


def test_assert_all_identities_rejects_unknown_axis():
    b = _baseline_scene()
    with pytest.raises(ValueError):
        assert_all_identities(b, _baseline_scene(), active_axes=["not_an_axis"])


def test_g4_identity_hook_shape():
    b = _baseline_scene()
    p = _baseline_scene()
    # Active=texture+camera → hook reports the 7 others.
    hook = g4_identity_hook(b, p, active_axes=["texture", "camera"])
    assert set(hook.keys()) == set(AXES) - {"texture", "camera"}
    assert all(v is True for v in hook.values())


def test_g4_identity_hook_detects_isolation_leak():
    """If texture is the only active axis but a position leaks, the hook
    must flag position as failing — that's the entire purpose of family A."""
    b = _baseline_scene()
    p = _baseline_scene()
    p.objects[0].position = (0.1, 0.2, 0.3 + 0.01)   # position leak
    hook = g4_identity_hook(b, p, active_axes=["texture"])
    assert hook["position"] is False
    # Other inactive axes still pass — leak is localized.
    for axis, ok in hook.items():
        if axis != "position":
            assert ok, f"axis {axis} unexpectedly failed"


def test_full_active_axes_yields_empty_hook():
    b = _baseline_scene()
    p = _baseline_scene()
    assert g4_identity_hook(b, p, active_axes=AXES) == {}


def test_deepcopy_baseline_passes_all_axes():
    b = _baseline_scene()
    p = copy.deepcopy(b)
    hook = g4_identity_hook(b, p, active_axes=[])
    assert set(hook.keys()) == set(AXES)
    assert all(hook.values())
