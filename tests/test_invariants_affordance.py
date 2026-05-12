"""Tests for G4 Family D (affordance, cheap) invariants."""

from __future__ import annotations

from dataclasses import dataclass, field

from libero_infinity.validation.invariants import (
    assert_aabb_clear_around_grasp,
    assert_affordance,
)


@dataclass
class _Obj:
    name: str
    object_class: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    aabb: tuple[float, float, float, float, float, float] | None = None
    is_fixed: bool = False


@dataclass
class _Scene:
    objects: list[_Obj] = field(default_factory=list)


class _Registry:
    def __init__(self, grasp_points):
        self.grasp_points = grasp_points


def test_grasp_clearance_pass():
    obj = _Obj("bowl_1", "bowl", position=(0.5, 0.5, 0.0))
    scene = _Scene(
        objects=[
            obj,
            _Obj("wall_1", "wall", aabb=(1.0, 1.5, -1.0, 1.0, 0.0, 1.0), is_fixed=True),
        ]
    )
    registry = _Registry({"bowl": (0.0, 0.0, 0.05)})
    r = assert_aabb_clear_around_grasp(obj, scene, registry, gripper_jaw_half_width=0.04)
    assert r.passed is True


def test_grasp_clearance_fail_blocked():
    obj = _Obj("bowl_1", "bowl", position=(0.0, 0.0, 0.0))
    blocker = _Obj("wall_1", "wall", aabb=(-0.01, 0.01, -0.5, 0.5, 0.0, 1.0), is_fixed=True)
    scene = _Scene(objects=[obj, blocker])
    registry = _Registry({"bowl": (0.0, 0.0, 0.05)})
    r = assert_aabb_clear_around_grasp(obj, scene, registry)
    assert r.passed is False
    assert r.payload["obstructions"][0]["occluder"] == "wall_1"


def test_grasp_clearance_skip_no_grasp_data():
    obj = _Obj("bowl_1", "bowl", position=(0, 0, 0))
    scene = _Scene(objects=[obj])
    registry = _Registry({})  # no grasp metadata
    r = assert_aabb_clear_around_grasp(obj, scene, registry)
    assert r.passed is None
    assert r.payload["reason"] == "no-grasp-data"


def test_grasp_clearance_skip_for_unknown_class_with_default_libero_registry():
    # The bundled asset_registry has no grasp_points field at all, so this
    # MUST skip — never silently pass.
    from libero_infinity import asset_registry as ar

    obj = _Obj("bowl_1", "akita_black_bowl", position=(0, 0, 0))
    scene = _Scene(objects=[obj])
    r = assert_aabb_clear_around_grasp(obj, scene, ar)
    assert r.passed is None


def test_grasp_clearance_grasp_points_param_overrides_registry():
    obj = _Obj("bowl_1", "akita_black_bowl", position=(0.0, 0.0, 0.0))
    scene = _Scene(objects=[obj])
    r = assert_aabb_clear_around_grasp(
        obj, scene, registry=None, grasp_points={"akita_black_bowl": (0, 0, 0)}
    )
    assert r.passed is True


def test_assert_affordance_skips_fixed_objects_from_iteration():
    grasp_points = {"bowl": (0, 0, 0)}
    scene = _Scene(
        objects=[
            _Obj("bowl_1", "bowl", position=(0, 0, 0)),
            _Obj("table_1", "table", is_fixed=True),
        ]
    )
    results = assert_affordance(scene, grasp_points=grasp_points)
    assert len(results) == 1
    assert results[0].payload["name"] == "bowl_1"
