"""G4 Family D — affordance (cheap) invariants.

The single check provided here is:

    assert_aabb_clear_around_grasp(obj, scene, registry)
        For ``obj``, look up a grasp-point in the asset registry. If the asset
        class has no grasp-point metadata, return a *skip* (``passed=None``)
        with reason ``"no-grasp-data"`` — **never** a pass. Otherwise, build a
        gripper-jaw xy-AABB around the grasp-point (half-width
        ``gripper_jaw_half_width``, default 0.04 m) and require that no
        *fixed* scene geometry (``is_fixed=True``) has an xy-AABB intersecting
        it.

Grasp-point lookup
------------------

The default registry adapter looks for a ``grasp_points`` dict on the
registry object::

    registry.grasp_points: dict[class_name, (gx, gy, gz)]

If the registry is a plain ``dict`` (such as ``ASSET_VARIANTS``), callers can
pass ``grasp_points=...`` explicitly. Per the validation plan, the current
``asset_registry.py`` has no grasp-point metadata — so without an explicit
``grasp_points`` argument, this assertion will *honestly skip* for every
object, which is a real signal that the metadata is missing upstream.
"""

from __future__ import annotations

from typing import Any, Mapping

from ._result import AssertionResult
from .domain import _iter_scene_objects, _obj_class

DEFAULT_GRIPPER_JAW_HALF_WIDTH = 0.04  # ~Panda jaw half-aperture, metres

__all__ = [
    "AssertionResult",
    "AFFORDANCE_ASSERTIONS",
    "DEFAULT_GRIPPER_JAW_HALF_WIDTH",
    "assert_aabb_clear_around_grasp",
    "assert_affordance",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_grasp_point(
    obj_class: str,
    registry: Any,
    grasp_points: Mapping[str, tuple[float, float, float]] | None,
) -> tuple[float, float, float] | None:
    """Return the (gx, gy, gz) grasp point for ``obj_class`` or ``None``."""
    if grasp_points is not None and obj_class in grasp_points:
        gp = grasp_points[obj_class]
        return (float(gp[0]), float(gp[1]), float(gp[2]))
    gp_attr = getattr(registry, "grasp_points", None) if registry is not None else None
    if isinstance(gp_attr, Mapping) and obj_class in gp_attr:
        gp = gp_attr[obj_class]
        return (float(gp[0]), float(gp[1]), float(gp[2]))
    return None


def _aabb_xy_intersects(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return (ax0 <= bx1) and (bx0 <= ax1) and (ay0 <= by1) and (by0 <= ay1)


# ---------------------------------------------------------------------------
# D1 — clearance around grasp point
# ---------------------------------------------------------------------------


def assert_aabb_clear_around_grasp(
    obj: Any,
    scene: Any,
    registry: Any = None,
    *,
    gripper_jaw_half_width: float = DEFAULT_GRIPPER_JAW_HALF_WIDTH,
    grasp_points: Mapping[str, tuple[float, float, float]] | None = None,
) -> AssertionResult:
    """Require ``gripper_jaw_half_width`` xy-clearance around the grasp point.

    Skip (``passed=None``) iff the asset class has no grasp-point metadata.
    """
    name = getattr(obj, "name", "?")
    cls = _obj_class(obj)
    if cls is None:
        return AssertionResult(
            name="aabb_clear_around_grasp",
            passed=False,
            detail=f"{name}: object has no asset class.",
            payload={"name": name},
        )
    gp = _resolve_grasp_point(cls, registry, grasp_points)
    if gp is None:
        return AssertionResult(
            name="aabb_clear_around_grasp",
            passed=None,
            detail=f"{name}: no-grasp-data for class {cls!r}.",
            payload={"name": name, "class": cls, "reason": "no-grasp-data"},
        )
    # Object world position is required to place the grasp point in world frame.
    obj_pos = getattr(obj, "position", None)
    if obj_pos is None or len(obj_pos) < 2:
        return AssertionResult(
            name="aabb_clear_around_grasp",
            passed=False,
            detail=f"{name}: missing scene position for grasp placement.",
            payload={"name": name, "class": cls},
        )
    gx_world = float(obj_pos[0]) + gp[0]
    gy_world = float(obj_pos[1]) + gp[1]
    h = float(gripper_jaw_half_width)
    grasp_aabb = (gx_world - h, gx_world + h, gy_world - h, gy_world + h)

    obstructions: list[dict[str, Any]] = []
    for other in _iter_scene_objects(scene):
        if other is obj:
            continue
        if not getattr(other, "is_fixed", False):
            continue
        other_aabb = getattr(other, "aabb", None)
        if other_aabb is None or len(other_aabb) < 4:
            continue
        ox0, ox1, oy0, oy1 = other_aabb[0], other_aabb[1], other_aabb[2], other_aabb[3]
        if _aabb_xy_intersects(grasp_aabb, (ox0, ox1, oy0, oy1)):
            obstructions.append(
                {
                    "occluder": getattr(other, "name", "?"),
                    "occluder_aabb_xy": (ox0, ox1, oy0, oy1),
                }
            )

    payload = {
        "name": name,
        "class": cls,
        "grasp_point_local": gp,
        "grasp_point_world": (
            gx_world,
            gy_world,
            float(obj_pos[2]) + gp[2] if len(obj_pos) >= 3 else None,
        ),
        "grasp_aabb_xy": grasp_aabb,
        "gripper_jaw_half_width": h,
    }
    if obstructions:
        return AssertionResult(
            name="aabb_clear_around_grasp",
            passed=False,
            detail=(
                f"{name}: {len(obstructions)} fixed-geometry obstruction(s) within "
                f"jaw-half-width {h}m of grasp point."
            ),
            payload={**payload, "obstructions": obstructions},
        )
    return AssertionResult(
        name="aabb_clear_around_grasp",
        passed=True,
        detail=f"{name}: grasp point clear (half-width={h}m).",
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


AFFORDANCE_ASSERTIONS: tuple[str, ...] = ("aabb_clear_around_grasp",)


def assert_affordance(
    scene: Any,
    registry: Any = None,
    *,
    gripper_jaw_half_width: float = DEFAULT_GRIPPER_JAW_HALF_WIDTH,
    grasp_points: Mapping[str, tuple[float, float, float]] | None = None,
    only_movable: bool = True,
) -> list[AssertionResult]:
    """Run the cheap-affordance check on each movable scene object.

    Fixed geometry is not graspable so it is skipped from iteration when
    ``only_movable`` is True (the default).
    """
    results: list[AssertionResult] = []
    for o in _iter_scene_objects(scene):
        if only_movable and getattr(o, "is_fixed", False):
            continue
        results.append(
            assert_aabb_clear_around_grasp(
                o,
                scene,
                registry,
                gripper_jaw_half_width=gripper_jaw_half_width,
                grasp_points=grasp_points,
            )
        )
    return results
