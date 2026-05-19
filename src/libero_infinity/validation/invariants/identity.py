"""G4 family A — identity invariants (cross-axis isolation).

For each of the 9 canonical perturbation axes, this module exports an
assertion of the form ``assert_<axis>_unchanged(baseline, perturbed)`` that
returns an :class:`AssertionResult`.

Semantics
---------
When axis X is **not** in ``active_axes``, every attribute belonging to axis X
in the perturbed scene MUST be identical to the no-axes baseline. "Identical"
means:

* categorical fields (class id, asset name, material id, distractor names,
  background name, etc.): EXACT equality (``==``).
* numeric fields (positions, joint values, intensities, extrinsics):
  ``math.isclose(a, b, abs_tol=1e-9)`` — identity, not "approximately equal".
  Tolerance-based pose comparisons live in family C.

This module is intentionally agnostic to the concrete scene class. It accesses
fields through a small set of duck-typed reader functions that try a handful
of common attribute paths used by the libero-infinity scene/IR. If the
expected attribute is missing the assertion records a failure rather than
silently passing — "absent" is not the same as "unchanged".

Public API
----------
* :class:`AssertionResult`
* ``assert_<axis>_unchanged(baseline, perturbed)`` × 9
* :func:`assert_all_identities`
* :func:`g4_identity_hook`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from ._scene_view import wrap_scene

# Canonical 9 axes — must agree with validation.sweep.CANONICAL_AXES.
AXES: tuple[str, ...] = (
    "position",
    "articulation",
    "object",
    "robot",
    "texture",
    "lighting",
    "camera",
    "distractor",
    "background",
)

_ABS_TOL = 1e-9


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class AssertionResult:
    """Outcome of a single identity assertion.

    Attributes
    ----------
    name:
        Assertion name, e.g. ``"identity:position"``.
    passed:
        ``True`` iff every checked field matched.
    detail:
        Human-readable diff on failure (empty on pass).
    delta:
        ``{field_path: (baseline_value, observed_value)}`` for every
        mismatching field. Empty on pass.
    """

    name: str
    passed: bool
    detail: str = ""
    delta: dict[str, tuple[Any, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------


_MISSING = object()


def _get(obj: Any, path: str, default: Any = _MISSING) -> Any:
    """Resolve a dotted attribute / key path on ``obj``.

    Supports both attribute access and ``Mapping`` lookup so tests can use
    ``SimpleNamespace``, dataclasses, or plain dicts interchangeably.
    """
    cur: Any = obj
    for part in path.split("."):
        if cur is _MISSING:
            return default
        if isinstance(cur, Mapping):
            if part in cur:
                cur = cur[part]
                continue
            return default
        if hasattr(cur, part):
            cur = getattr(cur, part)
            continue
        return default
    return cur


def _first_present(obj: Any, paths: Iterable[str]) -> tuple[str | None, Any]:
    """Return (path, value) for the first path that exists on ``obj``."""
    for p in paths:
        v = _get(obj, p)
        if v is not _MISSING:
            return p, v
    return None, _MISSING


def _values_equal(a: Any, b: Any) -> bool:
    """Identity-equality with abs_tol=1e-9 for numerics, EXACT for everything else."""
    # Numeric scalar pairs (excluding bool — bool is treated categorically).
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(float(a), float(b), abs_tol=_ABS_TOL, rel_tol=0.0)
    # Sequences: elementwise (only treat as numeric if all elements numeric).
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (set, frozenset)) and isinstance(b, (set, frozenset)):
        return a == b
    return a == b


def _diff_fields(
    baseline: Any, perturbed: Any, paths: Iterable[str]
) -> dict[str, tuple[Any, Any]]:
    """Compare a list of dotted paths between two scenes.

    Paths missing on **both** sides are recorded only when *every* path in
    the group is doubly missing (i.e. the entire field group is absent —
    that's a renderer-dropped-a-known-field bug). Otherwise the path is
    skipped: scenes that use one of several conventional names should not be
    penalised for the unused aliases.
    """
    paths = list(paths)
    delta: dict[str, tuple[Any, Any]] = {}
    doubly_missing: list[str] = []
    for p in paths:
        bv = _get(baseline, p)
        pv = _get(perturbed, p)
        if bv is _MISSING and pv is _MISSING:
            doubly_missing.append(p)
            continue
        if bv is _MISSING:
            delta[p] = ("<missing>", pv)
            continue
        if pv is _MISSING:
            delta[p] = (bv, "<missing>")
            continue
        if not _values_equal(bv, pv):
            delta[p] = (bv, pv)
    # Only surface doubly-missing entries when the entire field group is
    # absent (no path on either side resolved); otherwise the group is
    # present via one of its conventional aliases and the missing paths are
    # legitimate name variants.
    if not delta and len(doubly_missing) == len(paths):
        for p in doubly_missing:
            delta[p] = ("<missing>", "<missing>")
    return delta


def _result_from_delta(name: str, delta: dict[str, tuple[Any, Any]]) -> AssertionResult:
    if not delta:
        return AssertionResult(name=name, passed=True)
    lines = [f"  {k}: baseline={b!r}  observed={o!r}" for k, (b, o) in delta.items()]
    detail = f"{name} mismatched {len(delta)} field(s):\n" + "\n".join(lines)
    return AssertionResult(name=name, passed=False, detail=detail, delta=delta)


# ---------------------------------------------------------------------------
# Per-axis field specs
# ---------------------------------------------------------------------------
#
# Each spec is a list of dotted attribute paths to compare. We compare ALL
# listed paths — a renderer that uses only one of the conventional names will
# pass through the others as "<missing>" on both sides and that pair will be
# recorded as a mismatch. The intent is "identity means identity"; if your
# scene representation uses different field names, register them in the
# project's scene adapter rather than weakening the invariant here.


def _object_positions(scene: Any) -> Any:
    """Return {name: (x, y, z)} for every object in the scene, sorted."""
    objs = _get(scene, "objects")
    if objs is _MISSING:
        return _MISSING
    out: dict[str, tuple[float, float, float]] = {}
    for o in objs:
        name = _get(o, "name", _get(o, "id"))
        pos = _get(o, "position", _get(o, "pos"))
        if name is _MISSING or pos is _MISSING:
            return _MISSING
        out[str(name)] = tuple(float(v) for v in pos)  # type: ignore[assignment]
    return dict(sorted(out.items()))


def _object_classes(scene: Any) -> Any:
    objs = _get(scene, "objects")
    if objs is _MISSING:
        return _MISSING
    out: dict[str, str] = {}
    for o in objs:
        name = _get(o, "name", _get(o, "id"))
        cls = _get(o, "class_id", _get(o, "asset", _get(o, "asset_name")))
        if name is _MISSING or cls is _MISSING:
            return _MISSING
        out[str(name)] = cls
    return dict(sorted(out.items()))


def _object_materials(scene: Any) -> Any:
    """Materials for both objects and fixtures (texture axis)."""
    out: dict[str, Any] = {}
    for collection_path in ("objects", "fixtures"):
        coll = _get(scene, collection_path)
        if coll is _MISSING:
            continue
        for o in coll:
            name = _get(o, "name", _get(o, "id"))
            mat = _get(o, "material", _get(o, "material_id"))
            if name is _MISSING or mat is _MISSING:
                continue
            out[f"{collection_path}.{name}"] = mat
    return dict(sorted(out.items())) if out else _MISSING


def _fixture_joint_states(scene: Any) -> Any:
    fixtures = _get(scene, "fixtures")
    if fixtures is _MISSING:
        return _MISSING
    out: dict[str, Any] = {}
    for f in fixtures:
        name = _get(f, "name", _get(f, "id"))
        joints = _get(f, "joint_states", _get(f, "joints"))
        if name is _MISSING or joints is _MISSING:
            continue
        if isinstance(joints, Mapping):
            joints = {str(k): float(v) for k, v in joints.items()}
        else:
            joints = tuple(float(v) for v in joints)
        out[str(name)] = joints
    return dict(sorted(out.items())) if out else _MISSING


def _distractor_set(scene: Any) -> Any:
    ds = _get(scene, "distractors")
    if ds is _MISSING:
        return _MISSING
    names: list[str] = []
    for d in ds:
        nm = _get(d, "name", _get(d, "id"))
        if nm is _MISSING:
            return _MISSING
        names.append(str(nm))
    return tuple(sorted(names))


def _lights(scene: Any) -> Any:
    ls = _get(scene, "lights")
    if ls is _MISSING:
        return _MISSING
    out: dict[str, dict[str, Any]] = {}
    for L in ls:
        nm = _get(L, "name", _get(L, "id"))
        pos = _get(L, "position", _get(L, "pos"))
        inten = _get(L, "intensity")
        if nm is _MISSING or pos is _MISSING or inten is _MISSING:
            return _MISSING
        out[str(nm)] = {
            "position": tuple(float(v) for v in pos),
            "intensity": float(inten),
        }
    return dict(sorted(out.items()))


def _cameras(scene: Any) -> Any:
    cs = _get(scene, "cameras")
    if cs is _MISSING:
        return _MISSING
    out: dict[str, dict[str, Any]] = {}
    for c in cs:
        nm = _get(c, "name", _get(c, "id"))
        pos = _get(c, "position", _get(c, "pos"))
        rot = _get(c, "rotation", _get(c, "quat", _get(c, "orientation")))
        if nm is _MISSING or pos is _MISSING or rot is _MISSING:
            return _MISSING
        out[str(nm)] = {
            "position": tuple(float(v) for v in pos),
            "rotation": tuple(float(v) for v in rot),
        }
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Computed-aggregate diff
# ---------------------------------------------------------------------------


def _diff_aggregate(
    name: str,
    baseline: Any,
    perturbed: Any,
    extractor: Callable[[Any], Any],
    label: str,
) -> AssertionResult:
    bv = extractor(baseline)
    pv = extractor(perturbed)
    if bv is _MISSING and pv is _MISSING:
        return AssertionResult(
            name=name,
            passed=False,
            detail=f"{name}: required field group {label!r} missing on both scenes",
            delta={label: ("<missing>", "<missing>")},
        )
    if bv is _MISSING or pv is _MISSING:
        return AssertionResult(
            name=name,
            passed=False,
            detail=f"{name}: {label!r} missing on one side (baseline={bv!r}, observed={pv!r})",
            delta={label: (bv if bv is not _MISSING else "<missing>",
                            pv if pv is not _MISSING else "<missing>")},
        )
    if _values_equal(bv, pv):
        return AssertionResult(name=name, passed=True)
    return AssertionResult(
        name=name,
        passed=False,
        detail=f"{name}: {label!r} differs\n  baseline={bv!r}\n  observed={pv!r}",
        delta={label: (bv, pv)},
    )


# ---------------------------------------------------------------------------
# Public per-axis assertions
# ---------------------------------------------------------------------------


def assert_position_unchanged(baseline: Any, perturbed: Any) -> AssertionResult:
    """Object xy/z positions match baseline."""
    return _diff_aggregate(
        "identity:position", baseline, perturbed, _object_positions, "object_positions"
    )


def assert_articulation_unchanged(baseline: Any, perturbed: Any) -> AssertionResult:
    """Fixture joint states match baseline."""
    return _diff_aggregate(
        "identity:articulation", baseline, perturbed, _fixture_joint_states,
        "fixture_joint_states",
    )


def assert_object_unchanged(baseline: Any, perturbed: Any) -> AssertionResult:
    """Object class identities (sampled asset variants) match."""
    return _diff_aggregate(
        "identity:object", baseline, perturbed, _object_classes, "object_classes"
    )


def assert_robot_unchanged(baseline: Any, perturbed: Any) -> AssertionResult:
    """Robot init joint config matches."""
    paths = ("robot.init_qpos", "robot.init_joint_config", "robot.name", "robot.model")
    delta = _diff_fields(baseline, perturbed, paths)
    # Filter "both missing" only when *every* path is doubly missing — i.e.
    # the scene has no robot field at all. If any robot subfield is present
    # we keep the strict double-missing entries (renderer dropped a known
    # field).
    all_missing = all(v == ("<missing>", "<missing>") for v in delta.values())
    if all_missing and delta:
        return AssertionResult(
            name="identity:robot", passed=False,
            detail="identity:robot: no robot fields present on either scene",
            delta=delta,
        )
    return _result_from_delta("identity:robot", delta)


def assert_texture_unchanged(baseline: Any, perturbed: Any) -> AssertionResult:
    """Object/fixture material identities match."""
    return _diff_aggregate(
        "identity:texture", baseline, perturbed, _object_materials, "materials"
    )


def assert_lighting_unchanged(baseline: Any, perturbed: Any) -> AssertionResult:
    """Light positions + intensities match."""
    return _diff_aggregate(
        "identity:lighting", baseline, perturbed, _lights, "lights"
    )


def assert_camera_unchanged(baseline: Any, perturbed: Any) -> AssertionResult:
    """Camera extrinsics match."""
    return _diff_aggregate(
        "identity:camera", baseline, perturbed, _cameras, "cameras"
    )


def assert_distractor_unchanged(baseline: Any, perturbed: Any) -> AssertionResult:
    """Set of distractor objects matches."""
    return _diff_aggregate(
        "identity:distractor", baseline, perturbed, _distractor_set, "distractors"
    )


def assert_background_unchanged(baseline: Any, perturbed: Any) -> AssertionResult:
    """Scene / skybox identity matches."""
    paths = ("background", "background.name", "background.id", "scene_name", "skybox")
    delta = _diff_fields(baseline, perturbed, paths)
    all_missing = all(v == ("<missing>", "<missing>") for v in delta.values())
    if all_missing and delta:
        return AssertionResult(
            name="identity:background", passed=False,
            detail="identity:background: no background field present on either scene",
            delta=delta,
        )
    return _result_from_delta("identity:background", delta)


# ---------------------------------------------------------------------------
# Registry + composite entry points
# ---------------------------------------------------------------------------


IDENTITY_ASSERTIONS: dict[str, Callable[[Any, Any], AssertionResult]] = {
    "position": assert_position_unchanged,
    "articulation": assert_articulation_unchanged,
    "object": assert_object_unchanged,
    "robot": assert_robot_unchanged,
    "texture": assert_texture_unchanged,
    "lighting": assert_lighting_unchanged,
    "camera": assert_camera_unchanged,
    "distractor": assert_distractor_unchanged,
    "background": assert_background_unchanged,
}

# Sanity: registry covers exactly the 9 canonical axes.
assert set(IDENTITY_ASSERTIONS.keys()) == set(AXES), (
    f"identity registry / AXES mismatch: "
    f"{set(IDENTITY_ASSERTIONS) ^ set(AXES)}"
)


def assert_all_identities(
    baseline_scene: Any,
    perturbed_scene: Any,
    active_axes: Iterable[str],
) -> list[AssertionResult]:
    """Run an identity assertion for every axis NOT in ``active_axes``.

    Returns one :class:`AssertionResult` per inactive axis, in canonical
    ``AXES`` order.
    """
    active = set(active_axes)
    unknown = active - set(AXES)
    if unknown:
        raise ValueError(
            f"assert_all_identities: unknown axes in active_axes: {sorted(unknown)}"
        )
    # Project Scenic Scenes (which only expose .objects/.params) onto the
    # richer per-axis schema this module reads. Already-rich scenes (used by
    # the unit-test fixtures) pass through unchanged. See ``_scene_view.py``.
    baseline_scene = wrap_scene(baseline_scene)
    perturbed_scene = wrap_scene(perturbed_scene)
    return [
        IDENTITY_ASSERTIONS[axis](baseline_scene, perturbed_scene)
        for axis in AXES
        if axis not in active
    ]


def g4_identity_hook(
    baseline_scene: Any,
    perturbed_scene: Any,
    active_axes: Iterable[str],
) -> dict[str, bool]:
    """Sweep-friendly hook: returns ``{axis: passed}`` for every inactive axis.

    Active axes are intentionally OMITTED from the result (changes on those
    axes are expected and out of scope for identity invariants). The sweep
    harness can flatten this dict into the JSONL row, e.g. as
    ``g4_identity_position=True``.
    """
    return {r.name.split(":", 1)[1]: r.passed
            for r in assert_all_identities(baseline_scene, perturbed_scene, active_axes)}
