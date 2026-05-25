"""G4 Family B — domain invariants.

Operate on the parsed BDDL (``TaskConfig``), the sampled Scenic scene
(duck-typed; see ``SceneLike`` below), an asset-class registry
(duck-typed; defaults to ``libero_infinity.asset_registry.ASSET_VARIANTS``),
and optionally a MuJoCo (model, data) pair after settling.

Assertions (all return :class:`AssertionResult`):

* ``assert_bddl_objects_present``      — every BDDL ``:objects`` instance is in the scene
* ``assert_assets_in_registry``        — every sampled asset class is in the registry
* ``assert_no_initial_collisions``     — no penetrating contacts after settling
* ``assert_on_predicates_z``           — every ``(On A B)`` in ``:init`` satisfies
  ``A.z > B.z_top - tol``
* ``assert_goal_false_at_reset``       — ``:goal`` evaluates ``False`` at ``t=0``
* ``assert_goal_reachable_soft``       — heuristic feasibility hint (objects present + not occluded)

Design notes
------------

* The *scene* contract is intentionally duck-typed: anything with an
  iterable ``.objects`` attribute, where each object exposes ``name``,
  ``object_class`` (or ``class_``), and ``position = (x, y, z)``, will work.
  Optional fields used opportunistically: ``z_top``, ``aabb`` (a
  6-tuple ``(xmin, xmax, ymin, ymax, zmin, zmax)``), ``is_fixed``.

* ``passed=None`` is used **only** when the relevant input is absent — for
  example, no MuJoCo handle was supplied, or the BDDL has no ``(On …)``
  predicate to check.  Never as a substitute for ``False``.

* ``assert_goal_false_at_reset`` requires the caller to supply a
  ``goal_evaluator`` callable (typically ``env.check_success`` or a
  BDDL goal predicate evaluator).  We do *not* import LIBERO here.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from ._result import AssertionResult
from ._scene_view import is_scene_fixture, resolve_object_name

DEFAULT_POS_TOL = 1e-4

__all__ = [
    "AssertionResult",
    "DOMAIN_ASSERTIONS",
    "assert_bddl_objects_present",
    "assert_assets_in_registry",
    "assert_no_initial_collisions",
    "assert_on_predicates_z",
    "assert_goal_false_at_reset",
    "assert_goal_reachable_soft",
    "assert_domain",
]


# ---------------------------------------------------------------------------
# Duck-typed protocols (helpful for IDE / mypy; runtime checks are tolerant).
# ---------------------------------------------------------------------------


@runtime_checkable
class _SceneObj(Protocol):
    name: str
    position: tuple[float, float, float]


def _obj_class(o: Any) -> str | None:
    """Return the asset class of a scene object, tolerant to field naming."""
    for attr in ("object_class", "class_", "asset_class", "cls"):
        v = getattr(o, attr, None)
        if isinstance(v, str):
            return v
    return None


def _obj_position(o: Any) -> tuple[float, float, float] | None:
    pos = getattr(o, "position", None)
    if pos is None:
        return None
    try:
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    except (TypeError, ValueError, IndexError):
        return None
    return (x, y, z)


def _obj_z_top(o: Any) -> float | None:
    """Best-effort top-of-AABB Z. Falls back to z + 0.5 * height if available."""
    z_top = getattr(o, "z_top", None)
    if z_top is not None:
        return float(z_top)
    aabb = getattr(o, "aabb", None)
    if aabb is not None and len(aabb) == 6:
        return float(aabb[5])
    pos = _obj_position(o)
    height = getattr(o, "height", None)
    if pos is not None and height is not None:
        return pos[2] + 0.5 * float(height)
    return None


def _iter_scene_objects(scene: Any) -> list[Any]:
    objs = getattr(scene, "objects", None)
    if objs is None:
        return []
    return list(objs)


def _scene_by_name(scene: Any) -> dict[str, Any]:
    # Scenic LIBEROObject/LIBEROFixture expose identity as ``libero_name``,
    # not ``name`` — resolve via the shared scene-object adapter so the key
    # is the real instance name (see ``_scene_view.resolve_object_name``).
    return {resolve_object_name(o): o for o in _iter_scene_objects(scene)}


def _iter_sampled_objects(scene: Any) -> list[Any]:
    """Scene objects that are sampled task assets (movable objects + distractors).

    Excludes Scenic ``LIBEROFixture`` instances: fixtures are scene structure
    placed from the BDDL ``:fixtures`` block, not assets sampled from the
    asset-variant registry, so the registry / consistency invariants must not
    score them.
    """
    return [o for o in _iter_scene_objects(scene) if not is_scene_fixture(o)]


# ---------------------------------------------------------------------------
# B1 — every BDDL object is in the scene
# ---------------------------------------------------------------------------


def assert_bddl_objects_present(bddl: Any, scene: Any) -> AssertionResult:
    """Every movable object listed in BDDL ``:objects`` must appear in ``scene``."""
    bddl_objs = list(getattr(bddl, "movable_objects", []) or [])
    if not bddl_objs:
        return AssertionResult(
            name="bddl_objects_present",
            passed=None,
            detail="BDDL has no movable objects to verify.",
            payload={"bddl_objects": []},
        )
    scene_names = set(_scene_by_name(scene).keys())
    missing = [
        getattr(o, "instance_name", repr(o))
        for o in bddl_objs
        if getattr(o, "instance_name", None) not in scene_names
    ]
    if missing:
        return AssertionResult(
            name="bddl_objects_present",
            passed=False,
            detail=f"{len(missing)} BDDL object(s) missing from scene: {missing}",
            payload={"missing": missing, "scene_objects": sorted(scene_names)},
        )
    return AssertionResult(
        name="bddl_objects_present",
        passed=True,
        detail=f"All {len(bddl_objs)} BDDL objects present in scene.",
        payload={"count": len(bddl_objs)},
    )


# ---------------------------------------------------------------------------
# B2 — every sampled asset class is in the asset registry
# ---------------------------------------------------------------------------


def _default_registry() -> Iterable[str]:
    from libero_infinity.asset_registry import ASSET_VARIANTS

    return ASSET_VARIANTS.keys()


def assert_assets_in_registry(scene: Any, registry: Iterable[str] | None = None) -> AssertionResult:
    """Every asset class sampled into ``scene`` must exist in ``registry``."""
    if registry is None:
        registry_set: set[str] = set(_default_registry())
    else:
        registry_set = set(registry)
    objs = _iter_sampled_objects(scene)
    if not objs:
        return AssertionResult(
            name="assets_in_registry",
            passed=None,
            detail="Scene has no sampled objects to verify.",
            payload={},
        )
    unknown: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for o in objs:
        cls = _obj_class(o) or None
        if cls is None:
            unknown.append((resolve_object_name(o) or "?", None))
            continue
        seen.add(cls)
        if cls not in registry_set:
            unknown.append((resolve_object_name(o) or "?", cls))
    if unknown:
        return AssertionResult(
            name="assets_in_registry",
            passed=False,
            detail=f"{len(unknown)} object(s) with unknown asset class: {unknown}",
            payload={"unknown": unknown, "registry_size": len(registry_set)},
        )
    return AssertionResult(
        name="assets_in_registry",
        passed=True,
        detail=f"All {len(objs)} scene asset classes ({len(seen)} unique) are in registry.",
        payload={"unique_classes": sorted(seen)},
    )


# ---------------------------------------------------------------------------
# B3 — no initial penetrating contacts after MuJoCo settling
# ---------------------------------------------------------------------------


def assert_no_initial_collisions(
    scene: Any,
    mjmodel: Any | None,
    mjdata: Any | None,
    *,
    tol: float = DEFAULT_POS_TOL,
) -> AssertionResult:
    """After MuJoCo settling, no contact should have ``dist < -tol`` (penetrating).

    Returns ``passed=None`` if either ``mjmodel`` or ``mjdata`` is None — the
    caller did not provide a physics handle and we honestly can't check.
    """
    if mjmodel is None or mjdata is None:
        return AssertionResult(
            name="no_initial_collisions",
            passed=None,
            detail="No MuJoCo (model, data) supplied — cannot evaluate contacts.",
            payload={},
        )
    ncon = int(getattr(mjdata, "ncon", 0))
    contacts = getattr(mjdata, "contact", None)
    penetrating: list[dict[str, Any]] = []
    for i in range(ncon):
        c = contacts[i]
        dist = float(getattr(c, "dist", 0.0))
        if dist < -tol:
            penetrating.append(
                {
                    "index": i,
                    "dist": dist,
                    "geom1": int(getattr(c, "geom1", -1)),
                    "geom2": int(getattr(c, "geom2", -1)),
                }
            )
    if penetrating:
        worst = min(p["dist"] for p in penetrating)
        return AssertionResult(
            name="no_initial_collisions",
            passed=False,
            detail=(
                f"{len(penetrating)} penetrating contact(s) after settle "
                f"(worst dist={worst:.6f}, tol={tol})"
            ),
            payload={"penetrating": penetrating, "tol": tol, "ncon": ncon},
        )
    return AssertionResult(
        name="no_initial_collisions",
        passed=True,
        detail=f"No penetrating contacts (ncon={ncon}, tol={tol}).",
        payload={"ncon": ncon, "tol": tol},
    )


# ---------------------------------------------------------------------------
# B4 — On predicates: stacked object Z above support Z_top
# ---------------------------------------------------------------------------


_ON_RE = re.compile(r"\(\s*(?:On|on)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\)")


def _extract_on_predicates(init_text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in _ON_RE.finditer(init_text or "")]


def assert_on_predicates_z(
    bddl: Any, scene: Any, *, tol: float = DEFAULT_POS_TOL
) -> AssertionResult:
    """Each ``(On A B)`` in ``:init`` must satisfy ``A.z > B.z_top - tol``."""
    init_text = getattr(bddl, "init_text", "") or ""
    pairs = _extract_on_predicates(init_text)
    if not pairs:
        return AssertionResult(
            name="on_predicates_z",
            passed=None,
            detail="No (On …) predicates in :init — nothing to check.",
            payload={},
        )
    by_name = _scene_by_name(scene)
    # Pre-collect fixtures for prefix-based region/side resolution. BDDL
    # `(On <obj> <fixture>_<region_or_side>)` predicates reference *named
    # regions* on a fixture (e.g. `main_table_bowl_region`,
    # `wooden_cabinet_1_top_side`) that are not themselves materialised as
    # Scenic objects. We resolve the support to the longest fixture name
    # that is a prefix of the target token (separated by `_`), and use that
    # fixture's z_top as the support surface height. This is exactly the
    # semantics LIBERO uses (regions are surface patches on a fixture).
    fixtures_by_name = {
        resolve_object_name(o): o
        for o in _iter_scene_objects(scene)
        if is_scene_fixture(o)
    }

    def _resolve_support(target: str) -> Any:
        obj = by_name.get(target)
        if obj is not None:
            return obj
        # Longest-prefix fixture match. `wooden_cabinet_1_top_side` →
        # `wooden_cabinet_1`; `main_table_bowl_region` → `main_table`.
        best: tuple[int, Any] = (-1, None)
        for fname, fobj in fixtures_by_name.items():
            if target == fname or target.startswith(fname + "_"):
                if len(fname) > best[0]:
                    best = (len(fname), fobj)
        return best[1]

    violations: list[dict[str, Any]] = []
    missing: list[str] = []
    for a, b in pairs:
        oa = by_name.get(a)
        ob = _resolve_support(b)
        if oa is None or ob is None:
            missing.append(f"{a} on {b}")
            continue
        pa = _obj_position(oa)
        z_top_b = _obj_z_top(ob)
        if pa is None or z_top_b is None:
            missing.append(f"{a} on {b} (no z/z_top)")
            continue
        if not (pa[2] > z_top_b - tol):
            violations.append(
                {"a": a, "b": b, "a_z": pa[2], "b_z_top": z_top_b, "margin": pa[2] - z_top_b}
            )
    if violations or missing:
        return AssertionResult(
            name="on_predicates_z",
            passed=False,
            detail=(
                f"{len(violations)} On-predicate Z violation(s), "
                f"{len(missing)} missing/incomplete pair(s)."
            ),
            payload={"violations": violations, "missing": missing, "pairs": pairs, "tol": tol},
        )
    return AssertionResult(
        name="on_predicates_z",
        passed=True,
        detail=f"All {len(pairs)} (On …) predicates satisfy Z dominance.",
        payload={"pairs": pairs, "tol": tol},
    )


# ---------------------------------------------------------------------------
# B5 — goal must be False at reset (else task is trivially solved)
# ---------------------------------------------------------------------------


def assert_goal_false_at_reset(
    bddl: Any,
    env: Any,
    *,
    goal_evaluator: Callable[[Any, Any], bool] | None = None,
) -> AssertionResult:
    """The BDDL ``:goal`` predicate must be ``False`` at ``t=0``.

    Resolution order for the evaluator:
        1. Caller-supplied ``goal_evaluator(bddl, env) -> bool``.
        2. ``env.check_success()``.
        3. ``env._check_success()``.
        4. Skip (``passed=None``) — we will not silently assume success/failure.
    """
    if goal_evaluator is not None:
        result = goal_evaluator(bddl, env)
    elif env is not None and callable(getattr(env, "check_success", None)):
        result = env.check_success()
    elif env is not None and callable(getattr(env, "_check_success", None)):
        result = env._check_success()
    else:
        return AssertionResult(
            name="goal_false_at_reset",
            passed=None,
            detail="No goal_evaluator and env exposes no check_success method.",
            payload={},
        )
    truthy = bool(result)
    if truthy:
        return AssertionResult(
            name="goal_false_at_reset",
            passed=False,
            detail="Goal predicate evaluates True at t=0 — task is trivially solved.",
            payload={"goal_text": getattr(bddl, "goal_text", ""), "result": truthy},
        )
    return AssertionResult(
        name="goal_false_at_reset",
        passed=True,
        detail="Goal predicate is False at t=0.",
        payload={"goal_text": getattr(bddl, "goal_text", "")},
    )


# ---------------------------------------------------------------------------
# B6 — soft goal reachability heuristic
# ---------------------------------------------------------------------------


_PRED_NAME_RE = re.compile(r"\(\s*([A-Za-z][A-Za-z0-9_-]*)\s+([^()]*)\)")


def _extract_goal_object_refs(goal_text: str) -> set[str]:
    """Return the set of object identifiers referenced inside goal predicates.

    Heuristic: tokenize each ``(predicate args …)`` form and collect bare
    identifiers (``[A-Za-z_][A-Za-z0-9_]*``) that are not numbers or quoted.
    """
    refs: set[str] = set()
    for m in _PRED_NAME_RE.finditer(goal_text or ""):
        args = m.group(2).strip().split()
        for tok in args:
            tok = tok.strip("()")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
                refs.add(tok)
    return refs


def _xy_overlaps(a: Any, b: Any) -> bool:
    """Return True if AABBs of ``a`` and ``b`` overlap in xy."""
    aa = getattr(a, "aabb", None)
    ba = getattr(b, "aabb", None)
    if aa is None or ba is None:
        return False
    ax0, ax1, ay0, ay1, _, _ = aa
    bx0, bx1, by0, by1, _, _ = ba
    return (ax0 <= bx1) and (bx0 <= ax1) and (ay0 <= by1) and (by0 <= ay1)


def assert_goal_reachable_soft(bddl: Any, scene: Any) -> AssertionResult:
    """Lightweight feasibility hint for the goal predicate.

    Heuristic (documented, not a planner):

    1. Collect identifiers referenced inside the ``:goal`` block.
    2. For each identifier that matches a scene object name, require:
       a. the object exists in the scene; and
       b. no *fixed* geometry (``is_fixed=True``) has an xy-AABB that strictly
          *contains* the object's xy-AABB while being above it in Z (a crude
          "is it buried under static geometry?" probe).

    Identifiers that do not resolve to scene objects (BDDL fixtures, regions,
    predicates with no aabb metadata, …) are recorded in ``payload['unresolved']``
    and do **not** cause failure on their own — this is a soft check. A True
    pass only requires referenced *movable* scene objects to exist and not be
    obviously occluded by fixed geometry.
    """
    goal_text = getattr(bddl, "goal_text", "") or ""
    if not goal_text.strip():
        return AssertionResult(
            name="goal_reachable_soft",
            passed=None,
            detail="BDDL has no :goal block.",
            payload={},
        )
    refs = _extract_goal_object_refs(goal_text)
    if not refs:
        return AssertionResult(
            name="goal_reachable_soft",
            passed=None,
            detail="Could not extract object refs from :goal.",
            payload={"goal_text": goal_text},
        )
    by_name = _scene_by_name(scene)
    fixed = [o for o in _iter_scene_objects(scene) if getattr(o, "is_fixed", False)]
    missing: list[str] = []
    occluded: list[dict[str, Any]] = []
    resolved: list[str] = []
    unresolved: list[str] = []
    for ref in sorted(refs):
        obj = by_name.get(ref)
        if obj is None:
            unresolved.append(ref)
            continue
        resolved.append(ref)
        obj_z_top = _obj_z_top(obj) or 0.0
        for fx in fixed:
            if fx is obj:
                continue
            fx_aabb = getattr(fx, "aabb", None)
            obj_aabb = getattr(obj, "aabb", None)
            if fx_aabb is None or obj_aabb is None:
                continue
            fx0, fx1, fy0, fy1, fz0, _ = fx_aabb
            ox0, ox1, oy0, oy1, _, _ = obj_aabb
            contains_xy = (fx0 <= ox0) and (ox1 <= fx1) and (fy0 <= oy0) and (oy1 <= fy1)
            if contains_xy and fz0 > obj_z_top:
                occluded.append({"object": ref, "occluder": getattr(fx, "name", "?")})
    if missing or occluded:
        return AssertionResult(
            name="goal_reachable_soft",
            passed=False,
            detail=(f"Soft reachability failed: missing={missing} occluded={occluded}"),
            payload={
                "missing": missing,
                "occluded": occluded,
                "refs": sorted(refs),
                "unresolved": unresolved,
                "heuristic": (
                    "Goal refs that resolve to scene objects must exist and not "
                    "lie under any fixed geometry whose xy-AABB strictly contains "
                    "them and sits above their Z-top. Unresolved refs (fixtures, "
                    "regions) are reported but do not fail this check."
                ),
            },
        )
    return AssertionResult(
        name="goal_reachable_soft",
        passed=True,
        detail=(
            f"Soft reachability OK ({len(resolved)} resolved, {len(unresolved)} unresolved refs)."
        ),
        payload={
            "resolved": resolved,
            "unresolved": unresolved,
            "heuristic": ("xy-AABB containment + Z-occlusion probe; documented soft check."),
        },
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


DOMAIN_ASSERTIONS: tuple[str, ...] = (
    "bddl_objects_present",
    "assets_in_registry",
    "no_initial_collisions",
    "on_predicates_z",
    "goal_false_at_reset",
    "goal_reachable_soft",
)


def assert_domain(
    bddl: Any,
    scene: Any,
    *,
    registry: Iterable[str] | None = None,
    mjmodel: Any | None = None,
    mjdata: Any | None = None,
    env: Any | None = None,
    goal_evaluator: Callable[[Any, Any], bool] | None = None,
    tol: float = DEFAULT_POS_TOL,
) -> list[AssertionResult]:
    """Run all Family-B domain invariants and return their results in order."""
    return [
        assert_bddl_objects_present(bddl, scene),
        assert_assets_in_registry(scene, registry),
        assert_no_initial_collisions(scene, mjmodel, mjdata, tol=tol),
        assert_on_predicates_z(bddl, scene, tol=tol),
        assert_goal_false_at_reset(bddl, env, goal_evaluator=goal_evaluator),
        assert_goal_reachable_soft(bddl, scene),
    ]
