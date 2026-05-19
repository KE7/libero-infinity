"""Validation invariants (G4 families).

* Family A — identity      (``identity``)
* Family B — domain        (``domain``)
* Family C — consistency   (``consistency``)
* Family D — affordance    (``affordance``, cheap-only)

Family A's ``AssertionResult`` is a historical dataclass with ``delta``;
families B/C/D share a separate ``AssertionResult`` with ``payload`` (see
``_result.AssertionResult``) — the schemas serve different aggregation needs.
The combined sweep hook ``g4_domain_consistency_hook`` operates on the B/C/D
variant.
"""

from typing import Any, Callable, Iterable, Mapping

from ._result import AssertionResult as BCDAssertionResult
from .affordance import (
    AFFORDANCE_ASSERTIONS,
    assert_aabb_clear_around_grasp,
    assert_affordance,
)
from .consistency import (
    CONSISTENCY_ASSERTIONS,
    assert_class_match,
    assert_consistency,
    assert_pose_tolerance,
)
from .domain import (
    DOMAIN_ASSERTIONS,
    assert_assets_in_registry,
    assert_bddl_objects_present,
    assert_domain,
    assert_goal_false_at_reset,
    assert_goal_reachable_soft,
    assert_no_initial_collisions,
    assert_on_predicates_z,
)
from .identity import (
    IDENTITY_ASSERTIONS,
    AssertionResult,
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


def g4_domain_consistency_hook(
    scene: Any,
    env: Any,
    bddl: Any,
    registry: Any = None,
    *,
    mjmodel: Any = None,
    mjdata: Any = None,
    goal_evaluator: Callable[[Any, Any], bool] | None = None,
    grasp_points: Mapping[str, tuple[float, float, float]] | None = None,
    pos_tol: float = 5e-3,
    rot_tol_deg: float = 1.0,
    domain_tol: float = 1e-4,
) -> dict[str, BCDAssertionResult]:
    """Run G4 families B (domain), C (consistency), D (affordance) on a condition.

    Intended to be called from the sweep harness *after* G5 (env reset).
    Returns a flat dict keyed by an assertion identifier; per-object results
    (consistency, affordance) are disambiguated by the object name.

    Naming scheme:
        ``"domain:<assertion-name>"``
        ``"consistency:<assertion-name>:<object-name>"``
        ``"affordance:<assertion-name>:<object-name>"``
    """
    out: dict[str, BCDAssertionResult] = {}

    # Family B
    registry_iter: Iterable[str] | None
    if registry is None:
        registry_iter = None
    elif isinstance(registry, Mapping):
        registry_iter = registry.keys()
    elif hasattr(registry, "__iter__"):
        registry_iter = registry
    else:
        registry_iter = None

    for res in assert_domain(
        bddl,
        scene,
        registry=registry_iter,
        mjmodel=mjmodel,
        mjdata=mjdata,
        env=env,
        goal_evaluator=goal_evaluator,
        tol=domain_tol,
    ):
        out[f"domain:{res.name}"] = res

    # Family C — one (pose, class) pair per object; tag with object name from payload
    for res in assert_consistency(scene, env, pos_tol=pos_tol, rot_tol_deg=rot_tol_deg):
        nm = res.payload.get("name", "?")
        out[f"consistency:{res.name}:{nm}"] = res

    # Family D
    for res in assert_affordance(scene, registry, grasp_points=grasp_points):
        nm = res.payload.get("name", "?")
        out[f"affordance:{res.name}:{nm}"] = res

    return out


__all__ = [
    # Family A re-exports (legacy)
    "AssertionResult",
    "IDENTITY_ASSERTIONS",
    "assert_all_identities",
    "assert_articulation_unchanged",
    "assert_background_unchanged",
    "assert_camera_unchanged",
    "assert_distractor_unchanged",
    "assert_lighting_unchanged",
    "assert_object_unchanged",
    "assert_position_unchanged",
    "assert_robot_unchanged",
    "assert_texture_unchanged",
    "g4_identity_hook",
    # Families B/C/D
    "BCDAssertionResult",
    "DOMAIN_ASSERTIONS",
    "CONSISTENCY_ASSERTIONS",
    "AFFORDANCE_ASSERTIONS",
    "assert_domain",
    "assert_bddl_objects_present",
    "assert_assets_in_registry",
    "assert_no_initial_collisions",
    "assert_on_predicates_z",
    "assert_goal_false_at_reset",
    "assert_goal_reachable_soft",
    "assert_consistency",
    "assert_pose_tolerance",
    "assert_class_match",
    "assert_affordance",
    "assert_aabb_clear_around_grasp",
    "g4_domain_consistency_hook",
]
