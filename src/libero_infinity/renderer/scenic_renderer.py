"""Pure Scenic 3 renderer for the Libero-Infinity compiler pipeline.

PURITY INVARIANT: This module contains zero conditional logic based on task
semantics. No ``if fixture_class == "..."`` or ``if object_class in {...}``.
All task-semantic decisions live in the semantic graph builder and planner.
All perturbation decisions live in the planner.
This renderer is a deterministic function of the plan IR alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from libero_infinity.asset_metadata import TABLE_SURFACE_Z, surface_spawn_z
from libero_infinity.asset_registry import get_dimensions
from libero_infinity.ir.nodes import (
    FixtureNode,
    MovableSupportNode,
    ObjectNode,
)
from libero_infinity.ir.scene_graph import SemanticSceneGraph
from libero_infinity.planner.types import PerturbationPlan
from libero_infinity.robot_metadata import RobotFootprint, RobotLink, get_robot_footprint

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Semantic support / containment relations
# ---------------------------------------------------------------------------
#
# A "declared support relation" is a (child, support) pair where the BDDL
# author asserted that ``child`` rests on / inside / stacked-on ``support``.
# Such pairs are encoded in the SemanticSceneGraph as ``supported_by``,
# ``contained_in``, or ``stacked_on`` edges with a matching ``spatial_kind``.
#
# When the position-axis planner emits ``use_relative_positioning=True`` for
# such a child, the renderer expresses that relation in Scenic via the
# specifier
#
#     at <support> offset by Vector(Range(<dx_lo>, <dx_hi>),
#                                   Range(<dy_lo>, <dy_hi>), 0.0)
#
# where the offset envelope is already clipped (in
# ``planner/position.py``) to ``support_half_extents - child_half_extents``.
# That specifier therefore *is* the positive support / containment
# constraint — the child's footprint is constructively pinned inside the
# support region by the offset Range.
#
# What the renderer must NOT do, then, is emit any *contradictory* clearance
# constraint between the same child and support — e.g.,
# ``require (distance from child to support) > clearance`` — because that
# would carve out a measure-zero feasible region and Scenic would reject
# every sample.
#
# The helpers below make this explicit: ``_collect_support_relations``
# materializes the declared (child, support, kind) tuples, and
# ``_should_emit_fixture_clearance`` is the only place that decides whether
# an object↔fixture distance constraint is safe to emit.


@dataclass(frozen=True)
class _SupportRelation:
    """A renderer-visible declared support / containment relation.

    Attributes:
        child_var:        Scenic variable name for the child object.
        support_var:      Scenic variable name for the support entity.
        child_name:       Original instance name of the child.
        support_name:     Original instance name of the support entity.
        kind:             One of ``"on_surface"`` (resting on a fixture/movable
                          surface), ``"inside"`` (contained within a fixture
                          cavity such as a drawer), ``"stacked"`` (sitting on
                          top of another movable object), or ``"unknown"`` if
                          no matching IR edge was found (defensive fallback).
        support_is_fixture: True iff the support is an immobile FixtureNode.
                          Distinguishes fixture-anchored placements (cabinet,
                          stove) from movable supports (cookies_box).
    """

    child_var: str
    support_var: str
    child_name: str
    support_name: str
    kind: str
    support_is_fixture: bool


def _collect_support_relations(
    plan: PerturbationPlan, graph: SemanticSceneGraph
) -> list[_SupportRelation]:
    """Materialize declared support / containment relations.

    A relation is recorded for every ``(child, support)`` pair where the
    renderer will emit a relative-positioning specifier (i.e., the planner
    produced a ``PositionPlan`` with ``use_relative_positioning=True`` and a
    non-empty ``support_name``).  The semantic kind is read from the matching
    SceneEdge label so downstream logic can distinguish drawer/containment
    placements from open-top support placements.

    The renderer is a pure function of plan + graph (purity invariant), so
    no I/O or hidden side-state is involved — the result is fully determined
    by the inputs.
    """
    relations: list[_SupportRelation] = []
    edge_kind_by_label = {
        "supported_by": "on_surface",
        "contained_in": "inside",
        "stacked_on": "stacked",
    }
    seen: set[tuple[str, str]] = set()

    # 1. Relations implied by the position plan (relative-positioning entries).
    for child_name, pp in plan.position_plans.items():
        if pp is None:
            continue
        if not (pp.use_relative_positioning and pp.support_name):
            continue
        kind = "unknown"
        for edge in graph.edges_from(child_name):
            if edge.dst_id != pp.support_name:
                continue
            if edge.label in edge_kind_by_label:
                kind = edge_kind_by_label[edge.label]
                break
        support_node = graph.get_node(pp.support_name)
        is_fixture = isinstance(support_node, FixtureNode)
        seen.add((child_name, pp.support_name))
        relations.append(
            _SupportRelation(
                child_var=_to_var(child_name),
                support_var=_to_var(pp.support_name),
                child_name=child_name,
                support_name=pp.support_name,
                kind=kind,
                support_is_fixture=is_fixture,
            )
        )

    # 2. Declared-but-not-relative support edges. The BDDL author asserted
    # the child rests on / sits inside the support; even if the planner
    # chose absolute (workspace-coord) sampling for the child, the pair is
    # still semantically a support relation. We must suppress the
    # ``distance(child, fixture) > clearance`` constraint for it — that
    # constraint would forbid the BDDL-declared placement entirely (e.g.
    # ``On(bowl, cabinet_top_side)`` with bowl sampled near cabinet xy).
    for edge in graph.edges:
        if edge.label not in edge_kind_by_label:
            continue
        if (edge.src_id, edge.dst_id) in seen:
            continue
        support_node = graph.get_node(edge.dst_id)
        is_fixture = isinstance(support_node, FixtureNode)
        seen.add((edge.src_id, edge.dst_id))
        relations.append(
            _SupportRelation(
                child_var=_to_var(edge.src_id),
                support_var=_to_var(edge.dst_id),
                child_name=edge.src_id,
                support_name=edge.dst_id,
                kind=edge_kind_by_label[edge.label],
                support_is_fixture=is_fixture,
            )
        )
    return relations


def _is_declared_support_pair(var_a: str, var_b: str, relations: list[_SupportRelation]) -> bool:
    """Return True if ``{var_a, var_b}`` are a declared (child, support) pair.

    Order-insensitive: works for both the object-fixture clearance loop
    (where ``var_a`` is the child and ``var_b`` is the fixture) and the
    object-object clearance loop (where ``var_a`` is the child and ``var_b``
    is its movable support — e.g., a bowl stacked on a cookies box).
    """
    for rel in relations:
        if (rel.child_var == var_a and rel.support_var == var_b) or (
            rel.child_var == var_b and rel.support_var == var_a
        ):
            return True
    return False


def _should_emit_fixture_clearance(
    obj_var: str,
    obj_sampled: bool,
    fixture_var: str,
    relations: list[_SupportRelation],
) -> bool:
    """Decide whether to emit ``distance(obj, fixture) > clearance``.

    Semantics:
    1. If ``obj`` is not Scenic-sampled — i.e. it carries the BDDL author's
       canonical (x, y) position verbatim — trust the author and emit
       nothing.  The renderer's clearance threshold is a conservative AABB
       diagonal that frequently overlaps an author's hand-placed bowl/plate.
    2. If ``obj`` has a declared support relation against ``fixture`` (any
       kind: on_surface / inside / stacked-via-fixture-anchored-support),
       the relative-position specifier already pins ``obj`` *inside* the
       fixture footprint.  Emitting ``distance > clearance`` in that case
       is unsatisfiable — every Scenic sample would be rejected.
    3. Otherwise the pair is unrelated and we emit the standard footprint
       clearance constraint to prevent geometric overlap.
    """
    if not obj_sampled:
        return False
    if _is_declared_support_pair(obj_var, fixture_var, relations):
        return False
    return True


# ---------------------------------------------------------------------------
# Fixture footprint dimensions
# ---------------------------------------------------------------------------

# Conservative (width, length, height) estimates for LIBERO fixture classes.
# Used for object-fixture clearance constraints.  Also exported via compiler.py
# as _FIXTURE_DIMENSIONS for callers that need fixture footprint data.
_FIXTURE_DIMS: dict[str, tuple[float, float, float]] = {
    # MuJoCo-measured geom extents including door handles and protruding parts:
    # wooden_cabinet: door handle extends to y=+0.130 m from centre → use (0.30, 0.30)
    "wooden_cabinet": (0.30, 0.30, 0.24),
    "white_cabinet": (0.30, 0.30, 0.24),
    # flat_stove: x extends 0.15 m to the right of centre → use (0.36, 0.20)
    "flat_stove": (0.36, 0.20, 0.08),
    "wine_rack": (0.18, 0.12, 0.20),
    "microwave": (0.24, 0.18, 0.16),
    "bowl_drainer": (0.18, 0.14, 0.08),
    # desk_caddy: corrected to match MJCF geom-union AABB (was 0.14, 0.10, 0.06,
    # off by factor ~4 in y and ~3.5 in z) — see rca/g5_settle_drift_caddy.md §3.
    "desk_caddy": (0.14, 0.42, 0.22),
    "wooden_two_layer_shelf": (0.33, 0.20, 0.21),
    "table": (0.80, 0.60, 0.05),
    "kitchen_table": (0.80, 0.60, 0.05),
    "living_room_table": (0.55, 0.65, 0.05),
    "study_table": (0.50, 0.58, 0.05),
    "floor": (0.50, 0.55, 0.01),
}
_FIXTURE_DIM_DEFAULT = (0.20, 0.18, 0.18)  # conservative fallback


def _fixture_dims(fixture_class: str | None) -> tuple[float, float, float]:
    """Return (width, length, height) for a fixture class."""
    return _FIXTURE_DIMS.get(fixture_class or "", _FIXTURE_DIM_DEFAULT)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _footprint_clearance_xy(
    dims_a: tuple[float, float, float],
    dims_b: tuple[float, float, float],
) -> float:
    """Minimum centre-to-centre **radial** xy distance before two footprints overlap.

    Computes the diagonal radius of each object's xy footprint and returns
    their sum — the minimum separation needed to guarantee no overlap from
    *any* approach angle.  Duplicated from simulator.py to avoid circular
    imports.

    NOTE: For *axis-aligned* AABBs (which is what every LIBERO fixture and
    object uses), this radial form is unnecessarily conservative by a factor
    of ~√2 in the symmetric case (and worse for elongated fixtures such as
    flat_stove or desk_caddy).  The separating-axis theorem gives an *exact*
    AABB non-overlap test using only per-axis half-width sums
    (``_footprint_clearance_aabb`` below).  This radial form is retained for
    callers that need a single scalar clearance (e.g. settle-drift tolerance
    checks); the Scenic ``require`` emitters in ``_render_constraints`` use
    the SAT-correct OR-form instead, mirroring the object↔object pair form.
    """
    radius_a = math.hypot(dims_a[0], dims_a[1]) / 2.0
    radius_b = math.hypot(dims_b[0], dims_b[1]) / 2.0
    return radius_a + radius_b


def _footprint_clearance_aabb(
    dims_a: tuple[float, float, float],
    dims_b: tuple[float, float, float],
) -> tuple[float, float]:
    """SAT-correct per-axis half-width-sum clearance for two AABBs.

    Returns ``(dx_min, dy_min)`` such that two axis-aligned bounding boxes
    of footprint ``dims_a`` and ``dims_b`` are non-overlapping iff::

        abs(xa - xb) > dx_min  OR  abs(ya - yb) > dy_min

    This is the standard separating-axis theorem for axis-aligned rectangles
    and is the tightest possible non-overlap condition.
    """
    dx_min = (dims_a[0] + dims_b[0]) / 2.0
    dy_min = (dims_a[1] + dims_b[1]) / 2.0
    return dx_min, dy_min


# ---------------------------------------------------------------------------
# Well-formedness check
# ---------------------------------------------------------------------------


def _check_wellformed(plan: PerturbationPlan, graph: SemanticSceneGraph) -> None:
    """Raise ValueError if the plan or graph is malformed."""
    if not graph.nodes:
        raise ValueError("SemanticSceneGraph has no nodes — cannot render")
    if plan.diagnostics is None:
        raise ValueError("PerturbationPlan missing diagnostics — cannot render")
    if not graph.task_language:
        raise ValueError("SemanticSceneGraph has no task_language")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_scenic(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    """Render a PerturbationPlan + SemanticSceneGraph to a Scenic 3 program.

    This is a pure function:
    - No side effects.
    - Zero conditional logic based on task-semantic class names.
    - Deterministic: identical input → identical output.

    Args:
        plan: The perturbation plan produced by plan_perturbations().
        graph: The semantic scene graph produced by build_semantic_scene_graph().

    Returns:
        A valid Scenic 3 program string.

    Raises:
        ValueError: If the plan or graph fails the well-formedness check.
    """
    _check_wellformed(plan, graph)

    fragments: list[str] = []

    fragments.append(_render_header(graph))
    fragments.append(_render_global_params(plan, graph))
    fragments.append(_render_fixtures(plan, graph))
    fragments.append(_render_objects(plan, graph))
    fragments.append(_render_articulation(plan, graph))
    fragments.append(_render_robot(plan, graph))
    fragments.append(_render_camera(plan, graph))
    fragments.append(_render_lighting(plan, graph))
    fragments.append(_render_texture(plan, graph))
    fragments.append(_render_background(plan, graph))
    fragments.append(_render_distractors(plan, graph))
    fragments.append(_render_sensor_noise(plan, graph))
    fragments.append(_render_constraints(plan, graph))
    fragments.append(_render_visibility(plan, graph))

    return "\n".join(f for f in fragments if f)


# ---------------------------------------------------------------------------
# Fragment renderers (each is a pure function of plan + graph)
# ---------------------------------------------------------------------------


def _render_header(graph: SemanticSceneGraph) -> str:
    lang = graph.task_language.replace('"', '\\"')
    bddl = graph.bddl_path.replace('"', '\\"')
    return (
        f'"""Auto-generated Scenic program for: {lang}"""\n'
        f"\n"
        f"model libero_model\n"
        f"\n"
        f'param task = "{lang}"\n'
        f'param bddl_path = "{bddl}"\n'
    )


def _render_global_params(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    lines = [
        "",
        "# Perturbation parameters",
        "param ood_margin = 0.15",
        "",
        "_ood_margin = globalParameters.ood_margin",
        "",
    ]
    # Expose active axes in scene params
    axes_str = ",".join(sorted(plan.active_axes))
    lines.append(f'param active_axes = "{axes_str}"')
    lines.append("")
    return "\n".join(lines)


def _render_fixtures(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    lines = ["# Fixture declarations"]
    for node_id, node in graph.nodes.items():
        if not isinstance(node, FixtureNode):
            continue
        if node.init_x is None or node.init_y is None:
            continue
        x = node.init_x
        y = node.init_y
        # Use TABLE_Z as placeholder — simulator overrides with actual z
        var_name = _to_var(node.instance_name)
        # Scenic 3: specifiers are comma-separated; libero_name is the declared property
        lines.append(
            f"{var_name} = new LIBEROFixture "
            f"at Vector({x:.4f}, {y:.4f}, TABLE_Z), "
            f'with libero_name "{node.instance_name}"'
        )
    lines.append("")
    return "\n".join(lines)


def _resolve_surface_class(
    node: "ObjectNode | MovableSupportNode",
    plan: PerturbationPlan,
    graph: SemanticSceneGraph,
) -> str | None:
    """Resolve the class of the support surface ``node`` rests on, at codegen.

    The support is known from the BDDL ``:init`` semantics carried in the plan /
    graph: the position plan's ``support_name`` (when the planner anchored the
    object to a support) or, failing that, a declared ``supported_by`` /
    ``contained_in`` / ``stacked_on`` edge. Returns the support's object class
    (e.g. ``"flat_stove"``, ``"wooden_cabinet"``) so the per-(variant, surface)
    clearance table can resolve the object's settled z for *this* surface
    context (Finding A: clearance is not surface-invariant). Returns ``None`` for
    objects resting directly on the default workspace table — the legacy
    class-only clearance applies there.
    """
    pp = plan.position_plans.get(node.instance_name)
    support_name = pp.support_name if (pp is not None and pp.support_name) else None
    if support_name is None:
        # Try the node's own id and its instance name (the graph may key the
        # node under either) when scanning declared support/containment edges.
        candidate_ids = [getattr(node, "node_id", None), node.instance_name]
        for cid in candidate_ids:
            if cid is None:
                continue
            for edge in graph.edges_from(cid):
                if edge.label in {"supported_by", "contained_in", "stacked_on"}:
                    support_name = edge.dst_id
                    break
            if support_name is not None:
                break
    if support_name is None:
        return None
    support_node = graph.get_node(support_name)
    if support_node is None:
        return None
    return support_node.object_class or None


def _spawn_z_expr(
    obj_class: str,
    surface_class: str | None,
    scenic_class: str | None,
    variants: list[str] | None,
) -> str:
    """Return a Scenic expression string for an object's resolved spawn z.

    When the ``object`` axis substitutes this object's identity (``scenic_class``
    is the ``_chosen_<class>`` sampler and ``variants`` is its pool), the z is
    emitted as a conditional chain on the *same* sampled variable, so the chosen
    variant carries ITS measured seating height on the current surface — Scenic
    picks the (class, z) pair together (Fix 3 / Finding A). Otherwise a single
    measured float is emitted, as before.
    """
    if scenic_class and variants:
        # The variant chooser is a Uniform over (class, z) pairs, so the chosen
        # variant's measured spawn z is the second element of the SAME sample.
        # Indexing with a constant subscript is a deterministic op on a random
        # value (allowed by Scenic), unlike an `if`/`==` branch (forbidden).
        del obj_class, surface_class  # folded into the pair at chooser build time
        return f"{scenic_class}[1]"
    return f"{surface_spawn_z(TABLE_SURFACE_Z, obj_class, surface_class):.4f}"


def _render_objects(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    lines = ["# Object declarations"]

    # Asset variant sampling (object axis)
    seen_classes: set[str] = set()
    asset_var_map: dict[str, str] = {}  # object_class -> scenic var name

    if "object" in plan.active_axes and plan.object_substitutions:
        lines.append("# Asset variant sampling")
        # Each chooser is a single Uniform over (asset_class, resolved_spawn_z)
        # PAIRS, so the chosen variant and its measured seating height are drawn
        # together from ONE sample (Scenic forbids branching on a random value,
        # so a per-variant ternary is illegal; correlated tuples are the
        # idiomatic substitute). ``_chosen_X[0]`` is the asset class; the
        # object's z spec reads ``_chosen_X[1]`` (Fix 3 / Finding A). The z is
        # resolved against the support surface of the first object of this class.
        for obj_name, variants in plan.object_substitutions.items():
            node = graph.get_node(obj_name)
            if node is None:
                continue
            obj_class = node.object_class or obj_name
            if obj_class in seen_classes:
                continue
            seen_classes.add(obj_class)
            var_name = f"_chosen_{_sanitize(obj_class)}"
            surface_class = _resolve_surface_class(node, plan, graph)
            pairs = ", ".join(
                f'("{v}", {surface_spawn_z(TABLE_SURFACE_Z, v, surface_class):.4f})'
                for v in variants
            )
            lines.append(f"{var_name} = Uniform({pairs})")
            asset_var_map[obj_class] = var_name

        if asset_var_map:
            first_class = next(iter(asset_var_map))
            first_var = asset_var_map[first_class]
            lines.append(f'param perturb_class = "{first_class}"')
            lines.append(f"param chosen_asset = {first_var}[0]")
        lines.append("")

    # Object placements — topologically sorted so support objects are
    # declared before any object that references them via relative positioning.
    raw_obj_nodes: list[tuple[str, object]] = [
        (nid, n)
        for nid, n in graph.nodes.items()
        if isinstance(n, (ObjectNode, MovableSupportNode))
    ]
    # Build dependency map: obj_name -> support_name (or None)
    _dep: dict[str, str | None] = {}
    for _nid, _n in raw_obj_nodes:
        pp = plan.position_plans.get(_n.instance_name)
        _dep[_n.instance_name] = (
            pp.support_name if (pp is not None and pp.use_relative_positioning) else None
        )
    # Kahn's algorithm for topological sort
    _name_to_entry = {n.instance_name: (nid, n) for nid, n in raw_obj_nodes}
    _in_degree: dict[str, int] = {
        name: (1 if dep is not None and dep in _name_to_entry else 0) for name, dep in _dep.items()
    }
    _children: dict[str, list[str]] = {name: [] for name in _dep}
    for name, dep in _dep.items():
        if dep is not None and dep in _children:
            _children[dep].append(name)
    from collections import deque

    _queue: deque[str] = deque(n for n, d in _in_degree.items() if d == 0)
    _sorted_names: list[str] = []
    while _queue:
        _cur = _queue.popleft()
        _sorted_names.append(_cur)
        for _child in _children.get(_cur, []):
            _in_degree[_child] -= 1
            if _in_degree[_child] == 0:
                _queue.append(_child)
    # Fall back to original order if cycle detected (shouldn't happen)
    if len(_sorted_names) != len(raw_obj_nodes):
        _sorted_names = [n.instance_name for _, n in raw_obj_nodes]
    sorted_obj_nodes = [_name_to_entry[name] for name in _sorted_names]

    for node_id, node in sorted_obj_nodes:
        obj_name = node.instance_name
        obj_class = node.object_class or obj_name
        var_name = _to_var(obj_name)
        scenic_class = asset_var_map.get(obj_class)

        # Position plan
        pos_plan = plan.position_plans.get(obj_name)

        # Resolve the spawn z that the simulator will settle this object to, at
        # codegen, so the Scenic-sampled pose matches the post-reset MuJoCo pose
        # in the SAME frame (G4 family-C pose_tolerance; validation plan §4).
        # Previously every object was emitted at the bare ``TABLE_Z`` placeholder
        # and the simulator silently overrode the z, so pose_tolerance failed on
        # an 8–18 cm z-frame mismatch. We now emit the concrete resolved z via
        # the shared ``surface_spawn_z`` helper — the same function the simulator
        # calls — so the override becomes a no-op for agreeing objects.
        #
        # Fix 3 (Finding A): the clearance is resolved against the object's
        # ACTUAL support surface (stove vs cabinet top settle ~50 mm apart), and
        # under an active ``object`` axis the spawn z is emitted as a conditional
        # on the sampled variant identity, so the instantiated variant carries
        # its OWN measured seating height instead of inheriting the canonical
        # class's z. Scenic thus picks the (class, z) pair together.
        surface_class = _resolve_surface_class(node, plan, graph)
        object_axis_variants = (
            plan.object_substitutions.get(obj_name) if "object" in plan.active_axes else None
        )
        spawn_z_expr = _spawn_z_expr(obj_class, surface_class, scenic_class, object_axis_variants)

        if pos_plan is not None and not pos_plan.use_relative_positioning:
            x_lo = pos_plan.x_envelope.lo
            x_hi = pos_plan.x_envelope.hi
            y_lo = pos_plan.y_envelope.lo
            y_hi = pos_plan.y_envelope.hi
            pos_spec = (
                f"at Vector(Range({x_lo:.4f}, {x_hi:.4f}), "
                f"Range({y_lo:.4f}, {y_hi:.4f}), {spawn_z_expr})"
            )
        elif pos_plan is not None and pos_plan.use_relative_positioning:
            support_var = _to_var(pos_plan.support_name)
            x_lo = pos_plan.x_envelope.lo
            x_hi = pos_plan.x_envelope.hi
            y_lo = pos_plan.y_envelope.lo
            y_hi = pos_plan.y_envelope.hi
            # Relative placement inherits z from the support (offset 0.0): a
            # supported/contained child's z derives from its support relation,
            # not the table surface, so the resolved spawn z does not apply here.
            pos_spec = (
                f"at {support_var} offset by Vector(Range({x_lo:.4f}, {x_hi:.4f}), "
                f"Range({y_lo:.4f}, {y_hi:.4f}), 0.0)"
            )
        elif node.init_x is not None and node.init_y is not None:
            pos_spec = f"at Vector({node.init_x:.4f}, {node.init_y:.4f}, {spawn_z_expr})"
        else:
            # No position info — use workspace center
            pos_spec = "in SAFE_REGION"

        # Build specifier list — Scenic 3 requires comma-separated specifiers.
        # libero_name is the declared property on LIBEROObject (not 'name').
        specifiers: list[str] = [pos_spec]
        # asset_class MUST be emitted on every generated object so the G4
        # family-B/C/D invariants (assets_in_registry, class_match, affordance
        # class lookup) can read the real instantiated asset class. When the
        # ``object`` axis is active and this object's class was substituted to
        # a sampled OOD variant, asset_class is bound to the variant chooser
        # (``_chosen_<class>``) — i.e. it reports the asset that was actually
        # instantiated, not the canonical one. Otherwise it is the canonical
        # asset class from the TaskConfig / scene graph.
        if scenic_class:
            # ``_chosen_X`` is a (class, z) pair; element [0] is the class.
            specifiers.append(f"with asset_class {scenic_class}[0]")
        else:
            specifiers.append(f'with asset_class "{obj_class}"')
        specifiers.append(f'with libero_name "{obj_name}"')
        # support_parent_name is read by the simulator both to skip
        # footprint-overlap validation between the object and its declared
        # support, and (when the object is treated as a "supported child"
        # in the simulator's z-snap logic) to lift the object above the
        # support's AABB top. The latter behaviour is gated on the support
        # NOT being a fixed FixtureNode — see ``simulator.py`` use of
        # ``support_parent_name``. We emit the property whenever a support
        # is known so the validator skips the AABB-overlap reject for the
        # author-declared pair, even when the planner chose absolute (not
        # relative) sampling for the child.
        if pos_plan is not None and pos_plan.support_name:
            specifiers.append(f'with support_parent_name "{pos_plan.support_name}"')
        # Emit the resolved support-surface class so the simulator resolves the
        # object's settle z through the SAME per-(variant, surface) clearance the
        # renderer used for spawn_z_expr above — keeping the two in lockstep
        # (Fix 3 / Finding A). Empty/None means the default workspace table.
        if surface_class:
            specifiers.append(f'with support_surface_class "{surface_class}"')

        lines.append(f"{var_name} = new LIBEROObject " + ", ".join(specifiers))

    lines.append("")
    return "\n".join(lines)


def _render_articulation(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    if not plan.articulation_plans:
        return ""
    # When the articulation axis is INACTIVE the plan still carries the
    # baseline articulation values that BDDL :init asserts as task
    # preconditions (e.g. a closed cabinet). Those preconditions must be
    # applied to the simulator regardless of axes, but they must be
    # *deterministic* — emitting them as `Range(lo, hi)` would resample an
    # independent concrete joint angle every time, breaking the G4 identity
    # assertion that the baseline scene and any inactive-axis perturbed
    # scene agree on joint state. Mirror the semantic split PR #6
    # introduced for `plan_articulation_perturbation`: render
    # stochastic Range only when the articulation axis is active.
    art_active = "articulation" in plan.active_axes
    lines = ["# Articulation parameters"]
    for fixture_name, art_plan in plan.articulation_plans.items():
        var_name = _sanitize(fixture_name)
        lo = art_plan.lo
        hi = art_plan.hi
        if art_active and lo != hi:
            lines.append(f"param articulation_{var_name} = Range({lo:.4f}, {hi:.4f})")
        else:
            # Deterministic baseline value. Use lo as the canonical sample
            # so a degenerate range (lo == hi) reduces cleanly; this is the
            # band edge documented in the RCA follow-up.
            lines.append(f"param articulation_{var_name} = {lo:.4f}")
        lines.append(f'param articulation_{var_name}_state = "{art_plan.state_kind}"')
    lines.append("")
    return "\n".join(lines)


def _render_robot(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    del graph
    if plan.robot_plan is None or "robot" not in plan.active_axes:
        return ""
    rp = plan.robot_plan
    canonical_terms = ", ".join(f"{q:.8f}" for q in rp.canonical_qpos)
    lines = [
        "# Robot init perturbation",
        "import math",
        f"_robot_qpos_canonical = [{canonical_terms}]",
        f"_robot_radius = Range({rp.radius_lo:.4f}, {rp.radius_hi:.4f})",
    ]
    # Sample direction components from a standard normal rather than a
    # uniform-on-cube. Projecting cube-uniform draws onto the unit sphere
    # is the canonical biased spherical sampler — corner directions are
    # over-represented and axis-aligned directions under-represented, with
    # the bias growing in higher dimensions. A 7-vector of i.i.d.
    # Normal(0, 1) draws normalised by its L2 norm is the standard
    # uniform-on-S^6 construction (Marsaglia / Müller). The radius envelope
    # is unchanged.
    dir_terms: list[str] = []
    for idx in range(len(rp.canonical_qpos)):
        lines.append(f"_robot_dir_{idx} = Normal(0.0, 1.0)")
        dir_terms.append(f"_robot_dir_{idx}")
    norm_expr = " + ".join(f"({_term} * {_term})" for _term in dir_terms)
    lines.append(f"_robot_dir_norm = (({norm_expr}) + 1e-12) ** 0.5")
    # Per-joint perturbation delta (qpos - canonical), exposed as a reusable
    # Scenic local so the robot-clearance constraints in ``_render_constraints``
    # can express each link's *perturbed* world position as a linear function of
    # the SAME sampled deltas the qpos params consume (Fix 1: robot in the
    # require graph). Keeping one delta local avoids re-deriving the spherical
    # sample and guarantees the constraint graph and the applied qpos agree.
    for idx in range(len(rp.canonical_qpos)):
        lines.append(f"_robot_dq_{idx} = (_robot_radius * _robot_dir_{idx}) / _robot_dir_norm")
    qpos_refs: list[str] = []
    for idx, qpos in enumerate(rp.canonical_qpos):
        expr = f"{qpos:.8f} + _robot_dq_{idx}"
        lines.append(f"param robot_init_qpos_{idx} = {expr}")
        qpos_refs.append(f"globalParameters.robot_init_qpos_{idx}")
    lines.append("param robot_init_radius = _robot_radius")
    lines.append(f'param robot_model = "{rp.robot_model}"')
    lines.append(f"param robot_init_qpos = [{', '.join(qpos_refs)}]")
    lines.append("")
    return "\n".join(lines)


def _render_camera(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    if plan.camera_plan is None or "camera" not in plan.active_axes:
        return ""
    cp = plan.camera_plan
    lines = [
        "# Camera perturbation",
        f"param cam_azimuth = Range({cp.azimuth_lo:.2f}, {cp.azimuth_hi:.2f})",
        f"param cam_elevation = Range({cp.elevation_lo:.2f}, {cp.elevation_hi:.2f})",
        f"param cam_distance = Range({cp.distance_lo:.3f}, {cp.distance_hi:.3f})",
        "",
    ]
    return "\n".join(lines)


def _render_lighting(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    if plan.lighting_plan is None or "lighting" not in plan.active_axes:
        return ""
    lp = plan.lighting_plan
    jitter = lp.position_jitter
    lines = [
        "# Lighting perturbation",
        f"param light_intensity = Range({lp.intensity_lo}, {lp.intensity_hi})",
        f"param light_x_offset = Range({-jitter}, {jitter})",
        f"param light_y_offset = Range({-jitter}, {jitter})",
        f"param light_z_offset = Range({-jitter}, {jitter})",
        f"param ambient_level = Range({lp.ambient_lo}, {lp.ambient_hi})",
        "",
    ]
    return "\n".join(lines)


def _render_texture(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    if plan.texture_plan is None or "texture" not in plan.active_axes:
        return ""
    tp = plan.texture_plan
    # Emit param that the simulator reads in _apply_texture_perturbation().
    # If texture_candidates is non-empty, sample uniformly from that list;
    # otherwise fall back to the table_texture field (typically "random").
    if tp.texture_candidates:
        candidates_str = ", ".join(f'"{c}"' for c in tp.texture_candidates)
        tex_value = f"Uniform({candidates_str})"
    else:
        tex_value = f'"{tp.table_texture}"'
    lines = [
        "# Texture perturbation",
        f"param table_texture = {tex_value}",
        "",
    ]
    return "\n".join(lines)


def _render_background(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    """Render background (wall + floor) texture perturbation params."""
    if plan.background_plan is None or "background" not in plan.active_axes:
        return ""
    bp = plan.background_plan
    if bp.texture_candidates:
        candidates_str = ", ".join(f'"{c}"' for c in bp.texture_candidates)
        wall_val = f"Uniform({candidates_str})"
        floor_val = f"Uniform({candidates_str})"
    else:
        wall_val = f'"{bp.wall_texture}"'
        floor_val = f'"{bp.floor_texture}"'
    lines = [
        "# Background perturbation",
        f"param wall_texture = {wall_val}",
        f"param floor_texture = {floor_val}",
        "",
    ]
    return "\n".join(lines)


def _render_sensor_noise(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    """Emit ``param sensor_noise_kind`` and ``param sensor_noise_severity``.

    Sensor noise is purely an observation-pipeline transform (applied by
    ``simulator._apply_sensor_noise`` to the rendered RGB image at every
    ``step()``); it has no scene-geometry effect, so the only renderer
    output is the two parameters that the simulator consumes.
    """
    del graph
    if plan.sensor_noise_plan is None or "sensor_noise" not in plan.active_axes:
        return ""
    sn = plan.sensor_noise_plan
    kinds_str = ", ".join(f'"{k}"' for k in sn.kinds)
    lines = [
        "# Sensor / image-noise perturbation",
        f"param sensor_noise_kind = Uniform({kinds_str})",
        f"param sensor_noise_severity = DiscreteRange({sn.severity_lo}, {sn.severity_hi})",
        "",
    ]
    return "\n".join(lines)


def _render_distractors(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    del graph
    if plan.distractor_budget <= 0 or "distractor" not in plan.active_axes:
        return ""
    classes = plan.distractor_classes or []
    n = plan.distractor_budget
    lines = [
        "# Distractor objects",
        f"param n_distractors = DiscreteRange(1, {n})",
        "_n_distractors = globalParameters.n_distractors",
    ]
    if classes:
        cls_str = ", ".join(f'"{c}"' for c in classes)
        lines.append(f"_distractor_pool = [{cls_str}]")
    for i in range(n):
        if classes:
            lines.append(f"param distractor_{i}_class = Uniform(*_distractor_pool)")
        # Scenic 3: specifiers are comma-separated; libero_name is the declared property
        specifiers = [
            "in SAFE_REGION",
            f'with libero_name "distractor_{i}"',
            "with width 0.08",
            "with length 0.08",
            "with height 0.08",
            "with preserve_default_z False",
        ]
        if classes:
            specifiers.append(f"with asset_class globalParameters.distractor_{i}_class")
        lines.append(f"distractor_{i} = new LIBEROObject " + ", ".join(specifiers))
    lines.append("")
    return "\n".join(lines)


def _is_sampled(node: "ObjectNode | MovableSupportNode", plan: PerturbationPlan) -> bool:
    """Return True if this object's position is Scenic-sampled (Range or SAFE_REGION).

    An object is "sampled" when the renderer emits a ``Range``-based or
    ``in SAFE_REGION`` placement for it.  This happens when:
    - The planner produced a PositionPlan for this object (position axis active), OR
    - The object has no BDDL init position and the renderer falls back to SAFE_REGION.

    An object is "fixed" when it has a BDDL canonical init position *and* no
    position plan — the renderer emits ``at Vector(x, y, TABLE_Z)`` verbatim.
    """
    if node.instance_name in plan.position_plans:
        return True
    if node.init_x is None or node.init_y is None:
        return True
    return False


def _relative_parent_map(plan: PerturbationPlan) -> dict[str, str]:
    """Map relatively positioned object names to their direct support names."""
    return {
        obj_name: pp.support_name
        for obj_name, pp in plan.position_plans.items()
        if pp is not None and pp.use_relative_positioning and pp.support_name
    }


def _support_relation_label(
    graph: SemanticSceneGraph,
    child_name: str,
    parent_name: str,
) -> str | None:
    child_ids = [child_name]
    child_ids.extend(
        node_id
        for node_id, node in graph.nodes.items()
        if node.instance_name == child_name and node_id != child_name
    )
    for child_id in child_ids:
        edges = graph.edges_from(child_id)
        for edge in edges:
            parent = graph.get_node(edge.dst_id)
            parent_matches = edge.dst_id == parent_name or (
                parent is not None and parent.instance_name == parent_name
            )
            if parent_matches and edge.label in {
                "contained_in",
                "stacked_on",
                "supported_by",
            }:
                return edge.label
    return None


def _clearance_relationship(
    name_a: str,
    name_b: str,
    relative_parent: dict[str, str],
    graph: SemanticSceneGraph,
) -> str:
    """Classify how pairwise clearance should treat two object nodes."""
    parent_a = relative_parent.get(name_a)
    parent_b = relative_parent.get(name_b)

    if parent_a == name_b or parent_b == name_a:
        return "direct_parent"

    if parent_a is None or parent_b is None or parent_a != parent_b:
        return "independent"

    label_a = _support_relation_label(graph, name_a, parent_a)
    label_b = _support_relation_label(graph, name_b, parent_b)
    if label_a is None or label_b is None:
        raise ValueError(
            "Relative-positioned siblings share support "
            f"{parent_a!r}, but graph support edges are missing for "
            f"{name_a!r} and/or {name_b!r}"
        )
    if label_a != label_b:
        raise ValueError(
            "Relative-positioned siblings share support "
            f"{parent_a!r} with incompatible relationships: "
            f"{name_a!r}={label_a}, {name_b!r}={label_b}"
        )
    if label_a == "contained_in":
        return "contained_sibling"
    return "independent"


def _link_pos_exprs(link: RobotLink, n_dof: int) -> tuple[str, str, str]:
    """Scenic expressions for a link's *perturbed* world position.

    Each coordinate is the canonical origin plus the linearized FK contribution
    of the sampled joint deltas: ``c0 + Σ_k J_k * _robot_dq_k`` (near-zero
    Jacobian terms are dropped for compactness). The same ``_robot_dq_k`` locals
    drive the applied ``robot_init_qpos_k`` params, so the constraint graph and
    the simulator's applied pose are derived from one sample.
    """

    def _build(c0: float, jrow: tuple[float, ...]) -> str:
        terms = [f"{c0:.6f}"]
        for k in range(min(n_dof, len(jrow))):
            coef = jrow[k]
            if abs(coef) < 1e-6:
                continue
            terms.append(f"({coef:.6f} * _robot_dq_{k})")
        return " + ".join(terms)

    return _build(link.x0, link.jx), _build(link.y0, link.jy), _build(link.z0, link.jz)


def _render_robot_clearance(
    footprint: RobotFootprint, plan: PerturbationPlan, graph: SemanticSceneGraph
) -> list[str]:
    """Emit 3-D SAT clearance clauses between every movable robot link and every
    placed object / distractor / fixture (Fix 1; RCA Finding B).

    The z term is required, not optional: at the home pose the arm sits ~30 cm
    above the table, so a pure-xy shadow would reject every object beneath the
    (stationary) arm. The z projection only rejects samples where the perturbed
    link actually dips into the target's slab. A static z prune drops (link,
    target) pairs whose measured swept-z ranges can never overlap.
    """
    rp = plan.robot_plan
    if rp is None:
        return []
    active_links = footprint.active_links()
    if not active_links:
        return []
    n_dof = footprint.n_dof or len(rp.canonical_qpos)

    # target = (guard_prefix, cx_expr, cy_expr, cz_expr, thx, thy, thz, z_lo, z_hi)
    targets: list[tuple[str, str, str, str, float, float, float, float | None, float | None]] = []

    # Task objects / movable supports.
    for node_id, node in graph.nodes.items():
        if not isinstance(node, (ObjectNode, MovableSupportNode)):
            continue
        if isinstance(node, ObjectNode) and node.contained:
            continue
        var = _to_var(node.instance_name)
        obj_class = node.object_class or node.instance_name
        # FV MC #6: a substituted OOD variant may be WIDER (and taller) than the
        # canonical class. The clearance half-extents must bound the widest
        # footprint the object axis can instantiate -- mirroring the max-over-pool
        # already done for z (zs) below -- or canonical-only thx/thy admit an AABB
        # overlap for wide variants that the simulator must then shove.
        variants = (
            plan.object_substitutions.get(node.instance_name)
            if "object" in plan.active_axes
            else None
        )
        pool_classes = [obj_class] + list(variants or [])
        thx = max(get_dimensions(c)[0] for c in pool_classes) / 2.0
        thy = max(get_dimensions(c)[1] for c in pool_classes) / 2.0
        thz = max(get_dimensions(c)[2] for c in pool_classes) / 2.0
        pp = plan.position_plans.get(node.instance_name)
        surface_class = _resolve_surface_class(node, plan, graph)
        if pp is not None and pp.use_relative_positioning:
            # Support-inherited z: not statically known, so never prune; the
            # dynamic z term still guards correctly.
            z_lo = z_hi = None
        else:
            zs = [surface_spawn_z(TABLE_SURFACE_Z, obj_class, surface_class)]
            if variants:
                zs.extend(surface_spawn_z(TABLE_SURFACE_Z, v, surface_class) for v in variants)
            z_lo = min(zs) - thz
            z_hi = max(zs) + thz
        targets.append(
            (
                "",
                f"{var}.position.x",
                f"{var}.position.y",
                f"{var}.position.z",
                thx,
                thy,
                thz,
                z_lo,
                z_hi,
            )
        )

    # Distractors (guarded by the active-count gate).
    if plan.distractor_budget > 0 and "distractor" in plan.active_axes:
        dz_lo = footprint.table_world_z
        dz_hi = footprint.table_world_z + 0.08
        # FV SMT Finding G: the simulator instantiates each distractor at its
        # world settle z (surface_spawn_z ~= 0.92), NOT the Scenic SAFE_REGION
        # TABLE_Z (~0.82). Emitting `distractor_{i}.position.z` (the Scenic value)
        # into the robot<->distractor z-term guarded a phantom ~100 mm below
        # reality. Distractors rest on the table (surface_class=None) and their
        # class is sampled at runtime, so emit the canonical-distractor world
        # settle z as a constant; the static z-prune band [table_world_z, +0.08]
        # bounds the residual per-class variation. The constraint variable and the
        # prune band now live in the same (world) frame the simulator uses.
        dz_world = surface_spawn_z(TABLE_SURFACE_Z, "distractor", None)
        for i in range(plan.distractor_budget):
            guard = f"(_n_distractors <= {i}) or "
            targets.append(
                (
                    guard,
                    f"distractor_{i}.position.x",
                    f"distractor_{i}.position.y",
                    f"{dz_world:.4f}",
                    0.04,
                    0.04,
                    0.04,
                    dz_lo,
                    dz_hi,
                )
            )

    # Fixtures (immovable; world-z slab from the measured table height + height).
    for _node_id, fnode in graph.nodes.items():
        if not isinstance(fnode, FixtureNode):
            continue
        if fnode.init_x is None or fnode.init_y is None:
            continue
        fdims = _fixture_dims(fnode.object_class)
        fh = fdims[2]
        fz_center = footprint.table_world_z + fh / 2.0
        targets.append(
            (
                "",
                f"{fnode.init_x:.4f}",
                f"{fnode.init_y:.4f}",
                f"{fz_center:.4f}",
                fdims[0] / 2.0,
                fdims[1] / 2.0,
                fh / 2.0,
                footprint.table_world_z,
                footprint.table_world_z + fh,
            )
        )

    if not targets:
        return []

    lines = [
        "",
        "# Robot link clearance (Fix 1: perturbed robot init pose in the require graph).",
        "# Each link's perturbed world position is a linear (Jacobian) function of the",
        "# sampled joint deltas _robot_dq_k; the SAT-correct 3-D AABB clause keeps every",
        "# placed object, distractor and fixture out of the link's measured world AABB so",
        "# MuJoCo settle need not shove them (RCA Finding B). Footprints are measured by",
        "# scripts/measure_robot_link_footprints.py.",
    ]
    for link in active_links:
        lx, ly, lz = _link_pos_exprs(link, n_dof)
        sv = _sanitize(link.name)
        lines.append(f"_rc_{sv}_x = {lx}")
        lines.append(f"_rc_{sv}_y = {ly}")
        lines.append(f"_rc_{sv}_z = {lz}")
        for guard, cx, cy, cz, thx, thy, thz, z_lo, z_hi in targets:
            if z_lo is not None and (link.z_min > z_hi or link.z_max < z_lo):
                continue
            dx = link.hx + thx
            dy = link.hy + thy
            dz = link.hz + thz
            lines.append(
                f"require {guard}(abs(_rc_{sv}_x - {cx}) > {dx:.4f}) "
                f"or (abs(_rc_{sv}_y - {cy}) > {dy:.4f}) "
                f"or (abs(_rc_{sv}_z - {cz}) > {dz:.4f})"
            )
    return lines


def _render_constraints(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    lines = ["# Distance constraints"]

    # Materialize declared support / containment relations once.  The
    # relative-positioning specifier already encodes the positive
    # support / containment constraint via its offset Range; the constraint
    # block must only suppress contradictory clearance constraints between
    # the same (child, support) pair.
    support_relations = _collect_support_relations(plan, graph)

    # Annotate each declared relation in the generated Scenic so the
    # semantic intent is auditable from the .scenic alone.  These comments
    # are pure documentation — they emit no Scenic constraints.
    if support_relations:
        lines.append(
            "# Declared support / containment relations "
            "(positive constraints encoded via 'at <support> offset by Range' specifier above):"
        )
        for rel in support_relations:
            lines.append(
                f"#   {rel.child_name} {rel.kind} {rel.support_name}"
                f"{' (fixture)' if rel.support_is_fixture else ' (movable)'}"
            )

    # Collect (var_name, dims, instance_name, is_sampled) tuples for
    # non-contained objects.  is_sampled drives the constraint skip rule.
    obj_info: list[tuple[str, tuple[float, float, float], str, bool]] = []
    for node_id, node in graph.nodes.items():
        if not isinstance(node, (ObjectNode, MovableSupportNode)):
            continue
        if isinstance(node, ObjectNode) and node.contained:
            continue
        var_name = _to_var(node.instance_name)
        obj_class = node.object_class or node.instance_name
        # FV MC #6: bound the footprint by the WIDEST variant the object axis can
        # substitute, not just the canonical class (a wider OOD variant otherwise
        # slips through the pairwise / fixture / distractor AABB clearance and the
        # simulator shoves it). Mirrors the max-over-pool the robot-clearance
        # z-prune already does.
        _pool = [obj_class] + (
            list(plan.object_substitutions.get(node.instance_name) or [])
            if "object" in plan.active_axes
            else []
        )
        dims = tuple(max(get_dimensions(c)[k] for c in _pool) for k in range(3))
        sampled = _is_sampled(node, plan)
        obj_info.append((var_name, dims, node.instance_name, sampled))

    # Map of relatively-positioned child -> direct support, used for the
    # contained-sibling detection below. This is the same data
    # ``support_relations`` carries (child/support pairs), but indexed by
    # child name so ``_clearance_relationship`` can look up shared parents
    # without scanning the relations list.
    relative_parent = _relative_parent_map(plan)

    # Pairwise AABB clearance constraints.
    #
    # Rule 1 — fixed-vs-fixed: BOTH objects sit at BDDL canonical positions
    #   that a human set deliberately.  No constraint is emitted — the planner
    #   trusts that the author's positions are geometrically valid.
    #
    # Rule 2 — sampled-vs-fixed: one object is Scenic-sampled (position-perturbed
    #   task object), the other is at a fixed BDDL position.  Emit a footprint-
    #   based threshold so the sampled object cannot overlap the fixed one.
    #
    # Rule 3 — sampled-vs-sampled: both positions are Scenic-sampled.
    #   Use footprint-based thresholds — exact AABB non-overlap guarantee.
    #
    # Rule 4 — declared support pair (e.g., bowl stacked on cookies_box):
    #   skip — the relative-positioning specifier already pins the child
    #   on top of the support.
    #
    # Rule 5 — contained siblings (e.g., two soup cans inside one basket):
    #   skip — both children are sampled relative to the same container's
    #   interior envelope, which already enforces non-overlap via per-child
    #   slot ranges in the planner. Emitting an additional pairwise AABB
    #   clearance here would over-constrain the sampler and reject most
    #   valid layouts.
    for i in range(len(obj_info)):
        for j in range(i + 1, len(obj_info)):
            var_a, dims_a, _name_a, sampled_a = obj_info[i]
            var_b, dims_b, _name_b, sampled_b = obj_info[j]
            # Rule 4: declared (child, support) pair across any kind
            # (on_surface / inside / stacked).
            if _is_declared_support_pair(var_a, var_b, support_relations):
                continue
            # Rule 5: contained siblings sharing one movable container.
            # ``_clearance_relationship`` also re-detects the direct-parent
            # case, which is a no-op here because Rule 4 already short-
            # circuits it; the value of going through it is the
            # ``contained_sibling`` classification.
            relationship = _clearance_relationship(_name_a, _name_b, relative_parent, graph)
            if relationship in {"direct_parent", "contained_sibling"}:
                continue
            # Rule 1: both fixed — skip (positions are valid by BDDL design)
            if not sampled_a and not sampled_b:
                continue
            # Rules 2 & 3: at least one sampled — footprint-based AABB constraint
            dx_clearance = (dims_a[0] + dims_b[0]) / 2.0
            dy_clearance = (dims_a[1] + dims_b[1]) / 2.0
            lines.append(
                f"require (abs({var_a}.position.x - {var_b}.position.x) > {dx_clearance:.4f}) "
                f"or (abs({var_a}.position.y - {var_b}.position.y) > {dy_clearance:.4f})"
            )

    # Object-fixture footprint clearance: task objects must not overlap fixed
    # fixtures.  Use the **SAT-correct per-axis OR-form** — identical in
    # structure to the object↔object pairwise clearance at the top of this
    # function.  By the separating-axis theorem for axis-aligned rectangles,
    # two AABBs are non-overlapping iff |dx| > (wa+wb)/2 OR |dy| > (la+lb)/2.
    # The previous radial-diagonal form (`distance > _footprint_clearance_xy`)
    # was conservative by ~√2× (worse for elongated fixtures like flat_stove
    # 0.36×0.20 or desk_caddy 0.14×0.42), and the original justification
    # ("OR form permits diagonal corner penetration") is mathematically
    # incorrect — disjoint x-projections imply disjoint AABBs regardless of
    # y, and vice versa. The over-conservative radial form is the dominant
    # contributor to Scenic rejection-sampler budget exhaustion on the
    # 10-task `libero_goal/` drawer-family residual (campaign caveat c.1):
    # see rca/stage4_c1_addendum_10_task_footprint.md.
    #
    # Decision routed through ``_should_emit_fixture_clearance`` so that:
    #   - fixed BDDL placements stay un-constrained (trust author), and
    #   - declared support relations between the object and this fixture
    #     suppress the (otherwise unsatisfiable) clearance constraint.
    for node_id, fnode in graph.nodes.items():
        if not isinstance(fnode, FixtureNode):
            continue
        if fnode.init_x is None or fnode.init_y is None:
            continue
        fvar = _to_var(fnode.instance_name)
        fdims = _fixture_dims(fnode.object_class)
        for var_name, dims, _name, sampled in obj_info:
            if not _should_emit_fixture_clearance(var_name, sampled, fvar, support_relations):
                continue
            dx_min, dy_min = _footprint_clearance_aabb(fdims, dims)
            lines.append(
                f"require (abs({var_name}.position.x - {fvar}.position.x) > {dx_min:.4f}) "
                f"or (abs({var_name}.position.y - {fvar}.position.y) > {dy_min:.4f})"
            )

    # Anti-trivialization: note in params that it's active
    if plan.anti_trivialization_active:
        lines.append("")
        lines.append('param anti_trivialization = "active"')

    # Distractor clearance (fixed small margin — distractors are intentionally small).
    # Distractors do not currently carry a PositionPlan with a declared support
    # relation, so ``_should_emit_fixture_clearance`` for a synthetic
    # distractor var would always return True; we keep the existing
    # distractor-vs-fixture loop unconditional but wire it through the
    # helper to remain forward-compatible if the planner is later extended
    # to put distractors on a fixture.
    if plan.distractor_budget > 0 and "distractor" in plan.active_axes:
        distractor_dims = (0.08, 0.08, 0.08)
        # Diagonal-radius clearance between two AABBs of equal half-extent h
        # is 2 * sqrt(2) * h ≈ 2.828 h. For h = 0.04 m this is ≈ 0.1131 m.
        # The previous 0.06 threshold was the *axis-aligned* sum of half-
        # widths only; two distractors aligned diagonally could satisfy
        # `distance > 0.06` while their AABBs still overlapped on both axes.
        _DISTRACTOR_PAIR_CLEARANCE = math.sqrt(distractor_dims[0] ** 2 + distractor_dims[1] ** 2)
        for i in range(plan.distractor_budget):
            d_var = f"distractor_{i}"
            # Distractor↔object clearance: SAT-correct per-axis AABB OR-form,
            # the same fix PR #16 applied to the fixture pairs. The previous
            # ``distance from d to obj > 0.13`` was the radial point-distance
            # bug family — a hardcoded scalar that both over-constrains along
            # the axes and under-constrains on the diagonal, and ignored each
            # object's measured footprint entirely. Use the measured object
            # dims (NOT a hardcoded 0.13): two AABBs are non-overlapping iff
            # their projections are disjoint on x OR y.
            for var, dims, _n, _s in obj_info:
                dx_min, dy_min = _footprint_clearance_aabb(distractor_dims, dims)
                lines.append(
                    f"require (_n_distractors <= {i}) "
                    f"or (abs({d_var}.position.x - {var}.position.x) > {dx_min:.4f}) "
                    f"or (abs({d_var}.position.y - {var}.position.y) > {dy_min:.4f})"
                )
            for j in range(i + 1, plan.distractor_budget):
                lines.append(
                    f"require (_n_distractors <= {i}) "
                    f"or (_n_distractors <= {j}) "
                    f"or ((distance from {d_var} to distractor_{j}) "
                    f"> {_DISTRACTOR_PAIR_CLEARANCE:.4f})"
                )
            for node_id, fnode in graph.nodes.items():
                if not isinstance(fnode, FixtureNode):
                    continue
                if fnode.init_x is None or fnode.init_y is None:
                    continue
                fvar = _to_var(fnode.instance_name)
                # Distractors are always Scenic-sampled (they have no BDDL
                # canonical position) and currently have no support relations,
                # so the helper trivially returns True today.  Routing
                # through the helper keeps a future planner extension
                # (e.g., distractors-on-fixtures) automatically correct.
                if not _should_emit_fixture_clearance(d_var, True, fvar, support_relations):
                    continue
                fdims = _fixture_dims(fnode.object_class)
                # SAT-correct AABB OR-form (see object↔fixture comment block
                # above and rca/stage4_c1_addendum_10_task_footprint.md).
                dx_min, dy_min = _footprint_clearance_aabb(fdims, distractor_dims)
                lines.append(
                    f"require (_n_distractors <= {i}) "
                    f"or (abs({d_var}.position.x - {fvar}.position.x) > {dx_min:.4f}) "
                    f"or (abs({d_var}.position.y - {fvar}.position.y) > {dy_min:.4f})"
                )

    # Robot link clearance (Fix 1): put the perturbed robot init pose into the
    # require graph so Scenic rejects samples where a perturbed arm link's volume
    # intersects a placed object/distractor/fixture (otherwise MuJoCo settle
    # resolves the penetration by shoving the object 40–260 mm in xy — the
    # dominant pose_tolerance failure in RCA Finding B).
    if plan.robot_plan is not None and "robot" in plan.active_axes:
        footprint = get_robot_footprint(plan.robot_plan.robot_model)
        if footprint is not None:
            lines.extend(_render_robot_clearance(footprint, plan, graph))

    lines.append("")
    return "\n".join(lines)


def _render_visibility(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    """Emit visibility_targets param for objects with must_remain_visible_with edges."""
    vis_edges = graph.edges_by_label("must_remain_visible_with")
    targets = sorted(
        {e.src_id for e in vis_edges if isinstance(graph.get_node(e.src_id), ObjectNode)}
    )
    if not targets:
        return ""
    targets_str = ", ".join(f'"{t}"' for t in targets)
    return f"param visibility_targets = [{targets_str}]\n"


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------


def _to_var(name: str) -> str:
    """Convert an instance name to a valid Scenic variable name."""
    return name.replace("-", "_")


def _sanitize(name: str) -> str:
    """Sanitize a string for use in a Scenic variable name."""
    return name.replace("-", "_").replace(" ", "_")


def _to_class_name(name: str) -> str:
    """Convert a fixture/object class name to CamelCase."""
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_"))
