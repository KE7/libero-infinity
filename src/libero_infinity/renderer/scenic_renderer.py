"""Pure Scenic 3 renderer for the Libero-Infinity compiler pipeline.

PURITY INVARIANT: This module contains zero conditional logic based on task
semantics. No ``if fixture_class == "..."`` or ``if object_class in {...}``.
All task-semantic decisions live in the semantic graph builder and planner.
All perturbation decisions live in the planner.
This renderer is a deterministic function of the plan IR alone.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from libero_infinity.asset_registry import get_dimensions
from libero_infinity.ir.nodes import (
    FixtureNode,
    MovableSupportNode,
    ObjectNode,
)
from libero_infinity.ir.scene_graph import SemanticSceneGraph
from libero_infinity.planner.types import PerturbationPlan

if TYPE_CHECKING:
    pass


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
    "desk_caddy": (0.14, 0.10, 0.06),
    "wooden_two_layer_shelf": (0.24, 0.14, 0.22),
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
    """Minimum centre-to-centre xy distance before two footprints overlap.

    Computes the diagonal radius of each object's xy footprint and returns
    their sum — the minimum separation needed to guarantee no overlap.
    Duplicated from simulator.py to avoid circular imports.
    """
    radius_a = math.hypot(dims_a[0], dims_a[1]) / 2.0
    radius_b = math.hypot(dims_b[0], dims_b[1]) / 2.0
    return radius_a + radius_b


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
    fragments.append(_render_camera(plan, graph))
    fragments.append(_render_lighting(plan, graph))
    fragments.append(_render_texture(plan, graph))
    fragments.append(_render_background(plan, graph))
    fragments.append(_render_distractors(plan, graph))
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


def _render_objects(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    lines = ["# Object declarations"]

    # Asset variant sampling (object axis)
    seen_classes: set[str] = set()
    asset_var_map: dict[str, str] = {}  # object_class -> scenic var name

    if "object" in plan.active_axes and plan.object_substitutions:
        lines.append("# Asset variant sampling")
        for obj_name, variants in plan.object_substitutions.items():
            node = graph.get_node(obj_name)
            if node is None:
                continue
            obj_class = node.object_class or obj_name
            if obj_class in seen_classes:
                continue
            seen_classes.add(obj_class)
            var_name = f"_chosen_{_sanitize(obj_class)}"
            variants_str = ", ".join(f'"{v}"' for v in variants)
            lines.append(f"{var_name} = Uniform({variants_str})")
            asset_var_map[obj_class] = var_name

        if asset_var_map:
            first_class = next(iter(asset_var_map))
            first_var = asset_var_map[first_class]
            lines.append(f'param perturb_class = "{first_class}"')
            lines.append(f"param chosen_asset = {first_var}")
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
    _in_degree: dict[str, int] = {name: (1 if dep is not None else 0) for name, dep in _dep.items()}
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

        if pos_plan is not None and not pos_plan.use_relative_positioning:
            x_lo = pos_plan.x_envelope.lo
            x_hi = pos_plan.x_envelope.hi
            y_lo = pos_plan.y_envelope.lo
            y_hi = pos_plan.y_envelope.hi
            pos_spec = (
                f"at Vector(Range({x_lo:.4f}, {x_hi:.4f}), Range({y_lo:.4f}, {y_hi:.4f}), TABLE_Z)"
            )
        elif pos_plan is not None and pos_plan.use_relative_positioning:
            support_var = _to_var(pos_plan.support_name)
            x_lo = pos_plan.x_envelope.lo
            x_hi = pos_plan.x_envelope.hi
            y_lo = pos_plan.y_envelope.lo
            y_hi = pos_plan.y_envelope.hi
            pos_spec = (
                f"at {support_var} offset by Vector(Range({x_lo:.4f}, {x_hi:.4f}), "
                f"Range({y_lo:.4f}, {y_hi:.4f}), 0.0)"
            )
        elif node.init_x is not None and node.init_y is not None:
            pos_spec = f"at Vector({node.init_x:.4f}, {node.init_y:.4f}, TABLE_Z)"
        else:
            # No position info — use workspace center
            pos_spec = "in SAFE_REGION"

        # Build specifier list — Scenic 3 requires comma-separated specifiers.
        # libero_name is the declared property on LIBEROObject (not 'name').
        specifiers: list[str] = [pos_spec]
        if scenic_class:
            specifiers.append(f"with asset_class {scenic_class}")
        specifiers.append(f'with libero_name "{obj_name}"')
        # support_parent_name is read by the simulator to skip footprint overlap
        # validation between an object and the movable surface it sits on.
        if pos_plan is not None and pos_plan.use_relative_positioning and pos_plan.support_name:
            specifiers.append(f'with support_parent_name "{pos_plan.support_name}"')

        lines.append(f"{var_name} = new LIBEROObject " + ", ".join(specifiers))

    lines.append("")
    return "\n".join(lines)


def _render_articulation(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    if not plan.articulation_plans:
        return ""
    lines = ["# Articulation parameters"]
    for fixture_name, art_plan in plan.articulation_plans.items():
        var_name = _sanitize(fixture_name)
        lines.append(f"param articulation_{var_name} = Range({art_plan.lo:.4f}, {art_plan.hi:.4f})")
        lines.append(f'param articulation_{var_name}_state = "{art_plan.state_kind}"')
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


def _render_distractors(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    if plan.distractor_budget <= 0 or "distractor" not in plan.active_axes:
        return ""
    classes = plan.distractor_classes or []
    n = plan.distractor_budget
    lines = [
        "# Distractor objects",
        f"param n_distractors = {n}",
    ]
    if classes:
        cls_str = ", ".join(f'"{c}"' for c in classes[:10])
        lines.append(f"_distractor_pool = [{cls_str}]")
    for i in range(n):
        # Scenic 3: specifiers are comma-separated; libero_name is the declared property
        lines.append(
            f'distractor_{i} = new LIBEROObject in SAFE_REGION, with libero_name "distractor_{i}"'
        )
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


def _render_constraints(plan: PerturbationPlan, graph: SemanticSceneGraph) -> str:
    lines = ["# Distance constraints"]

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
        dims = get_dimensions(obj_class)
        sampled = _is_sampled(node, plan)
        obj_info.append((var_name, dims, node.instance_name, sampled))

    # Build the set of (child_var, support_var) pairs that use relative
    # positioning — no distance constraint should be emitted between these.
    relative_pairs: set[frozenset[str]] = set()
    for obj_name, pp in plan.position_plans.items():
        if pp is not None and pp.use_relative_positioning and pp.support_name:
            relative_pairs.add(frozenset({_to_var(obj_name), _to_var(pp.support_name)}))

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
    for i in range(len(obj_info)):
        for j in range(i + 1, len(obj_info)):
            var_a, dims_a, _name_a, sampled_a = obj_info[i]
            var_b, dims_b, _name_b, sampled_b = obj_info[j]
            if frozenset({var_a, var_b}) in relative_pairs:
                continue  # object sits on support — no separation constraint
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
    # fixtures.  Use the radial diagonal clearance (_footprint_clearance_xy)
    # rather than an AABB OR-based threshold — the OR form permits diagonal
    # positions that still penetrate fixture geometry (one axis satisfied but
    # the other not).  The diagonal radius sum guarantees clearance from all
    # approach angles.
    # Only emit for sampled objects — fixed task objects already have valid
    # BDDL-authored positions that the fixture placement was designed around.
    for node_id, fnode in graph.nodes.items():
        if not isinstance(fnode, FixtureNode):
            continue
        if fnode.init_x is None or fnode.init_y is None:
            continue
        fvar = _to_var(fnode.instance_name)
        fdims = _fixture_dims(fnode.object_class)
        for var_name, dims, _name, sampled in obj_info:
            if not sampled:
                continue  # fixed task object — trust BDDL author placement
            clearance = _footprint_clearance_xy(fdims, dims)
            lines.append(f"require (distance from {var_name} to {fvar}) > {clearance:.4f}")

    # Anti-trivialization: note in params that it's active
    if plan.anti_trivialization_active:
        lines.append("")
        lines.append('param anti_trivialization = "active"')

    # Distractor clearance (fixed small margin — distractors are intentionally small)
    obj_vars = [var for var, _, _n, _s in obj_info]
    if plan.distractor_budget > 0 and "distractor" in plan.active_axes:
        for i in range(plan.distractor_budget):
            for var in obj_vars:
                lines.append(f"require (distance from distractor_{i} to {var}) > 0.08")

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
