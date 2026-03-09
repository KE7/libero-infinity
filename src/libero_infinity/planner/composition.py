"""Full perturbation plan composition for Libero-Infinity.

Implements the plan_perturbations function: each axis is planned independently
(tuple construction), then a single-pass validation step checks cross-axis
constraints.
"""

from __future__ import annotations

from libero_infinity.asset_registry import get_dimensions
from libero_infinity.ir.nodes import (
    FixtureNode,
    MovableSupportNode,
    ObjectNode,
    PlanDiagnostics,
    WorkspaceNode,
)
from libero_infinity.ir.scene_graph import SemanticSceneGraph
from libero_infinity.planner.axes import (
    plan_articulation,
    plan_background,
    plan_camera,
    plan_distractor,
    plan_lighting,
    plan_object,
    plan_texture,
)
from libero_infinity.planner.position import (
    _CONTAINER_FIXTURE_INTERIOR,
    _CONTAINER_FIXTURE_INTERIOR_DEFAULT,
    plan_position,
)
from libero_infinity.planner.types import PerturbationPlan

# ---------------------------------------------------------------------------
# Preset expansion
# ---------------------------------------------------------------------------

AXIS_PRESETS: dict[str, frozenset[str]] = {
    "combined": frozenset(["position", "object", "camera", "lighting", "distractor", "background"]),
    "full": frozenset(
        [
            "position",
            "object",
            "camera",
            "lighting",
            "texture",
            "distractor",
            "articulation",
            "background",
        ]
    ),
}


def parse_axes(axes_str: str) -> frozenset[str]:
    """Parse comma-separated axes string, expanding presets.

    Args:
        axes_str: Axis specification, e.g. ``"position,camera"`` or ``"combined"``.

    Returns:
        Frozen set of individual axis name strings.
    """
    result: set[str] = set()
    for part in axes_str.split(","):
        part = part.strip()
        if part in AXIS_PRESETS:
            result.update(AXIS_PRESETS[part])
        elif part:
            result.add(part)
    return frozenset(result)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def plan_perturbations(
    graph: SemanticSceneGraph,
    request: str | frozenset[str],
) -> PerturbationPlan:
    """Compose a full perturbation plan from all active axes.

    Tuple construction: each axis is planned independently, then a single-pass
    validation step adds cross-axis constraints.

    Args:
        graph: The semantic scene graph for the task.
        request: Axis specification string or frozen set of axis names.

    Returns:
        A PerturbationPlan with all active axes populated.
    """
    if isinstance(request, str):
        axes = parse_axes(request)
    else:
        axes = request

    diagnostics = PlanDiagnostics()
    plan = PerturbationPlan(active_axes=axes, diagnostics=diagnostics)

    # STEP 1: Plan each axis independently (tuple construction)
    if "position" in axes:
        plan.position_plans = plan_position(graph, axes, diagnostics)

    if "object" in axes:
        plan.object_substitutions = plan_object(graph, axes, diagnostics)

    # Always plan articulation for goal-reachability safety
    if "articulation" in axes or True:  # noqa: SIM210
        plan.articulation_plans = plan_articulation(graph, axes, diagnostics)

    if "camera" in axes:
        plan.camera_plan = plan_camera(graph, axes, diagnostics)

    if "lighting" in axes:
        plan.lighting_plan = plan_lighting(graph, axes, diagnostics)

    if "texture" in axes:
        plan.texture_plan = plan_texture(graph, axes, diagnostics)

    if "distractor" in axes:
        plan.distractor_budget, plan.distractor_classes = plan_distractor(graph, axes, diagnostics)

    if "background" in axes:
        plan.background_plan = plan_background(graph, axes, diagnostics)

    # STEP 2: Single-pass validation (7 fixed steps)
    _validate_plan(plan, graph, diagnostics)

    return plan


# ---------------------------------------------------------------------------
# Single-pass validation (7 fixed steps per spec section 3.3)
# ---------------------------------------------------------------------------


def _validate_plan(
    plan: PerturbationPlan,
    graph: SemanticSceneGraph,
    diagnostics: PlanDiagnostics,
) -> None:
    """Single-pass validation in 7 fixed steps."""
    _check_support_preservation(plan, graph, diagnostics)
    _check_containment_dimensions(plan, graph, diagnostics)
    _check_articulation_noninterference(plan, graph, diagnostics)
    _check_distractor_budget(plan, graph, diagnostics)
    _check_camera_visibility(plan, graph, diagnostics)
    _check_anti_trivialization(plan, graph, diagnostics)
    _check_envelope_quality(plan, graph, diagnostics)


def _check_support_preservation(
    plan: PerturbationPlan,
    graph: SemanticSceneGraph,
    diagnostics: PlanDiagnostics,
) -> None:
    """Step 1: Verify child position envelopes lie within parent envelopes (best-effort).

    For objects whose position is expressed relative to a parent (stacked or
    contained, ``use_relative_positioning=True``), the envelope half-extents
    should not exceed the parent's half-footprint.  If they do, a diagnostic
    warning is recorded but the plan is not dropped (best-effort).
    """
    for obj_name, pos_plan in plan.position_plans.items():
        if not pos_plan.use_relative_positioning:
            continue

        support_node = graph.get_node(pos_plan.support_name)
        if support_node is None:
            diagnostics.narrow_axis(
                "position",
                f"{obj_name}: support node {pos_plan.support_name!r} not found "
                "during support-preservation check",
            )
            continue

        # WorkspaceNode is arbitrarily large — no footprint check needed.
        if isinstance(support_node, WorkspaceNode):
            continue

        support_class = support_node.object_class or ""
        try:
            sdims = get_dimensions(support_class)
            half_sx = sdims[0] / 2.0
            half_sy = sdims[1] / 2.0
        except Exception:
            continue

        x_half = (pos_plan.x_envelope.hi - pos_plan.x_envelope.lo) / 2.0
        y_half = (pos_plan.y_envelope.hi - pos_plan.y_envelope.lo) / 2.0

        if x_half > half_sx or y_half > half_sy:
            diagnostics.narrow_axis(
                "position",
                f"{obj_name}: relative envelope half-extents ({x_half:.3f}, "
                f"{y_half:.3f}) exceed parent '{pos_plan.support_name}' "
                f"half-footprint ({half_sx:.3f}, {half_sy:.3f})",
            )


def _check_containment_dimensions(
    plan: PerturbationPlan,
    graph: SemanticSceneGraph,
    diagnostics: PlanDiagnostics,
) -> None:
    """Step 2: Verify containment dimensional compatibility.

    For each object with a ``contained_in`` edge, checks that the object's
    bounding-box footprint fits inside the container's usable interior cavity.
    Records a diagnostic warning when the child is oversized; does NOT drop the
    position plan (best-effort — the physics simulator may still allow tight fits).
    """
    for node_id, node in graph.nodes.items():
        if not isinstance(node, (ObjectNode, MovableSupportNode)):
            continue

        contained_edges = [e for e in graph.edges_from(node_id) if e.label == "contained_in"]
        if not contained_edges:
            continue

        container_node = graph.get_node(contained_edges[0].dst_id)
        if container_node is None:
            continue

        # Determine container interior dimensions.
        container_class = container_node.object_class or ""
        if isinstance(container_node, FixtureNode):
            cont_x, cont_y = _CONTAINER_FIXTURE_INTERIOR.get(
                container_class, _CONTAINER_FIXTURE_INTERIOR_DEFAULT
            )
        else:
            try:
                cdims = get_dimensions(container_class)
                cont_x, cont_y = cdims[0], cdims[1]
            except Exception:
                continue

        # Child footprint.
        child_class = node.object_class or ""
        try:
            child_dims = get_dimensions(child_class)
            child_x, child_y = child_dims[0], child_dims[1]
        except Exception:
            continue

        if child_x > cont_x or child_y > cont_y:
            diagnostics.narrow_axis(
                "position",
                f"{node_id}: child footprint ({child_x:.3f}×{child_y:.3f}) "
                f"exceeds container '{contained_edges[0].dst_id}' interior "
                f"({cont_x:.3f}×{cont_y:.3f})",
            )


def _check_articulation_noninterference(
    plan: PerturbationPlan,
    graph: SemanticSceneGraph,
    diagnostics: PlanDiagnostics,
) -> None:
    """Step 3: Verify articulation plans don't block object placement goals.

    A fixture must be open/on whenever:
    - An object is currently *inside* it (``contained_in`` edge to the fixture).
    - A position plan places an object *inside* it
      (``use_relative_positioning=True`` with ``support_name`` == fixture id).

    If the ArticulationPlan for such a fixture has state_kind ``"Close"`` or
    ``"Turnoff"``, the plan is corrected to ``"Open"`` (or ``"Turnon"`` for
    stoves) and ``goal_reachability_ok`` is set to ``False`` to flag the
    correction.
    """
    # Collect fixture instance_names (and node_ids) that must be accessible.
    needs_open: set[str] = set()

    for node_id, node in graph.nodes.items():
        if not isinstance(node, (ObjectNode, MovableSupportNode)):
            continue

        # Objects currently contained inside a fixture → fixture must be open.
        for edge in graph.edges_from(node_id):
            if edge.label == "contained_in":
                container = graph.get_node(edge.dst_id)
                if isinstance(container, FixtureNode):
                    needs_open.add(container.instance_name)
                    needs_open.add(container.node_id)

        # Objects whose position plan anchors them relative to a fixture.
        for plan_key in (node_id, node.instance_name):
            pos_plan = plan.position_plans.get(plan_key)
            if pos_plan is not None and pos_plan.use_relative_positioning:
                support = graph.get_node(pos_plan.support_name)
                if isinstance(support, FixtureNode):
                    needs_open.add(support.instance_name)
                    needs_open.add(support.node_id)

    # Correct any conflicting articulation plans.
    for fixture_name, art_plan in plan.articulation_plans.items():
        if fixture_name not in needs_open:
            continue
        if art_plan.state_kind not in ("Close", "Turnoff"):
            continue

        # Determine the open/on equivalent for this fixture's family.
        corrected_state = "Turnon" if art_plan.state_kind == "Turnoff" else "Open"
        art_plan.state_kind = corrected_state
        art_plan.goal_reachability_ok = False
        diagnostics.narrow_axis(
            "articulation",
            f"{fixture_name}: corrected from 'Close/Turnoff' to "
            f"'{corrected_state}' — object access or placement requires "
            "open state",
        )


def _check_distractor_budget(
    plan: PerturbationPlan,
    graph: SemanticSceneGraph,
    diagnostics: PlanDiagnostics,
) -> None:
    """Step 4: Cap distractor budget at 5."""
    if plan.distractor_budget > 5:
        plan.distractor_budget = 5
        diagnostics.narrow_axis("distractor", "capped at 5")


def _check_camera_visibility(
    plan: PerturbationPlan,
    graph: SemanticSceneGraph,
    diagnostics: PlanDiagnostics,
) -> None:
    """Step 5: Verify camera visibility sub-envelope is non-empty.

    Checks that the CameraPlan's azimuth, elevation, and distance ranges are
    all strictly non-degenerate (lo < hi).  A degenerate range means the
    camera perturbation would be forced to a single value, defeating the
    purpose of the axis.  If any range is degenerate the entire camera plan is
    dropped (set to None) and the axis is recorded as dropped in diagnostics.
    """
    if plan.camera_plan is None:
        return

    cp = plan.camera_plan
    reasons: list[str] = []

    if cp.azimuth_lo >= cp.azimuth_hi:
        reasons.append(f"degenerate azimuth [{cp.azimuth_lo}, {cp.azimuth_hi}]")
    if cp.elevation_lo >= cp.elevation_hi:
        reasons.append(f"degenerate elevation [{cp.elevation_lo}, {cp.elevation_hi}]")
    if cp.distance_lo >= cp.distance_hi:
        reasons.append(f"degenerate distance [{cp.distance_lo}, {cp.distance_hi}]")

    if reasons:
        plan.camera_plan = None
        diagnostics.drop_axis("camera", "; ".join(reasons))


def _check_anti_trivialization(
    plan: PerturbationPlan,
    graph: SemanticSceneGraph,
    diagnostics: PlanDiagnostics,
) -> None:
    """Step 6: Anti-trivialization is always active."""
    plan.anti_trivialization_active = True


def _check_envelope_quality(
    plan: PerturbationPlan,
    graph: SemanticSceneGraph,
    diagnostics: PlanDiagnostics,
) -> None:
    """Step 7: Envelope quality check.

    Removes any PositionPlan whose x or y envelope is degenerate (lo >= hi).
    Degenerate envelopes cannot produce meaningful perturbations and would
    cause downstream samplers to fail or produce identical samples.  Each
    removed plan is recorded in diagnostics.
    """
    bad_keys = [
        obj_name
        for obj_name, pos_plan in plan.position_plans.items()
        if (
            pos_plan.x_envelope.lo >= pos_plan.x_envelope.hi
            or pos_plan.y_envelope.lo >= pos_plan.y_envelope.hi
        )
    ]
    for key in bad_keys:
        diagnostics.drop_axis(
            "position",
            f"{key}: degenerate position envelope removed in quality check",
        )
        del plan.position_plans[key]
