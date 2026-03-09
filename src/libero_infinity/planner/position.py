"""Position axis planner for Libero-Infinity.

Computes per-object PositionPlan entries from a SemanticSceneGraph.
Each plan is independent — no cross-axis logic here.
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
from libero_infinity.planner.types import (
    AxisEnvelope,
    InfeasiblePerturbationError,
    PositionPlan,
)

_WORKSPACE_X_MARGIN = 0.11  # from calibration
_WORKSPACE_Y_MARGIN = 0.11
_DEFAULT_PERTURB_RADIUS = 0.15
_FIXTURE_PERTURB_RADIUS = 0.08
_GOAL_COVERAGE_THRESHOLD = 0.8  # switch to distance-based if >80% covered

# Interior (x, y) sampling extent for known container fixtures.
# These are conservative estimates of the usable interior cavity;
# each value is roughly 60-70 % of the corresponding exterior footprint.
_CONTAINER_FIXTURE_INTERIOR: dict[str, tuple[float, float]] = {
    "wooden_cabinet": (0.20, 0.18),
    "white_cabinet": (0.20, 0.18),
    "microwave": (0.18, 0.14),
    "desk_caddy": (0.10, 0.07),
    "bowl_drainer": (0.12, 0.10),
    "wine_rack": (0.12, 0.08),
}
_CONTAINER_FIXTURE_INTERIOR_DEFAULT = (0.15, 0.12)  # conservative fallback


def plan_position(
    graph: SemanticSceneGraph,
    request_axes: frozenset[str],
    diagnostics: PlanDiagnostics | None = None,
) -> dict[str, PositionPlan]:
    """Compute per-object position plans from the scene graph.

    Args:
        graph: The semantic scene graph for the task.
        request_axes: Set of active perturbation axis names.
        diagnostics: Optional diagnostics collector; a fresh one is created if None.

    Returns:
        Dict mapping object node_id -> PositionPlan. Empty if 'position' not in request_axes.
    """
    if "position" not in request_axes:
        return {}
    if diagnostics is None:
        diagnostics = PlanDiagnostics()

    plans: dict[str, PositionPlan] = {}
    for node_id, node in graph.nodes.items():
        if not isinstance(node, (ObjectNode, MovableSupportNode)):
            continue
        plan = _plan_object_position(node, graph, diagnostics)
        if plan is not None:
            plans[node_id] = plan
    return plans


def _plan_object_position(
    node: ObjectNode | MovableSupportNode,
    graph: SemanticSceneGraph,
    diagnostics: PlanDiagnostics,
) -> PositionPlan | None:
    """Compute position plan for a single object node."""
    # Find the primary support edge
    support_edges = [
        e
        for e in graph.edges_from(node.node_id)
        if e.label in ("supported_by", "stacked_on", "contained_in")
    ]
    if not support_edges:
        return None

    edge = support_edges[0]

    # Contained objects: sample WITHIN parent bounds using relative positioning.
    if edge.label == "contained_in":
        return _plan_contained_position(node, edge.dst_id, graph, diagnostics)

    is_stacked = edge.label == "stacked_on"
    support_node = graph.get_node(edge.dst_id)

    # Compute envelope based on support relationship
    if is_stacked:
        # Relative to parent: small perturbation around parent origin
        x_env = AxisEnvelope(-0.05, 0.05, "x")
        y_env = AxisEnvelope(-0.05, 0.05, "y")
    elif isinstance(support_node, WorkspaceNode):
        # Workspace surface: center around init position with default radius
        cx = node.init_x or 0.0
        cy = node.init_y or 0.0
        r = _DEFAULT_PERTURB_RADIUS
        x_env = AxisEnvelope(cx - r, cx + r, "x")
        y_env = AxisEnvelope(cy - r, cy + r, "y")
    else:
        # Fixture or movable support: tighter perturbation around init position
        cx = node.init_x or 0.0
        cy = node.init_y or 0.0
        r = _FIXTURE_PERTURB_RADIUS
        x_env = AxisEnvelope(cx - r, cx + r, "x")
        y_env = AxisEnvelope(cy - r, cy + r, "y")

    # Range degeneracy guard
    try:
        x_env.validate()
        y_env.validate()
    except InfeasiblePerturbationError:
        diagnostics.drop_axis("position", f"degenerate envelope for {node.node_id}")
        return None

    # Goal region exclusion: avoid placing the object where the task is already solved
    exclusion_zones: list[tuple[float, float, float, float]] = []
    exclusion_min_distance: float | None = None

    goal_edges = [e for e in graph.edges_from(node.node_id) if e.label == "goal_target"]
    for ge in goal_edges:
        region_node = graph.get_node(ge.dst_id)
        if region_node is None or not hasattr(region_node, "x_min"):
            continue
        if region_node.x_min is None or region_node.y_min is None:
            continue

        zone = (
            float(region_node.x_min),
            float(region_node.y_min),
            float(region_node.x_max),
            float(region_node.y_max),
        )
        zone_area = (zone[2] - zone[0]) * (zone[3] - zone[1])
        env_area = (x_env.hi - x_env.lo) * (y_env.hi - y_env.lo)

        if env_area > 0 and zone_area / env_area > _GOAL_COVERAGE_THRESHOLD:
            # Exclusion zone covers too much — fall back to distance-based
            exclusion_min_distance = 0.05
            diagnostics.narrow_axis(
                "position",
                f"GoalRegionExclusion fallback to distance-based for {node.node_id}",
            )
        else:
            exclusion_zones.append(zone)

    return PositionPlan(
        object_name=node.instance_name,
        x_envelope=x_env,
        y_envelope=y_env,
        support_name=edge.dst_id,
        use_relative_positioning=is_stacked,
        exclusion_zones=exclusion_zones,
        exclusion_min_distance=exclusion_min_distance,
    )


def _plan_contained_position(
    node: ObjectNode | MovableSupportNode,
    container_id: str,
    graph: SemanticSceneGraph,
    diagnostics: PlanDiagnostics,
) -> PositionPlan | None:
    """Compute a relative position plan for an object contained inside a parent.

    The child is placed at `parent offset by Vector(Range(-dx, dx), Range(-dy, dy), 0)`
    where dx/dy are half the parent's interior cavity minus the child's half-footprint.

    This preserves containment under perturbation: perturbing the parent moves the
    child with it, while the child can independently sample within the parent's bounds.
    For recursive chains (ball in bowl in cabinet), each child anchors to its direct
    parent so the chain propagates automatically.
    """
    container_node = graph.get_node(container_id)
    if container_node is None:
        diagnostics.drop_axis(
            "position",
            f"contained_in parent {container_id!r} not found for {node.node_id}",
        )
        return None

    container_class = (
        (container_node.object_class or "") if hasattr(container_node, "object_class") else ""
    )
    child_class = node.object_class or ""

    # Determine parent interior extents (x, y).
    if isinstance(container_node, FixtureNode):
        parent_x, parent_y = _CONTAINER_FIXTURE_INTERIOR.get(
            container_class, _CONTAINER_FIXTURE_INTERIOR_DEFAULT
        )
    else:
        # Movable support or object: use registry footprint directly.
        pdims = get_dimensions(container_class)
        parent_x, parent_y = pdims[0], pdims[1]

    # Child half-footprint.
    cdims = get_dimensions(child_class)
    child_hx = cdims[0] / 2.0
    child_hy = cdims[1] / 2.0

    # Sampling half-extents: keep child fully inside parent interior.
    dx = max(0.02, parent_x / 2.0 - child_hx)
    dy = max(0.02, parent_y / 2.0 - child_hy)

    x_env = AxisEnvelope(-dx, dx, "x")
    y_env = AxisEnvelope(-dy, dy, "y")

    try:
        x_env.validate()
        y_env.validate()
    except InfeasiblePerturbationError:
        diagnostics.drop_axis(
            "position",
            f"degenerate contained envelope for {node.node_id} in {container_id}",
        )
        return None

    return PositionPlan(
        object_name=node.instance_name,
        x_envelope=x_env,
        y_envelope=y_env,
        support_name=container_id,
        use_relative_positioning=True,
    )
