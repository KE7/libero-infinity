"""Position axis planner for Libero-Infinity.

Computes per-object PositionPlan entries from a SemanticSceneGraph.
Each plan is independent — no cross-axis logic here.
"""

from __future__ import annotations

from libero_infinity.asset_registry import OBJECT_DIMENSIONS, get_dimensions
from libero_infinity.ir.nodes import (
    FixtureNode,
    MovableSupportNode,
    ObjectNode,
    PlanDiagnostics,
    RegionNode,
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
_TABLE_X_MIN = -0.40
_TABLE_X_MAX = 0.40
_TABLE_Y_MIN = -0.30
_TABLE_Y_MAX = 0.30
_TABLE_X_MARGIN = 0.04
_TABLE_Y_MARGIN = 0.04
_SUPPORT_UNBOUNDED_DEFAULT_RADIUS = 0.05

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

# Interior (x, y) sampling extent for known movable containers.
# Unlike fixtures, movable objects do not get a generic footprint fallback:
# treating every object footprint as usable interior silently turns arbitrary
# supports into containers.
_MOVABLE_CONTAINER_INTERIOR: dict[str, tuple[float, float]] = {
    "basket": (0.14, 0.09),
    "tray": (0.18, 0.12),
    "wooden_tray": (0.18, 0.12),
    "white_storage_box": (0.13, 0.09),
    "caddy": (0.10, 0.07),
    "desk_caddy": (0.10, 0.07),
    "bowl_drainer": (0.12, 0.10),
}


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

    def _support_half_extents(parent_node: FixtureNode | MovableSupportNode | ObjectNode | None) -> tuple[float, float]:
        """Return conservative half-width/half-length extents for a support.

        For non-workspace supports, we derive this from registry dimensions so
        relative sampling stays near the support footprint instead of drifting
        toward the parent origin from absolute coordinates.
        """
        if parent_node is None:
            return _SUPPORT_UNBOUNDED_DEFAULT_RADIUS, _SUPPORT_UNBOUNDED_DEFAULT_RADIUS
        if isinstance(parent_node, RegionNode):
            if (
                parent_node.x_min is not None
                and parent_node.x_max is not None
                and parent_node.y_min is not None
                and parent_node.y_max is not None
            ):
                return (
                    (parent_node.x_max - parent_node.x_min) / 2.0,
                    (parent_node.y_max - parent_node.y_min) / 2.0,
                )
            return _SUPPORT_UNBOUNDED_DEFAULT_RADIUS, _SUPPORT_UNBOUNDED_DEFAULT_RADIUS
        parent_class = getattr(parent_node, "object_class", "") or ""
        p_w, p_l, _ = get_dimensions(parent_class)
        return p_w / 2.0, p_l / 2.0

    def _clip_relative_to_table(
        parent_node: FixtureNode | MovableSupportNode | ObjectNode,
        rel_lo: float,
        rel_hi: float,
        axis: str,
    ) -> tuple[float, float]:
        """Clip relative offsets to a bounded table-safe absolute range."""
        parent_x = float(getattr(parent_node, "init_x", 0.0) or 0.0)
        parent_y = float(getattr(parent_node, "init_y", 0.0) or 0.0)

        width = rel_hi - rel_lo
        if width <= 0:
            return rel_lo, rel_hi

        if axis == "x":
            target_min = _TABLE_X_MIN + _TABLE_X_MARGIN
            target_max = _TABLE_X_MAX - _TABLE_X_MARGIN
            centre = parent_x + (rel_lo + rel_hi) / 2.0
        else:
            target_min = _TABLE_Y_MIN + _TABLE_Y_MARGIN
            target_max = _TABLE_Y_MAX - _TABLE_Y_MARGIN
            centre = parent_y + (rel_lo + rel_hi) / 2.0

        half_w = width / 2.0
        clipped_centre = min(max(centre, target_min + half_w), target_max - half_w)
        abs_lo = clipped_centre - half_w
        abs_hi = clipped_centre + half_w

        if abs_lo < target_min:
            abs_lo = target_min
            abs_hi = abs_lo + width
        if abs_hi > target_max:
            abs_hi = target_max
            abs_lo = abs_hi - width

        if axis == "x":
            return abs_lo - parent_x, abs_hi - parent_x
        return abs_lo - parent_y, abs_hi - parent_y

    def _clamped_relative_envelope(
        parent_node: FixtureNode | MovableSupportNode | ObjectNode,
        child_node: ObjectNode | MovableSupportNode,
    ) -> tuple[AxisEnvelope, AxisEnvelope]:
        """Compute bounded relative offsets for non-workspace support links."""
        support_hx, support_hy = _support_half_extents(parent_node)
        parent_class = getattr(parent_node, "object_class", "")
        has_known_support_dims = (
            (isinstance(parent_node, RegionNode) and parent_node.x_min is not None)
            or parent_class in OBJECT_DIMENSIONS
            or (
                isinstance(parent_node, FixtureNode)
                and parent_class in _CONTAINER_FIXTURE_INTERIOR
            )
        )
        child_class = child_node.object_class or ""
        child_w, child_l, _ = get_dimensions(child_class)
        child_hx = child_w / 2.0
        child_hy = child_l / 2.0

        # Keep objects on-top of the support footprint while staying bounded.
        # For explicit support footprints, clip to the support-relative footprint.
        # For unbounded regions, keep placement centered on the support with a
        # small, bounded fallback around the support center.
        if has_known_support_dims:
            x_rad = support_hx - child_hx
            y_rad = support_hy - child_hy
        else:
            x_rad = max(0.0, support_hx - child_hx)
            y_rad = max(0.0, support_hy - child_hy)
        x_rad = max(0.02, min(_FIXTURE_PERTURB_RADIUS, x_rad))
        y_rad = max(0.02, min(_FIXTURE_PERTURB_RADIUS, y_rad))
        rel_x_lo, rel_x_hi = -x_rad, x_rad
        rel_y_lo, rel_y_hi = -y_rad, y_rad
        if has_known_support_dims:
            rel_x_lo, rel_x_hi = _clip_relative_to_table(parent_node, rel_x_lo, rel_x_hi, "x")
            rel_y_lo, rel_y_hi = _clip_relative_to_table(parent_node, rel_y_lo, rel_y_hi, "y")
        return AxisEnvelope(rel_x_lo, rel_x_hi, "x"), AxisEnvelope(rel_y_lo, rel_y_hi, "y")

    # Compute envelope based on support relationship
    if is_stacked:
        # Keep stacked objects on a bounded support footprint around the parent
        # anchor. This avoids task-specific fallback constants and prevents
        # stacked objects from drifting off small support surfaces.
        x_env, y_env = _clamped_relative_envelope(support_node, node)
    elif isinstance(support_node, WorkspaceNode):
        # Workspace surface: center around init position with default radius
        cx = node.init_x or 0.0
        cy = node.init_y or 0.0
        r = _DEFAULT_PERTURB_RADIUS
        x_env = AxisEnvelope(cx - r, cx + r, "x")
        y_env = AxisEnvelope(cy - r, cy + r, "y")
    else:
        # Fixture or movable support: tighter perturbation around init position
        x_env, y_env = _clamped_relative_envelope(support_node, node)

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
        use_relative_positioning=is_stacked or not isinstance(support_node, WorkspaceNode),
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

    parent_interior = container_interior_xy(container_node)
    if parent_interior is None:
        diagnostics.drop_axis(
            "position",
            f"unsupported movable container {container_id!r} "
            f"({container_class or 'unknown class'}) for contained object {node.node_id}",
        )
        return None
    parent_x, parent_y = parent_interior

    # Child half-footprint.
    cdims = get_dimensions(child_class)
    child_hx = cdims[0] / 2.0
    child_hy = cdims[1] / 2.0

    x_lo, x_hi, y_lo, y_hi = _contained_child_bounds(
        node.node_id,
        container_id,
        graph,
        parent_x,
        parent_y,
        child_hx,
        child_hy,
    )

    x_env = AxisEnvelope(x_lo, x_hi, "x")
    y_env = AxisEnvelope(y_lo, y_hi, "y")

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


def container_interior_xy(container_node: object) -> tuple[float, float] | None:
    """Return usable interior xy dimensions for a known container node."""
    container_class = getattr(container_node, "object_class", "") or ""
    if isinstance(container_node, FixtureNode):
        return _CONTAINER_FIXTURE_INTERIOR.get(container_class, _CONTAINER_FIXTURE_INTERIOR_DEFAULT)
    return _MOVABLE_CONTAINER_INTERIOR.get(container_class)


def _contained_child_bounds(
    node_id: str,
    container_id: str,
    graph: SemanticSceneGraph,
    parent_x: float,
    parent_y: float,
    child_hx: float,
    child_hy: float,
) -> tuple[float, float, float, float]:
    """Return deterministic relative bounds for one contained child.

    Multiple children in the same container are split into stable slots along
    the container's longer axis so Scenic does not have to discover a
    non-overlapping packing from identical broad ranges.
    """
    siblings = sorted(
        e.src_id for e in graph.edges if e.label == "contained_in" and e.dst_id == container_id
    )
    if node_id not in siblings or len(siblings) <= 1:
        return (
            -(parent_x / 2.0 - child_hx),
            parent_x / 2.0 - child_hx,
            -(parent_y / 2.0 - child_hy),
            parent_y / 2.0 - child_hy,
        )

    index = siblings.index(node_id)
    count = len(siblings)
    if parent_x >= parent_y:
        slot = parent_x / count
        center = -parent_x / 2.0 + slot * (index + 0.5)
        return (
            center - (slot / 2.0 - child_hx),
            center + (slot / 2.0 - child_hx),
            -(parent_y / 2.0 - child_hy),
            parent_y / 2.0 - child_hy,
        )

    slot = parent_y / count
    center = -parent_y / 2.0 + slot * (index + 0.5)
    return (
        -(parent_x / 2.0 - child_hx),
        parent_x / 2.0 - child_hx,
        center - (slot / 2.0 - child_hy),
        center + (slot / 2.0 - child_hy),
    )
