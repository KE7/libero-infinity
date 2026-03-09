"""IR node types for the Libero-Infinity semantic scene graph.

Defines the typed vocabulary of nodes, edges, and models used to represent
a parsed BDDL task as a semantic graph — independent of any Scenic syntax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Literal type aliases
# ---------------------------------------------------------------------------

NodeType = Literal[
    "workspace",
    "fixture",
    "movable_support",
    "object",
    "region",
    "camera",
    "light",
    "distractor_slot",
]

EdgeLabel = Literal[
    "supported_by",
    "contained_in",
    "stacked_on",
    "anchored_to",
    "articulated_by",
    "must_remain_visible_with",
    "goal_target",
]

SpatialRelationKind = Literal["on_surface", "inside", "stacked"]

ArticulationFamily = Literal["microwave", "cabinet", "stove"]

ArticulationKind = Literal["door", "drawer", "knob"]


# ---------------------------------------------------------------------------
# Scene node hierarchy
# ---------------------------------------------------------------------------


@dataclass
class SceneNode:
    """Base scene node with identity and type information."""

    node_id: str
    node_type: NodeType
    instance_name: str
    object_class: str
    metadata: dict = field(default_factory=dict)


@dataclass
class WorkspaceNode(SceneNode):
    """Root workspace surface (table, floor, etc.)."""

    surface_bounds: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.node_type = "workspace"


@dataclass
class FixtureNode(SceneNode):
    """Non-movable fixture (cabinet, stove, microwave, etc.)."""

    placement_target: str | None = None
    init_x: float | None = None
    init_y: float | None = None
    init_yaw: float | None = None
    is_articulatable: bool = False

    def __post_init__(self) -> None:
        self.node_type = "fixture"


@dataclass
class MovableSupportNode(SceneNode):
    """Movable object that also serves as a support surface for another object."""

    placement_target: str | None = None
    init_x: float | None = None
    init_y: float | None = None
    init_yaw: float | None = None
    stacked_on: str | None = None

    def __post_init__(self) -> None:
        self.node_type = "movable_support"


@dataclass
class ObjectNode(SceneNode):
    """Standard movable object."""

    placement_target: str | None = None
    init_x: float | None = None
    init_y: float | None = None
    init_yaw: float | None = None
    stacked_on: str | None = None
    contained: bool = False

    def __post_init__(self) -> None:
        self.node_type = "object"


@dataclass
class RegionNode(SceneNode):
    """Named placement region from the BDDL (:regions ...) block."""

    target: str = ""
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    yaw_min: float | None = None
    yaw_max: float | None = None

    def __post_init__(self) -> None:
        self.node_type = "region"


@dataclass
class CameraNode(SceneNode):
    """Virtual camera node."""

    def __post_init__(self) -> None:
        self.node_type = "camera"


@dataclass
class LightNode(SceneNode):
    """Virtual light node."""

    def __post_init__(self) -> None:
        self.node_type = "light"


@dataclass
class DistractorSlotNode(SceneNode):
    """One of the 5 reserved distractor object slots."""

    slot_index: int = 0

    def __post_init__(self) -> None:
        self.node_type = "distractor_slot"


# ---------------------------------------------------------------------------
# Scene edge
# ---------------------------------------------------------------------------


@dataclass
class SceneEdge:
    """Directed edge between two scene nodes."""

    src_id: str
    dst_id: str
    label: EdgeLabel
    metadata: dict = field(default_factory=dict)
    spatial_kind: SpatialRelationKind | None = None


# ---------------------------------------------------------------------------
# Articulation model
# ---------------------------------------------------------------------------


@dataclass
class ArticulationModel:
    """Encodes knowledge about articulatable fixture families and joint ranges."""

    # fixture_class -> (family, kind)
    fixture_families: dict[str, tuple] = field(default_factory=dict)
    # fixture_class -> {state_name -> (lo, hi)}
    articulation_ranges: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    immobile_workspace_fixtures: frozenset = field(default_factory=lambda: frozenset())
    root_workspace_fixtures: frozenset = field(default_factory=lambda: frozenset())

    @classmethod
    def canonical(cls) -> ArticulationModel:
        """Return the canonical articulation model with all known LIBERO fixtures."""
        return cls(
            fixture_families={
                "microwave": ("microwave", "door"),
                "wooden_cabinet": ("cabinet", "drawer"),
                "white_cabinet": ("cabinet", "drawer"),
                "flat_stove": ("stove", "knob"),
            },
            articulation_ranges={
                "microwave": {
                    "Open": (-2.094, -1.3),
                    "Close": (-0.005, 0.0),
                },
                "wooden_cabinet": {
                    "Open": (-0.16, -0.14),
                    "Close": (0.0, 0.005),
                },
                "white_cabinet": {
                    "Open": (-0.16, -0.14),
                    "Close": (0.0, 0.005),
                },
                "flat_stove": {
                    "Turnon": (0.5, 2.1),
                    "Turnoff": (-0.005, 0.0),
                },
            },
            root_workspace_fixtures=frozenset(
                {
                    "table",
                    "kitchen_table",
                    "living_room_table",
                    "study_table",
                    "floor",
                }
            ),
        )

    def is_articulatable(self, fixture_class: str) -> bool:
        """Return True if fixture_class has known articulation."""
        return fixture_class in self.fixture_families

    def get_range(self, fixture_class: str, state: str) -> tuple | None:
        """Return the (lo, hi) joint range for the given fixture class and state."""
        ranges = self.articulation_ranges.get(fixture_class)
        if ranges is None:
            return None
        return ranges.get(state)

    def get_family(self, fixture_class: str) -> tuple | None:
        """Return the (family, kind) tuple for the given fixture class."""
        return self.fixture_families.get(fixture_class)


# ---------------------------------------------------------------------------
# Plan diagnostics
# ---------------------------------------------------------------------------


@dataclass
class PlanDiagnostics:
    """Tracks which perturbation axes were constrained, narrowed, or dropped."""

    constrained_axes: list = field(default_factory=list)
    narrowed_axes: list = field(default_factory=list)
    dropped_axes: list = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def drop_axis(self, axis: str, reason: str) -> None:
        """Mark an axis as completely dropped."""
        self.dropped_axes.append(axis)
        self.reasons[axis] = reason

    def narrow_axis(self, axis: str, reason: str) -> None:
        """Mark an axis as narrowed (reduced range but still active)."""
        self.narrowed_axes.append(axis)
        self.reasons[axis] = reason

    def constrain_axis(self, axis: str, reason: str) -> None:
        """Mark an axis as constrained (fixed to a specific value)."""
        self.constrained_axes.append(axis)
        self.reasons[axis] = reason
