"""Types for the Libero-Infinity perturbation planner.

Each axis planner produces an independent plan struct. The PerturbationPlan
is a pure tuple of axis plans — composition is tuple construction, not merged
state. Cross-axis constraints are added only in single-pass validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from libero_infinity.ir.nodes import PlanDiagnostics


class InfeasiblePerturbationError(Exception):
    """Raised when a perturbation envelope collapses to zero volume."""

    def __init__(self, msg: str, diagnostics: PlanDiagnostics | None = None) -> None:
        super().__init__(msg)
        self.diagnostics = diagnostics


@dataclass
class AxisEnvelope:
    """The perturbation envelope for a single axis."""

    lo: float
    hi: float
    axis: str

    def validate(self) -> None:
        """Range degeneracy guard: assert lo < hi."""
        if self.lo >= self.hi:
            raise InfeasiblePerturbationError(
                f"Degenerate envelope for axis '{self.axis}': lo={self.lo} >= hi={self.hi}"
            )


@dataclass
class LightingPlan:
    """Lighting perturbation envelope."""

    intensity_lo: float
    intensity_hi: float
    ambient_lo: float
    ambient_hi: float
    position_jitter: float  # max XY jitter on light source position


@dataclass
class TexturePlan:
    """Texture perturbation plan — table surface texture variation.

    table_texture: "random" picks a random MuJoCo texture at runtime;
                   any other string is treated as a named texture id.
    texture_candidates: optional list of named textures to sample from
                        (empty list means use "random").
    """

    table_texture: str = "random"
    texture_candidates: list[str] = field(default_factory=list)


@dataclass
class BackgroundPlan:
    """Background perturbation plan — wall and floor texture variation.

    wall_texture: "random" picks a random MuJoCo texture at runtime;
                  any other string is treated as a named texture to look up
                  via model.texture_name2id().  On miss the simulator falls
                  back to a random loaded texture.
    floor_texture: same semantics as wall_texture.
    texture_candidates: pool of texture names to sample from uniformly.
                        Empty list means use wall_texture/floor_texture directly.
    """

    wall_texture: str = "random"
    floor_texture: str = "random"
    texture_candidates: list[str] = field(default_factory=list)


@dataclass
class PositionPlan:
    """Position plan for a single object — independent of other axes."""

    object_name: str
    x_envelope: AxisEnvelope  # lo/hi in workspace coords
    y_envelope: AxisEnvelope
    support_name: str  # name of the support surface/object
    use_relative_positioning: bool = False  # True for stacked objects
    yaw_lo: float = -3.14159
    yaw_hi: float = 3.14159
    exclusion_zones: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )  # (x0, y0, x1, y1)
    exclusion_min_distance: float | None = None  # distance-based fallback
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerturbationPlan:
    """Axis tuple — composition is tuple construction, NOT merged state.

    Each axis plan is computed independently from the SemanticSceneGraph.
    Cross-axis constraints are added only in the single-pass validation.
    """

    # Position axis
    position_plans: dict[str, PositionPlan] = field(default_factory=dict)  # object_name -> plan
    # Object axis (filled by planner-axes)
    object_substitutions: dict[str, list[str]] = field(
        default_factory=dict
    )  # object_name -> variant list
    # Articulation axis (filled by planner-axes)
    articulation_plans: dict[str, Any] = field(
        default_factory=dict
    )  # fixture_name -> ArticulationPlan
    # Camera axis (filled by planner-axes)
    camera_plan: Any | None = None
    # Lighting axis (filled by planner-axes)
    lighting_plan: LightingPlan | None = None
    # Texture axis (filled by planner-axes)
    texture_plan: TexturePlan | None = None
    # Background axis (filled by planner-axes)
    background_plan: BackgroundPlan | None = None
    # Distractor axis (filled by planner-axes)
    distractor_budget: int = 0
    distractor_classes: list[str] = field(default_factory=list)
    # Active axes
    active_axes: frozenset[str] = field(default_factory=frozenset)
    # Diagnostics
    diagnostics: PlanDiagnostics = field(default_factory=PlanDiagnostics)
    # Anti-trivialization
    anti_trivialization_active: bool = False
    max_initial_goal_satisfaction: float = 0.5  # at most floor(k/2)/k goals satisfied
