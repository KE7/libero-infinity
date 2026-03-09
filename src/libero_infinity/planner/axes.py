"""Per-axis perturbation planners for Libero-Infinity.

Each function is a pure function of the SemanticSceneGraph that computes
an independent per-axis plan without cross-axis logic.
"""

from __future__ import annotations

import math
import pathlib as _pathlib
from dataclasses import dataclass

from libero_infinity.asset_registry import (
    ASSET_VARIANTS,
    UNLOADABLE_ASSET_CLASSES,
    get_dimensions,
    get_variants,
)
from libero_infinity.ir.nodes import (
    FixtureNode,
    MovableSupportNode,
    ObjectNode,
    PlanDiagnostics,
)
from libero_infinity.ir.scene_graph import SemanticSceneGraph
from libero_infinity.planner.types import BackgroundPlan, LightingPlan, TexturePlan

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ArticulationPlan:
    """Initial articulation state plan for a single fixture."""

    fixture_name: str
    state_kind: str  # 'Open', 'Close', 'Turnon', 'Turnoff'
    lo: float
    hi: float
    reason: str
    goal_reachability_ok: bool = True


@dataclass
class CameraPlan:
    """Camera perturbation envelope."""

    azimuth_lo: float = -15.0
    azimuth_hi: float = 15.0
    elevation_lo: float = -10.0
    elevation_hi: float = 10.0
    distance_lo: float = 0.9
    distance_hi: float = 1.1
    visibility_constrained: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CONTAINER_INTERIOR_SCALE = 0.85  # interior ≈ 85% of bounding box dims


def _container_interior_dims(container_class: str) -> tuple[float, float, float]:
    """Return estimated interior (w, l, h) for a container fixture class."""
    w, length, h = get_dimensions(container_class)
    s = _CONTAINER_INTERIOR_SCALE
    return (w * s, length * s, h * s)


# ---------------------------------------------------------------------------
# plan_object
# ---------------------------------------------------------------------------


def plan_object(
    graph: SemanticSceneGraph,
    request_axes: frozenset[str],
    diagnostics: PlanDiagnostics,
) -> dict[str, list[str]]:
    """Plan object variant substitutions for each movable object.

    Args:
        graph: The semantic scene graph for the task.
        request_axes: Set of active perturbation axis names.
        diagnostics: Diagnostics collector.

    Returns:
        Dict mapping object instance_name -> list of candidate variant classes.
        Only objects with 2+ reachable variants are included.
    """
    if "object" not in request_axes:
        return {}

    result: dict[str, list[str]] = {}

    for node_id, node in graph.nodes.items():
        if not isinstance(node, (ObjectNode, MovableSupportNode)):
            continue

        obj_class = node.object_class
        variants = get_variants(obj_class, include_canonical=True, require_loadable=True)

        # Containment-dimensional filtering: variant must fit inside container
        contained_edges = [e for e in graph.edges_from(node_id) if e.label == "contained_in"]
        if contained_edges:
            container_node = graph.get_node(contained_edges[0].dst_id)
            if container_node is not None:
                iw, il, ih = _container_interior_dims(container_node.object_class)
                filtered = []
                for v in variants:
                    vw, vl, vh = get_dimensions(v)
                    if vw <= iw and vl <= il and vh <= ih:
                        filtered.append(v)
                if filtered:
                    variants = filtered
                else:
                    variants = [obj_class]
                    diagnostics.narrow_axis(
                        "object",
                        f"{node_id}: all variants exceed container interior "
                        f"{contained_edges[0].dst_id}",
                    )

        # Stacking dimensional check: variant footprint must fit on support
        stacked_edges = [e for e in graph.edges_from(node_id) if e.label == "stacked_on"]
        if stacked_edges:
            support_node = graph.get_node(stacked_edges[0].dst_id)
            if support_node is not None:
                sw, sl, _ = get_dimensions(support_node.object_class)
                filtered = []
                for v in variants:
                    vw, vl, _ = get_dimensions(v)
                    # Allow 20% tolerance beyond support surface
                    if vw <= sw * 1.2 and vl <= sl * 1.2:
                        filtered.append(v)
                if filtered:
                    variants = filtered
                else:
                    variants = [obj_class]
                    diagnostics.narrow_axis(
                        "object",
                        f"{node_id}: all variants too large to stack on {stacked_edges[0].dst_id}",
                    )

        if not variants:
            diagnostics.drop_axis("object", f"{node_id}: variant pool collapsed to zero")
            continue

        # Skip objects with no actual substitution choice
        if len(variants) == 1 and variants[0] == obj_class:
            continue

        result[node.instance_name] = variants

    return result


# ---------------------------------------------------------------------------
# plan_articulation
# ---------------------------------------------------------------------------


def plan_articulation(
    graph: SemanticSceneGraph,
    request_axes: frozenset[str],
    diagnostics: PlanDiagnostics,
) -> dict[str, ArticulationPlan]:
    """Plan initial articulation states for articulatable fixtures.

    Always runs for articulatable fixtures regardless of request_axes, to
    ensure goal-reachability is never violated.

    Args:
        graph: The semantic scene graph for the task.
        request_axes: Set of active perturbation axis names (informational only).
        diagnostics: Diagnostics collector.

    Returns:
        Dict mapping fixture instance_name -> ArticulationPlan.
    """
    result: dict[str, ArticulationPlan] = {}

    for node_id, node in graph.nodes.items():
        if not isinstance(node, FixtureNode):
            continue
        if not node.is_articulatable:
            continue

        fixture_class = node.object_class
        art_model = graph.articulation_model
        ranges = art_model.articulation_ranges.get(fixture_class)
        if not ranges:
            continue

        # Check if any object must end up inside this fixture (In goal)
        need_open_at_init = False
        for edge in graph.edges_to(node_id):
            if edge.label == "goal_target":
                src_node = graph.get_node(edge.src_id)
                if isinstance(src_node, ObjectNode) and src_node.contained:
                    need_open_at_init = True
                    break

        # Determine initial state kind and range
        family = art_model.get_family(fixture_class)
        if family is None:
            continue

        family_name, _kind = family

        if family_name in ("microwave", "cabinet"):
            if need_open_at_init:
                state_kind = "Open"
                reason = "goal requires interior access — init must be Open"
            else:
                # Default: start Open (robot can work with fixture)
                state_kind = "Open"
                reason = "default articulation perturbation — Open init"
        elif family_name == "stove":
            # Stove starts off by default; goal is typically to turn it on
            state_kind = "Turnoff"
            reason = "stove default init — Turnoff"
        else:
            # Unknown family: use first available state
            state_kind = next(iter(ranges))
            reason = f"unknown family '{family_name}' — using first state"

        state_range = ranges.get(state_kind)
        if state_range is None:
            diagnostics.narrow_axis(
                "articulation",
                f"{node_id}: state_kind '{state_kind}' not in ranges {list(ranges)}",
            )
            continue

        lo, hi = state_range
        result[node.instance_name] = ArticulationPlan(
            fixture_name=node.instance_name,
            state_kind=state_kind,
            lo=lo,
            hi=hi,
            reason=reason,
            goal_reachability_ok=True,
        )

    return result


# ---------------------------------------------------------------------------
# plan_camera
# ---------------------------------------------------------------------------


def plan_camera(
    graph: SemanticSceneGraph,
    request_axes: frozenset[str],
    diagnostics: PlanDiagnostics,
) -> CameraPlan | None:
    """Plan camera perturbation envelope.

    Constrains the camera sub-envelope based on must_remain_visible_with edges.

    Args:
        graph: The semantic scene graph for the task.
        request_axes: Set of active perturbation axis names.
        diagnostics: Diagnostics collector.

    Returns:
        A CameraPlan, or None if the camera axis is dropped.
    """
    if "camera" not in request_axes:
        return None

    # Collect visibility targets (objects that must remain visible)
    vis_edges = graph.edges_by_label("must_remain_visible_with")
    n_targets = len(vis_edges)

    if n_targets == 0:
        # No visibility constraints — use full default envelope
        return CameraPlan()

    # Constrain sub-envelope based on number of visibility targets.
    # More targets → tighter ranges to keep everything in frame.
    if n_targets <= 2:
        az_lo, az_hi = -10.0, 10.0
        el_lo, el_hi = -7.0, 7.0
    else:
        az_lo, az_hi = -8.0, 8.0
        el_lo, el_hi = -5.0, 5.0

    # Sub-envelope degeneracy check (should never happen with above values)
    if az_lo >= az_hi or el_lo >= el_hi:
        diagnostics.drop_axis(
            "camera",
            f"visibility sub-envelope collapsed with {n_targets} targets",
        )
        return None

    diagnostics.narrow_axis(
        "camera",
        f"constrained to ±{az_hi}° azimuth for {n_targets} visibility targets",
    )
    return CameraPlan(
        azimuth_lo=az_lo,
        azimuth_hi=az_hi,
        elevation_lo=el_lo,
        elevation_hi=el_hi,
        distance_lo=0.9,
        distance_hi=1.1,
        visibility_constrained=True,
    )


# ---------------------------------------------------------------------------
# plan_lighting
# ---------------------------------------------------------------------------


def plan_lighting(
    graph: SemanticSceneGraph,
    request_axes: frozenset[str],
    diagnostics: PlanDiagnostics,
) -> LightingPlan | None:
    """Plan lighting perturbation. Returns fixed safe ranges.

    Ranges match scenic/lighting_perturbation.scenic exactly:
      intensity_min=0.4, intensity_max=2.0
      ambient_min=0.05,  ambient_max=0.6
      light_pos_range=0.5

    Args:
        graph: The semantic scene graph for the task.
        request_axes: Set of active perturbation axis names.
        diagnostics: Diagnostics collector.

    Returns:
        A LightingPlan with fixed safe ranges, or None if axis not requested.
    """
    if "lighting" not in request_axes:
        return None

    return LightingPlan(
        intensity_lo=0.4,
        intensity_hi=2.0,
        ambient_lo=0.05,
        ambient_hi=0.6,
        position_jitter=0.5,
    )


# ---------------------------------------------------------------------------
# plan_texture
# ---------------------------------------------------------------------------


def plan_texture(
    graph: SemanticSceneGraph,
    request_axes: frozenset[str],
    diagnostics: PlanDiagnostics,
) -> TexturePlan | None:
    """Plan texture perturbation for the table surface.

    Matches the simulator behaviour:
      - Emits ``table_texture = "random"`` so the simulator picks a random
        MuJoCo texture at runtime (``_apply_texture_perturbation``).
      - No scene-graph analysis required: texture variation is table-surface
        only and independent of task objects.

    Args:
        graph: The semantic scene graph for the task (unused; kept for API
               consistency with other axis planners).
        request_axes: Set of active perturbation axis names.
        diagnostics: Diagnostics collector.

    Returns:
        A TexturePlan with ``table_texture="random"``, or None if axis not
        requested.
    """
    if "texture" not in request_axes:
        return None

    return TexturePlan(table_texture="random", texture_candidates=[])


# ---------------------------------------------------------------------------
# plan_distractor
# ---------------------------------------------------------------------------


def plan_distractor(
    graph: SemanticSceneGraph,
    request_axes: frozenset[str],
    diagnostics: PlanDiagnostics,
    free_area: float = 0.09,
) -> tuple[int, list[str]]:
    """Plan distractor object budget and class pool.

    Dynamic budget: n = min(5, floor(free_area / distractor_footprint)).

    Args:
        graph: The semantic scene graph for the task.
        request_axes: Set of active perturbation axis names.
        diagnostics: Diagnostics collector.
        free_area: Estimated free workspace area in m² (default 0.09 = 30cm×30cm).

    Returns:
        Tuple of (n_distractors, distractor_classes_list).
    """
    if "distractor" not in request_axes:
        return 0, []

    distractor_footprint = 0.01  # 10cm × 10cm = 0.01 m²
    budget = min(5, math.floor(free_area / distractor_footprint))

    if budget < math.floor(free_area / distractor_footprint):
        diagnostics.narrow_axis(
            "distractor", f"budget capped at {budget} (free_area={free_area:.3f})"
        )

    # Collect task-scene object classes to exclude from distractors
    scene_classes: set[str] = set()
    for node in graph.nodes.values():
        if isinstance(node, (ObjectNode, MovableSupportNode)):
            scene_classes.add(node.object_class)

    # Available distractor classes: all ASSET_VARIANTS keys minus unloadable and task objects
    distractor_classes = [
        cls
        for cls in ASSET_VARIANTS.keys()
        if cls not in UNLOADABLE_ASSET_CLASSES and cls not in scene_classes
    ]

    return budget, distractor_classes


# ---------------------------------------------------------------------------
# Background texture constants and helpers
# ---------------------------------------------------------------------------

# Absolute path to the LIBERO textures directory.
# axes.py lives at src/libero_infinity/planner/axes.py  →  parents[3] = repo root
_LIBERO_TEXTURE_DIR: _pathlib.Path = (
    _pathlib.Path(__file__).resolve().parents[3]
    / "vendor"
    / "libero"
    / "libero"
    / "libero"
    / "assets"
    / "textures"
)

# Fallback list — used when disk enumeration fails (e.g. in unit tests without
# the full vendor tree present).  Derived from the 35 PNGs found in
# vendor/libero/libero/libero/assets/textures/ as of the initial implementation.
LIBERO_BACKGROUND_TEXTURES: tuple[str, ...] = (
    "brown_ceramic_tile",
    "canvas_sky_blue",
    "capriccio_sky",
    "ceramic",
    "cream-plaster",
    "dapper_gray_floor",
    "dark_blue_wall",
    "dark_floor_texture",
    "dark_gray_plaster",
    "dark_green_plaster_wall",
    "gray_ceramic_tile",
    "gray_floor",
    "gray_plaster",
    "gray_wall",
    "grigia_caldera_porcelain_floor",
    "kona_gotham",
    "light_blue_wall",
    "light_floor",
    "light-gray-floor-tile",
    "light-gray-plaster",
    "light_gray_plaster",
    "light_grey_plaster",
    "marble_floor",
    "martin_novak_wood_table",
    "meeka-beige-plaster",
    "new_light_gray_plaster",
    "rustic_floor",
    "seamless_wood_planks_floor",
    "smooth_light_gray_plaster",
    "stucco_wall",
    "table_light_wood",
    "tile_grigia_caldera_porcelain_floor",
    "white_marble_floor",
    "white_wall",
    "yellow_linen_wall_texture",
)


def _discover_background_textures() -> list[str]:
    """Enumerate background texture base-names from LIBERO assets on disk.

    Returns the stem (filename without extension) of every PNG file found in
    the LIBERO textures directory.  Falls back to the hardcoded
    ``LIBERO_BACKGROUND_TEXTURES`` tuple when the directory is missing or
    inaccessible (e.g. in unit tests that run without the vendor tree).
    """
    try:
        names = sorted(p.stem for p in _LIBERO_TEXTURE_DIR.glob("*.png"))
        return names if names else list(LIBERO_BACKGROUND_TEXTURES)
    except Exception:
        return list(LIBERO_BACKGROUND_TEXTURES)


# ---------------------------------------------------------------------------
# plan_background
# ---------------------------------------------------------------------------


def plan_background(
    graph: SemanticSceneGraph,
    request_axes: frozenset[str],
    diagnostics: PlanDiagnostics,
) -> BackgroundPlan | None:
    """Plan background (wall + floor) texture perturbation.

    Discovers the pool of available LIBERO texture assets on disk and returns
    a BackgroundPlan whose ``texture_candidates`` field lists every available
    texture name.  At Scenic scene-generation time the renderer emits a
    ``Uniform(...)`` distribution over these candidates so that each generated
    episode carries a specific (reproducible) texture name in its params.

    The simulator resolves the sampled name via ``model.texture_name2id()``;
    on a miss (the named texture is not loaded in the current MuJoCo model) it
    falls back to a random loaded texture rather than silently no-oping.

    Args:
        graph: The semantic scene graph (unused — kept for API consistency with
               other axis planners).
        request_axes: Set of active perturbation axis names.
        diagnostics: Diagnostics collector.

    Returns:
        A BackgroundPlan with the full texture candidate pool, or None if the
        ``"background"`` axis is not in ``request_axes``.
    """
    if "background" not in request_axes:
        return None

    candidates = _discover_background_textures()
    return BackgroundPlan(
        wall_texture="random",
        floor_texture="random",
        texture_candidates=candidates,
    )
