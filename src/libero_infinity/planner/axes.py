"""Per-axis perturbation planners for Libero-Infinity.

Each function is a pure function of the SemanticSceneGraph that computes
an independent per-axis plan without cross-axis logic.
"""

from __future__ import annotations

import math
import pathlib as _pathlib
from dataclasses import dataclass

from libero_infinity.asset_registry import (
    DEFAULT_DISTRACTOR_POOL,
    OBJECT_DIMENSIONS,
    UNLOADABLE_ASSET_CLASSES,
    get_dimensions,
    get_distractor_pool,
    get_variants,
)
from libero_infinity.ir.nodes import (
    FixtureNode,
    MovableSupportNode,
    ObjectNode,
    PlanDiagnostics,
)
from libero_infinity.ir.scene_graph import SemanticSceneGraph
from libero_infinity.planner.types import (
    BackgroundPlan,
    LightingPlan,
    RobotInitPlan,
    SensorNoisePlan,
    TexturePlan,
)

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

_CONTAINER_INTERIOR_SCALE = 0.85  # interior ≈ 85 % of bounding box dims (z-axis only)
_PANDA_INIT_QPOS = (
    0.0,
    -1.61037389e-01,
    0.0,
    -2.44459747e00,
    0.0,
    2.22675220e00,
    math.pi / 4.0,
)
_PANDA_JOINT_NAMES = (
    "robot0_joint1",
    "robot0_joint2",
    "robot0_joint3",
    "robot0_joint4",
    "robot0_joint5",
    "robot0_joint6",
    "robot0_joint7",
)


def _container_interior_dims(container_class: str) -> tuple[float, float, float]:
    """Return estimated interior (w, l, h) for a container.

    The (w, l) part is sourced from
    ``planner.position.container_interior_xy`` so that the variant-filter
    in ``plan_object`` and the in-container position sampler in
    ``plan_position`` agree on what "interior" means. The height (h) is
    still derived from the registry bounding box, scaled by
    ``_CONTAINER_INTERIOR_SCALE`` — there is no explicit per-class
    height table, and z-fitting is much less sensitive to the exact
    interior estimate than xy-footprint fitting.
    """
    from libero_infinity.planner.position import container_interior_xy_by_class

    xy = container_interior_xy_by_class(container_class)
    _w_bbox, _l_bbox, h = get_dimensions(container_class)
    if xy is not None:
        return (xy[0], xy[1], h * _CONTAINER_INTERIOR_SCALE)
    # Fallback: scale the full bounding box by the legacy 0.85 ratio.
    s = _CONTAINER_INTERIOR_SCALE
    return (_w_bbox * s, _l_bbox * s, h * s)


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

        # Containment-dimensional filtering: variant must fit inside container.
        # We can only apply this filter when the container's interior dimensions
        # are *known*. Fixture containers (microwave, cabinet, drawer, …) and
        # any class missing from OBJECT_DIMENSIONS fall through to the
        # conservative default (0.08 × 0.08 × 0.06), which is smaller than
        # almost every graspable variant — silently collapsing the variant pool
        # to the canonical class. Skipping the filter in that case preserves
        # the full variant pool; positional sampling provides the actual
        # geometric feasibility guard.
        contained_edges = [e for e in graph.edges_from(node_id) if e.label == "contained_in"]
        if contained_edges:
            container_node = graph.get_node(contained_edges[0].dst_id)
            if (
                container_node is not None
                and not isinstance(container_node, FixtureNode)
                and container_node.object_class in OBJECT_DIMENSIONS
            ):
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

        # Stacking dimensional check: variant footprint must fit on support.
        # The previous 20% over-footprint tolerance silently allowed stacks
        # whose centre of mass projected outside the support surface (e.g. a
        # plate that overhangs by 10% on each side). Tightened to a small
        # 5% tolerance — enough to absorb measurement noise on hand-edited
        # bounding-box JSON entries without admitting visibly unstable stacks.
        stacked_edges = [e for e in graph.edges_from(node_id) if e.label == "stacked_on"]
        if stacked_edges:
            support_node = graph.get_node(stacked_edges[0].dst_id)
            if support_node is not None:
                sw, sl, _ = get_dimensions(support_node.object_class)
                filtered = []
                for v in variants:
                    vw, vl, _ = get_dimensions(v)
                    if vw <= sw * 1.05 and vl <= sl * 1.05:
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

    Two emission paths:

    1. **Goal-reachability override (always active).** When the BDDL goal
       requires the robot to place an object *inside* an articulated fixture
       (``In <obj> <fixture_region>`` with ``contained=True``), the fixture
       must be Open at init regardless of which perturbation axes the user
       requested — otherwise the goal is geometrically unreachable.
    2. **Articulation axis (gated).** When ``"articulation"`` is in
       ``request_axes`` the planner additionally perturbs articulatable
       fixtures whose canonical init state is *not* dictated by goal
       reachability (e.g. opening the microwave for a "put-on-table" task).

    The previous implementation always emitted an "Open" plan for every
    microwave/cabinet — silently turning the canonical (no-perturbation)
    baseline into an *already-perturbed* environment for any task with one
    of those fixtures. That broke benchmark comparability: any reported
    "no-perturbation" success rate was actually measured on a perturbed
    init state. Gating non-mandatory plans on the axis request restores
    parity between the BDDL canonical state and the planner's "no axes"
    baseline.

    Args:
        graph: The semantic scene graph for the task.
        request_axes: Set of active perturbation axis names.
        diagnostics: Diagnostics collector.

    Returns:
        Dict mapping fixture instance_name -> ArticulationPlan.
    """
    result: dict[str, ArticulationPlan] = {}
    axis_requested = "articulation" in request_axes

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

        # Check if any object must end up *inside* this fixture (In goal).
        # We can't rely on ``ObjectNode.contained`` here because that flag
        # captures *initial* containment from the BDDL ``:init`` block, not
        # *goal* containment from the ``:goal`` block. We instead parse the
        # goal text directly for ``(In <obj> <region>)`` predicates whose
        # region is anchored to this fixture (either by name match or by
        # the region node's ``target`` attribute).
        need_open_at_init = False
        fixture_region_names: set[str] = {
            region.instance_name
            for region in graph.nodes.values()
            if getattr(region, "target", None) == node.instance_name
        }
        for edge in graph.edges:
            if edge.label != "goal_target":
                continue
            # Match either a region whose target is this fixture, or a
            # region name that contains the fixture's instance name as a
            # substring (covers BDDL region patterns like
            # ``<fixture>_<sub>_region`` even when the RegionNode wasn't
            # registered with a matching target).
            in_fixture_region = (
                edge.dst_id in fixture_region_names or node.instance_name in edge.dst_id
            )
            if not in_fixture_region:
                continue
            # The graph_builder doesn't preserve the predicate kind
            # (On/In) on the edge — treat any goal_target into a fixture-
            # anchored region as a containment requirement. This is
            # conservative: if the BDDL says (On X <fixture_region>) we'd
            # also flag Open. That matches the original behaviour for the
            # axis-requested case and the previous always-Open logic; the
            # alternative (false negative) silently breaks goal
            # reachability for In-style goals which is far worse.
            need_open_at_init = True
            break

        # If the axis isn't requested AND there's no goal-reachability
        # requirement, leave the fixture alone — its BDDL canonical init
        # state is what the simulator should load.
        if not axis_requested and not need_open_at_init:
            continue

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
                # axis_requested implied here (we returned earlier otherwise).
                state_kind = "Open"
                reason = "articulation axis requested — Open init"
        elif family_name == "stove":
            # Stove starts off by default; goal is typically to turn it on.
            state_kind = "Turnoff"
            reason = "stove default init — Turnoff"
            # Bug 17 fix: surface the silent fallback as a diagnostic.
            diagnostics.narrow_axis(
                "articulation",
                f"{node_id}: stove fixture defaulted to 'Turnoff' init state",
            )
        else:
            # Unknown family: use first available state.
            state_kind = next(iter(ranges))
            reason = f"unknown family '{family_name}' — using first state"
            # Bug 17 fix: surface the silent fallback as a diagnostic.
            diagnostics.narrow_axis(
                "articulation",
                f"{node_id}: unknown family '{family_name}' — defaulting to "
                f"first state '{state_kind}'",
            )

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


def plan_robot(
    graph: SemanticSceneGraph,
    request_axes: frozenset[str],
    diagnostics: PlanDiagnostics,
) -> RobotInitPlan | None:
    """Plan joint-space robot reset perturbation for Panda tasks."""
    del graph, diagnostics
    if "robot" not in request_axes:
        return None

    return RobotInitPlan(
        canonical_qpos=_PANDA_INIT_QPOS,
        radius_lo=0.1,
        radius_hi=0.5,
        joint_names=_PANDA_JOINT_NAMES,
        robot_model="Panda",
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
    free_area: float | None = None,
    *,
    position_plans: dict | None = None,
    object_substitutions: dict[str, list[str]] | None = None,
) -> tuple[int, list[str]]:
    """Plan distractor object budget and class pool.

    Budget is derived from the *actual* free area on the workspace, computed
    as

        free_area = table_area
                    - sum(planned position-envelope rectangles)
                    - sum(substituted asset bounding-box footprints)

    rather than the previous global default of 0.09 m². The previous
    implementation also applied a hard
    ``budget = min(2, budget) if {position, object, distractor} ⊆ axes``
    fudge to unblock Scenic's rejection sampler; that branch is removed —
    the empirical free-area calculation now produces a budget that is
    sampleable without the brute-force cap.

    Args:
        graph: The semantic scene graph for the task.
        request_axes: Set of active perturbation axis names.
        diagnostics: Diagnostics collector.
        free_area: Optional explicit free-area override (m²). When ``None``,
            the value is computed from ``position_plans`` /
            ``object_substitutions`` and the workspace bounds derived from
            the graph. The override remains available for unit tests.
        position_plans: The position planner's per-object plans (the
            envelope sizes are subtracted from the workspace area).
        object_substitutions: The object planner's per-object variant pools
            (used to take the *largest* substituted footprint per object,
            since asset substitution can grow object size).

    Returns:
        Tuple of (n_distractors, distractor_classes_list).
    """
    if "distractor" not in request_axes:
        return 0, []

    distractor_footprint = 0.01  # 10cm × 10cm = 0.01 m²

    if free_area is None:
        # Derive free area from the BDDL workspace bounds and the in-progress
        # plan. This runs after position / object planning in
        # ``plan_perturbations`` so the position envelopes and object
        # substitutions are already populated when this function executes.
        from libero_infinity.planner.position import _workspace_bounds_from_graph

        x_min, y_min, x_max, y_max = _workspace_bounds_from_graph(graph)
        table_area = max(0.0, (x_max - x_min) * (y_max - y_min))

        # Per-task-object reservation = (largest substituted footprint)²,
        # padded by the distractor-vs-task clearance margin (matches the
        # 0.13 m threshold the renderer emits in ``_render_constraints``).
        # We do *not* add the position envelope on top because envelopes
        # already overlap on the table — the task object's realised
        # footprint at any one sample is the un-padded AABB; clearance is
        # what the sampler enforces around that AABB.
        _DISTRACTOR_TASK_CLEARANCE = 0.13
        position_plans = position_plans or {}
        object_substitutions = object_substitutions or {}

        n_task = 0
        occupied = 0.0
        for node in graph.nodes.values():
            if not isinstance(node, (ObjectNode, MovableSupportNode)):
                continue
            instance_name = node.instance_name
            obj_class = node.object_class
            n_task += 1

            # Largest variant footprint (asset substitution can grow size).
            variants = object_substitutions.get(instance_name, [obj_class])
            footprint_w = footprint_l = 0.0
            for v in variants:
                vw, vl, _ = get_dimensions(v)
                footprint_w = max(footprint_w, vw)
                footprint_l = max(footprint_l, vl)
            if footprint_w == 0.0:
                vw, vl, _ = get_dimensions(obj_class)
                footprint_w, footprint_l = vw, vl

            occupied += (footprint_w + _DISTRACTOR_TASK_CLEARANCE) * (
                footprint_l + _DISTRACTOR_TASK_CLEARANCE
            )

        # Each distractor reserves (distractor_size + dist-dist clearance)²
        # of free area. Matches the renderer's
        # ``_DISTRACTOR_PAIR_CLEARANCE`` plus the distractor AABB side. Keep
        # the constants in sync with ``renderer/scenic_renderer.py`` (audit
        # E2): the renderer authors a 0.08 m AABB cube and emits a diagonal
        # clearance of ``sqrt(2 * 0.08²) ≈ 0.1131`` m between distractors.
        _DISTRACTOR_AABB_SIDE = 0.08
        _DISTRACTOR_PAIR_CLEARANCE = math.sqrt(2.0) * _DISTRACTOR_AABB_SIDE
        per_distractor_area = (_DISTRACTOR_AABB_SIDE + _DISTRACTOR_PAIR_CLEARANCE) ** 2

        free_area = max(0.0, table_area - occupied)

        # Joint-sampling safety factor: when position + object perturbations
        # are active the rejection-sampler must satisfy clearance against
        # task objects whose realised positions can fall anywhere in their
        # envelope. The combinatorics make budgets above ~table_area /
        # (n_task × 2 × per_distractor_area) unsampleable in practice — even
        # if the static free-area calculation says they should fit. We
        # therefore divide by an empirical density factor that scales with
        # the task-object count when {position, object, distractor} are
        # jointly requested.
        density_divisor = 1.0
        if {"position", "object", "distractor"}.issubset(request_axes) and n_task > 0:
            density_divisor = max(1.0, n_task * 1.5)

        raw_budget = math.floor(free_area / (per_distractor_area * density_divisor))
    else:
        raw_budget = math.floor(free_area / distractor_footprint)

    budget = min(5, max(0, raw_budget))

    if budget < raw_budget:
        diagnostics.narrow_axis(
            "distractor", f"budget capped at 5 (computed {raw_budget}, free_area={free_area:.3f})"
        )

    # Collect task-scene object classes to exclude from distractors
    scene_classes: set[str] = set()
    for node in graph.nodes.values():
        if isinstance(node, (ObjectNode, MovableSupportNode)):
            scene_classes.add(node.object_class)

    # Use the curated distractor pool instead of every asset variant class.
    distractor_classes = get_distractor_pool(exclude_classes=scene_classes)
    if not distractor_classes:
        distractor_classes = [
            cls
            for cls in DEFAULT_DISTRACTOR_POOL
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


# ---------------------------------------------------------------------------
# plan_sensor_noise
# ---------------------------------------------------------------------------


def plan_sensor_noise(
    graph: SemanticSceneGraph,
    request_axes: frozenset[str],
    diagnostics: PlanDiagnostics,
) -> SensorNoisePlan | None:
    """Plan sensor / image-noise perturbation.

    The plan exposes a (kinds, severity_range) pair that the renderer
    converts to two Scenic params:

        param sensor_noise_kind = Uniform("none", "gaussian_noise", …)
        param sensor_noise_severity = DiscreteRange(severity_lo, severity_hi)

    The simulator's ``_apply_sensor_noise`` post-processes
    ``obs["agentview_image"]`` (and the eye-in-hand camera, when present)
    at every ``step()`` by dispatching on the sampled kind.

    Args:
        graph: The semantic scene graph (unused; kept for API consistency).
        request_axes: Set of active perturbation axis names.
        diagnostics: Diagnostics collector.

    Returns:
        A SensorNoisePlan, or None if ``"sensor_noise"`` is not in
        ``request_axes``.
    """
    del graph, diagnostics
    if "sensor_noise" not in request_axes:
        return None
    return SensorNoisePlan()
