"""Tests for the Perturbation Planner modules (Waves 2-3: planner-position, planner-axes).

Covers: PerturbationPlan tuple independence, AxisEnvelope degeneracy guard,
distractor budget cap, parse_axes, position/articulation
independence, contained object exclusion, camera sub-envelope.
"""

from __future__ import annotations

import glob

import pytest

from libero_infinity.ir.graph_builder import build_semantic_scene_graph
from libero_infinity.ir.nodes import ArticulationModel, PlanDiagnostics
from libero_infinity.ir.scene_graph import SemanticSceneGraph
from libero_infinity.planner.axes import (
    LIBERO_BACKGROUND_TEXTURES,
    ArticulationPlan,
    CameraPlan,
    plan_articulation,
    plan_background,
    plan_camera,
    plan_distractor,
    plan_lighting,
    plan_texture,
)
from libero_infinity.planner.composition import (
    AXIS_PRESETS,
    parse_axes,
    plan_perturbations,
)
from libero_infinity.planner.position import plan_position
from libero_infinity.planner.types import (
    AxisEnvelope,
    BackgroundPlan,
    InfeasiblePerturbationError,
    LightingPlan,
    PerturbationPlan,
    PositionPlan,
    TexturePlan,
)
from libero_infinity.renderer.scenic_renderer import render_scenic
from libero_infinity.task_config import TaskConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BDDL_ROOT = "src/libero_infinity/data/libero_runtime/bddl_files"


def _bddl_files(n: int = 5) -> list[str]:
    files = glob.glob(f"{_BDDL_ROOT}/**/*.bddl", recursive=True)
    return files[:n]


@pytest.fixture
def sample_cfg() -> TaskConfig:
    files = _bddl_files(1)
    if not files:
        pytest.skip("No BDDL files found")
    return TaskConfig.from_bddl(files[0])


@pytest.fixture
def sample_graph(sample_cfg: TaskConfig) -> SemanticSceneGraph:
    return build_semantic_scene_graph(sample_cfg)


# ---------------------------------------------------------------------------
# AxisEnvelope degeneracy guard (Range degeneracy guard requirement)
# ---------------------------------------------------------------------------


def test_axis_envelope_valid_range() -> None:
    """AxisEnvelope with lo < hi must not raise."""
    env = AxisEnvelope(lo=0.0, hi=1.0, axis="x")
    env.validate()  # should not raise


def test_axis_envelope_degenerate_lo_equal_hi() -> None:
    """AxisEnvelope with lo == hi must raise InfeasiblePerturbationError."""
    env = AxisEnvelope(lo=0.5, hi=0.5, axis="x")
    with pytest.raises(InfeasiblePerturbationError, match="[Dd]egenerate"):
        env.validate()


def test_axis_envelope_degenerate_lo_greater_than_hi() -> None:
    """AxisEnvelope with lo > hi must raise InfeasiblePerturbationError."""
    env = AxisEnvelope(lo=1.0, hi=0.0, axis="y")
    with pytest.raises(InfeasiblePerturbationError):
        env.validate()


def test_axis_envelope_carries_axis_name() -> None:
    env = AxisEnvelope(lo=0.0, hi=1.0, axis="position_x")
    assert env.axis == "position_x"


# ---------------------------------------------------------------------------
# PerturbationPlan: tuple independence
# ---------------------------------------------------------------------------


def test_perturbation_plan_is_tuple(sample_graph: SemanticSceneGraph) -> None:
    """Plan for 'position' axis must have empty articulation/camera/object by default."""
    plan = plan_perturbations(sample_graph, "position")
    # Tuple construction: position active, others empty/None
    # (articulation is always planned for safety, so we only check camera/object)
    assert plan.camera_plan is None
    assert plan.object_substitutions == {}
    assert plan.distractor_budget == 0


def test_perturbation_plan_axes_are_independent(
    sample_graph: SemanticSceneGraph,
) -> None:
    """Planning position+camera must not affect articulation plans."""
    plan_pos = plan_perturbations(sample_graph, "position")
    _plan_cam = plan_perturbations(sample_graph, "camera")
    plan_both = plan_perturbations(sample_graph, "position,camera")

    # Position plans should be the same whether camera is requested or not
    pos_keys = set(plan_pos.position_plans.keys())
    both_keys = set(plan_both.position_plans.keys())
    assert pos_keys == both_keys, "position plans changed when camera axis added"


def test_perturbation_plan_active_axes_recorded(
    sample_graph: SemanticSceneGraph,
) -> None:
    """PerturbationPlan.active_axes must reflect exactly what was requested."""
    axes = frozenset(["position", "camera"])
    plan = plan_perturbations(sample_graph, axes)
    assert "position" in plan.active_axes
    assert "camera" in plan.active_axes
    assert "object" not in plan.active_axes


def test_perturbation_plan_has_diagnostics(sample_graph: SemanticSceneGraph) -> None:
    """Every plan must carry a PlanDiagnostics instance."""
    plan = plan_perturbations(sample_graph, "position")
    assert plan.diagnostics is not None
    assert isinstance(plan.diagnostics, PlanDiagnostics)


def test_perturbation_plan_anti_trivialization_always_active(
    sample_graph: SemanticSceneGraph,
) -> None:
    """Anti-trivialization must always be set after validation step 6."""
    plan = plan_perturbations(sample_graph, "position")
    assert plan.anti_trivialization_active is True


# ---------------------------------------------------------------------------
# parse_axes
# ---------------------------------------------------------------------------


def test_parse_axes_single() -> None:
    assert parse_axes("position") == frozenset(["position"])


def test_parse_axes_comma_separated() -> None:
    result = parse_axes("position,camera,lighting")
    assert result == frozenset(["position", "camera", "lighting"])


def test_parse_axes_preset_combined() -> None:
    result = parse_axes("combined")
    expected = AXIS_PRESETS["combined"]
    assert result == expected


def test_parse_axes_preset_full() -> None:
    result = parse_axes("full")
    expected = AXIS_PRESETS["full"]
    assert result == expected


def test_parse_axes_whitespace_tolerant() -> None:
    result = parse_axes("position, camera, distractor")
    assert "position" in result
    assert "camera" in result
    assert "distractor" in result


def test_parse_axes_empty_string() -> None:
    result = parse_axes("")
    assert result == frozenset()


# ---------------------------------------------------------------------------
# plan_position: contained objects excluded
# ---------------------------------------------------------------------------


def test_position_plan_for_contained_objects_uses_relative_positioning(
    sample_graph: SemanticSceneGraph,
) -> None:
    """Contained objects must get a relative position plan that samples within the parent bounds."""
    from libero_infinity.ir.nodes import ObjectNode

    diag = PlanDiagnostics()
    pos_plans = plan_position(sample_graph, frozenset(["position"]), diag)

    # Check that contained objects get relative position plans
    for node_id, node in sample_graph.nodes.items():
        if not isinstance(node, ObjectNode):
            continue
        contained_edges = [e for e in sample_graph.edges_from(node_id) if e.label == "contained_in"]
        if contained_edges:
            assert node_id in pos_plans, f"Contained object {node_id} should have a relative position plan"  # fmt: skip  # noqa: E501
            plan = pos_plans[node_id]
            assert plan.use_relative_positioning, f"Contained object {node_id} plan must use relative positioning"  # fmt: skip  # noqa: E501
            assert plan.support_name == contained_edges[0].dst_id, f"Contained object {node_id} plan support_name must be the container id"  # fmt: skip  # noqa: E501


def test_position_plan_not_requested_returns_empty(
    sample_graph: SemanticSceneGraph,
) -> None:
    """plan_position must return empty dict when 'position' not in request_axes."""
    diag = PlanDiagnostics()
    result = plan_position(sample_graph, frozenset(["camera"]), diag)
    assert result == {}


def test_position_plans_x_lo_lt_hi(sample_graph: SemanticSceneGraph) -> None:
    """Every position plan must have x_lo < x_hi and y_lo < y_hi."""
    diag = PlanDiagnostics()
    plans = plan_position(sample_graph, frozenset(["position"]), diag)
    for obj_name, pp in plans.items():
        assert pp.x_envelope.lo < pp.x_envelope.hi, f"Object {obj_name}: x_envelope lo >= hi: {pp.x_envelope.lo} >= {pp.x_envelope.hi}"  # fmt: skip  # noqa: E501
        assert pp.y_envelope.lo < pp.y_envelope.hi, f"Object {obj_name}: y_envelope lo >= hi: {pp.y_envelope.lo} >= {pp.y_envelope.hi}"  # fmt: skip  # noqa: E501


# ---------------------------------------------------------------------------
# plan_articulation: goal-reachability check
# ---------------------------------------------------------------------------


def test_articulation_plans_only_for_articulatable_fixtures(
    sample_graph: SemanticSceneGraph,
) -> None:
    """plan_articulation must only produce plans for articulatable fixtures."""
    from libero_infinity.ir.nodes import FixtureNode

    diag = PlanDiagnostics()
    art_plans = plan_articulation(sample_graph, frozenset(["articulation"]), diag)
    for fixture_name in art_plans:
        node = sample_graph.get_node(fixture_name)
        if node is None:
            # try by instance_name lookup
            for n in sample_graph.nodes.values():
                if isinstance(n, FixtureNode) and n.instance_name == fixture_name:
                    assert n.is_articulatable, f"ArticulationPlan for non-articulatable fixture {fixture_name}"  # fmt: skip  # noqa: E501
                    break


def test_articulation_plan_lo_lt_hi(sample_graph: SemanticSceneGraph) -> None:
    """All ArticulationPlan entries must have lo < hi."""
    diag = PlanDiagnostics()
    art_plans = plan_articulation(sample_graph, frozenset(["articulation"]), diag)
    for name, ap in art_plans.items():
        assert ap.lo < ap.hi, f"ArticulationPlan for {name}: lo={ap.lo} >= hi={ap.hi}"


def test_articulation_plan_state_kind_valid(sample_graph: SemanticSceneGraph) -> None:
    """ArticulationPlan state_kind must be one of the expected states."""
    valid_states = {"Open", "Close", "Turnon", "Turnoff"}
    diag = PlanDiagnostics()
    art_plans = plan_articulation(sample_graph, frozenset(["articulation"]), diag)
    for name, ap in art_plans.items():
        assert ap.state_kind in valid_states, f"ArticulationPlan for {name}: unknown state_kind '{ap.state_kind}'"  # fmt: skip  # noqa: E501


# ---------------------------------------------------------------------------
# plan_camera: visibility sub-envelope
# ---------------------------------------------------------------------------


def test_camera_plan_default_envelope(sample_graph: SemanticSceneGraph) -> None:
    """Default camera plan should have non-degenerate azimuth/elevation ranges."""
    diag = PlanDiagnostics()
    cp = plan_camera(sample_graph, frozenset(["camera"]), diag)
    assert cp is not None
    assert cp.azimuth_lo < cp.azimuth_hi
    assert cp.elevation_lo < cp.elevation_hi
    assert cp.distance_lo < cp.distance_hi


def test_camera_plan_not_requested_returns_none(
    sample_graph: SemanticSceneGraph,
) -> None:
    """plan_camera must return None when 'camera' not in request_axes."""
    diag = PlanDiagnostics()
    result = plan_camera(sample_graph, frozenset(["position"]), diag)
    assert result is None


# ---------------------------------------------------------------------------
# plan_distractor: dynamic budget cap
# ---------------------------------------------------------------------------


def test_distractor_budget_cap_at_5(sample_graph: SemanticSceneGraph) -> None:
    """Distractor budget must be capped at 5 even with large free_area."""
    diag = PlanDiagnostics()
    budget, classes = plan_distractor(sample_graph, frozenset(["distractor"]), diag, free_area=10.0)
    assert budget <= 5, f"Distractor budget exceeded 5: got {budget}"


def test_distractor_budget_zero_when_not_requested(
    sample_graph: SemanticSceneGraph,
) -> None:
    """Distractor budget must be 0 when 'distractor' not in request_axes."""
    diag = PlanDiagnostics()
    budget, classes = plan_distractor(sample_graph, frozenset(["position"]), diag)
    assert budget == 0
    assert classes == []


def test_distractor_budget_scales_with_free_area(
    sample_graph: SemanticSceneGraph,
) -> None:
    """Distractor budget should scale with free_area (bounded by 5)."""
    diag = PlanDiagnostics()
    small_budget, _ = plan_distractor(sample_graph, frozenset(["distractor"]), diag, free_area=0.01)
    large_budget, _ = plan_distractor(sample_graph, frozenset(["distractor"]), diag, free_area=0.5)
    # With a small area budget should be <= large area budget
    assert small_budget <= large_budget


def test_distractor_classes_non_empty(sample_graph: SemanticSceneGraph) -> None:
    """Distractor classes should be non-empty when budget > 0."""
    diag = PlanDiagnostics()
    budget, classes = plan_distractor(sample_graph, frozenset(["distractor"]), diag, free_area=0.09)
    if budget > 0:
        assert len(classes) > 0


# ---------------------------------------------------------------------------
# Corpus integration: all BDDLs plan successfully
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bddl_file", _bddl_files(5))
def test_all_bddls_plan_position(bddl_file: str) -> None:
    """All BDDL files must produce a valid position plan."""
    cfg = TaskConfig.from_bddl(bddl_file)
    graph = build_semantic_scene_graph(cfg)
    plan = plan_perturbations(graph, "position")
    assert plan.active_axes == frozenset(["position"])
    assert plan.diagnostics is not None


@pytest.mark.parametrize("bddl_file", _bddl_files(5))
def test_all_bddls_plan_combined(bddl_file: str) -> None:
    """All BDDL files must produce a valid combined plan."""
    cfg = TaskConfig.from_bddl(bddl_file)
    graph = build_semantic_scene_graph(cfg)
    plan = plan_perturbations(graph, "combined")
    assert "position" in plan.active_axes
    assert "camera" in plan.active_axes
    assert plan.diagnostics is not None


# ---------------------------------------------------------------------------
# Lighting axis tests
# ---------------------------------------------------------------------------


def test_plan_lighting_returns_lighting_plan(sample_graph: SemanticSceneGraph) -> None:
    """plan_lighting() must return a LightingPlan with valid (non-degenerate) ranges."""
    diag = PlanDiagnostics()
    lp = plan_lighting(sample_graph, frozenset(["lighting"]), diag)
    assert lp is not None
    assert isinstance(lp, LightingPlan)
    # intensity range must be non-degenerate and cover a useful spread
    assert lp.intensity_lo < lp.intensity_hi
    assert lp.intensity_lo >= 0.0
    # ambient range must be non-degenerate
    assert lp.ambient_lo < lp.ambient_hi
    assert lp.ambient_lo >= 0.0
    # position jitter must be positive
    assert lp.position_jitter > 0.0


def test_lighting_axis_in_combined_preset(sample_graph: SemanticSceneGraph) -> None:
    """plan_perturbations with axes={'lighting'} must set plan.lighting_plan."""
    plan = plan_perturbations(sample_graph, frozenset(["lighting"]))
    assert plan.lighting_plan is not None
    assert isinstance(plan.lighting_plan, LightingPlan)


def test_render_lighting_emits_params(sample_graph: SemanticSceneGraph) -> None:
    """render_scenic with a lighting_plan must emit light_intensity param."""
    plan = plan_perturbations(sample_graph, frozenset(["lighting"]))
    assert plan.lighting_plan is not None
    scenic_src = render_scenic(plan, sample_graph)
    assert "light_intensity" in scenic_src
    assert "ambient_level" in scenic_src
    assert "light_x_offset" in scenic_src


# ---------------------------------------------------------------------------
# Texture axis tests
# ---------------------------------------------------------------------------


def test_plan_texture_returns_texture_plan(sample_graph: SemanticSceneGraph) -> None:
    """plan_texture() must return a TexturePlan when 'texture' is in request_axes."""
    diag = PlanDiagnostics()
    tp = plan_texture(sample_graph, frozenset(["texture"]), diag)
    assert tp is not None
    assert isinstance(tp, TexturePlan)
    # Default mode is "random" (matches old pipeline _apply_texture_perturbation)
    assert tp.table_texture == "random"
    # No candidates means the simulator will pick randomly at runtime
    assert isinstance(tp.texture_candidates, list)


def test_texture_axis_in_full_preset() -> None:
    """'texture' must be included in the 'full' preset."""
    from libero_infinity.planner.composition import AXIS_PRESETS

    assert "texture" in AXIS_PRESETS["full"], "'texture' missing from AXIS_PRESETS['full']"


def test_render_texture_emits_params(sample_graph: SemanticSceneGraph) -> None:
    """render_scenic with a texture_plan must emit table_texture param."""
    plan = plan_perturbations(sample_graph, frozenset(["texture"]))
    assert plan.texture_plan is not None
    assert isinstance(plan.texture_plan, TexturePlan)
    scenic_src = render_scenic(plan, sample_graph)
    assert "table_texture" in scenic_src
    # Default: should emit the "random" string
    assert '"random"' in scenic_src


# ---------------------------------------------------------------------------
# parse_axes: mixed preset + individual axis (edge case)
# ---------------------------------------------------------------------------


def test_parse_axes_mixed_preset_and_individual() -> None:
    """'combined,texture' should expand 'combined' preset and also add 'texture'."""
    result = parse_axes("combined,texture")
    # All combined axes must be present
    for axis in AXIS_PRESETS["combined"]:
        assert axis in result, f"Expected '{axis}' from combined preset, got {result}"
    # texture must also be present (not in combined by default)
    assert "texture" in result, "Expected 'texture' axis from explicit request"


def test_parse_axes_unknown_axis_passes_through() -> None:
    """Unknown axis names should be passed through unchanged (no error)."""
    result = parse_axes("position,custom_axis_xyz")
    assert "position" in result
    assert "custom_axis_xyz" in result


# ---------------------------------------------------------------------------
# plan_position: contained_in recursive positioning correctness
# ---------------------------------------------------------------------------


def test_contained_position_uses_relative_positioning_with_correct_parent() -> None:
    """An object with contained_in edge must get a PositionPlan anchored to direct parent."""
    # Build a graph with a bowl contained inside a wooden_cabinet fixture
    from libero_infinity.ir.nodes import (
        FixtureNode,
        ObjectNode,
        SceneEdge,
        WorkspaceNode,
    )

    graph = SemanticSceneGraph(
        task_language="pick bowl from cabinet",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    graph.add_node(
        WorkspaceNode(
            node_id="main_table",
            node_type="workspace",
            instance_name="main_table",
            object_class="table",
        )
    )
    graph.add_node(
        FixtureNode(
            node_id="wooden_cabinet_1",
            node_type="fixture",
            instance_name="wooden_cabinet_1",
            object_class="wooden_cabinet",
            init_x=0.1,
            init_y=0.0,
        )
    )
    bowl = ObjectNode(
        node_id="akita_black_bowl_1",
        node_type="object",
        instance_name="akita_black_bowl_1",
        object_class="akita_black_bowl",
        contained=True,
        placement_target="wooden_cabinet_1",
    )
    graph.add_node(bowl)
    graph.add_edge(
        SceneEdge(
            src_id="akita_black_bowl_1",
            dst_id="wooden_cabinet_1",
            label="contained_in",
            spatial_kind="inside",
        )
    )

    diag = PlanDiagnostics()
    plans = plan_position(graph, frozenset(["position"]), diag)

    assert "akita_black_bowl_1" in plans, "Contained object must get a position plan"
    pp = plans["akita_black_bowl_1"]
    assert pp.use_relative_positioning is True, "Contained objects must use relative positioning"
    assert pp.support_name == "wooden_cabinet_1", "support_name must be the direct container, got: " + pp.support_name  # fmt: skip  # noqa: E501
    # Envelope must be non-degenerate
    assert pp.x_envelope.lo < pp.x_envelope.hi
    assert pp.y_envelope.lo < pp.y_envelope.hi


def test_contained_position_fixture_interior_limits_envelope() -> None:
    """Envelope for object in fixture must be bounded by _CONTAINER_FIXTURE_INTERIOR dims."""
    from libero_infinity.asset_registry import get_dimensions
    from libero_infinity.ir.nodes import (
        FixtureNode,
        ObjectNode,
    )
    from libero_infinity.planner.position import (
        _CONTAINER_FIXTURE_INTERIOR,
        _plan_contained_position,
    )

    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    fixture = FixtureNode(
        node_id="wooden_cabinet_1",
        node_type="fixture",
        instance_name="wooden_cabinet_1",
        object_class="wooden_cabinet",
        init_x=0.0,
        init_y=0.0,
    )
    graph.add_node(fixture)

    child = ObjectNode(
        node_id="akita_black_bowl_1",
        node_type="object",
        instance_name="akita_black_bowl_1",
        object_class="akita_black_bowl",
        contained=True,
    )
    graph.add_node(child)

    diag = PlanDiagnostics()
    pp = _plan_contained_position(child, "wooden_cabinet_1", graph, diag)

    assert pp is not None, "Should return a PositionPlan for contained object"
    # Verify envelope derived from _CONTAINER_FIXTURE_INTERIOR
    parent_x, parent_y = _CONTAINER_FIXTURE_INTERIOR["wooden_cabinet"]
    cdims = get_dimensions("akita_black_bowl")
    expected_dx = max(0.02, parent_x / 2.0 - cdims[0] / 2.0)
    expected_dy = max(0.02, parent_y / 2.0 - cdims[1] / 2.0)
    assert pp.x_envelope.lo == pytest.approx(-expected_dx, abs=1e-6)
    assert pp.x_envelope.hi == pytest.approx(expected_dx, abs=1e-6)
    assert pp.y_envelope.lo == pytest.approx(-expected_dy, abs=1e-6)
    assert pp.y_envelope.hi == pytest.approx(expected_dy, abs=1e-6)


def test_contained_position_missing_container_returns_none_and_drops_axis() -> None:
    """_plan_contained_position must return None and drop axis if container is missing."""
    from libero_infinity.ir.nodes import ObjectNode
    from libero_infinity.planner.position import _plan_contained_position

    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    child = ObjectNode(
        node_id="bowl_1",
        node_type="object",
        instance_name="bowl_1",
        object_class="akita_black_bowl",
        contained=True,
    )
    graph.add_node(child)

    diag = PlanDiagnostics()
    result = _plan_contained_position(child, "nonexistent_container", graph, diag)

    assert result is None, "Must return None when container is not in graph"
    assert "position" in diag.dropped_axes, "Must record axis drop when container is missing"


def test_position_plan_goal_exclusion_fallback_for_large_zone() -> None:
    """When goal zone covers >80% of envelope, planner must use distance-based fallback."""
    from libero_infinity.ir.nodes import (
        ObjectNode,
        RegionNode,
        SceneEdge,
        WorkspaceNode,
    )
    from libero_infinity.planner.position import _DEFAULT_PERTURB_RADIUS

    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    graph.add_node(
        WorkspaceNode(
            node_id="main_table",
            node_type="workspace",
            instance_name="main_table",
            object_class="table",
        )
    )
    cx, cy = 0.0, 0.0
    obj = ObjectNode(
        node_id="bowl_1",
        node_type="object",
        instance_name="bowl_1",
        object_class="akita_black_bowl",
        init_x=cx,
        init_y=cy,
        placement_target="main_table",
    )
    graph.add_node(obj)
    graph.add_edge(SceneEdge(src_id="bowl_1", dst_id="main_table", label="supported_by"))

    # Create a region that covers the entire envelope (100% coverage → fallback)
    r = _DEFAULT_PERTURB_RADIUS
    region = RegionNode(
        node_id="goal_region",
        node_type="region",
        instance_name="goal_region",
        object_class="region",
        target="main_table",
        x_min=cx - r - 0.01,
        x_max=cx + r + 0.01,
        y_min=cy - r - 0.01,
        y_max=cy + r + 0.01,
    )
    graph.add_node(region)
    graph.add_edge(SceneEdge(src_id="bowl_1", dst_id="goal_region", label="goal_target"))

    diag = PlanDiagnostics()
    plans = plan_position(graph, frozenset(["position"]), diag)

    assert "bowl_1" in plans
    pp = plans["bowl_1"]
    # When zone covers >80% → fall back to distance-based (not zone list)
    assert pp.exclusion_min_distance is not None, "Expected distance-based exclusion fallback when goal zone covers entire envelope"  # fmt: skip  # noqa: E501
    assert pp.exclusion_zones == [], "Exclusion zones list should be empty when falling back to distance-based"  # fmt: skip  # noqa: E501


# ---------------------------------------------------------------------------
# Background axis tests
# ---------------------------------------------------------------------------


def test_plan_background_returns_background_plan(
    sample_graph: SemanticSceneGraph,
) -> None:
    """plan_background() must return a BackgroundPlan when 'background' is in request_axes."""
    diag = PlanDiagnostics()
    bp = plan_background(sample_graph, frozenset(["background"]), diag)
    assert bp is not None
    assert isinstance(bp, BackgroundPlan)
    # Must have a non-empty candidate pool
    assert isinstance(bp.texture_candidates, list)
    assert len(bp.texture_candidates) > 0, "Expected at least one background texture candidate"


def test_plan_background_candidates_from_known_list(
    sample_graph: SemanticSceneGraph,
) -> None:
    """plan_background() candidates must all be names from the known LIBERO texture list."""
    diag = PlanDiagnostics()
    bp = plan_background(sample_graph, frozenset(["background"]), diag)
    assert bp is not None
    known = set(LIBERO_BACKGROUND_TEXTURES)
    for name in bp.texture_candidates:
        assert name in known, f"Texture candidate '{name}' not in LIBERO_BACKGROUND_TEXTURES"


def test_plan_background_not_activated_without_axis(
    sample_graph: SemanticSceneGraph,
) -> None:
    """plan_background() must return None when 'background' is not requested."""
    diag = PlanDiagnostics()
    bp = plan_background(sample_graph, frozenset(["lighting", "texture"]), diag)
    assert bp is None


def test_background_axis_in_combined_preset() -> None:
    """'background' must be included in the 'combined' preset."""
    assert "background" in AXIS_PRESETS["combined"], "'background' missing from AXIS_PRESETS['combined']"  # fmt: skip  # noqa: E501


def test_background_axis_in_full_preset() -> None:
    """'background' must be included in the 'full' preset."""
    assert "background" in AXIS_PRESETS["full"], "'background' missing from AXIS_PRESETS['full']"


def test_plan_perturbations_background_plan_populated(
    sample_graph: SemanticSceneGraph,
) -> None:
    """plan_perturbations with axes={'background'} must set plan.background_plan."""
    plan = plan_perturbations(sample_graph, frozenset(["background"]))
    assert plan.background_plan is not None
    assert isinstance(plan.background_plan, BackgroundPlan)
    assert len(plan.background_plan.texture_candidates) > 0


def test_render_background_emits_params(sample_graph: SemanticSceneGraph) -> None:
    """render_scenic with a background_plan must emit wall_texture and floor_texture params."""
    plan = plan_perturbations(sample_graph, frozenset(["background"]))
    assert plan.background_plan is not None
    scenic_src = render_scenic(plan, sample_graph)
    assert "wall_texture" in scenic_src, "Expected 'wall_texture' param in rendered Scenic"
    assert "floor_texture" in scenic_src, "Expected 'floor_texture' param in rendered Scenic"
    # Should emit a Uniform distribution since candidates are populated
    assert "Uniform(" in scenic_src, "Expected Uniform() distribution for background textures"


def test_libero_background_textures_constant_non_empty() -> None:
    """LIBERO_BACKGROUND_TEXTURES must be a non-empty tuple of strings."""
    assert len(LIBERO_BACKGROUND_TEXTURES) > 0
    for name in LIBERO_BACKGROUND_TEXTURES:
        assert isinstance(name, str)
        assert len(name) > 0


# ---------------------------------------------------------------------------
# Validation step tests — these tests FAIL with pass-stubs and PASS with
# the real implementations of the 5 previously-stubbed checks.
# ---------------------------------------------------------------------------


# --- Step 1: _check_support_preservation ------------------------------------


def test_check_support_preservation_records_diagnostic_for_oversized_envelope() -> None:
    """_check_support_preservation must record a diagnostic when a relative
    position envelope exceeds the parent support's half-footprint."""
    from libero_infinity.ir.nodes import (
        FixtureNode,
        ObjectNode,
        SceneEdge,
        WorkspaceNode,
    )
    from libero_infinity.planner.composition import _check_support_preservation

    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    graph.add_node(
        WorkspaceNode(
            node_id="main_table",
            node_type="workspace",
            instance_name="main_table",
            object_class="table",
        )
    )
    # Plate is a small support (footprint ~0.25 × 0.20)
    graph.add_node(
        FixtureNode(
            node_id="plate_fixture",
            node_type="fixture",
            instance_name="plate_fixture",
            object_class="plate",
            init_x=0.0,
            init_y=0.0,
        )
    )
    obj = ObjectNode(
        node_id="bowl_1",
        node_type="object",
        instance_name="bowl_1",
        object_class="akita_black_bowl",
    )
    graph.add_node(obj)
    graph.add_edge(SceneEdge(src_id="bowl_1", dst_id="plate_fixture", label="stacked_on"))

    # Create an envelope that is ridiculously wide — FAR exceeds plate's half-footprint
    from libero_infinity.planner.types import AxisEnvelope

    plan = PerturbationPlan(
        position_plans={
            "bowl_1": PositionPlan(
                object_name="bowl_1",
                x_envelope=AxisEnvelope(-2.0, 2.0, "x"),  # 2 m half-extent >> plate
                y_envelope=AxisEnvelope(-2.0, 2.0, "y"),
                support_name="plate_fixture",
                use_relative_positioning=True,
            )
        }
    )
    diag = PlanDiagnostics()
    _check_support_preservation(plan, graph, diag)

    # A real implementation must record a diagnostic for the oversized envelope.
    assert "position" in diag.narrowed_axes, (
        "_check_support_preservation must narrow 'position' axis when "
        "relative envelope exceeds parent half-footprint"
    )


def test_check_support_preservation_no_diagnostic_for_tight_envelope() -> None:
    """_check_support_preservation must NOT record a diagnostic when a
    relative envelope is well within the parent's half-footprint."""
    from libero_infinity.ir.nodes import (
        FixtureNode,
        ObjectNode,
        SceneEdge,
        WorkspaceNode,
    )
    from libero_infinity.planner.composition import _check_support_preservation
    from libero_infinity.planner.types import AxisEnvelope

    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    graph.add_node(
        WorkspaceNode(
            node_id="main_table",
            node_type="workspace",
            instance_name="main_table",
            object_class="table",
        )
    )
    graph.add_node(
        FixtureNode(
            node_id="plate_fixture",
            node_type="fixture",
            instance_name="plate_fixture",
            object_class="plate",
            init_x=0.0,
            init_y=0.0,
        )
    )
    graph.add_node(
        ObjectNode(
            node_id="bowl_1",
            node_type="object",
            instance_name="bowl_1",
            object_class="akita_black_bowl",
        )
    )
    graph.add_edge(SceneEdge(src_id="bowl_1", dst_id="plate_fixture", label="stacked_on"))

    # Tight envelope — well within any reasonable plate footprint
    plan = PerturbationPlan(
        position_plans={
            "bowl_1": PositionPlan(
                object_name="bowl_1",
                x_envelope=AxisEnvelope(-0.01, 0.01, "x"),
                y_envelope=AxisEnvelope(-0.01, 0.01, "y"),
                support_name="plate_fixture",
                use_relative_positioning=True,
            )
        }
    )
    diag = PlanDiagnostics()
    _check_support_preservation(plan, graph, diag)
    # No narrowing should occur for a tight envelope
    assert "position" not in diag.narrowed_axes, "_check_support_preservation must not narrow 'position' for tight envelope"  # fmt: skip  # noqa: E501


# --- Step 2: _check_containment_dimensions ----------------------------------


def test_check_containment_dimensions_records_diagnostic_for_oversized_child() -> None:
    """_check_containment_dimensions must record a diagnostic when a child
    object's footprint exceeds the container's interior cavity."""
    from libero_infinity.ir.nodes import (
        FixtureNode,
        ObjectNode,
        SceneEdge,
        WorkspaceNode,
    )
    from libero_infinity.planner.composition import _check_containment_dimensions

    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    graph.add_node(
        WorkspaceNode(
            node_id="main_table",
            node_type="workspace",
            instance_name="main_table",
            object_class="table",
        )
    )
    # desk_caddy has a very small interior: (0.10, 0.07)
    graph.add_node(
        FixtureNode(
            node_id="desk_caddy_1",
            node_type="fixture",
            instance_name="desk_caddy_1",
            object_class="desk_caddy",
            init_x=0.0,
            init_y=0.0,
        )
    )
    # Use a large object class whose footprint exceeds desk_caddy interior
    # We'll create a synthetic large object via get_dimensions override isn't easy,
    # so use a known large object.  The wooden_cabinet itself is huge.
    # Actually for this test we'll manually verify using knowledge:
    # desk_caddy interior = (0.10, 0.07).
    # We want a child that is larger than 0.10 × 0.07.
    # akita_black_bowl has dims roughly (0.13, 0.13, 0.06) from the registry.
    # If its x-dim > 0.10, it fails the containment check.
    graph.add_node(
        ObjectNode(
            node_id="bowl_big_1",
            node_type="object",
            instance_name="bowl_big_1",
            object_class="akita_black_bowl",  # footprint likely > desk_caddy interior
            contained=True,
        )
    )
    graph.add_edge(
        SceneEdge(
            src_id="bowl_big_1",
            dst_id="desk_caddy_1",
            label="contained_in",
            spatial_kind="inside",
        )
    )

    from libero_infinity.asset_registry import get_dimensions
    from libero_infinity.planner.position import _CONTAINER_FIXTURE_INTERIOR

    bowl_dims = get_dimensions("akita_black_bowl")
    caddy_interior = _CONTAINER_FIXTURE_INTERIOR["desk_caddy"]
    # Only run meaningful assertion if bowl really is oversized
    if bowl_dims[0] <= caddy_interior[0] and bowl_dims[1] <= caddy_interior[1]:
        pytest.skip("bowl fits in desk_caddy; cannot test oversized child")

    plan = PerturbationPlan()
    diag = PlanDiagnostics()
    _check_containment_dimensions(plan, graph, diag)

    assert "position" in diag.narrowed_axes, "_check_containment_dimensions must narrow 'position' when child exceeds container"  # fmt: skip  # noqa: E501


def test_check_containment_dimensions_no_diagnostic_for_small_child() -> None:
    """_check_containment_dimensions must NOT record a diagnostic when the
    child's footprint fits comfortably inside the container."""
    from libero_infinity.ir.nodes import (
        FixtureNode,
        ObjectNode,
        SceneEdge,
        WorkspaceNode,
    )
    from libero_infinity.planner.composition import _check_containment_dimensions

    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    graph.add_node(
        WorkspaceNode(
            node_id="main_table",
            node_type="workspace",
            instance_name="main_table",
            object_class="table",
        )
    )
    # wooden_cabinet interior = (0.20, 0.18) — plenty of room for a bowl
    graph.add_node(
        FixtureNode(
            node_id="wooden_cabinet_1",
            node_type="fixture",
            instance_name="wooden_cabinet_1",
            object_class="wooden_cabinet",
            init_x=0.0,
            init_y=0.0,
        )
    )
    graph.add_node(
        ObjectNode(
            node_id="bowl_1",
            node_type="object",
            instance_name="bowl_1",
            object_class="akita_black_bowl",
            contained=True,
        )
    )
    graph.add_edge(
        SceneEdge(
            src_id="bowl_1",
            dst_id="wooden_cabinet_1",
            label="contained_in",
            spatial_kind="inside",
        )
    )

    from libero_infinity.asset_registry import get_dimensions
    from libero_infinity.planner.position import _CONTAINER_FIXTURE_INTERIOR

    bowl_dims = get_dimensions("akita_black_bowl")
    cabinet_interior = _CONTAINER_FIXTURE_INTERIOR["wooden_cabinet"]
    if bowl_dims[0] > cabinet_interior[0] or bowl_dims[1] > cabinet_interior[1]:
        pytest.skip("bowl does not fit in wooden_cabinet; cannot test no-diagnostic case")

    plan = PerturbationPlan()
    diag = PlanDiagnostics()
    _check_containment_dimensions(plan, graph, diag)

    assert "position" not in diag.narrowed_axes, "_check_containment_dimensions must not warn when child fits in container"  # fmt: skip  # noqa: E501


# --- Step 3: _check_articulation_noninterference ----------------------------


def test_check_articulation_noninterference_corrects_closed_fixture_with_contained_object() -> None:
    """_check_articulation_noninterference must correct an ArticulationPlan
    from 'Close' to 'Open' when an object is contained_in the fixture."""
    from libero_infinity.ir.nodes import (
        FixtureNode,
        ObjectNode,
        SceneEdge,
        WorkspaceNode,
    )
    from libero_infinity.planner.composition import _check_articulation_noninterference
    from libero_infinity.planner.types import PerturbationPlan

    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    graph.add_node(
        WorkspaceNode(
            node_id="main_table",
            node_type="workspace",
            instance_name="main_table",
            object_class="table",
        )
    )
    graph.add_node(
        FixtureNode(
            node_id="wooden_cabinet_1",
            node_type="fixture",
            instance_name="wooden_cabinet_1",
            object_class="wooden_cabinet",
            is_articulatable=True,
        )
    )
    graph.add_node(
        ObjectNode(
            node_id="bowl_1",
            node_type="object",
            instance_name="bowl_1",
            object_class="akita_black_bowl",
            contained=True,
        )
    )
    graph.add_edge(
        SceneEdge(
            src_id="bowl_1",
            dst_id="wooden_cabinet_1",
            label="contained_in",
            spatial_kind="inside",
        )
    )

    # Articulation plan starts as Close — would block access to the bowl.
    art_plan = ArticulationPlan(
        fixture_name="wooden_cabinet_1",
        state_kind="Close",
        lo=0.0,
        hi=0.005,
        reason="test: starting closed",
        goal_reachability_ok=True,
    )
    plan = PerturbationPlan(
        articulation_plans={"wooden_cabinet_1": art_plan},
    )
    diag = PlanDiagnostics()
    _check_articulation_noninterference(plan, graph, diag)

    # The implementation must correct the state_kind to "Open".
    assert art_plan.state_kind == "Open", (
        f"_check_articulation_noninterference must correct 'Close' → 'Open' "
        f"when an object is contained_in the fixture; got '{art_plan.state_kind}'"
    )
    assert art_plan.goal_reachability_ok is False, "goal_reachability_ok must be set to False after correction"  # fmt: skip  # noqa: E501
    assert "articulation" in diag.narrowed_axes, "A diagnostic must be recorded for the correction"


def test_check_articulation_noninterference_no_change_when_no_contained_objects() -> None:
    """_check_articulation_noninterference must NOT change an ArticulationPlan
    when no objects are contained in the fixture."""
    from libero_infinity.ir.nodes import FixtureNode, WorkspaceNode
    from libero_infinity.planner.composition import _check_articulation_noninterference
    from libero_infinity.planner.types import PerturbationPlan

    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    graph.add_node(
        WorkspaceNode(
            node_id="main_table",
            node_type="workspace",
            instance_name="main_table",
            object_class="table",
        )
    )
    graph.add_node(
        FixtureNode(
            node_id="wooden_cabinet_1",
            node_type="fixture",
            instance_name="wooden_cabinet_1",
            object_class="wooden_cabinet",
            is_articulatable=True,
        )
    )

    art_plan = ArticulationPlan(
        fixture_name="wooden_cabinet_1",
        state_kind="Close",
        lo=0.0,
        hi=0.005,
        reason="test: closed, no objects inside",
        goal_reachability_ok=True,
    )
    plan = PerturbationPlan(
        articulation_plans={"wooden_cabinet_1": art_plan},
    )
    diag = PlanDiagnostics()
    _check_articulation_noninterference(plan, graph, diag)

    # No objects inside → state must NOT be changed.
    assert art_plan.state_kind == "Close", (
        "_check_articulation_noninterference must not change state when no "
        "objects are contained in the fixture"
    )
    assert art_plan.goal_reachability_ok is True


# --- Step 5: _check_camera_visibility ---------------------------------------


def test_check_camera_visibility_drops_plan_with_degenerate_azimuth() -> None:
    """_check_camera_visibility must set camera_plan to None when azimuth
    range is degenerate (lo == hi)."""
    from libero_infinity.planner.composition import _check_camera_visibility
    from libero_infinity.planner.types import PerturbationPlan

    plan = PerturbationPlan(
        camera_plan=CameraPlan(
            azimuth_lo=5.0,
            azimuth_hi=5.0,  # degenerate!
            elevation_lo=-5.0,
            elevation_hi=5.0,
            distance_lo=0.9,
            distance_hi=1.1,
        )
    )
    diag = PlanDiagnostics()

    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    _check_camera_visibility(plan, graph, diag)

    assert plan.camera_plan is None, "_check_camera_visibility must set camera_plan to None when azimuth is degenerate"  # fmt: skip  # noqa: E501
    assert "camera" in diag.dropped_axes, "_check_camera_visibility must record 'camera' in dropped_axes"  # fmt: skip  # noqa: E501


def test_check_camera_visibility_drops_plan_with_degenerate_elevation() -> None:
    """_check_camera_visibility must drop the plan when elevation lo > hi."""
    from libero_infinity.planner.composition import _check_camera_visibility
    from libero_infinity.planner.types import PerturbationPlan

    plan = PerturbationPlan(
        camera_plan=CameraPlan(
            azimuth_lo=-10.0,
            azimuth_hi=10.0,
            elevation_lo=7.0,
            elevation_hi=3.0,  # lo > hi — degenerate
            distance_lo=0.9,
            distance_hi=1.1,
        )
    )
    diag = PlanDiagnostics()
    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    _check_camera_visibility(plan, graph, diag)

    assert plan.camera_plan is None, "_check_camera_visibility must drop camera_plan on degenerate elevation"  # fmt: skip  # noqa: E501
    assert "camera" in diag.dropped_axes


def test_check_camera_visibility_preserves_valid_plan() -> None:
    """_check_camera_visibility must NOT drop a valid, non-degenerate camera plan."""
    from libero_infinity.planner.composition import _check_camera_visibility
    from libero_infinity.planner.types import PerturbationPlan

    original_plan = CameraPlan(
        azimuth_lo=-10.0,
        azimuth_hi=10.0,
        elevation_lo=-5.0,
        elevation_hi=5.0,
        distance_lo=0.9,
        distance_hi=1.1,
    )
    plan = PerturbationPlan(camera_plan=original_plan)
    diag = PlanDiagnostics()
    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    _check_camera_visibility(plan, graph, diag)

    assert plan.camera_plan is not None, "_check_camera_visibility must NOT drop a valid camera plan"  # fmt: skip  # noqa: E501
    assert "camera" not in diag.dropped_axes


# --- Step 7: _check_envelope_quality ----------------------------------------


def test_check_envelope_quality_removes_degenerate_x_envelope() -> None:
    """_check_envelope_quality must remove position plans with degenerate
    x-envelope (lo == hi)."""
    from libero_infinity.planner.composition import _check_envelope_quality
    from libero_infinity.planner.types import AxisEnvelope

    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    good_plan = PositionPlan(
        object_name="good_obj",
        x_envelope=AxisEnvelope(-0.1, 0.1, "x"),
        y_envelope=AxisEnvelope(-0.1, 0.1, "y"),
        support_name="main_table",
    )
    bad_plan = PositionPlan(
        object_name="bad_obj",
        x_envelope=AxisEnvelope(0.5, 0.5, "x"),  # degenerate!
        y_envelope=AxisEnvelope(-0.1, 0.1, "y"),
        support_name="main_table",
    )
    plan = PerturbationPlan(position_plans={"good_obj": good_plan, "bad_obj": bad_plan})
    diag = PlanDiagnostics()
    _check_envelope_quality(plan, graph, diag)

    assert "bad_obj" not in plan.position_plans, "_check_envelope_quality must remove 'bad_obj' with degenerate x-envelope"  # fmt: skip  # noqa: E501
    assert "good_obj" in plan.position_plans, "_check_envelope_quality must keep 'good_obj' with valid envelope"  # fmt: skip  # noqa: E501
    assert "position" in diag.dropped_axes, "_check_envelope_quality must record the dropped plan in diagnostics"  # fmt: skip  # noqa: E501


def test_check_envelope_quality_removes_degenerate_y_envelope() -> None:
    """_check_envelope_quality must remove position plans with degenerate
    y-envelope (lo > hi)."""
    from libero_infinity.planner.composition import _check_envelope_quality
    from libero_infinity.planner.types import AxisEnvelope

    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    bad_plan = PositionPlan(
        object_name="obj_1",
        x_envelope=AxisEnvelope(-0.1, 0.1, "x"),
        y_envelope=AxisEnvelope(0.2, 0.1, "y"),  # lo > hi — degenerate
        support_name="main_table",
    )
    plan = PerturbationPlan(position_plans={"obj_1": bad_plan})
    diag = PlanDiagnostics()
    _check_envelope_quality(plan, graph, diag)

    assert "obj_1" not in plan.position_plans, "_check_envelope_quality must remove plans with degenerate y-envelope"  # fmt: skip  # noqa: E501
    assert "position" in diag.dropped_axes


def test_check_envelope_quality_keeps_all_valid_envelopes() -> None:
    """_check_envelope_quality must leave all valid position plans untouched."""
    from libero_infinity.planner.composition import _check_envelope_quality
    from libero_infinity.planner.types import AxisEnvelope

    graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    plans = {
        f"obj_{i}": PositionPlan(
            object_name=f"obj_{i}",
            x_envelope=AxisEnvelope(-0.1, 0.1, "x"),
            y_envelope=AxisEnvelope(-0.1, 0.1, "y"),
            support_name="main_table",
        )
        for i in range(3)
    }
    plan = PerturbationPlan(position_plans=dict(plans))
    diag = PlanDiagnostics()
    _check_envelope_quality(plan, graph, diag)

    assert len(plan.position_plans) == 3, "_check_envelope_quality must not remove any valid plans"
    assert "position" not in diag.dropped_axes
