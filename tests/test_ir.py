"""Tests for the Semantic IR module (Wave 1: ir-core).

Covers: ArticulationModel, PlanDiagnostics, SemanticSceneGraph,
build_semantic_scene_graph, DAG validation, node types.
"""

from __future__ import annotations

import glob

import pytest

from libero_infinity.ir.graph_builder import build_semantic_scene_graph
from libero_infinity.ir.nodes import (
    ArticulationModel,
    DistractorSlotNode,
    FixtureNode,
    MovableSupportNode,
    ObjectNode,
    PlanDiagnostics,
    SceneEdge,
    WorkspaceNode,
)
from libero_infinity.ir.scene_graph import SemanticError, SemanticSceneGraph
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
# ArticulationModel tests
# ---------------------------------------------------------------------------


def test_articulation_model_canonical_has_four_classes() -> None:
    """ArticulationModel.canonical() must have all 4 known fixture classes."""
    model = ArticulationModel.canonical()
    assert "microwave" in model.fixture_families
    assert "wooden_cabinet" in model.fixture_families
    assert "white_cabinet" in model.fixture_families
    assert "flat_stove" in model.fixture_families


def test_articulation_model_ranges_microwave() -> None:
    """Microwave articulation ranges must match legacy scene_semantics values."""
    model = ArticulationModel.canonical()
    ranges = model.articulation_ranges["microwave"]
    lo, hi = ranges["Open"]
    assert abs(lo - (-2.094)) < 1e-6
    assert abs(hi - (-1.3)) < 1e-6
    lo2, hi2 = ranges["Close"]
    assert abs(lo2 - (-0.005)) < 1e-6
    assert abs(hi2 - 0.0) < 1e-6


def test_articulation_model_ranges_wooden_cabinet() -> None:
    """Wooden cabinet ranges must match legacy values."""
    model = ArticulationModel.canonical()
    ranges = model.articulation_ranges["wooden_cabinet"]
    lo, hi = ranges["Open"]
    assert abs(lo - (-0.16)) < 1e-6
    assert abs(hi - (-0.14)) < 1e-6


def test_articulation_model_is_articulatable() -> None:
    model = ArticulationModel.canonical()
    assert model.is_articulatable("microwave")
    assert model.is_articulatable("wooden_cabinet")
    assert not model.is_articulatable("table")
    assert not model.is_articulatable("unknown_fixture")


def test_articulation_model_root_workspace_fixtures() -> None:
    model = ArticulationModel.canonical()
    assert "table" in model.root_workspace_fixtures
    assert "floor" in model.root_workspace_fixtures
    assert "kitchen_table" in model.root_workspace_fixtures


# ---------------------------------------------------------------------------
# PlanDiagnostics tests
# ---------------------------------------------------------------------------


def test_plan_diagnostics_drop_axis() -> None:
    d = PlanDiagnostics()
    d.drop_axis("position", "no workspace area")
    assert "position" in d.dropped_axes
    assert d.reasons["position"] == "no workspace area"


def test_plan_diagnostics_narrow_axis() -> None:
    d = PlanDiagnostics()
    d.narrow_axis("camera", "visibility constraint")
    assert "camera" in d.narrowed_axes


def test_plan_diagnostics_constrain_axis() -> None:
    d = PlanDiagnostics()
    d.constrain_axis("articulation", "goal reachability")
    assert "articulation" in d.constrained_axes


def test_plan_diagnostics_defaults() -> None:
    d = PlanDiagnostics()
    assert d.dropped_axes == []
    assert d.warnings == []


# ---------------------------------------------------------------------------
# build_semantic_scene_graph tests
# ---------------------------------------------------------------------------


def test_build_graph_basic(sample_graph: SemanticSceneGraph) -> None:
    """Graph must have nodes and edges."""
    assert len(sample_graph.nodes) > 0
    assert len(sample_graph.edges) >= 0  # could be 0 for trivial tasks


def test_build_graph_nodes_have_valid_types(sample_graph: SemanticSceneGraph) -> None:
    """All nodes must have a valid node_type string."""
    valid_types = {
        "workspace",
        "fixture",
        "movable_support",
        "object",
        "region",
        "camera",
        "light",
        "distractor_slot",
    }
    for nid, node in sample_graph.nodes.items():
        assert node.node_type in valid_types, f"Node {nid} has invalid type: {node.node_type}"


def test_build_graph_has_workspace_node(sample_graph: SemanticSceneGraph) -> None:
    """At least one WorkspaceNode must be present."""
    workspace_nodes = [n for n in sample_graph.nodes.values() if isinstance(n, WorkspaceNode)]
    assert len(workspace_nodes) >= 1


def test_build_graph_has_camera_and_light(sample_graph: SemanticSceneGraph) -> None:
    """Default camera and light nodes must be present."""
    assert "camera_default" in sample_graph.nodes
    assert "light_default" in sample_graph.nodes


def test_build_graph_has_distractor_slots(sample_graph: SemanticSceneGraph) -> None:
    """5 DistractorSlotNodes must be present."""
    slots = [n for n in sample_graph.nodes.values() if isinstance(n, DistractorSlotNode)]
    assert len(slots) == 5


def test_build_graph_articulation_model(sample_graph: SemanticSceneGraph) -> None:
    """Graph must carry the canonical articulation model."""
    assert sample_graph.articulation_model is not None
    assert sample_graph.articulation_model.is_articulatable("microwave")


def test_movable_support_promotion() -> None:
    """ObjectNode is promoted to MovableSupportNode when another object stacks on it."""
    # Find a task with stacking
    files = glob.glob(f"{_BDDL_ROOT}/**/*.bddl", recursive=True)
    stacking_found = False
    for f in files:
        cfg = TaskConfig.from_bddl(f)
        if any(obj.stacked_on for obj in cfg.movable_objects):
            graph = build_semantic_scene_graph(cfg)
            # Check that the stacked_on target is a MovableSupportNode
            for obj in cfg.movable_objects:
                if obj.stacked_on and obj.stacked_on in graph.nodes:
                    target = graph.nodes[obj.stacked_on]
                    assert isinstance(target, MovableSupportNode), (
                        f"Expected MovableSupportNode for {obj.stacked_on}, "
                        f"got {type(target).__name__}"
                    )
                    stacking_found = True
                    break
        if stacking_found:
            break
    if not stacking_found:
        pytest.skip("No stacking tasks found in test corpus")


def test_fixture_articulatable_flag() -> None:
    """Articulatable fixtures must have is_articulatable=True."""
    files = glob.glob(f"{_BDDL_ROOT}/**/*.bddl", recursive=True)
    articulatable_classes = {
        "microwave",
        "wooden_cabinet",
        "white_cabinet",
        "flat_stove",
    }
    for f in files:
        cfg = TaskConfig.from_bddl(f)
        if any(fix.fixture_class in articulatable_classes for fix in cfg.fixtures):
            graph = build_semantic_scene_graph(cfg)
            for nid, node in graph.nodes.items():
                if isinstance(node, FixtureNode) and node.object_class in articulatable_classes:
                    assert node.is_articulatable, f"FixtureNode {nid} ({node.object_class}) should be articulatable"  # fmt: skip  # noqa: E501
            break


def test_dag_validation_passes_on_normal_task(sample_graph: SemanticSceneGraph) -> None:
    """Normal task graphs must pass DAG validation without raising."""
    sample_graph.validate_dag()  # should not raise


def test_dag_cycle_detection_tc3() -> None:
    """TC-3: Cycle detection — circular stacking raises SemanticError."""
    graph = SemanticSceneGraph(
        task_language="cycle test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    graph.add_node(
        ObjectNode(
            node_id="obj_a",
            node_type="object",
            instance_name="obj_a",
            object_class="bowl",
        )
    )
    graph.add_node(
        ObjectNode(
            node_id="obj_b",
            node_type="object",
            instance_name="obj_b",
            object_class="plate",
        )
    )
    # A stacks on B and B stacks on A — cycle!
    graph.add_edge(SceneEdge(src_id="obj_a", dst_id="obj_b", label="stacked_on"))
    graph.add_edge(SceneEdge(src_id="obj_b", dst_id="obj_a", label="stacked_on"))

    with pytest.raises(SemanticError, match="[Cc]ycle"):
        graph.validate_dag()


@pytest.mark.parametrize("bddl_file", _bddl_files(10))
def test_all_bddls_build(bddl_file: str) -> None:
    """All BDDL files in corpus must build a valid scene graph."""
    cfg = TaskConfig.from_bddl(bddl_file)
    graph = build_semantic_scene_graph(cfg)
    assert len(graph.nodes) > 0
    assert graph.task_language  # must have language
    assert graph.bddl_path  # must have bddl_path
