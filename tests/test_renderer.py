"""Tests for the Scenic renderer (Wave 4: renderer).

Covers: purity invariant (grep/AST check), stable output, well-formedness
check, render_scenic output structure, compile_task_to_scenic end-to-end.
"""

from __future__ import annotations

import ast
import glob
import re

import pytest

from libero_infinity.compiler import compile_task_to_scenic, generate_scenic
from libero_infinity.ir.graph_builder import build_semantic_scene_graph
from libero_infinity.ir.nodes import ArticulationModel
from libero_infinity.ir.scene_graph import SemanticSceneGraph
from libero_infinity.planner.composition import plan_perturbations
from libero_infinity.planner.types import PerturbationPlan
from libero_infinity.renderer.scenic_renderer import render_scenic
from libero_infinity.task_config import TaskConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BDDL_ROOT = "src/libero_infinity/data/libero_runtime/bddl_files"
_RENDERER_PATH = "src/libero_infinity/renderer/scenic_renderer.py"


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


@pytest.fixture
def sample_plan(sample_graph: SemanticSceneGraph) -> PerturbationPlan:
    return plan_perturbations(sample_graph, "position")


# ---------------------------------------------------------------------------
# TC-PURITY: Renderer purity invariant (grep check)
# ---------------------------------------------------------------------------


def _find_if_name_references(source: str, name: str) -> list[str]:
    """Find ast.If nodes whose test expression contains a Name(id=name) reference.

    This checks actual code, not docstrings or comments, so the docstring's
    ``if fixture_class == "..."`` example text is correctly excluded.
    """
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Check if the test expression contains a Name reference to `name`
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Name) and sub.id == name:
                    violations.append(ast.unparse(node.test))
                    break
    return violations


def test_renderer_has_no_fixture_class_conditionals() -> None:
    """PURITY: renderer must contain zero 'if fixture_class' conditional branches in code.

    Uses AST analysis to avoid false positives from docstring examples.
    """
    with open(_RENDERER_PATH) as f:
        source = f.read()
    violations = _find_if_name_references(source, "fixture_class")
    assert len(violations) == 0, (
        f"Renderer purity violated: found {len(violations)} 'if fixture_class' branch(es):\n"
        + "\n".join(violations)
    )


def test_renderer_has_no_object_class_conditionals() -> None:
    """PURITY: renderer must contain zero 'if object_class' conditional branches in code.

    Uses AST analysis to avoid false positives from docstring examples.
    """
    with open(_RENDERER_PATH) as f:
        source = f.read()
    violations = _find_if_name_references(source, "object_class")
    assert len(violations) == 0, (
        f"Renderer purity violated: found {len(violations)} 'if object_class' branch(es):\n"
        + "\n".join(violations)
    )


def test_renderer_has_no_task_semantic_class_checks() -> None:
    """PURITY: renderer must not hardcode task-semantic class names (microwave, table, etc.)."""
    with open(_RENDERER_PATH) as f:
        source = f.read()
    # Task-semantic class names should not appear in if/elif/else conditions
    forbidden_patterns = [
        r'if\b.*["\']microwave["\']',
        r'if\b.*["\']table["\']',
        r'if\b.*["\']cabinet["\']',
    ]
    for pat in forbidden_patterns:
        matches = re.findall(pat, source)
        assert len(matches) == 0, f"Renderer purity violated: found pattern '{pat}': {matches}"


def test_renderer_is_syntactically_valid_python() -> None:
    """Renderer source must be syntactically valid Python."""
    with open(_RENDERER_PATH) as f:
        source = f.read()
    try:
        ast.parse(source)
    except SyntaxError as e:
        pytest.fail(f"Renderer has syntax error: {e}")


# ---------------------------------------------------------------------------
# Well-formedness checks
# ---------------------------------------------------------------------------


def test_render_scenic_raises_on_empty_graph(sample_plan: PerturbationPlan) -> None:
    """render_scenic must raise ValueError when graph has no nodes."""
    empty_graph = SemanticSceneGraph(
        task_language="test",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    with pytest.raises(ValueError, match="[Nn]o.*nodes|nodes.*empty"):
        render_scenic(sample_plan, empty_graph)


def test_render_scenic_raises_on_missing_diagnostics(
    sample_graph: SemanticSceneGraph,
) -> None:
    """render_scenic must raise ValueError when plan has no diagnostics."""
    plan = PerturbationPlan(
        active_axes=frozenset(["position"]),
        diagnostics=None,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="[Dd]iagnostics"):
        render_scenic(plan, sample_graph)


def test_render_scenic_raises_on_missing_task_language(
    sample_plan: PerturbationPlan,
) -> None:
    """render_scenic must raise ValueError when graph has no task_language."""
    bad_graph = SemanticSceneGraph(
        task_language="",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    # Add at least one node so the nodes check passes
    from libero_infinity.ir.nodes import WorkspaceNode

    bad_graph.add_node(
        WorkspaceNode(
            node_id="ws",
            node_type="workspace",
            instance_name="table",
            object_class="table",
        )
    )
    with pytest.raises(ValueError, match="[Tt]ask_language|task language"):
        render_scenic(sample_plan, bad_graph)


# ---------------------------------------------------------------------------
# Output structure checks
# ---------------------------------------------------------------------------


def test_render_scenic_contains_model_declaration(
    sample_plan: PerturbationPlan,
    sample_graph: SemanticSceneGraph,
) -> None:
    """Output must contain 'model libero_model'."""
    output = render_scenic(sample_plan, sample_graph)
    assert "model libero_model" in output


def test_render_scenic_contains_param_task(
    sample_plan: PerturbationPlan,
    sample_graph: SemanticSceneGraph,
) -> None:
    """Output must contain 'param task =' declaration."""
    output = render_scenic(sample_plan, sample_graph)
    assert "param task" in output


def test_render_scenic_contains_active_axes(
    sample_plan: PerturbationPlan,
    sample_graph: SemanticSceneGraph,
) -> None:
    """Output must contain active_axes parameter declaration."""
    output = render_scenic(sample_plan, sample_graph)
    assert "active_axes" in output


def test_render_scenic_contains_per_pair_clearance(
    sample_plan: PerturbationPlan,
    sample_graph: SemanticSceneGraph,
) -> None:
    """Output must contain per-pair distance constraints (not a fixed global min_clearance)."""
    output = render_scenic(sample_plan, sample_graph)
    # Per-pair clearance replaces the old fixed param min_clearance = 0.10.
    # Accept either Euclidean ("distance from") or AABB ("abs(...) > ...") style.
    has_clearance = "require (distance from" in output or "require (abs(" in output
    assert has_clearance, "Expected per-pair distance/AABB constraint"
    assert "param min_clearance = 0.10" not in output, "Old fixed min_clearance should be gone"


def test_render_scenic_is_string(
    sample_plan: PerturbationPlan,
    sample_graph: SemanticSceneGraph,
) -> None:
    """render_scenic must return a string."""
    output = render_scenic(sample_plan, sample_graph)
    assert isinstance(output, str)
    assert len(output) > 0


# ---------------------------------------------------------------------------
# Determinism: identical input → identical output
# ---------------------------------------------------------------------------


def test_render_scenic_is_deterministic(
    sample_plan: PerturbationPlan,
    sample_graph: SemanticSceneGraph,
) -> None:
    """Same plan + graph must produce identical output (pure function)."""
    output1 = render_scenic(sample_plan, sample_graph)
    output2 = render_scenic(sample_plan, sample_graph)
    assert output1 == output2, "render_scenic produced different output for same input"


# ---------------------------------------------------------------------------
# Camera rendering
# ---------------------------------------------------------------------------


def test_render_scenic_camera_in_output(sample_graph: SemanticSceneGraph) -> None:
    """When camera axis is active, output must include cam_azimuth parameter."""
    plan = plan_perturbations(sample_graph, "camera")
    output = render_scenic(plan, sample_graph)
    assert "cam_azimuth" in output


def test_render_scenic_no_camera_without_axis(sample_graph: SemanticSceneGraph) -> None:
    """When camera axis is not active, output must NOT include cam_azimuth."""
    plan = plan_perturbations(sample_graph, "position")
    output = render_scenic(plan, sample_graph)
    assert "cam_azimuth" not in output


# ---------------------------------------------------------------------------
# compile_task_to_scenic end-to-end
# ---------------------------------------------------------------------------


def test_compile_task_to_scenic_returns_string(sample_cfg: TaskConfig) -> None:
    """compile_task_to_scenic must return a non-empty string."""
    output = compile_task_to_scenic(sample_cfg, "position")
    assert isinstance(output, str)
    assert len(output) > 0


def test_compile_task_to_scenic_contains_model(sample_cfg: TaskConfig) -> None:
    """compile_task_to_scenic output must contain 'model libero_model'."""
    output = compile_task_to_scenic(sample_cfg, "position")
    assert "model libero_model" in output


def test_compile_task_to_scenic_combined(sample_cfg: TaskConfig) -> None:
    """compile_task_to_scenic with 'combined' preset must include camera and position."""
    output = compile_task_to_scenic(sample_cfg, "combined")
    assert "active_axes" in output
    assert "cam_azimuth" in output  # camera axis


def test_generate_scenic_compatibility(sample_cfg: TaskConfig) -> None:
    """generate_scenic() (backward compat shim) must match compile_task_to_scenic()."""
    out_new = compile_task_to_scenic(sample_cfg, "position")
    out_compat = generate_scenic(sample_cfg, "position")
    assert out_new == out_compat


# ---------------------------------------------------------------------------
# Corpus integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bddl_file", _bddl_files(5))
def test_all_bddls_render_position(bddl_file: str) -> None:
    """All BDDL files must render a valid position-axis Scenic program."""
    cfg = TaskConfig.from_bddl(bddl_file)
    output = compile_task_to_scenic(cfg, "position")
    assert "model libero_model" in output
    assert "param task" in output


@pytest.mark.parametrize("bddl_file", _bddl_files(5))
def test_all_bddls_render_combined(bddl_file: str) -> None:
    """All BDDL files must render a valid combined-axis Scenic program."""
    cfg = TaskConfig.from_bddl(bddl_file)
    output = compile_task_to_scenic(cfg, "combined")
    assert "model libero_model" in output
    assert "cam_azimuth" in output


# ---------------------------------------------------------------------------
# Contained-object exclusion from pairwise constraints
# ---------------------------------------------------------------------------


def test_contained_objects_excluded_from_pairwise_aabb_constraints() -> None:
    """Contained objects (contained=True) must not appear in pairwise AABB require statements.

    _render_constraints skips objects with node.contained==True so they don't
    generate spurious clearance violations with objects outside the container.
    """
    from libero_infinity.ir.nodes import (
        FixtureNode,
        ObjectNode,
        SceneEdge,
        WorkspaceNode,
    )

    graph = SemanticSceneGraph(
        task_language="pick bowl from cabinet and place on plate",
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
    # Object inside cabinet: contained=True
    bowl_in = ObjectNode(
        node_id="akita_black_bowl_1",
        node_type="object",
        instance_name="akita_black_bowl_1",
        object_class="akita_black_bowl",
        contained=True,
        init_x=0.1,
        init_y=0.0,
    )
    graph.add_node(bowl_in)
    graph.add_edge(
        SceneEdge(
            src_id="akita_black_bowl_1",
            dst_id="wooden_cabinet_1",
            label="contained_in",
            spatial_kind="inside",
        )
    )
    # Object on table surface (not contained)
    plate = ObjectNode(
        node_id="plate_1",
        node_type="object",
        instance_name="plate_1",
        object_class="plate",
        contained=False,
        init_x=0.05,
        init_y=0.2,
        placement_target="main_table",
    )
    graph.add_node(plate)
    graph.add_edge(SceneEdge(src_id="plate_1", dst_id="main_table", label="supported_by"))

    plan = plan_perturbations(graph, "position")
    output = render_scenic(plan, graph)

    # The contained bowl should not appear in pairwise require lines
    require_lines = [ln for ln in output.splitlines() if ln.strip().startswith("require (abs(")]
    for line in require_lines:
        assert "akita_black_bowl_1" not in line, f"Contained object should not appear in pairwise AABB constraint:\n{line}"  # fmt: skip  # noqa: E501


# ---------------------------------------------------------------------------
# Relative positioning syntax in output
# ---------------------------------------------------------------------------


def test_stacked_object_uses_offset_by_syntax() -> None:
    """Stacked objects must use 'offset by Vector(Range(...), Range(...), 0.0)' syntax."""
    from libero_infinity.ir.nodes import (
        MovableSupportNode,
        ObjectNode,
        SceneEdge,
        WorkspaceNode,
    )

    graph = SemanticSceneGraph(
        task_language="pick bowl on cookie box and place on plate",
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
    # Cookie box is the support (becomes MovableSupportNode)
    cookies = MovableSupportNode(
        node_id="cookies_1",
        node_type="movable_support",
        instance_name="cookies_1",
        object_class="cookies",
        init_x=0.0,
        init_y=0.1,
        placement_target="main_table",
    )
    graph.add_node(cookies)
    graph.add_edge(SceneEdge(src_id="cookies_1", dst_id="main_table", label="supported_by"))
    # Bowl sits on top of cookies (stacked_on)
    bowl = ObjectNode(
        node_id="akita_black_bowl_1",
        node_type="object",
        instance_name="akita_black_bowl_1",
        object_class="akita_black_bowl",
        stacked_on="cookies_1",
    )
    graph.add_node(bowl)
    graph.add_edge(
        SceneEdge(
            src_id="akita_black_bowl_1",
            dst_id="cookies_1",
            label="stacked_on",
            spatial_kind="stacked",
        )
    )

    plan = plan_perturbations(graph, "position")
    output = render_scenic(plan, graph)

    # Stacked object must use "offset by Vector(Range(...)..." syntax
    assert "offset by Vector(Range(" in output, (
        "Stacked objects must emit 'offset by Vector(Range(...)' relative positioning syntax\n"
        f"Output was:\n{output}"
    )


# ---------------------------------------------------------------------------
# Distractor clearance constraints
# ---------------------------------------------------------------------------


def test_distractor_clearance_constraints_emitted_when_axis_active() -> None:
    """When distractor axis is active, 'distance from distractor_N' constraints must appear."""
    output = compile_task_to_scenic(
        TaskConfig.from_bddl(_bddl_files(1)[0]),
        "distractor",
    )
    assert "distractor" in output, "Distractor objects must be declared"
    # Clearance constraints should be present
    has_distractor_clearance = "distance from distractor_" in output or "distractor_0" in output
    assert has_distractor_clearance, "Expected distractor clearance constraints in output"


def test_distractor_clearance_absent_without_axis() -> None:
    """Without distractor axis, no distractor_N declarations should appear."""
    output = compile_task_to_scenic(
        TaskConfig.from_bddl(_bddl_files(1)[0]),
        "position",
    )
    assert "distractor_0" not in output, "distractor_0 must not appear when distractor axis is not active"  # fmt: skip  # noqa: E501


# ---------------------------------------------------------------------------
# Visibility targets rendered
# ---------------------------------------------------------------------------


def test_visibility_targets_param_emitted_for_task_objects() -> None:
    """Objects with must_remain_visible_with edges must appear in visibility_targets param."""
    from libero_infinity.ir.nodes import (
        CameraNode,
        ObjectNode,
        SceneEdge,
        WorkspaceNode,
    )

    graph = SemanticSceneGraph(
        task_language="put bowl on plate",
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
        CameraNode(
            node_id="camera_default",
            node_type="camera",
            instance_name="camera_default",
            object_class="camera",
        )
    )
    bowl = ObjectNode(
        node_id="akita_black_bowl_1",
        node_type="object",
        instance_name="akita_black_bowl_1",
        object_class="akita_black_bowl",
        init_x=0.0,
        init_y=0.1,
        placement_target="main_table",
    )
    graph.add_node(bowl)
    graph.add_edge(
        SceneEdge(src_id="akita_black_bowl_1", dst_id="main_table", label="supported_by")
    )
    # Add must_remain_visible_with edge
    graph.add_edge(
        SceneEdge(
            src_id="akita_black_bowl_1",
            dst_id="camera_default",
            label="must_remain_visible_with",
        )
    )

    plan = plan_perturbations(graph, "position")
    output = render_scenic(plan, graph)

    assert "visibility_targets" in output, "visibility_targets param must be emitted when objects have must_remain_visible_with edges"  # fmt: skip  # noqa: E501
    assert "akita_black_bowl_1" in output, "The visible object name must appear in visibility_targets list"  # fmt: skip  # noqa: E501
