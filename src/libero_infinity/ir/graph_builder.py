"""Build a SemanticSceneGraph from a parsed TaskConfig.

Converts the flat BDDL task representation into a typed, validated scene graph
with explicit support/containment/articulation edges.
"""

from __future__ import annotations

import re

from libero_infinity.ir.nodes import (
    ArticulationModel,
    CameraNode,
    DistractorSlotNode,
    FixtureNode,
    LightNode,
    MovableSupportNode,
    ObjectNode,
    RegionNode,
    SceneEdge,
    WorkspaceNode,
)
from libero_infinity.ir.scene_graph import SemanticSceneGraph
from libero_infinity.task_config import TaskConfig

_GOAL_PRED_RE = re.compile(r"\((?:On|In)\s+(\w+)\s+(\w+)\)")
_INIT_ARTIC_RE = re.compile(r"\((Open|Close|Turnon|Turnoff)\s+([^\s()]+)\)")


def _resolve_articulation_init_states(cfg: TaskConfig) -> dict[str, str]:
    """Parse BDDL ``:init`` articulation predicates and bind them to fixtures.

    BDDL allows articulation predicates to target either a fixture instance
    directly (``(Open wooden_cabinet_1)``) or one of its sub-regions
    (``(Open wooden_cabinet_1_top_region)``). Both forms encode the same
    task precondition — the fixture must be in the asserted state at init.
    This helper normalises them to a ``{fixture_instance_name: state_kind}``
    mapping by matching region names against the declared fixtures.
    """
    fixture_names = {f.instance_name for f in cfg.fixtures}
    init_states: dict[str, str] = {}
    for match in _INIT_ARTIC_RE.finditer(cfg.init_text or ""):
        state, target = match.group(1), match.group(2)
        if target in fixture_names:
            init_states[target] = state
            continue
        # Region-style target: find longest fixture prefix that matches.
        owner = None
        for fname in fixture_names:
            if target.startswith(fname + "_") and (owner is None or len(fname) > len(owner)):
                owner = fname
        if owner is not None:
            init_states[owner] = state
    return init_states


def build_semantic_scene_graph(cfg: TaskConfig) -> SemanticSceneGraph:
    """Build and validate a SemanticSceneGraph from a TaskConfig.

    Args:
        cfg: Parsed BDDL task configuration.

    Returns:
        A validated SemanticSceneGraph with all nodes and edges populated.
    """
    art_model = ArticulationModel.canonical()
    graph = SemanticSceneGraph(
        task_language=cfg.language,
        bddl_path=cfg.bddl_path,
        articulation_model=art_model,
    )

    # ------------------------------------------------------------------
    # Determine which movable objects serve as stacking targets
    # (they get promoted to MovableSupportNode)
    # ------------------------------------------------------------------
    movable_support_targets: set[str] = set()
    for obj in cfg.movable_objects:
        if obj.stacked_on:
            movable_support_targets.add(obj.stacked_on)

    # BDDL :init articulation predicates, normalised to {fixture: state}.
    init_articulation_states = _resolve_articulation_init_states(cfg)

    # ------------------------------------------------------------------
    # Fixture nodes
    # ------------------------------------------------------------------
    for fixture in cfg.fixtures:
        fclass = fixture.fixture_class
        node_id = fixture.instance_name

        if fclass in art_model.root_workspace_fixtures:
            node: WorkspaceNode | FixtureNode = WorkspaceNode(
                node_id=node_id,
                node_type="workspace",
                instance_name=fixture.instance_name,
                object_class=fclass,
            )
            graph.add_node(node)
        else:
            is_artic = art_model.is_articulatable(fclass)
            fnode = FixtureNode(
                node_id=node_id,
                node_type="fixture",
                instance_name=fixture.instance_name,
                object_class=fclass,
                placement_target=fixture.placement_target,
                init_x=fixture.init_x,
                init_y=fixture.init_y,
                init_yaw=fixture.init_yaw,
                is_articulatable=is_artic,
                init_state_kind=init_articulation_states.get(fixture.instance_name),
            )
            graph.add_node(fnode)

            # anchored_to edge pointing at placement target workspace/fixture
            if fixture.placement_target:
                graph.add_edge(
                    SceneEdge(
                        src_id=node_id,
                        dst_id=fixture.placement_target,
                        label="anchored_to",
                    )
                )

            # self-loop articulated_by edge for articulatable fixtures
            if is_artic:
                graph.add_edge(
                    SceneEdge(
                        src_id=node_id,
                        dst_id=node_id,
                        label="articulated_by",
                    )
                )

    # ------------------------------------------------------------------
    # Movable object nodes
    # ------------------------------------------------------------------
    for obj in cfg.movable_objects:
        node_id = obj.instance_name

        if obj.instance_name in movable_support_targets:
            onode: MovableSupportNode | ObjectNode = MovableSupportNode(
                node_id=node_id,
                node_type="movable_support",
                instance_name=obj.instance_name,
                object_class=obj.object_class,
                placement_target=obj.placement_target,
                init_x=obj.init_x,
                init_y=obj.init_y,
                init_yaw=obj.init_yaw,
                stacked_on=obj.stacked_on,
            )
        else:
            onode = ObjectNode(
                node_id=node_id,
                node_type="object",
                instance_name=obj.instance_name,
                object_class=obj.object_class,
                placement_target=obj.placement_target,
                init_x=obj.init_x,
                init_y=obj.init_y,
                init_yaw=obj.init_yaw,
                stacked_on=obj.stacked_on,
                contained=obj.contained,
            )

        graph.add_node(onode)

        # Support edge
        if obj.stacked_on:
            graph.add_edge(
                SceneEdge(
                    src_id=node_id,
                    dst_id=obj.stacked_on,
                    label="stacked_on",
                    spatial_kind="stacked",
                )
            )
        elif obj.contained and obj.placement_target:
            graph.add_edge(
                SceneEdge(
                    src_id=node_id,
                    dst_id=obj.placement_target,
                    label="contained_in",
                    spatial_kind="inside",
                )
            )
        elif obj.placement_target:
            graph.add_edge(
                SceneEdge(
                    src_id=node_id,
                    dst_id=obj.placement_target,
                    label="supported_by",
                    spatial_kind="on_surface",
                )
            )

    # ------------------------------------------------------------------
    # Region nodes
    # ------------------------------------------------------------------
    for region_name, region in cfg.regions.items():
        node_id = f"region_{region_name}"
        rnode = RegionNode(
            node_id=node_id,
            node_type="region",
            instance_name=region_name,
            object_class="region",
            target=region.target,
            x_min=region.x_min,
            x_max=region.x_max,
            y_min=region.y_min,
            y_max=region.y_max,
            yaw_min=region.yaw_min,
            yaw_max=region.yaw_max,
        )
        graph.add_node(rnode)

    # ------------------------------------------------------------------
    # Camera and light nodes
    # ------------------------------------------------------------------
    graph.add_node(
        CameraNode(
            node_id="camera_default",
            node_type="camera",
            instance_name="camera_default",
            object_class="camera",
        )
    )
    graph.add_node(
        LightNode(
            node_id="light_default",
            node_type="light",
            instance_name="light_default",
            object_class="light",
        )
    )

    # ------------------------------------------------------------------
    # Distractor slot nodes (always 5 slots)
    # ------------------------------------------------------------------
    for i in range(5):
        graph.add_node(
            DistractorSlotNode(
                node_id=f"distractor_slot_{i}",
                node_type="distractor_slot",
                instance_name=f"distractor_slot_{i}",
                object_class="distractor",
                slot_index=i,
            )
        )

    # ------------------------------------------------------------------
    # must_remain_visible_with edges for task-relevant objects
    # ------------------------------------------------------------------
    for obj_name in cfg.obj_of_interest:
        if obj_name in graph.nodes:
            graph.add_edge(
                SceneEdge(
                    src_id=obj_name,
                    dst_id="camera_default",
                    label="must_remain_visible_with",
                )
            )

    # ------------------------------------------------------------------
    # goal_target edges from goal predicates
    # ------------------------------------------------------------------
    for m in _GOAL_PRED_RE.finditer(cfg.goal_text):
        obj_name = m.group(1)
        target = m.group(2)
        if obj_name in graph.nodes:
            graph.add_edge(
                SceneEdge(
                    src_id=obj_name,
                    dst_id=target,
                    label="goal_target",
                )
            )

    # ------------------------------------------------------------------
    # Structural validation
    # ------------------------------------------------------------------
    graph.validate_dag()

    return graph
