"""Libero-Infinity Intermediate Representation (IR).

Provides a typed semantic scene graph derived from parsed BDDL tasks,
independent of any Scenic syntax.

Public API::

    from libero_infinity.ir import (
        # Node types
        SceneNode, WorkspaceNode, FixtureNode, MovableSupportNode,
        ObjectNode, RegionNode, CameraNode, LightNode, DistractorSlotNode,
        # Edge / relation types
        SceneEdge,
        # Type literals
        NodeType, EdgeLabel, SpatialRelationKind,
        ArticulationFamily, ArticulationKind,
        # Models
        ArticulationModel, PlanDiagnostics,
        # Graph
        SemanticSceneGraph, SemanticError,
        # Builder
        build_semantic_scene_graph,
    )
"""

from __future__ import annotations

from libero_infinity.ir.graph_builder import build_semantic_scene_graph
from libero_infinity.ir.nodes import (
    ArticulationFamily,
    ArticulationKind,
    ArticulationModel,
    CameraNode,
    DistractorSlotNode,
    EdgeLabel,
    FixtureNode,
    LightNode,
    MovableSupportNode,
    NodeType,
    ObjectNode,
    PlanDiagnostics,
    RegionNode,
    SceneEdge,
    SceneNode,
    SpatialRelationKind,
    WorkspaceNode,
)
from libero_infinity.ir.scene_graph import SemanticError, SemanticSceneGraph

__all__ = [
    # Node classes
    "SceneNode",
    "WorkspaceNode",
    "FixtureNode",
    "MovableSupportNode",
    "ObjectNode",
    "RegionNode",
    "CameraNode",
    "LightNode",
    "DistractorSlotNode",
    # Edge
    "SceneEdge",
    # Literals
    "NodeType",
    "EdgeLabel",
    "SpatialRelationKind",
    "ArticulationFamily",
    "ArticulationKind",
    # Models
    "ArticulationModel",
    "PlanDiagnostics",
    # Graph
    "SemanticSceneGraph",
    "SemanticError",
    # Builder
    "build_semantic_scene_graph",
]
