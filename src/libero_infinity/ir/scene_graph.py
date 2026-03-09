"""Semantic scene graph for Libero-Infinity IR.

The SemanticSceneGraph holds all nodes and edges parsed from a BDDL task
and provides graph traversal, structural queries, and DAG validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from libero_infinity.ir.nodes import (
    ArticulationModel,
    EdgeLabel,
    SceneEdge,
    SceneNode,
)


class SemanticError(Exception):
    """Raised when the scene graph fails semantic validation."""


@dataclass
class SemanticSceneGraph:
    """Full semantic representation of a BDDL task as a typed scene graph."""

    nodes: dict[str, SceneNode] = field(default_factory=dict)
    edges: list[SceneEdge] = field(default_factory=list)
    articulation_model: ArticulationModel = field(default_factory=ArticulationModel.canonical)
    task_language: str = ""
    bddl_path: str = ""

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_node(self, node: SceneNode) -> None:
        """Register a node in the graph (keyed by node_id)."""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: SceneEdge) -> None:
        """Append a directed edge to the graph."""
        self.edges.append(edge)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> SceneNode | None:
        """Return the node with the given id, or None if absent."""
        return self.nodes.get(node_id)

    def edges_from(self, node_id: str) -> list[SceneEdge]:
        """Return all edges whose source is node_id."""
        return [e for e in self.edges if e.src_id == node_id]

    def edges_to(self, node_id: str) -> list[SceneEdge]:
        """Return all edges whose destination is node_id."""
        return [e for e in self.edges if e.dst_id == node_id]

    def edges_by_label(self, label: EdgeLabel) -> list[SceneEdge]:
        """Return all edges with the given label."""
        return [e for e in self.edges if e.label == label]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_dag(self) -> None:
        """Topological-sort validation on support/containment/stacking edges.

        Raises SemanticError with a cycle path string if a cycle is detected
        in the supported_by / contained_in / stacked_on sub-graph.
        """
        support_labels: set[str] = {"supported_by", "contained_in", "stacked_on"}

        # Build dependency map: node -> list of nodes it depends on (its support)
        # supported_by edge: src depends on dst
        deps: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for edge in self.edges:
            if edge.label in support_labels and edge.src_id != edge.dst_id:
                if edge.src_id in deps:
                    deps[edge.src_id].append(edge.dst_id)

        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node_id: str, path: list[str]) -> None:
            visited.add(node_id)
            in_stack.add(node_id)
            path.append(node_id)

            for dep in deps.get(node_id, []):
                if dep not in visited:
                    dfs(dep, path)
                elif dep in in_stack:
                    cycle_start = path.index(dep)
                    cycle_path = path[cycle_start:] + [dep]
                    cycle_str = " -> ".join(cycle_path)
                    raise SemanticError(f"Cycle detected in support graph: {cycle_str}")

            path.pop()
            in_stack.discard(node_id)

        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id, [])
