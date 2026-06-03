"""Goal-region resolution — shared between the Scenic renderer (Fix 1: the
distractor↔goal-region require clause) and the G4 domain invariant
(``assert_goal_region_admits_object``).

Motivation
----------
A distractor placed where the task object must *end up* (e.g. on the stove
burner when the task is ``On(bowl, flat_stove_1_cook_region)``) makes the goal
physically unsatisfiable, so the scene is unsolvable — a generator-validity
violation (the dual of G4 "no accidental trivialization": *goal
impossibilization*). Under option (i) distractors may legitimately sit on a
*non-goal* fixture (a wine rack while the task is "bowl on stove"), so we cannot
simply ban distractors from all fixtures; we must ban them only from the
**goal-relevant region** — the place the goal object must occupy.

This module resolves, from the semantic scene graph alone, the set of
goal-relevant regions: for every ``goal_target`` edge (``On``/``In`` predicate),
the world-frame AABB of the target support patch plus the goal object's own
footprint. The renderer turns each into a SAT-correct AABB ``require`` keeping
every distractor out, and the invariant re-derives the same regions to assert
the goal object still fits after distractors are placed.

Frame
-----
Region coordinates from the BDDL ``:regions`` block are expressed relative to
the region's ``:target`` origin. The workspace table sits at the world origin
``(0, 0)`` (table centre projected to the floor), so table-targeted region
bounds are already in workspace coordinates; fixture-targeted regions add the
fixture's world ``(init_x, init_y)``. Regions declared with no explicit
``:ranges`` (e.g. ``cook_region``, ``top_side``) denote the whole top surface of
their target fixture, so we resolve them to that fixture's measured footprint.
"""

from __future__ import annotations

from dataclasses import dataclass

from libero_infinity.asset_metadata import fixture_footprint
from libero_infinity.asset_registry import get_dimensions
from libero_infinity.ir.nodes import (
    FixtureNode,
    MovableSupportNode,
    ObjectNode,
    RegionNode,
    WorkspaceNode,
)
from libero_infinity.ir.scene_graph import SemanticSceneGraph


@dataclass(frozen=True)
class GoalRegion:
    """A world-frame goal-relevant placement region.

    Attributes:
        goal_obj_name:  Instance name of the object the goal places.
        target_name:    Raw goal-predicate target token (region / fixture).
        cx, cy:         World-frame centre of the target region (m).
        half_x, half_y: Half-extents of the target region footprint (m).
        obj_half_x,
        obj_half_y:     Half-extents of the goal object's footprint (m). The
                        goal object, when placed anywhere in the region, sweeps
                        an area up to ``(half + obj_half)`` from the centre; a
                        distractor outside that inflated box can never block the
                        object's final footprint.
        fixture_name:   Instance name of the fixture this region sits on, if the
                        target resolves to (or onto) a fixture; else ``None``.
                        Used to suppress the on-fixture distractor that is
                        *assigned* to a different patch of the same fixture from
                        being double-counted (it is already kept off the goal
                        box by the require clause).
    """

    goal_obj_name: str
    target_name: str
    cx: float
    cy: float
    half_x: float
    half_y: float
    obj_half_x: float
    obj_half_y: float
    fixture_name: str | None


def _anchor_xy(graph: SemanticSceneGraph, target_name: str) -> tuple[float, float] | None:
    """World ``(x, y)`` origin of a region's ``:target`` entity.

    The workspace table is at the world origin, so a WorkspaceNode anchors at
    ``(0, 0)``; a fixture anchors at its ``(init_x, init_y)``. Returns ``None``
    when the target is unknown or has no resolved position.
    """
    node = graph.get_node(target_name)
    if isinstance(node, WorkspaceNode):
        return (0.0, 0.0)
    if isinstance(node, FixtureNode):
        if node.init_x is None or node.init_y is None:
            return None
        return (float(node.init_x), float(node.init_y))
    return None


def _find_region_node(graph: SemanticSceneGraph, target_token: str) -> RegionNode | None:
    """Resolve a goal-predicate target token to its RegionNode, if any.

    BDDL goal predicates reference a region by its fully-qualified name
    ``<target>_<region_name>`` (e.g. ``flat_stove_1_cook_region``). RegionNodes
    store the short name in ``instance_name`` and the owner in ``target``, so we
    match on the reconstructed full name (and, defensively, on the bare name).
    """
    for node in graph.nodes.values():
        if not isinstance(node, RegionNode):
            continue
        full = f"{node.target}_{node.instance_name}" if node.target else node.instance_name
        if target_token in (full, node.instance_name):
            return node
    return None


def _longest_prefix_fixture(graph: SemanticSceneGraph, target_token: str) -> FixtureNode | None:
    """Resolve ``target_token`` to a fixture by exact id or longest-prefix match.

    A goal target such as ``wooden_cabinet_1_top_side`` references a surface on
    ``wooden_cabinet_1``; we pick the longest fixture instance name that is a
    prefix of the token (mirroring ``domain.assert_on_predicates_z``).
    """
    direct = graph.get_node(target_token)
    if isinstance(direct, FixtureNode):
        return direct
    best: FixtureNode | None = None
    for node in graph.nodes.values():
        if not isinstance(node, FixtureNode):
            continue
        if target_token == node.instance_name or target_token.startswith(node.instance_name + "_"):
            if best is None or len(node.instance_name) > len(best.instance_name):
                best = node
    return best


def _fixture_footprint_region(
    fnode: FixtureNode, goal_obj_name: str, target_name: str, obj_half: tuple[float, float]
) -> GoalRegion | None:
    if fnode.init_x is None or fnode.init_y is None:
        return None
    fw, fl = fixture_footprint(fnode.object_class)
    return GoalRegion(
        goal_obj_name=goal_obj_name,
        target_name=target_name,
        cx=float(fnode.init_x),
        cy=float(fnode.init_y),
        half_x=fw / 2.0,
        half_y=fl / 2.0,
        obj_half_x=obj_half[0],
        obj_half_y=obj_half[1],
        fixture_name=fnode.instance_name,
    )


def resolve_goal_regions(graph: SemanticSceneGraph) -> list[GoalRegion]:
    """Return the world-frame goal-relevant regions for the task.

    One :class:`GoalRegion` per resolvable ``goal_target`` edge. Edges whose
    target is itself a movable object (e.g. ``On(bowl, plate)``) are skipped:
    the target's pose is Scenic-sampled and cannot be pinned at codegen, and the
    distractor↔object clearance already keeps distractors off that object.
    Unresolvable targets are skipped (no spurious constraint).
    """
    regions: list[GoalRegion] = []
    seen: set[tuple[str, str]] = set()
    for edge in graph.edges_by_label("goal_target"):
        goal_obj = graph.get_node(edge.src_id)
        if not isinstance(goal_obj, (ObjectNode, MovableSupportNode)):
            continue
        key = (edge.src_id, edge.dst_id)
        if key in seen:
            continue
        seen.add(key)
        obj_class = goal_obj.object_class or goal_obj.instance_name
        odims = get_dimensions(obj_class)
        obj_half = (odims[0] / 2.0, odims[1] / 2.0)
        target = edge.dst_id

        region_node = _find_region_node(graph, target)
        if region_node is not None:
            if (
                region_node.x_min is not None
                and region_node.x_max is not None
                and region_node.y_min is not None
                and region_node.y_max is not None
            ):
                anchor = _anchor_xy(graph, region_node.target)
                if anchor is None:
                    # Owner position unknown — fall back to the fixture footprint
                    # path below if the owner is a fixture.
                    fnode = _longest_prefix_fixture(graph, region_node.target)
                    if fnode is not None:
                        gr = _fixture_footprint_region(fnode, edge.src_id, target, obj_half)
                        if gr is not None:
                            regions.append(gr)
                    continue
                ax, ay = anchor
                rcx = (float(region_node.x_min) + float(region_node.x_max)) / 2.0
                rcy = (float(region_node.y_min) + float(region_node.y_max)) / 2.0
                fnode = _longest_prefix_fixture(graph, region_node.target)
                regions.append(
                    GoalRegion(
                        goal_obj_name=edge.src_id,
                        target_name=target,
                        cx=ax + rcx,
                        cy=ay + rcy,
                        half_x=(float(region_node.x_max) - float(region_node.x_min)) / 2.0,
                        half_y=(float(region_node.y_max) - float(region_node.y_min)) / 2.0,
                        obj_half_x=obj_half[0],
                        obj_half_y=obj_half[1],
                        fixture_name=fnode.instance_name if fnode is not None else None,
                    )
                )
                continue
            # Region with no explicit ranges → the whole top surface of its
            # target fixture (e.g. cook_region on flat_stove_1).
            fnode = _longest_prefix_fixture(graph, region_node.target)
            if fnode is not None:
                gr = _fixture_footprint_region(fnode, edge.src_id, target, obj_half)
                if gr is not None:
                    regions.append(gr)
            continue

        # No matching region node — try a direct/longest-prefix fixture target.
        fnode = _longest_prefix_fixture(graph, target)
        if fnode is not None:
            gr = _fixture_footprint_region(fnode, edge.src_id, target, obj_half)
            if gr is not None:
                regions.append(gr)
            continue
        # Otherwise the target is a movable object / region with no geometry —
        # skip (handled by distractor↔object clearance or simply unconstrained).
    return regions
