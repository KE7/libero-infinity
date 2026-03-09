"""Typed task semantics derived from a parsed TaskConfig."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from libero_infinity.task_config import RegionInfo, TaskConfig

GoalPredicateType = Literal["On", "In", "Open", "Close", "Turnon", "Turnoff"]
EntityKind = Literal["object", "fixture", "region", "unknown"]
SupportRelationLabel = Literal[
    "on_root_region",
    "on_fixture_region",
    "on_object_region",
    "in_fixture_region",
    "in_object_region",
    "on_object",
]
Phase = Literal["init", "goal"]
VisibilityTargetKind = Literal["object", "fixture", "region"]
ArticulationFamily = Literal["cabinet", "microwave"]
ArticulationKind = Literal["drawer", "door"]
ArticulationState = Literal["open", "closed", "unknown"]

_ATOMIC_PREDICATE_RE = re.compile(r"\((\w+)\s+([^\s()]+)(?:\s+([^\s()]+))?\)")
_SUPPORTED_GOAL_PREDICATES: set[GoalPredicateType] = {
    "On",
    "In",
    "Open",
    "Close",
    "Turnon",
    "Turnoff",
}
_ARTICULATABLE_FIXTURE_CLASSES: dict[str, tuple[ArticulationFamily, ArticulationKind]] = {
    "microwave": ("microwave", "door"),
    "white_cabinet": ("cabinet", "drawer"),
    "wooden_cabinet": ("cabinet", "drawer"),
}
_ROOT_WORKSPACE_FIXTURES = {
    "table",
    "kitchen_table",
    "living_room_table",
    "study_table",
    "floor",
}


@dataclass(frozen=True)
class AtomicGoalPredicate:
    """A single typed goal predicate with target decomposition."""

    predicate: GoalPredicateType
    raw_arguments: tuple[str, ...]
    primary_name: str
    primary_kind: EntityKind
    target_name: str | None = None
    target_kind: EntityKind | None = None
    support_instance_name: str | None = None
    support_region_name: str | None = None


@dataclass(frozen=True)
class GoalRegionExclusion:
    """A bounded goal region that should be excluded from reset placement."""

    object_name: str
    region_full_name: str
    region_name: str
    support_instance_name: str
    bounds: tuple[float, float, float, float]
    yaw_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class SupportGraphEdge:
    """A labeled support/containment relation in the task scene graph."""

    phase: Phase
    child_name: str
    parent_name: str
    relation: SupportRelationLabel
    region_full_name: str | None = None
    region_name: str | None = None


@dataclass(frozen=True)
class ArticulatableFixtureSemantics:
    """Metadata for task-relevant articulatable fixtures."""

    fixture_name: str
    fixture_class: str
    family: ArticulationFamily
    articulation_kind: ArticulationKind
    control_target_name: str
    compartment_name: str | None
    init_state: ArticulationState
    goal_state: ArticulationState


@dataclass(frozen=True)
class VisibilityTarget:
    """An entity that should stay visible for task execution."""

    name: str
    kind: VisibilityTargetKind
    reason: str


@dataclass(frozen=True)
class YawHint:
    """Yaw prior sourced from a BDDL region annotation."""

    entity_name: str
    phase: Phase
    region_full_name: str
    support_instance_name: str
    yaw_range: tuple[float, float]


@dataclass(frozen=True)
class CoordinationGroup:
    """Objects which share the same resolved goal support layout."""

    root_support_name: str
    member_names: tuple[str, ...]


@dataclass(frozen=True)
class TaskSemantics:
    """Typed scene/task semantics for a parsed TaskConfig."""

    goal_predicates: tuple[AtomicGoalPredicate, ...]
    goal_region_exclusions: tuple[GoalRegionExclusion, ...]
    init_support_graph: tuple[SupportGraphEdge, ...]
    goal_support_graph: tuple[SupportGraphEdge, ...]
    articulatable_fixtures: tuple[ArticulatableFixtureSemantics, ...]
    visibility_targets: tuple[VisibilityTarget, ...]
    yaw_hints: tuple[YawHint, ...]
    coordination_groups: tuple[CoordinationGroup, ...]


def derive_task_semantics(cfg: TaskConfig) -> TaskSemantics:
    """Derive typed task semantics from a parsed TaskConfig."""
    goal_atoms = tuple(_build_goal_predicates(cfg))
    goal_region_exclusions = tuple(_build_goal_region_exclusions(goal_atoms, cfg))
    init_support_graph = tuple(_build_init_support_graph(cfg))
    goal_support_graph = tuple(_build_goal_support_graph(goal_atoms, cfg))
    articulatable_fixtures = tuple(_build_articulatable_fixtures(cfg, goal_atoms))
    visibility_targets = tuple(
        _build_visibility_targets(
            cfg,
            goal_atoms=goal_atoms,
            articulatable_fixtures=articulatable_fixtures,
        )
    )
    yaw_hints = tuple(_build_yaw_hints(cfg, goal_atoms))
    coordination_groups = tuple(_build_coordination_groups(goal_atoms, cfg))
    return TaskSemantics(
        goal_predicates=goal_atoms,
        goal_region_exclusions=goal_region_exclusions,
        init_support_graph=init_support_graph,
        goal_support_graph=goal_support_graph,
        articulatable_fixtures=articulatable_fixtures,
        visibility_targets=visibility_targets,
        yaw_hints=yaw_hints,
        coordination_groups=coordination_groups,
    )


def _build_goal_predicates(cfg: TaskConfig) -> list[AtomicGoalPredicate]:
    fixture_names = {fixture.instance_name for fixture in cfg.fixtures}
    movable_names = {obj.instance_name for obj in cfg.movable_objects}
    goal_atoms: list[AtomicGoalPredicate] = []

    for pred_name, args in _parse_atomic_predicates(cfg.goal_text):
        if pred_name not in _SUPPORTED_GOAL_PREDICATES:
            continue
        primary_name = args[0]
        primary_kind = _entity_kind(primary_name, movable_names, fixture_names, cfg.region_refs)
        target_name = args[1] if len(args) > 1 else None
        target_kind = None
        support_instance_name = None
        support_region_name = None
        if target_name is not None:
            target_kind = _entity_kind(target_name, movable_names, fixture_names, cfg.region_refs)
            region = cfg.region_refs.get(target_name)
            if region is not None:
                support_instance_name = region.target
                support_region_name = region.name
            elif target_name in movable_names or target_name in fixture_names:
                support_instance_name = target_name
        elif primary_name in cfg.region_refs:
            region = cfg.region_refs[primary_name]
            support_instance_name = region.target
            support_region_name = region.name
        elif primary_name in fixture_names:
            support_instance_name = primary_name

        goal_atoms.append(
            AtomicGoalPredicate(
                predicate=pred_name,
                raw_arguments=args,
                primary_name=primary_name,
                primary_kind=primary_kind,
                target_name=target_name,
                target_kind=target_kind,
                support_instance_name=support_instance_name,
                support_region_name=support_region_name,
            )
        )
    return goal_atoms


def _build_goal_region_exclusions(
    goal_atoms: tuple[AtomicGoalPredicate, ...],
    cfg: TaskConfig,
) -> list[GoalRegionExclusion]:
    exclusions: list[GoalRegionExclusion] = []
    for atom in goal_atoms:
        if atom.predicate not in {"On", "In"} or atom.target_name is None:
            continue
        region = cfg.region_refs.get(atom.target_name)
        if region is None or not region.has_bounds:
            continue
        yaw_range = None
        if region.has_yaw_hint:
            yaw_range = (float(region.yaw_min), float(region.yaw_max))
        exclusions.append(
            GoalRegionExclusion(
                object_name=atom.primary_name,
                region_full_name=atom.target_name,
                region_name=region.name,
                support_instance_name=region.target,
                bounds=(
                    float(region.x_min),
                    float(region.x_max),
                    float(region.y_min),
                    float(region.y_max),
                ),
                yaw_range=yaw_range,
            )
        )
    return exclusions


def _build_init_support_graph(cfg: TaskConfig) -> list[SupportGraphEdge]:
    movable_names = {obj.instance_name for obj in cfg.movable_objects}
    fixture_names = {fixture.instance_name for fixture in cfg.fixtures}
    edges: list[SupportGraphEdge] = []

    for obj in cfg.movable_objects:
        parent_name: str | None = None
        region_full_name: str | None = None
        region_name: str | None = None
        relation: SupportRelationLabel

        if obj.stacked_on and obj.stacked_on in movable_names:
            parent_name = obj.stacked_on
            relation = "on_object"
        elif obj.region_name and obj.placement_target:
            parent_name = obj.placement_target
            region_name = obj.region_name
            region_full_name = f"{obj.placement_target}_{obj.region_name}"
            if obj.contained:
                relation = (
                    "in_fixture_region"
                    if obj.placement_target in fixture_names
                    else "in_object_region"
                )
            elif obj.placement_target in fixture_names:
                relation = (
                    "on_root_region"
                    if _fixture_class(cfg, obj.placement_target) in _ROOT_WORKSPACE_FIXTURES
                    else "on_fixture_region"
                )
            else:
                relation = "on_object_region"
        else:
            continue

        edges.append(
            SupportGraphEdge(
                phase="init",
                child_name=obj.instance_name,
                parent_name=parent_name,
                relation=relation,
                region_full_name=region_full_name,
                region_name=region_name,
            )
        )
    return edges


def _build_goal_support_graph(
    goal_atoms: tuple[AtomicGoalPredicate, ...],
    cfg: TaskConfig,
) -> list[SupportGraphEdge]:
    movable_names = {obj.instance_name for obj in cfg.movable_objects}
    fixture_names = {fixture.instance_name for fixture in cfg.fixtures}
    edges: list[SupportGraphEdge] = []

    for atom in goal_atoms:
        if atom.predicate not in {"On", "In"} or atom.target_name is None:
            continue
        relation: SupportRelationLabel
        parent_name = atom.support_instance_name or atom.target_name
        region_full_name = None
        region_name = None

        if atom.target_name in cfg.region_refs:
            region = cfg.region_refs[atom.target_name]
            parent_name = region.target
            region_full_name = atom.target_name
            region_name = region.name
            if atom.predicate == "In":
                relation = (
                    "in_fixture_region" if parent_name in fixture_names else "in_object_region"
                )
            elif parent_name in fixture_names:
                relation = (
                    "on_root_region"
                    if _fixture_class(cfg, parent_name) in _ROOT_WORKSPACE_FIXTURES
                    else "on_fixture_region"
                )
            else:
                relation = "on_object_region"
        elif atom.target_name in movable_names:
            relation = "on_object"
        else:
            continue

        edges.append(
            SupportGraphEdge(
                phase="goal",
                child_name=atom.primary_name,
                parent_name=parent_name,
                relation=relation,
                region_full_name=region_full_name,
                region_name=region_name,
            )
        )
    return edges


def _build_articulatable_fixtures(
    cfg: TaskConfig,
    goal_atoms: tuple[AtomicGoalPredicate, ...],
) -> list[ArticulatableFixtureSemantics]:
    fixtures_by_name = {fixture.instance_name: fixture for fixture in cfg.fixtures}
    init_states = _collect_articulation_states(cfg.init_text, cfg)
    goal_states = _collect_articulation_states(cfg.goal_text, cfg)
    specs: dict[tuple[str, str], ArticulatableFixtureSemantics] = {}

    def add_fixture(
        fixture_name: str, control_target_name: str, compartment_name: str | None
    ) -> None:
        fixture = fixtures_by_name.get(fixture_name)
        if fixture is None:
            return
        articulation = _ARTICULATABLE_FIXTURE_CLASSES.get(fixture.fixture_class)
        if articulation is None:
            return
        family, articulation_kind = articulation
        spec = ArticulatableFixtureSemantics(
            fixture_name=fixture_name,
            fixture_class=fixture.fixture_class,
            family=family,
            articulation_kind=articulation_kind,
            control_target_name=control_target_name,
            compartment_name=compartment_name,
            init_state=init_states.get(control_target_name, "unknown"),
            goal_state=goal_states.get(control_target_name, "unknown"),
        )
        specs[(fixture_name, control_target_name)] = spec

    for atom in goal_atoms:
        if atom.predicate in {"Open", "Close"}:
            control_target = atom.primary_name
            region = cfg.region_refs.get(control_target)
            if region is not None:
                add_fixture(
                    fixture_name=region.target,
                    control_target_name=control_target,
                    compartment_name=_compartment_name(region.name),
                )
            elif atom.primary_name in fixtures_by_name:
                add_fixture(
                    fixture_name=atom.primary_name,
                    control_target_name=atom.primary_name,
                    compartment_name=(
                        "heating"
                        if fixtures_by_name[atom.primary_name].fixture_class == "microwave"
                        else None
                    ),
                )
        elif atom.predicate == "In" and atom.target_name in cfg.region_refs:
            region = cfg.region_refs[atom.target_name]
            control_target = (
                region.target
                if fixtures_by_name.get(region.target, None) is not None
                and fixtures_by_name[region.target].fixture_class == "microwave"
                else atom.target_name
            )
            add_fixture(
                fixture_name=region.target,
                control_target_name=control_target,
                compartment_name=_compartment_name(region.name),
            )
    return list(specs.values())


def _build_visibility_targets(
    cfg: TaskConfig,
    *,
    goal_atoms: tuple[AtomicGoalPredicate, ...],
    articulatable_fixtures: tuple[ArticulatableFixtureSemantics, ...],
) -> list[VisibilityTarget]:
    movable_names = {obj.instance_name for obj in cfg.movable_objects}
    fixture_names = {fixture.instance_name for fixture in cfg.fixtures}
    targets: list[VisibilityTarget] = []
    seen: set[tuple[str, VisibilityTargetKind]] = set()

    def add_target(name: str, kind: VisibilityTargetKind, reason: str) -> None:
        key = (name, kind)
        if key in seen:
            return
        seen.add(key)
        targets.append(VisibilityTarget(name=name, kind=kind, reason=reason))

    for name in cfg.obj_of_interest:
        if name in movable_names:
            add_target(name, "object", "obj_of_interest")
        elif name in fixture_names:
            add_target(name, "fixture", "obj_of_interest")
        elif name in cfg.region_refs:
            add_target(name, "region", "obj_of_interest")
            add_target(cfg.region_refs[name].target, "fixture", "goal_region_support")

    for atom in goal_atoms:
        if atom.primary_name in movable_names:
            add_target(atom.primary_name, "object", "goal_primary")
        elif atom.primary_name in fixture_names:
            add_target(atom.primary_name, "fixture", "goal_primary")
        elif atom.primary_name in cfg.region_refs:
            add_target(atom.primary_name, "region", "goal_primary")
            add_target(
                cfg.region_refs[atom.primary_name].target,
                "fixture",
                "goal_region_support",
            )

        if atom.target_name is None:
            continue
        if atom.target_name in movable_names:
            add_target(atom.target_name, "object", "goal_target")
        elif atom.target_name in fixture_names:
            add_target(atom.target_name, "fixture", "goal_target")
        elif atom.target_name in cfg.region_refs:
            add_target(atom.target_name, "region", "goal_target")
            parent = cfg.region_refs[atom.target_name].target
            kind: VisibilityTargetKind = "fixture" if parent in fixture_names else "object"
            add_target(parent, kind, "goal_region_support")

    for fixture in articulatable_fixtures:
        add_target(fixture.fixture_name, "fixture", "articulatable_fixture")
        if fixture.control_target_name in cfg.region_refs:
            add_target(fixture.control_target_name, "region", "articulatable_control")
    return targets


def _build_yaw_hints(
    cfg: TaskConfig,
    goal_atoms: tuple[AtomicGoalPredicate, ...],
) -> list[YawHint]:
    hints: list[YawHint] = []
    seen: set[tuple[str, Phase, str]] = set()

    def add_hint(entity_name: str, phase: Phase, region: RegionInfo) -> None:
        if not region.has_yaw_hint:
            return
        key = (entity_name, phase, region.full_name)
        if key in seen:
            return
        seen.add(key)
        hints.append(
            YawHint(
                entity_name=entity_name,
                phase=phase,
                region_full_name=region.full_name,
                support_instance_name=region.target,
                yaw_range=(float(region.yaw_min), float(region.yaw_max)),
            )
        )

    for obj in cfg.movable_objects:
        if obj.region_name and obj.placement_target:
            region = cfg.region_refs.get(f"{obj.placement_target}_{obj.region_name}")
            if region is not None:
                add_hint(obj.instance_name, "init", region)
    for fixture in cfg.fixtures:
        if fixture.region_name and fixture.placement_target:
            region = cfg.region_refs.get(f"{fixture.placement_target}_{fixture.region_name}")
            if region is not None:
                add_hint(fixture.instance_name, "init", region)
    for atom in goal_atoms:
        if atom.target_name in cfg.region_refs:
            add_hint(atom.primary_name, "goal", cfg.region_refs[atom.target_name])
    return hints


def _build_coordination_groups(
    goal_atoms: tuple[AtomicGoalPredicate, ...],
    cfg: TaskConfig,
) -> list[CoordinationGroup]:
    parent_by_child: dict[str, str] = {}
    root_by_child: dict[str, str] = {}
    order: list[str] = []

    for atom in goal_atoms:
        if atom.predicate not in {"On", "In"} or atom.target_name is None:
            continue
        child = atom.primary_name
        parent_by_child[child] = atom.target_name
        order.append(child)

    def root_support_name(name: str) -> str | None:
        parent = parent_by_child.get(name)
        if parent is None:
            return None
        if parent not in parent_by_child:
            return parent
        # Follow stacked goals until we reach the non-object support anchor.
        return root_support_name(parent)

    for child in order:
        root = root_support_name(child)
        if root is not None:
            root_by_child[child] = root

    grouped: dict[str, list[str]] = {}
    for child in order:
        root = root_by_child.get(child)
        if root is None:
            continue
        grouped.setdefault(root, [])
        if child not in grouped[root]:
            grouped[root].append(child)

    return [
        CoordinationGroup(root_support_name=root, member_names=tuple(members))
        for root, members in grouped.items()
        if len(members) > 1
    ]


def _parse_atomic_predicates(
    block_text: str,
) -> list[tuple[GoalPredicateType, tuple[str, ...]]]:
    atoms: list[tuple[GoalPredicateType, tuple[str, ...]]] = []
    for match in _ATOMIC_PREDICATE_RE.finditer(block_text):
        predicate = match.group(1)
        if predicate == "And" or predicate not in _SUPPORTED_GOAL_PREDICATES:
            continue
        args = (match.group(2),) if match.group(3) is None else (match.group(2), match.group(3))
        atoms.append((predicate, args))
    return atoms


def _entity_kind(
    name: str,
    movable_names: set[str],
    fixture_names: set[str],
    region_refs: dict[str, RegionInfo],
) -> EntityKind:
    if name in movable_names:
        return "object"
    if name in fixture_names:
        return "fixture"
    if name in region_refs:
        return "region"
    return "unknown"


def _fixture_class(cfg: TaskConfig, fixture_name: str) -> str | None:
    for fixture in cfg.fixtures:
        if fixture.instance_name == fixture_name:
            return fixture.fixture_class
    return None


def _collect_articulation_states(
    block_text: str,
    cfg: TaskConfig,
) -> dict[str, ArticulationState]:
    states: dict[str, ArticulationState] = {}
    for predicate, args in _parse_atomic_predicates(block_text):
        if predicate not in {"Open", "Close"}:
            continue
        target = args[0]
        region = cfg.region_refs.get(target)
        fixture_name = region.target if region is not None else target
        fixture_class = _fixture_class(cfg, fixture_name)
        if fixture_class not in _ARTICULATABLE_FIXTURE_CLASSES:
            continue
        states[target] = "open" if predicate == "Open" else "closed"
    return states


def _compartment_name(region_name: str) -> str | None:
    for suffix in ("top_region", "middle_region", "bottom_region"):
        if region_name == suffix:
            return suffix.removesuffix("_region")
    if region_name == "heating_region":
        return "heating"
    return None
