"""Automatic BDDL task analysis for multi-task Scenic program generation.

Parses any BDDL file to extract the structured information needed to
auto-generate a Scenic perturbation program, without hand-writing
per-task .scenic files.

Usage::

    from libero_infinity.task_config import TaskConfig
    cfg = TaskConfig.from_bddl("path/to/task.bddl")
    print(cfg.movable_objects)   # [ObjectInfo(...), ...]
    print(cfg.fixtures)          # [FixtureInfo(...), ...]
    print(cfg.obj_of_interest)   # ["bowl_1", "plate_1"]
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from functools import cached_property

from libero_infinity.bddl_preprocessor import (
    _extract_block,
    _find_closing_paren,
    _parse_declarations,
    _parse_language,
)


@dataclass
class RegionInfo:
    """A named placement region from the BDDL (:regions ...) block."""

    name: str
    target: str  # e.g. "main_table" or "wooden_cabinet_1"
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    yaw_min: float | None = None
    yaw_max: float | None = None

    @property
    def has_bounds(self) -> bool:
        return all(v is not None for v in (self.x_min, self.x_max, self.y_min, self.y_max))

    @property
    def full_name(self) -> str:
        return f"{self.target}_{self.name}"

    @property
    def has_yaw_hint(self) -> bool:
        return self.yaw_min is not None and self.yaw_max is not None

    @property
    def yaw_centre(self) -> float | None:
        if not self.has_yaw_hint:
            return None
        return (self.yaw_min + self.yaw_max) / 2.0

    @property
    def centre(self) -> tuple[float, float] | None:
        if not self.has_bounds:
            return None
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )


# Fixture classes that are fixed workspace surfaces (not movable goal fixtures).
# Used by perturbation planning to distinguish between "table" (immobile) and
# "flat_stove" / "wooden_cabinet" (potentially movable goal fixtures).
IMMOBILE_WORKSPACE_FIXTURES: frozenset[str] = frozenset(
    {
        "table",
        "kitchen_table",
        "living_room_table",
        "study_table",
        "floor",
    }
)

# Backward-compat alias (private name used historically)
_IMMOBILE_WORKSPACE_FIXTURES = IMMOBILE_WORKSPACE_FIXTURES


def _support_parent_names(
    objects: list["ObjectInfo"],
    *,
    moving_fixture_names: set[str],
) -> dict[str, str | None]:
    """Resolve the movable support parent, if any, for each task object.

    Returns a mapping from object instance_name to the instance_name of its
    movable support parent (or None if the object rests on a fixed workspace surface).
    """
    movable_names = {obj.instance_name for obj in objects}
    parent_names: dict[str, str | None] = {}
    for obj in objects:
        if obj.stacked_on and obj.stacked_on in movable_names:
            parent_names[obj.instance_name] = obj.stacked_on
        elif obj.placement_target in movable_names:
            parent_names[obj.instance_name] = obj.placement_target
        elif obj.placement_target in moving_fixture_names:
            parent_names[obj.instance_name] = obj.placement_target
        else:
            parent_names[obj.instance_name] = None
    return parent_names


@dataclass
class ObjectInfo:
    """A movable object from the BDDL (:objects ...) block."""

    instance_name: str  # e.g. "akita_black_bowl_1"
    object_class: str  # e.g. "akita_black_bowl"
    region_name: str | None = None  # e.g. "akita_black_bowl_region"
    init_x: float | None = None  # centre of placement region
    init_y: float | None = None
    init_yaw: float | None = None
    stacked_on: str | None = None  # instance name of object this sits on
    placement_target: str | None = None  # support surface / container instance
    contained: bool = False


@dataclass
class FixtureInfo:
    """A non-movable fixture from the BDDL (:fixtures ...) block."""

    instance_name: str  # e.g. "wooden_cabinet_1"
    fixture_class: str  # e.g. "wooden_cabinet"
    region_name: str | None = None  # e.g. "cabinet_region"
    placement_target: str | None = None  # e.g. "main_table"
    init_x: float | None = None  # centre of placement region
    init_y: float | None = None
    init_yaw: float | None = None


@dataclass
class TaskConfig:
    """Structured representation of a BDDL task file."""

    bddl_path: str
    language: str
    movable_objects: list[ObjectInfo] = field(default_factory=list)
    fixtures: list[FixtureInfo] = field(default_factory=list)
    regions: dict[str, RegionInfo] = field(default_factory=dict)
    region_refs: dict[str, RegionInfo] = field(default_factory=dict)
    obj_of_interest: list[str] = field(default_factory=list)
    init_text: str = ""
    goal_text: str = ""

    @classmethod
    def from_bddl(cls, bddl_path: str | pathlib.Path) -> "TaskConfig":
        """Parse a BDDL file and extract all task structure."""
        bddl_path = pathlib.Path(bddl_path)
        content = bddl_path.read_text()
        return cls._parse(str(bddl_path.resolve()), content)

    @classmethod
    def from_string(cls, bddl_content: str, path: str = "<string>") -> "TaskConfig":
        """Parse BDDL content from a string."""
        return cls._parse(path, bddl_content)

    @classmethod
    def _parse(cls, path: str, content: str) -> "TaskConfig":
        cfg = cls(bddl_path=path, language=_parse_language(content))
        cfg.regions = _parse_regions(content)
        cfg.region_refs = _parse_region_refs(content)
        cfg.fixtures = _parse_fixtures(content)
        cfg.movable_objects = _parse_objects(content, cfg.regions)
        cfg.obj_of_interest = _parse_obj_of_interest(content)
        cfg.init_text = _parse_init(content)
        cfg.goal_text = _parse_goal(content)

        # Resolve initial positions from region centres + init predicates
        _resolve_init_positions(content, cfg)

        return cfg

    @property
    def perturbable_classes(self) -> set[str]:
        """Object classes present in this task that have OOD variants."""
        from libero_infinity.asset_registry import ASSET_VARIANTS

        return {
            obj.object_class for obj in self.movable_objects if obj.object_class in ASSET_VARIANTS
        }

    @property
    def goal_fixture_names(self) -> set[str]:
        """Fixture instances referenced by the goal or object-of-interest anchors."""
        refs = set(self.obj_of_interest)
        refs.add(self.goal_text)
        return {
            fixture.instance_name
            for fixture in self.fixtures
            if any(
                ref == fixture.instance_name
                or ref.startswith(f"{fixture.instance_name}_")
                or fixture.instance_name in ref
                for ref in refs
            )
        }

    @cached_property
    def semantics(self):
        """Typed task semantics derived from the parsed BDDL structure."""
        from libero_infinity.task_semantics import derive_task_semantics

        return derive_task_semantics(self)


# ---------------------------------------------------------------------------
# BDDL parsing helpers
# ---------------------------------------------------------------------------


def _parse_regions(content: str) -> dict[str, RegionInfo]:
    """Parse all regions from the (:regions ...) block."""
    regions: dict[str, RegionInfo] = {}
    for region in _iter_regions(content):
        regions[region.name] = region
    return regions


def _parse_region_refs(content: str) -> dict[str, RegionInfo]:
    """Parse regions keyed by fully qualified target reference."""
    return {region.full_name: region for region in _iter_regions(content)}


def _parse_fixtures(content: str) -> list[FixtureInfo]:
    body = _extract_block(content, "fixtures")
    if not body:
        return []
    return [
        FixtureInfo(instance_name=inst, fixture_class=cls)
        for inst, cls in _parse_declarations(body)
    ]


def _parse_objects(
    content: str,
    regions: dict[str, RegionInfo],
) -> list[ObjectInfo]:
    body = _extract_block(content, "objects")
    if not body:
        return []
    return [
        ObjectInfo(instance_name=inst, object_class=cls) for inst, cls in _parse_declarations(body)
    ]


def _parse_obj_of_interest(content: str) -> list[str]:
    body = _extract_block(content, "obj_of_interest")
    if not body:
        return []
    return [line.strip() for line in body.splitlines() if line.strip()]


def _parse_goal(content: str) -> str:
    body = _extract_block(content, "goal")
    if not body:
        return ""
    return body.strip()


def _parse_init(content: str) -> str:
    body = _extract_block(content, "init")
    if not body:
        return ""
    return body.strip()


def _iter_regions(content: str):
    """Yield every parsed region, preserving duplicate short names."""
    body = _extract_block(content, "regions")
    if not body:
        return

    region_starts = [
        match
        for match in re.finditer(r"^\s*\(([^:\s()]+)\s*$", body, re.MULTILINE)
        if not match.group(1).startswith(":")
    ]

    for rm in region_starts:
        name = rm.group(1)
        start = rm.start()
        try:
            end = _find_closing_paren(body, start) + 1
        except ValueError:
            continue
        inner = body[start:end]

        target_m = re.search(r"\(:target\s+(\w+)\)", inner)
        target = target_m.group(1) if target_m else ""
        region = RegionInfo(name=name, target=target)

        ranges_m = re.search(r"\(:ranges\s*\(\s*\(([^)]+)\)", inner)
        if ranges_m:
            vals = [float(v) for v in ranges_m.group(1).split()]
            if len(vals) == 4:
                region.x_min = vals[0]
                region.y_min = vals[1]
                region.x_max = vals[2]
                region.y_max = vals[3]

        yaw_m = re.search(r"\(:yaw_rotation\s*\(\s*\(([^)]+)\)", inner)
        if yaw_m:
            vals = [float(v) for v in yaw_m.group(1).split()]
            if len(vals) == 2:
                region.yaw_min = vals[0]
                region.yaw_max = vals[1]

        yield region


def _resolve_init_positions(content: str, cfg: TaskConfig) -> None:
    """Map objects to their initial placement regions from (:init ...) block.

    Handles three init patterns:
      - ``(On bowl_1 main_table_bowl_region)`` — object on a table region
      - ``(On bowl_1 plate_1)`` — object stacked on another movable object
      - ``(In bowl_1 cabinet_1_top_region)`` — object inside a container

    The region reference format is "{target}_{region_name}".
    """
    body = _extract_block(content, "init")
    if not body:
        return

    movable_names = {obj.instance_name for obj in cfg.movable_objects}
    fixture_names = {fixture.instance_name for fixture in cfg.fixtures}

    on_re = re.compile(r"\(On\s+(\w+)\s+(\w+)\)")
    in_re = re.compile(r"\(In\s+(\w+)\s+(\w+)\)")

    placements: dict[str, str] = {}
    stacking: dict[str, str] = {}  # obj_a → obj_b (obj_a is on obj_b)
    contained_in: dict[str, str] = {}  # obj_a -> container region ref

    for om in on_re.finditer(body):
        obj_name = om.group(1)
        target = om.group(2)
        if obj_name in movable_names and target in movable_names:
            # Object-on-object stacking (e.g., bowl on plate)
            stacking[obj_name] = target
        elif obj_name in movable_names or obj_name in fixture_names:
            placements[obj_name] = target
        else:
            continue

    for im in in_re.finditer(body):
        obj_name = im.group(1)
        if obj_name in movable_names:
            contained_in[obj_name] = im.group(2)

    # Build a lookup for resolved positions by instance name
    resolved: dict[str, tuple[float, float]] = {}

    for obj in cfg.movable_objects:
        if obj.instance_name in contained_in:
            obj.contained = True
            full_region = contained_in[obj.instance_name]
            rinfo = cfg.region_refs.get(full_region)
            if rinfo is not None:
                obj.region_name = rinfo.name
                obj.placement_target = rinfo.target
            # Objects inside fixtures are intentionally not given table x/y.
            continue

        full_region = placements.get(obj.instance_name)
        if not full_region:
            continue

        rinfo = cfg.region_refs.get(full_region)
        if rinfo is not None and rinfo.has_bounds:
            obj.region_name = rinfo.name
            obj.placement_target = rinfo.target
            centre = rinfo.centre
            if centre:
                obj.init_x, obj.init_y = centre
                obj.init_yaw = rinfo.yaw_centre
                resolved[obj.instance_name] = centre

    # Resolve stacking dependencies
    for obj in cfg.movable_objects:
        parent_name = stacking.get(obj.instance_name)
        if parent_name:
            obj.stacked_on = parent_name
            obj.placement_target = parent_name
            # Use parent's position as approximate init
            parent_pos = resolved.get(parent_name)
            if parent_pos:
                obj.init_x, obj.init_y = parent_pos
                parent_obj = next(
                    (
                        candidate
                        for candidate in cfg.movable_objects
                        if candidate.instance_name == parent_name
                    ),
                    None,
                )
                if parent_obj is not None:
                    obj.init_yaw = parent_obj.init_yaw

    for fixture in cfg.fixtures:
        full_region = placements.get(fixture.instance_name)
        if not full_region:
            continue

        rinfo = cfg.region_refs.get(full_region)
        if rinfo is not None and rinfo.has_bounds:
            fixture.region_name = rinfo.name
            fixture.placement_target = rinfo.target
            centre = rinfo.centre
            if centre:
                fixture.init_x, fixture.init_y = centre
                fixture.init_yaw = rinfo.yaw_centre
