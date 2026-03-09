"""Helpers for richer position perturbation policies.

This module is intentionally standalone so new placement policies can be
developed and tested without editing the Scenic generator or simulator.
"""

from __future__ import annotations

import math
import pathlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping

from libero_infinity.bddl_preprocessor import _extract_block, _find_closing_paren
from libero_infinity.task_config import (
    _IMMOBILE_WORKSPACE_FIXTURES,
    FixtureInfo,
    ObjectInfo,
    TaskConfig,
)

_REGION_START_RE = re.compile(r"^\s*\(([^:\s()]+)\s*$", re.MULTILINE)
_YAW_ROTATION_RE = re.compile(
    r"\(:yaw_rotation\s*\(\s*\(\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)\s*\)\s*\)\s*\)"
)

_SUPPORT_SCALE_BY_TYPE: dict[str, tuple[float, float]] = {
    "contained": (0.45, 0.45),
    "cook_surface": (0.75, 0.75),
    "shelf_surface": (0.60, 0.60),
    "workspace": (0.60, 0.60),
    "object_surface": (0.60, 0.60),
}

_CONTAINER_SUPPORT_CLASSES = frozenset(
    {
        "basket",
        "desk_caddy",
        "microwave",
    }
)

_SHELF_SUPPORT_CLASSES = frozenset(
    {
        "bowl_drainer",
        "white_cabinet",
        "wine_rack",
        "wooden_two_layer_shelf",
        "wooden_cabinet",
    }
)


@dataclass(frozen=True)
class NumericRange:
    """Closed interval used for sampling policy parameters."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("Range bounds must be finite")
        if self.maximum < self.minimum:
            raise ValueError(
                f"Invalid range [{self.minimum}, {self.maximum}] with maximum < minimum"
            )

    @property
    def span(self) -> float:
        return self.maximum - self.minimum

    def sample(self, rng: random.Random) -> float:
        if self.minimum == self.maximum:
            return self.minimum
        return rng.uniform(self.minimum, self.maximum)


@dataclass(frozen=True)
class YawRange(NumericRange):
    """Yaw interval in radians."""


@dataclass(frozen=True)
class LocalEnvelope:
    """Support-relative local perturbation envelope."""

    x_half_extent: float
    y_half_extent: float
    support_type: str


@dataclass(frozen=True)
class GroupTransform:
    """Shared transform applied to a support/workspace group."""

    translation: tuple[float, float]
    shared_yaw: float | None = None
    local_jitter: dict[str, tuple[float, float]] = field(default_factory=dict)


def parse_region_yaw_ranges_from_text(bddl_text: str) -> dict[str, YawRange]:
    """Extract per-region yaw ranges from a BDDL ``:regions`` block."""
    body = _extract_block(bddl_text, "regions")
    if not body:
        return {}

    ranges: dict[str, YawRange] = {}
    for match in _REGION_START_RE.finditer(body):
        name = match.group(1)
        if name.startswith(":"):
            continue
        start = match.start()
        try:
            end = _find_closing_paren(body, start) + 1
        except ValueError:
            continue
        inner = body[start:end]
        yaw_match = _YAW_ROTATION_RE.search(inner)
        if yaw_match is None:
            continue
        lo = float(yaw_match.group(1))
        hi = float(yaw_match.group(2))
        minimum = min(lo, hi)
        maximum = max(lo, hi)
        ranges[name] = YawRange(minimum, maximum)
    return ranges


def parse_region_yaw_ranges_from_file(path: str | pathlib.Path) -> dict[str, YawRange]:
    """Load a BDDL file and extract its region yaw ranges."""
    return parse_region_yaw_ranges_from_text(pathlib.Path(path).read_text())


def resolve_object_yaw_ranges(
    cfg: TaskConfig,
    *,
    default: YawRange | None = None,
) -> dict[str, YawRange | None]:
    """Return the yaw range for each movable object when one can be inferred.

    Objects inherit their range from the region referenced by the BDDL init
    predicate. If the referenced region has no ``:yaw_rotation`` metadata, the
    optional ``default`` is returned instead.
    """
    region_yaws = parse_region_yaw_ranges_from_file(cfg.bddl_path)
    resolved: dict[str, YawRange | None] = {}
    for obj in cfg.movable_objects:
        resolved[obj.instance_name] = region_yaws.get(obj.region_name or "", default)
    return resolved


def infer_support_type(
    *,
    support_class: str | None,
    region_name: str | None,
    contained: bool,
) -> str:
    """Classify the support type for local envelope sizing."""
    region_name = (region_name or "").lower()
    support_class = (support_class or "").lower()

    if contained or "contain" in region_name or support_class in _CONTAINER_SUPPORT_CLASSES:
        return "contained"
    if region_name == "cook_region" or support_class == "flat_stove":
        return "cook_surface"
    if support_class in _IMMOBILE_WORKSPACE_FIXTURES:
        return "workspace"
    if support_class in _SHELF_SUPPORT_CLASSES or region_name in {
        "top_side",
        "top_region",
        "middle_region",
        "bottom_region",
        "left_region",
        "right_region",
    }:
        return "shelf_surface"
    return "object_surface"


def support_local_envelope(
    *,
    support_dims: tuple[float, float, float],
    child_dims: tuple[float, float, float],
    support_class: str | None,
    region_name: str | None,
    contained: bool,
    clearance_margin: float = 0.02,
    support_scales: Mapping[str, tuple[float, float]] | None = None,
) -> LocalEnvelope:
    """Return support-relative xy half-extents for local perturbations."""
    support_type = infer_support_type(
        support_class=support_class,
        region_name=region_name,
        contained=contained,
    )
    scales = dict(_SUPPORT_SCALE_BY_TYPE)
    if support_scales:
        scales.update(support_scales)

    usable_x = max(0.0, support_dims[0] - child_dims[0] - clearance_margin)
    usable_y = max(0.0, support_dims[1] - child_dims[1] - clearance_margin)
    scale_x, scale_y = scales[support_type]

    region = (region_name or "").lower()
    if "left" in region or "right" in region:
        scale_x *= 0.5
    if "front" in region or "back" in region:
        scale_y *= 0.5

    return LocalEnvelope(
        x_half_extent=max(0.0, usable_x * scale_x / 2.0),
        y_half_extent=max(0.0, usable_y * scale_y / 2.0),
        support_type=support_type,
    )


def coordination_groups(cfg: TaskConfig) -> dict[str, tuple[ObjectInfo, ...]]:
    """Group movable objects by their shared support or root workspace."""
    fixture_by_name = {fixture.instance_name: fixture for fixture in cfg.fixtures}
    movable_names = {obj.instance_name for obj in cfg.movable_objects}
    groups: dict[str, list[ObjectInfo]] = defaultdict(list)

    for obj in cfg.movable_objects:
        groups[_coordination_group_key(obj, movable_names, fixture_by_name)].append(obj)

    return {key: tuple(group) for key, group in groups.items()}


def sample_group_transform(
    object_names: list[str],
    *,
    translation_x_range: NumericRange,
    translation_y_range: NumericRange,
    rng: random.Random | None = None,
    shared_yaw_range: YawRange | None = None,
    local_jitter_range: tuple[NumericRange, NumericRange] | None = None,
    local_jitter_ranges: Mapping[str, tuple[NumericRange, NumericRange]] | None = None,
) -> GroupTransform:
    """Sample a coordinated transform for objects sharing a support/workspace."""
    rng = rng or random.Random()
    local_jitter_ranges = local_jitter_ranges or {}
    default_jitter = local_jitter_range or (
        NumericRange(0.0, 0.0),
        NumericRange(0.0, 0.0),
    )

    jitter: dict[str, tuple[float, float]] = {}
    for name in object_names:
        x_range, y_range = local_jitter_ranges.get(name, default_jitter)
        jitter[name] = (x_range.sample(rng), y_range.sample(rng))

    shared_yaw = None
    if shared_yaw_range is not None:
        shared_yaw = shared_yaw_range.sample(rng)

    return GroupTransform(
        translation=(
            translation_x_range.sample(rng),
            translation_y_range.sample(rng),
        ),
        shared_yaw=shared_yaw,
        local_jitter=jitter,
    )


def apply_group_transform(
    canonical_positions: Mapping[str, tuple[float, float]],
    *,
    anchor_xy: tuple[float, float],
    transform: GroupTransform,
) -> dict[str, tuple[float, float]]:
    """Apply a shared translation/yaw plus per-object jitter to xy positions."""
    cos_yaw = 1.0
    sin_yaw = 0.0
    if transform.shared_yaw is not None:
        cos_yaw = math.cos(transform.shared_yaw)
        sin_yaw = math.sin(transform.shared_yaw)

    transformed: dict[str, tuple[float, float]] = {}
    for name, (x, y) in canonical_positions.items():
        dx = x - anchor_xy[0]
        dy = y - anchor_xy[1]
        rot_x = (dx * cos_yaw) - (dy * sin_yaw)
        rot_y = (dx * sin_yaw) + (dy * cos_yaw)
        jitter_x, jitter_y = transform.local_jitter.get(name, (0.0, 0.0))
        transformed[name] = (
            anchor_xy[0] + transform.translation[0] + rot_x + jitter_x,
            anchor_xy[1] + transform.translation[1] + rot_y + jitter_y,
        )
    return transformed


def _coordination_group_key(
    obj: ObjectInfo,
    movable_names: set[str],
    fixture_by_name: Mapping[str, FixtureInfo],
) -> str:
    if obj.stacked_on and obj.stacked_on in movable_names:
        return f"support:{obj.stacked_on}"
    if obj.placement_target in movable_names:
        return f"support:{obj.placement_target}"

    fixture = fixture_by_name.get(obj.placement_target or "")
    if fixture is None:
        return "workspace:<unknown>"
    if fixture.fixture_class in _IMMOBILE_WORKSPACE_FIXTURES:
        return f"workspace:{fixture.instance_name}"
    return f"support:{fixture.instance_name}"
