"""Utilities for auditing generated perturbations and visible changes."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np

from libero_infinity.task_config import (
    _IMMOBILE_WORKSPACE_FIXTURES,
    FixtureInfo,
    ObjectInfo,
    TaskConfig,
    _support_parent_names,
)

_HARD_REQUIRE_RE = re.compile(r"^\s*require(?!\[)\s+(?P<body>.+)$")
_SOFT_REQUIRE_RE = re.compile(r"^\s*require\[(?P<weight>[^\]]+)\]\s+(?P<body>.+)$")
_TEMPORAL_OPERATOR_RE = re.compile(r"\b(always|eventually|until|next|monitor|implies)\b")


@dataclass(frozen=True)
class ConstraintAudit:
    hard_require_total: int = 0
    soft_require_total: int = 0
    hard_axis_clearance: int = 0
    hard_distance_clearance: int = 0
    soft_ood_bias: int = 0
    temporal_require_total: int = 0
    temporal_operators: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NumericSummary:
    count: int
    mean: float | None
    median: float | None
    p10: float | None
    p90: float | None
    minimum: float | None
    maximum: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnchorPixelRecord:
    """Pixel-space anchor observations for canonical and perturbed renders."""

    name: str
    canonical_pixel: tuple[float, float] | None = None
    perturbed_pixel: tuple[float, float] | None = None
    canonical_visible: bool | None = None
    perturbed_visible: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnchorPixelSummary:
    """Aggregate anchor visibility and displacement statistics."""

    anchor_count: int
    canonical_visible_fraction: float | None
    perturbed_visible_fraction: float | None
    canonical_in_frame_fraction: float | None
    perturbed_in_frame_fraction: float | None
    matched_fraction: float | None
    mean_displacement_px: float | None
    max_displacement_px: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisibleChangeScoreConfig:
    """Configuration for deterministic visible-change scoring."""

    rgb_delta_material_threshold: float = 0.015
    anchor_displacement_reference_px: float = 12.0
    anchor_motion_material_threshold_px: float = 6.0
    minimum_perturbed_visibility_fraction: float = 0.5
    minimum_perturbed_in_frame_fraction: float = 0.5
    rgb_weight: float = 0.55
    anchor_displacement_weight: float = 0.30
    anchor_visibility_weight: float = 0.15
    combined_material_threshold: float = 0.35
    vlm_ambiguity_lower: float = 0.25
    vlm_ambiguity_upper: float = 0.75


@dataclass(frozen=True)
class VisibleChangeScore:
    """Combined RGB and anchor-based visible-change signals."""

    rgb_mean_delta: float
    rgb_score: float
    anchor_summary: AnchorPixelSummary
    anchor_displacement_score: float
    anchor_visibility_score: float
    combined_score: float
    material_rgb_change: bool
    material_anchor_motion: bool
    anchor_visibility_ok: bool
    material_visible_change: bool
    should_run_vlm_check: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["anchor_summary"] = self.anchor_summary.to_dict()
        return payload


def analyze_generated_constraints(scenic_code: str) -> ConstraintAudit:
    """Classify hard/soft constraints in a generated Scenic program."""
    hard_total = 0
    soft_total = 0
    hard_axis = 0
    hard_distance = 0
    soft_ood = 0
    temporal_total = 0
    temporal_ops: set[str] = set()

    for raw_line in scenic_code.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        soft_match = _SOFT_REQUIRE_RE.match(line)
        if soft_match:
            soft_total += 1
            body = soft_match.group("body")
            if "distance from" in body and "> _ood_margin" in body:
                soft_ood += 1
            matches = set(_TEMPORAL_OPERATOR_RE.findall(body))
            if matches:
                temporal_total += 1
                temporal_ops.update(matches)
            continue

        hard_match = _HARD_REQUIRE_RE.match(line)
        if not hard_match:
            continue

        hard_total += 1
        body = hard_match.group("body")
        if "abs(" in body and (".position.x" in body or ".position.y" in body):
            hard_axis += 1
        elif "distance from" in body:
            hard_distance += 1

        matches = set(_TEMPORAL_OPERATOR_RE.findall(body))
        if matches:
            temporal_total += 1
            temporal_ops.update(matches)

    return ConstraintAudit(
        hard_require_total=hard_total,
        soft_require_total=soft_total,
        hard_axis_clearance=hard_axis,
        hard_distance_clearance=hard_distance,
        soft_ood_bias=soft_ood,
        temporal_require_total=temporal_total,
        temporal_operators=tuple(sorted(temporal_ops)),
    )


def canonical_xy_for_object(cfg: TaskConfig, obj: ObjectInfo) -> tuple[float, float] | None:
    """Return the canonical xy anchor for an object, including contained cases."""
    if obj.init_x is not None and obj.init_y is not None:
        return (float(obj.init_x), float(obj.init_y))

    if not obj.region_name:
        return _fallback_support_anchor_xy(cfg, obj)

    region = cfg.regions.get(obj.region_name)
    if region is None or not region.has_bounds:
        return _fallback_support_anchor_xy(cfg, obj)
    if obj.placement_target and region.target != obj.placement_target:
        return _fallback_support_anchor_xy(cfg, obj)
    centre = region.centre
    if centre is None:
        return _fallback_support_anchor_xy(cfg, obj)
    return (float(centre[0]), float(centre[1]))


def moving_support_names(
    cfg: TaskConfig,
) -> tuple[set[str], set[str], dict[str, str | None]]:
    """Return movable support fixture names, movable support object names, parent map."""
    fixture_by_name = {fixture.instance_name: fixture for fixture in cfg.fixtures}
    support_fixture_names = {
        obj.placement_target
        for obj in cfg.movable_objects
        if obj.placement_target in fixture_by_name
        if fixture_by_name[obj.placement_target].fixture_class not in _IMMOBILE_WORKSPACE_FIXTURES
    }
    moving_fixture_names = cfg.goal_fixture_names | support_fixture_names
    support_parent_map = _support_parent_names(
        cfg.movable_objects,
        moving_fixture_names=moving_fixture_names,
    )
    movable_names = {obj.instance_name for obj in cfg.movable_objects}
    movable_support_names = {
        parent_name for parent_name in support_parent_map.values() if parent_name in movable_names
    }
    return moving_fixture_names, movable_support_names, support_parent_map


def object_displacements(
    cfg: TaskConfig,
    scene_objects: list[Any],
) -> dict[str, float]:
    """Compute xy displacement of each movable object from its canonical pose."""
    positions = _scene_xy_positions(scene_objects)
    displacements: dict[str, float] = {}
    for obj in cfg.movable_objects:
        if obj.instance_name not in positions:
            continue
        canonical_xy = canonical_xy_for_object(cfg, obj)
        if canonical_xy is None:
            continue
        displacements[obj.instance_name] = _xy_distance(positions[obj.instance_name], canonical_xy)
    return displacements


def support_displacements(
    cfg: TaskConfig,
    scene_objects: list[Any],
) -> dict[str, float]:
    """Compute xy displacement of movable support anchors from canonical pose."""
    positions = _scene_xy_positions(scene_objects)
    moving_fixture_names, movable_support_names, _support_parent_map = moving_support_names(cfg)

    displacements: dict[str, float] = {}
    fixture_by_name = {fixture.instance_name: fixture for fixture in cfg.fixtures}
    for fixture_name in moving_fixture_names:
        fixture = fixture_by_name.get(fixture_name)
        if (
            fixture is None
            or fixture.init_x is None
            or fixture.init_y is None
            or fixture_name not in positions
        ):
            continue
        displacements[fixture_name] = _xy_distance(
            positions[fixture_name],
            (float(fixture.init_x), float(fixture.init_y)),
        )

    object_by_name = {obj.instance_name: obj for obj in cfg.movable_objects}
    for object_name in movable_support_names:
        obj = object_by_name.get(object_name)
        if obj is None or object_name not in positions:
            continue
        canonical_xy = canonical_xy_for_object(cfg, obj)
        if canonical_xy is None:
            continue
        displacements[object_name] = _xy_distance(positions[object_name], canonical_xy)

    return displacements


def mean_absolute_image_delta(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Return mean absolute RGB delta in [0, 1]."""
    if frame_a.shape != frame_b.shape:
        raise ValueError(f"Image shapes differ: {frame_a.shape} vs {frame_b.shape}")
    diff = np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32)) / 255.0
    return float(diff.mean())


def parse_anchor_pixel_records(payload: object) -> list[AnchorPixelRecord]:
    """Parse JSON-like anchor payloads into typed records."""
    if isinstance(payload, Mapping):
        if "name" in payload:
            return [_parse_anchor_pixel_record(payload)]
        return [
            _parse_anchor_pixel_record({**value, "name": name}) for name, value in payload.items()
        ]

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [_parse_anchor_pixel_record(item) for item in payload]

    raise ValueError("Anchor payload must be a mapping or a sequence of mappings")


def summarize_anchor_pixel_records(
    anchor_records: Sequence[AnchorPixelRecord],
    frame_shape: Sequence[int],
) -> AnchorPixelSummary:
    """Summarize anchor visibility/in-frame fractions and pixel displacement."""
    if len(frame_shape) < 2:
        raise ValueError(f"Expected image-like frame shape, got {tuple(frame_shape)!r}")

    total = len(anchor_records)
    if total == 0:
        return AnchorPixelSummary(
            anchor_count=0,
            canonical_visible_fraction=None,
            perturbed_visible_fraction=None,
            canonical_in_frame_fraction=None,
            perturbed_in_frame_fraction=None,
            matched_fraction=None,
            mean_displacement_px=None,
            max_displacement_px=None,
        )

    height = int(frame_shape[0])
    width = int(frame_shape[1])
    canonical_visible = 0
    perturbed_visible = 0
    canonical_in_frame = 0
    perturbed_in_frame = 0
    matched = 0
    displacements: list[float] = []

    for record in anchor_records:
        canonical_is_visible = _resolve_visibility(record.canonical_visible, record.canonical_pixel)
        perturbed_is_visible = _resolve_visibility(record.perturbed_visible, record.perturbed_pixel)

        canonical_visible += int(canonical_is_visible)
        perturbed_visible += int(perturbed_is_visible)

        canonical_is_in_frame = canonical_is_visible and _pixel_in_frame(
            record.canonical_pixel,
            width=width,
            height=height,
        )
        perturbed_is_in_frame = perturbed_is_visible and _pixel_in_frame(
            record.perturbed_pixel,
            width=width,
            height=height,
        )

        canonical_in_frame += int(canonical_is_in_frame)
        perturbed_in_frame += int(perturbed_is_in_frame)

        if canonical_is_visible and perturbed_is_visible:
            if record.canonical_pixel is not None and record.perturbed_pixel is not None:
                matched += 1
                displacements.append(_xy_distance(record.canonical_pixel, record.perturbed_pixel))

    return AnchorPixelSummary(
        anchor_count=total,
        canonical_visible_fraction=_fraction(canonical_visible, total),
        perturbed_visible_fraction=_fraction(perturbed_visible, total),
        canonical_in_frame_fraction=_fraction(canonical_in_frame, total),
        perturbed_in_frame_fraction=_fraction(perturbed_in_frame, total),
        matched_fraction=_fraction(matched, total),
        mean_displacement_px=float(mean(displacements)) if displacements else None,
        max_displacement_px=float(max(displacements)) if displacements else None,
    )


def score_visible_change(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    anchor_records: Sequence[AnchorPixelRecord] = (),
    *,
    config: VisibleChangeScoreConfig | None = None,
) -> VisibleChangeScore:
    """Combine RGB delta with anchor displacement and visibility checks."""
    config = config or VisibleChangeScoreConfig()
    rgb_mean_delta = mean_absolute_image_delta(frame_a, frame_b)
    anchor_summary = summarize_anchor_pixel_records(anchor_records, frame_b.shape)

    rgb_score = _normalize_score(rgb_mean_delta, config.rgb_delta_material_threshold)
    anchor_displacement_score = _normalize_score(
        anchor_summary.mean_displacement_px,
        config.anchor_displacement_reference_px,
    )
    anchor_visibility_score = (
        _mean_defined(
            (
                anchor_summary.perturbed_visible_fraction,
                anchor_summary.perturbed_in_frame_fraction,
            )
        )
        or 0.0
    )

    total_weight = (
        config.rgb_weight + config.anchor_displacement_weight + config.anchor_visibility_weight
    )
    if total_weight <= 0.0:
        raise ValueError("VisibleChangeScoreConfig weights must sum to a positive value")

    combined_score = (
        config.rgb_weight * rgb_score
        + config.anchor_displacement_weight * anchor_displacement_score
        + config.anchor_visibility_weight * anchor_visibility_score
    ) / total_weight

    material_rgb_change = rgb_mean_delta >= config.rgb_delta_material_threshold
    material_anchor_motion = (
        anchor_summary.mean_displacement_px is not None
        and anchor_summary.mean_displacement_px >= config.anchor_motion_material_threshold_px
    )
    anchor_visibility_ok = anchor_summary.anchor_count == 0 or (
        (anchor_summary.perturbed_visible_fraction or 0.0)
        >= config.minimum_perturbed_visibility_fraction
        and (anchor_summary.perturbed_in_frame_fraction or 0.0)
        >= config.minimum_perturbed_in_frame_fraction
    )
    material_visible_change = (
        combined_score >= config.combined_material_threshold
        and anchor_visibility_ok
        and (material_rgb_change or material_anchor_motion)
    )
    should_run_vlm_check = (
        config.vlm_ambiguity_lower <= combined_score <= config.vlm_ambiguity_upper
        or material_rgb_change != material_anchor_motion
        or not anchor_visibility_ok
    )

    return VisibleChangeScore(
        rgb_mean_delta=rgb_mean_delta,
        rgb_score=rgb_score,
        anchor_summary=anchor_summary,
        anchor_displacement_score=anchor_displacement_score,
        anchor_visibility_score=anchor_visibility_score,
        combined_score=combined_score,
        material_rgb_change=material_rgb_change,
        material_anchor_motion=material_anchor_motion,
        anchor_visibility_ok=anchor_visibility_ok,
        material_visible_change=material_visible_change,
        should_run_vlm_check=should_run_vlm_check,
    )


def summarize_numeric(values: list[float]) -> NumericSummary:
    """Return basic summary statistics for a list of floats."""
    if not values:
        return NumericSummary(
            count=0,
            mean=None,
            median=None,
            p10=None,
            p90=None,
            minimum=None,
            maximum=None,
        )

    ordered = sorted(float(value) for value in values)
    return NumericSummary(
        count=len(ordered),
        mean=float(mean(ordered)),
        median=float(median(ordered)),
        p10=_percentile(ordered, 0.10),
        p90=_percentile(ordered, 0.90),
        minimum=float(ordered[0]),
        maximum=float(ordered[-1]),
    )


def fixture_canonical_xy(fixture: FixtureInfo) -> tuple[float, float] | None:
    if fixture.init_x is None or fixture.init_y is None:
        return None
    return (float(fixture.init_x), float(fixture.init_y))


def _scene_xy_positions(scene_objects: list[Any]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for obj in scene_objects:
        libero_name = getattr(obj, "libero_name", "")
        if not libero_name:
            continue
        positions[libero_name] = (float(obj.position.x), float(obj.position.y))
    return positions


def _xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _parse_anchor_pixel_record(payload: object) -> AnchorPixelRecord:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping for anchor record, got {type(payload).__name__}")
    if "name" not in payload:
        raise ValueError("Anchor record is missing required field 'name'")
    name = str(payload["name"])
    return AnchorPixelRecord(
        name=name,
        canonical_pixel=_coerce_pixel(payload.get("canonical_pixel")),
        perturbed_pixel=_coerce_pixel(payload.get("perturbed_pixel")),
        canonical_visible=_coerce_optional_bool(payload.get("canonical_visible")),
        perturbed_visible=_coerce_optional_bool(payload.get("perturbed_visible")),
    )


def _coerce_pixel(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return (float(value["x"]), float(value["y"]))
        if "u" in value and "v" in value:
            return (float(value["u"]), float(value["v"]))
        raise ValueError("Pixel mappings must contain either x/y or u/v")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 2:
            raise ValueError(f"Expected pixel pair, got {value!r}")
        return (float(value[0]), float(value[1]))
    raise ValueError(f"Unsupported pixel payload: {value!r}")


def _coerce_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"Expected bool or None, got {value!r}")


def _resolve_visibility(
    explicit_visible: bool | None,
    pixel: tuple[float, float] | None,
) -> bool:
    if explicit_visible is not None:
        return explicit_visible
    return pixel is not None


def _pixel_in_frame(
    pixel: tuple[float, float] | None,
    *,
    width: int,
    height: int,
) -> bool:
    if pixel is None:
        return False
    return 0.0 <= pixel[0] < float(width) and 0.0 <= pixel[1] < float(height)


def _fraction(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _normalize_score(value: float | None, reference: float) -> float:
    if value is None or reference <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(reference)))


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None
    return float(mean(defined))


def _fallback_support_anchor_xy(
    cfg: TaskConfig,
    obj: ObjectInfo,
) -> tuple[float, float] | None:
    if obj.placement_target:
        for fixture in cfg.fixtures:
            if fixture.instance_name == obj.placement_target:
                return fixture_canonical_xy(fixture)
        for candidate in cfg.movable_objects:
            if candidate.instance_name == obj.placement_target:
                return canonical_xy_for_object(cfg, candidate)
    if obj.stacked_on:
        for candidate in cfg.movable_objects:
            if candidate.instance_name == obj.stacked_on:
                return canonical_xy_for_object(cfg, candidate)
    return None


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty list")
    if len(values) == 1:
        return float(values[0])
    index = (len(values) - 1) * q
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return float(values[lo])
    frac = index - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


class GeminiVisibilityChecker:
    """Optional LiteLLM helper for curated visibility / occlusion audits."""

    def __init__(
        self,
        *,
        model: str = "vertex/gemini-3-flash-preview",
        project: str | None = None,
        location: str = "global",
    ) -> None:
        self.model = model
        self.project = project
        self.location = location

    def describe_visibility(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> str:
        try:
            import base64

            import litellm
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LiteLLM visibility checks require litellm to be installed") from exc

        extra_body = {"vertex_location": self.location}
        if self.project:
            extra_body["vertex_project"] = self.project
        data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        response = litellm.completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            extra_body=extra_body,
        )
        return str(response.choices[0].message.content).strip()
