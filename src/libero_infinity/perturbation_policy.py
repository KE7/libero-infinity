"""Policy helpers for richer support-preserving position perturbations."""

from __future__ import annotations

_SUPPORT_SCALE_BY_CLASS: dict[str, tuple[float, float]] = {
    "plate": (0.55, 0.55),
    "tray": (0.65, 0.65),
    "wooden_tray": (0.65, 0.65),
    "basket": (0.60, 0.60),
    "desk_caddy": (0.60, 0.60),
    "microwave": (0.55, 0.55),
    "wooden_cabinet": (0.58, 0.52),
    "white_cabinet": (0.58, 0.52),
    "flat_stove": (0.70, 0.70),
    "wine_rack": (0.45, 0.45),
    "wooden_two_layer_shelf": (0.55, 0.45),
}

_SUPPORT_SCALE_BY_REGION_KEYWORD: dict[str, tuple[float, float]] = {
    "cook_region": (0.72, 0.72),
    "heating_region": (0.45, 0.45),
    "contain_region": (0.55, 0.55),
    "top_region": (0.55, 0.45),
    "middle_region": (0.52, 0.42),
    "bottom_region": (0.52, 0.42),
    "top_side": (0.62, 0.52),
}

_YAW_SPAN_BY_CLASS: dict[str, float] = {
    "plate": 0.45,
    "bowl": 0.55,
    "mug": 0.65,
    "wine_bottle": 0.45,
    "bottle": 0.45,
    "stove": 0.18,
    "cabinet": 0.12,
    "microwave": 0.14,
    "rack": 0.18,
    "tray": 0.35,
}


def support_offset_bounds(
    *,
    support_dims: tuple[float, float, float],
    child_dims: tuple[float, float, float],
    support_class: str | None,
    region_name: str | None,
    contained: bool,
) -> tuple[float, float]:
    """Compute local x/y envelopes for a child supported by another entity."""
    support_w, support_l, _support_h = support_dims
    child_w, child_l, _child_h = child_dims
    base_x = max((support_w - child_w) / 2.0, 0.0)
    base_y = max((support_l - child_l) / 2.0, 0.0)

    scale_x, scale_y = _support_scale(support_class=support_class, region_name=region_name)
    if contained:
        scale_x *= 0.85
        scale_y *= 0.85
    return (base_x * scale_x, base_y * scale_y)


def yaw_bounds(
    *,
    canonical_yaw: float | None,
    asset_class: str,
    support_class: str | None = None,
) -> tuple[float, float] | None:
    """Return a safe yaw interval around the canonical yaw."""
    centre = 0.0 if canonical_yaw is None else float(canonical_yaw)
    key = _yaw_key(asset_class)
    if support_class:
        key = _yaw_key(support_class) or key
    span = _YAW_SPAN_BY_CLASS.get(key)
    if span is None:
        span = 0.30 if _looks_round(asset_class) else 0.18
    return (centre - span, centre + span)


def coordinated_group_offset(
    *,
    member_count: int,
    support_dims: tuple[float, float, float],
) -> tuple[float, float]:
    """Shared translation range for multiple objects on the same support."""
    if member_count <= 1:
        return (0.0, 0.0)
    support_w, support_l, _support_h = support_dims
    scale = min(0.18 + 0.04 * member_count, 0.30)
    return (support_w * scale / 2.0, support_l * scale / 2.0)


def _support_scale(*, support_class: str | None, region_name: str | None) -> tuple[float, float]:
    if region_name:
        for key, scale in _SUPPORT_SCALE_BY_REGION_KEYWORD.items():
            if key in region_name:
                return scale
    if support_class:
        support_key = _yaw_key(support_class)
        if support_key in _SUPPORT_SCALE_BY_CLASS:
            return _SUPPORT_SCALE_BY_CLASS[support_key]
        if support_class in _SUPPORT_SCALE_BY_CLASS:
            return _SUPPORT_SCALE_BY_CLASS[support_class]
    return (0.50, 0.50)


def _yaw_key(name: str) -> str:
    lowered = name.lower()
    for key in _YAW_SPAN_BY_CLASS:
        if key in lowered:
            return key
    return lowered


def _looks_round(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("bowl", "plate", "mug", "cup"))
