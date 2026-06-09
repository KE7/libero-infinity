"""Runtime loaders for externalized semantic registries (WS-4 hardcoding audit).

Three semantic registries that were previously hard-coded are now served from
``data/*.json`` so they stop drifting from the asset set and from each other:

* ``surface_compatibility.json`` — physical-type taxonomy + forbidden
  (category, surface) pairs for counterfactual (CF) placement plausibility
  (was ``bddl_preprocessor._VISUAL_CATEGORY`` + ``_INCOMPATIBLE``).
* ``cf_category_groups.json`` — visual-similarity groups for CF grounding
  difficulty (harmonizes ``bddl_preprocessor._CF_CATEGORY`` and
  ``scripts/rate_cf_bddls.py:_CATEGORY``).
* ``support_scale_factors.json`` — per-support-type local-perturbation scale
  factors (was ``perturbation_policy_helpers._SUPPORT_SCALE_BY_TYPE``).

Every loader is *fallback-safe*: if the data file is missing or malformed it
returns ``None`` and the caller keeps using its in-code dict, so behavior is
unchanged when the file is absent.
"""

from __future__ import annotations

import json
import pkgutil
from typing import Optional


def _load_json(filename: str) -> Optional[dict]:
    """Return parsed ``data/<filename>`` or ``None`` if absent/malformed."""
    try:
        raw = pkgutil.get_data("libero_infinity", f"data/{filename}")
    except (OSError, ModuleNotFoundError):
        return None
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def load_cf_category_groups() -> Optional[dict[str, str]]:
    """Return the harmonized ``{asset: cf_category}`` map, or ``None``."""
    data = _load_json("cf_category_groups.json")
    if data is None:
        return None
    groups = data.get("cf_category")
    if not isinstance(groups, dict) or not groups:
        return None
    return {str(k): str(v) for k, v in groups.items()}


def load_surface_compatibility() -> Optional[tuple[dict[str, str], set[tuple[str, str]]]]:
    """Return ``(physical_category, incompatible_pairs)``, or ``None``.

    ``physical_category`` maps ``{asset: physical_type}``; ``incompatible_pairs``
    is a set of ``(category, surface)`` tuples that forbid a CF placement.
    """
    data = _load_json("surface_compatibility.json")
    if data is None:
        return None
    categories = data.get("physical_category")
    pairs = data.get("incompatible")
    if not isinstance(categories, dict) or not categories:
        return None
    if not isinstance(pairs, list):
        return None
    incompatible: set[tuple[str, str]] = set()
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        incompatible.add((str(pair[0]), str(pair[1])))
    category_map = {str(k): str(v) for k, v in categories.items()}
    return category_map, incompatible


def load_support_scale_factors() -> Optional[dict[str, tuple[float, float]]]:
    """Return ``{support_type: (scale_x, scale_y)}``, or ``None``."""
    data = _load_json("support_scale_factors.json")
    if data is None:
        return None
    scales = data.get("support_scale_by_type")
    if not isinstance(scales, dict) or not scales:
        return None
    resolved: dict[str, tuple[float, float]] = {}
    for key, value in scales.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            resolved[str(key)] = (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    return resolved
