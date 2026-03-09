"""Asset variant registry — single source of truth loaded from data/asset_variants.json.

Maps each LIBERO canonical object class to a list of OOD visual variants.
The Scenic object perturbation program draws from these lists uniformly.

Convention: the first entry in each list is the canonical (training) object.
All subsequent entries are OOD variants for evaluation.

Geometric difficulty levels (subjective):
  EASY   — same shape, different color/texture
  MEDIUM — similar shape, visibly different
  HARD   — same functional category, distinct geometry
"""

from __future__ import annotations

import json
import pkgutil

_raw = pkgutil.get_data("libero_infinity", "data/asset_variants.json")
assert _raw is not None, "asset_variants.json not found in package data"
_registry = json.loads(_raw)

ASSET_VARIANTS: dict[str, list[str]] = _registry["variants"]
OBJECT_DIMENSIONS: dict[str, list[float]] = _registry.get("dimensions", {})
UNLOADABLE_ASSET_CLASSES: frozenset[str] = frozenset({"cherries", "corn", "mayo"})


def get_variants(
    object_class: str,
    include_canonical: bool = True,
    require_loadable: bool = False,
) -> list[str]:
    """Return OOD variant list for an object class.

    Args:
        object_class: BDDL object type name, e.g. "akita_black_bowl".
        include_canonical: If False, strip the canonical first entry.
        require_loadable: If True, filter out known classes whose MuJoCo XML
            assets are not available in the bundled LIBERO runtime.

    Returns:
        List of asset class strings usable as BDDL object types.
    """
    variants = ASSET_VARIANTS.get(object_class, [object_class])
    if require_loadable:
        filtered = [v for v in variants if v not in UNLOADABLE_ASSET_CLASSES]
        if filtered:
            variants = filtered
    if not include_canonical and len(variants) > 1:
        return variants[1:]
    return variants


def has_variants(object_class: str) -> bool:
    """Return True if the object class has at least one OOD variant."""
    return len(get_variants(object_class, include_canonical=False)) > 0


def get_dimensions(object_class: str) -> tuple[float, float, float]:
    """Return (width, length, height) in metres for the given object class.

    Falls back to a conservative default if not in the registry.
    """
    dims = OBJECT_DIMENSIONS.get(object_class, [0.08, 0.08, 0.06])
    return (dims[0], dims[1], dims[2])


# Flat set of all object classes that appear in any LIBERO suite
ALL_LIBERO_CLASSES: frozenset[str] = frozenset(ASSET_VARIANTS.keys())

# Default pool of small graspable objects suitable as distractors.
# Loaded from the canonical "distractor_pool" key in asset_variants.json so
# that asset_registry.py and the Scenic model stay in sync automatically.
DEFAULT_DISTRACTOR_POOL: list[str] = list(_registry.get("distractor_pool", []))


def get_distractor_pool(
    exclude_classes: set[str] | None = None,
    custom_pool: list[str] | None = None,
) -> list[str]:
    """Return a list of valid distractor object classes.

    Args:
        exclude_classes: Classes to exclude (e.g., task objects already in scene).
        custom_pool: Override the default pool with a custom list.

    Returns:
        List of asset class names valid for use as distractors.
    """
    pool = list(custom_pool) if custom_pool else list(DEFAULT_DISTRACTOR_POOL)
    if exclude_classes:
        pool = [c for c in pool if c not in exclude_classes]
    return pool
