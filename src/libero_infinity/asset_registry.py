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

import functools
import json
import pathlib
import pkgutil
import re

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


# ---------------------------------------------------------------------------
# Asset affordance probe: which classes expose a ``contain_region`` site?
# ---------------------------------------------------------------------------
#
# LIBERO's BDDL goal predicates refer to per-instance sub-regions like
# ``basket_1_contain_region``. These names resolve at runtime to MuJoCo sites
# named ``contain_region`` declared inside the asset's MJCF; LIBERO's
# ``_load_sites_in_arena`` matches them by suffix and registers a
# ``SiteObjectState`` in ``object_states_dict``. If the asset MJCF lacks a
# ``contain_region`` site, the site state is never registered and
# ``_eval_predicate`` raises ``KeyError`` the first time ``check_success()``
# runs (see ``simulator.py``'s post-settle observable refresh).
#
# The object-axis perturbation pipeline picks variant classes from
# ``ASSET_VARIANTS``; some of those (e.g. ``white_storage_box``,
# ``chefmate_8_frypan``) lack a ``contain_region`` site. When chosen as a
# substitute for a goal-required container (basket / wooden_tray /
# desk_caddy), they break the goal predicate. The planner therefore needs to
# know which classes preserve the ``contain_region`` affordance so it can
# filter container variant pools.
#
# This is determined by scanning the bundled LIBERO asset MJCF files. The
# result is cached on first use.


@functools.lru_cache(maxsize=1)
def _libero_assets_root() -> pathlib.Path | None:
    """Locate the bundled LIBERO ``assets/`` directory.

    The runtime resolves LIBERO assets via the installed ``libero`` package,
    so we use the same root. Returns ``None`` if the package is not importable
    (in which case affordance probing must fail closed — see callers).
    """
    try:
        import libero.libero  # type: ignore
    except Exception:
        return None
    pkg_dir = pathlib.Path(libero.libero.__file__).parent
    candidate = pkg_dir / "assets"
    return candidate if candidate.is_dir() else None


_CONTAIN_REGION_SITE_RE = re.compile(rb'name="([A-Za-z0-9_]*contain_region)"')


@functools.lru_cache(maxsize=512)
def contain_region_sites(asset_class: str) -> frozenset[str]:
    """Return the set of ``*contain_region``-suffixed MJCF site names that
    ``asset_class`` declares.

    LIBERO matches BDDL region references like ``<instance>_<site>`` against
    the underlying MJCF site names; a missing site causes
    ``_load_sites_in_arena`` to omit the entry from ``object_states_dict``
    and ``_eval_predicate`` then raises ``KeyError`` on the first
    ``check_success()`` (see ``rca/stage3_run2b_contain_region_family.md``).

    The basket / wooden_tray classes expose a single site named
    ``contain_region``; the desk_caddy class exposes four directional ones
    (``left_contain_region``, ``right_contain_region``,
    ``back_contain_region``, ``front_contain_region``). The object-axis
    planner uses this set to require that any substitute variant exposes a
    superset of the canonical class's containment sites.

    Probing is done by scanning the bundled LIBERO ``assets/`` tree and is
    cached. Returns the empty set if the asset has no MJCF (fail-closed:
    callers must treat the canonical class as the only safe choice in that
    case).
    """
    root = _libero_assets_root()
    if root is None:
        return frozenset()
    # Convention: each asset class lives in ``<group>/<class>/<class>.xml``.
    for group_dir in root.iterdir():
        if not group_dir.is_dir():
            continue
        asset_dir = group_dir / asset_class
        if not asset_dir.is_dir():
            continue
        xml_path = asset_dir / f"{asset_class}.xml"
        if not xml_path.is_file():
            continue
        try:
            with open(xml_path, "rb") as fh:
                data = fh.read()
        except OSError:
            return frozenset()
        return frozenset(m.group(1).decode("ascii") for m in _CONTAIN_REGION_SITE_RE.finditer(data))
    return frozenset()


def class_provides_contain_region(asset_class: str) -> bool:
    """Backwards-compatible: True iff the class declares any contain_region site."""
    return bool(contain_region_sites(asset_class))
