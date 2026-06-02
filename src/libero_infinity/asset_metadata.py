"""Shared asset spawn-pose metadata — single source of truth for the resolved
spawn z of a movable object resting on a workspace surface.

Both the Scenic renderer (at code generation) and the MuJoCo simulator (at
``env.reset``) resolve an object's spawn z through :func:`surface_spawn_z`.
Because they call the *same* pure function with the *same* table-surface
constant, the Scenic-sampled object pose and the post-reset MuJoCo pose live in
the same frame, so the G4 family-C ``pose_tolerance`` invariant can compare them
1-to-1 (validation plan §4: "sampled Scenic object poses … must match MuJoCo
object poses … within 5 mm position").

Before this module existed, the renderer emitted a bare ``TABLE_Z`` placeholder
z for every object while the simulator resolved the real settled z — so every
``pose_tolerance`` check failed on an 8–18 cm z-frame mismatch (see
``rca/stage1_g4_consistency_pose_frame_mismatch.md``; Option A).

Per-class spawn clearance
-------------------------
The *spawn clearance* of an asset class is the height (m) of its settled MuJoCo
body-origin above the Scenic table-surface constant ``TABLE_Z`` when the object
rests on the workspace table::

    clearance(class) = body_xpos_z(settled) - TABLE_Z

It is **not** half the bounding-box height (the body origin is generally not the
geometric centre, and LIBERO objects settle onto collision geometry that the
bounding box does not capture). The clearances are therefore *measured* from the
authoritative LIBERO MuJoCo assets by ``scripts/measure_spawn_clearances.py``
and stored in ``data/spawn_clearances.json``. Re-run that generator after any
asset upgrade. Asset classes absent from the registry fall back to the legacy
half-bounding-box approximation so the function is always total.
"""

from __future__ import annotations

import json
import pkgutil
import warnings

# Canonical workspace table-surface height in the Scenic / MuJoCo world frame
# (floor → 0). This is the surface objects are sampled to rest on. It MUST stay
# equal to ``TABLE_Z`` in ``scenic/libero_model.scenic`` and ``simulator.py``;
# ``test_invariants_consistency`` asserts they agree so the three never drift.
# Measured spawn clearances are expressed relative to this value, so the
# renderer passes it to :func:`surface_spawn_z` to emit a concrete spawn z.
TABLE_SURFACE_Z: float = 0.82

# Minimum physical clearance of an object's body origin above its support (m).
# Guards against degenerate/zero-height registry entries.
_MIN_CLEARANCE: float = 0.01


def _load_clearances() -> dict[str, float]:
    raw = pkgutil.get_data("libero_infinity", "data/spawn_clearances.json")
    if raw is None:
        return {}
    data = json.loads(raw)
    return {str(k): float(v) for k, v in data.get("clearances", {}).items()}


def _variant_key(asset_class: str, surface_class: str) -> str:
    """Composite key for the per-(variant, surface) clearance table."""
    return f"{asset_class}|{surface_class}"


def _load_variant_clearances() -> dict[str, float]:
    """Load the per-(variant_class, surface_class) settled-clearance table.

    This table (``data/spawn_clearances_variants.json``) resolves Finding A of
    the RCA: an object's settled clearance is NOT class-invariant across support
    surfaces (the same white_bowl settles ~50 mm higher on a cabinet top than on
    a stove), and an OOD object-axis variant generally settles at a different
    clearance than the canonical class on the SAME surface. It is keyed by
    ``"<variant_class>|<surface_class>"`` and measured by
    ``scripts/measure_spawn_clearances.py``. Absent when not yet generated, in
    which case :func:`surface_spawn_z` falls back to the canonical class table.
    """
    try:
        raw = pkgutil.get_data("libero_infinity", "data/spawn_clearances_variants.json")
    except FileNotFoundError:
        # Optional resource: when the generator has not been run the renderer
        # falls back to the canonical per-class table. This is an expected
        # absence, not a swallowed error.
        return {}
    if raw is None:
        return {}
    data = json.loads(raw)
    out: dict[str, float] = {}
    for k, v in data.get("clearances", {}).items():
        out[str(k)] = float(v)
    return out


SPAWN_CLEARANCES: dict[str, float] = _load_clearances()
VARIANT_CLEARANCES: dict[str, float] = _load_variant_clearances()

if not VARIANT_CLEARANCES:
    # FV SMT D2 / MC Prop 2: make the inert state visible. When the per-(variant,
    # surface) table is absent, ``surface_spawn_z`` ignores its ``surface_class``
    # argument and every (variant, surface) pair resolves to the canonical
    # per-class clearance (or the median ``DEFAULT_CLEARANCE`` for OOD variants),
    # re-introducing the seating-height settle the table is meant to remove.
    warnings.warn(
        "libero_infinity.asset_metadata: spawn_clearances_variants.json is "
        "absent or empty — surface_spawn_z ignores its surface argument and "
        "per-(variant,surface) seating heights fall back to the canonical "
        "per-class clearance (FV SMT D2). Run "
        "scripts/measure_spawn_clearances.py to populate it.",
        RuntimeWarning,
        stacklevel=2,
    )


def _default_clearance() -> float:
    """Data-derived prior for an unmeasured class: the median measured clearance.

    A movable LIBERO object's settled body-origin sits ~0.10 m above the table
    in a tight band (the measured clearances cluster at 0.087–0.152 m, median
    ≈ 0.10). The pre-fix ``bbox_height / 2`` approximation is *known wrong* — it
    is exactly the model the z-frame RCA refuted (the body origin is not the
    geometric centre), and it systematically under-estimates by ~5–9 cm, which
    is what made unmeasured classes (object-axis OOD variants, distractor-pool
    objects) fail pose_tolerance. The median measured clearance is an unbiased,
    data-derived prior — far closer than the bounding box — used until the class
    is measured (regenerate via ``scripts/measure_spawn_clearances.py``).
    """
    if SPAWN_CLEARANCES:
        vals = sorted(SPAWN_CLEARANCES.values())
        n = len(vals)
        mid = n // 2
        return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0
    return 0.10


DEFAULT_CLEARANCE: float = _default_clearance()


# Workspace-table fixture classes — the arena tables objects rest on directly
# (the TABLE_SURFACE_Z surface). A ``surface_class`` in this set is a *table*
# (use the per-class / per-(class,table) clearance), whereas any other named
# surface is an *elevated fixture* whose top sits ``fixture_top_z_above_table``
# above the table — a distractor resting on it seats that much higher. Kept in
# sync with ``ArticulationModel.root_workspace_fixtures``.
_WORKSPACE_TABLE_CLASSES: frozenset[str] = frozenset(
    {"table", "kitchen_table", "living_room_table", "study_table", "floor"}
)


def _is_fixture_surface(surface_class: str | None) -> bool:
    """True iff ``surface_class`` names an elevated fixture (not a table).

    A fixture surface is one that has known geometry (measured or fallback) and
    is not one of the workspace-table classes.
    """
    if not surface_class or surface_class in _WORKSPACE_TABLE_CLASSES:
        return False
    return surface_class in FIXTURE_GEOMETRY or surface_class in _FIXTURE_DIMS_FALLBACK


def spawn_clearance(asset_class: str, surface_class: str | None = None) -> float:
    """Return the resting body-origin height (m) above ``TABLE_SURFACE_Z``.

    Resolution order (most specific first), so an object-axis variant carries
    its own measured seating height on the *actual* support surface:

    1. The measured per-(variant, surface) clearance, when ``surface_class`` is
       given and the pair was measured. This captures both Finding-A sub-causes
       (geometry-different variants and surface-dependent seating) AND the
       measured per-(distractor, fixture) entries produced by Fix 2.
    2. **On-fixture analytic** (Fix 2): when ``surface_class`` is an elevated
       fixture not yet in the measured table, the distractor rests on the
       fixture's top face, so its body origin sits
       ``fixture_top_z_above_table(surface_class)`` higher than it would on the
       table, plus its own table-resting body-origin offset
       (``spawn_clearance(asset_class, None)``). Both terms are measured
       geometry (or a conservative fallback), so the renderer and simulator
       resolve the SAME on-fixture z even before the per-pair settle has been
       recorded — no hardcoded fixture heights, no chicken-and-egg with the
       measured table. The settle measurement validates this analytic value and,
       once recorded, supersedes it via rule 1.
    3. The measured per-canonical-class clearance (legacy table-resting table).
    4. The median measured clearance (:data:`DEFAULT_CLEARANCE`) — a data-derived
       prior, NOT the discredited bounding-box approximation — so the function is
       total for every (class, surface).
    """
    if surface_class is not None:
        measured = VARIANT_CLEARANCES.get(_variant_key(asset_class, surface_class))
        if measured is not None:
            return max(float(measured), _MIN_CLEARANCE)
        if _is_fixture_surface(surface_class):
            on_table = spawn_clearance(asset_class, None)
            return max(fixture_top_z_above_table(surface_class) + on_table, _MIN_CLEARANCE)
    measured = SPAWN_CLEARANCES.get(asset_class)
    if measured is not None:
        return max(float(measured), _MIN_CLEARANCE)
    return max(DEFAULT_CLEARANCE, _MIN_CLEARANCE)


def surface_spawn_z(surface_z: float, asset_class: str, surface_class: str | None = None) -> float:
    """Resolved spawn z for ``asset_class`` resting on ``surface_class``.

    Pure function: identical output for identical inputs, on both the renderer
    and the simulator sides. ``surface_z`` is the Scenic table-surface constant
    (``TABLE_Z``); the measured clearance already folds in the gap between that
    frame and the object's true MuJoCo settled pose on the given surface.

    ``surface_class`` is the class of the support the object rests on (e.g.
    ``"flat_stove"``, ``"wooden_cabinet"``); pass ``None`` for the default
    workspace table to preserve the legacy class-only behaviour.
    """
    return float(surface_z) + spawn_clearance(asset_class, surface_class)


def is_measured(asset_class: str, surface_class: str | None = None) -> bool:
    """True iff ``asset_class`` (on ``surface_class``) has a measured clearance."""
    if surface_class is not None and _variant_key(asset_class, surface_class) in VARIANT_CLEARANCES:
        return True
    return asset_class in SPAWN_CLEARANCES


# ---------------------------------------------------------------------------
# Fixture geometry — measured footprints and top-surface heights
# ---------------------------------------------------------------------------
#
# Under option (i) a distractor can be placed to rest ON a scene fixture (stove
# burner, cabinet top, wine-rack shelf). Two pieces of fixture geometry are then
# needed and must be MEASURED, not hand-guessed:
#
#   * footprint (width, length) — the xy area the distractor's sampled (x, y)
#     must stay within so it actually lands on the fixture top. The previously
#     hand-coded ``_FIXTURE_DIMS`` in the renderer under-estimated several
#     fixtures (which is exactly why distractors slipped onto fixtures
#     unexpectedly); the measured table replaces them.
#   * top_z (above ``TABLE_SURFACE_Z``) — the height of the fixture's top face.
#     The per-(distractor, fixture) spawn clearance is ``top_z + <body-origin
#     offset>`` (see ``scripts/measure_spawn_clearances.py``), so the renderer
#     and simulator inject the distractor exactly where it settles.
#
# Both are produced by ``scripts/measure_spawn_clearances.py`` into
# ``data/fixture_geometry.json``. Until that file exists the conservative
# fallback below keeps the renderer total (matching the legacy hand-coded
# values), and ``surface_spawn_z`` falls back to the median prior — no crash.

# Conservative fallback (width, length, height) in metres — used only when the
# measured ``fixture_geometry.json`` is absent. Mirrors the legacy renderer
# ``_FIXTURE_DIMS`` so behaviour is unchanged before the generator runs.
_FIXTURE_DIMS_FALLBACK: dict[str, tuple[float, float, float]] = {
    "wooden_cabinet": (0.30, 0.30, 0.24),
    "white_cabinet": (0.30, 0.30, 0.24),
    "flat_stove": (0.36, 0.20, 0.08),
    "wine_rack": (0.18, 0.12, 0.20),
    "microwave": (0.24, 0.18, 0.16),
    "bowl_drainer": (0.18, 0.14, 0.08),
    "desk_caddy": (0.14, 0.42, 0.22),
    "wooden_two_layer_shelf": (0.33, 0.20, 0.21),
    "table": (0.80, 0.60, 0.05),
    "kitchen_table": (0.80, 0.60, 0.05),
    "living_room_table": (0.55, 0.65, 0.05),
    "study_table": (0.50, 0.58, 0.05),
    "floor": (0.50, 0.55, 0.01),
}
_FIXTURE_DIM_DEFAULT: tuple[float, float, float] = (0.20, 0.18, 0.18)


def _load_fixture_geometry() -> dict[str, dict]:
    try:
        raw = pkgutil.get_data("libero_infinity", "data/fixture_geometry.json")
    except FileNotFoundError:
        return {}
    if raw is None:
        return {}
    data = json.loads(raw)
    out: dict[str, dict] = {}
    for k, v in data.get("fixtures", {}).items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


FIXTURE_GEOMETRY: dict[str, dict] = _load_fixture_geometry()


def fixture_footprint(fixture_class: str | None) -> tuple[float, float]:
    """Return the measured (width, length) xy footprint of a fixture class (m).

    Falls back to the conservative legacy dimensions when the fixture is not in
    the measured ``fixture_geometry.json`` table.
    """
    geom = FIXTURE_GEOMETRY.get(fixture_class or "")
    if geom is not None:
        fp = geom.get("footprint")
        if isinstance(fp, (list, tuple)) and len(fp) >= 2:
            return float(fp[0]), float(fp[1])
    dims = _FIXTURE_DIMS_FALLBACK.get(fixture_class or "", _FIXTURE_DIM_DEFAULT)
    return dims[0], dims[1]


def fixture_height(fixture_class: str | None) -> float:
    """Return the measured z-extent (height, m) of a fixture class."""
    geom = FIXTURE_GEOMETRY.get(fixture_class or "")
    if geom is not None and geom.get("height") is not None:
        return float(geom["height"])
    dims = _FIXTURE_DIMS_FALLBACK.get(fixture_class or "", _FIXTURE_DIM_DEFAULT)
    return dims[2]


def fixture_top_z_above_table(fixture_class: str | None) -> float:
    """Return the fixture top-surface height above ``TABLE_SURFACE_Z`` (m).

    This is the measured world-frame z of the fixture's top face minus the
    table-surface constant — the surface a distractor settles onto. Falls back
    to the fixture's height (i.e. assuming the fixture base sits at the table)
    when no measured value is present.
    """
    geom = FIXTURE_GEOMETRY.get(fixture_class or "")
    if geom is not None and geom.get("top_z") is not None:
        return float(geom["top_z"])
    return fixture_height(fixture_class)


def is_fixture_measured(fixture_class: str | None) -> bool:
    """True iff ``fixture_class`` has measured geometry in the data table."""
    return (fixture_class or "") in FIXTURE_GEOMETRY
