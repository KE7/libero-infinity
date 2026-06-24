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
import logging
import math
import pkgutil
import warnings

log = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Per-arena table-surface height (arena-aware spawn z)
# ---------------------------------------------------------------------------
#
# ``TABLE_SURFACE_Z`` (0.82) is the Scenic table-surface constant the *kitchen /
# default tabletop* arena is calibrated against: the measured per-class spawn
# clearances in ``spawn_clearances.json`` are ``settled_body_z − TABLE_SURFACE_Z``
# on that arena, so ``surface_spawn_z(TABLE_SURFACE_Z, …)`` reproduces the
# kitchen settled z exactly (pose_tolerance ≈ 0 mm). But the kitchen table top
# is not the only arena: LIBERO's living-room / coffee tables sit ~0.49 m LOWER
# and the study table ~0.03 m lower. Emitting the kitchen spawn z for ALL arenas
# placed every living-room / study object hundreds of mm too high, so every
# object failed pose_tolerance on the z axis (RCA task_robot_shove.md §4).
#
# The fix is arena-aware WITHOUT remeasuring per-arena clearances: a movable's
# settled body-origin height ABOVE its (flat) table top is a property of the
# object's geometry, not the arena, so it is arena-invariant. Shifting only the
# Scenic surface constant by the table-top delta therefore reproduces the
# arena's settled z while reusing the SAME measured clearance:
#
#     arena_surface_z(A) = TABLE_SURFACE_Z + (table_top_z[A] − table_top_z[ref])
#     surface_spawn_z(arena_surface_z(A), c) = arena_surface_z(A) + clearance(c)
#                                            = table_top_z[A] + (body-origin offset)
#
# ``table_top_z`` is the per-arena ``workspace_offset[2]`` hard-coded in LIBERO's
# arena problem classes (vendored at
# ``site-packages/libero/libero/envs/problems/libero_*_manipulation.py``). These
# are LIBERO geometry constants, sourced (not hand-tuned), mirroring the
# provenance of ``planner.position._LIBERO_TABLE_HALF_EXTENTS``.
#
#   kitchen_table / table  : workspace_offset (0, 0, 0.90)   (libero_kitchen_*, libero_tabletop_*)
#   study_table            : workspace_offset (-0.2, 0, 0.867) (libero_study_*)
#   living_room_table      : workspace_offset (0, 0, 0.41)   (libero_living_room_*)
#   coffee_table           : workspace_offset (0, 0, 0.41)   (libero_coffee_*)
#   floor                  : floor_offset     (0, 0, -0.035) (libero_floor_*)
_LIBERO_ARENA_TABLE_TOP_Z: dict[str, float] = {
    "table": 0.90,
    "main_table": 0.90,
    "kitchen_table": 0.90,
    "study_table": 0.867,
    "living_room_table": 0.41,
    "coffee_table": 0.41,
    "floor": -0.035,
}

# The reference arena ``TABLE_SURFACE_Z`` is calibrated against (kitchen/default
# tabletop). The measured clearances are expressed relative to TABLE_SURFACE_Z
# on this arena, so its delta is 0 and kitchen behaviour is byte-identical.
_REFERENCE_ARENA_TABLE_TOP_Z: float = _LIBERO_ARENA_TABLE_TOP_Z["kitchen_table"]

# Workspace-table classes whose per-(class, table) settled clearances are
# MEASURED (in spawn_clearances_variants.json, by measure_arena_tables) and
# threaded as the ``surface_class`` for table-resting objects. These are the
# arenas LIBERO seats objects materially differently from the kitchen reference
# — the ~0.49 m-lower living-room / coffee tables, where LIBERO additionally
# places several tall objects at an elevated metastable rest no rigid arena
# shift can predict. Reference arenas (kitchen / default ``table``) and the study
# table reproduce their settled z via ``arena_surface_z`` + the canonical
# per-class clearance, so they are NOT threaded and stay byte-identical.
#
# ``floor`` (the libero_object suite arena) is also threaded: its table top is
# ~0.935 m below the kitchen reference, and — like the living-room table — the
# canonical kitchen clearance does NOT transfer because the libero_object suite
# authors several classes (ketchup, salad_dressing, milk, orange_juice, …) in a
# DIFFERENT rest orientation than the kitchen, so the same class settles at a
# different body-origin clearance. The per-(class, floor) settled clearance is
# fully deterministic (0.00 mm spread across seeds; measured by
# ``measure_arena_tables``), so emitting it reproduces the suite's settled z and
# closes the G4 ``pose_tolerance`` z mismatch (RCA g4_task_pose_drift.md, Regime A).
PER_ARENA_TABLE_CLASSES: frozenset[str] = frozenset({"living_room_table", "coffee_table", "floor"})


def arena_surface_z(workspace_class: str | None) -> float:
    """Scenic table-surface constant for the arena whose workspace is ``workspace_class``.

    Returns the per-arena equivalent of :data:`TABLE_SURFACE_Z`: the kitchen
    constant shifted by the arena's table-top height delta (see the
    ``_LIBERO_ARENA_TABLE_TOP_Z`` block). Feeding this to
    :func:`surface_spawn_z` reproduces the arena's LIBERO settled z while reusing
    the arena-invariant per-class measured clearance, so the renderer's emitted
    spawn z matches the simulator's per-arena settled pose (pose_tolerance).

    An unknown / absent workspace class falls back to the reference
    ``TABLE_SURFACE_Z`` (the legacy single-constant behaviour), so kitchen and
    any not-yet-registered arena are unchanged.
    """
    if not workspace_class:
        return TABLE_SURFACE_Z
    top_z = _LIBERO_ARENA_TABLE_TOP_Z.get(workspace_class)
    if top_z is None:
        return TABLE_SURFACE_Z
    return TABLE_SURFACE_Z + (top_z - _REFERENCE_ARENA_TABLE_TOP_Z)


def spawn_clearance(
    asset_class: str, surface_class: str | None = None, *, distractor: bool = False
) -> float:
    """Return the resting body-origin height (m) above ``TABLE_SURFACE_Z``.

    Resolution order (most specific first), so an object-axis variant carries
    its own measured seating height on the *actual* support surface:

    1. The measured per-(variant, surface) clearance, when ``surface_class`` is
       given and the pair was measured. This captures both Finding-A sub-causes
       (geometry-different variants and surface-dependent seating) AND the
       measured per-(distractor, fixture) entries produced by Fix 2.
    2. **On-fixture analytic** (Fix 2): when ``surface_class`` is an elevated
       fixture not yet in the measured table, the object rests on the fixture's
       top face, so its body origin sits ``<rest surface above table>`` higher
       than it would on the table, plus its own table-resting body-origin offset
       (``spawn_clearance(asset_class, None)``). The rest surface depends on the
       resting object:

         * a DISTRACTOR (``distractor=True``) is a FLAT object that settles at
           the height a flat object actually comes to rest on the fixture top —
           ``fixture_distractor_rest_z_above_table`` (the settle-measured
           ``settle_top_z``). For a flat fixture (stove/cabinet/table top) this
           equals ``top_z``; for an OPEN-FRAME fixture (wine_rack, shelf) it is
           LOWER than ``top_z`` because the flat object sinks between the frame
           rails rather than perching on the highest collision edge (WS-1 RCA:
           wine_rack top_z≈0.396 collision edge vs ≈0.361 flat-settle rest).
         * a TASK object (``distractor=False``, default) keeps the raw
           ``fixture_top_z_above_table`` (``top_z``) — task seating already
           no-regresses and must be left unchanged.

       Both terms are measured geometry (or a conservative fallback), so the
       renderer and simulator resolve the SAME on-fixture z even before the
       per-pair settle has been recorded. The per-pair settle measurement
       validates this analytic value and, once recorded, supersedes it via rule 1.
    3. The measured per-canonical-class clearance (legacy table-resting table).
    4. The median measured clearance (:data:`DEFAULT_CLEARANCE`) — a data-derived
       prior, NOT the discredited bounding-box approximation — so the function is
       total for every (class, surface).
    """
    if surface_class is not None:
        # Cradle seating (angled open-frame fixture, e.g. wine_rack): a flat
        # distractor seats in the slot at a MEASURED body-centre clearance
        # ``dz`` above the table — NOT the additive top_z/settle_top_z model,
        # which does not hold in the inclined cradle. Resolved identically on the
        # renderer and the simulator so injected z == settled z.
        if distractor:
            # Per-(class, fixture) SEAT: the MEASURED flat-plate rest dz —
            # supersedes the (body-centred, knob-contaminated) variant clearance,
            # so injected z == settled z on the relocated seat (flat_stove burner).
            seat = distractor_fixture_seat(surface_class, asset_class)
            if seat is not None:
                return max(seat["dz"], _MIN_CLEARANCE)
            cr = cradle_rest(surface_class, asset_class)
            if cr is not None:
                return max(cr[1], _MIN_CLEARANCE)
        measured = VARIANT_CLEARANCES.get(_variant_key(asset_class, surface_class))
        if measured is not None:
            return max(float(measured), _MIN_CLEARANCE)
        if _is_fixture_surface(surface_class):
            on_table = spawn_clearance(asset_class, None)
            rest_z = (
                fixture_distractor_rest_z_above_table(surface_class)
                if distractor
                else fixture_top_z_above_table(surface_class)
            )
            return max(rest_z + on_table, _MIN_CLEARANCE)
    measured = SPAWN_CLEARANCES.get(asset_class)
    if measured is not None:
        return max(float(measured), _MIN_CLEARANCE)
    return max(DEFAULT_CLEARANCE, _MIN_CLEARANCE)


def surface_spawn_z(
    surface_z: float,
    asset_class: str,
    surface_class: str | None = None,
    *,
    distractor: bool = False,
) -> float:
    """Resolved spawn z for ``asset_class`` resting on ``surface_class``.

    Pure function: identical output for identical inputs, on both the renderer
    and the simulator sides. ``surface_z`` is the Scenic table-surface constant
    (``TABLE_Z``); the measured clearance already folds in the gap between that
    frame and the object's true MuJoCo settled pose on the given surface.

    ``surface_class`` is the class of the support the object rests on (e.g.
    ``"flat_stove"``, ``"wooden_cabinet"``); pass ``None`` for the default
    workspace table to preserve the legacy class-only behaviour.

    ``distractor`` selects the on-fixture rest surface for the un-measured
    analytic fallback (rule 2 of :func:`spawn_clearance`): a flat distractor
    rests at the settle-measured ``settle_top_z`` (which is below ``top_z`` for
    open-frame fixtures), while a task object keeps ``top_z``. It has no effect
    when a per-(class, surface) row is measured (rule 1) or for the table.
    """
    return float(surface_z) + spawn_clearance(asset_class, surface_class, distractor=distractor)


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


# Classes for which the hand-coded fallback is a documented, accepted source —
# the workspace tables/floor are ARENA surfaces (their per-arena geometry is
# served by #29's arena-table rows, NOT fixture_geometry.json), so resolving them
# through the fallback dict is correct and must not warn. Every OTHER non-empty
# fixture class hitting the fallback is an UNMEASURED real support surface and is
# warned about (once per class) so it cannot silently degrade spawn z / footprint.
_FALLBACK_OK_CLASSES: frozenset[str] = _WORKSPACE_TABLE_CLASSES
_warned_fixture_fallback: set[str] = set()


def _warn_fixture_fallback(fixture_class: str | None, quantity: str) -> None:
    """Emit a one-shot WARNING when a real (non-table) fixture resolves ``quantity``
    from the hand-coded fallback instead of measured ``fixture_geometry.json``."""
    fc = fixture_class or ""
    if not fc or fc in _FALLBACK_OK_CLASSES or fc in _warned_fixture_fallback:
        return
    _warned_fixture_fallback.add(fc)
    log.warning(
        "fixture %r has no MEASURED %s in fixture_geometry.json — using the "
        "hand-coded _FIXTURE_DIMS_FALLBACK (inaccurate). Run "
        "`scripts/measure_spawn_clearances.py --support-fixtures-only` to measure it.",
        fc,
        quantity,
    )


def fixture_footprint(fixture_class: str | None) -> tuple[float, float]:
    """Return the measured (width, length) xy footprint of a fixture class (m).

    Falls back to the conservative legacy dimensions when the fixture is not in
    the measured ``fixture_geometry.json`` table (warns for real fixtures).
    """
    geom = FIXTURE_GEOMETRY.get(fixture_class or "")
    if geom is not None:
        fp = geom.get("footprint")
        if isinstance(fp, (list, tuple)) and len(fp) >= 2:
            return float(fp[0]), float(fp[1])
    _warn_fixture_fallback(fixture_class, "footprint")
    dims = _FIXTURE_DIMS_FALLBACK.get(fixture_class or "", _FIXTURE_DIM_DEFAULT)
    return dims[0], dims[1]


def fixture_offset(fixture_class: str | None) -> tuple[float, float]:
    """Return the (dx, dy) of a fixture's geom-AABB center relative to its body
    position (the value the renderer emits as ``<fixture>.position``).

    Irregular fixtures (e.g. ``flat_stove``) carry collision geometry offset
    ~100 mm from the body origin; an origin-centered clearance box then misses
    part of the real fixture and a table distractor is injected penetrating it,
    causing the contact solver to launch it (RCA ``robot_distractor_settle.md``).
    Returns (0.0, 0.0) when no offset is recorded — so centered fixtures are
    unaffected. Measured by ``scripts/measure_fixture_offsets.py``.
    """
    geom = FIXTURE_GEOMETRY.get(fixture_class or "")
    if geom is not None:
        off = geom.get("offset")
        if isinstance(off, (list, tuple)) and len(off) >= 2:
            return float(off[0]), float(off[1])
    return 0.0, 0.0


def fixture_height(fixture_class: str | None) -> float:
    """Return the measured z-extent (height, m) of a fixture class."""
    geom = FIXTURE_GEOMETRY.get(fixture_class or "")
    if geom is not None and geom.get("height") is not None:
        return float(geom["height"])
    _warn_fixture_fallback(fixture_class, "height")
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
    _warn_fixture_fallback(fixture_class, "top_z")
    return fixture_height(fixture_class)


def fixture_distractor_rest_z_above_table(fixture_class: str | None) -> float:
    """Height above ``TABLE_SURFACE_Z`` (m) at which a FLAT distractor actually
    settles on the fixture top — the contact surface a flat object comes to rest
    on, NOT the highest collision edge.

    For a flat-topped fixture (stove burner, cabinet/microwave/table top) this is
    the same as :func:`fixture_top_z_above_table` (``top_z``): a flat object rests
    right on the flat surface. For an OPEN-FRAME fixture (wine_rack, two-layer
    shelf, desk_caddy) the flat object sinks between the top rails until it
    catches lower, so it settles BELOW the highest-collision-edge ``top_z`` (WS-1
    RCA ``robot_distractor_settle.md``: wine_rack ``top_z``≈0.396 vs flat-settle
    rest ≈0.361). The settle-measured value is stored per fixture as
    ``settle_top_z`` by ``scripts/measure_spawn_clearances.py
    --distractor-settle-z``; when absent it falls back to ``top_z`` so behaviour
    is unchanged for any not-yet-measured fixture (and exactly equal for flat
    fixtures, whose ``settle_top_z`` == ``top_z``).
    """
    geom = FIXTURE_GEOMETRY.get(fixture_class or "")
    if geom is not None and geom.get("settle_top_z") is not None:
        return float(geom["settle_top_z"])
    return fixture_top_z_above_table(fixture_class)


# Minimum ``top_z − settle_top_z`` divergence (m) that marks an OPEN-FRAME
# fixture — a slatted/lattice top with no flat platform. Equal to the
# pose-tolerance: a smaller divergence means the top is effectively flat
# (``settle_top_z`` ≈ ``top_z``) and a flat object rests on it.
_OPEN_FRAME_SETTLE_DROP: float = 0.005


def is_open_frame_fixture(fixture_class: str | None) -> bool:
    """True iff ``fixture_class`` is an OPEN-FRAME fixture that cannot hold a flat
    distractor on its top.

    An open-frame fixture (``wine_rack``, ``wooden_two_layer_shelf``) has a
    lattice / slatted top with **no flat platform**: a flat clutter object placed
    on it sinks between the rails and keeps falling rather than perching on the
    highest collision edge — for the wine_rack it falls THROUGH the frame to the
    table entirely (WS-1 RCA ``robot_distractor_settle.md``; verified: collision
    geometry is thin vertical panels, and a box injected on top drops ~35 mm in
    the 50-step settle window and continues to the table over a full settle).

    The original WS-1 fix tried to seat such a distractor at the settle-measured
    ``settle_top_z`` (below ``top_z``), but that 50-step value is a *mid-fall*
    artifact, not a stable rest: injecting there merely starts the object lower
    and it keeps sinking, so the injected target never matches the settled pose.
    Because no stable on-top rest exists, an open-frame fixture is **not a valid
    flat-distractor support**; :func:`_assignable_fixtures` excludes it and the
    distractor is routed to the table or a flat fixture (cabinet/stove/caddy),
    where it settles within pose-tolerance — matching the pre-fixture (main)
    table-resting behaviour.

    Detection is data-driven and general: the settle-measurement records a
    ``settle_top_z`` that diverges from ``top_z`` by more than the pose tolerance
    **only** for these open-frame tops (a flat top measures ``settle_top_z`` ≈
    ``top_z`` and carries no field), so any future open-frame fixture is excluded
    automatically without a hard-coded name.
    """
    geom = FIXTURE_GEOMETRY.get(fixture_class or "")
    if geom is None:
        return False
    top_z = geom.get("top_z")
    settle = geom.get("settle_top_z")
    if top_z is None or settle is None:
        return False
    return (float(top_z) - float(settle)) > _OPEN_FRAME_SETTLE_DROP


# ---------------------------------------------------------------------------
# Cradle seating — angled open-frame fixtures (wine_rack)
# ---------------------------------------------------------------------------
#
# An open-frame fixture whose top is an ANGLED slotted surface (the wine_rack: a
# 62°-inclined V-cradle, MJCF ``top_region`` site) cannot hold a flat distractor
# laid on its bounding top (it falls through — see ``is_open_frame_fixture``),
# but a FLAT-enough distractor placed IN the cradle slides to the bottom of the
# slant and comes to a STABLE rest, wedged against the lower rail / stopper (user
# directive C). That rest is measured per distractor class — drop the class into
# the cradle and long-settle to a fixed point (injected-vs-settled error 0 mm) —
# and stored in the fixture's ``cradle`` block:
#
#   cradle = {
#     "tilt_quat_wxyz": [...],   # class-independent rest orientation (the incline)
#     "x_half": <m>,             # half-range along the cradle's free (x) axis
#     "rests": { <class>: {"dy": <m>, "dz": <m>} },  # per eligible (flat) class
#   }
#
# ``dy`` is the +y offset (toward the downslope) of the rest from the fixture
# BODY position; ``dz`` is the resting body-centre clearance above
# ``TABLE_SURFACE_Z``. Only classes that seat WITHOUT tumbling appear in
# ``rests`` — tall/cube classes (e.g. alphabet_soup) tip and fall through, so
# they are absent and fall back to the table. Both the renderer (placement) and
# the simulator (injection) read this block so the injected pose == the settled
# pose, keeping the distractor seated ON the rack within pose-tolerance.


def cradle_geometry(fixture_class: str | None) -> dict | None:
    """Return the fixture's measured ``cradle`` block, or ``None`` if it is not an
    angled-slot (cradle) fixture."""
    geom = FIXTURE_GEOMETRY.get(fixture_class or "")
    if geom is None:
        return None
    c = geom.get("cradle")
    return c if isinstance(c, dict) and c.get("rests") else None


def is_cradle_fixture(fixture_class: str | None) -> bool:
    """True iff ``fixture_class`` is an angled-slot fixture with measured per-class
    cradle rests (so flat distractors can be SEATED in it rather than excluded)."""
    return cradle_geometry(fixture_class) is not None


def cradle_rest(fixture_class: str | None, asset_class: str | None) -> tuple[float, float] | None:
    """Return ``(dy, dz)`` cradle rest for ``asset_class`` on ``fixture_class``.

    ``dy`` is the +y offset from the fixture body toward the downslope; ``dz`` is
    the resting body-centre clearance above ``TABLE_SURFACE_Z``. Returns ``None``
    when the fixture is not a cradle fixture or the class is not cradle-seatable
    (too tall/round to seat without tumbling) — the caller then routes the
    distractor to the table."""
    c = cradle_geometry(fixture_class)
    if c is None:
        return None
    r = (c.get("rests") or {}).get(asset_class or "")
    if not isinstance(r, dict) or r.get("dy") is None or r.get("dz") is None:
        return None
    return float(r["dy"]), float(r["dz"])


def cradle_tilt_quat(fixture_class: str | None) -> tuple[float, float, float, float] | None:
    """Return the (class-independent) cradle rest orientation as a MuJoCo wxyz
    quaternion, or ``None`` for a non-cradle fixture."""
    c = cradle_geometry(fixture_class)
    if c is None:
        return None
    q = c.get("tilt_quat_wxyz")
    if isinstance(q, (list, tuple)) and len(q) == 4:
        return tuple(float(x) for x in q)  # type: ignore[return-value]
    return None


def cradle_x_half(fixture_class: str | None) -> float:
    """Half-range along the cradle's free (x) axis for distractor placement (m)."""
    c = cradle_geometry(fixture_class)
    if c is None:
        return 0.0
    return max(float(c.get("x_half", 0.0)), 0.0)


def is_cradle_seatable(fixture_class: str | None, asset_class: str | None) -> bool:
    """True iff ``asset_class`` has a measured stable cradle rest on ``fixture_class``."""
    return cradle_rest(fixture_class, asset_class) is not None


# ---------------------------------------------------------------------------
# Per-(class, fixture) distractor SEAT — flat-top fixture, off-centre support
# ---------------------------------------------------------------------------
#
# Some flat-top fixtures present their real support surface OFFSET from the body
# origin, with non-support clutter (a knob, a lip) intruding into the
# body-centred placement region. The ``flat_stove`` is the canonical case: its
# real burner plate is a clean flat collision surface (the ``burnerplate`` visual
# "grate" mesh is ``contype=0/conaffinity=0`` — NON-colliding), but it sits
# ``+0.15 m`` in x from the fixture body, while the stove KNOB occupies the
# body-centred region. A distractor placed at the body centre therefore straddles
# the knob and the burner-left edge: a TALL class (the alphabet_soup CAN,
# r≈0.029/h≈0.075) tips off the knob and tumbles OFF the stove (long-settle drift
# > 800 mm), while shorter classes survive the 50-step gate by luck. PR #32
# excluded alphabet_soup via ``unstable_distractors``; the SEAT instead relocates
# it onto the burner plate, where it settles to a true fixed point (injected-vs-
# settled 0.25 mm at the 50-step AND a >1500-step settle, reproducibly across
# tasks/seeds — WS alphabet_soup-stove RCA).
#
# A seat is stored per (fixture, class) in ``fixture_geometry.json`` under the
# fixture's ``distractor_seats`` map::
#
#   "distractor_seats": {
#     "<class>": {"dx": <m>, "dy": <m>, "dz": <m>, "x_half": <m>, "y_half": <m>}
#   }
#
# ``dx``/``dy`` offset the seat CENTRE from the fixture BODY toward the support
# surface; ``dz`` is the resting body-centre clearance above ``TABLE_SURFACE_Z``
# (the MEASURED flat-plate rest — it SUPERSEDES the stale body-centred variant
# clearance); ``x_half``/``y_half`` bound the (stable) placement box on the plate.
# Unlike a cradle (which seats EVERY fitting class), a seat is per-CLASS: only the
# listed classes relocate; the rest keep the default flat-fixture placement. Both
# the renderer (placement) and the simulator (z via :func:`spawn_clearance`) read
# the seat so injected pose == settled pose. Absent ⇒ no seat (every fixture
# without a ``distractor_seats`` map is byte-identical).


def distractor_fixture_seat(
    fixture_class: str | None, asset_class: str | None
) -> dict[str, float] | None:
    """Return the measured per-(class, fixture) distractor SEAT, or ``None``.

    The returned dict has keys ``dx, dy, dz, x_half, y_half`` (all metres, with
    missing offsets/halves defaulting to 0.0); ``dz`` (the resting body-centre
    clearance above ``TABLE_SURFACE_Z``) is required. ``None`` when the fixture
    has no ``distractor_seats`` map or the class is not seated there."""
    geom = FIXTURE_GEOMETRY.get(fixture_class or "")
    if geom is None:
        return None
    seats = geom.get("distractor_seats")
    if not isinstance(seats, dict):
        return None
    s = seats.get(asset_class or "")
    if not isinstance(s, dict) or s.get("dz") is None:
        return None
    return {
        "dx": float(s.get("dx", 0.0)),
        "dy": float(s.get("dy", 0.0)),
        "dz": float(s["dz"]),
        "x_half": float(s.get("x_half", 0.0)),
        "y_half": float(s.get("y_half", 0.0)),
    }


def is_distractor_fixture_seated(fixture_class: str | None, asset_class: str | None) -> bool:
    """True iff ``asset_class`` has a measured per-(class, fixture) seat on
    ``fixture_class`` (so it is relocated to the measured support-surface rest
    rather than placed at the body-centred footprint region)."""
    return distractor_fixture_seat(fixture_class, asset_class) is not None


def is_fixture_measured(fixture_class: str | None) -> bool:
    """True iff ``fixture_class`` has measured geometry in the data table."""
    return (fixture_class or "") in FIXTURE_GEOMETRY


def is_distractor_fixture_unstable(asset_class: str | None, fixture_class: str | None) -> bool:
    """True iff distractor ``asset_class`` cannot SETTLE STABLY on ``fixture_class``'s
    top, so it must NOT be seated there (route it to the table / a flat fixture).

    Footprint fit (:func:`_distractor_fits_fixture`) only checks that a class's
    base fits within the fixture top — necessary but NOT sufficient. Some fixture
    tops are not a clean flat platform (the ``flat_stove`` top is a raised burner
    GRATE), so a TALL class that fits the footprint still tips/slides on the grate
    edges and settles >5 mm from its injected pose, with a large HORIZONTAL slide
    no spawn-z or rest-orientation can remove (WS follow-up RCA: ``alphabet_soup``
    on ``flat_stove`` — 7/10 seeds fail, ~6.9 mm median xy slide — while the SAME
    class settles within 0.5 mm on the flat table and 0.3 mm on a flat cabinet
    top). Such a (class, fixture) seating is excluded, exactly as the wine_rack
    cradle lists only the classes that seat WITHOUT tumbling (``rests``).

    The exclusion is MEASURED per (class, fixture): a fixture's
    ``unstable_distractors`` list in ``fixture_geometry.json`` (verified by
    ``scripts/measure_spawn_clearances.py --distractor-fixture-stability``).
    Absent ⇒ no exclusion, so every fixture with a clean flat top is unchanged.
    """
    geom = FIXTURE_GEOMETRY.get(fixture_class or "")
    if geom is None:
        return False
    unstable = geom.get("unstable_distractors")
    return bool(unstable) and (asset_class or "") in set(unstable)


# ---------------------------------------------------------------------------
# Distractor footprint geometry — measured per-class settled AABB extents
# ---------------------------------------------------------------------------
#
# The renderer historically declared EVERY distractor as a uniform 8 cm cube
# (``_DISTRACTOR_HALF = 0.04``) while MuJoCo loaded the REAL asset mesh. For a
# box-like distractor (cream_cheese, popcorn, …) the 8 cm proxy is close enough,
# but for an anisotropic / irregular class (desk_caddy: real footprint
# 0.14×0.42×0.22 m; bowl_drainer: 0.18×0.14) the proxy is off by hundreds of mm —
# so the placement engine reasoned about a 4 cm half-cube while a 0.42 m caddy
# was instantiated, overhanging an undersized fixture top and tilting to an
# xy-dependent settle no single clearance-z could satisfy (RCA
# ``distractor_z_convergence.md`` §"structural proxy").
#
# These footprints are MEASURED from the authoritative LIBERO MuJoCo assets by
# ``scripts/measure_spawn_clearances.py --distractor-footprints-only`` and stored
# in ``data/distractor_geometry.json``. Each entry holds:
#   * ``footprint`` = [w, l] — median settled geom-AABB horizontal extents (m),
#     informational (the settle yaw is free, so per-axis w/l are not load-bearing).
#   * ``height``    = settled geom-AABB z-extent (m), used for the z-prune band.
#   * ``radius``    = the dominant-mode of the per-sample circumscribed planar
#     half-extent ``0.5·sqrt(wx² + wy²)`` — a yaw-ROBUST half-extent that bounds
#     the distractor's footprint under ANY settle yaw (and the consistent ~90°
#     settle tip). This is the value threaded into every clearance constraint, so
#     orientation enters the clearance instead of a single scalar half-extent.

# Legacy uniform-proxy fallback (w, l, h) in metres — used only for a distractor
# class with no measured geometry, preserving the historical 8 cm-cube behaviour.
_DISTRACTOR_DIM_DEFAULT: tuple[float, float, float] = (0.08, 0.08, 0.08)


def _load_distractor_geometry() -> dict[str, dict]:
    try:
        raw = pkgutil.get_data("libero_infinity", "data/distractor_geometry.json")
    except FileNotFoundError:
        return {}
    if raw is None:
        return {}
    data = json.loads(raw)
    out: dict[str, dict] = {}
    for k, v in data.get("distractors", {}).items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


DISTRACTOR_GEOMETRY: dict[str, dict] = _load_distractor_geometry()


def distractor_footprint(asset_class: str | None) -> tuple[float, float, float]:
    """Return the measured (width, length, height) settled AABB extents (m).

    Falls back to the legacy uniform 8 cm proxy for an unmeasured class so the
    function is total.
    """
    geom = DISTRACTOR_GEOMETRY.get(asset_class or "")
    if geom is not None:
        fp = geom.get("footprint")
        h = geom.get("height")
        if isinstance(fp, (list, tuple)) and len(fp) >= 2 and h is not None:
            return float(fp[0]), float(fp[1]), float(h)
    return _DISTRACTOR_DIM_DEFAULT


def distractor_planar_half(asset_class: str | None) -> float:
    """Yaw-robust planar half-extent (circumscribed radius, m) of a distractor.

    Used as the per-class clearance half-extent on BOTH x and y so an arbitrary
    settle yaw (and the consistent ~90° settle tip the irregular distractors
    take) is bounded. Reads the measured ``radius`` when present; otherwise
    derives the circumscribed radius from the footprint (legacy proxy for an
    unmeasured class).
    """
    geom = DISTRACTOR_GEOMETRY.get(asset_class or "")
    if geom is not None and geom.get("radius") is not None:
        return max(float(geom["radius"]), _MIN_CLEARANCE)
    w, length, _ = distractor_footprint(asset_class)
    return max(0.5 * math.hypot(w, length), _MIN_CLEARANCE)


def distractor_fit_half(asset_class: str | None) -> float:
    """Resting-orientation footprint half-extent (m) for the support-FIT test.

    The larger horizontal half-extent of the settled footprint
    (``max(w, l) / 2``). Unlike :func:`distractor_planar_half` (the 45°-robust
    circumscribed radius used for *clearance*), this is the half-extent the
    distractor actually occupies in its roughly axis-aligned resting pose, so the
    "does it sit on this support top?" decision and the placement-region sizing
    are neither under-counted (the old uniform 0.04 proxy ignored the real mesh)
    nor over-counted (the diagonal radius would wrongly exclude a box from a
    narrow stove it physically fits). For the legacy 8 cm cube this is 0.04 —
    identical to the historical proxy, so box-distractor support assignment is
    unchanged; only genuinely-oversized classes (desk_caddy) are newly excluded.
    """
    w, length, _ = distractor_footprint(asset_class)
    return max(max(w, length) / 2.0, _MIN_CLEARANCE)


def distractor_half_height(asset_class: str | None) -> float:
    """Half of the measured settled z-extent (m) of a distractor class."""
    return max(distractor_footprint(asset_class)[2] / 2.0, _MIN_CLEARANCE)


def is_distractor_measured(asset_class: str | None) -> bool:
    """True iff ``asset_class`` has measured distractor geometry."""
    return (asset_class or "") in DISTRACTOR_GEOMETRY


# ---------------------------------------------------------------------------
# Per-class distractor table REST ORIENTATION
# ---------------------------------------------------------------------------
#
# A table distractor with no Scenic-authored orientation is injected at the
# generic upright fallback ``DEFAULT_ORIENTATIONS["_default"]`` — a 90°-about-x
# rotation that stands the typical GoogleScannedObject box upright. That default
# is correct for the box-like pool classes (alphabet_soup settles 0.5 mm on the
# table at x90), but WRONG for an irregular class whose stable flat rest is NOT
# the x90 upright: injected on edge, it tips a few degrees AND slides several mm
# during the 50-step settle, so the injected pose never equals the settled pose
# (the pose_tolerance miss). The fix mirrors the cradle ``tilt_quat`` and the
# pre-existing ``DEFAULT_ORIENTATIONS["simple_rack"] = identity`` precedent:
# inject such a class at its MEASURED stable table rest orientation so injected
# pose == settled pose.
#
# Measured by sweeping the candidate rest orientations of the class at its real
# table placement and 50-step settling each (scripts/measure_spawn_clearances.py
# does not author this; it is a per-asset geometric constant, like a cradle tilt).
# For ``bowl_drainer`` (a flat-bottomed dish rack, footprint 0.21×0.23, the
# largest pool class) the identity rest (base flat on the table) settles with
# dxy ≈ 0 mm and tip ≈ 0° across table placements, whereas the x90 upright tips
# it onto a rim and slides ~6–9 mm (pose_err 11.8 mm — WS follow-up RCA). The
# value is a MuJoCo wxyz quaternion; a class absent here keeps the generic x90
# default unchanged, so every other distractor is byte-identical.
_DISTRACTOR_TABLE_REST_QUAT: dict[str, tuple[float, float, float, float]] = {
    "bowl_drainer": (1.0, 0.0, 0.0, 0.0),  # identity — flat base-down dish rack
}


def distractor_table_rest_quat(
    asset_class: str | None,
) -> tuple[float, float, float, float] | None:
    """Measured stable TABLE rest orientation (MuJoCo wxyz) for a distractor class.

    Returns ``None`` for a class that settles stably at the generic upright
    default (every box-like pool class), so the simulator leaves its injected
    orientation unchanged. Returns the measured wxyz quaternion for an irregular
    class whose flat rest differs from the x90 upright (``bowl_drainer``), so the
    simulator injects it there and the 50-step settle leaves it in place."""
    q = _DISTRACTOR_TABLE_REST_QUAT.get(asset_class or "")
    return tuple(float(x) for x in q) if q is not None else None
