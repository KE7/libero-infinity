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

from libero_infinity.asset_registry import get_dimensions

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

# Tiny anti-interpenetration lift used only by the legacy bounding-box fallback,
# preserved so unmeasured classes keep their historical spawn z exactly.
_FALLBACK_EPS: float = 1e-3


def _load_clearances() -> dict[str, float]:
    raw = pkgutil.get_data("libero_infinity", "data/spawn_clearances.json")
    if raw is None:
        return {}
    data = json.loads(raw)
    return {str(k): float(v) for k, v in data.get("clearances", {}).items()}


SPAWN_CLEARANCES: dict[str, float] = _load_clearances()


def spawn_clearance(asset_class: str) -> float:
    """Return the resting body-origin height (m) above the workspace surface.

    Uses the measured per-class clearance when available; otherwise falls back
    to half the registry bounding-box height plus the legacy anti-penetration
    epsilon (the pre-fix approximation), so the function is total for every
    class — including OOD object-axis substitutions not yet measured.
    """
    measured = SPAWN_CLEARANCES.get(asset_class)
    if measured is not None:
        return max(float(measured), _MIN_CLEARANCE)
    _w, _l, h = get_dimensions(asset_class)
    return max(float(h) / 2.0, _MIN_CLEARANCE) + _FALLBACK_EPS


def surface_spawn_z(surface_z: float, asset_class: str) -> float:
    """Resolved spawn z for ``asset_class`` resting on a surface at ``surface_z``.

    Pure function: identical output for identical inputs, on both the renderer
    and the simulator sides. ``surface_z`` is the Scenic table-surface constant
    (``TABLE_Z``); the measured clearance already folds in the small gap between
    that frame and the object's true MuJoCo settled pose.
    """
    return float(surface_z) + spawn_clearance(asset_class)


def is_measured(asset_class: str) -> bool:
    """True iff ``asset_class`` has a measured (non-fallback) spawn clearance."""
    return asset_class in SPAWN_CLEARANCES
