"""libero_model.scenic — Layer 2: LIBERO world vocabulary for Scenic 3.

All per-perturbation Scenic programs import this model with:

    model libero_model

This file defines:
  * LIBEROObject — Scenic Object subclass for all graspable table objects
  * LIBEROFixture — non-movable scene elements (cabinet, stove, rack, …)
  * TABLE_REGION — the valid placement volume above the main_table workspace
  * ASSET_VARIANTS — OOD variant registry (mirrors asset_registry.py)
  * Helper functions for constraint-building

Coordinate system (MuJoCo world frame)
───────────────────────────────────────
  +x  →  forward (away from robot)
  +y  →  left
  +z  →  up
  Origin: table centre projected onto the floor
  Table surface: z ≈ TABLE_Z  (0.82 m from floor)

Table workspace limits (from BDDL region definitions)
──────────────────────────────────────────────────────
  x ∈ [-0.40, 0.40]   (TABLE_WIDTH = 0.80 m)
  y ∈ [-0.30, 0.30]   (TABLE_LENGTH = 0.60 m)
  z ≈ TABLE_Z          (objects rest on surface)

Scenic 3 notes
──────────────
* Properties are declared with a colon + default value inside class bodies.
* `new ClassName [specifiers]` creates an instance; specifiers apply left-to-right.
* `require expr` states a hard constraint that Scenic's rejection sampler must satisfy.
* `require[p] expr` states a soft constraint satisfied with probability ≥ p.
* `Uniform(a, b, c)` samples uniformly from {a, b, c}.
* `Range(lo, hi)` samples uniformly from [lo, hi].
* `distance from A to B` gives the Euclidean distance between two positioned entities.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants (metres, matching LIBERO arena XML)
# ──────────────────────────────────────────────────────────────────────────────
TABLE_Z      = 0.82      # table-surface height in MuJoCo world frame (floor → 0)
TABLE_WIDTH  = 0.80      # full extent in x
TABLE_LENGTH = 0.60      # full extent in y
TABLE_X_MIN  = -0.40
TABLE_X_MAX  =  0.40
TABLE_Y_MIN  = -0.30
TABLE_Y_MAX  =  0.30

# Minimum object-object clearance for the rejection sampler (metres)
MIN_OBJ_CLEARANCE = 0.10

# Per-side workspace inset for SAFE_REGION (keeps object centres ≥ this far
# from workspace edges, ensuring robot reachability).
WORKSPACE_MARGIN = 0.05

# Per-side workspace inset for PLATE_SAFE_REGION — larger objects (plates, etc.)
# need more clearance from edges so their footprint doesn't overhang.
PLATE_WORKSPACE_MARGIN = 0.08

# Very thin z-extent that "pins" sampled z near TABLE_Z without allowing vertical
# variation. Increasing this would let TABLE_REGION overlap with non-table surfaces.
TABLE_REGION_Z_THICKNESS = 0.01

# ──────────────────────────────────────────────────────────────────────────────
# Workspace region
# BoxRegion: a 3-D box aligned with the world frame.
# Objects placed `in TABLE_REGION` will have their centre inside the box.
# The very thin z-extent (TABLE_REGION_Z_THICKNESS) pins the sampled z near TABLE_Z.
# ──────────────────────────────────────────────────────────────────────────────
TABLE_REGION = BoxRegion(
    dimensions=(TABLE_WIDTH, TABLE_LENGTH, TABLE_REGION_Z_THICKNESS),
    position=Vector(0.0, 0.0, TABLE_Z),
)

# Inner safe region — inset by WORKSPACE_MARGIN per side.
# Objects placed `in SAFE_REGION` have their centre ≥ WORKSPACE_MARGIN from
# workspace edges, ensuring robot reachability without explicit boundary requires.
SAFE_REGION = BoxRegion(
    dimensions=(TABLE_WIDTH - 2 * WORKSPACE_MARGIN, TABLE_LENGTH - 2 * WORKSPACE_MARGIN, TABLE_REGION_Z_THICKNESS),
    position=Vector(0.0, 0.0, TABLE_Z),
)

# Wider-margin safe region for large objects (plates, etc.) that need extra
# edge clearance so their footprint does not overhang the table boundary.
PLATE_SAFE_REGION = BoxRegion(
    dimensions=(TABLE_WIDTH - 2 * PLATE_WORKSPACE_MARGIN, TABLE_LENGTH - 2 * PLATE_WORKSPACE_MARGIN, TABLE_REGION_Z_THICKNESS),
    position=Vector(0.0, 0.0, TABLE_Z),
)

# ──────────────────────────────────────────────────────────────────────────────
# Object classes
# ──────────────────────────────────────────────────────────────────────────────

class LIBEROObject(Object):
    """A movable object on the workspace table.

    Properties
    ──────────
    libero_name : str
        Instance name used in the BDDL file, e.g. "akita_black_bowl_1".
        The simulator uses this to look up the MuJoCo free joint.

    asset_class : str
        BDDL object type, e.g. "akita_black_bowl".
        For object perturbation, swap this to an OOD variant class.

    graspable : bool
        Whether the robot can attempt to grasp this object.

    width, length, height : float  [metres]
        Approximate axis-aligned bounding box for collision margin calculations.

    allowCollisions : bool
        Set True so Scenic does NOT use FCL geometry intersection for rejection.
        Clearance is enforced via explicit `require distance > threshold`
        constraints instead — which are physics-correct and don't need FCL meshes.

    preserve_default_z : bool
        Whether LIBEROSimulation.setup() should override the sampled z with the
        canonical LIBERO placement height.

    support_parent_name : str
        Optional Scenic-level support parent for support-preserving perturbation.
        Used for debugging/tests; empty string means this object is sampled as
        an independent root object.
    """
    libero_name:     ""
    asset_class:     ""
    graspable:       True
    allowCollisions: True
    preserve_default_z: True
    support_parent_name: ""
    width:           0.08
    length:          0.08
    height:          0.06


class LIBEROFixture(Object):
    """A non-movable scene element (cabinet, stove, wine rack, …).

    Fixtures are declared in the Scenic scene so that spatial constraints
    (e.g. require the bowl is NOT on the stove region) can reference them.
    Their positions come from the BDDL :fixtures block and are injected
    by LIBEROSimulation.setup() via body_pos, not via free joints.
    """
    libero_name:  ""
    asset_class:  ""
    graspable:    False
    allowCollisions: True
    width:        0.20
    length:       0.30
    height:       0.40


# ──────────────────────────────────────────────────────────────────────────────
# Asset variant registry — loaded from the canonical JSON source of truth
# (src/libero_infinity/data/asset_variants.json)
# ──────────────────────────────────────────────────────────────────────────────
import json as _json
import pathlib as _pathlib

_asset_json_path = _pathlib.Path(__file__).resolve().parent.parent / "src" / "libero_infinity" / "data" / "asset_variants.json"
_asset_data = _json.loads(_asset_json_path.read_text())
ASSET_VARIANTS = _asset_data["variants"]
OBJECT_DIMENSIONS = _asset_data.get("dimensions", {})
# Authoritative pool of small graspable objects suitable as distractors.
# Kept in sync with asset_registry.DEFAULT_DISTRACTOR_POOL via the JSON key.
DISTRACTOR_POOL = _asset_data.get("distractor_pool", [])

# ──────────────────────────────────────────────────────────────────────────────
# Background texture pool — all PNG texture names in the LIBERO textures dir.
# Used by background_perturbation.scenic and the Scenic renderer for the
# "background" perturbation axis (wall + floor texture swapping).
# ──────────────────────────────────────────────────────────────────────────────
_LIBERO_TEXTURE_DIR = (
    _pathlib.Path(__file__).resolve().parent.parent
    / "vendor" / "libero" / "libero" / "libero" / "assets" / "textures"
)
LIBERO_BACKGROUND_TEXTURES = (
    sorted([_p.stem for _p in _LIBERO_TEXTURE_DIR.glob("*.png")])
    if _LIBERO_TEXTURE_DIR.exists()
    else []
)
