"""Layer 1: Scenic 3 ↔ LIBERO simulator bridge.

Implements the two classes Scenic requires to drive a new simulator:

  LIBEROSimulator   — subclasses scenic.core.simulators.Simulator;
                       creates LIBEROSimulation instances from Scenic scenes
  LIBEROSimulation  — subclasses scenic.core.simulators.Simulation;
                       injects Scenic-sampled poses into MuJoCo, steps physics,
                       reads back state for Scenic's monitor/require constraints

Key design decisions
────────────────────
* We do NOT bypass the LIBERO env entirely. Instead we call env.reset() first
  (which loads the BDDL scene with its default placements), then override each
  movable object's joint-qpos with the Scenic-sampled position. This keeps the
  full LIBERO physics model (arena XML, fixtures, robot, cameras) intact.

* setup() is the injection point. It calls env.reset() first, then iterates
  over scene.objects to inject each LIBEROObject's sampled position.
  createObjectInSimulator() is a no-op (objects already exist via BDDL).

* step() advances MuJoCo physics with a zero action. For full policy
  evaluation, see step_with_action() and eval.py.

* getProperties() reads position and orientation back from the live MuJoCo
  data so that Scenic's temporal monitors and require-always constraints work
  correctly on the evolving sim state.

Coordinate systems
──────────────────
LIBERO/MuJoCo world frame:
  +x  →  forward (away from robot base)
  +y  →  left
  +z  →  up
  Table surface ≈ z = TABLE_Z (see libero_model.scenic for the exact value)

Scenic positions are passed as scenic.core.vectors.Vector(x, y, z) which map
directly to the MuJoCo world frame — no coordinate transform needed.

Quaternion convention
─────────────────────
robosuite / MuJoCo uses (x, y, z, w) order (scalar last).
Scenic's Orientation is a scipy Rotation; .as_quat() returns (x, y, z, w).
For objects that Scenic has not given an explicit orientation, we use the
per-class canonical rotation recorded in DEFAULT_ORIENTATIONS below.
"""

from __future__ import annotations

import logging
import pathlib
import re
from typing import Any

import numpy as np
from scenic.core.simulators import Simulation, Simulator
from scenic.core.vectors import Vector
from scipy.spatial.transform import Rotation as _Rotation

from libero_infinity.asset_metadata import cradle_tilt_quat as _cradle_tilt_quat
from libero_infinity.asset_metadata import distractor_table_rest_quat as _distractor_table_rest_quat
from libero_infinity.asset_metadata import is_cradle_seatable as _is_cradle_seatable
from libero_infinity.asset_metadata import spawn_clearance as _spawn_clearance
from libero_infinity.asset_metadata import surface_spawn_z as _shared_surface_spawn_z
from libero_infinity.asset_registry import get_dimensions
from libero_infinity.planner.axes import LIBERO_BACKGROUND_TEXTURES
from libero_infinity.validation_errors import (  # noqa: F401 — re-exported for callers
    MAX_VISIBILITY_RETRIES,
    RECOVERY_STRATEGY,
    CollisionError,
    InfeasibleScenarioError,
    ScenarioValidationError,
    VisibilityError,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants (metres, matching LIBERO arena XML / libero_model.scenic)
# ---------------------------------------------------------------------------
TABLE_Z = 0.82  # table surface height in MuJoCo world frame (floor → 0)
TABLE_X_MIN = -0.40
TABLE_X_MAX = 0.40
TABLE_Y_MIN = -0.30
TABLE_Y_MAX = 0.30

# Objects whose LIBERO default z exceeds this threshold are considered "elevated"
# (e.g. starting on a stove or cabinet shelf).  When their XY position is being
# perturbed to the table area, their z is recomputed from the inferred root surface
# rather than copied from the LIBERO default placement.
# 0.15 m above TABLE_Z is above any table-top object's normal resting height and
# below typical shelf/stove surface heights (~0.20 m above the table).
ELEVATED_Z_THRESHOLD = TABLE_Z + 0.15

# Physics-settling validation thresholds — calibrated empirically via
# scripts/calibrate_drift.py (see calibration_results.json).
# Objects that violate these bounds after env.reset() + settling trigger a retry.
MAX_SETTLE_XY_DRIFT = 0.20  # max allowed xy drift from the Scenic-sampled position (m)
MAX_SETTLE_Z_DROP = 0.08  # max allowed z drop (objects falling through fixtures)
MAX_SETTLE_ROT_DRIFT = np.deg2rad(35.0)  # max rotation from default LIBERO pose (35° ≈ 0.61 rad)
MIN_SETTLED_Z = TABLE_Z - 0.05  # min z after settling; below this = fallen off the table


def _is_workspace_surface_body(body_name: str) -> bool:
    """True if ``body_name`` is the scene's workspace support surface.

    A table-spawned object (task object or distractor) resting on the
    workspace surface is in *expected* persistent contact with it under
    gravity — that contact must not be flagged as a bad-placement overlap by
    ``_validate_settled_positions``.

    LIBERO names every workspace surface consistently: the robosuite default
    arena body is ``table``; suite arenas use ``<room>_table`` (``kitchen_table``,
    ``living_room_table``, ``study_table``, ``main_table``); floor-based scenes
    use ``floor``. The previous check only matched ``table``/``table*`` and so
    silently failed for every multi-word table name (e.g. ``study_table``),
    which made distractors that legitimately rest on the STUDY-scene table read
    as overlaps and exhausted the reset retries (run3 g5 RCA). Matching the
    naming convention covers all suites without hardcoding a per-scene list.
    """
    name = body_name.lower()
    return name == "table" or name.startswith("table") or name.endswith("table") or "floor" in name


# Regex matching robosuite's compiled ``<size .../>`` element. robosuite's
# base.xml hardcodes ``<size nconmax="5000" njmax="5000"/>``, which caps the
# MuJoCo contact (``nconmax``) and constraint (``njmax``) arenas at 5000. A
# single dense scene that momentarily exceeds that cap triggers MuJoCo's
# ``Too many contacts (ncon = 5000)`` warning and SILENTLY TRUNCATES the
# excess contacts — the dropped contacts mean missing constraint forces, i.e.
# corrupted physics for that step (penetration / ejection). This is distinct
# from the cross-sample accumulation overflow that PR #33 addressed via
# process-per-sample; this is the single-dense-scene mode.
_SIZE_ELEM_RE = re.compile(r"<size\b[^>]*/?>")
_NCONMAX_RE = re.compile(r'\bnconmax\s*=\s*"[^"]*"')
_NJMAX_RE = re.compile(r'\bnjmax\s*=\s*"[^"]*"')


def _autosize_contact_arena(xml: str) -> str:
    """MJCF processor: let MuJoCo size the contact/constraint arena dynamically.

    Rewrites any explicit ``nconmax`` / ``njmax`` on the model's ``<size>``
    element to ``-1`` — MuJoCo's documented sentinel for *automatic* arena
    sizing (the engine grows the contact/constraint buffers from the arena
    memory pool as the scene requires, rather than capping at a fixed count).
    This is the principled, scene-complexity-adaptive replacement for the
    hardcoded 5000 cap: no magic constant, the ceiling scales to whatever the
    scene actually produces.

    Verified physics-invariant for normal scenes: for any scene whose contact
    count never reached the old cap, the contacts detected and solved are
    bit-identical (the buffer ceiling is the only thing that changes). The
    behaviour differs ONLY for scenes that previously overflowed, where the
    formerly-truncated contacts are now retained — the intended correctness
    fix. Installed via robosuite's ``set_xml_processor`` hook so it runs inside
    every ``_initialize_sim`` rebuild, with no edits to vendored assets.
    """

    def _rewrite_size(m: "re.Match[str]") -> str:
        elem = m.group(0)
        elem = _NCONMAX_RE.sub('nconmax="-1"', elem)
        elem = _NJMAX_RE.sub('njmax="-1"', elem)
        return elem

    return _SIZE_ELEM_RE.sub(_rewrite_size, xml)


# ---------------------------------------------------------------------------
# Canonical upright orientations per asset class (robosuite (x,y,z,w) format)
# Most GoogleScannedObjects ship with rotation=(π/2) around x so they stand
# upright on the table.  Values match the defaults in google_scanned_objects.py.
# ---------------------------------------------------------------------------
_QUAT_UPRIGHT_X90 = np.array(
    [np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)], dtype=float
)  # 90° rotation around x axis

DEFAULT_ORIENTATIONS: dict[str, np.ndarray] = {
    "_default": _QUAT_UPRIGHT_X90,
    "simple_rack": np.array([0.0, 0.0, 0.0, 1.0]),  # flat (no rotation)
}
EXPECTED_PANDA_ARM_DOF = 7


def _footprint_clearance_xy(
    dims_a: tuple[float, float, float],
    dims_b: tuple[float, float, float],
) -> float:
    """Minimum centre-to-centre xy distance before two footprints overlap."""
    radius_a = float(np.hypot(dims_a[0], dims_a[1])) / 2.0
    radius_b = float(np.hypot(dims_b[0], dims_b[1])) / 2.0
    return radius_a + radius_b


def _axis_overlap_xy(
    pos_a: np.ndarray,
    dims_a: tuple[float, float, float],
    pos_b: np.ndarray,
    dims_b: tuple[float, float, float],
    margin: float = 0.0,
) -> bool:
    """Whether two settled axis-aligned xy footprints overlap."""
    min_dx = (dims_a[0] + dims_b[0]) / 2.0 + margin
    min_dy = (dims_a[1] + dims_b[1]) / 2.0 + margin
    dx = abs(float(pos_a[0] - pos_b[0]))
    dy = abs(float(pos_a[1] - pos_b[1]))
    return dx < min_dx and dy < min_dy


def _scenic_quat(scenic_orientation) -> np.ndarray:
    """Convert a Scenic Orientation to scipy xyzw quaternion.

    NOTE: returns xyzw (scalar-last), NOT wxyz. Callers that write to
    MuJoCo qpos must convert: [q[3], q[0], q[1], q[2]].
    """
    try:
        return np.array(scenic_orientation.as_quat(), dtype=float)
    except Exception:
        return DEFAULT_ORIENTATIONS["_default"].copy()


def _surface_spawn_z(
    surface_z: float,
    asset_class: str,
    surface_class: str | None = None,
    *,
    distractor: bool = False,
) -> float:
    """Resolved object-centre z for spawning directly on a root surface.

    Thin delegate to the shared, pure :func:`asset_metadata.surface_spawn_z` so
    the simulator and the Scenic renderer resolve byte-identical spawn z for the
    same ``(surface_z, asset_class, surface_class, distractor)`` — the G4
    family-C ``pose_tolerance`` invariant relies on both sides agreeing (see
    ``asset_metadata`` docstring and
    ``rca/stage1_g4_consistency_pose_frame_mismatch.md``). ``distractor`` routes
    a flat distractor onto a fixture's settle-measured rest height instead of its
    raw collision-edge ``top_z`` (WS-1 open-frame seating fix); it must mirror the
    renderer's distractor branch exactly.
    """
    return _shared_surface_spawn_z(surface_z, asset_class, surface_class, distractor=distractor)


def _bddl_contained_object_names(bddl_path: str) -> set[str]:
    """Return objects whose authored initial relation is true containment."""
    try:
        from libero_infinity.task_config import TaskConfig

        cfg = TaskConfig.from_bddl(bddl_path)
    except Exception:
        log.debug(
            "Could not parse BDDL containment metadata from %s",
            bddl_path,
            exc_info=True,
        )
        return set()
    return {obj.instance_name for obj in cfg.movable_objects if obj.contained}


def _infer_root_surface_z(scene_objects, default_pose: dict[str, np.ndarray]) -> float:
    """Infer the canonical root support height from default LIBERO placements."""
    surface_candidates: list[float] = []
    for obj in scene_objects:
        libero_name = getattr(obj, "libero_name", None)
        if not libero_name or libero_name not in default_pose:
            continue
        if getattr(obj, "support_parent_name", ""):
            continue
        # Invert the spawn model: surface = settled body-origin z − clearance.
        # Using the SAME clearance that ``_surface_spawn_z`` applies keeps the
        # round trip consistent — for a table object this recovers ≈ TABLE_Z, so
        # re-spawning an elevated object onto the inferred surface reproduces its
        # natural settled z rather than a bounding-box approximation.
        clearance = _spawn_clearance(getattr(obj, "asset_class", "_default"))
        surface_candidates.append(float(default_pose[libero_name][2]) - clearance)
    if not surface_candidates:
        return TABLE_Z
    return float(np.median(surface_candidates))


def _visibility_anchor_points(
    center: np.ndarray,
    dims: tuple[float, float, float],
) -> list[np.ndarray]:
    """Anchor points used to approximate object visibility."""
    half_x = max(float(dims[0]) * 0.30, 0.01)
    half_y = max(float(dims[1]) * 0.30, 0.01)
    half_z = max(float(dims[2]) * 0.20, 0.01)
    offsets = [
        np.array((0.0, 0.0, 0.0), dtype=float),
        np.array((half_x, 0.0, 0.0), dtype=float),
        np.array((-half_x, 0.0, 0.0), dtype=float),
        np.array((0.0, half_y, 0.0), dtype=float),
        np.array((0.0, -half_y, 0.0), dtype=float),
        np.array((0.0, 0.0, half_z), dtype=float),
    ]
    return [center + offset for offset in offsets]


def _geom_world_aabb(sim, geom_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a world-frame AABB for a MuJoCo geom when enough data is exposed."""
    model = sim.model
    data = sim.data
    try:
        center = np.array(data.geom_xpos[geom_id], dtype=float)
        rot = np.array(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    except Exception:
        return None

    try:
        import mujoco

        mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    except Exception:
        mesh_type = 7

    geom_type = int(model.geom_type[geom_id])
    if geom_type == mesh_type and int(model.geom_dataid[geom_id]) >= 0:
        mesh_id = int(model.geom_dataid[geom_id])
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        if vert_num > 0:
            verts = np.array(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=float)
            points = center + verts @ rot.T
            return points.min(axis=0), points.max(axis=0)

    try:
        radius = float(model.geom_rbound[geom_id])
    except Exception:
        return None
    extent = np.array((radius, radius, radius), dtype=float)
    return center - extent, center + extent


def _body_world_aabb(sim, object_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a live world-frame AABB for an object/fixture body prefix."""
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for geom_id in range(int(sim.model.ngeom)):
        body_name = sim.model.body_id2name(sim.model.geom_bodyid[geom_id]) or ""
        if body_name != object_name and not body_name.startswith(f"{object_name}_"):
            continue
        aabb = _geom_world_aabb(sim, geom_id)
        if aabb is None:
            continue
        geom_min, geom_max = aabb
        mins.append(geom_min)
        maxs.append(geom_max)
    if not mins:
        return None
    return np.min(mins, axis=0), np.max(maxs, axis=0)


def _body_origin_z(sim, object_name: str) -> float | None:
    """World-frame z of a movable's body origin (``body_xpos``), or ``None``.

    The body origin is the frame ``spawn_clearance`` / ``surface_spawn_z`` are
    expressed in (``body_xpos_z − surface_z``), so the restack compares and lifts
    in this frame to stay in lockstep with the renderer's emitted spawn z.
    """
    for cand in (object_name, f"{object_name}_main"):
        try:
            body_id = sim.model.body_name2id(cand)
        except Exception:
            continue
        return float(sim.data.body_xpos[body_id][2])
    return None


def _visibility_anchor_points_for_body(
    *,
    sim,
    object_name: str,
    center: np.ndarray,
    dims: tuple[float, float, float],
) -> list[np.ndarray]:
    """Anchor visibility checks on the live object surface instead of its interior."""
    aabb = _body_world_aabb(sim, object_name)
    if aabb is None:
        return _visibility_anchor_points(center, dims)

    min_corner, max_corner = aabb
    if not np.all(np.isfinite(min_corner)) or not np.all(np.isfinite(max_corner)):
        return _visibility_anchor_points(center, dims)
    if np.any(max_corner <= min_corner):
        return _visibility_anchor_points(center, dims)

    mid = (min_corner + max_corner) / 2.0
    xs = (float(min_corner[0]), float(mid[0]), float(max_corner[0]))
    ys = (float(min_corner[1]), float(mid[1]), float(max_corner[1]))
    top_z = float(max_corner[2])
    mid_z = float(mid[2])

    anchors = [np.array((x, y, top_z), dtype=float) for x in xs for y in ys]
    anchors.extend(
        np.array((x, y, mid_z), dtype=float)
        for x in (float(min_corner[0]), float(max_corner[0]))
        for y in (float(min_corner[1]), float(max_corner[1]))
    )
    return anchors


def _visibility_depth_tolerance(
    *,
    sim,
    object_name: str,
    base_tolerance: float = 0.05,
) -> float:
    """Depth slack for anchors lying behind the visible front surface."""
    aabb = _body_world_aabb(sim, object_name)
    if aabb is None:
        return base_tolerance
    min_corner, max_corner = aabb
    if not np.all(np.isfinite(min_corner)) or not np.all(np.isfinite(max_corner)):
        return base_tolerance
    extent = np.maximum(max_corner - min_corner, 0.0)
    front_surface_slack = min(0.20, max(0.02, float(np.max(extent)) * 1.5))
    return base_tolerance + front_surface_slack


def _anchor_visible(
    *,
    point: np.ndarray,
    world_to_pixel: np.ndarray,
    world_to_camera: np.ndarray,
    depth_map: np.ndarray,
    image_height: int,
    image_width: int,
    depth_tolerance: float = 0.05,
) -> bool:
    """Whether a projected 3-D anchor is inside the frame and not depth-occluded."""
    hom = np.concatenate([point.astype(float), np.array([1.0])], axis=0)
    camera_point = world_to_camera @ hom
    if camera_point[2] <= 1e-6:
        return False

    pixel_hom = world_to_pixel @ hom
    if pixel_hom[2] <= 1e-6:
        return False
    col = int(round(float(pixel_hom[0] / pixel_hom[2])))
    row = int(round(float(pixel_hom[1] / pixel_hom[2])))
    if row < 0 or row >= image_height or col < 0 or col >= image_width:
        return False

    observed_depth = float(depth_map[row, col])
    if not np.isfinite(observed_depth):
        return False
    return observed_depth + depth_tolerance >= float(camera_point[2])


def _camera_transforms(
    *,
    sim,
    camera_name: str,
    camera_height: int,
    camera_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return world->pixel and world->camera transforms for a MuJoCo camera."""
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = float(sim.model.cam_fovy[cam_id])
    focal = 0.5 * camera_height / np.tan(fovy * np.pi / 360.0)
    intrinsic = np.array(
        [
            [focal, 0.0, camera_width / 2.0],
            [0.0, focal, camera_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    camera_pos = np.array(sim.data.cam_xpos[cam_id], dtype=float)
    camera_rot = np.array(sim.data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
    extrinsic = np.eye(4, dtype=float)
    extrinsic[:3, :3] = camera_rot
    extrinsic[:3, 3] = camera_pos
    axis_correction = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    corrected_extrinsic = extrinsic @ axis_correction
    world_to_camera = np.linalg.inv(corrected_extrinsic)

    intrinsic_4 = np.eye(4, dtype=float)
    intrinsic_4[:3, :3] = intrinsic
    world_to_pixel = intrinsic_4 @ world_to_camera
    return world_to_pixel, world_to_camera


_NOISE_KIND_OFFSETS: dict[str, int] = {
    "gaussian_noise": 0,
    "shot_noise": 1,
    "impulse_noise": 2,
    "brightness_jitter": 3,
    "contrast_jitter": 4,
}


def _apply_image_corruption(
    image: np.ndarray,
    kind: str,
    severity: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a sensor-noise corruption to a single RGB image.

    Severity follows the *Common Image Corruptions* convention (1..5).
    All transforms preserve dtype (uint8 in, uint8 out) and shape (H, W, 3).
    Implementations are deliberately lightweight and dependency-free —
    we use only numpy here to avoid pulling in scipy.ndimage / opencv at
    the simulator-import level. The corruption family follows the
    *Common Image Corruptions* taxonomy (Hendrycks & Dietterich, 2019)
    so users can swap in the higher-fidelity ``imagecorruptions``
    package later if needed.
    """
    if image.ndim != 3 or image.shape[-1] not in (3, 4) or image.dtype != np.uint8:
        return image
    img = image[..., :3].astype(np.float32)

    def _rng(kind_offset: int) -> np.random.Generator:
        # Combine the optional per-scene ``seed`` with kind/severity so the
        # realised noise pattern varies across scenes when a non-None seed
        # is supplied (E5 audit fix). With ``seed=None`` the legacy
        # severity-only seeding is preserved for backward compatibility.
        if seed is None:
            return np.random.default_rng(severity + kind_offset)
        return np.random.default_rng((int(seed), int(severity), int(kind_offset)))

    if kind == "gaussian_noise":
        sigma = [4.0, 8.0, 16.0, 24.0, 32.0][severity - 1]
        rng = _rng(_NOISE_KIND_OFFSETS["gaussian_noise"])
        noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
        out = img + noise
    elif kind == "shot_noise":
        scale = [60.0, 25.0, 12.0, 5.0, 3.0][severity - 1]
        rng = _rng(_NOISE_KIND_OFFSETS["shot_noise"])
        out = rng.poisson(np.clip(img, 0, 255) / 255.0 * scale) / scale * 255.0
    elif kind == "impulse_noise":
        prob = [0.01, 0.02, 0.04, 0.08, 0.15][severity - 1]
        rng = _rng(_NOISE_KIND_OFFSETS["impulse_noise"])
        mask = rng.random(img.shape[:2])
        out = img.copy()
        out[mask < prob / 2.0] = 0.0
        out[mask > 1.0 - prob / 2.0] = 255.0
    elif kind == "gaussian_blur":
        sigma = [0.6, 1.0, 1.5, 2.5, 4.0][severity - 1]
        out = _gaussian_blur(img, sigma)
    elif kind == "motion_blur":
        # Approximate motion blur as a horizontal moving-average box.
        radius = [2, 3, 5, 8, 12][severity - 1]
        out = _moving_average(img, radius=radius, axis=1)
    elif kind == "defocus_blur":
        sigma = [0.8, 1.2, 1.8, 2.6, 4.0][severity - 1]
        out = _gaussian_blur(img, sigma)
    elif kind == "jpeg_compression":
        # Severity-driven coarse quantisation in YCbCr-ish space; cheap
        # standin for a real JPEG round-trip without bringing in PIL.
        steps = [4, 8, 16, 32, 48][severity - 1]
        out = (np.round(img / steps) * steps).astype(np.float32)
    elif kind == "brightness_jitter":
        delta = [10, 25, 45, 70, 100][severity - 1]
        rng = _rng(_NOISE_KIND_OFFSETS["brightness_jitter"])
        shift = rng.uniform(-delta, delta)
        out = img + shift
    elif kind == "contrast_jitter":
        gain = [0.95, 0.9, 0.8, 0.65, 0.5][severity - 1]
        rng = _rng(_NOISE_KIND_OFFSETS["contrast_jitter"])
        scale = rng.uniform(gain, 1.0 / gain)
        mean = float(img.mean())
        out = (img - mean) * scale + mean
    elif kind == "saturation_jitter":
        gain = [0.9, 0.75, 0.5, 0.25, 0.1][severity - 1]
        gray = img.mean(axis=-1, keepdims=True)
        out = gray + (img - gray) * gain
    else:
        return image  # unknown kind; pass-through
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    if image.shape[-1] == 4:
        out_rgba = image.copy()
        out_rgba[..., :3] = out
        return out_rgba
    return out


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur using a numpy 1-D convolution."""
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2.0 * max(sigma, 1e-6) ** 2))
    kernel /= kernel.sum()
    out = img.copy()
    for axis in (0, 1):
        out = np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode="same"),
            axis=axis,
            arr=out,
        )
    return out


def _moving_average(img: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Approximate motion-blur kernel along a single axis."""
    width = 2 * radius + 1
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.apply_along_axis(
        lambda v: np.convolve(v, kernel, mode="same"),
        axis=axis,
        arr=img,
    )


def _real_depth_map(sim, depth_map: np.ndarray) -> np.ndarray:
    """Convert MuJoCo's normalized depth image to metric depth.

    Defensively sanitises the input rather than ``assert``-ing — NaN
    or out-of-range pixels (occasionally produced by EGL drivers on
    edge cases) would otherwise crash the rollout. Anomalous values
    are clipped into ``[0, 1]`` and NaNs replaced with ``1.0`` (far
    plane) so downstream consumers see a finite metric depth.
    """
    arr = np.asarray(depth_map, dtype=float)
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=1.0, posinf=1.0, neginf=0.0)
    if arr.min() < 0.0 or arr.max() > 1.0:
        log.debug(
            "depth_map out of [0,1] (min=%.4f max=%.4f); clipping",
            float(arr.min()),
            float(arr.max()),
        )
        arr = np.clip(arr, 0.0, 1.0)
    extent = float(sim.model.stat.extent)
    far = float(sim.model.vis.map.zfar) * extent
    near = float(sim.model.vis.map.znear) * extent
    return near / (1.0 - arr * (1.0 - near / far))


# ---------------------------------------------------------------------------
# LIBEROSimulator
# ---------------------------------------------------------------------------


class LIBEROSimulator(Simulator):
    """Scenic Simulator subclass that wraps the LIBERO environment factory.

    Parameters
    ----------
    bddl_path:
        Absolute path to the BDDL task file.
    env_kwargs:
        Extra kwargs forwarded to OffScreenRenderEnv (cameras, render flags, …).
    """

    def __init__(
        self,
        bddl_path: str,
        env_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.bddl_path = bddl_path
        self.env_kwargs = env_kwargs or {}

    def createSimulation(self, scene, **kwargs) -> "LIBEROSimulation":
        """Required by Scenic. Instantiate a LIBEROSimulation for the sampled scene."""
        return LIBEROSimulation(
            scene,
            bddl_path=self.bddl_path,
            env_kwargs=self.env_kwargs,
            **kwargs,
        )

    def simulate(
        self,
        scene,
        maxSteps: int = 500,
        verbosity: int = 0,
        render_live: str | None = None,
        camera: str = "agentview",
        **kwargs,
    ) -> "LIBEROSimulation":
        """Run one LIBERO episode: setup → step × maxSteps → return simulation.

        This bypasses Scenic's eager Simulation.__init__ loop (which would
        re-evaluate require constraints at every physics step and reject on
        soft-constraint misses).  For evaluation use eval.py instead.

        Parameters
        ----------
        render_live : None | "cv2" | "viewer"
            None     — no live display (default; headless / batch use)
            "cv2"    — stream rendered frames to an OpenCV window.  Needs a
                       display (DISPLAY env var set, e.g. ":1" or forwarded
                       via SSH -X).  No extra packages beyond opencv-python.
            "viewer" — launch MuJoCo's interactive passive viewer in a
                       background thread.  Gives full orbit/pan/zoom GUI.
                       Also needs a display (GLFW/X11).
        camera : str
            Camera name to show in "cv2" mode (default "agentview").
        """
        sim = self.createSimulation(scene, maxSteps=maxSteps, verbosity=verbosity, **kwargs)
        sim.setup()

        if render_live == "viewer":
            self._simulate_viewer(sim)
        elif render_live == "cv2":
            self._simulate_cv2(sim, camera=camera)
        else:
            for _ in range(sim._max_steps):
                sim.step()
                if sim._done:
                    break

        return sim

    def _simulate_viewer(self, sim: "LIBEROSimulation") -> None:
        """Run episode with MuJoCo's interactive passive viewer.

        The viewer opens in a background thread; the main thread drives
        physics.  `handle.sync()` pushes each new physics state into the
        viewer.  The episode ends when done or the user closes the window.

        Requires: DISPLAY env var set (X11); glfw installed (already a
        mujoco dependency).  On macOS use `mjpython` instead of `python`.
        """
        try:
            import mujoco.viewer as _mjv
        except ImportError as e:
            raise RuntimeError("mujoco.viewer not available — install mujoco >= 2.3.3") from e

        mjmodel, mjdata = sim.mj_handles

        with _mjv.launch_passive(mjmodel, mjdata) as handle:
            for _ in range(sim._max_steps):
                if not handle.is_running():
                    break
                sim.step()
                with handle.lock():
                    handle.sync()
                if sim._done:
                    break

    def _simulate_cv2(self, sim: "LIBEROSimulation", camera: str = "agentview") -> None:
        """Run episode streaming frames to an OpenCV window.

        Each frame is read from obs["{camera}_image"] (an EGL-rendered
        numpy uint8 array in OpenGL convention: origin bottom-left, RGB).
        We flip vertically and swap channels to BGR for cv2.

        Requires: DISPLAY env var set; opencv-python installed.
        Press 'q' to quit early.
        """
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("opencv-python not installed — uv pip install opencv-python") from e

        win = f"LIBERO — {camera}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        try:
            for _ in range(sim._max_steps):
                sim.step()
                obs = sim.last_obs
                if obs is not None:
                    frame = obs.get(f"{camera}_image")
                    if frame is not None:
                        # OpenGL origin is bottom-left; cv2 expects top-left.
                        # obs is RGB; cv2.imshow expects BGR.
                        cv2.imshow(win, frame[::-1, :, ::-1])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if sim._done:
                    break
        finally:
            cv2.destroyWindow(win)

    def destroy(self):
        """No shared simulator resources to release."""
        super().destroy()


# ---------------------------------------------------------------------------
# LIBEROSimulation
# ---------------------------------------------------------------------------


class LIBEROSimulation(Simulation):
    """Scenic Simulation that executes one LIBERO episode.

    Lifecycle (called by Scenic's simulate() loop)
    ────────────────────────────────────────────────
    1. setup()                 — init env, reset, inject Scenic poses
    2. step() × N             — advance physics
    3. getProperties() × M    — read back state for monitors
    4. destroy()               — close env

    The `scene` object (scenic.core.scenarios.Scene) carries:
      scene.objects   — list of all Scenic objects with sampled properties
      scene.params    — dict of sampled global parameters (task name, etc.)
      scene.egoObject — the designated ego agent (not used for LIBERO arm)
    """

    def __init__(
        self,
        scene,
        *,
        bddl_path: str,
        env_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        # We do NOT call super().__init__() here.
        #
        # Scenic's Simulation.__init__ runs the entire simulation loop eagerly
        # (setup → N steps → requires checked at each step → result).  That
        # design works for Scenic's built-in simulate() driver but is
        # incompatible with our lazy lifecycle:
        #
        #   sim.createSimulation(scene, ...) → episode
        #   episode.setup()                  → init env, inject positions
        #   episode.step() × N               → physics advance
        #   episode.destroy()                → cleanup
        #
        # Additionally, Scenic re-evaluates all require / require[p] statements
        # at every simulation step, including soft constraints (require[0.8])
        # that were only meant to bias the sampling distribution.  This causes
        # spurious RejectSimulationException on 20 % of scenes.
        #
        # Minimal Scenic Simulation state (attributes inspected externally).
        self.scene = scene
        self.objects = list(scene.objects) if scene is not None else []
        self.agents: list = []
        self.result = None
        self.currentTime = 0
        self.timestep = float(kwargs.get("timestep") or 0.05)
        self.verbosity = int(kwargs.get("verbosity") or 0)
        self.name = str(kwargs.get("name") or "")
        self.worker_num = 0

        # LIBERO-specific state.
        self.bddl_path = bddl_path
        self.env_kwargs = env_kwargs or {}
        self.libero_env = None
        self._last_obs: dict | None = None
        self._done: bool = False
        self._max_steps = int(kwargs.get("maxSteps") or 500)

        # Snapshot of model arrays taken at env-creation time so the
        # ``_apply_*_perturbation`` passes can restore the canonical baseline
        # before each application. This prevents the cumulative-mutation bug
        # where additive writes (cam_pos += dx, light_diffuse *=
        # intensity, mat_texid = tex_id) drift the model further from baseline
        # on every reuse of the same env.
        self._model_baseline: dict | None = None

    # ------------------------------------------------------------------
    # setup — called once before stepping begins
    # ------------------------------------------------------------------

    def setup(self):
        """Initialise LIBERO env and inject Scenic-sampled object positions.

        Flow
        ────
        1. Build OffScreenRenderEnv from self.bddl_path
        2. env.reset() — loads BDDL scene with default placements
        3. For each LIBEROObject in scene.objects, override its joint qpos
           with the Scenic-sampled position / orientation
        4. mj_forward() — settle physics so collision detection is fresh

        We do NOT call super().setup() here because the default implementation
        calls createObjectInSimulator() for each Scenic object, but our objects
        are already instantiated in the LIBERO environment via the BDDL file.
        Instead we inject positions directly after env.reset().
        """
        from libero.libero.envs.env_wrapper import OffScreenRenderEnv

        env_cfg = dict(
            bddl_file_name=self.bddl_path,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="agentview",
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=128,
            camera_widths=128,
            camera_depths=True,
            control_freq=20,
            horizon=self._max_steps,
            ignore_done=False,
            hard_reset=True,
        )
        env_cfg.update(self.env_kwargs)

        # ── handle distractor objects from Scenic scene ─────────────────
        effective_bddl = self.bddl_path
        self._distractor_bddl_path = None
        self._active_distractor_names: set[str] = set()
        params = getattr(self.scene, "params", {})

        # Auto-detect: scan scene objects for distractor_* names
        n_distractors = params.get("n_distractors", 0)
        if isinstance(n_distractors, float):
            n_distractors = int(n_distractors)

        distractor_objs: list[tuple[str, str]] = []
        for obj in self.scene.objects:
            name = getattr(obj, "libero_name", "")
            if name.startswith("distractor_"):
                asset_cls = getattr(obj, "asset_class", "")
                # Read per-slot class from scene.params if available
                slot_cls = params.get(f"{name}_class", asset_cls)
                if slot_cls:
                    distractor_objs.append((name, str(slot_cls)))

        # Sort by name for deterministic ordering (distractor_0 < distractor_1 ...)
        distractor_objs.sort(key=lambda x: x[0])

        # Take only the first n_distractors (rest are inactive Scenic slots)
        active_distractors = distractor_objs[:n_distractors] if n_distractors > 0 else []

        # Fallback: also check legacy distractor_objects param
        if not active_distractors:
            legacy_specs = params.get("distractor_objects")
            if legacy_specs and isinstance(legacy_specs, list):
                active_distractors = list(legacy_specs)

        if active_distractors:
            import tempfile

            from libero_infinity.bddl_preprocessor import add_distractor_objects

            bddl_text = pathlib.Path(self.bddl_path).read_text()
            patched = add_distractor_objects(bddl_text, active_distractors)
            f = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".bddl",
                prefix="libero_inf_dist_",
                delete=False,
            )
            f.write(patched)
            f.close()
            effective_bddl = f.name
            self._distractor_bddl_path = f.name
            env_cfg["bddl_file_name"] = effective_bddl
            self._active_distractor_names = {name for name, _ in active_distractors}
        else:
            env_cfg["bddl_file_name"] = self.bddl_path

        log.debug("LIBEROSimulation.setup: creating env from %s", effective_bddl)
        self.libero_env = OffScreenRenderEnv(**env_cfg)
        # Auto-size the MuJoCo contact/constraint arena before the sim is
        # (re)built. robosuite's base.xml hardcodes nconmax/njmax=5000; a single
        # dense scene that exceeds that cap truncates contacts and corrupts the
        # step's physics (+ emits the ``ncon = 5000`` warning spam seen in the
        # run3 sweep). The processor runs inside ``_initialize_sim`` on the
        # hard-reset rebuild below, so the model we actually use has a dynamic
        # arena. See ``_autosize_contact_arena``.
        try:
            self.libero_env.env.set_xml_processor(_autosize_contact_arena)
        except AttributeError:  # pragma: no cover — older robosuite without the hook
            log.warning("robosuite env has no set_xml_processor; contact arena stays capped")
        self._last_obs = self.libero_env.reset()

        # Snapshot canonical model arrays before any perturbation runs so the
        # apply pass below — and any future re-application — always starts
        # from the XML-loaded baseline rather than a previously-perturbed
        # state. See ``_capture_model_baseline`` docstring.
        self._capture_model_baseline()

        # ── capture LIBERO's default pose for each object / fixture ────
        # After env.reset(), LIBERO places objects at correct z heights
        # via its region samplers.  We preserve those z values and only
        # override x, y from Scenic.  This avoids hardcoding TABLE_Z.
        sim_data = self.libero_env.env.sim.data
        sim_model = self.libero_env.env.sim.model
        default_pose: dict[str, np.ndarray] = {}
        default_rot: dict[str, np.ndarray] = {}
        for obj in self.scene.objects:
            libero_name = getattr(obj, "libero_name", None)
            if not libero_name:
                continue
            # Distractors are injected via patched BDDL and do not have a meaningful
            # canonical reset pose. Their qpos often starts near the origin before
            # Scenic injection, which would corrupt root-surface inference.
            if libero_name.startswith("distractor_"):
                continue
            joint_name = f"{libero_name}_joint0"
            try:
                qpos = sim_data.get_joint_qpos(joint_name)
                default_pose[libero_name] = np.array(qpos[:3], dtype=float)
                # MuJoCo free-joint qpos stores quaternion as wxyz [qw,qx,qy,qz].
                # Convert to scipy xyzw [qx,qy,qz,qw] so all downstream code
                # (from_quat, as_quat, Rotation composition) uses one convention.
                _q_wxyz = qpos[3:7]
                default_rot[libero_name] = np.array(
                    [_q_wxyz[1], _q_wxyz[2], _q_wxyz[3], _q_wxyz[0]], dtype=float
                )  # → xyzw
            except Exception:
                for body_name in (libero_name, libero_name + "_main"):
                    try:
                        body_id = sim_model.body_name2id(body_name)
                        default_pose[libero_name] = np.array(
                            sim_data.body_xpos[body_id][:3],
                            dtype=float,
                        )
                        default_rot[libero_name] = np.array(
                            sim_data.body_xmat[body_id],
                            dtype=float,
                        ).reshape(3, 3)
                        break
                    except Exception:
                        continue

        self._canonical_rot = dict(default_rot)
        root_surface_z = _infer_root_surface_z(self.scene.objects, default_pose)
        contained_object_names = _bddl_contained_object_names(effective_bddl)

        # Restore the canonical XML-loaded model state ONCE, before any
        # perturbation writes (robot, object/fixture injection, articulation,
        # camera/lighting/texture/background). This gives the additive (`+=`)
        # and multiplicative (`*=`) env-perturbation passes a clean baseline
        # while — critically — running BEFORE the fixture-injection body_pos
        # writes below. A jointless fixture (e.g. ``desk_caddy_1``,
        # ``wooden_two_layer_shelf_1``) can only be relocated by writing
        # ``sim.model.body_pos`` directly (see ``_inject_object_pose``
        # fallback). The previous ordering restored the baseline *after* the
        # injection loop, which reverted those fixture writes to the XML
        # default — the fixture then "drifted" the full sample-vs-default
        # distance (~0.20 m) and the settle validator rejected every retry
        # (g5 STUDY_SCENE reset failures, run3). Restoring first preserves the
        # injection: the env-perturbation passes only touch cam_pos / light_*
        # / mat_texid, never body_pos, so they cannot clobber it.
        self._restore_model_baseline()
        self._apply_robot_perturbation()

        # ── inject Scenic-sampled positions ───────────────────────────────
        n_injected = 0
        injected_targets: dict[str, np.ndarray] = {}
        object_dimensions: dict[str, tuple[float, float, float]] = {}
        support_parent_names: dict[str, str] = {}
        table_spawned_names: set[str] = set()
        movable_names: set[str] = set()
        for obj in self.scene.objects:
            libero_name = getattr(obj, "libero_name", None)
            if not libero_name:
                continue
            # Only graspable LIBEROObjects go into movable_names.
            # LIBEROFixture instances (graspable=False) must stay OUT so that
            # object-fixture contacts are flagged by _validate_settled_positions
            # rather than silently skipped.
            if getattr(obj, "graspable", True):
                movable_names.add(libero_name)
            support_parent = getattr(obj, "support_parent_name", "")
            if support_parent:
                support_parent_names[libero_name] = support_parent
            # Skip inactive distractor slots (exist in Scenic but not in MuJoCo)
            if libero_name.startswith("distractor_"):
                if libero_name not in self._active_distractor_names:
                    continue
            pos = np.array(obj.position, dtype=float)  # (x, y, z) MuJoCo frame
            preserve_default_z = bool(getattr(obj, "preserve_default_z", True))
            # True containment comes from authored BDDL "(In ...)" relations.
            # support_parent_name is broader: it is also used for ordinary
            # "On" support stacks such as bowl-on-ramekin or bowl-on-stove.
            is_contained = libero_name in contained_object_names
            # A "supported child" inherits z from its support (e.g. bowl on a
            # cookies box) and gets lifted above the support's AABB top in
            # ``_restack_supported_children``. That behaviour is correct for
            # MOVABLE supports (stack relationships) but incorrect for
            # FIXED FIXTURE exterior supports — there, the position planner
            # samples the child's xy in absolute workspace coords and
            # expects the child to settle on the WORKSPACE (table) surface,
            # not be teleported to the fixture's AABB top. We treat the
            # child as "supported" only when its declared support is NOT a
            # fixed fixture (or it is contained inside one).
            raw_support_parent = support_parent_names.get(libero_name, "")
            support_is_fixed_fixture = bool(raw_support_parent) and any(
                getattr(o, "libero_name", "") == raw_support_parent
                and getattr(o, "graspable", True) is False
                for o in self.scene.objects
            )
            is_supported_child = (
                bool(raw_support_parent) and not is_contained and not support_is_fixed_fixture
            )
            # Use LIBERO's default support height only when the generated
            # Scenic object explicitly opts into it AND the object starts near
            # the table surface (or is inside a container at any height).
            # Objects with elevated default_z (e.g. starting on a stove or
            # cabinet shelf that the robot placed them on) should be placed at
            # table-surface z when their XY position is being perturbed to the
            # table area. Contained objects and supported children are the
            # exceptions: both derive their z from an authored support relation.
            if (
                preserve_default_z
                and libero_name in default_pose
                and (
                    default_pose[libero_name][2] <= ELEVATED_Z_THRESHOLD
                    or is_contained
                    or is_supported_child
                )
            ):
                pos[2] = default_pose[libero_name][2]
            else:
                # surface_class lets the variant table resolve a per-(variant,
                # surface) clearance identical to the renderer's emitted spawn z
                # (Fix 3 / Finding A). Empty string → default workspace table.
                _surface_class = getattr(obj, "support_surface_class", "") or None
                # A distractor is a flat clutter object: on an open-frame fixture
                # it settles BELOW the raw collision-edge top_z, so route it
                # through the settle-measured rest-z (mirrors the renderer's
                # distractor branch — lockstep). Task objects keep top_z.
                _is_distractor = libero_name.startswith("distractor_")
                pos[2] = _surface_spawn_z(
                    root_surface_z,
                    getattr(obj, "asset_class", "_default"),
                    _surface_class,
                    distractor=_is_distractor,
                )
                table_spawned_names.add(libero_name)
            self._inject_object_pose(libero_name, pos, obj)
            injected_targets[libero_name] = pos.copy()
            object_dimensions[libero_name] = get_dimensions(getattr(obj, "asset_class", "_default"))
            n_injected += 1

        self._apply_articulation_perturbation()

        # ── apply environment perturbations from Scenic params ──────────
        # The canonical baseline was restored above (before injection) so the
        # additive (`+=`) / multiplicative (`*=`) writes inside the _apply_*
        # helpers still start from canonical cam_pos / light_* / mat_texid.
        # We must NOT restore again here: that would revert the fixture
        # body_pos injection performed in the loop above (g5 STUDY_SCENE RCA).
        self._apply_camera_perturbation()
        self._apply_lighting_perturbation()
        self._apply_texture_perturbation()
        self._apply_background_perturbation()

        if n_injected > 0 or self._has_env_perturbation() or self._has_robot_perturbation():
            import mujoco

            mjmodel = self.libero_env.env.sim.model._model
            mjdata = self.libero_env.env.sim.data._data

            # Zero all velocities so injected objects don't inherit stale momentum.
            mjdata.qvel[:] = 0
            mujoco.mj_forward(mjmodel, mjdata)

            # Run settling steps so objects come to rest on the table surface
            # before the episode begins.  Re-zero velocities afterwards so
            # the policy starts from a quiescent state.
            # Resolve body ids for the injected objects once, so we can measure
            # a CONVERGENCE signal over the last few settle steps below.
            _settle_bids: dict[str, int] = {}
            for _vname in injected_targets:
                for _cand in (_vname, _vname + "_main"):
                    try:
                        _settle_bids[_vname] = mjmodel.body(_cand).id
                        break
                    except Exception:  # noqa: BLE001 — body named either way
                        continue

            # Instantaneous end-of-settle velocity is a POOR "at rest" signal on
            # this path: a resting object in steady contact reads a large, frame-
            # dependent spatial velocity (measured: clean strict-passing cans read
            # ~0.8 m/s while their NET displacement is ~0). A short-window
            # displacement has the same problem — a persistent contact vibration
            # floor (~9 mm over 5 steps for tall/large objects) swamps genuine net
            # motion. So we measure the NET DRIFT of the VIBRATION-AVERAGED
            # position over the settle tail: the mean body position over the first
            # half of the last ``_CONV_WINDOW`` steps vs the mean over the second
            # half. A converged object (a true fixed point) has ~0 net drift even
            # while vibrating; one still mid-settle (a transient that has not
            # reached its rest) drifts. Angular drift uses the orientation at each
            # half's midpoint. Captured WITHIN the existing 50 steps — no extra
            # dynamics — so the settled pose the rest of setup() scores is
            # byte-identical to before. The G4 pose_tolerance alt-rest path reads
            # these via ``get_object_state`` to require an object be genuinely
            # converged before admitting a non-exact-pose settle as a valid
            # alternate rest (see invariants/consistency.py).
            _CONV_WINDOW = 10
            _pos_hist: dict[str, list[np.ndarray]] = {n: [] for n in _settle_bids}
            _quat_hist: dict[str, list[np.ndarray]] = {n: [] for n in _settle_bids}
            for _step in range(50):
                if _step >= 50 - _CONV_WINDOW:
                    for _vname, _bid in _settle_bids.items():
                        _pos_hist[_vname].append(np.array(mjdata.xpos[_bid], dtype=float))
                        _quat_hist[_vname].append(np.array(mjdata.xquat[_bid], dtype=float))
                mujoco.mj_step(mjmodel, mjdata)

            _half = _CONV_WINDOW // 2
            self._settle_convergence: dict[str, tuple[float, float]] = {}
            for _vname in _settle_bids:
                _ph = _pos_hist[_vname]
                _qh = _quat_hist[_vname]
                if len(_ph) < _CONV_WINDOW:
                    continue
                try:
                    _mean1 = np.mean(np.stack(_ph[:_half]), axis=0)
                    _mean2 = np.mean(np.stack(_ph[_half:]), axis=0)
                    _lin = float(np.linalg.norm(_mean2 - _mean1))  # net drift (m)
                    # Angular net drift: orientation at each half's midpoint.
                    _qa = _qh[_half // 2]
                    _qb = _qh[_half + _half // 2]
                    _dot = float(np.clip(abs(np.dot(_qa, _qb)), -1.0, 1.0))
                    _ang = float(2.0 * np.degrees(np.arccos(_dot)))  # net drift (deg)
                    self._settle_convergence[_vname] = (_lin, _ang)
                except Exception:  # noqa: BLE001 — never let telemetry break setup
                    continue

            # Re-zero velocities afterwards so the policy starts from a quiescent
            # state.
            mjdata.qvel[:] = 0
            mujoco.mj_forward(mjmodel, mjdata)
            # Restrict the re-stack lift to genuine stack relationships
            # (movable supports). Fixed-fixture exterior supports such as
            # ``On(bowl, cabinet_top_side)`` are recorded in
            # ``support_parent_names`` so the validator skips the AABB
            # overlap pair-check, but the child is intentionally sampled on
            # the workspace (or fixture exterior), not on the parent's AABB
            # top — lifting it would re-introduce the PR #6 z-height
            # regression.
            #
            # We allow restack ONLY when the parent is a *movable* scene
            # object (graspable=True). Two failure modes occur if we
            # instead filter by the negative set "explicit fixed fixtures
            # in scene.objects":
            #
            #   1. Workspace tables (``living_room_table``,
            #      ``kitchen_table``, …) are NOT enumerated in
            #      ``scene.objects`` at all — they are the implicit
            #      arena. The old filter ``parent not in
            #      fixture_support_names`` was therefore vacuously true
            #      for every workspace-supported child, and the restack
            #      lifted every basket / soup / mug / book to its arena's
            #      AABB top (z ≈ 1.30 m on living_room_table), which is
            #      above the agentview camera (z = 0.96 m) and triggers
            #      Scenic visibility rejection until the 10-retry cap
            #      exhausts.
            #
            #   2. Same hazard for any future fixture not registered as a
            #      Scenic object.
            #
            # Using the positive set "movable scene objects" closes both:
            # restack lifts exist only for ``stacked_on`` (movable→movable)
            # relationships; everything else falls through to physics
            # settling alone.
            movable_scene_names = {
                getattr(o, "libero_name", "")
                for o in self.scene.objects
                if getattr(o, "graspable", True) is True
            }
            stack_support_parent_names = {
                child: parent
                for child, parent in support_parent_names.items()
                if parent in movable_scene_names
            }
            self._restack_supported_children(
                support_parent_names=stack_support_parent_names,
                contained_object_names=contained_object_names,
            )
            mjdata.qvel[:] = 0
            mujoco.mj_forward(mjmodel, mjdata)

            self._validate_settled_positions(
                injected_targets=injected_targets,
                default_pose=default_pose,
                default_rot=default_rot,
                object_dimensions=object_dimensions,
                movable_names=movable_names,
                support_parent_names=support_parent_names,
                table_spawned_names=table_spawned_names,
            )

            # Settling steps can nudge the arm slightly under controller dynamics.
            # Re-apply the sampled reset so the first policy observation matches
            # the Scenic-sampled robot start state exactly.
            if self._has_robot_perturbation():
                self._apply_robot_perturbation()

            # Refresh observables so the first frame reflects the settled state
            # without advancing the episode by an extra control step.
            self.libero_env.check_success()
            self.libero_env._post_process()
            self.libero_env._update_observables(force=True)
            self._last_obs = self.libero_env.env._get_observations(force_update=True)
            self._validate_task_relevant_visibility(object_dimensions=object_dimensions)

        # Cache action dimension for step() — avoids per-step lookups.
        self._nact = self.libero_env.env.action_spec[0].shape[0]
        self._zero_action = np.zeros(self._nact, dtype=float)

        # Cache body_id lookups for getProperties() — avoids per-step try/except.
        self._body_ids: dict[str, int | None] = {}
        sim = self.libero_env.env.sim
        for obj in self.scene.objects:
            libero_name = getattr(obj, "libero_name", None)
            if not libero_name:
                continue
            bid = None
            for candidate in (libero_name, libero_name + "_main"):
                try:
                    bid = sim.model.body_name2id(candidate)
                    break
                except Exception:
                    pass
            self._body_ids[libero_name] = bid

        log.debug("setup complete: injected %d object poses", n_injected)

    # ------------------------------------------------------------------
    # createObjectInSimulator — required abstract method
    # ------------------------------------------------------------------

    def createObjectInSimulator(self, obj):
        """Required by Scenic's ABC. No-op here — objects are loaded via BDDL.

        Position injection happens in setup() instead. This method exists to
        satisfy the abstract method contract.
        """
        pass

    # ------------------------------------------------------------------
    # step — advance simulation by one timestep (Scenic's control loop)
    # ------------------------------------------------------------------

    def step(self):
        """Advance MuJoCo physics by one control timestep.

        Called by Scenic's internal simulation loop (maxSteps times).
        Applies a zero-torque action — the robot holds position.

        For policy-driven evaluation, use step_with_action() instead
        (called directly by eval.py, bypassing Scenic's loop).
        """
        if self.libero_env is None or self._done:
            return

        obs, _reward, done, _info = self.libero_env.step(self._zero_action)
        obs = self._apply_sensor_noise(obs)
        self._last_obs = obs
        self._done = bool(done)

    # ------------------------------------------------------------------
    # getProperties — called by Scenic to read back object state
    # ------------------------------------------------------------------

    def getProperties(self, obj, properties: set[str]) -> dict:
        """Read current simulator state for the given Scenic object.

        Scenic calls this after every step() to track dynamic objects for
        temporal monitors and require-always constraints.

        Supported properties
        ────────────────────
        position    → scenic.core.vectors.Vector(x, y, z)
        orientation → scipy.spatial.transform.Rotation
        velocity    → scenic.core.vectors.Vector(vx, vy, vz)
        speed       → float
        """
        libero_name = getattr(obj, "libero_name", None)
        result: dict[str, Any] = {}

        if not libero_name or self.libero_env is None:
            for prop in properties:
                result[prop] = getattr(obj, prop)
            return result

        sim = self.libero_env.env.sim

        # Use cached body_id (resolved in setup()) to avoid per-step try/except.
        bid = self._body_ids.get(libero_name)

        for prop in properties:
            if prop == "position":
                if bid is not None:
                    result["position"] = Vector(*sim.data.body_xpos[bid].copy())
                else:
                    result["position"] = obj.position

            elif prop == "orientation":
                if bid is not None:
                    try:
                        mat = sim.data.body_xmat[bid].reshape(3, 3)
                        result["orientation"] = _Rotation.from_matrix(mat)
                    except Exception:
                        result["orientation"] = obj.orientation
                else:
                    result["orientation"] = obj.orientation

            elif prop == "velocity":
                if bid is not None:
                    cvel = sim.data.cvel[bid]  # (6,): [angular(3), linear(3)]
                    result["velocity"] = Vector(*cvel[3:])
                else:
                    result["velocity"] = Vector(0, 0, 0)

            elif prop == "speed":
                if bid is not None:
                    cvel = sim.data.cvel[bid]
                    result["speed"] = float(np.linalg.norm(cvel[3:]))
                else:
                    result["speed"] = 0.0

            else:
                result[prop] = getattr(obj, prop, None)

        return result

    # ------------------------------------------------------------------
    # destroy — cleanup
    # ------------------------------------------------------------------

    def destroy(self):
        """Release the LIBERO env and clean up temp files."""
        if self.libero_env is not None:
            try:
                self.libero_env.close()
            except Exception:
                pass
            self.libero_env = None
        if getattr(self, "_distractor_bddl_path", None):
            pathlib.Path(self._distractor_bddl_path).unlink(missing_ok=True)
            self._distractor_bddl_path = None
        super().destroy()

    # ------------------------------------------------------------------
    # Public helpers (used by eval.py — not part of Scenic ABC)
    # ------------------------------------------------------------------

    def step_with_action(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Drive env with a real policy action (used by eval.py).

        This bypasses Scenic's internal loop and gives full control to the
        evaluation harness.

        Returns:
            (obs, reward, done, info) from LIBERO.
        """
        if self.libero_env is None:
            raise RuntimeError("Call setup() before step_with_action()")
        obs, reward, done, info = self.libero_env.step(action)
        obs = self._apply_sensor_noise(obs)
        self._last_obs = obs
        self._done = bool(done)
        return obs, reward, done, info

    def check_success(self) -> bool:
        """Query LIBERO task success predicate."""
        if self.libero_env is None:
            return False
        return bool(self.libero_env.check_success())

    @property
    def last_obs(self) -> dict | None:
        """Most recent environment observation dict."""
        return self._last_obs

    @property
    def mj_handles(self) -> tuple:
        """Raw MuJoCo (model, data) handles for the underlying simulation."""
        return (
            self.libero_env.env.sim.model._model,
            self.libero_env.env.sim.data._data,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_object_pose(
        self,
        libero_name: str,
        pos: np.ndarray,
        scenic_obj,
    ) -> None:
        """Set a movable object's MuJoCo pose to the Scenic-sampled values.

        The caller is responsible for setting the correct z (typically
        preserved from LIBERO's default placement via setup()).

        Uses set_joint_qpos for free-joint objects (all standard graspables).
        Falls back to body_pos/body_quat for fixtures without free joints.
        """
        sim = self.libero_env.env.sim

        asset_class = getattr(scenic_obj, "asset_class", "_default")
        # Use LIBERO canonical orientation + scenic yaw delta (not yaw-only).
        # This prevents objects like bowls/ketchup from toppling during settling.
        # _canonical_rot stores quaternions in scipy xyzw convention (converted
        # from MuJoCo's wxyz in setup()).
        _rot_store = getattr(self, "_canonical_rot", {})
        canonical = _rot_store.get(libero_name)
        if canonical is not None:
            try:
                yaw = float(scenic_obj.orientation.yaw)
            except Exception:
                yaw = 0.0
            if canonical.shape == (3, 3):
                R_can = _Rotation.from_matrix(canonical)
            else:
                R_can = _Rotation.from_quat(canonical)  # xyzw
            R_yaw = _Rotation.from_euler("z", yaw)
            quat_xyzw = (R_can * R_yaw).as_quat()  # scipy xyzw output
            # set_joint_qpos writes directly to MuJoCo qpos which stores wxyz.
            # Convert xyzw → wxyz: [qx,qy,qz,qw] → [qw,qx,qy,qz]
            quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        else:
            try:
                q_xyzw = _scenic_quat(scenic_obj.orientation)
            except Exception:
                q_xyzw = DEFAULT_ORIENTATIONS.get(
                    asset_class,
                    DEFAULT_ORIENTATIONS["_default"],
                ).copy()
            # _scenic_quat() and DEFAULT_ORIENTATIONS both return xyzw
            # (scipy scalar-last).  MuJoCo free-joint qpos expects wxyz
            # (scalar-first), so convert: [qx,qy,qz,qw] → [qw,qx,qy,qz].
            quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

        # Angled-slot CRADLE seating (wine_rack): a flat distractor seated in the
        # slot rests TILTED on the incline. Inject it at the measured cradle tilt
        # (class-independent, MuJoCo wxyz) so it stays put at the slant-bottom
        # rest (fixed-point: injected pose == settled pose) instead of toppling
        # out from an upright start. Only for cradle-seatable distractors; the
        # renderer placed the matching slant-bottom xy/z, so this completes the
        # injected==settled pose. Rotation is not scored for distractors.
        if libero_name.startswith("distractor_"):
            _ssc = getattr(scenic_obj, "support_surface_class", "") or None
            _tilt = _cradle_tilt_quat(_ssc) if _ssc else None
            if _tilt is not None and _is_cradle_seatable(_ssc, asset_class):
                quat = np.array(_tilt, dtype=float)
            else:
                # TABLE distractor (no cradle): an irregular class whose stable
                # flat rest is NOT the generic x90 upright is injected at its
                # MEASURED table rest orientation (e.g. bowl_drainer → identity,
                # base-down) so the 50-step settle leaves it in place instead of
                # tipping onto a rim and sliding (WS follow-up distractor-settle
                # RCA). A class with no measured rest keeps the x90 default.
                _rest = _distractor_table_rest_quat(asset_class)
                if _rest is not None:
                    quat = np.array(_rest, dtype=float)

        pos = pos.copy()

        # 7-vector: [x, y, z, qw, qx, qy, qz]  (MuJoCo free-joint wxyz convention)
        qpos7 = np.concatenate([pos, quat])

        # Try free-joint first (all graspable objects)
        joint_name = f"{libero_name}_joint0"
        try:
            sim.data.set_joint_qpos(joint_name, qpos7)
            log.debug("  set_joint_qpos %s → pos=%s", joint_name, pos)
            return
        except Exception:
            pass

        # Fallback: directly set body position (fixtures without free joints).
        # LIBERO/robosuite names the body "{libero_name}_main"; try both.
        for body_name in (libero_name, libero_name + "_main"):
            try:
                body_id = sim.model.body_name2id(body_name)
                body_quat = quat
                if not bool(getattr(scenic_obj, "graspable", True)):
                    body_quat = sim.model.body_quat[body_id].copy()
                sim.model.body_pos[body_id] = pos
                sim.model.body_quat[body_id] = body_quat
                log.debug("  body_pos fallback %s → pos=%s", body_name, pos)
                return
            except Exception:
                pass

        log.warning("Could not inject pose for %s: not found as joint or body", libero_name)

    def _restack_supported_children(
        self,
        *,
        support_parent_names: dict[str, str],
        contained_object_names: set[str],
    ) -> None:
        """Lift supported children that settled into their parent support.

        A stacked child (movable→movable, e.g. a bowl stacked on a plate) is
        rendered ``at <support> offset by (.., .., 0.0)`` and dropped onto its
        parent, then physics settles it. If it sinks INTO the parent, it is
        lifted so its body origin sits at the MEASURED seating height above the
        parent's top: ``parent_top + spawn_clearance(child_class, parent_class)``.

        ``spawn_clearance`` is the measured body-origin height above a flat
        contact surface (``body_xpos_z − TABLE_SURFACE_Z`` for a table-rester),
        so adding it to the parent's AABB top places the child's base exactly on
        the parent — the SAME measured quantity the renderer/injector use for an
        object's spawn z (``surface_spawn_z``), keeping injected z == settled z.
        This replaces the prior heuristic gap ``max(0.003, min(0.010,
        child_height*0.05))`` (3–10 mm), which ignored the asset's real
        body-origin offset and under-seated tall children by 50–250 % (audit
        WS-1), inducing a renderer/simulator z-frame mismatch. The guard is
        unchanged: a child already resting at/above the measured height is left
        untouched (only sunk children are lifted, never pulled down).
        """
        if self.libero_env is None:
            return

        # Resolve each movable's asset class so the per-(child, parent) measured
        # clearance can be looked up — the SAME class the renderer emits.
        class_by_name: dict[str, str] = {}
        for o in getattr(self.scene, "objects", []) or []:
            nm = getattr(o, "libero_name", "")
            if nm:
                class_by_name[nm] = getattr(o, "asset_class", "") or "_default"

        sim = self.libero_env.env.sim
        for child_name, parent_name in support_parent_names.items():
            if not parent_name or child_name in contained_object_names:
                continue

            child_aabb = _body_world_aabb(sim, child_name)
            parent_aabb = _body_world_aabb(sim, parent_name)
            if child_aabb is None or parent_aabb is None:
                continue

            child_min, _child_max = child_aabb
            _parent_min, parent_max = parent_aabb
            if not np.all(np.isfinite(child_min)) or not np.all(np.isfinite(parent_max)):
                continue

            child_origin_z = _body_origin_z(sim, child_name)
            if child_origin_z is None or not np.isfinite(child_origin_z):
                continue

            # Measured body-origin seating height above the parent's top face.
            clearance = _spawn_clearance(
                class_by_name.get(child_name, "_default"),
                class_by_name.get(parent_name) or None,
            )
            min_child_origin_z = float(parent_max[2]) + clearance
            if child_origin_z >= min_child_origin_z:
                continue

            dz = min_child_origin_z - child_origin_z
            joint_name = f"{child_name}_joint0"
            try:
                qpos = sim.data.get_joint_qpos(joint_name).copy()
                qpos[2] = float(qpos[2]) + dz
                sim.data.set_joint_qpos(joint_name, qpos)
                log.debug(
                    "Lifted supported child %s by %.4f m above %s",
                    child_name,
                    dz,
                    parent_name,
                )
                continue
            except Exception:
                pass

            for body_name in (child_name, child_name + "_main"):
                try:
                    body_id = sim.model.body_name2id(body_name)
                    sim.model.body_pos[body_id][2] = float(sim.model.body_pos[body_id][2]) + dz
                    log.debug(
                        "Lifted supported child body %s by %.4f m above %s",
                        body_name,
                        dz,
                        parent_name,
                    )
                    break
                except Exception:
                    continue

    def _validate_settled_positions(
        self,
        *,
        injected_targets: dict[str, np.ndarray],
        default_pose: dict[str, np.ndarray],
        default_rot: dict[str, np.ndarray],
        object_dimensions: dict[str, tuple[float, float, float]],
        movable_names: set[str],
        support_parent_names: dict[str, str],
        table_spawned_names: set[str],
    ) -> None:
        """Fail fast when settling reveals a sample with unstable placement.

        The absolute table/floor geometry varies across LIBERO suites, so the
        validator only checks failures which are robust across scenes:
        large xy drift from the Scenic sample, non-finite settled poses,
        excessive rotation from the default pose, or post-settle overlaps.
        """
        if not injected_targets:
            return

        sim = self.libero_env.env.sim
        failures: list[str] = []
        for libero_name, target in injected_targets.items():
            if libero_name.startswith("distractor_"):
                continue
            body_id = None
            for candidate in (libero_name, libero_name + "_main"):
                try:
                    body_id = sim.model.body_name2id(candidate)
                    break
                except Exception:
                    continue
            if body_id is None:
                continue

            final_pos = np.array(sim.data.body_xpos[body_id][:3], dtype=float)
            if not np.all(np.isfinite(final_pos)):
                failures.append(f"{libero_name} settled to a non-finite pose")
                continue
            xy_drift = float(np.linalg.norm(final_pos[:2] - target[:2]))
            if xy_drift > MAX_SETTLE_XY_DRIFT:
                failures.append(
                    f"{libero_name} drifted {xy_drift:.3f} m from its sampled xy target"
                )

            ref_rot = default_rot.get(libero_name)
            if ref_rot is not None:
                final_rot = np.array(sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
                # default_rot is xyzw (converted from MuJoCo wxyz in setup()).
                # Body fallback stores a 3×3 matrix directly.
                if ref_rot.shape == (4,):
                    ref_rot_mat = _Rotation.from_quat(ref_rot).as_matrix()
                else:
                    ref_rot_mat = ref_rot.reshape(3, 3)
                rel_rot = ref_rot_mat.T @ final_rot
                rot_drift = float(np.arccos(np.clip((np.trace(rel_rot) - 1.0) * 0.5, -1.0, 1.0)))
                if rot_drift > MAX_SETTLE_ROT_DRIFT:
                    failures.append(f"{libero_name}: rot drift {np.rad2deg(rot_drift):.1f} deg")

        settled_positions: dict[str, np.ndarray] = {}
        for libero_name in injected_targets:
            if libero_name.startswith("distractor_"):
                continue
            body_id = None
            for candidate in (libero_name, libero_name + "_main"):
                try:
                    body_id = sim.model.body_name2id(candidate)
                    break
                except Exception:
                    continue
            if body_id is None:
                continue
            settled_positions[libero_name] = np.array(
                sim.data.body_xpos[body_id][:3],
                dtype=float,
            )

        names = sorted(settled_positions)
        for i, name_a in enumerate(names):
            dims_a = object_dimensions.get(name_a)
            if dims_a is None:
                continue
            for name_b in names[i + 1 :]:
                # Only police overlap for pairs influenced by table-surface
                # injection. Canonical BDDL-authored pairs can be close by
                # design, and this validator is meant to catch Scenic-induced
                # bad samples rather than task-author layout choices.
                if name_a not in table_spawned_names and name_b not in table_spawned_names:
                    continue
                if (
                    support_parent_names.get(name_a) == name_b
                    or support_parent_names.get(name_b) == name_a
                ):
                    continue
                dims_b = object_dimensions.get(name_b)
                if dims_b is None:
                    continue
                if _axis_overlap_xy(
                    settled_positions[name_a],
                    dims_a,
                    settled_positions[name_b],
                    dims_b,
                    margin=-0.03,  # allow 3 cm of AABB slack: registry dims are
                    # conservative bounding boxes; actual meshes
                    # are smaller, so minor AABB overlaps after
                    # settling are normal physics artefacts.
                ):
                    failures.append(
                        f"{name_a} overlaps {name_b} after settling "
                        "(axis-aligned footprints intersect)"
                    )

        for i in range(int(sim.data.ncon)):
            contact = sim.data.contact[i]
            geom_a = int(contact.geom1)
            geom_b = int(contact.geom2)
            body_a = sim.model.body_id2name(sim.model.geom_bodyid[geom_a]) or ""
            body_b = sim.model.body_id2name(sim.model.geom_bodyid[geom_b]) or ""

            owner_a = next((name for name in table_spawned_names if body_a.startswith(name)), None)
            owner_b = next((name for name in table_spawned_names if body_b.startswith(name)), None)
            if owner_a is None and owner_b is None:
                continue

            other_body = body_b if owner_a is not None else body_a
            if _is_workspace_surface_body(other_body):
                continue
            if any(other_body.startswith(prefix) for prefix in movable_names):
                continue
            # Skip contacts between a contained object and its support-parent
            # fixture (e.g. a bowl inside a cabinet drawer contacts the drawer
            # walls/bottom — this is expected and must not be flagged).
            contact_owner = owner_a if owner_a is not None else owner_b
            if contact_owner is not None:
                parent_name = support_parent_names.get(contact_owner, "")
                if parent_name and other_body.startswith(parent_name):
                    continue

            failures.append(
                f"{owner_a or owner_b} remains in contact with {other_body} after settling"
            )

        if failures:
            raise CollisionError(
                "Invalid Scenic sample after MuJoCo settling: " + "; ".join(failures),
                object_names=failures,
            )

    def _has_env_perturbation(self) -> bool:
        """True if any environment perturbation params are set in the scene."""
        if self.scene is None:
            return False
        params = getattr(self.scene, "params", {})
        return any(
            params.get(k) is not None
            for k in (
                "cam_azimuth",
                "cam_elevation",
                "cam_distance",
                "camera_x_offset",
                "camera_y_offset",
                "camera_z_offset",
                "camera_tilt",
                "light_intensity",
                "light_x_offset",
                "light_y_offset",
                "light_z_offset",
                "ambient_level",
                "table_texture",
            )
        )

    # ------------------------------------------------------------------
    # Model state snapshot / restore (Bug 1 fix)
    # ------------------------------------------------------------------

    def _capture_model_baseline(self) -> None:
        """Snapshot the MuJoCo model arrays mutated by ``_apply_*_perturbation``.

        Called once per env, immediately after ``OffScreenRenderEnv`` is
        constructed. Stores deep copies of the canonical (XML-loaded) values so
        ``_restore_model_baseline`` can return the model to that baseline before
        each apply pass. This makes the perturbation pipeline idempotent under
        repeated application — without it, additive writes to cam_pos /
        light_pos and multiplicative writes to light_diffuse / light_specular
        accumulate every time the apply functions run, silently mutating the
        realised perturbation envelope.
        """
        if self.libero_env is None:
            return
        sim = self.libero_env.env.sim
        m = sim.model
        try:
            self._model_baseline = {
                "cam_pos": np.array(m.cam_pos, dtype=float, copy=True),
                "cam_quat": np.array(m.cam_quat, dtype=float, copy=True),
                "light_pos": np.array(m.light_pos, dtype=float, copy=True),
                "light_diffuse": np.array(m.light_diffuse, dtype=float, copy=True),
                "light_specular": np.array(m.light_specular, dtype=float, copy=True),
                "light_ambient": np.array(m.light_ambient, dtype=float, copy=True),
                "mat_texid": np.array(m.mat_texid, dtype=int, copy=True),
                "headlight_ambient": np.array(m.vis.headlight.ambient, dtype=float, copy=True),
                # body_pos / body_quat must be snapshotted too: the
                # ``_inject_object_pose`` and ``_restack_supported_children``
                # paths write directly to ``sim.model.body_pos[id]`` (and
                # occasionally body_quat) as a fallback when a body has no
                # free joint. Without a baseline restore, those writes
                # persist into subsequent resets and compound across
                # episodes (RCA: ~/.omar/ea/4/pr6_zheight_rca.md §3).
                "body_pos": np.array(m.body_pos, dtype=float, copy=True),
                "body_quat": np.array(m.body_quat, dtype=float, copy=True),
            }
        except Exception as exc:
            log.debug("Model baseline capture failed: %s", exc)
            self._model_baseline = None

    def _restore_model_baseline(self) -> None:
        """Restore the snapshotted MuJoCo model arrays.

        Called before each apply pass so that perturbations always start from
        the canonical XML-loaded state, never from a previously-perturbed
        state. No-op if no snapshot has been captured (e.g. baseline capture
        failed).
        """
        if self.libero_env is None or self._model_baseline is None:
            return
        sim = self.libero_env.env.sim
        m = sim.model
        baseline = self._model_baseline
        try:
            m.cam_pos[:] = baseline["cam_pos"]
            m.cam_quat[:] = baseline["cam_quat"]
            m.light_pos[:] = baseline["light_pos"]
            m.light_diffuse[:] = baseline["light_diffuse"]
            m.light_specular[:] = baseline["light_specular"]
            m.light_ambient[:] = baseline["light_ambient"]
            m.mat_texid[:] = baseline["mat_texid"]
            m.vis.headlight.ambient[:] = baseline["headlight_ambient"]
            if "body_pos" in baseline:
                m.body_pos[:] = baseline["body_pos"]
            if "body_quat" in baseline:
                m.body_quat[:] = baseline["body_quat"]
        except Exception as exc:
            log.debug("Model baseline restore failed: %s", exc)

    def _has_robot_perturbation(self) -> bool:
        """True if Scenic sampled a robot init-qpos vector for this scene."""
        if self.scene is None:
            return False
        params = getattr(self.scene, "params", {})
        robot_qpos = params.get("robot_init_qpos")
        if isinstance(robot_qpos, (list, tuple)):
            return len(robot_qpos) > 0
        return any(
            params.get(f"robot_init_qpos_{idx}") is not None
            for idx in range(EXPECTED_PANDA_ARM_DOF)
        )

    def _apply_robot_perturbation(self) -> None:
        """Apply a Scenic-sampled Panda init_qpos perturbation to the arm joints."""
        if self.scene is None or self.libero_env is None:
            return
        params = getattr(self.scene, "params", {})
        robot_qpos = params.get("robot_init_qpos")
        if robot_qpos is None:
            per_joint = [
                params.get(f"robot_init_qpos_{idx}") for idx in range(EXPECTED_PANDA_ARM_DOF)
            ]
            if not any(value is not None for value in per_joint):
                return
            robot_qpos = per_joint
        if not isinstance(robot_qpos, (list, tuple)):
            return

        qpos = np.asarray(robot_qpos, dtype=float)
        if qpos.shape != (EXPECTED_PANDA_ARM_DOF,):
            raise ScenarioValidationError(
                f"robot_init_qpos must be length {EXPECTED_PANDA_ARM_DOF}, got shape {qpos.shape}"
            )
        if not np.all(np.isfinite(qpos)):
            raise ScenarioValidationError("robot_init_qpos contains non-finite values")

        env = self.libero_env.env
        robot = env.robots[0]
        sim = env.sim
        joint_indexes = np.asarray(getattr(robot, "_ref_joint_pos_indexes", ()), dtype=int)
        joint_names = tuple(getattr(robot, "robot_joints", ()))
        if (
            joint_indexes.shape != (EXPECTED_PANDA_ARM_DOF,)
            or len(joint_names) != EXPECTED_PANDA_ARM_DOF
        ):
            raise ScenarioValidationError(
                "Unexpected Panda joint layout while applying robot_init_qpos perturbation"
            )

        lower = []
        upper = []
        for joint_name in joint_names:
            joint_id = int(sim.model.joint_name2id(joint_name))
            lo, hi = sim.model.jnt_range[joint_id]
            lower.append(float(lo))
            upper.append(float(hi))
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)
        clipped = np.clip(qpos, lower_arr, upper_arr)
        if not np.allclose(qpos, clipped, atol=1e-8):
            log.debug("  robot init qpos clipped to joint limits")
            qpos = clipped

        sim.data.qpos[joint_indexes] = qpos
        sim.data.qvel[joint_indexes] = 0.0
        try:
            sim.forward()
        except Exception:
            pass
        if hasattr(robot, "init_qpos"):
            robot.init_qpos = qpos.copy()
        if hasattr(robot, "recent_qpos") and hasattr(robot.recent_qpos, "push"):
            robot.recent_qpos.push(qpos.copy())

        controller = getattr(robot, "controller", None)
        if controller is not None:
            controller.update()
            controller.joint_pos = qpos.copy()
            if hasattr(controller, "update_initial_joints"):
                controller.update_initial_joints(qpos.copy())

        self._applied_robot_init_qpos = qpos.copy()
        log.debug("  robot init qpos: %s", np.array2string(qpos, precision=4))

    def _apply_articulation_perturbation(self) -> None:
        """Apply sampled articulation qpos values from Scenic params."""
        if self.scene is None or self.libero_env is None:
            return
        params = getattr(self.scene, "params", {})
        if not params:
            return

        object_states = getattr(self.libero_env.env, "object_states_dict", {})
        if not object_states:
            return

        for key, value in params.items():
            if (
                not key.startswith("articulation_")
                or key.startswith("articulation_state_")
                or key.startswith("articulation_control_")
                or key.endswith("_state")
            ):
                continue
            fixture_name = key.removeprefix("articulation_")
            control_target = params.get(f"articulation_control_{fixture_name}", fixture_name)
            state = object_states.get(control_target)
            if state is None:
                state = object_states.get(fixture_name)
            if state is None:
                log.debug("No articulation state handle found for %s", control_target)
                continue
            try:
                state.set_joint(float(value))
            except Exception:
                log.debug("Failed to set articulation for %s", control_target, exc_info=True)

    def _validate_task_relevant_visibility(
        self,
        *,
        object_dimensions: dict[str, tuple[float, float, float]],
    ) -> None:
        """Reject settled samples where key task objects are out of frame or occluded."""
        if self.scene is None or self.libero_env is None or self._last_obs is None:
            return
        params = getattr(self.scene, "params", {})
        target_names = list(params.get("visibility_targets", []))
        if not target_names:
            return
        depth = self._last_obs.get("agentview_depth")
        if depth is None:
            return

        sim = self.libero_env.env.sim
        height = int(depth.shape[0])
        width = int(depth.shape[1])
        world_to_pixel, world_to_camera = _camera_transforms(
            sim=sim,
            camera_name="agentview",
            camera_height=height,
            camera_width=width,
        )
        depth_map = _real_depth_map(sim, depth[..., 0])

        failures: list[str] = []
        for target_name in target_names:
            body_id = None
            if hasattr(self, "_body_ids"):
                body_id = self._body_ids.get(target_name)
            if body_id is None:
                for candidate in (target_name, target_name + "_main"):
                    try:
                        body_id = sim.model.body_name2id(candidate)
                        break
                    except Exception:
                        continue
            if body_id is None:
                continue
            center = np.array(sim.data.body_xpos[body_id][:3], dtype=float)
            dims = object_dimensions.get(target_name, (0.06, 0.06, 0.06))
            anchor_points = _visibility_anchor_points_for_body(
                sim=sim,
                object_name=target_name,
                center=center,
                dims=dims,
            )
            depth_tolerance = _visibility_depth_tolerance(
                sim=sim,
                object_name=target_name,
            )
            visible = 0
            anchors = 0
            for point in anchor_points:
                anchors += 1
                visible += int(
                    _anchor_visible(
                        point=point,
                        world_to_pixel=world_to_pixel,
                        world_to_camera=world_to_camera,
                        depth_map=depth_map,
                        image_height=height,
                        image_width=width,
                        depth_tolerance=depth_tolerance,
                    )
                )
            if visible == 0:
                failures.append(f"{target_name} is out of frame or fully occluded")
            elif visible < max(1, min(3, anchors // 4)):
                failures.append(f"{target_name} is only weakly visible in agentview")

        if failures:
            raise VisibilityError(
                "Invalid Scenic sample after visibility check: " + "; ".join(failures),
                invisible_names=failures,
            )

    def _apply_camera_perturbation(self) -> None:
        """Perturb agentview camera pose from Scenic params.

        Scenic params read from scene.params:
          cam_azimuth    — orbit angle around the table target, in degrees
          cam_elevation  — elevation delta around the table target, in degrees
          cam_distance   — multiplicative distance scale from the table target
          camera_x_offset  — additive x offset (metres)
          camera_y_offset  — additive y offset (metres)
          camera_z_offset  — additive z offset (metres)
          camera_tilt      — tilt angle in degrees (added to elevation)
        """
        if self.scene is None or self.libero_env is None:
            return
        params = getattr(self.scene, "params", {})

        azimuth = params.get("cam_azimuth")
        elevation = params.get("cam_elevation")
        distance = params.get("cam_distance")
        dx = params.get("camera_x_offset", 0.0)
        dy = params.get("camera_y_offset", 0.0)
        dz = params.get("camera_z_offset", 0.0)
        tilt = params.get("camera_tilt", 0.0)

        has_orbit = azimuth is not None or elevation is not None or distance is not None
        if not has_orbit and dx == 0 and dy == 0 and dz == 0 and tilt == 0:
            return

        sim = self.libero_env.env.sim

        # Find agentview camera
        try:
            cam_id = sim.model.camera_name2id("agentview")
        except Exception:
            log.warning("agentview camera not found; skipping camera perturbation")
            return

        if has_orbit:
            base_pos = np.array(sim.model.cam_pos[cam_id], dtype=float)
            target = np.array(
                [
                    float(params.get("cam_target_x", 0.0)),
                    float(params.get("cam_target_y", 0.0)),
                    float(params.get("cam_target_z", TABLE_Z)),
                ],
                dtype=float,
            )
            vec = base_pos - target
            radius = float(np.linalg.norm(vec))
            if radius > 1e-9:
                az0 = float(np.arctan2(vec[1], vec[0]))
                el0 = float(np.arcsin(np.clip(vec[2] / radius, -1.0, 1.0)))
                az = az0 + np.deg2rad(float(azimuth or 0.0))
                el = np.clip(
                    el0 + np.deg2rad(float(elevation or 0.0)),
                    np.deg2rad(-85.0),
                    np.deg2rad(85.0),
                )
                radius *= float(distance if distance is not None else 1.0)
                radius = max(radius, 1e-6)

                cos_el = float(np.cos(el))
                new_pos = target + np.array(
                    [
                        radius * cos_el * np.cos(az),
                        radius * cos_el * np.sin(az),
                        radius * np.sin(el),
                    ],
                    dtype=float,
                )
                sim.model.cam_pos[cam_id] = new_pos
                self._set_camera_lookat(cam_id, new_pos, target)
                log.debug(
                    "  camera orbit: azimuth=%.3f elevation=%.3f distance=%.3f",
                    float(azimuth or 0.0),
                    float(elevation or 0.0),
                    float(distance if distance is not None else 1.0),
                )

        if dx != 0 or dy != 0 or dz != 0:
            sim.model.cam_pos[cam_id][0] += float(dx)
            sim.model.cam_pos[cam_id][1] += float(dy)
            sim.model.cam_pos[cam_id][2] += float(dz)
            log.debug("  camera offset: dx=%.3f dy=%.3f dz=%.3f", dx, dy, dz)

        if tilt != 0:
            # Tilt is applied as a rotation around the camera's local x-axis
            # by modifying the camera quaternion.
            current_quat = sim.model.cam_quat[cam_id].copy()
            # MuJoCo uses (w,x,y,z) quaternion convention
            r_current = _Rotation.from_quat(
                [
                    current_quat[1],
                    current_quat[2],
                    current_quat[3],
                    current_quat[0],
                ]
            )
            r_tilt = _Rotation.from_euler("x", float(tilt), degrees=True)
            r_new = r_current * r_tilt
            q = r_new.as_quat()  # (x,y,z,w)
            sim.model.cam_quat[cam_id] = [q[3], q[0], q[1], q[2]]
            log.debug("  camera tilt: %.1f degrees", tilt)

        try:
            sim.forward()
        except Exception:
            pass

    def _set_camera_lookat(self, cam_id: int, camera_pos: np.ndarray, target: np.ndarray) -> None:
        """Orient a MuJoCo fixed camera so its local -z axis points at target."""
        forward = target - camera_pos
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-9:
            return
        forward = forward / norm

        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm <= 1e-9:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            right = right / right_norm
        up = np.cross(right, forward)
        rot = np.column_stack([right, up, -forward])
        quat_xyzw = _Rotation.from_matrix(rot).as_quat()
        self.libero_env.env.sim.model.cam_quat[cam_id] = [
            quat_xyzw[3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2],
        ]

    def _apply_lighting_perturbation(self) -> None:
        """Perturb scene lighting from Scenic params.

        Scenic params read from scene.params:
          light_intensity    — multiplier for diffuse/specular light (default 1.0)
          light_x_offset     — additive x offset for light position
          light_y_offset     — additive y offset for light position
          light_z_offset     — additive z offset for light position
          ambient_level      — override ambient light level (0.0-1.0)
        """
        if self.scene is None or self.libero_env is None:
            return
        params = getattr(self.scene, "params", {})

        intensity = params.get("light_intensity")
        ldx = params.get("light_x_offset", 0.0)
        ldy = params.get("light_y_offset", 0.0)
        ldz = params.get("light_z_offset", 0.0)
        ambient = params.get("ambient_level")

        has_change = (
            (intensity is not None and intensity != 1.0)
            or ldx != 0
            or ldy != 0
            or ldz != 0
            or ambient is not None
        )
        if not has_change:
            return

        sim = self.libero_env.env.sim

        # Perturb all lights
        n_lights = sim.model.nlight
        for i in range(n_lights):
            if ldx != 0 or ldy != 0 or ldz != 0:
                sim.model.light_pos[i][0] += float(ldx)
                sim.model.light_pos[i][1] += float(ldy)
                sim.model.light_pos[i][2] += float(ldz)

            if intensity is not None and intensity != 1.0:
                sim.model.light_diffuse[i] *= float(intensity)
                sim.model.light_specular[i] *= float(intensity)

            # Apply ambient to each declared light, not just the headlight.
            # MuJoCo's headlight is only active when the model declares no
            # other lights — most LIBERO scenes do declare lights, so writing
            # only to vis.headlight.ambient was a no-op for the rendered
            # frame. Per-light ``light_ambient[i]`` is the channel that
            # actually contributes to scene shading.
            if ambient is not None:
                sim.model.light_ambient[i][:] = float(ambient)

        if ambient is not None:
            # Also set the global headlight ambient as a belt-and-braces
            # fallback for any scenes that *don't* declare lights.
            sim.model.vis.headlight.ambient[:] = float(ambient)
            log.debug("  ambient level: %.2f", ambient)

        log.debug(
            "  lighting: intensity=%.2f offset=(%.2f,%.2f,%.2f)",
            intensity or 1.0,
            ldx,
            ldy,
            ldz,
        )

    def _curated_loaded_tex_ids(self, sim) -> list[int]:
        """Return texture IDs whose names appear in the curated background pool.

        This is the intersection of ``LIBERO_BACKGROUND_TEXTURES`` (the
        curated wall/floor/table-looking texture names the planner draws over)
        with the textures actually loaded into the current MuJoCo model. It is
        used by the ``"random"`` resolution path so that a "random table /
        background texture" cannot pick a robot, gripper, or character mesh
        texture that just happens to be loaded.
        """
        n_tex = int(getattr(sim.model, "ntex", 0))
        if n_tex <= 0:
            return []
        curated: list[int] = []
        for name in LIBERO_BACKGROUND_TEXTURES:
            try:
                curated.append(int(sim.model.texture_name2id(name)))
            except Exception:
                continue
        return curated

    def _apply_texture_perturbation(self) -> None:
        """Perturb table texture from Scenic params.

        Scenic params read from scene.params:
          table_texture  — texture name to apply to table surface,
                          or "random" to pick from available textures
        """
        if self.scene is None or self.libero_env is None:
            return
        params = getattr(self.scene, "params", {})

        texture_name = params.get("table_texture")
        if not texture_name:
            return

        sim = self.libero_env.env.sim

        # Find the table body and its geom
        table_body_id = None
        for name_candidate in ("main_table", "table_main"):
            try:
                table_body_id = sim.model.body_name2id(name_candidate)
                break
            except Exception:
                pass

        if table_body_id is None:
            log.warning("Table body not found; skipping texture perturbation")
            return

        if texture_name == "random":
            # Pick a random texture from the *curated* loaded subset so the
            # table cannot accidentally adopt a robot/gripper/character mesh
            # texture that happens to be loaded in the model.
            curated = self._curated_loaded_tex_ids(sim)
            if curated:
                tex_id = int(curated[np.random.randint(0, len(curated))])
            else:
                # Fall back to any loaded texture only if no curated entries
                # are present in the model (e.g. minimal arenas).
                n_tex = sim.model.ntex
                if n_tex <= 0:
                    return
                tex_id = int(np.random.randint(0, n_tex))
                log.debug(
                    "  texture: no curated textures loaded; falling back to any-texture random"
                )
        else:
            # Look up by name
            try:
                tex_id = sim.model.texture_name2id(texture_name)
            except Exception:
                log.warning("Texture '%s' not found; skipping", texture_name)
                return

        # Find material(s) used by geoms of the table body
        for geom_id in range(sim.model.ngeom):
            if sim.model.geom_bodyid[geom_id] == table_body_id:
                mat_id = sim.model.geom_matid[geom_id]
                if mat_id >= 0:
                    sim.model.mat_texid[mat_id] = tex_id
                    log.debug("  table texture: geom %d → tex %d", geom_id, tex_id)

    def _apply_sensor_noise(self, obs: dict | None) -> dict | None:
        """Post-process visual observations with the sampled corruption.

        Reads ``sensor_noise_kind`` and ``sensor_noise_severity`` from
        ``scene.params`` and applies the corresponding image transform to
        every key in ``obs`` whose name ends in ``"_image"`` (RGB only;
        depth and segmentation channels are left untouched). The
        transform is a deterministic function of (kind, severity) so the
        same scene + step yields the same corrupted image.

        Severity follows the C-level convention from the *Common Image
        Corruptions* benchmark — ``1`` = barely visible, ``5`` = severe.
        """
        if not isinstance(obs, dict) or self.scene is None:
            return obs
        params = getattr(self.scene, "params", {}) or {}
        kind = params.get("sensor_noise_kind")
        if not kind or kind == "none":
            return obs
        severity = int(params.get("sensor_noise_severity") or 1)
        severity = max(1, min(5, severity))

        # Per-scene seed: combine any explicit ``scenic_seed`` (set by the
        # planner) with the step counter so identical (kind, severity) pairs
        # still produce *different* noise patterns across scenes (E5 fix).
        seed_source = params.get("scenic_seed")
        if seed_source is None:
            seed_source = params.get("seed")
        if seed_source is None:
            # Fall back to id(self.scene) for at-least-per-episode variation.
            seed_source = id(self.scene)
        try:
            scene_seed = int(seed_source) & 0x7FFFFFFF
        except (TypeError, ValueError):
            scene_seed = abs(hash(seed_source)) & 0x7FFFFFFF

        out = dict(obs)
        for key, value in obs.items():
            if not (isinstance(key, str) and key.endswith("_image")):
                continue
            if not isinstance(value, np.ndarray):
                continue
            try:
                out[key] = _apply_image_corruption(value, kind, severity, seed=scene_seed)
            except Exception as exc:
                log.debug("Sensor-noise transform '%s' failed: %s", kind, exc)
        return out

    def _apply_background_perturbation(self) -> None:
        """Perturb wall and floor textures from Scenic params.

        Scenic params read from scene.params:
          wall_texture   — texture name for wall material (``walls_mat``),
                           or ``"random"`` to pick any loaded texture.
          floor_texture  — texture name for floor material (``floorplane``),
                           or ``"random"`` to pick any loaded texture.

        Material names (``walls_mat`` and ``floorplane``) are the names used
        across all LIBERO scene XMLs — confirmed by inspecting the installed
        LIBERO scene assets. Missing material or
        texture names are handled gracefully so that scenes without these
        materials (e.g. custom arenas) are unaffected.
        """
        if self.scene is None or self.libero_env is None:
            return
        params = getattr(self.scene, "params", {})

        wall_texture = params.get("wall_texture")
        floor_texture = params.get("floor_texture")

        if not wall_texture and not floor_texture:
            return

        sim = self.libero_env.env.sim

        def _resolve_tex_id(texture_name: str) -> int | None:
            """Resolve a texture name to a loaded MuJoCo texture ID.

            "random" and named-but-not-loaded fallbacks both go through the
            curated wall/floor texture subset rather than ``randint(0, ntex)``,
            which would otherwise pick robot/gripper/object mesh textures.

            Returns:
                Integer texture ID, or None if the model has no textures.
            """
            n_tex = sim.model.ntex
            if n_tex <= 0:
                return None
            curated = self._curated_loaded_tex_ids(sim)
            if texture_name == "random":
                if curated:
                    return int(curated[np.random.randint(0, len(curated))])
                log.debug(
                    "  background: no curated textures loaded; falling back to any-texture random"
                )
                return int(np.random.randint(0, n_tex))
            try:
                return int(sim.model.texture_name2id(texture_name))
            except Exception:
                # Named texture not loaded in this model — fall back to a
                # random *curated* texture so the realised distribution stays
                # close to the marketed pool.
                log.debug(
                    "  background: texture '%s' not in model; using curated random",
                    texture_name,
                )
                if curated:
                    return int(curated[np.random.randint(0, len(curated))])
                return int(np.random.randint(0, n_tex))

        def _apply_mat_texture(mat_name: str, texture_name: str) -> None:
            """Swap the texture referenced by material mat_name."""
            try:
                mat_id = sim.model.material_name2id(mat_name)
            except Exception:
                log.debug("  background: material '%s' not found in model; skipping", mat_name)
                return
            tex_id = _resolve_tex_id(texture_name)
            if tex_id is None:
                return
            sim.model.mat_texid[mat_id] = tex_id
            log.debug("  background: %s → tex_id=%d", mat_name, tex_id)

        if wall_texture:
            _apply_mat_texture("walls_mat", str(wall_texture))
        if floor_texture:
            _apply_mat_texture("floorplane", str(floor_texture))


# ---------------------------------------------------------------------------
# Validation feedback loop (Stage 5 of compiler pipeline)
# ---------------------------------------------------------------------------


def run_with_validation_loop(
    scenario,
    simulator: "LIBEROSimulator",
    *,
    max_visibility_retries: int = MAX_VISIBILITY_RETRIES,
    max_steps: int = 500,
) -> "LIBEROSimulation":
    """Run a Scenic scenario with typed error recovery for VisibilityError only.

    Implements P7 (Bounded Termination): terminates in at most
    ``max_visibility_retries + 1`` simulation attempts, then raises
    ``InfeasibleScenarioError``.

    Recovery strategy mapping:
    - CollisionError  → propagates immediately as InfeasibleScenarioError
                        (renderer emits per-pair clearance; collision = bug)
    - VisibilityError → re-sample Scenic scenario, up to max_visibility_retries
                        (three sub-cases: camera frustum, distractor occlusion,
                         articulation occlusion — all resolved by re-sampling)

    Parameters
    ----------
    scenario:
        A compiled Scenic scenario (output of scenic.scenarioFromFile or equivalent).
    simulator:
        A LIBEROSimulator instance attached to the task BDDL.
    max_visibility_retries:
        Maximum re-sample attempts for VisibilityError. Default MAX_VISIBILITY_RETRIES.
    max_steps:
        Episode horizon for each simulation attempt.

    Returns
    -------
    LIBEROSimulation
        A successfully validated simulation.

    Raises
    ------
    InfeasibleScenarioError
        When all retry budgets are exhausted without a valid scenario.
    CollisionError
        Propagated immediately — indicates a renderer bug, not a transient failure.
    """
    n_visibility = 0

    while True:
        try:
            # Generate a scene (Scenic handles its own rejection sampling internally)
            scene, _ = scenario.generate(maxIterations=max_visibility_retries - n_visibility + 1)
            sim = simulator.simulate(scene, maxSteps=max_steps)
            return sim  # success

        except CollisionError as exc:
            # CollisionError = renderer bug (per-pair clearance should have prevented this).
            # Do NOT retry — propagate immediately as a hard failure.
            raise InfeasibleScenarioError(
                f"CollisionError (renderer bug — should not occur with per-pair clearance): {exc}",
                n_resample=0,
                n_replan=0,
            ) from exc

        except VisibilityError as exc:
            n_visibility += 1
            log.debug(
                "VisibilityError (retry %d/%d): %s",
                n_visibility,
                max_visibility_retries,
                exc,
            )
            if n_visibility >= max_visibility_retries:
                raise InfeasibleScenarioError(
                    f"Exhausted {max_visibility_retries} retries after VisibilityError",
                    n_resample=n_visibility,
                    n_replan=0,
                ) from exc
