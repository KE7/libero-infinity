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
from typing import Any

import numpy as np
from scenic.core.simulators import Simulation, Simulator
from scenic.core.vectors import Vector
from scipy.spatial.transform import Rotation as _Rotation

from libero_infinity.asset_registry import get_dimensions
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


def _surface_spawn_z(surface_z: float, asset_class: str) -> float:
    """Approximate object-centre z for spawning directly on a root surface."""
    _w, _l, h = get_dimensions(asset_class)
    return surface_z + max(float(h) / 2.0, 0.01) + 1e-3


def _infer_root_surface_z(scene_objects, default_pose: dict[str, np.ndarray]) -> float:
    """Infer the canonical root support height from default LIBERO placements."""
    surface_candidates: list[float] = []
    for obj in scene_objects:
        libero_name = getattr(obj, "libero_name", None)
        if not libero_name or libero_name not in default_pose:
            continue
        if getattr(obj, "support_parent_name", ""):
            continue
        _w, _l, h = get_dimensions(getattr(obj, "asset_class", "_default"))
        surface_candidates.append(float(default_pose[libero_name][2]) - max(float(h) / 2.0, 0.01))
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


def _real_depth_map(sim, depth_map: np.ndarray) -> np.ndarray:
    """Convert MuJoCo's normalized depth image to metric depth."""
    assert np.all(depth_map >= 0.0) and np.all(depth_map <= 1.0)
    extent = float(sim.model.stat.extent)
    far = float(sim.model.vis.map.zfar) * extent
    near = float(sim.model.vis.map.znear) * extent
    return near / (1.0 - depth_map * (1.0 - near / far))


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
        self._last_obs = self.libero_env.reset()

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
            # Contained objects (support_parent_name is set) must stay at their
            # init height inside the container, regardless of ELEVATED_Z_THRESHOLD.
            # This handles bowls inside cabinet drawers, which sit above the normal
            # table surface but must not be relocated to table-surface z.
            is_contained = bool(support_parent_names.get(libero_name, ""))
            # Use LIBERO's default support height only when the generated
            # Scenic object explicitly opts into it AND the object starts near
            # the table surface (or is inside a container at any height).
            # Objects with elevated default_z (e.g. starting on a stove or
            # cabinet shelf that the robot placed them on) should be placed at
            # table-surface z when their XY position is being perturbed to the
            # table area — but contained objects are the exception since their
            # fixture holds them at the correct height.
            if (
                preserve_default_z
                and libero_name in default_pose
                and (default_pose[libero_name][2] <= ELEVATED_Z_THRESHOLD or is_contained)
            ):
                pos[2] = default_pose[libero_name][2]
            else:
                pos[2] = _surface_spawn_z(
                    root_surface_z,
                    getattr(obj, "asset_class", "_default"),
                )
                table_spawned_names.add(libero_name)
            self._inject_object_pose(libero_name, pos, obj)
            injected_targets[libero_name] = pos.copy()
            object_dimensions[libero_name] = get_dimensions(getattr(obj, "asset_class", "_default"))
            n_injected += 1

        self._apply_articulation_perturbation()

        # ── apply environment perturbations from Scenic params ──────────
        self._apply_camera_perturbation()
        self._apply_lighting_perturbation()
        self._apply_texture_perturbation()
        self._apply_background_perturbation()

        if n_injected > 0 or self._has_env_perturbation():
            import mujoco

            mjmodel = self.libero_env.env.sim.model._model
            mjdata = self.libero_env.env.sim.data._data

            # Zero all velocities so injected objects don't inherit stale momentum.
            mjdata.qvel[:] = 0
            mujoco.mj_forward(mjmodel, mjdata)

            # Run settling steps so objects come to rest on the table surface
            # before the episode begins.  Re-zero velocities afterwards so
            # the policy starts from a quiescent state.
            for _ in range(50):
                mujoco.mj_step(mjmodel, mjdata)
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

            # Refresh observation so the first frame reflects settled state.
            self._last_obs = self.libero_env.env._get_observations()
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
            if other_body == "table" or other_body.startswith("table"):
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
            visible = 0
            anchors = 0
            for point in _visibility_anchor_points(center, dims):
                anchors += 1
                visible += int(
                    _anchor_visible(
                        point=point,
                        world_to_pixel=world_to_pixel,
                        world_to_camera=world_to_camera,
                        depth_map=depth_map,
                        image_height=height,
                        image_width=width,
                    )
                )
            if visible == 0:
                failures.append(f"{target_name} is out of frame or fully occluded")
            elif visible < max(1, anchors // 3):
                failures.append(f"{target_name} is only weakly visible in agentview")

        if failures:
            raise VisibilityError(
                "Invalid Scenic sample after visibility check: " + "; ".join(failures),
                invisible_names=failures,
            )

    def _apply_camera_perturbation(self) -> None:
        """Perturb agentview camera position and/or tilt from Scenic params.

        Scenic params read from scene.params:
          camera_x_offset  — additive x offset (metres)
          camera_y_offset  — additive y offset (metres)
          camera_z_offset  — additive z offset (metres)
          camera_tilt      — tilt angle in degrees (added to elevation)
        """
        if self.scene is None or self.libero_env is None:
            return
        params = getattr(self.scene, "params", {})

        dx = params.get("camera_x_offset", 0.0)
        dy = params.get("camera_y_offset", 0.0)
        dz = params.get("camera_z_offset", 0.0)
        tilt = params.get("camera_tilt", 0.0)

        if dx == 0 and dy == 0 and dz == 0 and tilt == 0:
            return

        sim = self.libero_env.env.sim

        # Find agentview camera
        try:
            cam_id = sim.model.camera_name2id("agentview")
        except Exception:
            log.warning("agentview camera not found; skipping camera perturbation")
            return

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

        if ambient is not None:
            # Set global ambient light (headlight ambient in MuJoCo model)
            sim.model.vis.headlight.ambient[:] = float(ambient)
            log.debug("  ambient level: %.2f", ambient)

        log.debug(
            "  lighting: intensity=%.2f offset=(%.2f,%.2f,%.2f)",
            intensity or 1.0,
            ldx,
            ldy,
            ldz,
        )

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
            # Pick a random texture from available textures
            n_tex = sim.model.ntex
            if n_tex > 0:
                tex_id = np.random.randint(0, n_tex)
            else:
                return
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

    def _apply_background_perturbation(self) -> None:
        """Perturb wall and floor textures from Scenic params.

        Scenic params read from scene.params:
          wall_texture   — texture name for wall material (``walls_mat``),
                           or ``"random"`` to pick any loaded texture.
          floor_texture  — texture name for floor material (``floorplane``),
                           or ``"random"`` to pick any loaded texture.

        Material names (``walls_mat`` and ``floorplane``) are the names used
        across all LIBERO scene XMLs — confirmed by inspecting every style XML
        in vendor/libero/libero/libero/assets/scenes/.  Missing material or
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

            Returns:
                Integer texture ID, or None if the model has no textures.
            """
            n_tex = sim.model.ntex
            if n_tex <= 0:
                return None
            if texture_name == "random":
                return int(np.random.randint(0, n_tex))
            try:
                return int(sim.model.texture_name2id(texture_name))
            except Exception:
                # Named texture not loaded in this model — fall back to random
                log.debug(
                    "  background: texture '%s' not in model; using random",
                    texture_name,
                )
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
