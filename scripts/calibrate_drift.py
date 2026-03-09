#!/usr/bin/env python3
"""Calibration loop for Scenic program drift parameters.

Identifies worst-case ("adversarial") scenes for each perturbation axis,
runs them through MuJoCo physics (or a zero-drift stub when MuJoCo is
unavailable), measures per-object drift, and binary-searches each
threshold parameter until:

  hard_failure_rate  = 0
  drift_95th_pct     < DRIFT_TARGET_95PCT (default 0.05 m)

Parameters calibrated
─────────────────────
  min_clearance        — minimum object–object clearance (position axis)
  workspace_margin     — inset from workspace edges (position axis)
  distractor_clearance — min clearance between distractors and task objects

Usage
─────
  python scripts/calibrate_drift.py --axis position --workers 4
  python scripts/calibrate_drift.py --axis all --workers 8 --dry-run
  python scripts/calibrate_drift.py --axis distractor --scenes-per-step 20
  python scripts/calibrate_drift.py --axis position --list-scenes

When MuJoCo is not importable the physics step is replaced by a stub that
returns zero drift for every scene.  The calibration infrastructure runs
end-to-end and the results JSON records stub_mode=true, so the recommended
values reflect the stub (zero failures → minimum search bound).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import multiprocessing
import os
import pathlib
import random
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
_SRC_PATH = _REPO_ROOT / "src"
_BDDL_DIR = _REPO_ROOT / "src" / "libero_infinity" / "data" / "libero_runtime" / "bddl_files"
_SCENIC_GEN_FILE = _REPO_ROOT / "src" / "libero_infinity" / "scenic_generator.py"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

# Ensure src is on the path for all workers too.
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

# ---------------------------------------------------------------------------
# Physical constants (mirror simulator.py)
# ---------------------------------------------------------------------------
TABLE_Z = 0.82
TABLE_X_MIN, TABLE_X_MAX = -0.40, 0.40
TABLE_Y_MIN, TABLE_Y_MAX = -0.30, 0.30
MAX_SETTLE_XY_DRIFT = 0.20   # hard limit from simulator.py
MAX_SETTLE_Z_DROP = 0.08     # hard limit from simulator.py
MIN_SETTLED_Z = TABLE_Z - 0.05  # 0.77
SETTLE_WORKSPACE_MARGIN = 0.03

# Calibration targets
DRIFT_TARGET_95PCT = 0.05   # 95th percentile xy drift target (metres)
HARD_FAILURE_RATE_TARGET = 0.0  # fraction of scenes allowed to hard-fail

# Current (uncalibrated) default values in scenic_generator.py
CURRENT_DEFAULTS = {
    "min_clearance": 0.10,
    "workspace_margin": 0.05,
    "distractor_clearance": 0.08,
}

# Search ranges for binary search: (lo, hi, tolerance, n_bisect_steps)
SEARCH_RANGES = {
    "min_clearance":      (0.04, 0.25, 0.005, 8),
    "workspace_margin":   (0.01, 0.12, 0.003, 7),
    "distractor_clearance": (0.03, 0.15, 0.004, 7),
}

# Axes and which parameters they govern
AXIS_PARAMS = {
    "position":   ["min_clearance", "workspace_margin"],
    "distractor": ["distractor_clearance"],
    "object":     [],  # object variant selection — no drift params
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("calibrate_drift")


# ---------------------------------------------------------------------------
# Availability probes (run once at module load)
# ---------------------------------------------------------------------------

def _probe_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401
        return True
    except ImportError:
        return False


def _probe_scenic() -> bool:
    try:
        import scenic  # noqa: F401
        return True
    except ImportError:
        return False


def _probe_libero() -> bool:
    try:
        from libero.libero.envs.env_wrapper import OffScreenRenderEnv  # noqa: F401
        return True
    except ImportError:
        return False


MUJOCO_AVAILABLE = _probe_mujoco()
SCENIC_AVAILABLE = _probe_scenic()
LIBERO_AVAILABLE = _probe_libero()

# Full simulation requires all three.
SIM_AVAILABLE = MUJOCO_AVAILABLE and SCENIC_AVAILABLE and LIBERO_AVAILABLE


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    """Per-scene measurement result."""

    bddl_path: str
    param_name: str
    param_value: float
    scene_id: int = 0

    # Failure flags
    hard_failure: bool = False
    inject_failure: bool = False
    settle_unstable: bool = False
    outside_workspace: bool = False
    scenic_rejection: bool = False  # Scenic couldn't sample a valid scene
    failure_reason: str = ""

    # Per-object measurements (one entry per movable object)
    xy_drifts: list[float] = field(default_factory=list)
    z_drops: list[float] = field(default_factory=list)
    n_objects: int = 0

    # Stub / timing metadata
    stub_mode: bool = False
    elapsed_s: float = 0.0

    @property
    def max_xy_drift(self) -> float:
        return max(self.xy_drifts) if self.xy_drifts else 0.0

    @property
    def mean_xy_drift(self) -> float:
        return sum(self.xy_drifts) / len(self.xy_drifts) if self.xy_drifts else 0.0


@dataclass
class StepSummary:
    """Aggregate statistics for one calibration step (one param value, N scenes)."""

    param_name: str
    param_value: float
    n_scenes: int
    n_hard_failures: int
    hard_failure_rate: float
    median_xy_drift: float
    drift_95pct: float
    max_xy_drift: float
    stub_mode: bool


@dataclass
class AxisResult:
    """Final calibration result for one parameter."""

    axis: str
    param_name: str
    current_value: float
    recommended_value: float
    n_scenes_evaluated: int
    hard_failure_rate_before: float
    drift_95pct_before: float
    hard_failure_rate_after: float
    drift_95pct_after: float
    stub_mode: bool
    search_history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Adversarial scene selection
# ---------------------------------------------------------------------------

def _load_all_bddls() -> list[pathlib.Path]:
    """Return all BDDL files in the data directory."""
    return sorted(_BDDL_DIR.glob("**/*.bddl"))


def _score_bddl_for_position(bddl_path: pathlib.Path) -> float:
    """Score a BDDL file for adversarial position-perturbation testing.

    Higher score = worse-case for min_clearance / workspace_margin:
      +2 per movable object (more objects = more pairwise constraints)
      +3 per non-table goal fixture (fixture placement tightens space)
      +4 if any stacking (stacked objects constrain each other tightly)
      +2 if large object present (plate ≥ 0.20, frypan ≥ 0.22)
    """
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.asset_registry import get_dimensions

    try:
        cfg = TaskConfig.from_bddl(bddl_path)
    except Exception:
        return 0.0

    score = 0.0
    score += 2.0 * len(cfg.movable_objects)

    # Non-table goal fixtures
    goal_fix = [
        f for f in cfg.fixtures
        if f.fixture_class not in {"table", "kitchen_table"}
        and f.instance_name in cfg.goal_fixture_names
    ]
    score += 3.0 * len(goal_fix)

    # Stacking
    if any(o.stacked_on for o in cfg.movable_objects):
        score += 4.0

    # Large objects
    large_classes = {"plate", "chefmate_8_frypan", "wooden_tray", "basket", "bowl_drainer"}
    if any(o.object_class in large_classes for o in cfg.movable_objects):
        score += 2.0

    return score


def _score_bddl_for_distractor(bddl_path: pathlib.Path) -> float:
    """Score for adversarial distractor testing.

    Most adversarial: many task objects (less room for distractors) OR
    few task objects (distractors are the only things crowding space).
    We want both extremes — score = crowding potential.
    """
    from libero_infinity.task_config import TaskConfig

    try:
        cfg = TaskConfig.from_bddl(bddl_path)
    except Exception:
        return 0.0

    n = len(cfg.movable_objects)
    # Adversarial: very crowded (many task objects + max distractors)
    # OR very sparse (task objects + max distractors at high density)
    return float(n)


def select_adversarial_scenes(
    axis: str,
    n_boundary: int = 6,
    n_random: int = 8,
    rng_seed: int = 42,
) -> list[pathlib.Path]:
    """Select adversarial BDDL files for the given perturbation axis.

    Returns up to ``n_boundary + n_random`` files:
      - The ``n_boundary`` highest-scoring files (hand-picked worst cases).
      - ``n_random`` randomly sampled files from the remaining pool.

    Args:
        axis:       Perturbation axis (``"position"``, ``"distractor"``, ``"object"``).
        n_boundary: Number of worst-case (highest-score) files.
        n_random:   Number of randomly sampled files.
        rng_seed:   RNG seed for reproducibility.

    Returns:
        Deduplicated list of BDDL paths.
    """
    rng = random.Random(rng_seed)
    all_bddls = _load_all_bddls()

    if axis in ("position", "object"):
        scored = [(b, _score_bddl_for_position(b)) for b in all_bddls]
    elif axis == "distractor":
        scored = [(b, _score_bddl_for_distractor(b)) for b in all_bddls]
    else:
        scored = [(b, 1.0) for b in all_bddls]

    scored.sort(key=lambda x: x[1], reverse=True)

    boundary = [b for b, _ in scored[:n_boundary]]
    remaining = [b for b, _ in scored[n_boundary:]]
    sampled = rng.sample(remaining, min(n_random, len(remaining)))

    selected = list(dict.fromkeys(boundary + sampled))  # dedup, preserve order
    return selected


# ---------------------------------------------------------------------------
# Physics measurement — stub (no MuJoCo)
# ---------------------------------------------------------------------------

def _measure_stub(
    bddl_path: str,
    param_name: str,
    param_value: float,
    scene_id: int,
) -> DriftResult:
    """Zero-drift stub used when MuJoCo is unavailable.

    Returns a DriftResult with all measurements zero and hard_failure=False.
    This allows the calibration infrastructure to run end-to-end and produce
    a valid calibration_results.json.  Recommended values will equal the
    current defaults (zero failures → minimum tested value).
    """
    t0 = time.perf_counter()

    # Parse BDDL to get object count (no simulation needed)
    from libero_infinity.task_config import TaskConfig
    try:
        cfg = TaskConfig.from_bddl(bddl_path)
        n_obj = len(cfg.movable_objects)
    except Exception:
        n_obj = 1

    elapsed = time.perf_counter() - t0
    return DriftResult(
        bddl_path=bddl_path,
        param_name=param_name,
        param_value=param_value,
        scene_id=scene_id,
        hard_failure=False,
        inject_failure=False,
        settle_unstable=False,
        outside_workspace=False,
        scenic_rejection=False,
        failure_reason="",
        xy_drifts=[0.0] * n_obj,
        z_drops=[0.0] * n_obj,
        n_objects=n_obj,
        stub_mode=True,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Physics measurement — real (MuJoCo + Scenic + LIBERO)
# ---------------------------------------------------------------------------

def _make_scenic_params(param_name: str, param_value: float, axis: str) -> dict:
    """Build kwargs for generate_scenic() based on which parameter we're tuning."""
    kwargs: dict[str, Any] = {}
    if param_name == "min_clearance":
        kwargs["min_clearance"] = param_value
        kwargs["perturbation"] = "position"
    elif param_name == "workspace_margin":
        kwargs["workspace_margin"] = param_value
        kwargs["perturbation"] = "position"
    elif param_name == "distractor_clearance":
        kwargs["distractor_clearance"] = param_value
        kwargs["perturbation"] = "distractor"
    else:
        kwargs["perturbation"] = "position"
    return kwargs


def _ensure_model_symlink(out_dir: pathlib.Path) -> None:
    """Ensure libero_model.scenic is reachable from out_dir (symlink if needed)."""
    scenic_root = (_REPO_ROOT / "scenic").resolve()
    model_src = scenic_root / "libero_model.scenic"
    model_link = out_dir / "libero_model.scenic"
    if model_src.exists() and not model_link.exists() and not model_link.is_symlink():
        try:
            model_link.symlink_to(model_src)
        except OSError:
            pass


def _measure_real(
    bddl_path: str,
    param_name: str,
    param_value: float,
    scene_id: int,
    max_scenic_iters: int = 3000,
    settle_steps: int = 50,
) -> DriftResult:
    """Full MuJoCo measurement for one BDDL + parameter configuration.

    Flow:
      1. Parse BDDL → TaskConfig
      2. generate_scenic() with param overridden
      3. Write .scenic to temp file; import via scenic.scenarioFromFile()
      4. Sample one scene (up to max_scenic_iters)
      5. Inject sampled positions into MuJoCo; run settle_steps physics steps
      6. Measure per-object xy_drift and z_drop relative to injected targets

    Returns a DriftResult.  On any exception the hard_failure flag is set.
    """
    import numpy as np
    import mujoco

    from libero_infinity.task_config import TaskConfig
    from libero_infinity.scenic_generator import generate_scenic
    from libero_infinity.simulator import (
        TABLE_X_MIN, TABLE_X_MAX, TABLE_Y_MIN, TABLE_Y_MAX,
        DEFAULT_ORIENTATIONS,
        _scenic_quat,
    )
    _SETTLE_MARGIN = 0.05  # local fallback margin for workspace check
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    t0 = time.perf_counter()
    result = DriftResult(
        bddl_path=bddl_path,
        param_name=param_name,
        param_value=param_value,
        scene_id=scene_id,
        stub_mode=False,
    )

    # ── 1. Parse BDDL ──────────────────────────────────────────────────────
    try:
        cfg = TaskConfig.from_bddl(bddl_path)
        result.n_objects = len(cfg.movable_objects)
    except Exception as exc:
        result.hard_failure = True
        result.inject_failure = True
        result.failure_reason = f"BDDL parse error: {exc}"
        result.elapsed_s = time.perf_counter() - t0
        return result

    # ── 2. Generate Scenic program ─────────────────────────────────────────
    axis = "position" if param_name in ("min_clearance", "workspace_margin") else "distractor"
    kwargs = _make_scenic_params(param_name, param_value, axis)
    try:
        scenic_code = generate_scenic(cfg, **kwargs)
    except ValueError as exc:
        # workspace too small for fixture — this is a hard failure mode
        result.hard_failure = True
        result.failure_reason = f"Scenic generation error: {exc}"
        result.elapsed_s = time.perf_counter() - t0
        return result

    # ── 3. Write .scenic to temp dir ───────────────────────────────────────
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="calib_scenic_"))
    _ensure_model_symlink(tmp_dir)
    scenic_path = tmp_dir / f"calib_{scene_id}.scenic"
    scenic_path.write_text(scenic_code)

    try:
        # ── 4. Sample a Scenic scene ───────────────────────────────────────
        import scenic as sc
        try:
            scenario = sc.scenarioFromFile(str(scenic_path))
            scene, _n_iters = scenario.generate(
                maxIterations=max_scenic_iters, verbosity=0
            )
        except Exception as exc:
            result.hard_failure = True
            result.scenic_rejection = True
            result.failure_reason = f"Scenic sampling failed: {exc}"
            result.elapsed_s = time.perf_counter() - t0
            return result

        # ── 5. Inject + settle in MuJoCo ──────────────────────────────────
        # Build env — offscreen renderer required for proper physics init
        # (same flags as LIBEROSimulation.setup() in simulator.py)
        env_cfg = dict(
            bddl_file_name=str(bddl_path),
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="agentview",
            camera_names=["agentview"],
            camera_heights=64,
            camera_widths=64,
            control_freq=20,
            horizon=1,
            ignore_done=True,
            hard_reset=True,
        )
        try:
            env = OffScreenRenderEnv(**env_cfg)
            env.reset()
        except Exception as exc:
            result.hard_failure = True
            result.inject_failure = True
            result.failure_reason = f"Env init/reset failed: {exc}"
            result.elapsed_s = time.perf_counter() - t0
            return result

        sim = env.env.sim
        mjmodel = sim.model._model
        mjdata = sim.data._data

        # Record default z heights
        default_pose: dict[str, np.ndarray] = {}
        for obj in scene.objects:
            name = getattr(obj, "libero_name", None)
            if not name:
                continue
            joint_name = f"{name}_joint0"
            try:
                qpos = sim.data.get_joint_qpos(joint_name)
                default_pose[name] = np.array(qpos[:3], dtype=float)
            except Exception:
                for body_name in (name, name + "_main"):
                    try:
                        bid = sim.model.body_name2id(body_name)
                        default_pose[name] = np.array(
                            sim.data.body_xpos[bid][:3], dtype=float
                        )
                        break
                    except Exception:
                        pass

        # Inject Scenic-sampled positions
        injected_targets: dict[str, np.ndarray] = {}
        for obj in scene.objects:
            name = getattr(obj, "libero_name", None)
            if not name or name.startswith("distractor_"):
                continue
            pos = np.array(obj.position, dtype=float)
            if name in default_pose:
                pos[2] = default_pose[name][2]

            asset_class = getattr(obj, "asset_class", "_default")
            try:
                quat = _scenic_quat(obj.orientation)
            except Exception:
                quat = DEFAULT_ORIENTATIONS.get(asset_class, DEFAULT_ORIENTATIONS["_default"]).copy()

            qpos7 = np.concatenate([pos, quat])
            joint_name = f"{name}_joint0"
            injected = False
            try:
                sim.data.set_joint_qpos(joint_name, qpos7)
                injected = True
            except Exception:
                for body_name in (name, name + "_main"):
                    try:
                        bid = sim.model.body_name2id(body_name)
                        sim.model.body_pos[bid] = pos
                        injected = True
                        break
                    except Exception:
                        pass
            if injected:
                injected_targets[name] = pos.copy()

        # Zero velocities and settle
        mjdata.qvel[:] = 0
        mujoco.mj_forward(mjmodel, mjdata)
        for _ in range(settle_steps):
            mujoco.mj_step(mjmodel, mjdata)
        mjdata.qvel[:] = 0
        mujoco.mj_forward(mjmodel, mjdata)

        # ── 6. Measure drift ──────────────────────────────────────────────
        xy_drifts: list[float] = []
        z_drops: list[float] = []
        failures: list[str] = []

        # Compute task-specific minimum settled z from default poses.
        # Different LIBERO suites use different arena/table setups:
        #   libero_goal / libero_spatial: objects at z ≈ 0.93-0.97
        #   libero_10 / libero_90 / libero_100: objects at z ≈ 0.44-0.48
        # Using the default_pose minimum avoids false positives for non-goal suites.
        if default_pose:
            task_min_z = min(
                p[2] for p in default_pose.values()
            ) - MAX_SETTLE_Z_DROP
        else:
            task_min_z = MIN_SETTLED_Z

        for name, target in injected_targets.items():
            body_id = None
            for candidate in (name, name + "_main"):
                try:
                    body_id = sim.model.body_name2id(candidate)
                    break
                except Exception:
                    pass
            if body_id is None:
                continue

            final_pos = np.array(sim.data.body_xpos[body_id][:3], dtype=float)

            # Non-finite check (physics exploded)
            if not np.all(np.isfinite(final_pos)):
                failures.append(f"{name}: non-finite position after settle")
                result.settle_unstable = True
                continue

            xy_drift = float(np.linalg.norm(final_pos[:2] - target[:2]))
            xy_drifts.append(xy_drift)

            z_drop = float(target[2] - final_pos[2])
            z_drops.append(z_drop)

            # Hard-failure conditions (mirrors simulator.py _validate_settled_positions):
            # 1. Large xy drift from injected target
            if xy_drift > MAX_SETTLE_XY_DRIFT:
                failures.append(
                    f"{name}: xy_drift={xy_drift:.3f} > {MAX_SETTLE_XY_DRIFT}"
                )
            # 2. Significant z drop from LIBERO's own default placement height
            dflt = default_pose.get(name)
            if dflt is not None and final_pos[2] < dflt[2] - MAX_SETTLE_Z_DROP:
                failures.append(
                    f"{name}: z-drop {dflt[2]:.3f}→{final_pos[2]:.3f}"
                )

            # Supplementary workspace check (not in simulator.py but useful
            # for calibration — records outside_workspace flag for metrics)
            if not (
                TABLE_X_MIN - _SETTLE_MARGIN <= final_pos[0] <= TABLE_X_MAX + _SETTLE_MARGIN
                and TABLE_Y_MIN - _SETTLE_MARGIN <= final_pos[1] <= TABLE_Y_MAX + _SETTLE_MARGIN
            ):
                result.outside_workspace = True
                # Don't add to failures for workspace check — simulator.py
                # doesn't reject on this; it's informational only.

        result.xy_drifts = xy_drifts
        result.z_drops = z_drops
        result.n_objects = len(xy_drifts)

        if failures:
            result.hard_failure = True
            result.settle_unstable = True
            result.failure_reason = "; ".join(failures)

        try:
            env.close()
        except Exception:
            pass

    finally:
        # Cleanup temp dir
        for f in tmp_dir.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
        try:
            tmp_dir.rmdir()
        except Exception:
            pass

    result.elapsed_s = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Worker function (top-level for pickle / multiprocessing)
# ---------------------------------------------------------------------------

def _worker_fn(args: tuple) -> DriftResult:
    """Top-level worker function executed in a subprocess.

    Args (positional tuple):
        bddl_path    (str)
        param_name   (str)
        param_value  (float)
        scene_id     (int)
        use_stub     (bool)  — True when MuJoCo is not available
    """
    bddl_path, param_name, param_value, scene_id, use_stub = args
    if str(_SRC_PATH) not in sys.path:
        sys.path.insert(0, str(_SRC_PATH))

    if use_stub:
        return _measure_stub(bddl_path, param_name, param_value, scene_id)
    else:
        return _measure_real(bddl_path, param_name, param_value, scene_id)


# ---------------------------------------------------------------------------
# Parallel evaluation engine
# ---------------------------------------------------------------------------

def evaluate_param(
    param_name: str,
    param_value: float,
    bddl_paths: list[pathlib.Path],
    n_workers: int,
    n_repeats: int = 3,
    use_stub: bool = True,
) -> tuple[StepSummary, list[DriftResult]]:
    """Evaluate ``param_value`` across adversarial scenes using parallel workers.

    Each BDDL is run ``n_repeats`` times (different random seeds in Scenic)
    to reduce variance.

    Returns (StepSummary, list[DriftResult]).
    """
    import numpy as np

    tasks = [
        (str(b), param_name, param_value, idx * n_repeats + rep, use_stub)
        for idx, b in enumerate(bddl_paths)
        for rep in range(n_repeats)
    ]

    results: list[DriftResult] = []
    if n_workers <= 1:
        # Single-process mode — avoids multiprocessing overhead in tests
        for task_args in tasks:
            results.append(_worker_fn(task_args))
    else:
        # Use 'spawn' start method to avoid EGL/OpenGL context conflicts
        # that arise when forking a process that has already imported mujoco/EGL.
        # 'spawn' creates a fresh interpreter per worker — slightly slower
        # startup but avoids GPU context corruption in parallel physics workers.
        mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp_ctx
        ) as exe:
            futures = {exe.submit(_worker_fn, a): a for a in tasks}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    # Worker crashed — treat as hard failure
                    args_ = futures[fut]
                    results.append(
                        DriftResult(
                            bddl_path=args_[0],
                            param_name=param_name,
                            param_value=param_value,
                            scene_id=args_[3],
                            hard_failure=True,
                            failure_reason=f"Worker exception: {exc}",
                        )
                    )

    # ── Aggregate ──────────────────────────────────────────────────────────
    n_hard = sum(1 for r in results if r.hard_failure)
    all_xy = [d for r in results for d in r.xy_drifts]

    if all_xy:
        all_xy_np = np.array(all_xy)
        median_xy = float(np.median(all_xy_np))
        drift_95 = float(np.percentile(all_xy_np, 95))
        max_xy = float(np.max(all_xy_np))
    else:
        median_xy = drift_95 = max_xy = 0.0

    summary = StepSummary(
        param_name=param_name,
        param_value=param_value,
        n_scenes=len(results),
        n_hard_failures=n_hard,
        hard_failure_rate=n_hard / len(results) if results else 0.0,
        median_xy_drift=median_xy,
        drift_95pct=drift_95,
        max_xy_drift=max_xy,
        stub_mode=use_stub,
    )
    return summary, results


# ---------------------------------------------------------------------------
# Binary-search calibration loop
# ---------------------------------------------------------------------------

def calibrate_param(
    axis: str,
    param_name: str,
    bddl_paths: list[pathlib.Path],
    n_workers: int,
    n_scenes_per_step: int = 20,
    n_repeats: int = 1,
    drift_target_95pct: float = DRIFT_TARGET_95PCT,
    use_stub: bool = True,
) -> AxisResult:
    """Binary-search for the minimum ``param_name`` value that meets targets.

    The search minimises the parameter value subject to:
      hard_failure_rate ≤ HARD_FAILURE_RATE_TARGET  AND
      drift_95pct       ≤ drift_target_95pct

    Strategy:
      1. Evaluate at ``current_value`` (baseline).
      2. Binary-search between (lo, hi) for the minimum safe value.
         - If mid satisfies both targets → try smaller (hi = mid).
         - Otherwise → need larger (lo = mid).
      3. Return ``AxisResult`` with recommended = lowest safe value found.

    With the zero-drift stub every value satisfies the targets, so the
    recommended value = lo (the smallest searched value).
    """
    import numpy as np

    lo, hi, tol, n_bisect = SEARCH_RANGES[param_name]
    current = CURRENT_DEFAULTS[param_name]

    log.info(
        "Calibrating %s (axis=%s): current=%.4f, search=[%.4f, %.4f], "
        "n_bddl=%d, n_workers=%d, stub=%s",
        param_name, axis, current, lo, hi, len(bddl_paths), n_workers, use_stub,
    )

    # ── Baseline at current default ────────────────────────────────────────
    log.info("  Baseline evaluation at current=%.4f ...", current)
    baseline_summary, baseline_results = evaluate_param(
        param_name=param_name,
        param_value=current,
        bddl_paths=bddl_paths[:min(len(bddl_paths), n_scenes_per_step)],
        n_workers=n_workers,
        n_repeats=n_repeats,
        use_stub=use_stub,
    )
    _log_step(baseline_summary, label="baseline")

    search_history: list[dict] = [
        {"step": "baseline", **_summary_dict(baseline_summary)}
    ]

    # ── Binary search ──────────────────────────────────────────────────────
    best_value = current  # conservative: keep current if nothing else satisfies
    best_summary = baseline_summary

    for step in range(n_bisect):
        mid = (lo + hi) / 2.0
        if abs(hi - lo) < tol:
            break

        log.info("  Step %d/%d: testing %.4f ...", step + 1, n_bisect, mid)
        step_summary, step_results = evaluate_param(
            param_name=param_name,
            param_value=mid,
            bddl_paths=bddl_paths[:min(len(bddl_paths), n_scenes_per_step)],
            n_workers=n_workers,
            n_repeats=n_repeats,
            use_stub=use_stub,
        )
        _log_step(step_summary, label=f"step {step + 1}")
        search_history.append(
            {"step": step + 1, "lo": lo, "hi": hi, **_summary_dict(step_summary)}
        )

        passes = (
            step_summary.hard_failure_rate <= HARD_FAILURE_RATE_TARGET
            and step_summary.drift_95pct <= drift_target_95pct
        )
        if passes:
            best_value = mid
            best_summary = step_summary
            hi = mid  # try to go smaller
        else:
            lo = mid  # need larger

    # Final evaluation at the recommended value (if different from baseline)
    if abs(best_value - current) > 0.001:
        log.info("  Final check at recommended=%.4f ...", best_value)
        final_summary, final_results = evaluate_param(
            param_name=param_name,
            param_value=best_value,
            bddl_paths=bddl_paths[:min(len(bddl_paths), n_scenes_per_step)],
            n_workers=n_workers,
            n_repeats=n_repeats,
            use_stub=use_stub,
        )
        _log_step(final_summary, label="final")
        search_history.append({"step": "final", **_summary_dict(final_summary)})
    else:
        final_summary = baseline_summary

    return AxisResult(
        axis=axis,
        param_name=param_name,
        current_value=current,
        recommended_value=round(best_value, 4),
        n_scenes_evaluated=sum(d["n_scenes"] for d in search_history),
        hard_failure_rate_before=baseline_summary.hard_failure_rate,
        drift_95pct_before=baseline_summary.drift_95pct,
        hard_failure_rate_after=final_summary.hard_failure_rate,
        drift_95pct_after=final_summary.drift_95pct,
        stub_mode=use_stub,
        search_history=search_history,
    )


def _summary_dict(s: StepSummary) -> dict:
    return {
        "param_value": round(s.param_value, 5),
        "n_scenes": s.n_scenes,
        "n_hard_failures": s.n_hard_failures,
        "hard_failure_rate": round(s.hard_failure_rate, 4),
        "median_xy_drift": round(s.median_xy_drift, 5),
        "drift_95pct": round(s.drift_95pct, 5),
        "max_xy_drift": round(s.max_xy_drift, 5),
        "stub_mode": s.stub_mode,
    }


def _log_step(s: StepSummary, label: str = "") -> None:
    tag = f"[{label}] " if label else ""
    log.info(
        "  %s%s=%.4f: failures=%d/%d (%.0f%%), "
        "drift median=%.4f p95=%.4f max=%.4f%s",
        tag, s.param_name, s.param_value,
        s.n_hard_failures, s.n_scenes,
        s.hard_failure_rate * 100,
        s.median_xy_drift, s.drift_95pct, s.max_xy_drift,
        " [stub]" if s.stub_mode else "",
    )


# ---------------------------------------------------------------------------
# Write-back to scenic_generator.py
# ---------------------------------------------------------------------------

def write_recommended_values(
    recommendations: dict[str, float],
    dry_run: bool = True,
    out_file: pathlib.Path | None = None,
) -> None:
    """Patch the default parameter values in scenic_generator.py.

    Only ``min_clearance``, ``workspace_margin``, and ``distractor_clearance``
    are patched — these are the default argument values in ``generate_scenic()``.

    Args:
        recommendations: Mapping param_name → recommended_value.
        dry_run:          If True, print the diff but do not write.
        out_file:         Override output path (default: scenic_generator.py).
    """
    target = out_file or _SCENIC_GEN_FILE
    if not target.exists():
        log.warning("scenic_generator.py not found at %s, skipping write-back", target)
        return

    src = target.read_text()
    patched = src

    # Patterns to match default argument values in generate_scenic() signature
    replacements = {
        "min_clearance": r"(min_clearance:\s*float\s*=\s*)([\d.]+)",
        "workspace_margin": r"(workspace_margin:\s*float\s*=\s*)([\d.]+)",
        "distractor_clearance": r"(distractor_clearance:\s*float\s*=\s*)([\d.]+)",
    }

    changed: list[tuple[str, float, float]] = []
    for param, rec_val in recommendations.items():
        pattern = replacements.get(param)
        if pattern is None:
            continue
        m = re.search(pattern, patched)
        if m is None:
            log.warning("Could not find %s default in %s", param, target.name)
            continue
        old_val = float(m.group(2))
        new_src = re.sub(pattern, rf"\g<1>{rec_val}", patched, count=1)
        if new_src != patched:
            patched = new_src
            changed.append((param, old_val, rec_val))

    if not changed:
        log.info("No changes needed in %s", target.name)
        return

    print("\nRecommended changes to %s:" % target.name)
    for param, old_val, new_val in changed:
        direction = "↑" if new_val > old_val else "↓"
        print(f"  {param}: {old_val:.4f} → {new_val:.4f}  {direction}")

    if dry_run:
        print("  (dry-run: not written)")
    else:
        target.write_text(patched)
        log.info("Wrote updated %s", target.name)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_report(axis_results: list[AxisResult], stub_mode: bool) -> None:
    """Print a human-readable calibration report to stdout."""
    sep = "─" * 70
    print()
    print(sep)
    print("  LIBERO Scenic Drift Calibration Report")
    if stub_mode:
        print("  *** STUB MODE (MuJoCo unavailable — zero-drift mock) ***")
    print(sep)
    print()

    for r in axis_results:
        direction = "↑" if r.recommended_value > r.current_value else "↓"
        change = r.recommended_value - r.current_value
        print(f"  Parameter : {r.param_name}  (axis: {r.axis})")
        print(f"  Current   : {r.current_value:.4f}")
        print(f"  Recommended: {r.recommended_value:.4f}  ({direction}{abs(change):.4f})")
        print()
        print(f"  Before calibration:")
        print(f"    Hard-failure rate : {r.hard_failure_rate_before * 100:.1f}%")
        print(f"    95th-pct xy drift : {r.drift_95pct_before * 100:.2f} cm")
        print()
        print(f"  After calibration:")
        print(f"    Hard-failure rate : {r.hard_failure_rate_after * 100:.1f}%")
        print(f"    95th-pct xy drift : {r.drift_95pct_after * 100:.2f} cm")
        print()
        print(f"  Scenes evaluated : {r.n_scenes_evaluated}")
        print(f"  Binary-search steps: {len(r.search_history) - 2}")  # excl. baseline + final
        print(sep)
        print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _build_axis_list(axis_arg: str) -> list[str]:
    if axis_arg == "all":
        return list(AXIS_PARAMS.keys())
    if axis_arg not in AXIS_PARAMS and axis_arg != "all":
        raise ValueError(
            f"Unknown axis {axis_arg!r}. Valid: {list(AXIS_PARAMS.keys()) + ['all']}"
        )
    return [axis_arg]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate Scenic drift parameters via adversarial scene stress-testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--axis",
        default="all",
        choices=list(AXIS_PARAMS.keys()) + ["all"],
        help="Which perturbation axis to calibrate.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Number of parallel workers (ProcessPoolExecutor). "
             "Default: cpu_count // 2.",
    )
    parser.add_argument(
        "--scenes-per-step",
        type=int,
        default=16,
        dest="scenes_per_step",
        help="Number of adversarial scenes to run at each binary-search step.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of Scenic re-samples per BDDL per step (reduces variance).",
    )
    parser.add_argument(
        "--drift-target",
        type=float,
        default=DRIFT_TARGET_95PCT,
        dest="drift_target",
        help="95th-percentile drift target in metres.",
    )
    parser.add_argument(
        "--n-boundary",
        type=int,
        default=6,
        dest="n_boundary",
        help="Number of worst-case (highest-scoring) BDDL files to include.",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=8,
        dest="n_random",
        help="Number of randomly sampled BDDL files to add.",
    )
    parser.add_argument(
        "--bisect-steps",
        type=int,
        default=7,
        dest="bisect_steps",
        help="Number of binary-search steps per parameter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print recommended values but do not write back to scenic_generator.py.",
    )
    parser.add_argument(
        "--write-back",
        action="store_true",
        dest="write_back",
        help="Write recommended values to scenic_generator.py (default: dry-run only).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=_REPO_ROOT / "calibration_results.json",
        help="Path for the calibration results JSON.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        dest="list_scenes",
        help="Print the adversarial scene list for each axis and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for scene selection.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Print environment info ─────────────────────────────────────────────
    print(f"MuJoCo available : {MUJOCO_AVAILABLE}")
    print(f"Scenic available : {SCENIC_AVAILABLE}")
    print(f"LIBERO available : {LIBERO_AVAILABLE}")
    print(f"Full sim mode    : {SIM_AVAILABLE}")
    print(f"Workers          : {args.workers}")
    print(f"Scenes per step  : {args.scenes_per_step}")
    print(f"Repeats per scene: {args.repeats}")
    print()

    use_stub = not SIM_AVAILABLE
    if use_stub:
        log.warning(
            "Full simulation not available — running in STUB mode "
            "(zero drift). Recommended values will equal the search lower bound."
        )

    axes = _build_axis_list(args.axis)

    # ── List-scenes mode ───────────────────────────────────────────────────
    if args.list_scenes:
        for axis in axes:
            scenes = select_adversarial_scenes(
                axis, n_boundary=args.n_boundary, n_random=args.n_random,
                rng_seed=args.seed,
            )
            print(f"\nAdversarial scenes for axis={axis!r} ({len(scenes)} files):")
            for s in scenes:
                print(f"  {s.relative_to(_REPO_ROOT)}")
        return 0

    # ── Calibration loop ───────────────────────────────────────────────────
    all_axis_results: list[AxisResult] = []
    t_start = time.time()

    for axis in axes:
        params = AXIS_PARAMS[axis]
        if not params:
            log.info("Axis %r has no drift parameters — skipping", axis)
            continue

        log.info("=== Axis: %s ===", axis)
        scenes = select_adversarial_scenes(
            axis,
            n_boundary=args.n_boundary,
            n_random=args.n_random,
            rng_seed=args.seed,
        )
        log.info("Selected %d adversarial scenes for axis=%r", len(scenes), axis)

        for param_name in params:
            # Override binary-search step count from args
            lo, hi, tol, default_n_bisect = SEARCH_RANGES[param_name]
            SEARCH_RANGES[param_name] = (lo, hi, tol, args.bisect_steps)

            result = calibrate_param(
                axis=axis,
                param_name=param_name,
                bddl_paths=scenes,
                n_workers=args.workers,
                n_scenes_per_step=args.scenes_per_step,
                n_repeats=args.repeats,
                drift_target_95pct=args.drift_target,
                use_stub=use_stub,
            )
            all_axis_results.append(result)

    # ── Print report ────────────────────────────────────────────────────────
    print_report(all_axis_results, stub_mode=use_stub)

    elapsed = time.time() - t_start
    log.info("Total calibration time: %.1f s", elapsed)

    # ── Write-back ──────────────────────────────────────────────────────────
    recommendations = {r.param_name: r.recommended_value for r in all_axis_results}
    if recommendations:
        write_recommended_values(
            recommendations,
            dry_run=not args.write_back,
        )

    # ── Save results JSON ───────────────────────────────────────────────────
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stub_mode": use_stub,
        "mujoco_available": MUJOCO_AVAILABLE,
        "scenic_available": SCENIC_AVAILABLE,
        "libero_available": LIBERO_AVAILABLE,
        "workers": args.workers,
        "scenes_per_step": args.scenes_per_step,
        "repeats": args.repeats,
        "drift_target_95pct": args.drift_target,
        "elapsed_seconds": round(elapsed, 2),
        "axes_calibrated": axes,
        "results": [asdict(r) for r in all_axis_results],
        "recommendations": recommendations,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2))
    log.info("Calibration results written to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
