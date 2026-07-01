"""Gym-compatible environment wrapper for LIBERO-Infinity.

Wraps the full Scenic perturbation pipeline (BDDL → TaskConfig →
compiler → constraint solver → LIBERO simulation) into a standard
``gym.Env`` that RL and VLA training loops can use directly.

Each ``reset()`` samples a new scene from the Scenic program (randomising
object positions, assets, camera, lighting, etc. according to the selected
perturbation mode) and returns a fresh observation dict.

Uses gym 0.25 API (4-tuple ``step()`` returns).

Usage::

    from libero_infinity.gym_env import LIBEROScenicEnv

    env = LIBEROScenicEnv(
        bddl_path="src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/"
                  "put_the_bowl_on_the_plate.bddl",
        perturbation="combined",
        resolution=256,
    )

    obs = env.reset()
    for _ in range(300):
        action = my_policy(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()

Parallel rollouts::

    from libero_infinity.gym_env import make_vec_env

    vec_env = make_vec_env(
        bddl_path="path/to/task.bddl",
        n_envs=4,
        perturbation="position",
    )
    obs = vec_env.reset()                # (4, ...) batched observations
    obs, rewards, dones, infos = vec_env.step(actions)
    vec_env.close()
"""

from __future__ import annotations

import contextlib
import logging
import pathlib
import warnings
from typing import Any

# gym 0.25.2 is pinned for robosuite 1.4.0 compatibility.
# Suppress the "Gym has been unmaintained since 2022" deprecation warning.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
    import gym
    from gym import spaces

import numpy as np

log = logging.getLogger(__name__)


class LIBEROScenicEnv(gym.Env):
    """Gym environment that samples perturbed LIBERO scenes via Scenic 3.

    Each ``reset()`` generates a new scene from the compiled Scenic program,
    resolves any BDDL substitutions, creates a fresh LIBEROSimulation, and
    returns the first observation.

    Parameters
    ----------
    bddl_path :
        Path to the BDDL task file.
    perturbation :
        Perturbation specification. Accepts a single axis (``"position"``),
        a preset (``"combined"``, ``"full"``), or a comma-separated list
        (``"position,camera,distractor"``).
    scenic_path :
        Optional path to a hand-written ``.scenic`` program. If ``None``
        (default), a program is auto-generated from the BDDL.
    resolution :
        Camera image resolution in pixels (default 128).
    max_steps :
        Episode horizon (default 300).
    seed :
        Optional RNG seed. Seeds Python ``random``, ``numpy.random``, and
        ``torch`` (if available) on first ``reset()``. Also seeds Scenic's
        rejection-sampler because Scenic uses Python ``random`` internally.
    reverse :
        If ``True``, reverse the task (goal becomes init, init becomes goal).
    scenic_params :
        Extra overrides for Scenic ``globalParameters``.
    env_kwargs :
        Extra kwargs forwarded to ``OffScreenRenderEnv``.
    scenic_generate_kwargs :
        Extra kwargs for ``generate_scenic()`` (e.g. ``min_clearance``,
        ``max_distractors``).
    max_scenic_iterations :
        Cap on Scenic rejection-sampling iterations per scene
        (``Scenario.generate(maxIterations=...)``). If ``None`` (default), the
        budget is resolved per perturbation mode from the measured calibration
        artifact (``data/scenic_iteration_budgets.json``) via
        :func:`libero_infinity.scenic_budget.resolve_iteration_budget` — harder
        modes (``combined``/``full``) get larger budgets while simple modes keep
        the historical 5000. Pass an int to override.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        bddl_path: str | pathlib.Path,
        perturbation: str = "position",
        scenic_path: str | pathlib.Path | None = None,
        resolution: int = 128,
        max_steps: int = 300,
        seed: int | None = None,
        reverse: bool = False,
        scenic_params: dict[str, Any] | None = None,
        env_kwargs: dict[str, Any] | None = None,
        scenic_generate_kwargs: dict[str, Any] | None = None,
        scene: Any = None,
        max_scenic_iterations: int | None = None,
    ):
        super().__init__()

        self._bddl_path = str(pathlib.Path(bddl_path).resolve())
        # Optional pre-sampled Scenic scene. When provided (e.g. by the
        # validation sweep harness via ``make_env``), the next ``reset()``
        # call will use this scene instead of generating a fresh one. This
        # lets G5 exercise the exact scene that earlier G-gates validated.
        # Consumed on first use; subsequent resets resample as usual.
        self._preset_scene: Any = scene
        self._perturbation = perturbation
        self._resolution = resolution
        self._max_steps = max_steps
        self._seed = seed
        self._reverse = reverse
        self._scenic_params = scenic_params or {}
        self._env_kwargs = env_kwargs or {}
        self._scenic_generate_kwargs = scenic_generate_kwargs or {}

        # Managed resources (init before any code that could raise, so __del__
        # cleanup never hits a missing attribute).
        self._exit_stack = contextlib.ExitStack()  # for long-lived resources (reversed BDDL)
        self._scenario = None
        self._sim: Any = None  # LIBEROSimulation
        self._generated_scenic_path: str | None = None
        self._per_reset_stack: contextlib.ExitStack | None = None  # per-episode resources

        # Resolve the per-mode Scenic iteration budget once. An explicit value
        # wins; otherwise the budget is derived from the perturbation mode using
        # the measured calibration artifact (back-compat default 5000).
        from libero_infinity.scenic_budget import resolve_iteration_budget

        self._max_scenic_iterations = resolve_iteration_budget(perturbation, max_scenic_iterations)

        # Action space: 7D continuous [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        # Observation space is constructed lazily after first reset() because
        # the exact keys and shapes depend on the BDDL task.
        self.observation_space = spaces.Dict({})
        self._obs_space_set = False

        # Compile the Scenic scenario once (expensive — involves Scenic
        # parsing + Python code generation).
        self._compile_scenario(scenic_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    # Maximum number of times reset() will resample a Scenic scene when
    # MuJoCo settling validation rejects it (e.g. a tall object topples).
    _MAX_SETTLE_RETRIES: int = 10

    def reset(self) -> dict[str, np.ndarray]:
        """Sample a new perturbed scene and return initial observation.

        Returns
        -------
        obs : dict[str, np.ndarray]
            Observation dict with visual, proprioceptive, and object-state
            keys. See ``docs/observations-actions.md`` for full schema.
        """
        if self._seed is not None:
            import random

            random.seed(self._seed)
            np.random.seed(self._seed)
            try:
                import torch

                torch.manual_seed(self._seed)
            except ImportError:
                pass
            self._seed = None  # only seed once

        # Destroy previous simulation and clean up per-episode resources.
        self._cleanup_sim()
        self._cleanup_per_reset()

        # Create and set up the LIBERO simulation.
        from libero_infinity.simulator import LIBEROSimulator

        env_kw = {
            "camera_heights": self._resolution,
            "camera_widths": self._resolution,
        }
        env_kw.update(self._env_kwargs)

        # Retry loop: if MuJoCo settling validation rejects the sample
        # (e.g. a tall object topples and drifts off the table), resample
        # a fresh Scenic scene rather than propagating the error.
        for attempt in range(self._MAX_SETTLE_RETRIES + 1):
            # Generate a new scene from the Scenic program.
            # The iteration budget is task/mode-adaptive (WS-3): radial
            # footprint-clearance constraints (task objects vs fixtures) are
            # tight, and harder modes (combined/full) need far more rejection-
            # sampling attempts than the historical global 5000.
            #
            # If a preset scene was injected (sweep harness contract via
            # ``make_env``), consume it on the first attempt of the first
            # reset; on retries (settle rejection) fall back to resampling.
            if self._preset_scene is not None and attempt == 0:
                scene = self._preset_scene
                self._preset_scene = None  # consume once
            else:
                scene, _n_iters = self._scenario.generate(
                    maxIterations=self._max_scenic_iterations,
                    verbosity=0,
                )
                # Early signal that the budget is too tight for this mode/task.
                from libero_infinity.scenic_budget import warn_if_near_budget

                warn_if_near_budget(
                    _n_iters,
                    self._max_scenic_iterations,
                    mode=self._perturbation,
                    logger=log,
                )

            # Resolve BDDL substitutions (asset swaps) via proper context manager.
            self._per_reset_stack = contextlib.ExitStack()
            effective_bddl = self._per_reset_stack.enter_context(
                self._resolve_bddl_for_scene(scene)
            )

            # Parse the *effective* BDDL (post-substitution) for the
            # asset-class map consumed by ``get_object_state`` (G4 family-C
            # env-side accessor). The effective BDDL is what LIBERO/MuJoCo
            # actually loaded, so its class strings are the ground truth.
            try:
                from libero_infinity.bddl_preprocessor import parse_object_classes

                self._effective_obj_classes = parse_object_classes(
                    pathlib.Path(effective_bddl).read_text()
                )
            except Exception:  # noqa: BLE001 — class map is best-effort
                self._effective_obj_classes = {}

            simulator = LIBEROSimulator(
                bddl_path=effective_bddl,
                env_kwargs=env_kw,
            )
            self._sim = simulator.createSimulation(
                scene,
                maxSteps=self._max_steps,
                timestep=0.05,
                verbosity=0,
            )

            try:
                self._sim.setup()
                break  # scene settled successfully
            except Exception as exc:
                from libero_infinity.validation_errors import (
                    CollisionError,
                    ScenarioValidationError,
                    VisibilityError,
                )

                # CollisionError is raised both for true object-object overlaps AND
                # for post-settle rotation drift (added by _validate_settled_positions).
                # Rotation drift is a transient physics artifact — retry on a fresh
                # Scenic sample. Both cases are recoverable by resampling.
                if not isinstance(exc, (CollisionError, VisibilityError, ScenarioValidationError)):
                    raise  # unrelated error — propagate immediately
                if attempt >= self._MAX_SETTLE_RETRIES:
                    raise RuntimeError(
                        f"reset() failed to find a valid scene after "
                        f"{self._MAX_SETTLE_RETRIES} retries. Last error: {exc}"
                    ) from exc
                log.warning(
                    "Validation failed (attempt %d/%d): %s — resampling",
                    attempt + 1,
                    self._MAX_SETTLE_RETRIES,
                    exc,
                )
                self._cleanup_sim()
                self._cleanup_per_reset()

        obs = self._sim.last_obs
        if obs is None:
            obs = {}

        # Build observation space on first reset.
        if not self._obs_space_set:
            self._build_obs_space(obs)
            self._obs_space_set = True

        self._steps = 0
        return obs

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        """Execute one control step.

        Parameters
        ----------
        action : np.ndarray
            Shape ``(7,)`` with values in ``[-1, 1]``.

        Returns
        -------
        obs : dict[str, np.ndarray]
        reward : float
            ``1.0`` if the task is completed at this step, ``0.0`` otherwise.
        done : bool
            ``True`` if horizon reached or task completed.
        info : dict
            Contains ``"success"`` (bool) and ``"steps"`` (int).
        """
        if self._sim is None:
            raise RuntimeError("Call reset() before step()")

        action = np.asarray(action, dtype=np.float64)
        obs, _reward, done, _info = self._sim.step_with_action(action)
        self._steps += 1

        success = self._sim.check_success()
        if success:
            done = True

        reward = 1.0 if success else 0.0
        info = {"success": success, "steps": self._steps}

        if self._steps >= self._max_steps:
            done = True

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # G4 family-C env-side accessor
    # ------------------------------------------------------------------

    def get_object_state(self, name: str) -> dict[str, Any] | None:
        """Return ``{position, orientation, class}`` for a Scenic-named object.

        Used by :func:`libero_infinity.validation.invariants.assert_consistency`
        to compare the Scenic-sampled scene against the live MuJoCo state
        after :meth:`reset` — the G4 family-C consistency invariant. Returns
        ``None`` when ``name`` is unknown so the caller can surface it as a
        genuine consistency failure (silent omission would defeat the check).

        - ``position`` is the world-frame body XYZ from MuJoCo
          (``sim.data.body_xpos``).
        - ``orientation`` is the body quaternion ``(w, x, y, z)`` from
          ``sim.data.body_xquat`` (MuJoCo's native convention).
        - ``class`` is the asset class string parsed from the *effective*
          (post-substitution) BDDL — what LIBERO/MuJoCo actually loaded.
        """
        if self._sim is None:
            return None
        body_ids = getattr(self._sim, "_body_ids", None) or {}
        bid = body_ids.get(name)
        if bid is None:
            return None
        try:
            sim_data = self._sim.libero_env.env.sim.data
            pos = sim_data.body_xpos[bid]
            quat = sim_data.body_xquat[bid]
            position = (float(pos[0]), float(pos[1]), float(pos[2]))
            orientation = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        except Exception:  # noqa: BLE001 — MuJoCo handle gone is a real fail
            return None
        cls_map = getattr(self, "_effective_obj_classes", {}) or {}
        cls = cls_map.get(name)
        state: dict[str, Any] = {
            "position": position,
            "orientation": orientation,
            "class": cls,
        }
        # End-of-settle CONVERGENCE signal captured by the simulator over the
        # last few steps of its settle (linear displacement in metres, angular
        # in degrees). This is the genuine "did the object converge to a fixed
        # point?" signal the G4 pose_tolerance alt-rest path needs — a live qvel
        # read after reset is vacuously ~0 (velocities are zeroed), and the
        # instantaneous end-of-settle spatial velocity is a frame-dependent
        # artifact for a body in steady contact. Absent for objects that never
        # went through the settle path (no injection); the scorer then declines
        # the alt-rest path and falls back to the strict gate (fail-safe).
        settle_conv = getattr(self._sim, "_settle_convergence", None) or {}
        sc = settle_conv.get(name)
        if sc is not None:
            state["settle_conv_lin"] = float(sc[0])
            state["settle_conv_ang"] = float(sc[1])
        # Canonical (as-placed) orientation, so the alt-rest path can measure
        # "upright" as env-settled vs canonical — the Scenic object's own
        # orientation attribute does not coerce to a quaternion for real
        # LIBEROObjects, so the strict scenic-vs-env rotation term is vacuous
        # there and cannot be relied on to detect a tip. ``_canonical_rot`` is
        # stored xyzw (scipy convention); expose it as wxyz (MuJoCo convention,
        # what ``_coerce_quat`` expects). Skip the 3x3-matrix fallback form.
        canon = getattr(self._sim, "_canonical_rot", None) or {}
        cq = canon.get(name)
        if cq is not None:
            try:
                cq = list(cq)
                if len(cq) == 4:  # xyzw → wxyz
                    state["canonical_orientation"] = (
                        float(cq[3]),
                        float(cq[0]),
                        float(cq[1]),
                        float(cq[2]),
                    )
            except (TypeError, ValueError):
                pass
        return state

    def render(self, mode: str = "rgb_array") -> np.ndarray | None:
        """Return the current agentview image.

        Parameters
        ----------
        mode :
            Only ``"rgb_array"`` is supported.

        Returns
        -------
        np.ndarray or None
            RGB image of shape ``(H, W, 3)`` (OpenGL convention, origin
            bottom-left). Flip with ``frame[::-1]`` for standard display.
        """
        if mode != "rgb_array":
            return None
        if self._sim is None or self._sim.last_obs is None:
            return None
        return self._sim.last_obs.get("agentview_image")

    @property
    def realized_scene(self) -> Any:
        """The Scenic scene the env ACTUALLY realized on the last ``reset()``.

        This differs from the scene handed to ``make_env``/``reset`` whenever the
        settle-validation retry loop resampled a fresh scene (a preset is consumed
        only on attempt 0; settle rejections on later attempts call
        ``self._scenario.generate()`` again — see ``reset``). Consumers that score
        the realized state against Scenic intent (e.g. ``pose_tolerance`` / the G4
        consistency family) MUST compare against THIS scene, not the externally
        held preset, otherwise a retried sample is scored against the rejected
        sample's poses and reports spurious mismatches.

        Returns ``None`` before the first successful ``reset()``.
        """
        return getattr(self._sim, "scene", None) if self._sim is not None else None

    def close(self):
        """Release all resources."""
        self._cleanup_sim()
        self._cleanup_per_reset()

        # Clean up generated scenic file.
        if self._generated_scenic_path:
            pathlib.Path(self._generated_scenic_path).unlink(missing_ok=True)
            self._generated_scenic_path = None

        self._exit_stack.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compile_scenario(
        self,
        scenic_path: str | pathlib.Path | None,
    ) -> None:
        """Compile the Scenic scenario (one-time cost)."""
        import scenic

        effective_bddl = self._bddl_path

        # Handle task reversal.
        if self._reverse:
            from libero_infinity.bddl_preprocessor import patched_bddl_from_string
            from libero_infinity.task_reverser import reverse_bddl

            original_text = pathlib.Path(self._bddl_path).read_text()
            reversed_text = reverse_bddl(original_text)
            effective_bddl = self._exit_stack.enter_context(patched_bddl_from_string(reversed_text))

        self._effective_bddl = effective_bddl

        if scenic_path is not None:
            resolved_scenic = str(pathlib.Path(scenic_path).resolve())
        else:
            # Auto-generate from BDDL using the compiler pipeline.
            from libero_infinity.compiler import generate_scenic_file
            from libero_infinity.task_config import TaskConfig

            cfg = TaskConfig.from_bddl(effective_bddl)
            resolved_scenic = generate_scenic_file(cfg, self._perturbation)
            self._generated_scenic_path = resolved_scenic

        params = {"bddl_path": effective_bddl}
        params.update(self._scenic_params)

        log.info("Compiling Scenic scenario: %s", resolved_scenic)
        self._scenario = scenic.scenarioFromFile(
            resolved_scenic,
            params=params,
        )

        # Parse original BDDL for asset substitution tracking.
        from libero_infinity.bddl_preprocessor import parse_object_classes

        self._orig_obj_classes = parse_object_classes(pathlib.Path(effective_bddl).read_text())

    def _resolve_bddl_for_scene(self, scene):
        """Return a context manager yielding the effective BDDL for this scene."""
        from libero_infinity.bddl_preprocessor import bddl_for_scene

        return bddl_for_scene(scene, self._effective_bddl, self._orig_obj_classes)

    def _cleanup_sim(self) -> None:
        """Destroy the current simulation if active."""
        if self._sim is not None:
            try:
                self._sim.destroy()
            except Exception:
                log.debug("Exception during sim cleanup", exc_info=True)
            self._sim = None

    def _cleanup_per_reset(self) -> None:
        """Close per-episode resources (temp BDDL context manager)."""
        if self._per_reset_stack is not None:
            self._per_reset_stack.close()
            self._per_reset_stack = None

    def _build_obs_space(self, obs: dict) -> None:
        """Construct the observation space from a sample observation."""
        obs_dict: dict[str, spaces.Space] = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                if val.dtype == np.uint8:
                    obs_dict[key] = spaces.Box(
                        low=0,
                        high=255,
                        shape=val.shape,
                        dtype=np.uint8,
                    )
                else:
                    obs_dict[key] = spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=val.shape,
                        dtype=np.float32,
                    )
        self.observation_space = spaces.Dict(obs_dict)


# ---------------------------------------------------------------------------
# Single-condition environment factory (sweep harness contract)
# ---------------------------------------------------------------------------


def make_env(scene: Any, *, bddl_path: str) -> "LIBEROScenicEnv":
    """Construct a single-condition env matching the sweep harness contract.

    Sweep-driven counterpart to :func:`make_vec_env`: returns one
    ``LIBEROScenicEnv`` initialised with a pre-sampled Scenic ``scene`` so the
    next ``reset()`` exercises exactly the scene that the upstream G-gates
    validated, rather than resampling a fresh one. Used by the validation
    sweep at G5 (env create + reset).
    """
    return LIBEROScenicEnv(scene=scene, bddl_path=bddl_path)


# ---------------------------------------------------------------------------
# Vectorized environment factory
# ---------------------------------------------------------------------------


def make_vec_env(
    bddl_path: str | pathlib.Path,
    n_envs: int = 4,
    perturbation: str = "position",
    resolution: int = 128,
    max_steps: int = 300,
    reverse: bool = False,
    scenic_params: dict[str, Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    scenic_generate_kwargs: dict[str, Any] | None = None,
    max_scenic_iterations: int | None = None,
    use_subprocess: bool = True,
) -> gym.vector.VectorEnv:
    """Create a vectorized environment for parallel rollouts.

    Uses ``gym.vector.AsyncVectorEnv`` (subprocess-based) by default for
    true parallelism, or ``gym.vector.SyncVectorEnv`` if ``use_subprocess``
    is ``False``.

    Parameters
    ----------
    bddl_path :
        Path to the BDDL task file (shared across all envs).
    n_envs :
        Number of parallel environments.
    perturbation :
        Perturbation mode passed to each env.
    resolution :
        Camera resolution for each env.
    max_steps :
        Episode horizon for each env.
    reverse :
        Whether to reverse the task.
    scenic_params :
        Scenic parameter overrides.
    env_kwargs :
        Extra kwargs for ``OffScreenRenderEnv``.
    scenic_generate_kwargs :
        Extra kwargs for ``generate_scenic()``.
    max_scenic_iterations :
        Per-env Scenic iteration budget (see ``LIBEROScenicEnv``). ``None``
        (default) → resolved per perturbation mode from calibration.
    use_subprocess :
        If ``True`` (default), use ``AsyncVectorEnv`` for true parallelism.
        If ``False``, use ``SyncVectorEnv`` (sequential, useful for debugging).

    Returns
    -------
    gym.vector.VectorEnv
        Batched environment that accepts/returns arrays of shape ``(n_envs, ...)``.

    Example
    -------
    ::

        vec_env = make_vec_env("path/to/task.bddl", n_envs=4)
        obs = vec_env.reset()
        actions = np.zeros((4, 7))
        obs, rewards, dones, infos = vec_env.step(actions)
        vec_env.close()
    """
    bddl_path = str(pathlib.Path(bddl_path).resolve())

    def _make_env(idx: int):
        def _thunk():
            return LIBEROScenicEnv(
                bddl_path=bddl_path,
                perturbation=perturbation,
                resolution=resolution,
                max_steps=max_steps,
                seed=None,  # each env gets independent randomness
                reverse=reverse,
                scenic_params=scenic_params,
                env_kwargs=env_kwargs,
                scenic_generate_kwargs=scenic_generate_kwargs,
                max_scenic_iterations=max_scenic_iterations,
            )

        return _thunk

    env_fns = [_make_env(i) for i in range(n_envs)]

    if use_subprocess:
        return gym.vector.AsyncVectorEnv(env_fns)
    return gym.vector.SyncVectorEnv(env_fns)
