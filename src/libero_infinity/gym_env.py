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
    ):
        super().__init__()

        self._bddl_path = str(pathlib.Path(bddl_path).resolve())
        self._perturbation = perturbation
        self._resolution = resolution
        self._max_steps = max_steps
        self._seed = seed
        self._reverse = reverse
        self._scenic_params = scenic_params or {}
        self._env_kwargs = env_kwargs or {}
        self._scenic_generate_kwargs = scenic_generate_kwargs or {}

        # Managed resources
        self._exit_stack = contextlib.ExitStack()  # for long-lived resources (reversed BDDL)
        self._scenario = None
        self._sim: Any = None  # LIBEROSimulation
        self._generated_scenic_path: str | None = None
        self._per_reset_stack: contextlib.ExitStack | None = None  # per-episode resources

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
            # Use 5000 iterations: radial footprint-clearance constraints
            # (task objects vs fixtures) are tighter than the old AABB form
            # and may need more rejection-sampling attempts.
            scene, _n_iters = self._scenario.generate(
                maxIterations=5000,
                verbosity=0,
            )

            # Resolve BDDL substitutions (asset swaps) via proper context manager.
            self._per_reset_stack = contextlib.ExitStack()
            effective_bddl = self._per_reset_stack.enter_context(
                self._resolve_bddl_for_scene(scene)
            )

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
            )

        return _thunk

    env_fns = [_make_env(i) for i in range(n_envs)]

    if use_subprocess:
        return gym.vector.AsyncVectorEnv(env_fns)
    return gym.vector.SyncVectorEnv(env_fns)
