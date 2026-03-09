"""Evaluation harness for LIBERO-Infinity.

Drives Scenic scene sampling → LIBERO policy evaluation → metric aggregation.

Two evaluation modes
────────────────────
1. **Standard** (default):
   Sample N scenes i.i.d. from the Scenic program, run the robot policy on
   each, report mean success rate and confidence interval.

2. **Adversarial** (--mode adversarial):
   Use VerifAI's Bayesian optimisation (or halving search) to find the scene
   distribution cell that *maximally degrades* the policy.  Requires the
   optional `verifai` dependency.

CLI usage
─────────
    libero-eval \\
      --bddl path/to/task.bddl \\
      --perturbation combined --n-scenes 200 --verbose

To evaluate a VLA policy, use the Python API::

    from libero_infinity.eval import evaluate
    results = evaluate(
        scenic_path="scenic/position_perturbation.scenic",
        bddl_path="...",
        policy=my_policy_fn,   # callable: obs_dict -> action_array (7,)
        n_scenes=100,
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from libero_infinity.grounding import (
    GroundingResult,
    GroundingTracker,
    aggregate_grounding,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Outcome of a single sampled scene × policy episode."""

    scene_index: int
    success: bool
    steps: int
    n_scenic_rejections: int  # how many Scenic samples were rejected
    scenic_params: dict  # globalParameters from the scene
    object_positions: dict[str, list]  # libero_name → [x, y, z]
    object_classes: dict[str, str]  # libero_name → chosen asset_class
    elapsed_s: float
    grounding: Optional[GroundingResult] = None  # None when no target_object_name in scene


@dataclass
class EvalResults:
    """Aggregate results from an evaluation run."""

    scenic_path: str
    bddl_path: str
    n_scenes: int
    n_success: int
    success_rate: float
    ci_95: float  # 95 % Wilson confidence interval half-width
    episodes: list[EpisodeResult] = field(default_factory=list)
    grounding_metrics: dict = field(default_factory=dict)  # from aggregate_grounding()

    def summary(self) -> str:
        base = (
            f"Success rate: {self.success_rate:.1%} ± {self.ci_95:.1%} "
            f"({self.n_success}/{self.n_scenes} scenes)"
        )
        if self.grounding_metrics:
            gr = self.grounding_metrics.get("grounding_rate")
            base += f"  |  Grounding rate: {gr:.1%}" if gr is not None else ""
        return base

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ---------------------------------------------------------------------------
# Shared helpers (used by both evaluate and evaluate_adversarial)
# ---------------------------------------------------------------------------


def _bddl_for_scene(scene, bddl_path, orig_obj_classes):
    """Delegate to shared implementation in bddl_preprocessor."""
    from libero_infinity.bddl_preprocessor import bddl_for_scene

    return bddl_for_scene(scene, bddl_path, orig_obj_classes)


def _collect_episode_result(
    scene,
    scene_index: int,
    success: bool,
    steps: int,
    n_iters: int,
    t0: float,
    grounding: Optional[GroundingResult] = None,
) -> EpisodeResult:
    """Build an EpisodeResult from a completed episode."""
    obj_positions: dict[str, list] = {}
    obj_classes: dict[str, str] = {}
    for obj in scene.objects:
        name = getattr(obj, "libero_name", None)
        if name:
            obj_positions[name] = list(map(float, obj.position))
            obj_classes[name] = getattr(obj, "asset_class", "")

    return EpisodeResult(
        scene_index=scene_index,
        success=success,
        steps=steps,
        n_scenic_rejections=n_iters - 1,
        scenic_params=dict(scene.params),
        object_positions=obj_positions,
        object_classes=obj_classes,
        elapsed_s=time.monotonic() - t0,
        grounding=grounding,
    )


def _wilson_ci_margin(n_success: int, n_total: int) -> float:
    """Compute 95% Wilson score confidence interval half-width."""
    if n_total == 0:
        return 0.0
    p = n_success / n_total
    z = 1.96
    denom = 1 + z**2 / n_total
    margin = (z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denom
    return float(margin)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate(
    scenic_path: str | pathlib.Path,
    bddl_path: str | pathlib.Path,
    policy: Callable[[dict], np.ndarray],
    n_scenes: int = 100,
    max_steps: int = 300,
    scenic_params: dict | None = None,
    env_kwargs: dict | None = None,
    verbose: bool = False,
    seed: int | None = None,
    render_live: str | None = None,
    camera: str = "agentview",
) -> EvalResults:
    """Sample N scenes from the Scenic program and evaluate `policy` on each.

    Args:
        scenic_path: Path to a .scenic perturbation program (Layer 3).
        bddl_path:   Path to the BDDL task file for this evaluation.
        policy:      Callable (obs_dict) → action_array.  The policy is
                     called once per control step.
        n_scenes:    Number of independent scenes to sample and evaluate.
        max_steps:   Maximum episode length per scene.
        scenic_params: Override globalParameters in the Scenic program.
        env_kwargs:  Extra kwargs forwarded to OffScreenRenderEnv.
        verbose:     Print per-episode summaries.
        seed:        Optional RNG seed for reproducibility.
        render_live: None (headless, default), "cv2" (stream frames to an
                     OpenCV window, needs DISPLAY), or "viewer" (interactive
                     MuJoCo passive viewer, needs DISPLAY + GLFW).
                     "cv2" keeps one window open across all episodes.
                     "viewer" opens a fresh viewer per episode.
        camera:      Camera name used for "cv2" mode (default "agentview").

    Returns:
        EvalResults with per-episode data and aggregate statistics.
    """
    import scenic
    from libero_infinity.bddl_preprocessor import parse_object_classes

    scenic_path = str(pathlib.Path(scenic_path).resolve())
    bddl_path = str(pathlib.Path(bddl_path).resolve())

    if seed is not None:
        import random

        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            pass

    params = {"bddl_path": bddl_path}
    params.update(scenic_params or {})

    log.info("Compiling Scenic scenario: %s", scenic_path)
    scenario = scenic.scenarioFromFile(scenic_path, params=params)

    from libero_infinity.simulator import LIBEROSimulator

    episodes: list[EpisodeResult] = []

    # Pre-validate optional rendering imports (once, not per-episode).
    _cv2_mod = None
    _cv2_win: str | None = None
    if render_live == "cv2":
        try:
            import cv2 as _cv2_mod
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python not installed — uv pip install opencv-python"
            ) from exc
        _cv2_win = f"LIBERO — {camera}"
        _cv2_mod.namedWindow(_cv2_win, _cv2_mod.WINDOW_NORMAL)

    _mjv = None
    if render_live == "viewer":
        try:
            import mujoco.viewer as _mjv
        except ImportError as exc:
            raise RuntimeError("mujoco.viewer not available — install mujoco >= 2.3.3") from exc

    _camera_key = f"{camera}_image"

    # Parse original BDDL to know the canonical object classes (for substitution)
    _orig_bddl_text = pathlib.Path(bddl_path).read_text()
    _orig_obj_classes = parse_object_classes(_orig_bddl_text)

    try:
        for i in range(n_scenes):
            t0 = time.monotonic()

            scene, n_iters = scenario.generate(maxIterations=5000, verbosity=0)

            with _bddl_for_scene(scene, bddl_path, _orig_obj_classes) as effective_bddl:
                simulator = LIBEROSimulator(
                    bddl_path=effective_bddl,
                    env_kwargs=env_kwargs,
                )
                sim = simulator.createSimulation(
                    scene,
                    maxSteps=max_steps,
                    timestep=0.05,
                    verbosity=0,
                )
                try:
                    sim.setup()

                    obs = sim.last_obs
                    success = False
                    steps = 0

                    # --- Grounding tracker (Section 1 of eval_plan.md) ---
                    _target_name: str = scene.params.get("target_object_name", "")
                    _tracker = GroundingTracker(
                        target_object_name=_target_name,
                        episode_id=f"ep_{i:04d}",
                    )

                    if obs is not None:
                        if _mjv is not None:
                            mjmodel, mjdata = sim.mj_handles
                            with _mjv.launch_passive(mjmodel, mjdata) as _viewer:
                                for step_idx in range(max_steps):
                                    if not _viewer.is_running():
                                        break
                                    action = policy(obs)
                                    obs, _reward, done, _info = sim.step_with_action(
                                        np.asarray(action, dtype=float)
                                    )
                                    _tracker.step(sim.libero_env.env.sim)
                                    steps = step_idx + 1
                                    with _viewer.lock():
                                        _viewer.sync()
                                    if done:
                                        success = sim.check_success()
                                        break
                                else:
                                    success = sim.check_success()
                        else:
                            for step_idx in range(max_steps):
                                action = policy(obs)
                                obs, _reward, done, _info = sim.step_with_action(
                                    np.asarray(action, dtype=float)
                                )
                                _tracker.step(sim.libero_env.env.sim)
                                steps = step_idx + 1
                                if _cv2_win is not None:
                                    frame = obs.get(_camera_key)
                                    if frame is not None:
                                        _cv2_mod.imshow(_cv2_win, frame[::-1, :, ::-1])
                                    if _cv2_mod.waitKey(1) & 0xFF == ord("q"):
                                        done = True
                                if done:
                                    success = sim.check_success()
                                    break
                            else:
                                success = sim.check_success()

                    _grounding = _tracker.result(task_success=success)
                    ep = _collect_episode_result(
                        scene,
                        i,
                        success,
                        steps,
                        n_iters,
                        t0,
                        grounding=_grounding if _target_name else None,
                    )
                    episodes.append(ep)

                    if verbose:
                        status = "✓" if success else "✗"
                        print(
                            f"[{i + 1:4d}/{n_scenes}] {status} "
                            f"steps={steps:3d}  iters={n_iters:3d}  "
                            f"t={ep.elapsed_s:.1f}s  "
                            f"positions={ep.object_positions}"
                        )
                finally:
                    sim.destroy()

    finally:
        if _cv2_win is not None:
            _cv2_mod.destroyWindow(_cv2_win)

    # Aggregate
    n_success = sum(e.success for e in episodes)
    p = n_success / n_scenes if n_scenes > 0 else 0.0
    ci_95 = _wilson_ci_margin(n_success, n_scenes)

    # Aggregate grounding metrics (only for episodes where tracking was active)
    _grounding_results = [e.grounding for e in episodes if e.grounding is not None]
    _grounding_metrics = aggregate_grounding(_grounding_results)
    if _grounding_metrics:
        log.info(
            "Grounding: rate=%.1f%%  biased=%.1f%%  no_contact=%.1f%%",
            _grounding_metrics["grounding_rate"] * 100,
            _grounding_metrics["biased_rate"] * 100,
            _grounding_metrics["no_contact_rate"] * 100,
        )

    results = EvalResults(
        scenic_path=scenic_path,
        bddl_path=bddl_path,
        n_scenes=n_scenes,
        n_success=n_success,
        success_rate=float(p),
        ci_95=ci_95,
        episodes=episodes,
        grounding_metrics=_grounding_metrics,
    )

    log.info(results.summary())
    return results


# ---------------------------------------------------------------------------
# Adversarial search via VerifAI
# ---------------------------------------------------------------------------


def evaluate_adversarial(
    scenic_path: str | pathlib.Path,
    bddl_path: str | pathlib.Path,
    policy: Callable[[dict], np.ndarray],
    n_samples: int = 200,
    max_steps: int = 300,
    scenic_params: dict | None = None,
    env_kwargs: dict | None = None,
    verbose: bool = False,
) -> EvalResults:
    """Find the worst-case scene using cross-entropy adversarial search.

    Uses Scenic's feedback-driven sampling: after each episode, pass
    rho = 0.0 (success) or 1.0 (failure) back to the scenario. If the
    Scenic program uses VerifaiRange instead of Range, the cross-entropy
    sampler concentrates on failure-inducing regions over iterations.

    Even without VerifaiRange, this function works as a standard evaluation
    loop that records worst-case episodes.

    Args:
        scenic_path: Path to .scenic perturbation program.
        bddl_path:   Path to BDDL task file.
        policy:      Callable (obs_dict) -> action_array.
        n_samples:   Number of adversarial search iterations.
        max_steps:   Max episode steps.
        scenic_params: Override globalParameters.
        env_kwargs:  Extra kwargs for OffScreenRenderEnv.
        verbose:     Print per-episode summaries.

    Returns:
        EvalResults sorted by failure (worst-case first).
    """
    import scenic
    from libero_infinity.bddl_preprocessor import parse_object_classes

    scenic_path = str(pathlib.Path(scenic_path).resolve())
    bddl_path = str(pathlib.Path(bddl_path).resolve())

    params = {"bddl_path": bddl_path}
    params.update(scenic_params or {})

    log.info("Compiling Scenic scenario for adversarial search: %s", scenic_path)
    scenario = scenic.scenarioFromFile(scenic_path, params=params)

    from libero_infinity.simulator import LIBEROSimulator

    _orig_bddl_text = pathlib.Path(bddl_path).read_text()
    _orig_obj_classes = parse_object_classes(_orig_bddl_text)

    episodes: list[EpisodeResult] = []
    last_feedback = None  # None for first sample (no feedback yet)

    for i in range(n_samples):
        t0 = time.monotonic()

        gen_kwargs = dict(maxIterations=5000, verbosity=0)
        if last_feedback is not None:
            gen_kwargs["feedback"] = last_feedback
        scene, n_iters = scenario.generate(**gen_kwargs)

        with _bddl_for_scene(scene, bddl_path, _orig_obj_classes) as effective_bddl:
            simulator = LIBEROSimulator(
                bddl_path=effective_bddl,
                env_kwargs=env_kwargs,
            )
            sim = simulator.createSimulation(
                scene,
                maxSteps=max_steps,
                timestep=0.05,
                verbosity=0,
            )
            try:
                sim.setup()

                obs = sim.last_obs
                success = False
                steps = 0

                # --- Grounding tracker ---
                _target_name_adv: str = scene.params.get("target_object_name", "")
                _tracker_adv = GroundingTracker(
                    target_object_name=_target_name_adv,
                    episode_id=f"adv_{i:04d}",
                )

                if obs is not None:
                    for step_idx in range(max_steps):
                        action = policy(obs)
                        obs, _, done, _ = sim.step_with_action(np.asarray(action, dtype=float))
                        _tracker_adv.step(sim.libero_env.env.sim)
                        steps = step_idx + 1
                        if done:
                            success = sim.check_success()
                            break
                    else:
                        success = sim.check_success()

                last_feedback = 0.0 if success else 1.0

                _grounding_adv = _tracker_adv.result(task_success=success)
                ep = _collect_episode_result(
                    scene,
                    i,
                    success,
                    steps,
                    n_iters,
                    t0,
                    grounding=_grounding_adv if _target_name_adv else None,
                )
                episodes.append(ep)

                if verbose:
                    status = "✓" if success else "✗"
                    print(
                        f"[ADV {i + 1:4d}/{n_samples}] {status} "
                        f"steps={steps:3d}  rho={last_feedback:.1f}  "
                        f"t={ep.elapsed_s:.1f}s"
                    )
            finally:
                sim.destroy()

    # Sort: failures first (adversarial worst-case at the top)
    episodes.sort(key=lambda e: (e.success, e.steps))

    n_success = sum(e.success for e in episodes)
    p = n_success / n_samples if n_samples > 0 else 0.0
    ci_95 = _wilson_ci_margin(n_success, n_samples)

    _adv_grounding_results = [e.grounding for e in episodes if e.grounding is not None]
    _adv_grounding_metrics = aggregate_grounding(_adv_grounding_results)

    results = EvalResults(
        scenic_path=scenic_path,
        bddl_path=bddl_path,
        n_scenes=n_samples,
        n_success=n_success,
        success_rate=float(p),
        ci_95=ci_95,
        episodes=episodes,
        grounding_metrics=_adv_grounding_metrics,
    )

    log.info("Adversarial search: %s", results.summary())
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for ``libero-eval``."""
    p = argparse.ArgumentParser(
        prog="libero-eval",
        description="LIBERO-Infinity evaluation harness",
    )
    p.add_argument(
        "--scenic",
        default=None,
        help="Path to .scenic program (auto-generated from BDDL if omitted)",
    )
    p.add_argument("--bddl", required=True, help="Path to BDDL task file")
    p.add_argument(
        "--perturbation",
        default="position",
        help=(
            "Perturbation axes. Accepts a single axis (position, object, "
            "camera, lighting, distractor), a preset (combined, full), or "
            "a comma-separated list (e.g. position,camera,distractor). "
            "Default: position"
        ),
    )
    p.add_argument(
        "--max-distractors",
        type=int,
        default=5,
        help="Max distractor objects for distractor mode (default: 5)",
    )
    p.add_argument(
        "--min-distractors",
        type=int,
        default=1,
        help="Min distractor objects for distractor mode (default: 1)",
    )
    p.add_argument("--n-scenes", type=int, default=100, help="Number of sampled scenes")
    p.add_argument("--max-steps", type=int, default=300, help="Episode horizon")
    p.add_argument("--mode", choices=["standard", "adversarial"], default="standard")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None, help="JSON output path")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--watch",
        choices=["cv2", "viewer"],
        default=None,
        metavar="{cv2,viewer}",
        help=(
            "Enable live rendering. 'cv2' streams frames to an OpenCV window "
            "(needs DISPLAY). 'viewer' opens MuJoCo's interactive passive "
            "viewer per episode (needs DISPLAY + GLFW)."
        ),
    )
    p.add_argument(
        "--camera",
        default="agentview",
        help="Camera to display with --watch cv2 (default: agentview)",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Camera image resolution in pixels (default: 512)",
    )
    p.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the task (object starts at goal, must return to original position)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for ``libero-eval``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _build_parser().parse_args(argv)  # allows --help to work

    raise SystemExit(
        "error: libero-eval requires a running robo-eval server.\n"
        "\n"
        "The built-in zero-action policy has been removed.\n"
        "\n"
        "  Connect a VLA via the robo-eval server:\n"
        "    bash examples/05_robo_eval_cli.sh\n"
        "\n"
        "  Or use the Python API with your own policy callable:\n"
        "    see examples/04_custom_vla.py\n"
        "    from libero_infinity.eval import evaluate"
    )


if __name__ == "__main__":
    main()
