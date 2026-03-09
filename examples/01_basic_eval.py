"""Basic evaluation loop using the Python API.

Demonstrates:
  - Auto-generating a Scenic perturbation program from a BDDL task file
  - Running ``evaluate()`` with a random policy for 3 episodes
  - Printing per-episode success, Scenic scene parameters, and object positions

Run from the repo root:

    MUJOCO_GL=egl python examples/01_basic_eval.py

MuJoCo rendering backends:
  MUJOCO_GL=egl     — EGL (default on headless Linux servers)
  MUJOCO_GL=osmesa  — software renderer (fallback if EGL is unavailable)
  (unset)           — macOS / desktop Linux with a display
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Resolve paths relative to the repo root so the example works regardless of
# the current working directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(os.environ.get("LIBERO_ROOT", pathlib.Path(__file__).parent.parent))
_BDDL_PATH = (
    _REPO_ROOT
    / "src/libero_infinity/data/libero_runtime/bddl_files/libero_goal"
    / "put_the_bowl_on_the_plate.bddl"
)


# ---------------------------------------------------------------------------
# Random policy — used here purely to exercise the evaluation pipeline.
# Real policies should return a shape-(7,) action in [-1, 1].
# ---------------------------------------------------------------------------
def random_policy(obs: dict) -> np.ndarray:
    """Return a uniformly random 7-DoF action.

    Action format: [dx, dy, dz, droll, dpitch, dyaw, gripper] in [-1, 1].
    """
    return np.random.uniform(-1.0, 1.0, size=(7,)).astype(np.float64)


def main() -> None:
    # --- Graceful error if MuJoCo is missing ---
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print(
            "ERROR: mujoco is not installed.\n"
            "Install it with:  uv pip install mujoco\n"
            "Then re-run:      MUJOCO_GL=egl python examples/01_basic_eval.py"
        )
        sys.exit(1)

    # --- Verify the BDDL file exists ---
    if not _BDDL_PATH.exists():
        print(f"ERROR: BDDL file not found: {_BDDL_PATH}")
        print(
            "Make sure you are running from the libero-infinity repo root, or set "
            "LIBERO_ROOT to the repo path."
        )
        sys.exit(1)

    # --- Step 1: Auto-generate a Scenic perturbation program from the BDDL ---
    #
    # ``generate_scenic_file`` inspects the BDDL task (objects, workspace
    # geometry) and emits a Scenic 3 program that samples perturbed scenes.
    # The perturbation="position" mode randomises object (x, y) placements
    # uniformly over the reachable workspace with pairwise clearance checks.
    #
    # Other perturbation modes: "object", "camera", "lighting",
    # "distractor", "combined" (position + object), "full" (all axes).
    print("=== LIBERO-Infinity: Basic Evaluation Example ===\n")
    print(f"Task BDDL : {_BDDL_PATH.name}")
    print("Perturbation: position\n")

    from libero_infinity.compiler import generate_scenic_file
    from libero_infinity.task_config import TaskConfig

    cfg = TaskConfig.from_bddl(str(_BDDL_PATH))
    print(f"Task instruction: \"{cfg.language}\"\n")

    scenic_path = generate_scenic_file(cfg, perturbation="position")
    print(f"Generated Scenic program: {scenic_path}\n")

    # --- Step 2: Define the evaluate() call ---
    #
    # ``evaluate()`` compiles the Scenic scenario once, then loops over
    # n_scenes episodes. For each episode it:
    #   1. Samples a new scene from the Scenic program (rejection-sampling
    #      until all spatial constraints are satisfied)
    #   2. Injects the sampled object poses into MuJoCo via env.reset()
    #   3. Runs the policy for up to max_steps steps
    #   4. Records success / failure and scene metadata
    from libero_infinity.eval import evaluate

    print("Running 3 episodes with a random policy …\n")
    results = evaluate(
        scenic_path=scenic_path,
        bddl_path=str(_BDDL_PATH),
        policy=random_policy,
        n_scenes=3,          # keep short for a quick smoke-test
        max_steps=100,       # short horizon — random policy won't succeed
        verbose=False,       # we'll print our own per-episode output below
        seed=42,             # reproducible sampling
    )

    # --- Step 3: Inspect per-episode results ---
    print("Per-episode results:")
    print("-" * 60)
    for ep in results.episodes:
        status = "SUCCESS ✓" if ep.success else "FAILURE ✗"
        print(f"  Episode {ep.scene_index + 1}: {status}  ({ep.steps} steps)")

        # Scenic scene parameters contain the sampled values for every
        # distribution in the perturbation program (e.g. object positions,
        # camera offsets, lighting intensity, …).
        print(f"  Scenic scene params : {ep.scenic_params}")

        # Object positions are the MuJoCo world-frame (x, y, z) coordinates
        # that were injected for each task object.
        for obj_name, pos in ep.object_positions.items():
            print(f"    {obj_name:40s} → ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")

        print()

    # --- Step 4: Print aggregate statistics ---
    print("-" * 60)
    print(f"Aggregate: {results.summary()}")
    print(
        "\nNote: 0% success is expected with a random policy.\n"
        "      Replace random_policy() with your VLA to measure real performance."
    )

    # --- Cleanup: remove the generated Scenic file ---
    pathlib.Path(scenic_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
