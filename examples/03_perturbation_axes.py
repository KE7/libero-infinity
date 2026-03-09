"""Perturbation axes survey — demonstrates all 6 axes.

For each perturbation axis, the environment is reset twice and the sampled
Scenic scene parameters are printed.  This shows concretely what each axis
randomises on every call to reset().

Perturbation axes:
  position    — object (x, y) placement over the reachable workspace
  object      — visual identity (mesh + texture) from asset variant pools
  camera      — agentview position offset and tilt angle
  lighting    — scene illumination (intensity, ambient level)
  distractor  — clutter objects placed near task objects
  combined    — preset: position + object identity (most common for eval)

Run from the repo root:

    MUJOCO_GL=egl python examples/03_perturbation_axes.py

MuJoCo rendering backends:
  MUJOCO_GL=egl     — EGL (default on headless Linux servers)
  MUJOCO_GL=osmesa  — software renderer (fallback if EGL is unavailable)
  (unset)           — macOS / desktop Linux with a display
"""

from __future__ import annotations

import os
import pathlib
import sys


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(os.environ.get("LIBERO_ROOT", pathlib.Path(__file__).parent.parent))

# Use a task with several task objects so position/object perturbations show
# clear diversity across resets.
_BDDL_PATH = (
    _REPO_ROOT
    / "src/libero_infinity/data/libero_runtime/bddl_files/libero_goal"
    / "put_the_bowl_on_the_plate.bddl"
)

# All single-axis perturbation modes, plus the "combined" preset.
# "full" (all axes simultaneously) is also valid but takes longer per reset.
PERTURBATION_AXES = [
    "position",
    "object",
    "camera",
    "lighting",
    "distractor",
    "combined",
]


def _check_deps() -> None:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print(
            "ERROR: mujoco is not installed.\n"
            "Install it with:  uv pip install mujoco\n"
            "Then re-run:      MUJOCO_GL=egl python examples/03_perturbation_axes.py"
        )
        sys.exit(1)

    if not _BDDL_PATH.exists():
        print(f"ERROR: BDDL file not found: {_BDDL_PATH}")
        print("Run from the libero-infinity repo root or set LIBERO_ROOT.")
        sys.exit(1)


def demo_axis(axis: str) -> None:
    """Create an env for ``axis``, reset twice, print Scenic params."""
    print(f"\n{'='*60}")
    print(f"  Perturbation axis: {axis!r}")
    print(f"{'='*60}")

    # What does this axis randomise?
    _DESCRIPTIONS = {
        "position":   "Object (x, y) placement — uniform over reachable workspace",
        "object":     "Visual identity (mesh + texture) from 34 asset variant pools",
        "camera":     "Agentview camera position offset (±0.10 m) and tilt (±15°)",
        "lighting":   "Illumination — intensity [0.4, 2.0], ambient [0.05, 0.6]",
        "distractor": "1–5 clutter objects placed near task objects",
        "combined":   "Preset: position + object identity (most common for evaluation)",
    }
    print(f"  What varies: {_DESCRIPTIONS.get(axis, '(see docs/scenic_perturbations.md)')}\n")

    from libero_infinity.gym_env import LIBEROScenicEnv

    # The constructor compiles the Scenic scenario once for this axis.
    # Subsequent reset() calls reuse the compiled scenario.
    env = LIBEROScenicEnv(
        bddl_path=str(_BDDL_PATH),
        perturbation=axis,
        resolution=64,    # small resolution — we only care about params, not images
        max_steps=10,
    )

    for trial in range(1, 3):   # two resets to show diversity
        obs = env.reset()

        # The Scenic scene parameters live on the underlying simulation object.
        # They reflect the sampled values for every distribution in the
        # perturbation program (e.g. object positions, camera offsets, …).
        scene_params = {}
        if env._sim is not None and hasattr(env._sim, "_scene"):
            scene_params = dict(env._sim._scene.params)

        # Object positions are also informative for position/combined axes.
        obj_positions: dict[str, list] = {}
        if env._sim is not None and hasattr(env._sim, "_scene"):
            for obj in env._sim._scene.objects:
                name = getattr(obj, "libero_name", None)
                if name:
                    obj_positions[name] = [round(float(c), 4) for c in obj.position]

        # Object asset classes (informative for object/combined axes).
        obj_classes: dict[str, str] = {}
        if env._sim is not None and hasattr(env._sim, "_scene"):
            for obj in env._sim._scene.objects:
                name = getattr(obj, "libero_name", None)
                cls = getattr(obj, "asset_class", None)
                if name and cls:
                    obj_classes[name] = cls

        print(f"  Reset #{trial}:")
        if scene_params:
            print(f"    Scenic params  : {scene_params}")
        if obj_positions:
            for name, pos in obj_positions.items():
                print(f"    {name:40s} pos={pos}")
        if obj_classes:
            for name, cls in obj_classes.items():
                print(f"    {name:40s} class={cls!r}")
        if not scene_params and not obj_positions and not obj_classes:
            # Fallback: show obs shape so the user can confirm things ran
            print(f"    obs keys: {list(obs.keys())[:5]} …")

    env.close()
    print(f"\n  [axis={axis!r} done — env closed]")


def main() -> None:
    _check_deps()

    print("=== LIBERO-Infinity: Perturbation Axes Survey ===\n")
    print(f"Task BDDL : {_BDDL_PATH.name}")
    print(
        "For each of the 6 perturbation axes, the environment is reset twice\n"
        "and the sampled Scenic scene parameters are printed.\n"
    )

    for axis in PERTURBATION_AXES:
        demo_axis(axis)

    print("\n" + "=" * 60)
    print("Summary of perturbation axes")
    print("=" * 60)
    print(
        "  position   — continuous uniform XY over the table workspace\n"
        "  object     — swap object meshes + textures from asset pools\n"
        "  camera     — shift agentview camera position and viewing angle\n"
        "  lighting   — vary scene illumination (intensity + ambient)\n"
        "  distractor — add 1–5 random clutter objects to the scene\n"
        "  combined   — position + object (preset for standard evaluation)\n"
        "\nAll axes are composable: e.g. perturbation='position,camera,lighting'\n"
        "See docs/scenic_perturbations.md for full parameter details."
    )


if __name__ == "__main__":
    main()
