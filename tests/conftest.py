"""Shared fixtures and helpers for Libero-∞ tests."""

import pathlib
import sys

import numpy as np
import pytest

# Exclude the backward-compat shim from default collection to avoid running
# every test twice (test_e2e.py re-exports from the split test files).
collect_ignore = ["test_e2e.py"]

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent
SCENIC_DIR = REPO_ROOT / "scenic"

# Add src/ to path so libero_infinity is importable without installing
sys.path.insert(0, str(REPO_ROOT / "src"))

from libero_infinity.runtime import get_bddl_dir  # noqa: E402

BDDL_DIR = get_bddl_dir()


# ── BDDL helpers ─────────────────────────────────────────────────────────────


def _find_bddl(glob_pattern: str) -> pathlib.Path | None:
    """Return first BDDL file matching glob, or None."""
    matches = list(BDDL_DIR.glob(glob_pattern))
    return matches[0] if matches else None


BOWL_BDDL = _find_bddl("**/put_the_bowl_on_the_plate.bddl")
OPEN_DRAWER_BDDL = _find_bddl("**/open_the_middle_drawer_of_the_cabinet.bddl")
DRAWER_BOWL_BDDL = _find_bddl("**/open_the_top_drawer_and_put_the_bowl_inside.bddl")
STOVE_BDDL = _find_bddl("**/turn_on_the_stove.bddl")
MICROWAVE_BDDL = _find_bddl(
    "**/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl"
)
OPEN_MICROWAVE_BDDL = _find_bddl("**/KITCHEN_SCENE7_open_the_microwave.bddl")
DRAWER_PICK_BOWL_BDDL = _find_bddl(
    "**/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate.bddl"
)
FLOOR_BASKET_BDDL = _find_bddl("**/*pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl")
LIVING_BASKET_BDDL = _find_bddl("**/*pick_up_the_butter_and_put_it_in_the_basket.bddl")
STUDY_SHELF_BDDL = _find_bddl(
    "**/*pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.bddl"
)


# ── LIBERO availability ─────────────────────────────────────────────────────

LIBERO_AVAILABLE = False
try:
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv  # noqa: F401

    LIBERO_AVAILABLE = True
except ImportError:
    pass

BDDL_AVAILABLE = BOWL_BDDL is not None and BOWL_BDDL.exists()

requires_libero = pytest.mark.skipif(
    not (LIBERO_AVAILABLE and BDDL_AVAILABLE),
    reason="LIBERO or BDDL files not installed",
)


# ── shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def bowl_config():
    """TaskConfig for the bowl-on-plate task (session-scoped)."""
    from libero_infinity.task_config import TaskConfig

    return TaskConfig.from_bddl(BOWL_BDDL)


# ── shared assertion helpers ─────────────────────────────────────────────────


def assert_pairwise_clearance(scene, min_clearance: float, xy_only: bool = False):
    """Assert that all objects in scene satisfy minimum clearance."""
    objs = [o for o in scene.objects if getattr(o, "libero_name", "")]
    for i, a in enumerate(objs):
        for b in objs[i + 1 :]:
            if xy_only:
                pa = np.array([a.position.x, a.position.y])
                pb = np.array([b.position.x, b.position.y])
            else:
                pa = np.array([a.position.x, a.position.y, a.position.z])
                pb = np.array([b.position.x, b.position.y, b.position.z])
            dist = np.linalg.norm(pa - pb)
            assert dist >= min_clearance - 0.001, (
                f"Clearance violation: {a.libero_name} ↔ {b.libero_name} "
                f"dist={dist:.3f} < {min_clearance}"
            )


def assert_pairwise_axis_clearance(
    scene,
    dims_by_name: dict[str, tuple[float, float, float]],
    margin: float = 0.0,
):
    """Assert that all object pairs are separated by axis-aligned footprints."""
    objs = [
        o
        for o in scene.objects
        if getattr(o, "libero_name", "") and getattr(o, "libero_name", "") in dims_by_name
    ]
    for i, a in enumerate(objs):
        for b in objs[i + 1 :]:
            dx = abs(float(a.position.x - b.position.x))
            dy = abs(float(a.position.y - b.position.y))
            min_dx = (
                dims_by_name[a.libero_name][0] + dims_by_name[b.libero_name][0]
            ) / 2.0 + margin
            min_dy = (
                dims_by_name[a.libero_name][1] + dims_by_name[b.libero_name][1]
            ) / 2.0 + margin
            assert dx >= min_dx - 0.001 or dy >= min_dy - 0.001, (
                f"Axis clearance violation: {a.libero_name} ↔ {b.libero_name} "
                f"dx={dx:.3f} < {min_dx:.3f} and dy={dy:.3f} < {min_dy:.3f}"
            )
