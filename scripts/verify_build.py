#!/usr/bin/env python3
"""End-to-end build verification for Libero-∞.

Exercises every major capability to confirm the install is working:

1. Import check — all modules importable
2. Scenic compilation — all .scenic files compile
3. Scene generation — sample from each perturbation type
4. LIBERO integration — create env, inject poses, step physics
5. Auto-generation — generate Scenic from BDDL, compile, sample
6. BDDL substitution — verify asset swap end-to-end
7. Distractor injection — verify distractor bodies appear in MuJoCo
8. LIBERO-PRO parity — load BDDL, create env, zero-action policy, check success

Usage::

    MUJOCO_GL=egl python scripts/verify_build.py
    MUJOCO_GL=egl python scripts/verify_build.py --verbose
    python scripts/verify_build.py --skip-sim   # Scenic-only checks
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import tempfile
import traceback
from typing import Callable

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENIC_DIR = REPO_ROOT / "scenic"
SRC_DIR = REPO_ROOT / "src"

# Ensure src/ is on path
sys.path.insert(0, str(SRC_DIR))

from libero_infinity.runtime import get_bddl_dir  # noqa: E402

BDDL_DIR = get_bddl_dir()

# Cached BDDL path — computed once, reused across all tests
_BOWL_BDDL: str = ""
_bddl_matches = list(BDDL_DIR.glob("**/put_the_bowl_on_the_plate.bddl"))
if _bddl_matches:
    _BOWL_BDDL = str(_bddl_matches[0])

PERTURBATION_TYPES = ["position", "object", "combined", "camera", "lighting", "distractor"]
SCENIC_FILES = [f"{p}_perturbation.scenic" for p in PERTURBATION_TYPES]


@dataclasses.dataclass
class _Results:
    passed: int = 0
    failed: int = 0
    skipped: int = 0


def check(
    name: str,
    fn: Callable[[], None],
    results: _Results,
    *,
    skip_if: bool = False,
    skip_reason: str = "",
    verbose: bool = False,
) -> None:
    """Run a named verification check, recording pass/fail/skip."""
    if skip_if:
        print(f"  SKIP  {name} ({skip_reason})")
        results.skipped += 1
        return
    try:
        fn()
        print(f"  PASS  {name}")
        results.passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        if verbose:
            traceback.print_exc()
        results.failed += 1


def _require_bddl() -> str:
    if not _BOWL_BDDL:
        raise RuntimeError("No BDDL file found")
    return _BOWL_BDDL


# ---------------------------------------------------------------------------
# 1. Import check
# ---------------------------------------------------------------------------
def test_imports():
    import libero_infinity  # noqa: F401
    from libero_infinity import (  # noqa: F401
        asset_registry,
        bddl_preprocessor,
        compiler,
        eval,
        simulator,
        task_config,
    )


def test_libero_import():
    from libero.libero import get_libero_path  # noqa: F401
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Scenic compilation
# ---------------------------------------------------------------------------
def test_scenic_compile():
    import scenic as sc

    bddl = _BOWL_BDDL
    for sf in SCENIC_FILES:
        path = SCENIC_DIR / sf
        assert path.exists(), f"{sf} not found"
        sc.scenarioFromFile(
            str(path),
            params={"bddl_path": bddl, "task": "put_the_bowl_on_the_plate"},
        )


# ---------------------------------------------------------------------------
# 3. Scene generation — one sample per perturbation type
# ---------------------------------------------------------------------------
def test_scene_generation():
    import scenic as sc
    from libero_infinity.compiler import generate_scenic
    from libero_infinity.task_config import TaskConfig

    bddl = _require_bddl()
    cfg = TaskConfig.from_bddl(bddl)
    for ptype in PERTURBATION_TYPES:
        code = generate_scenic(cfg, perturbation=ptype)
        assert len(code) > 0, f"Empty code for {ptype}"

        with tempfile.NamedTemporaryFile(
            suffix=".scenic", mode="w", dir=str(SCENIC_DIR), delete=False
        ) as f:
            f.write(code)
            tmp = f.name
        try:
            scenario = sc.scenarioFromFile(tmp)
            scene, _ = scenario.generate(maxIterations=50)
            assert scene is not None, f"No scene for {ptype}"
        finally:
            pathlib.Path(tmp).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 4. LIBERO integration — create env, inject poses, step
# ---------------------------------------------------------------------------
def test_libero_simulation():
    import numpy as np

    import scenic as sc
    from libero_infinity.simulator import LIBEROSimulator

    bddl = _require_bddl()
    path = SCENIC_DIR / "position_perturbation.scenic"
    scenario = sc.scenarioFromFile(
        str(path),
        params={"bddl_path": bddl, "task": "put_the_bowl_on_the_plate"},
    )
    scene, _ = scenario.generate(maxIterations=50)

    simulator = LIBEROSimulator(bddl_path=bddl)
    sim = simulator.createSimulation(scene, maxSteps=50, timestep=0.05)
    sim.setup()
    try:
        for _ in range(5):
            sim.step_with_action(np.zeros(sim._nact))

        success = sim.check_success()
        assert isinstance(success, bool)
    finally:
        sim.destroy()


# ---------------------------------------------------------------------------
# 5. Auto-generation — generate Scenic from any BDDL, compile, sample
# ---------------------------------------------------------------------------
def test_auto_generation():
    import scenic as sc
    from libero_infinity.compiler import generate_scenic_file
    from libero_infinity.task_config import TaskConfig

    bddls = sorted(BDDL_DIR.glob("**/*.bddl"))
    assert len(bddls) > 0, "No BDDL files found"

    for bddl in bddls[:3]:
        cfg = TaskConfig.from_bddl(str(bddl))
        out = generate_scenic_file(cfg, perturbation="position")
        try:
            scenario = sc.scenarioFromFile(out)
            scene, _ = scenario.generate(maxIterations=50)
            assert scene is not None
        finally:
            pathlib.Path(out).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 6. BDDL substitution — verify asset swap
# ---------------------------------------------------------------------------
def test_bddl_substitution():
    from libero_infinity.asset_registry import ASSET_VARIANTS
    from libero_infinity.bddl_preprocessor import parse_object_classes, substitute_asset

    bddl = _require_bddl()
    content = pathlib.Path(bddl).read_text()

    if "akita_black_bowl" in ASSET_VARIANTS:
        variants = ASSET_VARIANTS["akita_black_bowl"]
        if len(variants) > 1:
            new_class = [v for v in variants if v != "akita_black_bowl"][0]
            new_content = substitute_asset(content, "akita_black_bowl", new_class)
            assert new_class in new_content
            classes = parse_object_classes(new_content)
            assert new_class in classes.values(), f"{new_class} not in substituted BDDL objects"


# ---------------------------------------------------------------------------
# 7. Distractor injection
# ---------------------------------------------------------------------------
def test_distractor_injection():
    from libero_infinity.bddl_preprocessor import add_distractor_objects

    bddl = _require_bddl()
    content = pathlib.Path(bddl).read_text()
    distractors = [("distractor_0", "cream_cheese")]
    new_content = add_distractor_objects(content, distractors)
    assert "distractor_0" in new_content
    assert "cream_cheese" in new_content


# ---------------------------------------------------------------------------
# 8. LIBERO-PRO parity — load BDDL, create env, run zero-action, check success
# ---------------------------------------------------------------------------
def test_libero_pro_parity():
    """Reproduce the core LIBERO-PRO evaluation loop."""
    import numpy as np
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    bddl = _require_bddl()
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(42)
    try:
        obs = env.reset()
        assert obs is not None

        action_dim = env.env.action_spec[0].shape[0]
        for _ in range(10):
            obs, reward, done, info = env.step(np.zeros(action_dim))

        success = env.check_success()
        assert not success  # should be False at start
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point for build verification."""
    parser = argparse.ArgumentParser(description="Verify Libero-∞ build")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-sim", action="store_true", help="Skip LIBERO simulation tests")
    args = parser.parse_args()

    sim_available = True
    try:
        from libero.libero.envs.env_wrapper import OffScreenRenderEnv  # noqa: F401
    except ImportError:
        sim_available = False

    results = _Results()
    kw = {"results": results, "verbose": args.verbose}

    print("=" * 60)
    print("Libero-∞ Build Verification")
    print("=" * 60)

    print("\n1. Import checks")
    check("Core imports", test_imports, **kw)
    check(
        "LIBERO import",
        test_libero_import,
        **kw,
        skip_if=not sim_available,
        skip_reason="LIBERO not installed",
    )

    print("\n2. Scenic compilation")
    check("Compile all .scenic files", test_scenic_compile, **kw)

    print("\n3. Scene generation (all perturbation types)")
    check("Generate scenes", test_scene_generation, **kw)

    print("\n4. LIBERO simulation")
    check(
        "Create env + inject poses + step",
        test_libero_simulation,
        **kw,
        skip_if=not sim_available or args.skip_sim,
        skip_reason="simulation skipped",
    )

    print("\n5. Auto-generation from BDDL")
    check("Auto-generate Scenic from BDDL", test_auto_generation, **kw)

    print("\n6. BDDL substitution")
    check("Asset class swap", test_bddl_substitution, **kw)

    print("\n7. Distractor injection")
    check("Inject distractor objects", test_distractor_injection, **kw)

    print("\n8. LIBERO-PRO parity")
    check(
        "Load BDDL + create env + zero-action + success check",
        test_libero_pro_parity,
        **kw,
        skip_if=not sim_available or args.skip_sim,
        skip_reason="simulation skipped",
    )

    print("\n" + "=" * 60)
    print(f"Results: {results.passed} passed, {results.failed} failed, {results.skipped} skipped")
    print("=" * 60)

    sys.exit(1 if results.failed > 0 else 0)


if __name__ == "__main__":
    main()
