#!/usr/bin/env python3
"""Render and cache canonical agentview scene images for all LIBERO tasks.

For each BDDL task file, this script:
  1. Initialises the LIBERO OffScreenRenderEnv (no Scenic, no perturbations)
  2. Resets to the canonical initial state
  3. Captures the agentview RGB frame
  4. Saves it to  assets/scene_images/<suite>/<task_stem>.png

The per-scene images are used by rate_cf_bddls_llm.py to ground LLM scoring in
the actual scene layout rather than a single generic placeholder image.

Usage
-----
    # Render all suites (resumes automatically if images already exist):
    python scripts/render_scene_images.py

    # Render a single suite:
    python scripts/render_scene_images.py --suite libero_spatial

    # Render a single BDDL file:
    python scripts/render_scene_images.py --bddl path/to/task.bddl

    # Use a higher resolution (default: 512):
    python scripts/render_scene_images.py --resolution 512

Environment
-----------
    PKG_CONFIG_PATH=/home/batman/.micromamba/envs/libero_libs/lib/pkgconfig

Requires:  mujoco, libero (via vendor/libero symlink), Pillow
If mujoco is not available the script prints a diagnostic and exits with
status 2 so callers can detect the missing-dependency case cleanly.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# ── repository root paths ──────────────────────────────────────────────────
_REPO_ROOT  = pathlib.Path(__file__).resolve().parent.parent
_ASSETS_DIR = _REPO_ROOT / "assets" / "scene_images"
_BDDL_ROOT  = _REPO_ROOT / "src" / "libero_infinity" / "data" / "libero_runtime" / "bddl_files"

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]

sys.path.insert(0, str(_REPO_ROOT / "src"))


# ── dependency checks ──────────────────────────────────────────────────────

def _check_deps() -> tuple[bool, str]:
    """Return (ok, message).  ok=False means rendering is impossible."""
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False, (
            "mujoco is not importable from this Python interpreter.\n"
            "Rendering requires the mujoco Python package.\n"
            "Hint: activate the correct conda env or run with the libero_libs "
            "interpreter, e.g.:\n"
            "    /path/to/libero_env/bin/python scripts/render_scene_images.py\n"
            "The assets/scene_images/ directory structure has been created so "
            "the scorer infrastructure works; populate it by running this script "
            "from an environment where 'import mujoco' succeeds."
        )
    try:
        sys.path.insert(0, str(_REPO_ROOT / "vendor" / "libero"))
        from libero.libero.envs.env_wrapper import OffScreenRenderEnv  # noqa: F401
    except ImportError as exc:
        return False, (
            f"Could not import libero: {exc}\n"
            "Make sure the vendor/libero symlink is in place and libero's "
            "dependencies are installed."
        )
    try:
        from libero_infinity.runtime import ensure_runtime
        ensure_runtime()
    except Exception as exc:
        return False, f"LIBERO runtime bootstrap failed: {exc}"
    return True, "ok"


# ── rendering ──────────────────────────────────────────────────────────────

def render_one(bddl_path: pathlib.Path, out_path: pathlib.Path, resolution: int) -> None:
    """Render a single task and save the agentview PNG."""
    import numpy as np
    from PIL import Image
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        camera_names=["agentview"],
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=20,
        horizon=1,           # only need one step for a frame
        ignore_done=True,
        hard_reset=True,
    )
    try:
        obs = env.reset()
        # agentview_image is (H, W, 3) uint8 in OpenGL convention (bottom-left origin).
        frame: np.ndarray | None = obs.get("agentview_image")
        if frame is None:
            raise RuntimeError("'agentview_image' not in observation dict")

        # Flip vertically (OpenGL → standard image convention: top-left origin).
        frame = frame[::-1].copy()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame, mode="RGB").save(out_path, format="PNG")
    finally:
        try:
            env.close()
        except Exception:
            pass


def render_suite(
    suite_name: str,
    resolution: int,
    *,
    skip_existing: bool = True,
    verbose: bool = True,
) -> tuple[int, int, int]:
    """Render all tasks in a suite.  Returns (rendered, skipped, failed)."""
    suite_dir = _BDDL_ROOT / suite_name
    if not suite_dir.exists():
        print(f"  [WARN] Suite directory not found: {suite_dir}", flush=True)
        return 0, 0, 0

    bddl_files = sorted(suite_dir.glob("*.bddl"))
    if not bddl_files:
        if verbose:
            print(f"  [WARN] No .bddl files in {suite_dir}", flush=True)
        return 0, 0, 0

    rendered = skipped = failed = 0

    for bddl_path in bddl_files:
        task_name = bddl_path.stem
        out_path  = _ASSETS_DIR / suite_name / f"{task_name}.png"

        if skip_existing and out_path.exists():
            skipped += 1
            if verbose:
                print(f"  [SKIP] {suite_name}/{task_name}", flush=True)
            continue

        if verbose:
            print(f"  [RENDER] {suite_name}/{task_name} ...", end="", flush=True)
        try:
            render_one(bddl_path, out_path, resolution)
            rendered += 1
            if verbose:
                size_kb = out_path.stat().st_size // 1024
                print(f" OK ({size_kb} KB)", flush=True)
        except Exception as exc:
            failed += 1
            if verbose:
                print(f" FAILED: {exc}", flush=True)

    return rendered, skipped, failed


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=SUITES,
        default=None,
        help="Render only this suite (default: all suites)",
    )
    parser.add_argument(
        "--bddl",
        type=pathlib.Path,
        default=None,
        help="Render a single BDDL file and save next to it in assets/scene_images/",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Camera resolution in pixels (default: 512)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render even if the image already exists",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-task output",
    )
    args = parser.parse_args()

    # Always ensure the directory structure exists (even without mujoco).
    for suite in SUITES:
        (_ASSETS_DIR / suite).mkdir(parents=True, exist_ok=True)
    print(f"Scene image cache directory: {_ASSETS_DIR}", flush=True)

    # Dependency check.
    ok, msg = _check_deps()
    if not ok:
        print(f"\n[RENDERING UNAVAILABLE]\n{msg}", flush=True)
        sys.exit(2)

    # ── single BDDL mode ──────────────────────────────────────────────────
    if args.bddl is not None:
        bddl_path = args.bddl.resolve()
        if not bddl_path.exists():
            print(f"ERROR: BDDL file not found: {bddl_path}", flush=True)
            sys.exit(1)

        # Infer suite from parent directory name.
        suite_name = bddl_path.parent.name
        out_path   = _ASSETS_DIR / suite_name / f"{bddl_path.stem}.png"

        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path} already exists (use --overwrite to re-render)")
            sys.exit(0)

        print(f"Rendering {bddl_path.name} → {out_path}", flush=True)
        render_one(bddl_path, out_path, args.resolution)
        print("Done.", flush=True)
        return

    # ── suite / all-suites mode ───────────────────────────────────────────
    suites_to_run = [args.suite] if args.suite else SUITES

    total_rendered = total_skipped = total_failed = 0
    for suite_name in suites_to_run:
        print(f"\n── {suite_name} ──", flush=True)
        r, s, f = render_suite(
            suite_name,
            args.resolution,
            skip_existing=not args.overwrite,
            verbose=not args.quiet,
        )
        total_rendered += r
        total_skipped  += s
        total_failed   += f

    print(
        f"\n{'─'*60}\n"
        f"  rendered: {total_rendered}  |  skipped: {total_skipped}"
        f"  |  failed: {total_failed}\n"
        f"  output dir: {_ASSETS_DIR}\n",
        flush=True,
    )
    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
