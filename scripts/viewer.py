#!/usr/bin/env python3
"""Interactive MuJoCo viewer for LIBERO-Infinity Scenic perturbations.

Opens the MuJoCo passive viewer for a LIBERO task with live Scenic-sampled
perturbations.  Press R / Space / Enter to resample a fresh scene; press Q to
quit.

Usage
-----
    # Auto-generate Scenic from the BDDL (default mode)
    python scripts/viewer.py \\
        --suite libero_spatial \\
        --bddl pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate \\
        --perturbation position

    # Pass an explicit Scenic file (advanced — must match the BDDL's
    # libero_name vocabulary; the validator will reject mismatches):
    python scripts/viewer.py \\
        --suite libero_spatial \\
        --bddl pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate \\
        --scenic /path/to/custom.scenic

    # Or pass a full BDDL path directly (path shortened here for layout):
    python scripts/viewer.py \\
        --bddl <BDDL_DIR>/libero_spatial/pick_up_the_black_bowl.bddl \\
        --perturbation combined

Requirements
------------
* A working display (DISPLAY environment variable set, X11/Wayland available).
* mujoco >= 3.x with the ``mujoco.viewer`` sub-module (ships with mujoco 3+).
* The full libero-infinity simulation stack installed
  (``uv sync --extra simulation``).

Notes
-----
* Set ``MUJOCO_GL=egl`` if your headless server has EGL but *not* a full X11
  stack.  The passive viewer still requires GLFW (and therefore a DISPLAY),
  but EGL is used for off-screen rendering inside the LIBERO env.
* On macOS, replace ``python`` with ``mjpython`` as required by the MuJoCo
  passive viewer on that platform.
"""

from __future__ import annotations

import argparse
import contextlib
import pathlib
import re
import sys
import textwrap

# ---------------------------------------------------------------------------
# Repository root — make src/ importable when running as a plain script
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_BDDL_ROOT = _REPO_ROOT / "src" / "libero_infinity" / "data" / "libero_runtime" / "bddl_files"
_SCENIC_ROOT = _REPO_ROOT / "scenic"
sys.path.insert(0, str(_REPO_ROOT / "src"))

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]

PERTURBATION_CHOICES = [
    "position",
    "object",
    "distractor",
    "combined",
    "camera",
    "lighting",
    "full",
]

_VIEWER_SCENIC_OVERRIDES: dict[str, pathlib.Path] = {
    # No more handwritten Scenic overrides: the renderer-emitted Scenic
    # programs (compile_task / generate_scenic_file) cover every axis
    # (camera, lighting, distractor, …) for every BDDL task. The legacy
    # handwritten files in scenic/ have been deleted; any external caller
    # may still pass an explicit --scenic <path> argument.
}

# ---------------------------------------------------------------------------
# Viewer availability check
# ---------------------------------------------------------------------------

_VIEWER_IMPORT_ERROR: str | None = None

try:
    import mujoco.viewer as _mjv  # noqa: F401
except Exception as _exc:
    _VIEWER_IMPORT_ERROR = str(_exc)


def _require_viewer() -> None:
    """Abort with a helpful message if mujoco.viewer is not importable."""
    if _VIEWER_IMPORT_ERROR is not None:
        sys.exit(textwrap.dedent(f"""
                ERROR: Interactive viewer requires a display.

                mujoco.viewer could not be imported:
                    {_VIEWER_IMPORT_ERROR}

                To fix this:
                  1. Make sure you have mujoco >= 3.x installed.
                     (pip install mujoco  or  uv add mujoco)
                  2. Make sure a display is available:
                       - Linux desktop / VNC: DISPLAY=:0 python scripts/viewer.py ...
                       - X11 forwarding:      ssh -X host; DISPLAY=localhost:10.0 ...
                       - Headless server without display:
                           use the headless eval pipeline instead:
                           MUJOCO_GL=egl libero-eval --bddl ... --n-scenes 1 --watch cv2
                  3. On macOS the viewer requires mjpython:
                       mjpython scripts/viewer.py ...
                """).strip())


# ---------------------------------------------------------------------------
# BDDL path resolution
# ---------------------------------------------------------------------------


def _resolve_bddl(bddl_arg: str, suite: str | None) -> pathlib.Path:
    """Resolve the BDDL path from a task name or explicit path.

    Resolution order:
    1. If ``bddl_arg`` is an existing file path → use directly.
    2. If ``suite`` is given → look in _BDDL_ROOT/<suite>/<bddl_arg>.bddl
    3. Search all suites for ``<bddl_arg>.bddl`` (first match wins).
    """
    # Direct path
    candidate = pathlib.Path(bddl_arg)
    if candidate.exists():
        return candidate.resolve()

    # Add .bddl extension if missing
    stem = bddl_arg if bddl_arg.endswith(".bddl") else bddl_arg + ".bddl"

    if suite:
        suite_dir = _BDDL_ROOT / suite
        p = suite_dir / stem
        if p.exists():
            return p.resolve()
        sys.exit(
            f"ERROR: BDDL task '{bddl_arg}' not found in suite '{suite}'.\n"
            f"  Looked in: {suite_dir}\n"
            f"  Available tasks: {sorted(f.stem for f in suite_dir.glob('*.bddl'))[:5]} ..."
        )

    # Search all suites
    for s in SUITES:
        p = _BDDL_ROOT / s / stem
        if p.exists():
            return p.resolve()

    sys.exit(
        f"ERROR: BDDL task '{bddl_arg}' not found in any suite.\n"
        f"  Searched: {[str(_BDDL_ROOT / s) for s in SUITES]}\n"
        f"  Tip: pass --suite <suite> to narrow the search."
    )


def _resolve_scenic(scenic_arg: str) -> pathlib.Path:
    """Resolve a Scenic path or shorthand name.

    Resolution order:
    1. Exact existing path.
    2. `scenic/<name>`
    3. `scenic/<name>.scenic`
    """
    candidate = pathlib.Path(scenic_arg)
    if candidate.exists():
        return candidate.resolve()

    root_candidate = _SCENIC_ROOT / scenic_arg
    if root_candidate.exists():
        return root_candidate.resolve()

    stem = scenic_arg if scenic_arg.endswith(".scenic") else f"{scenic_arg}.scenic"
    root_candidate = _SCENIC_ROOT / stem
    if root_candidate.exists():
        return root_candidate.resolve()

    available = sorted(p.name for p in _SCENIC_ROOT.glob("*.scenic"))
    sys.exit(
        f"ERROR: Scenic program '{scenic_arg}' not found.\n"
        f"  Looked for: {candidate}, {_SCENIC_ROOT / scenic_arg}, {_SCENIC_ROOT / stem}\n"
        f"  Available hand-written Scenic files: {available}"
    )


def _arg_supplied(argv: list[str] | None, *names: str) -> bool:
    """Return True if any of the given CLI flags were explicitly provided."""
    args = list(sys.argv[1:] if argv is None else argv)
    return any(name in args for name in names)


def _viewer_default_scenic(perturbation: str) -> pathlib.Path | None:
    """Return a hand-written Scenic file the viewer should prefer, if any."""
    path = _VIEWER_SCENIC_OVERRIDES.get(perturbation)
    if path is None:
        return None
    return path.resolve()


def _extract_scenic_libero_names(scenic_path: pathlib.Path) -> set[str]:
    """Best-effort static extraction of literal libero_name references."""
    text = scenic_path.read_text()
    pattern = re.compile(r'with\s+libero_name\s+(?:f)?["\']([^"\']+)["\']')
    return set(pattern.findall(text))


def _validate_scenic_bddl_compatibility(
    scenic_path: pathlib.Path,
    bddl_path: pathlib.Path,
) -> None:
    """Reject explicit Scenic files that mention names absent from the BDDL."""
    from libero_infinity.task_config import TaskConfig

    scenic_names = {
        name
        for name in _extract_scenic_libero_names(scenic_path)
        if not name.startswith("distractor_")
    }
    if not scenic_names:
        return

    cfg = TaskConfig.from_bddl(bddl_path)
    valid_names = {obj.instance_name for obj in cfg.movable_objects}
    valid_names.update(fixture.instance_name for fixture in cfg.fixtures)

    missing = sorted(name for name in scenic_names if name not in valid_names)
    if not missing:
        return

    raise SystemExit(
        "ERROR: Scenic file does not match the requested BDDL task.\n"
        f"  Scenic: {scenic_path}\n"
        f"  BDDL:   {bddl_path}\n"
        f"  Names referenced by Scenic but absent from BDDL: {missing}\n"
        "  Use a task-compatible hand-written Scenic file or omit --scenic."
    )


# ---------------------------------------------------------------------------
# Scene sampler — reusable across resets
# ---------------------------------------------------------------------------


class SceneSession:
    """Manages a compiled Scenic scenario and samples episodes from it.

    Keeps the Scenic scenario compiled once and reuses it across resamplings.
    """

    def __init__(
        self,
        bddl_path: pathlib.Path,
        perturbation: str,
        scenic_path: pathlib.Path | None = None,
    ) -> None:
        self.bddl_path = bddl_path
        self.perturbation = perturbation
        self.scenic_path = scenic_path

        self._scenario = None
        self._orig_obj_classes: dict = {}
        self._generated_scenic_path: str | None = None
        self._exit_stack = contextlib.ExitStack()

        self._compile()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _compile(self) -> None:
        """Compile the Scenic scenario (one-time cost, ~2-5 s)."""
        import scenic
        from libero_infinity.compiler import generate_scenic_file
        from libero_infinity.task_config import TaskConfig

        print(f"  Loading task:    {self.bddl_path.name}")
        print(f"  Perturbation:    {self.perturbation}")
        if self.scenic_path is None:
            print("  Scenic source:   generated", flush=True)
            print("  Compiling Scenic scenario …", flush=True)
            cfg = TaskConfig.from_bddl(str(self.bddl_path))
            scenic_path = generate_scenic_file(cfg, perturbation=self.perturbation)
            self._generated_scenic_path = scenic_path
        else:
            _validate_scenic_bddl_compatibility(self.scenic_path, self.bddl_path)
            scenic_path = str(self.scenic_path)
            print(f"  Scenic source:   {self.scenic_path.name}", flush=True)
            print("  Compiling Scenic scenario …", flush=True)

        params = {"bddl_path": str(self.bddl_path)}
        self._scenario = scenic.scenarioFromFile(scenic_path, params=params)

        from libero_infinity.bddl_preprocessor import parse_object_classes

        self._orig_obj_classes = parse_object_classes(self.bddl_path.read_text())
        print("  Scenario ready.", flush=True)

    # ------------------------------------------------------------------
    # Per-episode helpers
    # ------------------------------------------------------------------

    def sample_simulation(self) -> tuple:
        """Sample a Scenic scene and set up a LIBEROSimulation.

        Returns
        -------
        (sim, per_reset_stack)
            ``sim``              — a ready LIBEROSimulation (setup() called)
            ``per_reset_stack``  — ExitStack that owns temp BDDL file;
                                   caller must close it when done with the episode.
        """
        from libero_infinity.bddl_preprocessor import bddl_for_scene
        from libero_infinity.simulator import LIBEROSimulator

        max_retries = 10
        for attempt in range(max_retries + 1):
            scene, _n_iters = self._scenario.generate(maxIterations=1000, verbosity=0)

            per_reset_stack = contextlib.ExitStack()
            effective_bddl = per_reset_stack.enter_context(
                bddl_for_scene(scene, str(self.bddl_path), self._orig_obj_classes)
            )

            simulator = LIBEROSimulator(
                bddl_path=effective_bddl,
                env_kwargs={
                    "camera_heights": 128,
                    "camera_widths": 128,
                },
            )
            sim = simulator.createSimulation(scene, maxSteps=500, verbosity=0)

            try:
                sim.setup()
                return sim, per_reset_stack
            except RuntimeError as exc:
                per_reset_stack.close()
                try:
                    sim.destroy()
                except Exception:
                    pass
                if "Invalid Scenic sample after MuJoCo settling" not in str(exc):
                    raise
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"Could not find a stable scene after {max_retries} retries: {exc}"
                    ) from exc
                print(
                    f"  [warn] Settling failed (attempt {attempt + 1}/{max_retries}), resampling …",
                    flush=True,
                )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._exit_stack.close()
        if self._generated_scenic_path:
            pathlib.Path(self._generated_scenic_path).unlink(missing_ok=True)
            self._generated_scenic_path = None


# ---------------------------------------------------------------------------
# Interactive viewer loop
# ---------------------------------------------------------------------------


def run_viewer(session: SceneSession) -> None:
    """Open the MuJoCo passive viewer and handle R/Space/Enter/Q key presses.

    The viewer is kept alive across scene resamples — the same window is
    reused; only the MuJoCo model/data handles change.
    """
    import mujoco
    import mujoco.viewer as mjv

    print(
        "\n"
        "┌─────────────────────────────────────────────────────────┐\n"
        "│  LIBERO-Infinity Interactive Viewer                      │\n"
        "├─────────────────────────────────────────────────────────┤\n"
        "│  R / Space / Enter  →  resample a new scene              │\n"
        "│  Left drag        →  rotate / orbit camera               │\n"
        "│  Right drag       →  pan camera                          │\n"
        "│  Scroll / middle  →  zoom camera                         │\n"
        "│  Q                →  quit                                │\n"
        "└─────────────────────────────────────────────────────────┘",
        flush=True,
    )

    # State shared between the key callback and the main loop.
    _state = {"resample": False, "quit": False}

    def _key_callback(keycode: int) -> None:
        """mujoco.viewer key callback (called on the viewer thread)."""
        # GLFW key codes:
        #   R = 82, Space = 32, Enter = 257, Q = 81
        if keycode in (82, 32, 257):  # R, Space, Enter
            _state["resample"] = True
        elif keycode == 81:  # Q
            _state["quit"] = True

    scene_index = 0

    def _load_scene() -> tuple:
        nonlocal scene_index
        scene_index += 1
        print(f"\n  [Scene {scene_index}] Sampling …", flush=True)
        sim, stack = session.sample_simulation()
        mjmodel, mjdata = sim.mj_handles
        print(f"  [Scene {scene_index}] Ready.", flush=True)
        return sim, stack, mjmodel, mjdata

    # Load the first scene.
    sim, per_reset_stack, mjmodel, mjdata = _load_scene()

    try:
        with mjv.launch_passive(
            mjmodel,
            mjdata,
            key_callback=_key_callback,
        ) as handle:
            while handle.is_running():
                if _state["quit"]:
                    print("  Quitting …", flush=True)
                    break

                if _state["resample"]:
                    _state["resample"] = False
                    print("  Resampling scene …", flush=True)

                    # Tear down the current episode.
                    try:
                        sim.destroy()
                    except Exception:
                        pass
                    per_reset_stack.close()

                    # Load the new scene.
                    sim, per_reset_stack, mjmodel_new, mjdata_new = _load_scene()

                    # The passive viewer's model/data cannot be hot-swapped
                    # through the handle — we need to reload the viewer.
                    # Re-launch with the new handles by breaking here; the
                    # outer while-loop will restart via the nested approach.
                    #
                    # MuJoCo >= 3.x passive viewer supports reload via
                    # handle.load(mjmodel_new, mjdata_new) when available.
                    try:
                        _reload_viewer_handle(handle, mjmodel_new, mjdata_new)
                        mjmodel, mjdata = mjmodel_new, mjdata_new
                    except AttributeError:
                        # Older mujoco.viewer without load() — close and
                        # re-launch the viewer with the new model.
                        break  # handled by the outer retry loop below

                # Advance physics one step (zero action — robot holds pose).
                mujoco.mj_step(mjmodel, mjdata)

                with handle.lock():
                    handle.sync()
    finally:
        try:
            sim.destroy()
        except Exception:
            pass
        per_reset_stack.close()


def run_viewer_restart_on_resample(session: SceneSession) -> None:
    """Viewer loop that restarts the window on each resample.

    Fallback for older mujoco.viewer that lacks ``handle.load()``.
    Keeps looping until the user presses Q (detected by the window closing
    immediately after R/Space/Enter when a sentinel flag is set).
    """
    import mujoco
    import mujoco.viewer as mjv

    print(
        "\n"
        "┌─────────────────────────────────────────────────────────┐\n"
        "│  LIBERO-Infinity Interactive Viewer                      │\n"
        "├─────────────────────────────────────────────────────────┤\n"
        "│  R / Space / Enter  →  resample a new scene              │\n"
        "│  Left drag        →  rotate / orbit camera               │\n"
        "│  Right drag       →  pan camera                          │\n"
        "│  Scroll / middle  →  zoom camera                         │\n"
        "│  Q / close window   →  quit                              │\n"
        "└─────────────────────────────────────────────────────────┘",
        flush=True,
    )

    _state = {"resample": False, "quit": False}

    def _key_callback(keycode: int) -> None:
        if keycode in (82, 32, 257):
            _state["resample"] = True
        elif keycode == 81:
            _state["quit"] = True

    scene_index = 0

    while not _state["quit"]:
        scene_index += 1
        print(f"\n  [Scene {scene_index}] Sampling …", flush=True)
        sim, per_reset_stack = session.sample_simulation()
        mjmodel, mjdata = sim.mj_handles
        print(f"  [Scene {scene_index}] Ready.  (Close window or press Q to quit)", flush=True)

        _state["resample"] = False  # clear before opening window

        try:
            with mjv.launch_passive(
                mjmodel,
                mjdata,
                key_callback=_key_callback,
            ) as handle:
                while handle.is_running():
                    if _state["quit"] or _state["resample"]:
                        break
                    mujoco.mj_step(mjmodel, mjdata)
                    with handle.lock():
                        handle.sync()
        finally:
            try:
                sim.destroy()
            except Exception:
                pass
            per_reset_stack.close()

        # If window was closed without pressing R/Q, treat as quit.
        if not _state["resample"] and not _state["quit"]:
            _state["quit"] = True


def _reload_viewer_handle(handle, mjmodel, mjdata) -> None:
    """Reload a passive viewer handle across MuJoCo API variants."""
    load = getattr(handle, "load", None)
    if callable(load):
        load(mjmodel, mjdata)
        return

    sim_ref = getattr(handle, "_sim", None)
    sim = sim_ref() if callable(sim_ref) else None
    sim_load = getattr(sim, "load", None)
    if callable(sim_load):
        sim_load(mjmodel, mjdata, "")
        return

    raise AttributeError("passive viewer handle has no reload support")


def _viewer_supports_reload(mjv_module) -> bool:
    """True if this mujoco.viewer build can hot-reload model/data."""
    handle_cls = getattr(mjv_module, "Handle", object)
    if hasattr(handle_cls, "load"):
        return True
    simulate_cls = getattr(mjv_module, "_Simulate", None)
    return simulate_cls is not None and hasattr(simulate_cls, "load")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bddl",
        required=True,
        metavar="TASK_OR_PATH",
        help=(
            "Task name (e.g. pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate) "
            "or explicit path to a .bddl file."
        ),
    )
    p.add_argument(
        "--suite",
        choices=SUITES,
        default=None,
        metavar="SUITE",
        help=(
            f"Suite to search for the task (one of {SUITES}). "
            "Required when --bddl is a task name that appears in multiple suites."
        ),
    )
    p.add_argument(
        "--perturbation",
        choices=PERTURBATION_CHOICES,
        default="position",
        metavar="PERTURBATION",
        help=(
            f"Perturbation axis (one of {PERTURBATION_CHOICES}, default: position). "
            "'combined' = position + object identity; 'full' = all axes."
        ),
    )
    p.add_argument(
        "--scenic",
        default=None,
        metavar="SCENIC",
        help=(
            "Optional Scenic program to load instead of auto-generating one from the BDDL. "
            "Accepts an explicit path or a hand-written file name from scenic/, "
            "e.g. 'position_perturbation' or 'scenic/position_perturbation.scenic'."
        ),
    )
    p.add_argument(
        "--no-restart",
        action="store_true",
        default=False,
        help=(
            "Prefer in-place model reload (mujoco >= 3.x handle.load). "
            "Ignored — the script auto-detects and falls back gracefully."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    _require_viewer()

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.scenic and _arg_supplied(argv, "--perturbation"):
        print(
            "WARNING: --perturbation is ignored when --scenic is provided; "
            "the viewer will load the explicit Scenic program.",
            file=sys.stderr,
            flush=True,
        )

    bddl_path = _resolve_bddl(args.bddl, args.suite)
    scenic_path = (
        _resolve_scenic(args.scenic) if args.scenic else _viewer_default_scenic(args.perturbation)
    )
    print(f"\nBDDL:          {bddl_path}")
    if scenic_path is not None:
        print(f"Scenic:        {scenic_path}")

    # Bootstrap LIBERO runtime paths (downloads assets to ~/.libero if needed).
    try:
        from libero_infinity.runtime import ensure_runtime

        ensure_runtime()
    except ImportError:
        pass  # older installations without runtime module

    session = SceneSession(bddl_path, args.perturbation, scenic_path=scenic_path)

    try:
        import mujoco.viewer as mjv

        supports_in_place_reload = _viewer_supports_reload(mjv)
        if supports_in_place_reload:
            run_viewer(session)
        else:
            run_viewer_restart_on_resample(session)
    except Exception as exc:
        # If run_viewer raised (e.g. GLFW issue), fall back to restart mode
        # only when this build truly lacks passive-viewer reload support.
        err_str = str(exc).lower()
        if "display" in err_str or "glfw" in err_str or "x11" in err_str:
            sys.exit(
                "\nERROR: Interactive viewer requires a display.\n"
                "  Run with: DISPLAY=:0 python scripts/viewer.py ...\n"
                "  Or use headless mode: MUJOCO_GL=egl libero-eval --bddl ... --watch cv2\n"
                f"\nOriginal error: {exc}"
            )
        print(
            f"  [info] Falling back to restart-on-resample viewer ({exc})",
            flush=True,
        )
        if _viewer_supports_reload(mjv):
            raise
        run_viewer_restart_on_resample(session)
    finally:
        session.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
