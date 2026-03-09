"""Top-level compiler API for Libero-Infinity.

Wires together the four compiler stages:
  1. Parse       — TaskConfig.from_bddl()
  2. Semantic    — build_semantic_scene_graph()
  3. Plan        — plan_perturbations()
  4. Render      — render_scenic()

Usage::

    from libero_infinity.compiler import compile_task_to_scenic, compile_task_to_scenario
    from libero_infinity.task_config import TaskConfig

    cfg = TaskConfig.from_bddl("path/to/task.bddl")

    # Get the Scenic program as a string:
    scenic_str = compile_task_to_scenic(cfg, "position")

    # Or compile all the way to a Scenic Scenario object:
    scenario = compile_task_to_scenario(cfg, "position")
    scene, n_iters = scenario.generate()

Public API
----------
- ``compile_task_to_scenic``   → str  (Scenic program text)
- ``compile_task_to_scenario`` → Scenario  (compiled, ready for .generate())
- ``generate_scenic``          → str  (backward-compat alias)
"""

from __future__ import annotations

import hashlib
import pathlib
import tempfile

from libero_infinity.ir.graph_builder import build_semantic_scene_graph
from libero_infinity.planner.composition import plan_perturbations
from libero_infinity.renderer.scenic_renderer import (
    _FIXTURE_DIMS as _FIXTURE_DIMENSIONS,  # noqa: F401  (re-export for callers)
)
from libero_infinity.renderer.scenic_renderer import render_scenic
from libero_infinity.task_config import TaskConfig

__all__ = [
    "compile_task_to_scenic",
    "compile_task_to_scenario",
    "generate_scenic",
    "generate_scenic_file",
    "_FIXTURE_DIMENSIONS",
]


def _scenic_model_dir() -> pathlib.Path:
    """Return the ``scenic/`` directory that contains ``libero_model.scenic``.

    Scenic resolves ``model libero_model`` relative to the calling ``.scenic``
    file's location, so any temp file used by compile_task_to_scenario() must
    live in this directory.
    """
    # compiler.py lives at src/libero_infinity/compiler.py; scenic/ is at repo root
    return pathlib.Path(__file__).parent.parent.parent / "scenic"


def compile_task_to_scenic(
    cfg: TaskConfig,
    request: str | frozenset[str] = "position",
) -> str:
    """Full compiler pipeline: parse → semantic graph → plan → render.

    This is the primary entry point for the new compiler.

    Args:
        cfg: Parsed BDDL task config (from TaskConfig.from_bddl()).
        request: Perturbation axes to activate. Accepts a comma-separated
            string (``"position,camera"``), a preset name (``"combined"``,
            ``"full"``), or a frozenset of axis names.

    Returns:
        A valid Scenic 3 program string.
    """
    graph = build_semantic_scene_graph(cfg)
    plan = plan_perturbations(graph, request)
    return render_scenic(plan, graph)


def compile_task_to_scenario(
    cfg: TaskConfig,
    request: str | frozenset[str] = "position",
    params: dict | None = None,
):
    """Compile a BDDL task config all the way to a Scenic Scenario object.

    Calls :func:`compile_task_to_scenic` to obtain the Scenic program string,
    writes it to a temporary file inside the ``scenic/`` directory (so that
    the ``model libero_model`` import resolves correctly), compiles it via
    ``scenic.scenarioFromFile()``, then removes the temp file.

    Args:
        cfg: Parsed BDDL task config (from TaskConfig.from_bddl()).
        request: Perturbation axes to activate. Accepted forms are identical
            to :func:`compile_task_to_scenic`.
        params: Optional dict of Scenic global-parameter overrides forwarded
            to ``scenarioFromFile(params=…)``. The ``task`` and ``bddl_path``
            globals are already embedded in the generated program string; only
            pass this if you need to override e.g. ``min_clearance``.

    Returns:
        A compiled ``scenic.core.scenarios.Scenario`` object — the same type
        returned by ``scenic.scenarioFromFile()``.  Call
        ``scenario.generate()`` to sample a scene.

    Raises:
        ImportError: If the ``scenic`` package is not installed.
    """
    import scenic  # noqa: PLC0415  (lazy import — scenic is an optional dep)

    scenic_str = compile_task_to_scenic(cfg, request)
    model_dir = _scenic_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    # Write to a temp file inside scenic/ so `model libero_model` resolves.
    # NamedTemporaryFile with delete=False: we close it before Scenic reads it
    # (required on all platforms), then clean up in the finally block.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".scenic",
        dir=model_dir,
        delete=False,
        encoding="utf-8",
    ) as fh:
        fh.write(scenic_str)
        tmp_path = pathlib.Path(fh.name)

    try:
        return scenic.scenarioFromFile(str(tmp_path), params=dict(params or {}))
    finally:
        tmp_path.unlink(missing_ok=True)


def generate_scenic(cfg: TaskConfig, perturbation: str = "position") -> str:
    """Thin wrapper for backward compatibility during migration.

    Delegates to compile_task_to_scenic(). Callers should migrate to
    compile_task_to_scenic() directly.
    """
    return compile_task_to_scenic(cfg, perturbation)


def _sanitize(name: str) -> str:
    """Convert a BDDL name to a valid Python/Scenic identifier."""
    return name.replace("-", "_").replace(" ", "_")


def generate_scenic_file(
    cfg: TaskConfig,
    perturbation: str = "position",
    output_dir: str | pathlib.Path | None = None,
    **kwargs,
) -> str:
    """Generate a .scenic program and write it to a file.

    Generates a .scenic program file using the compiler pipeline.
    Writes the compiler output to a temp file in scenic/ (or a custom directory)
    so that ``model libero_model`` resolves correctly.

    Args:
        cfg: Parsed BDDL task config.
        perturbation: Perturbation type (e.g. ``"position"``, ``"combined"``).
        output_dir: Optional output directory. Defaults to the scenic/ root so
            that ``model libero_model`` is automatically resolvable.
        **kwargs: Ignored (accepted for backward-compat with old API).

    Returns:
        Absolute path string to the written .scenic file.
    """
    scenic_root = _scenic_model_dir()
    if output_dir is not None:
        out_dir = pathlib.Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Ensure libero_model.scenic is reachable from the output directory
        if out_dir.resolve() != scenic_root.resolve():
            model_link = out_dir / "libero_model.scenic"
            if not model_link.exists() and not model_link.is_symlink():
                model_target = scenic_root / "libero_model.scenic"
                model_link.symlink_to(model_target)
    else:
        out_dir = scenic_root
        out_dir.mkdir(parents=True, exist_ok=True)

    scenic_src = compile_task_to_scenic(cfg, perturbation)

    task_slug = _sanitize(cfg.language.lower().replace(" ", "_"))
    path_digest = hashlib.sha1(cfg.bddl_path.encode("utf-8")).hexdigest()[:8]
    out_path = out_dir / f"_gen_{task_slug}_{path_digest}_{perturbation}.scenic"
    out_path.write_text(scenic_src, encoding="utf-8")
    return str(out_path.resolve())
