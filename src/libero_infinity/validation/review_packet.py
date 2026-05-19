"""G7 human-review packet generator.

For a given (task BDDL, axis subset, seed) tuple this writes a self-contained
directory of files that a human reviewer needs to grade per the rubric in
``docs/human_review_rubric.md``.

CLI
---
::

    python -m libero_infinity.validation.review_packet \\
        --task path/to/task.bddl \\
        --subset position,object,distractor \\
        --seed 0 \\
        --out /tmp/packet_dir

Programmatic API
----------------
- :func:`generate_packet` — build all packet artifacts on disk.
- :func:`PACKET_FILES` — manifest of relative paths the generator writes.
- :func:`subset_token` — canonical "subset" token used in directory names.

The generator is *defensive*: when an optional step (rendering, baseline
compile) fails it writes a placeholder file and a `.error` companion so the
packet is still complete and human-inspectable.  This is important because
the §3 sampling design generates tens of thousands of packets and a single
flaky task should not halt the run.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import logging
import pathlib
import random
import sys
import traceback
import warnings
from typing import Any

import yaml

from libero_infinity.compiler import compile_task_to_scenic
from libero_infinity.planner.composition import AXIS_PRESETS, parse_axes
from libero_infinity.task_config import TaskConfig

logger = logging.getLogger(__name__)

# Canonical list of perturbation axes (matches the planner's `full` preset).
ALL_AXES: tuple[str, ...] = (
    "position",
    "object",
    "robot",
    "camera",
    "lighting",
    "texture",
    "distractor",
    "background",
    "articulation",
)

# Relative paths the packet contains, in spec order.
PACKET_FILES: tuple[str, ...] = (
    "bddl/original.bddl",
    "task_language.md",
    "task_graph.md",
    "scenic/generated.scenic",
    "scenic/baseline_diff.md",
    "axis_params.json",
    "compiler_diagnostics.txt",
    "rendered/scene.png",
    "checklist.yaml",
)

# The eleven rubric questions, mirrored from docs/human_review_rubric.md §3.
CHECKLIST_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("q1_goal_semantics", "Does the Scenic program preserve the BDDL goal semantics?"),
    (
        "q2_entities_present",
        "Are all BDDL-declared objects, fixtures, and regions present in the Scenic program?",
    ),
    ("q3_init_predicates", "Are the init predicates respected?"),
    ("q4_axis_envelopes", "Are the perturbations within their declared envelopes?"),
    (
        "q5_object_affordances",
        "If the object axis is active, do substituted objects preserve the task's affordances?",
    ),
    (
        "q6_distractor_blocking",
        "If the distractor axis is active, do distractors avoid blocking the goal?",
    ),
    (
        "q7_articulation_state",
        "If the articulation axis is active, are joint sample ranges consistent with the goal?",
    ),
    (
        "q8_language_faithful",
        "Is the natural-language instruction still faithful to the rendered scene?",
    ),
    (
        "q9_compiler_diagnostics",
        "Are there compiler diagnostics or warnings that hint at a semantic issue?",
    ),
    (
        "q10_physical_plausibility",
        (
            "Is the rendered scene physically plausible "
            "(no float-throughs, intersections, fallen fixtures)?"
        ),
    ),
    (
        "q11_baseline_diff_clean",
        (
            "Does the baseline diff isolate the perturbation cleanly, "
            "or does it leak unrelated changes?"
        ),
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def subset_token(subset: frozenset[str] | list[str] | str) -> str:
    """Return a stable filesystem-safe token for a subset.

    - empty (`none`) → "none"
    - canonical presets `combined`/`full` are preserved when they match
    - otherwise: sorted axis names joined with `+`
    """
    if isinstance(subset, str):
        subset = parse_axes(subset)
    axes = frozenset(subset)
    if not axes:
        return "none"
    for name, preset in AXIS_PRESETS.items():
        if axes == preset:
            return name
    return "+".join(sorted(axes))


def _write(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _safe(fn, *args, **kwargs):
    """Call ``fn``; return ``(result, None)`` or ``(None, traceback_str)``."""
    try:
        return fn(*args, **kwargs), None
    except Exception:  # noqa: BLE001 — packet must be best-effort
        return None, traceback.format_exc()


# ---------------------------------------------------------------------------
# Artifact builders
# ---------------------------------------------------------------------------


def _task_graph_markdown(cfg: TaskConfig) -> str:
    """Render the parsed task graph as readable markdown."""
    out: list[str] = [
        "# Task Graph",
        "",
        f"**Source BDDL:** `{cfg.bddl_path}`",
        "",
        f"**Language:** {cfg.language}",
        "",
        "## Fixtures",
        "",
    ]
    if cfg.fixtures:
        out.append("| name | class |")
        out.append("|---|---|")
        for fx in cfg.fixtures:
            name = getattr(fx, "name", "?")
            klass = getattr(fx, "fixture_class", getattr(fx, "klass", "?"))
            out.append(f"| `{name}` | `{klass}` |")
    else:
        out.append("_(none)_")

    out += ["", "## Movable objects", ""]
    if cfg.movable_objects:
        out.append("| name | class |")
        out.append("|---|---|")
        for ob in cfg.movable_objects:
            name = getattr(ob, "name", "?")
            klass = getattr(ob, "object_class", getattr(ob, "klass", "?"))
            out.append(f"| `{name}` | `{klass}` |")
    else:
        out.append("_(none)_")

    out += ["", "## Regions", ""]
    if cfg.regions:
        out.append("| name | full_name |")
        out.append("|---|---|")
        for rname, region in cfg.regions.items():
            full = getattr(region, "full_name", rname)
            out.append(f"| `{rname}` | `{full}` |")
    else:
        out.append("_(none)_")

    out += ["", "## Init predicates", "", "```", cfg.init_text.strip() or "(empty)", "```", ""]
    out += ["## Goal predicates", "", "```", cfg.goal_text.strip() or "(empty)", "```", ""]
    return "\n".join(out) + "\n"


def _baseline_diff(baseline_src: str, perturbed_src: str, subset_label: str) -> str:
    """Unified diff `baseline → perturbed`, rendered as markdown."""
    diff = list(
        difflib.unified_diff(
            baseline_src.splitlines(keepends=False),
            perturbed_src.splitlines(keepends=False),
            fromfile="baseline (axes=none)",
            tofile=f"perturbed (axes={subset_label})",
            n=3,
            lineterm="",
        )
    )
    if not diff:
        body = "_(no diff — perturbed program is byte-identical to the baseline)_"
    else:
        body = "```diff\n" + "\n".join(diff) + "\n```"
    return f"# Baseline diff\n\nAxes active: `{subset_label}`\n\n{body}\n"


def _axis_params(plan_diagnostics: Any, subset: frozenset[str], seed: int) -> dict:
    """Extract concrete per-axis parameters from the plan's diagnostics, if any."""
    params: dict[str, Any] = {
        "_seed": seed,
        "_axes_requested": sorted(subset),
    }
    # plan_diagnostics may carry per-axis records; surface whatever's present.
    for axis in ALL_AXES:
        params[axis] = {"active": axis in subset}
    if plan_diagnostics is not None:
        for attr in ("dropped_axes", "warnings", "info"):
            val = getattr(plan_diagnostics, attr, None)
            if val:
                params.setdefault("_diagnostics", {})[attr] = (
                    list(val) if not isinstance(val, dict) else dict(val)
                )
    return params


def _empty_checklist(task_path: pathlib.Path, subset_label: str, seed: int) -> str:
    """Render an empty `checklist.yaml` ready for a reviewer to fill in."""
    doc: dict[str, Any] = {
        "packet": {
            "task": str(task_path),
            "axis_subset": subset_label,
            "seed": seed,
        },
        "coi_disclosed": None,  # reviewer fills in
        "reviewer": None,  # reviewer handle
        "reviewer2": None,
        "adjudicator": None,
        "items": [
            {
                "id": qid,
                "question": qtext,
                "severity": None,  # 0..4
                "reviewer2_severity": None,
                "adjudicator_severity": None,
                "notes": "",
            }
            for qid, qtext in CHECKLIST_QUESTIONS
        ],
        "overall_severity": None,  # computed by aggregator
    }
    return yaml.safe_dump(doc, sort_keys=False, width=88)


def _render_scene_png(scenic_src: str, out_path: pathlib.Path) -> str | None:
    """Best-effort render: compile + sample once + dump a PNG.

    Returns ``None`` on success, an error string on failure (caller writes
    `.error` companion).  Falls back to a matplotlib placeholder if Scenic
    cannot be sampled in this environment.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Write to a temp file inside scenic/ for model resolution
        import tempfile

        import scenic  # noqa: PLC0415
        from libero_infinity.compiler import _scenic_model_dir  # type: ignore

        model_dir = _scenic_model_dir()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".scenic", dir=model_dir, delete=False, encoding="utf-8"
        ) as fh:
            fh.write(scenic_src)
            tmp = pathlib.Path(fh.name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scenario = scenic.scenarioFromFile(str(tmp))
                scene, _iters = scenario.generate(maxIterations=2000)
        finally:
            tmp.unlink(missing_ok=True)

        # Scenic ships a built-in matplotlib renderer:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        try:
            scene.show2D(ax=ax, zoom=1)  # type: ignore[attr-defined]
        except Exception:
            # Fall back to a textual placeholder render.
            ax.text(
                0.5,
                0.5,
                f"scene sampled; {len(scene.objects)} objects\n(2-D render unavailable)",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return None
    except Exception:  # noqa: BLE001
        # Write a 1x1 placeholder PNG so the file exists.
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415

            fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
            ax.text(0.5, 0.5, "render unavailable", ha="center", va="center")
            ax.set_axis_off()
            fig.savefig(out_path)
            plt.close(fig)
        except Exception:
            out_path.write_bytes(b"")  # absolute last resort
        return traceback.format_exc()


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PacketResult:
    out_dir: pathlib.Path
    files_written: list[str]
    errors: dict[str, str]  # filename → traceback


def generate_packet(
    task_bddl: str | pathlib.Path,
    subset: str | frozenset[str],
    seed: int,
    out_dir: str | pathlib.Path,
    *,
    render: bool = True,
) -> PacketResult:
    """Generate every file listed in :data:`PACKET_FILES`.

    Args:
        task_bddl: Path to the BDDL file.
        subset: Comma-separated axes, a preset name (``combined``/``full``),
            ``"none"`` for the empty subset, or a ``frozenset``.
        seed: Sampling seed (recorded in ``axis_params.json`` and the
            checklist; influences any axis whose planner is seeded).
        out_dir: Output directory (created).
        render: If ``False``, skip the rendered PNG step entirely.
    """
    task_bddl = pathlib.Path(task_bddl).resolve()
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(subset, str):
        if subset.strip() in ("", "none"):
            axes = frozenset()
        else:
            axes = parse_axes(subset)
    else:
        axes = frozenset(subset)
    label = subset_token(axes)

    errors: dict[str, str] = {}
    written: list[str] = []

    # Seed both random and numpy for any seeded planner code paths.
    random.seed(seed)
    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
    except Exception:
        pass

    # 1. bddl/original.bddl
    bddl_src = task_bddl.read_text(encoding="utf-8")
    _write(out_dir / "bddl/original.bddl", bddl_src)
    written.append("bddl/original.bddl")

    # 2. Parse + 3. task graph
    cfg, parse_err = _safe(TaskConfig.from_bddl, str(task_bddl))
    if parse_err is not None:
        errors["bddl/original.bddl.parse"] = parse_err
        _write(
            out_dir / "task_language.md",
            "# Task language\n\n_(BDDL parse failed; see compiler_diagnostics.txt)_\n",
        )
        _write(
            out_dir / "task_graph.md",
            "# Task graph\n\n_(BDDL parse failed; see compiler_diagnostics.txt)_\n",
        )
    else:
        _write(out_dir / "task_language.md", f"# Task language\n\n> {cfg.language}\n")
        _write(out_dir / "task_graph.md", _task_graph_markdown(cfg))
    written.append("task_language.md")
    written.append("task_graph.md")

    # 4. scenic/generated.scenic
    if cfg is None:
        _write(out_dir / "scenic/generated.scenic", "// (parse failed — no scenic source)\n")
        scenic_src = ""
    else:
        scenic_src, err = _safe(compile_task_to_scenic, cfg, axes)
        if err is not None:
            errors["scenic/generated.scenic"] = err
            scenic_src = f"// (compile failed)\n// {err.splitlines()[-1] if err else ''}\n"
        _write(out_dir / "scenic/generated.scenic", scenic_src or "")
    written.append("scenic/generated.scenic")

    # 5. scenic/baseline_diff.md (vs no-axes baseline for same task)
    baseline_src = ""
    if cfg is not None:
        baseline_src, err = _safe(compile_task_to_scenic, cfg, frozenset())
        if err is not None:
            errors["scenic/baseline_diff.md"] = err
            baseline_src = ""
    _write(
        out_dir / "scenic/baseline_diff.md",
        _baseline_diff(baseline_src or "", scenic_src or "", label),
    )
    written.append("scenic/baseline_diff.md")

    # 6. axis_params.json — populate from planner if possible
    plan_diag = None
    if cfg is not None and axes:
        try:
            from libero_infinity.ir.graph_builder import build_semantic_scene_graph  # noqa: PLC0415
            from libero_infinity.planner.composition import plan_perturbations  # noqa: PLC0415

            graph = build_semantic_scene_graph(cfg)
            plan = plan_perturbations(graph, axes)
            plan_diag = plan.diagnostics
        except Exception:  # noqa: BLE001
            errors["axis_params.json.plan"] = traceback.format_exc()
    _write(
        out_dir / "axis_params.json",
        json.dumps(_axis_params(plan_diag, axes, seed), indent=2, default=str),
    )
    written.append("axis_params.json")

    # 7. compiler_diagnostics.txt
    diag_lines: list[str] = [f"task: {task_bddl}", f"axis_subset: {label}", f"seed: {seed}", ""]
    if errors:
        for name, tb in errors.items():
            diag_lines += [f"=== {name} ===", tb, ""]
    else:
        diag_lines.append("(no errors)")
    _write(out_dir / "compiler_diagnostics.txt", "\n".join(diag_lines))
    written.append("compiler_diagnostics.txt")

    # 8. rendered/scene.png
    png_path = out_dir / "rendered/scene.png"
    if render and scenic_src and not scenic_src.startswith("// (compile failed"):
        err = _render_scene_png(scenic_src, png_path)
        if err is not None:
            (out_dir / "rendered/scene.png.error").write_text(err)
            errors["rendered/scene.png"] = err
    else:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_bytes(b"")
    written.append("rendered/scene.png")

    # 9. checklist.yaml
    _write(out_dir / "checklist.yaml", _empty_checklist(task_bddl, label, seed))
    written.append("checklist.yaml")

    return PacketResult(out_dir=out_dir, files_written=written, errors=errors)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m libero_infinity.validation.review_packet",
        description="Generate a single G7 review packet.",
    )
    p.add_argument("--task", required=True, help="Path to a BDDL file.")
    p.add_argument("--subset", default="none", help="Comma-separated axes, preset name, or 'none'.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--no-render", action="store_true", help="Skip the rendered PNG step.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = generate_packet(
        args.task,
        args.subset,
        args.seed,
        args.out,
        render=not args.no_render,
    )
    print(f"wrote {len(result.files_written)} files to {result.out_dir}")
    if result.errors:
        print(f"  with {len(result.errors)} non-fatal error(s):", file=sys.stderr)
        for name in result.errors:
            print(f"    {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
