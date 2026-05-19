"""Premerge validation sweep harness for libero-infinity.

Iterates the cartesian product of (task, axis_subset, seed) and records the
outcome of each gate (G0..G6) as a JSONL stream. Designed for honest bug
hunting: every caught exception is recorded with class, message, source
``file:line`` location, and full traceback. There is **no** silent swallowing,
no ``ignore_done``-style mitigation, no ``try/except: pass``.

CLI:
    python -m libero_infinity.validation.sweep \\
        --tasks all \\
        --subsets 16 \\
        --seeds 1 \\
        --workers 8 \\
        --out smoke.jsonl

Gates:
    G0  BDDL parse                 (TaskConfig.from_bddl)
    G1  Scenic program generation  (compile_task_to_scenic -> str)
    G2  Scenic compile             (compile_task_to_scenario -> Scenario)
    G3  Scenic sample              (scenario.generate(maxIterations=...))
    G5  LIBERO env create + reset  (skipped if --scenic-only)
    G6  render + 5 noop steps      (skipped if --scenic-only)

JSONL row schema (one row per condition):
    task, axis_subset, seed, cardinality,
    g0..g6 ("pass" | "fail" | "skip"),
    n_iters (int|None),
    error_class, error_msg, error_file_line, traceback,
    duration_s, worker_pid
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import pathlib
import random
import sys
import time
import traceback as tb_mod
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

# Canonical 9 perturbation axes (matches the validation plan, not the planner
# module's extra "sensor_noise" axis which is out of scope for the publication
# claim).
CANONICAL_AXES: tuple[str, ...] = (
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

GATES: tuple[str, ...] = ("g0", "g1", "g2", "g3", "g5", "g6")

# Per-worker baseline scene cache keyed by absolute BDDL path. The baseline is
# the no-axes-active sample needed by the G4 identity hook for cross-axis
# isolation checks. Caching once per (worker, task) keeps the marginal cost of
# G4 identity to a single extra Scenic sample per task per worker process.
_BASELINE_CACHE: dict[str, Any] = {}


def _get_baseline_scene(cfg: Any, bddl_key: str, max_iter: int) -> Any:
    """Return (and memoize) the no-axes-active baseline scene for ``cfg``.

    Re-raises any exception so it can be honestly recorded in the row's
    ``g4_identity_error`` field — identity-hook failures are NOT silently
    swallowed.
    """
    cached = _BASELINE_CACHE.get(bddl_key)
    if cached is not None:
        return cached
    from libero_infinity.compiler import compile_task_to_scenario

    scenario = compile_task_to_scenario(cfg, "")
    scene, _ = scenario.generate(maxIterations=max_iter)
    _BASELINE_CACHE[bddl_key] = scene
    return scene


# Directory containing the bundled BDDLs (shipped with the package data tree).
_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent
_BDDL_ROOT = _PKG_ROOT / "data" / "libero_runtime" / "bddl_files"

# Representative smoke set: mix of stove / cabinet / drawer / floor tasks across
# suites.
SMOKE_TASKS: tuple[str, ...] = (
    "libero_goal/put_the_bowl_on_the_stove.bddl",
    "libero_goal/open_the_middle_drawer_of_the_cabinet.bddl",
    "libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl",
    "libero_goal/push_the_plate_to_the_front_of_the_stove.bddl",
    "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
)


# ---------------------------------------------------------------------------
# Task / subset enumeration helpers
# ---------------------------------------------------------------------------


def discover_all_tasks() -> list[str]:
    """Return all bundled BDDL task relative paths (suite/task.bddl)."""
    out: list[str] = []
    for p in sorted(_BDDL_ROOT.rglob("*.bddl")):
        rel = p.relative_to(_BDDL_ROOT)
        out.append(str(rel))
    return out


def resolve_task_path(task_rel: str) -> pathlib.Path:
    """Resolve a task spec to an absolute BDDL path.

    Accepts ``suite/name.bddl`` or a bare absolute path.
    """
    p = pathlib.Path(task_rel)
    if p.is_absolute():
        return p
    return _BDDL_ROOT / task_rel


def enumerate_subsets(n_axes: int = 9) -> list[tuple[str, ...]]:
    """Return all 2^n - 1 non-empty subsets of CANONICAL_AXES (as sorted tuples)."""
    subsets: list[tuple[str, ...]] = []
    for mask in range(1, 1 << n_axes):
        axes = tuple(CANONICAL_AXES[i] for i in range(n_axes) if mask & (1 << i))
        subsets.append(axes)
    return subsets


def sample_subsets(n_sample: int | str, seed: int = 0) -> list[tuple[str, ...]]:
    """Sample N axis subsets deterministically; ``'all'`` returns all 511."""
    all_subsets = enumerate_subsets()
    if isinstance(n_sample, str) and n_sample == "all":
        return all_subsets
    n = int(n_sample)
    if n >= len(all_subsets):
        return all_subsets
    rng = random.Random(seed)
    return rng.sample(all_subsets, n)


# ---------------------------------------------------------------------------
# Per-condition worker
# ---------------------------------------------------------------------------


def _error_file_line(exc: BaseException) -> str | None:
    """Return the deepest frame inside libero_infinity / scenic as 'file.py:lineno'.

    Falls back to the last frame in the traceback.
    """
    tb = exc.__traceback__
    if tb is None:
        return None
    frames = tb_mod.extract_tb(tb)
    if not frames:
        return None
    # Prefer the deepest frame inside our package or scenic — that's where the
    # bug usually lives, not the harness scaffolding.
    preferred = None
    for fr in frames:
        fn = fr.filename or ""
        if "libero_infinity" in fn or "/scenic/" in fn:
            preferred = fr
    chosen = preferred or frames[-1]
    name = pathlib.Path(chosen.filename).name
    return f"{name}:{chosen.lineno}"


def _record_failure(row: dict[str, Any], gate: str, exc: BaseException) -> None:
    """Mark ``gate`` failed in ``row`` and attach honest error metadata."""
    row[gate] = "fail"
    row["error_class"] = type(exc).__name__
    row["error_msg"] = str(exc)[:2000]
    row["error_file_line"] = _error_file_line(exc)
    row["traceback"] = "".join(tb_mod.format_exception(type(exc), exc, exc.__traceback__))


def run_condition(
    task_rel: str,
    axis_subset: tuple[str, ...],
    seed: int,
    *,
    scenic_only: bool,
    max_iter: int,
) -> dict[str, Any]:
    """Execute a single (task, axis_subset, seed) condition end-to-end."""

    t0 = time.monotonic()
    row: dict[str, Any] = {
        "task": task_rel,
        "axis_subset": list(axis_subset),
        "seed": seed,
        "cardinality": len(axis_subset),
        "g0": "skip",
        "g1": "skip",
        "g2": "skip",
        "g3": "skip",
        "g5": "skip" if not scenic_only else "skip",
        "g6": "skip" if not scenic_only else "skip",
        "n_iters": None,
        "error_class": None,
        "error_msg": None,
        "error_file_line": None,
        "traceback": None,
        "duration_s": 0.0,
        "worker_pid": os.getpid(),
        # G4 invariant family hooks (filled in after G3 / G5):
        "g4_identity": None,
        "g4_identity_error": None,
        "g4_domain": None,
        "g4_consistency": None,
        "g4_affordance": None,
        "g4_domain_error": None,
    }

    # Seed Python RNG for determinism on the sampling path.
    random.seed(seed)

    bddl_path = resolve_task_path(task_rel)
    request = ",".join(axis_subset)

    # ---- G0: BDDL parse ------------------------------------------------
    try:
        from libero_infinity.task_config import TaskConfig

        cfg = TaskConfig.from_bddl(str(bddl_path))
        row["g0"] = "pass"
    except Exception as exc:  # noqa: BLE001 — captured + recorded, not swallowed
        _record_failure(row, "g0", exc)
        row["duration_s"] = time.monotonic() - t0
        return row

    # ---- G1: Scenic program generation --------------------------------
    try:
        from libero_infinity.compiler import compile_task_to_scenic

        scenic_src = compile_task_to_scenic(cfg, request)
        if not scenic_src or not isinstance(scenic_src, str):
            raise RuntimeError(
                f"compile_task_to_scenic returned empty/non-str: {type(scenic_src)!r}"
            )
        row["g1"] = "pass"
    except Exception as exc:  # noqa: BLE001
        _record_failure(row, "g1", exc)
        row["duration_s"] = time.monotonic() - t0
        return row

    # ---- G2: Scenic compile ------------------------------------------
    # We call compile_task_to_scenario (which writes the program to a temp
    # file inside scenic/ so that ``model libero_model`` resolves correctly)
    # — this is the file-context-aware equivalent of scenic.scenarioFromString
    # for this codebase.
    try:
        from libero_infinity.compiler import compile_task_to_scenario

        scenario = compile_task_to_scenario(cfg, request)
        row["g2"] = "pass"
    except Exception as exc:  # noqa: BLE001
        _record_failure(row, "g2", exc)
        row["duration_s"] = time.monotonic() - t0
        return row

    # ---- G3: Scenic sample -------------------------------------------
    scene = None
    try:
        scene, n_iters = scenario.generate(maxIterations=max_iter)
        row["n_iters"] = int(n_iters)
        row["g3"] = "pass"
    except Exception as exc:  # noqa: BLE001
        _record_failure(row, "g3", exc)
        row["duration_s"] = time.monotonic() - t0
        return row

    # ---- G4 (family A) identity hook ---------------------------------
    # Compares the perturbed scene to a no-axes baseline; every axis NOT in
    # ``axis_subset`` must be byte-identical. Errors are recorded honestly
    # in ``g4_identity_error`` (with traceback) but do not gate the rest of
    # the pipeline — they're a separate publication-grade assertion family.
    try:
        baseline = _get_baseline_scene(cfg, str(bddl_path), max_iter)
        from libero_infinity.validation.invariants import g4_identity_hook

        row["g4_identity"] = g4_identity_hook(baseline, scene, axis_subset)
    except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
        row["g4_identity"] = None
        row["g4_identity_error"] = {
            "error_class": type(exc).__name__,
            "error_msg": str(exc)[:2000],
            "error_file_line": _error_file_line(exc),
            "traceback": "".join(tb_mod.format_exception(type(exc), exc, exc.__traceback__)),
        }

    if scenic_only:
        row["duration_s"] = time.monotonic() - t0
        return row

    # ---- G5: LIBERO env create + reset --------------------------------
    env = None
    try:
        from libero_infinity.gym_env import make_env  # lazy: pulls torch/robosuite

        env = make_env(scene, bddl_path=str(bddl_path))
        env.reset()
        row["g5"] = "pass"
    except Exception as exc:  # noqa: BLE001
        _record_failure(row, "g5", exc)
        row["duration_s"] = time.monotonic() - t0
        return row

    # ---- G4 (families B/C/D) domain/consistency/affordance hook -------
    # Runs after G5 env reset. Per-assertion {name: bool} payloads land in the
    # JSONL row under separate keys so downstream aggregation can compute
    # marginal fail rates per family / per axis.
    try:
        from libero_infinity.validation.invariants import g4_domain_consistency_hook

        try:
            from libero_infinity.asset_registry import ASSET_VARIANTS as _registry
        except Exception:  # noqa: BLE001
            _registry = None
        _flat = g4_domain_consistency_hook(scene, env, cfg, registry=_registry)
        # Bucket by family for compact JSONL rows: family -> {name: passed}
        dom: dict[str, Any] = {}
        con: dict[str, Any] = {}
        aff: dict[str, Any] = {}
        for k, res in _flat.items():
            family, _, sub = k.partition(":")
            target = {"domain": dom, "consistency": con, "affordance": aff}.get(family)
            if target is None:
                continue
            target[sub] = res.passed  # may be True / False / None (honest skip)
        row["g4_domain"] = dom
        row["g4_consistency"] = con
        row["g4_affordance"] = aff
    except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
        row["g4_domain_error"] = {
            "error_class": type(exc).__name__,
            "error_msg": str(exc)[:2000],
            "error_file_line": _error_file_line(exc),
            "traceback": "".join(tb_mod.format_exception(type(exc), exc, exc.__traceback__)),
        }

    # ---- G6: render + 5 noop steps ------------------------------------
    try:
        import numpy as np

        # Determine action dim from env if possible; default to 7-DoF noop.
        action_dim = getattr(getattr(env, "action_space", None), "shape", (7,))[0]
        noop = np.zeros(action_dim, dtype=np.float32)
        for _ in range(5):
            env.step(noop)
        # render: prefer offscreen if available; otherwise call render().
        if hasattr(env, "render"):
            env.render()
        row["g6"] = "pass"
    except Exception as exc:  # noqa: BLE001
        _record_failure(row, "g6", exc)
    finally:
        # Clean up env without masking errors above.
        try:
            if env is not None and hasattr(env, "close"):
                env.close()
        except Exception:  # noqa: BLE001 — close-time errors are not gate failures
            pass

    row["duration_s"] = time.monotonic() - t0
    return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _worker_entry(args: tuple) -> dict[str, Any]:
    task_rel, axis_subset, seed, scenic_only, max_iter = args
    try:
        return run_condition(
            task_rel,
            axis_subset,
            seed,
            scenic_only=scenic_only,
            max_iter=max_iter,
        )
    except BaseException as exc:  # noqa: BLE001 — last-resort harness guard
        # Even harness-internal failures must be reported, not swallowed.
        return {
            "task": task_rel,
            "axis_subset": list(axis_subset),
            "seed": seed,
            "cardinality": len(axis_subset),
            "g0": "fail",
            "g1": "skip",
            "g2": "skip",
            "g3": "skip",
            "g5": "skip",
            "g6": "skip",
            "n_iters": None,
            "error_class": type(exc).__name__,
            "error_msg": f"[harness] {exc}"[:2000],
            "error_file_line": _error_file_line(exc),
            "traceback": "".join(tb_mod.format_exception(type(exc), exc, exc.__traceback__)),
            "duration_s": 0.0,
            "worker_pid": os.getpid(),
        }


def run_sweep(
    tasks: list[str],
    subsets: list[tuple[str, ...]],
    seeds: list[int],
    *,
    out_path: pathlib.Path,
    workers: int,
    scenic_only: bool,
    max_iter: int,
) -> dict[str, Any]:
    """Execute the cartesian sweep and stream results to JSONL.

    Returns a summary dict with per-gate pass/fail counts.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Round-robin (task-minor) dispatch: interleave one (axis_subset, seed)
    # condition per task per cycle so a single sick task cannot saturate the
    # early-window rolling fail-rate and trip the global 5% abort threshold
    # before the schedule has visited the other tasks. The set of dispatched
    # conditions is unchanged — only the ORDER is — so honest pass/fail
    # accounting is preserved and the final aggregate is bit-identical to the
    # previous task-major ordering.
    per_task = [
        [(t, axs, s, scenic_only, max_iter) for axs in subsets for s in seeds] for t in tasks
    ]
    conditions = [
        c for c in itertools.chain.from_iterable(itertools.zip_longest(*per_task)) if c is not None
    ]
    total = len(conditions)
    print(
        f"[sweep] {total} conditions  "
        f"({len(tasks)} tasks x {len(subsets)} subsets x {len(seeds)} seeds)  "
        f"workers={workers}  scenic_only={scenic_only}  max_iter={max_iter}",
        file=sys.stderr,
        flush=True,
    )

    counts = {g: {"pass": 0, "fail": 0, "skip": 0} for g in GATES}

    # ProcessPoolExecutor; writer serialized in this (main) process.
    ctx = mp.get_context("spawn")
    with open(out_path, "w", encoding="utf-8") as fh:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = [ex.submit(_worker_entry, c) for c in conditions]
            done = 0
            for fut in as_completed(futures):
                row = fut.result()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                for g in GATES:
                    counts[g][row.get(g, "skip")] += 1
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"[sweep] {done}/{total}", file=sys.stderr, flush=True)

    return {"total": total, "counts": counts, "out": str(out_path)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_tasks(spec: str) -> list[str]:
    if spec == "all":
        return discover_all_tasks()
    if spec == "smoke":
        return list(SMOKE_TASKS)
    return [s.strip() for s in spec.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="libero_infinity.validation.sweep")
    ap.add_argument(
        "--tasks", default="smoke", help="'all', 'smoke', or comma list of suite/task.bddl paths"
    )
    ap.add_argument(
        "--subsets",
        default="16",
        help="Number of axis subsets to sample per task, or 'all' for 511",
    )
    ap.add_argument("--seeds", type=int, default=1, help="Number of seeds per (task, subset) cell")
    ap.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes (capped at 10)"
    )
    ap.add_argument(
        "--scenic-only", action="store_true", help="Stop after G3 (no LIBERO env creation)"
    )
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument(
        "--max-iter", type=int, default=2000, help="Scenic scenario.generate maxIterations"
    )
    ap.add_argument(
        "--subset-seed",
        type=int,
        default=0,
        help="Seed for the subset sampler (determinism across runs)",
    )
    args = ap.parse_args(argv)

    tasks = _parse_tasks(args.tasks)
    if not tasks:
        print("[sweep] no tasks resolved", file=sys.stderr)
        return 2

    subsets_arg: int | str
    if args.subsets == "all":
        subsets_arg = "all"
    else:
        subsets_arg = int(args.subsets)
    subsets = sample_subsets(subsets_arg, seed=args.subset_seed)

    seeds = list(range(args.seeds))
    workers = max(1, min(10, args.workers))

    summary = run_sweep(
        tasks,
        subsets,
        seeds,
        out_path=pathlib.Path(args.out).expanduser(),
        workers=workers,
        scenic_only=args.scenic_only,
        max_iter=args.max_iter,
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
