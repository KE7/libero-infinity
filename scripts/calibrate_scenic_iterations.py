#!/usr/bin/env python
"""Calibrate per-perturbation-mode Scenic iteration budgets (WS-3).

For each perturbation mode we measure the distribution of ``n_iters`` returned
by ``Scenario.generate()`` (the number of rejection-sampling iterations needed
to find ONE valid scene) across a diverse task corpus, then derive a budget
that covers the tail of that distribution with margin.

Methodology
-----------
Scenic's scene generator is a rejection sampler: each ``generate()`` call draws
candidate scenes until one satisfies every (hard) ``require`` constraint, and
returns how many iterations that took. For a mode/task with per-iteration
success probability ``p`` the count is ~Geometric(p), so the tail is
``P(n > N) = (1-p)^N``. Two complementary tail estimators are combined and the
larger is taken (robust to small-sample noise *and* to heavy mixture tails):

  1. Empirical: the pooled ``p99`` of observed ``n_iters``.
  2. Geometric model: ``mean * ln(1/(1-target))`` with ``target = 0.999``
     (factor ~6.9), estimated from the pooled mean. This extrapolates the tail
     past the largest sample we actually drew.

The recommended budget is ``ceil_round(max(p99, geom_999) * SAFETY)`` with a
floor of 5000 (back-compat) — rounded to a human-legible step.

Run (background, ~30-45 min):
    MUJOCO_GL=egl PYTHONPATH=src python scripts/calibrate_scenic_iterations.py
Writes: src/libero_infinity/data/scenic_iteration_budgets.json
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import time

from libero_infinity.compiler import compile_task_to_scenario
from libero_infinity.task_config import TaskConfig

REPO = pathlib.Path(__file__).resolve().parent.parent
BDDL_ROOT = REPO / "src/libero_infinity/data/libero_runtime/bddl_files"
OUT = REPO / "src/libero_infinity/data/scenic_iteration_budgets.json"

# Diverse corpus: flat placement, articulation+containment, basket pick, spatial
# reference, long-horizon kitchen.
CORPUS = [
    "libero_goal/put_the_bowl_on_the_plate.bddl",
    "libero_goal/open_the_top_drawer_and_put_the_bowl_inside.bddl",
    "libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl",
    "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl",
    "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
]

# (mode, n_tasks_used, samples_per_task). Cheap single axes get more samples;
# the heavy presets get fewer (each generate() can cost ~15s).
MODE_PLAN = [
    ("position", 5, 25),
    ("object", 5, 25),
    ("camera", 5, 25),
    ("lighting", 5, 25),
    ("distractor", 5, 20),
    ("background", 5, 20),
    ("robot", 5, 20),
    ("combined", 4, 18),
    ("full", 4, 16),
]

CAP = 300_000          # high enough to observe the real tail
SAFETY = 1.30          # margin over the tail estimate
TARGET_COVERAGE = 0.999


def ceil_round(x: float) -> int:
    """Round up to a human-legible step that grows with magnitude."""
    if x <= 5_000:
        return 5_000
    if x < 10_000:
        step = 1_000
    elif x < 100_000:
        step = 5_000
    else:
        step = 25_000
    return int(math.ceil(x / step) * step)


def pct(data: list[int], q: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    if len(s) == 1:
        return float(s[0])
    idx = q * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def main() -> None:
    results: dict = {}
    geom_factor = math.log(1.0 / (1.0 - TARGET_COVERAGE))  # ~6.908

    for mode, n_tasks, n_samples in MODE_PLAN:
        pooled: list[int] = []
        per_task: dict[str, dict] = {}
        for bddl_rel in CORPUS[:n_tasks]:
            bddl = BDDL_ROOT / bddl_rel
            cfg = TaskConfig.from_bddl(str(bddl))
            scenario = compile_task_to_scenario(
                cfg, mode, params={"bddl_path": str(bddl)}
            )
            iters: list[int] = []
            t0 = time.monotonic()
            for _ in range(n_samples):
                try:
                    _scene, n = scenario.generate(maxIterations=CAP, verbosity=0)
                    iters.append(int(n))
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! {mode} {bddl_rel}: generate failed: {exc}", flush=True)
            dt = time.monotonic() - t0
            pooled.extend(iters)
            per_task[bddl_rel] = {
                "n": len(iters),
                "mean": statistics.mean(iters) if iters else None,
                "max": max(iters) if iters else None,
                "wall_s": round(dt, 1),
            }
            print(
                f"  {mode:11s} {bddl_rel:65s} n={len(iters):3d} "
                f"mean={per_task[bddl_rel]['mean']!s:>8.8} max={max(iters) if iters else 0:>7d} "
                f"wall={dt:6.1f}s",
                flush=True,
            )

        mean = statistics.mean(pooled) if pooled else 0.0
        p99 = pct(pooled, 0.99)
        geom_999 = mean * geom_factor
        budget = ceil_round(max(p99, geom_999) * SAFETY)
        results[mode] = {
            "n_total": len(pooled),
            "mean": round(mean, 1),
            "p50": round(pct(pooled, 0.50), 1),
            "p90": round(pct(pooled, 0.90), 1),
            "p95": round(pct(pooled, 0.95), 1),
            "p99": round(p99, 1),
            "max": max(pooled) if pooled else 0,
            "geom_999": round(geom_999, 1),
            "budget": budget,
            "per_task": per_task,
        }
        print(
            f"==> {mode}: mean={mean:.0f} p99={p99:.0f} geom999={geom_999:.0f} "
            f"=> BUDGET={budget}",
            flush=True,
        )

    artifact = {
        "_meta": {
            "description": "WS-3 task-adaptive Scenic iteration budgets. "
            "budget = ceil_round(max(p99, mean*ln(1/(1-0.999))) * 1.30), floor 5000.",
            "cap": CAP,
            "safety": SAFETY,
            "target_coverage": TARGET_COVERAGE,
            "corpus": CORPUS,
            "generator": "scripts/calibrate_scenic_iterations.py",
        },
        "modes": results,
    }
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nWROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
