#!/usr/bin/env python
"""Before/after demo: a budget-starved combined-mode task (WS-3).

Shows that the OLD global maxIterations=5000 starves a hard combined-mode scene
(RejectionException), while the NEW task-adaptive budget generates it.
"""

from __future__ import annotations

import pathlib

from libero_infinity.compiler import compile_task_to_scenario
from libero_infinity.scenic_budget import resolve_iteration_budget
from libero_infinity.task_config import TaskConfig

REPO = pathlib.Path(__file__).resolve().parent.parent
BDDL = (
    REPO
    / "src/libero_infinity/data/libero_runtime/bddl_files"
    / "libero_goal/open_the_top_drawer_and_put_the_bowl_inside.bddl"
)
MODE = "combined"
TRIALS = 12


def run(cap: int) -> tuple[int, list[int]]:
    cfg = TaskConfig.from_bddl(str(BDDL))
    scenario = compile_task_to_scenario(cfg, MODE, params={"bddl_path": str(BDDL)})
    ok = 0
    iters: list[int] = []
    for _ in range(TRIALS):
        try:
            _scene, n = scenario.generate(maxIterations=cap, verbosity=0)
            ok += 1
            iters.append(int(n))
        except Exception:  # noqa: BLE001 — RejectionException on budget exhaustion
            pass
    return ok, iters


def main() -> None:
    budget = resolve_iteration_budget(MODE)
    print(f"task: {BDDL.name}\nmode: {MODE}  trials: {TRIALS}")
    print(f"resolved adaptive budget for '{MODE}': {budget}\n")

    old_ok, _ = run(5000)
    print(f"OLD global cap=5000      -> {old_ok}/{TRIALS} scenes generated")

    new_ok, new_iters = run(budget)
    mx = max(new_iters) if new_iters else 0
    print(
        f"NEW adaptive cap={budget:<7} -> {new_ok}/{TRIALS} scenes generated (max iters used={mx})"
    )


if __name__ == "__main__":
    main()
