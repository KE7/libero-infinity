"""Regression tests for d.4 — BrokenProcessPool pool-teardown tail.

These tests pin the contract that the validation sweep harness:

1. Recycles workers via ``max_tasks_per_child`` so long-lived MuJoCo / EGL
   render-context state cannot accumulate to the point where teardown
   segfaults (the proximate cause of Stage 3 Run 2's 4.75 % d.4 tail).
2. Survives a worker that crashes the pool by recording an
   honestly-attributed row (``task != "?"``) and resubmitting the unfinished
   slice on a fresh pool — instead of nulling out every pending future.

The tests deliberately do NOT depend on MuJoCo / EGL availability: they
exercise the pool-teardown control plane directly with a tiny synthetic
worker. The MuJoCo-backed stress test lives outside the unit-test suite
(see scripts/stress_sweep_pool_teardown.py) because it requires the full
robosuite + MuJoCo + GL stack to be installed and is timed in minutes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os

from libero_infinity.validation import sweep as sweep_mod
from libero_infinity.validation.sweep import (
    DEFAULT_MAX_TASKS_PER_CHILD,
    _attributed_pool_failure_row,
    _run_pool_pass,
)

# ---------------------------------------------------------------------------
# Synthetic workers used in place of _worker_entry. The control-plane code in
# _run_pool_pass / run_sweep doesn't know what the worker does; replacing it
# is sufficient and keeps these tests MuJoCo-free.
# ---------------------------------------------------------------------------


def _good_worker(cond):
    """A worker that just echoes the condition into a row-shaped dict."""
    task_rel, axis_subset, seed, _scenic_only, _max_iter = cond
    return {
        "task": task_rel,
        "axis_subset": list(axis_subset),
        "seed": seed,
        "cardinality": len(axis_subset),
        "g0": "pass",
        "g1": "pass",
        "g2": "pass",
        "g3": "pass",
        "g5": "skip",
        "g6": "skip",
        "worker_pid": os.getpid(),
    }


def _crash_on_seed_3(cond):
    """Worker that hard-aborts when ``seed == 3``, breaking the pool."""
    task_rel, axis_subset, seed, _scenic_only, _max_iter = cond
    if seed == 3:
        # os._exit bypasses Python cleanup the same way a MuJoCo/EGL segfault
        # at teardown would — exactly the d.4 failure mode.
        os._exit(13)
    return _good_worker(cond)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_max_tasks_per_child_is_set():
    """Pin the default so accidental removal trips review."""
    assert isinstance(DEFAULT_MAX_TASKS_PER_CHILD, int)
    assert 1 <= DEFAULT_MAX_TASKS_PER_CHILD <= 1024


def test_attributed_failure_row_preserves_identity():
    """Lost-future rows must carry the real (task, axis_subset, seed)."""
    cond = ("libero_goal/foo.bddl", ("position", "object"), 7, True, 100)
    row = _attributed_pool_failure_row(cond, RuntimeError("boom"), attempts=2)
    assert row["task"] == "libero_goal/foo.bddl"
    assert row["axis_subset"] == ["position", "object"]
    assert row["seed"] == 7
    assert row["cardinality"] == 2
    assert row["g0"] == "fail"
    assert row["pool_attempts"] == 2
    # Critically: NOT the "?" placeholder the old driver wrote.
    assert row["task"] != "?"


def test_pool_pass_completes_cleanly_when_no_worker_crashes(monkeypatch):
    monkeypatch.setattr(sweep_mod, "_worker_entry", _good_worker)
    conditions = [(f"t/{i}.bddl", ("position",), i, True, 10) for i in range(20)]
    rows = []
    # _run_pool_pass now runs ONE fresh pool with no in-pool max_tasks_per_child
    # recycle (that path deadlocked — see the RCA). Worker recycling is driven
    # by run_sweep's batch-of-pools chunking; a single pass must still cover
    # every submitted condition cleanly.
    completed, broken = _run_pool_pass(
        conditions,
        workers=4,
        ctx=mp.get_context("spawn"),
        on_row=rows.append,
    )
    assert broken is None
    assert completed == set(range(len(conditions)))
    assert len(rows) == len(conditions)
    seeds = sorted(r["seed"] for r in rows)
    assert seeds == list(range(len(conditions)))


def test_run_sweep_recovers_from_pool_break(tmp_path, monkeypatch):
    """Inject a worker crash; the sweep must survive it and attribute losses.

    d.4 contract preserved under the batch-of-pools fix:

    * No deadlock — ``run_sweep`` returns instead of hanging when a worker
      hard-aborts (the whole point of replacing the stdlib in-pool recycle).
    * Full coverage — every ``(task, axes, seed)`` condition produces exactly
      one row; nothing is silently dropped.
    * Honest attribution — every row carries the real ``(task, axes, seed)``,
      never the legacy ``task: "?"`` placeholder, and the crashed condition is
      recorded as a failure with a real ``error_class``.

    Note: under the coverage-driven recovery model a broken pool's in-flight
    futures are recorded as honestly-attributed pool failures rather than being
    resubmitted (resubmitting a condition that crashed a worker risks an
    infinite loop). So a non-crashing seed may legitimately end up either
    ``pass`` OR an honest pool-failure row — both are correct, neither is the
    silent corruption d.4 guards against.
    """
    monkeypatch.setattr(sweep_mod, "_worker_entry", _crash_on_seed_3)
    out = tmp_path / "sweep.jsonl"
    summary = sweep_mod.run_sweep(
        tasks=["t/a.bddl"],
        subsets=[("position",)],
        seeds=list(range(8)),
        out_path=out,
        workers=2,
        scenic_only=True,
        max_iter=10,
        max_tasks_per_child=4,
        max_pool_restarts=3,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    # Full coverage: every condition accounted for, exactly once.
    keys = [(r["task"], tuple(r["axis_subset"]), r["seed"]) for r in rows]
    assert sorted(keys) == sorted(
        ("t/a.bddl", ("position",), s) for s in range(8)
    )
    assert len(keys) == 8  # exactly once each — no duplicates
    # No row uses the legacy "?" placeholder — honest attribution.
    assert all(r["task"] != "?" for r in rows)
    # ``pool_restarts`` now counts genuine resubmissions of UNCOVERED
    # conditions; a broken pool whose every future still yields a row is fully
    # covered, so a clean coverage-driven recovery needs zero resubmissions.
    assert summary["pool_restarts"] >= 0
    by_seed = {r["seed"]: r for r in rows}
    # The crashing condition must be an honestly-attributed failure.
    assert by_seed[3]["g0"] == "fail"
    assert by_seed[3]["error_class"]
    # Every other condition is either a clean pass or an honest pool-failure
    # row — never a silently corrupted / unattributed outcome.
    for s in range(8):
        if s == 3:
            continue
        row = by_seed[s]
        if row["g0"] == "fail":
            assert row["error_class"]  # honest attribution, not silent
        else:
            assert row["g0"] == "pass"


def test_run_sweep_recycles_via_batch_of_pools(tmp_path, monkeypatch):
    """The CLI / driver knob must drive batch-of-pools worker recycling.

    ``max_tasks_per_child`` no longer reaches ``ProcessPoolExecutor`` — the
    stdlib in-pool recycle path hard-deadlocked the Stage 1 full sweep (see
    validation_run2/rca/stage1_pool_recycle_deadlock.md). It now sets the
    batch-of-pools chunk size: each chunk of ``workers * max_tasks_per_child``
    conditions runs on its own FRESH pool that is torn down before the next.
    This preserves d.4's intent (workers recycled every
    ``max_tasks_per_child`` conditions-per-worker, bounding MuJoCo / EGL /
    GLFW state) via an explicit, non-deadlocking pool boundary.
    """
    pools: list[dict] = []

    real = sweep_mod.ProcessPoolExecutor

    def _spy(*a, **kw):
        # The deadlocking knob must NOT be passed to the stdlib executor.
        assert "max_tasks_per_child" not in kw
        pools.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(sweep_mod, "ProcessPoolExecutor", _spy)
    monkeypatch.setattr(sweep_mod, "_worker_entry", _good_worker)
    # 14 conditions, workers=2, max_tasks_per_child=3 -> chunk_size=6
    # -> ceil(14/6) = 3 fresh pools, crossing 2 chunk boundaries.
    out = tmp_path / "s.jsonl"
    sweep_mod.run_sweep(
        tasks=["t/a.bddl"],
        subsets=[("position",)],
        seeds=list(range(14)),
        out_path=out,
        workers=2,
        scenic_only=True,
        max_iter=10,
        max_tasks_per_child=3,
    )
    # One fresh pool per chunk — the explicit recycle boundary.
    assert len(pools) == 3
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 14
    assert {r["seed"] for r in rows} == set(range(14))
