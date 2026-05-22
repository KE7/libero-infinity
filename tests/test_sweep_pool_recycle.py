"""Regression tests for the Stage 1 pool-recycle DEADLOCK.

Root cause (RCA: validation_run2/rca/stage1_pool_recycle_deadlock.md):
the CPython ``spawn`` + ``ProcessPoolExecutor(max_tasks_per_child=N)`` in-pool
worker-recycle path hard-deadlocks at the ``workers * max_tasks_per_child``
boundary — when the whole worker cohort retires in the same tick the
``_ExecutorManagerThread`` never spawns the replacement cohort, and no
``BrokenProcessPool`` is raised, so the sweep hangs forever with no error.

The fix replaces in-pool recycling with **batch-of-pools** chunking:
``run_sweep`` partitions the condition list into chunks of
``workers * max_tasks_per_child`` and runs each chunk on its own FRESH
``ProcessPoolExecutor`` (created WITHOUT ``max_tasks_per_child``) which is
torn down before the next chunk. Every worker process is therefore destroyed
every ``max_tasks_per_child`` conditions-per-worker — the same d.4 guarantee,
via a deterministic pool boundary instead of the deadlocking stdlib path.

These tests exercise the recycle control plane across >= 2 chunk boundaries
with a TRIVIAL synthetic worker (no libero / MuJoCo / GL stack), so they are
fast and hermetic. If the deadlock regresses, these tests hang rather than
fail — pytest's per-test timeout (or CI wall clock) surfaces it.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os

from libero_infinity.validation import sweep as sweep_mod
from libero_infinity.validation.sweep import (
    GATES,
    _run_pool_pass,
    chunk_conditions,
    run_sweep,
)

# ---------------------------------------------------------------------------
# Trivial, MuJoCo-free worker. The recycle control plane in run_sweep /
# _run_pool_pass does not care what the worker does; a synthetic one keeps
# these tests hermetic and sub-second.
# ---------------------------------------------------------------------------


def _trivial_worker(cond):
    task_rel, axis_subset, seed, _scenic_only, _max_iter = cond
    row = {
        "task": task_rel,
        "axis_subset": list(axis_subset),
        "seed": seed,
        "cardinality": len(axis_subset),
        "worker_pid": os.getpid(),
    }
    for g in GATES:
        row[g] = "pass"
    return row


# ---------------------------------------------------------------------------
# chunk_conditions — the partition primitive
# ---------------------------------------------------------------------------


def test_chunk_conditions_partitions_contiguously():
    conds = list(range(20))
    chunks = chunk_conditions(conds, 8)
    # 20 conditions / chunk_size 8 -> 2.5 batches -> 3 chunks (8, 8, 4).
    assert [len(c) for c in chunks] == [8, 8, 4]
    # Contiguous and lossless: concatenation reproduces the input exactly.
    assert [x for c in chunks for x in c] == conds


def test_chunk_conditions_rejects_bad_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_conditions([1, 2, 3], 0)


# ---------------------------------------------------------------------------
# _run_pool_pass — one fresh pool, no in-pool max_tasks_per_child
# ---------------------------------------------------------------------------


def test_run_pool_pass_runs_a_fresh_pool_without_max_tasks_per_child(monkeypatch):
    """A single pass on a fresh pool must cover every condition with no recycle
    knob — and must NOT deadlock."""
    monkeypatch.setattr(sweep_mod, "_worker_entry", _trivial_worker)
    conditions = [(f"t/{i}.bddl", ("position",), i, True, 10) for i in range(20)]
    rows: list[dict] = []
    completed, broken = _run_pool_pass(
        conditions,
        workers=2,
        ctx=mp.get_context("spawn"),
        on_row=rows.append,
    )
    assert broken is None
    assert completed == set(range(20))
    assert len(rows) == 20


# ---------------------------------------------------------------------------
# run_sweep — full batch-of-pools driver across >= 2 chunk boundaries
# ---------------------------------------------------------------------------


def test_run_sweep_crosses_chunk_boundaries_without_deadlock(tmp_path, monkeypatch):
    """End-to-end recycle test: 20 conditions, workers=2, max_tasks_per_child=4.

    chunk_size = workers * max_tasks_per_child = 8, so 20 conditions span
    20 / 8 = 2.5 batches -> 3 fresh pools, crossing 2 recycle boundaries.

    Asserts:
      (a) no deadlock — the test simply COMPLETES (a hang fails via timeout);
      (b) completed == submitted — zero dropped conditions;
      (c) every condition's result row is present exactly once.
    """
    monkeypatch.setattr(sweep_mod, "_worker_entry", _trivial_worker)
    out = tmp_path / "recycle.jsonl"

    seeds = list(range(20))
    summary = run_sweep(
        tasks=["t/a.bddl"],
        subsets=[("position",)],
        seeds=seeds,
        out_path=out,
        workers=2,
        scenic_only=True,
        max_iter=10,
        max_tasks_per_child=4,  # chunk_size = 2 * 4 = 8 -> 3 chunks for 20
    )

    # (a) No deadlock: reaching here means the driver returned. Also no
    # pool restarts were needed for a clean trivial worker.
    assert summary["total"] == 20
    assert summary["pool_restarts"] == 0

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]

    # (b) completed == submitted: exactly one row per submitted condition,
    # none lost to silent truncation.
    assert len(rows) == 20

    # (c) every condition's result row is present, exactly once, with no
    # legacy "?" placeholder and no pool-failure attribution.
    seen_seeds = sorted(r["seed"] for r in rows)
    assert seen_seeds == seeds
    assert all(r["task"] == "t/a.bddl" for r in rows)
    assert all(r["task"] != "?" for r in rows)
    assert all("pool_attempts" not in r for r in rows)
    # The trivial worker passes every gate — recovery path did not corrupt it.
    assert all(r["g0"] == "pass" for r in rows)


def test_run_sweep_one_fresh_pool_per_chunk(tmp_path, monkeypatch):
    """Pin the recycle MECHANISM: one fresh ProcessPoolExecutor per chunk, and
    max_tasks_per_child is never handed to the deadlocking stdlib path."""
    monkeypatch.setattr(sweep_mod, "_worker_entry", _trivial_worker)

    pools: list[dict] = []
    real = sweep_mod.ProcessPoolExecutor

    def _spy(*a, **kw):
        assert "max_tasks_per_child" not in kw  # the deadlocking knob
        pools.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(sweep_mod, "ProcessPoolExecutor", _spy)

    out = tmp_path / "recycle.jsonl"
    run_sweep(
        tasks=["t/a.bddl"],
        subsets=[("position",)],
        seeds=list(range(20)),
        out_path=out,
        workers=2,
        scenic_only=True,
        max_iter=10,
        max_tasks_per_child=4,  # chunk_size 8 -> 3 chunks -> 3 fresh pools
    )
    assert len(pools) == 3
