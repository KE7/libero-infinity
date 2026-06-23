"""Regression tests for the g3 sample-budget fix (R1 + R3b) in the sweep.

R1: the sweep's G3 Scenic sampling stage must route its ``maxIterations`` through
the SAME ``resolve_iteration_budget`` the eval/gym path uses (so the sweep and
eval agree by construction), with an explicit ``--max-iter`` still overriding.

R3b: a ``RejectionException`` from ``generate`` triggers a BOUNDED resample
retry (mirroring the eval reset() settle-loop), recorded in ``g3_attempts`` —
never an unbounded loop.

These tests are hermetic: they stub the BDDL parse / Scenic compile so no LIBERO
/ MuJoCo / GL stack is touched, and ``scenic_only=True`` stops the pipeline after
the G4 identity hook.
"""

from __future__ import annotations

import libero_infinity.compiler as _compiler
import libero_infinity.task_config as _task_config
import libero_infinity.validation.invariants as _invariants
from libero_infinity.scenic_budget import resolve_iteration_budget
from libero_infinity.validation import sweep as sweep_mod
from libero_infinity.validation.sweep import G3_RESAMPLE_RETRIES, run_condition

try:
    from scenic.core.distributions import RejectionException
except Exception:  # pragma: no cover
    RejectionException = RuntimeError  # type: ignore[misc,assignment]


class _FakeScenario:
    """Records every ``maxIterations`` it is asked to generate at."""

    def __init__(self, captured: list[int], fail_times: int = 0) -> None:
        self._captured = captured
        self._fail_times = fail_times
        self._calls = 0

    def generate(self, maxIterations=None, **_kw):  # noqa: N803 — Scenic API name
        self._captured.append(maxIterations)
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RejectionException("synthetic rejection")
        return (object(), 7)


def _install_stubs(monkeypatch, fail_times: int = 0) -> dict[str, list[int]]:
    """Stub the BDDL/Scenic/G4 layer; return the captured maxIterations list.

    Only the FIRST compiled scenario (the G3 perturbed sample) uses ``fail_times``;
    the baseline scenario (G4 identity, empty request) never fails.
    """
    g3_caps: list[int] = []
    state = {"compiled": 0}

    monkeypatch.setattr(_task_config.TaskConfig, "from_bddl", staticmethod(lambda _p: object()))
    monkeypatch.setattr(_compiler, "compile_task_to_scenic", lambda _cfg, _req: "scenic-src")

    def _fake_compile_scenario(_cfg, request):
        # The G3 sample is compiled first (non-empty request); the baseline G4
        # scene is compiled second with an empty request.
        if request:
            return _FakeScenario(g3_caps, fail_times=fail_times)
        return _FakeScenario([], fail_times=0)

    monkeypatch.setattr(_compiler, "compile_task_to_scenario", _fake_compile_scenario)
    monkeypatch.setattr(_invariants, "g4_identity_hook", lambda *_a, **_k: {})
    # Avoid touching the real BDDL tree for the baseline cache key / path.
    monkeypatch.setattr(sweep_mod, "resolve_task_path", lambda t: t)
    sweep_mod._BASELINE_CACHE.clear()
    return {"g3": g3_caps, "_state": state}


def test_sweep_routes_g3_budget_through_resolver(monkeypatch):
    # No explicit --max-iter (None) → budget resolved per (task, subset) by the
    # SAME resolver as eval. ``position,robot,distractor`` must get the large
    # combined budget (R2), proving sweep+eval agreement by construction.
    caps = _install_stubs(monkeypatch)
    subset = ("position", "robot", "distractor")
    row = run_condition("t.bddl", subset, 0, scenic_only=True, max_iter=None)

    expected = resolve_iteration_budget(",".join(subset), None)
    assert row["g3"] == "pass"
    assert row["max_iter_resolved"] == expected
    assert caps["g3"][0] == expected  # the budget actually handed to generate()
    assert expected == resolve_iteration_budget("combined")  # the resolver-gap fix


def test_sweep_explicit_max_iter_overrides_resolver(monkeypatch):
    caps = _install_stubs(monkeypatch)
    subset = ("position", "robot", "distractor")
    row = run_condition("t.bddl", subset, 0, scenic_only=True, max_iter=1234)

    assert row["g3"] == "pass"
    assert row["max_iter_resolved"] == 1234
    assert caps["g3"][0] == 1234


def test_sweep_cheap_subset_does_not_balloon(monkeypatch):
    caps = _install_stubs(monkeypatch)
    row = run_condition("t.bddl", ("position",), 0, scenic_only=True, max_iter=None)
    assert row["max_iter_resolved"] == 5000
    assert caps["g3"][0] == 5000


def test_sweep_resamples_on_rejection_then_passes(monkeypatch):
    # First draw raises RejectionException; the bounded retry re-draws and the
    # second draw succeeds — recorded in g3_attempts.
    caps = _install_stubs(monkeypatch, fail_times=1)
    row = run_condition(
        "t.bddl", ("position", "robot", "distractor"), 0, scenic_only=True, max_iter=None
    )
    assert row["g3"] == "pass"
    assert row["g3_attempts"] == 2  # one failed draw + one success
    assert len(caps["g3"]) == 2


def test_sweep_resample_is_bounded(monkeypatch):
    # Every draw rejects → the retry is bounded at G3_RESAMPLE_RETRIES + 1 draws
    # and then records an HONEST g3 failure (no unbounded loop, no masking).
    caps = _install_stubs(monkeypatch, fail_times=10_000)
    row = run_condition(
        "t.bddl", ("position", "robot", "distractor"), 0, scenic_only=True, max_iter=None
    )
    assert row["g3"] == "fail"
    assert row["g3_attempts"] == G3_RESAMPLE_RETRIES + 1
    assert len(caps["g3"]) == G3_RESAMPLE_RETRIES + 1
    assert row["error_class"] == "RejectionException"
