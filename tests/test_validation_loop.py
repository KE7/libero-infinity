"""Tests for the typed validation error classes and validation feedback loop.

Wave 5: runtime-validation. Covers:
- CollisionError, VisibilityError (the two active error types)
- All extend RuntimeError (backward compat with existing except RuntimeError handlers)
- InfeasibleScenarioError
- MAX_VISIBILITY_RETRIES constant
- RECOVERY_STRATEGY mapping
- Error attribute presence (object_names, invisible_names, etc.)

Note: AlreadySolvedError and SettleUnsafeError were removed as dead code.
      MAX_RESAMPLE and MAX_REPLAN were removed; use MAX_VISIBILITY_RETRIES.
"""

from __future__ import annotations

import pytest

from libero_infinity.validation_errors import (
    MAX_VISIBILITY_RETRIES,
    RECOVERY_STRATEGY,
    CollisionError,
    InfeasibleScenarioError,
    ScenarioValidationError,
    VisibilityError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_max_visibility_retries_is_at_least_5() -> None:
    """MAX_VISIBILITY_RETRIES must be at least 5."""
    assert MAX_VISIBILITY_RETRIES >= 5


def test_max_visibility_retries_is_10() -> None:
    """MAX_VISIBILITY_RETRIES must be exactly 10 per spec."""
    assert MAX_VISIBILITY_RETRIES == 10


# ---------------------------------------------------------------------------
# ScenarioValidationError base class
# ---------------------------------------------------------------------------


def test_scenario_validation_error_extends_runtime_error() -> None:
    """ScenarioValidationError must extend RuntimeError for backward compat."""
    assert issubclass(ScenarioValidationError, RuntimeError), (
        "ScenarioValidationError must extend RuntimeError so existing "
        "'except RuntimeError' handlers catch it"
    )


def test_scenario_validation_error_is_exception() -> None:
    """ScenarioValidationError must also be an Exception."""
    assert issubclass(ScenarioValidationError, Exception)


# ---------------------------------------------------------------------------
# CollisionError
# ---------------------------------------------------------------------------


def test_collision_error_extends_scenario_validation_error() -> None:
    assert issubclass(CollisionError, ScenarioValidationError)


def test_collision_error_extends_runtime_error() -> None:
    """CollisionError must be catchable as RuntimeError."""
    assert issubclass(CollisionError, RuntimeError)


def test_collision_error_carries_object_names() -> None:
    err = CollisionError("collision", object_names=["obj_a", "obj_b"])
    assert err.object_names == ["obj_a", "obj_b"]


def test_collision_error_default_object_names_empty() -> None:
    err = CollisionError("collision")
    assert err.object_names == []


def test_collision_error_message_preserved() -> None:
    err = CollisionError("test message")
    assert "test message" in str(err)


def test_collision_error_raised_and_caught_as_runtime_error() -> None:
    """CollisionError must be catchable in an 'except RuntimeError' block."""
    caught = False
    try:
        raise CollisionError("test", object_names=["foo"])
    except RuntimeError:
        caught = True
    assert caught, "CollisionError was not caught by 'except RuntimeError'"


# ---------------------------------------------------------------------------
# VisibilityError
# ---------------------------------------------------------------------------


def test_visibility_error_extends_scenario_validation_error() -> None:
    assert issubclass(VisibilityError, ScenarioValidationError)


def test_visibility_error_extends_runtime_error() -> None:
    assert issubclass(VisibilityError, RuntimeError)


def test_visibility_error_carries_invisible_names() -> None:
    err = VisibilityError("not visible", invisible_names=["obj_c"])
    assert err.invisible_names == ["obj_c"]


def test_visibility_error_default_invisible_names_empty() -> None:
    err = VisibilityError("not visible")
    assert err.invisible_names == []


def test_visibility_error_raised_and_caught_as_runtime_error() -> None:
    caught = False
    try:
        raise VisibilityError("test")
    except RuntimeError:
        caught = True
    assert caught


# ---------------------------------------------------------------------------
# InfeasibleScenarioError
# ---------------------------------------------------------------------------


def test_infeasible_scenario_error_is_exception() -> None:
    """InfeasibleScenarioError must be an Exception."""
    assert issubclass(InfeasibleScenarioError, Exception)


def test_infeasible_scenario_error_not_runtime_error() -> None:
    """InfeasibleScenarioError is NOT a ScenarioValidationError (it's a hard stop)."""
    assert not issubclass(InfeasibleScenarioError, ScenarioValidationError)


def test_infeasible_scenario_error_carries_n_resample() -> None:
    err = InfeasibleScenarioError("exhausted", n_resample=10, n_replan=0)
    assert err.n_resample == 10


def test_infeasible_scenario_error_carries_n_replan() -> None:
    err = InfeasibleScenarioError("exhausted", n_resample=10, n_replan=0)
    assert err.n_replan == 0


def test_infeasible_scenario_error_defaults() -> None:
    err = InfeasibleScenarioError("no msg")
    assert err.n_resample == 0
    assert err.n_replan == 0


def test_infeasible_scenario_error_raised_without_catching_runtime() -> None:
    """InfeasibleScenarioError must NOT be caught by 'except RuntimeError'."""
    caught_runtime = False
    caught_exception = False
    try:
        raise InfeasibleScenarioError("too many attempts")
    except RuntimeError:
        caught_runtime = True
    except Exception:
        caught_exception = True
    assert not caught_runtime, "InfeasibleScenarioError should NOT be a RuntimeError"
    assert caught_exception


# ---------------------------------------------------------------------------
# RECOVERY_STRATEGY mapping
# ---------------------------------------------------------------------------


def test_recovery_strategy_has_collision_entry() -> None:
    assert CollisionError in RECOVERY_STRATEGY


def test_recovery_strategy_has_visibility_entry() -> None:
    assert VisibilityError in RECOVERY_STRATEGY


def test_recovery_strategy_collision_is_propagate_immediately() -> None:
    assert RECOVERY_STRATEGY[CollisionError] == "propagate_immediately"


def test_recovery_strategy_visibility_is_resample_scenario() -> None:
    assert RECOVERY_STRATEGY[VisibilityError] == "resample_scenario"


def test_recovery_strategy_all_values_are_strings() -> None:
    for key, value in RECOVERY_STRATEGY.items():
        assert isinstance(value, str), f"RECOVERY_STRATEGY[{key}] is not a string"


# ---------------------------------------------------------------------------
# Polymorphic catching (isinstance checks)
# ---------------------------------------------------------------------------


def test_all_validation_errors_are_scenario_validation_error() -> None:
    """All typed errors must be recognizable as ScenarioValidationError via isinstance."""
    errors = [
        CollisionError("c"),
        VisibilityError("v"),
    ]
    for err in errors:
        assert isinstance(err, ScenarioValidationError), f"{type(err).__name__} is not an instance of ScenarioValidationError"  # fmt: skip  # noqa: E501


def test_all_validation_errors_are_runtime_error() -> None:
    """All typed errors must be catchable as RuntimeError."""
    errors = [
        CollisionError("c"),
        VisibilityError("v"),
    ]
    for err in errors:
        assert isinstance(err, RuntimeError), f"{type(err).__name__} is not a RuntimeError"


def test_validation_errors_are_distinct_types() -> None:
    """CollisionError and VisibilityError must be distinct (not aliased)."""
    assert CollisionError is not VisibilityError


# ---------------------------------------------------------------------------
# Validation feedback loop: run_with_validation_loop
# ---------------------------------------------------------------------------


def test_run_with_validation_loop_exists() -> None:
    """run_with_validation_loop must exist in simulator.py and be callable."""
    from libero_infinity.simulator import run_with_validation_loop

    assert callable(run_with_validation_loop)


def test_run_with_validation_loop_raises_infeasible_on_visibility_exhaustion() -> None:
    """run_with_validation_loop must raise InfeasibleScenarioError when VisibilityError retries exhausted."""  # noqa: E501
    from libero_infinity.simulator import run_with_validation_loop

    class _MockScene:
        params = {}

    class _MockScenario:
        def generate(self, maxIterations=100):
            return _MockScene(), 1

    class _MockSimulator:
        def simulate(self, scene, maxSteps=500):
            raise VisibilityError("mock visibility failure")

    scenario = _MockScenario()
    simulator = _MockSimulator()

    with pytest.raises(InfeasibleScenarioError) as exc_info:
        run_with_validation_loop(scenario, simulator, max_visibility_retries=3)

    err = exc_info.value
    assert err.n_resample >= 3


def test_run_with_validation_loop_collision_propagates_immediately() -> None:
    """CollisionError must propagate immediately as InfeasibleScenarioError (no retry)."""
    from libero_infinity.simulator import run_with_validation_loop

    call_count = 0

    class _MockScene:
        params = {}

    class _MockScenario:
        def generate(self, maxIterations=100):
            return _MockScene(), 1

    class _MockSimulator:
        def simulate(self, scene, maxSteps=500):
            nonlocal call_count
            call_count += 1
            raise CollisionError("mock collision")

    with pytest.raises(InfeasibleScenarioError):
        run_with_validation_loop(_MockScenario(), _MockSimulator(), max_visibility_retries=10)

    # CollisionError should NOT be retried — only one call
    assert call_count == 1, f"CollisionError was retried {call_count} times; expected 1"
