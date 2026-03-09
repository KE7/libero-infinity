"""Typed validation error classes for the Libero-Infinity simulation pipeline.

Replaces the single generic ERR_VALIDATE pattern with typed exceptions
that carry recovery strategy metadata. Each error type maps to a specific
recovery action in the validation feedback loop.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Retry budget constants
# ---------------------------------------------------------------------------

# Maximum retries for VisibilityError. Three sub-cases require retrying:
# 1. Camera projection uncertainty: a task-relevant object placed near the
#    workspace edge may fall outside the camera frustum at sample time.
# 2. Distractor occlusion: a distractor object may be spawned at a position
#    that blocks the line-of-sight to a task-relevant object. The distractor
#    placement is valid (no collision), but occludes the view.
# 3. Articulation occlusion: a microwave or cabinet door opened post-settle
#    may swing into view and occlude a nearby task-relevant object.
# In all three cases, re-sampling the Scenic scenario may resolve the issue.
MAX_VISIBILITY_RETRIES: int = 10


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ScenarioValidationError(RuntimeError):
    """Base class for all scenario post-settle validation errors.

    Inherits from RuntimeError for backward compatibility with existing
    ``except RuntimeError`` handlers. Subclasses indicate the failure mode
    and thus the appropriate recovery strategy in the validation feedback loop.
    """

    pass


# ---------------------------------------------------------------------------
# Typed validation errors (one per recovery strategy)
# ---------------------------------------------------------------------------


class CollisionError(ScenarioValidationError):
    """Objects are in collision after MuJoCo settling.

    Since the renderer now emits per-pair clearance constraints computed
    from object footprint dimensions, a CollisionError indicates a bug in
    the renderer or an infeasible scene — it is NOT retried.
    """

    def __init__(self, msg: str = "", object_names: list[str] | None = None) -> None:
        super().__init__(msg)
        self.object_names: list[str] = object_names or []


class VisibilityError(ScenarioValidationError):
    """Task-relevant objects are not visible after settling.

    Recovery strategy: re-sample the Scenic scenario (position or camera
    parameters). See MAX_VISIBILITY_RETRIES for the three sub-cases that
    make retrying necessary.
    """

    def __init__(self, msg: str = "", invisible_names: list[str] | None = None) -> None:
        super().__init__(msg)
        self.invisible_names: list[str] = invisible_names or []


# ---------------------------------------------------------------------------
# Exhaustion error (not a validation error per se)
# ---------------------------------------------------------------------------


class InfeasibleScenarioError(Exception):
    """Raised when MAX_VISIBILITY_RETRIES attempts are all exhausted.

    This is a hard failure — the caller should not retry further.
    Carries the number of attempts made for diagnostics.
    """

    def __init__(
        self,
        msg: str = "",
        n_resample: int = 0,
        n_replan: int = 0,
    ) -> None:
        super().__init__(msg)
        self.n_resample: int = n_resample
        self.n_replan: int = n_replan


# ---------------------------------------------------------------------------
# Recovery strategy mapping
# ---------------------------------------------------------------------------

#: Maps each typed error class to a short recovery strategy name.
#: Used by the validation feedback loop to decide what to retry.
RECOVERY_STRATEGY: dict[type[ScenarioValidationError], str] = {
    CollisionError: "propagate_immediately",
    VisibilityError: "resample_scenario",
}
