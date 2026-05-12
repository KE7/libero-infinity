"""Shared ``AssertionResult`` dataclass for all G4 invariant families.

Separated from ``identity.py`` so families B/C/D can import it without pulling
in the identity machinery (which is independent and not yet present in this
worktree).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AssertionResult:
    """Outcome of a single G4 invariant check.

    Attributes:
        name: Stable, short identifier (e.g. ``"bddl_objects_present"``).
        passed: ``True`` / ``False`` / ``None``.  ``None`` is reserved for
            *legitimate* skips where the input genuinely lacks data
            (e.g. no MuJoCo handle attached, asset has no grasp-point
            metadata).  ``None`` must never be used to dodge a failure.
        detail: Human-readable one-line summary of the outcome.
        payload: Structured diagnostics for downstream aggregation
            (field-level diffs, contact lists, missing objects, ...).
    """

    name: str
    passed: bool | None
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover - defensive
        # Truthiness on a skip would be ambiguous; force callers to compare
        # against ``passed`` explicitly.
        raise TypeError(
            "AssertionResult is not directly boolean; check .passed (True/False/None)"
        )


__all__ = ["AssertionResult"]
