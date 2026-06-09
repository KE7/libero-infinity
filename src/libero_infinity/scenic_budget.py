"""Task/mode-adaptive Scenic iteration budgets (WS-3).

The Scenic scene generator is a rejection sampler: ``Scenario.generate()`` draws
candidate scenes until one satisfies every hard ``require`` constraint. A single
GLOBAL ``maxIterations=5000`` under-provisions the harder perturbation modes —
calibration (``scripts/calibrate_scenic_iterations.py``) shows ``combined`` and
``full`` need tens-of-thousands of iterations on hard tasks, so 5000 silently
exhausts the budget, trips ``MAX_SETTLE_RETRIES`` / ``RejectionException``, and
corrupts the valid-scene distribution without a clear diagnostic.

This module resolves a per-mode budget from a measured artifact
(``data/scenic_iteration_budgets.json``) and provides an under-budget warning.

Back-compat: an explicit ``max_scenic_iterations`` always wins, and any mode the
artifact does not cover resolves to :data:`DEFAULT_MAX_ITERATIONS` (5000).
"""

from __future__ import annotations

import functools
import json
import logging
import pathlib

log = logging.getLogger(__name__)

# Back-compat default — the historical global constant.
DEFAULT_MAX_ITERATIONS: int = 5000

# Emit an under-budget warning when generation consumes >= this fraction of the
# allotted budget (early signal that the budget is too tight for the mode/task).
BUDGET_WARN_FRACTION: float = 0.90

_ARTIFACT = pathlib.Path(__file__).parent / "data" / "scenic_iteration_budgets.json"


@functools.lru_cache(maxsize=1)
def _load_mode_budgets() -> dict[str, int]:
    """Load measured per-mode budgets from the calibration artifact.

    Returns an empty dict if the artifact is missing/unreadable, in which case
    every mode falls back to :data:`DEFAULT_MAX_ITERATIONS`.
    """
    try:
        data = json.loads(_ARTIFACT.read_text())
    except (FileNotFoundError, ValueError, OSError):
        return {}
    out: dict[str, int] = {}
    for mode, info in (data.get("modes") or {}).items():
        budget = info.get("budget") if isinstance(info, dict) else info
        if isinstance(budget, (int, float)) and budget > 0:
            out[mode] = int(budget)
    return out


def resolve_iteration_budget(
    perturbation: str | None,
    explicit: int | None = None,
) -> int:
    """Resolve the Scenic ``maxIterations`` budget for a perturbation request.

    ``explicit`` (caller override) always wins, for full back-compat. Otherwise
    the budget is the **max** of every applicable measured estimate, so it is
    *monotone in axis-set inclusion* — a strictly larger request never gets a
    smaller budget. The candidates are:

    - the :data:`DEFAULT_MAX_ITERATIONS` floor (5000);
    - the exact mode-name budget, if the spec names a calibrated mode
      (``"combined"``, ``"full"``, ``"position"``, …);
    - every calibrated preset whose axis-set is fully contained in the request
      (so a custom comma-list equal to — or a superset of — a preset inherits
      that preset's budget). This is also why ``"full"`` ⊇ ``"combined"`` folds
      in the (empirically larger) ``combined`` budget rather than dropping below
      it on sampling noise;
    - every measured single-axis budget present in the request.

    Args:
        perturbation: The perturbation spec string (axis, preset, or
            comma-separated list). ``None``/empty → default.
        explicit: An explicit override; bypasses all auto-resolution.

    Returns:
        The iteration budget (always >= :data:`DEFAULT_MAX_ITERATIONS`).
    """
    if explicit is not None:
        return int(explicit)
    if not perturbation:
        return DEFAULT_MAX_ITERATIONS

    modes = _load_mode_budgets()
    candidates = [DEFAULT_MAX_ITERATIONS]

    key = perturbation.strip()
    if key in modes:
        candidates.append(modes[key])

    try:
        from libero_infinity.planner.composition import AXIS_PRESETS, parse_axes

        axes = parse_axes(perturbation)
    except Exception:  # noqa: BLE001 — never let budget resolution crash a run
        return max(candidates)

    for preset_name, preset_axes in AXIS_PRESETS.items():
        if preset_name in modes and preset_axes <= axes:
            candidates.append(modes[preset_name])
    for axis in axes:
        if axis in modes:
            candidates.append(modes[axis])
    return max(candidates)


def warn_if_near_budget(
    n_iters: int,
    budget: int,
    *,
    mode: str = "",
    logger: logging.Logger | None = None,
) -> bool:
    """Warn when a generation came within ``BUDGET_WARN_FRACTION`` of ``budget``.

    This is an early signal that the budget is under-provisioned for the
    mode/task: scenes that need slightly more iterations would silently fail to
    generate. Returns ``True`` iff a warning was emitted.
    """
    logger = logger or log
    if budget <= 0:
        return False
    if n_iters >= budget * BUDGET_WARN_FRACTION:
        logger.warning(
            "Scenic generation used %d/%d iterations (>=%.0f%% of budget) "
            "for perturbation mode '%s' — budget may be under-provisioned; "
            "harder scenes risk silent rejection-sampling failures. Consider "
            "raising max_scenic_iterations or re-running the calibration sweep.",
            n_iters,
            budget,
            BUDGET_WARN_FRACTION * 100,
            mode or "?",
        )
        return True
    return False
