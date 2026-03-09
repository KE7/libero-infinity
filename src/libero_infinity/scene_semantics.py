"""Structured task semantics derived from parsed LIBERO BDDL tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from libero_infinity.task_config import TaskConfig


@dataclass(frozen=True)
class AtomicPredicate:
    predicate: str
    arg1: str
    arg2: str | None = None


@dataclass(frozen=True)
class ArticulationSpec:
    fixture_name: str
    state_kind: str
    lo: float
    hi: float
    reason: str


_ATOMIC_RE = re.compile(r"\((On|In|Open|Close|Turnon|Turnoff)\s+([^\s()]+)(?:\s+([^\s()]+))?\)")

_ARTICULATION_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "microwave": {
        "Open": (-2.094, -1.3),
        "Close": (-0.005, 0.0),
    },
    "wooden_cabinet": {
        "Open": (-0.16, -0.14),
        "Close": (0.0, 0.005),
    },
    "white_cabinet": {
        "Open": (-0.16, -0.14),
        "Close": (0.0, 0.005),
    },
    "flat_stove": {
        "Turnon": (0.5, 2.1),
        "Turnoff": (-0.005, 0.0),
    },
}


def parse_atomic_predicates(text: str) -> list[AtomicPredicate]:
    """Extract flat atomic predicates from a BDDL init/goal expression."""
    return [
        AtomicPredicate(match.group(1), match.group(2), match.group(3))
        for match in _ATOMIC_RE.finditer(text or "")
    ]


def goal_predicates(cfg: TaskConfig) -> list[AtomicPredicate]:
    return parse_atomic_predicates(cfg.goal_text)


def init_predicates(cfg: TaskConfig) -> list[AtomicPredicate]:
    return parse_atomic_predicates(cfg.init_text)


def task_relevant_object_names(cfg: TaskConfig) -> set[str]:
    """Movable task objects the policy must still be able to see."""
    movable_names = {obj.instance_name for obj in cfg.movable_objects}
    relevant = {name for name in cfg.obj_of_interest if name in movable_names}
    for pred in goal_predicates(cfg):
        if pred.arg1 in movable_names:
            relevant.add(pred.arg1)
        if pred.arg2 in movable_names:
            relevant.add(pred.arg2)
    return relevant


def coordination_groups(cfg: TaskConfig) -> dict[str, list[str]]:
    """Group objects that share the same support surface / container."""
    groups: dict[str, list[str]] = {}
    for obj in cfg.movable_objects:
        parent = obj.placement_target or obj.stacked_on
        if not parent:
            continue
        groups.setdefault(parent, []).append(obj.instance_name)
    return {parent: names for parent, names in groups.items() if len(names) > 1}


def articulated_fixture_specs(cfg: TaskConfig) -> dict[str, ArticulationSpec]:
    """Choose safe articulation bands for the current task.

    Goal-articulated fixtures are initialized in the complementary band so the
    task cannot start solved. Non-goal articulated fixtures stay in their
    canonical open/close/on/off band when it can be inferred from the init.
    """
    fixture_by_name = {fixture.instance_name: fixture for fixture in cfg.fixtures}
    init_by_fixture: dict[str, str] = {}
    for pred in init_predicates(cfg):
        if pred.predicate in {"Open", "Close", "Turnon", "Turnoff"}:
            init_by_fixture[pred.arg1] = pred.predicate

    specs: dict[str, ArticulationSpec] = {}
    for pred in goal_predicates(cfg):
        if pred.predicate not in {"Open", "Close", "Turnon", "Turnoff"}:
            continue
        fixture = fixture_by_name.get(pred.arg1)
        if fixture is None:
            continue
        class_ranges = _ARTICULATION_RANGES.get(fixture.fixture_class)
        if class_ranges is None:
            continue
        complementary = _complementary_state(pred.predicate)
        lo, hi = class_ranges[complementary]
        specs[pred.arg1] = ArticulationSpec(
            fixture_name=pred.arg1,
            state_kind=complementary,
            lo=lo,
            hi=hi,
            reason=f"goal_{pred.predicate.lower()}",
        )

    for fixture in cfg.fixtures:
        if fixture.instance_name in specs:
            continue
        class_ranges = _ARTICULATION_RANGES.get(fixture.fixture_class)
        if class_ranges is None:
            continue
        canonical = init_by_fixture.get(fixture.instance_name)
        if canonical is None:
            canonical = _default_state_for_fixture_class(fixture.fixture_class)
        if canonical is None or canonical not in class_ranges:
            continue
        lo, hi = class_ranges[canonical]
        specs[fixture.instance_name] = ArticulationSpec(
            fixture_name=fixture.instance_name,
            state_kind=canonical,
            lo=lo,
            hi=hi,
            reason="canonical_init_band",
        )
    return specs


def support_contains_articulated_compartment(cfg: TaskConfig, fixture_name: str) -> bool:
    """Whether a fixture has task objects inside a moving compartment."""
    for obj in cfg.movable_objects:
        if obj.placement_target != fixture_name or not obj.contained:
            continue
        return True
    return False


def _complementary_state(goal_state: str) -> str:
    if goal_state == "Open":
        return "Close"
    if goal_state == "Close":
        return "Open"
    if goal_state == "Turnon":
        return "Turnoff"
    if goal_state == "Turnoff":
        return "Turnon"
    raise ValueError(f"Unsupported articulated goal state: {goal_state}")


def _default_state_for_fixture_class(fixture_class: str) -> str | None:
    if fixture_class in {"microwave", "wooden_cabinet", "white_cabinet"}:
        return "Close"
    if fixture_class == "flat_stove":
        return "Turnoff"
    return None
