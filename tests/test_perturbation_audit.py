"""Unit tests for perturbation-audit helpers."""

from __future__ import annotations

from libero_infinity.perturbation_audit import (
    analyze_generated_constraints,
)


def test_analyze_generated_constraints_counts_temporal_and_clearance_requirements():
    scenic_code = """
        require abs(obj_a.position.x - obj_b.position.x) > _axis_margin
        require distance from obj_a to obj_b > _min_clearance
        require[eventually] monitor collision_free
        require[0.2] distance from obj_a to obj_b > _ood_margin
    """

    audit = analyze_generated_constraints(scenic_code)

    assert audit.hard_require_total == 2
    assert audit.soft_require_total == 2
    assert audit.hard_axis_clearance == 1
    assert audit.hard_distance_clearance == 1
    assert audit.soft_ood_bias == 1
    assert audit.temporal_require_total == 1
    assert audit.temporal_operators == ("monitor",)
