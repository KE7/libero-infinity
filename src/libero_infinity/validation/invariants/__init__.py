"""Validation invariants (G4 family).

Family A — identity invariants: cross-axis isolation. When perturbation axis X
is NOT in ``active_axes``, attributes belonging to axis X must be identical to
the no-axes baseline scene. This proves perturbation logic does not leak
across axes.
"""

from .identity import (
    AssertionResult,
    IDENTITY_ASSERTIONS,
    assert_all_identities,
    assert_articulation_unchanged,
    assert_background_unchanged,
    assert_camera_unchanged,
    assert_distractor_unchanged,
    assert_lighting_unchanged,
    assert_object_unchanged,
    assert_position_unchanged,
    assert_robot_unchanged,
    assert_texture_unchanged,
    g4_identity_hook,
)

__all__ = [
    "AssertionResult",
    "IDENTITY_ASSERTIONS",
    "assert_all_identities",
    "assert_articulation_unchanged",
    "assert_background_unchanged",
    "assert_camera_unchanged",
    "assert_distractor_unchanged",
    "assert_lighting_unchanged",
    "assert_object_unchanged",
    "assert_position_unchanged",
    "assert_robot_unchanged",
    "assert_texture_unchanged",
    "g4_identity_hook",
]
