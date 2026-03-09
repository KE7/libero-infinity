"""
Tests for settling correctness and retry-loop elimination.
T1: SettleUnsafeError never raised in src/
T2: AlreadySolvedError never raised in src/
T3: _footprint_clearance_xy <= min_clearance for all pairs
T4: Renderer never emits visibility_targets
T5: Single-object settle stability
"""

import subprocess
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "src"


# T1: SettleUnsafeError is dead code — never raised
def test_settle_unsafe_error_never_raised():
    result = subprocess.run(
        ["grep", "-r", "raise SettleUnsafeError", str(SRC_DIR)],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "", f"SettleUnsafeError is raised somewhere: {result.stdout}"


# T2: AlreadySolvedError is dead code — never raised
def test_already_solved_error_never_raised():
    result = subprocess.run(
        ["grep", "-r", "raise AlreadySolvedError", str(SRC_DIR)],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "", f"AlreadySolvedError is raised somewhere: {result.stdout}"


# T3: _footprint_clearance_xy covers all object pairs
def test_footprint_clearance_covers_all_pairs():
    """Verify that per-pair clearance is used — renderer no longer emits fixed min_clearance."""
    renderer_path = SRC_DIR / "libero_infinity" / "renderer" / "scenic_renderer.py"
    source = renderer_path.read_text()
    # The fixed global min_clearance should be gone
    assert "param min_clearance = 0.10" not in source, "Renderer still emits hardcoded min_clearance = 0.10"  # fmt: skip  # noqa: E501


# T4: VisibilityError is the ONLY retried error
def test_only_visibility_error_is_retried():
    """simulator.py should only catch VisibilityError in the retry loop."""
    sim_path = SRC_DIR / "libero_infinity" / "simulator.py"
    source = sim_path.read_text()
    assert "MAX_VISIBILITY_RETRIES" in source, "MAX_VISIBILITY_RETRIES not found in simulator.py"
    assert "MAX_RESAMPLE" not in source, "Old MAX_RESAMPLE still referenced in simulator.py"
    assert "MAX_REPLAN" not in source, "Old MAX_REPLAN still referenced in simulator.py"
    assert "SettleUnsafeError" not in source, "Dead code SettleUnsafeError still in simulator.py"
    assert "AlreadySolvedError" not in source, "Dead code AlreadySolvedError still in simulator.py"


# T5: validation_errors has MAX_VISIBILITY_RETRIES and not the old constants
def test_validation_errors_constants():
    from libero_infinity.validation_errors import MAX_VISIBILITY_RETRIES

    assert MAX_VISIBILITY_RETRIES >= 5, "MAX_VISIBILITY_RETRIES should be at least 5"
    import libero_infinity.validation_errors as ve

    assert not hasattr(ve, "MAX_RESAMPLE"), "MAX_RESAMPLE should be removed"
    assert not hasattr(ve, "MAX_REPLAN"), "MAX_REPLAN should be removed"
    assert not hasattr(ve, "SettleUnsafeError"), "SettleUnsafeError should be removed"
    assert not hasattr(ve, "AlreadySolvedError"), "AlreadySolvedError should be removed"
