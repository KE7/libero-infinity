"""Focused tests for the audit-followups branch.

These cover the partial / low-severity items addressed from
``codex_branch_claims_audit.md``:

- C3: ``_real_depth_map`` no longer ``assert``-crashes on NaN /
  out-of-range pixels; it sanitises and returns a finite array.
- E5: ``_apply_image_corruption`` now mixes a per-scene ``seed`` into
  the RNG so identical (kind, severity) pairs produce *different*
  noise patterns across scenes, while remaining reproducible for the
  same seed.
- F2: ``yaw_bounds`` always returns either ``None`` or a strictly
  ordered ``(lo, hi)`` interval.
"""

from __future__ import annotations

import numpy as np

from libero_infinity.perturbation_policy import yaw_bounds
from libero_infinity.simulator import _apply_image_corruption, _real_depth_map


class _StubModel:
    class _Stat:
        extent = 1.0

    class _Vis:
        class _Map:
            zfar = 10.0
            znear = 0.1

        map = _Map()

    stat = _Stat()
    vis = _Vis()


class _StubSim:
    model = _StubModel()


def test_real_depth_map_handles_nan_and_out_of_range():
    depth = np.array([[0.0, 0.5, 1.0], [np.nan, -0.1, 1.5]], dtype=float)
    out = _real_depth_map(_StubSim(), depth)
    assert np.all(np.isfinite(out)), "depth output must be finite after sanitisation"
    assert out.shape == depth.shape


def test_sensor_noise_varies_across_scene_seeds():
    img = (np.ones((16, 16, 3), dtype=np.uint8) * 128)
    a = _apply_image_corruption(img, "gaussian_noise", 3, seed=1)
    b = _apply_image_corruption(img, "gaussian_noise", 3, seed=2)
    c = _apply_image_corruption(img, "gaussian_noise", 3, seed=1)
    # Different seeds must give different patterns…
    assert not np.array_equal(a, b)
    # …but the same seed remains reproducible.
    assert np.array_equal(a, c)


def test_sensor_noise_legacy_path_unchanged_when_no_seed():
    img = (np.ones((16, 16, 3), dtype=np.uint8) * 128)
    a = _apply_image_corruption(img, "gaussian_noise", 3)
    b = _apply_image_corruption(img, "gaussian_noise", 3)
    # Severity-only seeding remains deterministic when seed is omitted.
    assert np.array_equal(a, b)


def test_yaw_bounds_strictly_ordered_or_none():
    interval = yaw_bounds(canonical_yaw=0.0, asset_class="bowl_drainer")
    assert interval is not None
    lo, hi = interval
    assert lo < hi

    # An unknown asset still gets the default fallback.
    interval2 = yaw_bounds(canonical_yaw=1.5, asset_class="totally_unknown_class")
    assert interval2 is not None
    lo2, hi2 = interval2
    assert lo2 < hi2
