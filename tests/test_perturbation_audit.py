"""Unit tests for deterministic perturbation-audit helpers."""

from __future__ import annotations

import numpy as np
import pytest

from libero_infinity.perturbation_audit import (
    AnchorPixelRecord,
    VisibleChangeScoreConfig,
    analyze_generated_constraints,
    parse_anchor_pixel_records,
    score_visible_change,
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


def test_parse_anchor_pixel_records_supports_named_mapping_payloads():
    payload = {
        "bowl": {
            "canonical_pixel": [10, 12],
            "perturbed_pixel": {"x": 18, "y": 21},
            "perturbed_visible": True,
        },
        "plate": {
            "canonical_pixel": {"u": 30, "v": 32},
            "perturbed_pixel": None,
            "canonical_visible": True,
            "perturbed_visible": False,
        },
    }

    records = parse_anchor_pixel_records(payload)

    assert records == [
        AnchorPixelRecord(
            name="bowl",
            canonical_pixel=(10.0, 12.0),
            perturbed_pixel=(18.0, 21.0),
            canonical_visible=None,
            perturbed_visible=True,
        ),
        AnchorPixelRecord(
            name="plate",
            canonical_pixel=(30.0, 32.0),
            perturbed_pixel=None,
            canonical_visible=True,
            perturbed_visible=False,
        ),
    ]


def test_score_visible_change_marks_material_visible_change_when_signals_agree():
    frame_a = np.zeros((8, 8, 3), dtype=np.uint8)
    frame_b = np.full((8, 8, 3), 64, dtype=np.uint8)
    anchors = [
        AnchorPixelRecord("bowl", canonical_pixel=(1, 1), perturbed_pixel=(7, 7)),
        AnchorPixelRecord("plate", canonical_pixel=(2, 2), perturbed_pixel=(6, 6)),
    ]
    config = VisibleChangeScoreConfig(
        rgb_delta_material_threshold=0.1,
        anchor_displacement_reference_px=6.0,
        anchor_motion_material_threshold_px=4.0,
        combined_material_threshold=0.5,
        vlm_ambiguity_lower=0.2,
        vlm_ambiguity_upper=0.6,
    )

    score = score_visible_change(frame_a, frame_b, anchors, config=config)

    assert score.material_rgb_change is True
    assert score.material_anchor_motion is True
    assert score.anchor_visibility_ok is True
    assert score.material_visible_change is True
    assert score.combined_score > 0.6
    assert score.should_run_vlm_check is False


def test_score_visible_change_requests_vlm_when_visibility_is_bad():
    frame_a = np.zeros((10, 10, 3), dtype=np.uint8)
    frame_b = np.full((10, 10, 3), 48, dtype=np.uint8)
    anchors = [
        AnchorPixelRecord(
            "target",
            canonical_pixel=(4, 4),
            perturbed_pixel=None,
            canonical_visible=True,
            perturbed_visible=False,
        )
    ]
    config = VisibleChangeScoreConfig(
        rgb_delta_material_threshold=0.1,
        combined_material_threshold=0.2,
        minimum_perturbed_visibility_fraction=0.8,
        minimum_perturbed_in_frame_fraction=0.8,
    )

    score = score_visible_change(frame_a, frame_b, anchors, config=config)

    assert score.material_rgb_change is True
    assert score.material_anchor_motion is False
    assert score.anchor_visibility_ok is False
    assert score.material_visible_change is False
    assert score.should_run_vlm_check is True


def test_parse_anchor_pixel_records_rejects_invalid_payload():
    with pytest.raises(ValueError):
        parse_anchor_pixel_records("not-a-record")
