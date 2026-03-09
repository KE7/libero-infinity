import math
import random

import pytest
from conftest import BDDL_DIR, BOWL_BDDL

from libero_infinity.perturbation_policy_helpers import (
    GroupTransform,
    NumericRange,
    YawRange,
    apply_group_transform,
    coordination_groups,
    parse_region_yaw_ranges_from_file,
    resolve_object_yaw_ranges,
    sample_group_transform,
    support_local_envelope,
)
from libero_infinity.task_config import TaskConfig


def _require_bddl(path):
    if path is None or not path.exists():
        pytest.skip("Required BDDL file not found")
    return path


class TestYawRanges:
    def test_parse_region_yaw_ranges_reads_exact_bounds(self):
        ranges = parse_region_yaw_ranges_from_file(_require_bddl(BOWL_BDDL))

        assert ranges["cabinet_region"].minimum == pytest.approx(math.pi)
        assert ranges["cabinet_region"].maximum == pytest.approx(math.pi)
        assert ranges["wine_rack_region"].minimum == pytest.approx(math.pi)
        assert ranges["wine_rack_region"].maximum == pytest.approx(math.pi)

    def test_resolve_object_yaw_ranges_uses_explicit_region_yaw_when_present(self):
        path = next(
            BDDL_DIR.glob(
                "**/STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.bddl"
            ),
            None,
        )
        cfg = TaskConfig.from_bddl(_require_bddl(path))
        ranges = resolve_object_yaw_ranges(cfg)

        assert ranges["black_book_1"].minimum == pytest.approx(-math.pi / 2.0)
        assert ranges["black_book_1"].maximum == pytest.approx(-math.pi / 4.0)

    def test_resolve_object_yaw_ranges_uses_default_when_region_has_no_yaw(self):
        cfg = TaskConfig.from_bddl(_require_bddl(BOWL_BDDL))
        default = YawRange(-0.25, 0.25)

        ranges = resolve_object_yaw_ranges(cfg, default=default)

        assert ranges["plate_1"] == default


class TestSupportEnvelope:
    def test_cook_surface_envelope_is_wider_than_default_surface(self):
        cook = support_local_envelope(
            support_dims=(0.24, 0.18, 0.08),
            child_dims=(0.10, 0.10, 0.16),
            support_class="flat_stove",
            region_name="cook_region",
            contained=False,
        )
        generic = support_local_envelope(
            support_dims=(0.24, 0.18, 0.08),
            child_dims=(0.10, 0.10, 0.16),
            support_class=None,
            region_name=None,
            contained=False,
        )

        assert cook.support_type == "cook_surface"
        assert cook.x_half_extent > generic.x_half_extent
        assert cook.y_half_extent > generic.y_half_extent

    def test_front_compartment_squeezes_the_forward_axis(self):
        envelope = support_local_envelope(
            support_dims=(0.14, 0.10, 0.06),
            child_dims=(0.04, 0.03, 0.02),
            support_class="desk_caddy",
            region_name="front_contain_region",
            contained=True,
        )

        assert envelope.support_type == "contained"
        assert envelope.y_half_extent < envelope.x_half_extent

    def test_side_region_squeezes_the_lateral_axis(self):
        envelope = support_local_envelope(
            support_dims=(0.26, 0.26, 0.24),
            child_dims=(0.08, 0.08, 0.06),
            support_class="wooden_cabinet",
            region_name="left_region",
            contained=False,
        )

        assert envelope.x_half_extent < envelope.y_half_extent


class TestCoordinationGroups:
    def test_root_workspace_objects_share_a_workspace_group(self):
        cfg = TaskConfig.from_bddl(_require_bddl(BOWL_BDDL))

        groups = coordination_groups(cfg)
        workspace_group = {obj.instance_name for obj in groups["workspace:main_table"]}

        assert {"akita_black_bowl_1", "plate_1"} <= workspace_group

    def test_stacked_object_gets_its_support_group(self):
        path = next(
            BDDL_DIR.glob(
                "**/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl"
            ),
            None,
        )
        cfg = TaskConfig.from_bddl(_require_bddl(path))

        groups = coordination_groups(cfg)
        support_group = {obj.instance_name for obj in groups["support:cookies_1"]}

        assert "akita_black_bowl_1" in support_group


class TestCoordinatedTransforms:
    def test_apply_group_transform_rotates_translates_and_jitters(self):
        canonical = {"obj_a": (1.0, 0.0), "obj_b": (0.0, 1.0)}
        transform = GroupTransform(
            translation=(1.0, -1.0),
            shared_yaw=math.pi / 2.0,
            local_jitter={"obj_b": (0.1, -0.2)},
        )

        transformed = apply_group_transform(
            canonical,
            anchor_xy=(0.0, 0.0),
            transform=transform,
        )

        assert transformed["obj_a"][0] == pytest.approx(1.0, abs=1e-6)
        assert transformed["obj_a"][1] == pytest.approx(0.0, abs=1e-6)
        assert transformed["obj_b"][0] == pytest.approx(0.1, abs=1e-6)
        assert transformed["obj_b"][1] == pytest.approx(-1.2, abs=1e-6)

    def test_sample_group_transform_handles_optional_shared_yaw(self):
        rng = random.Random(7)

        transform = sample_group_transform(
            ["obj_a", "obj_b"],
            translation_x_range=NumericRange(0.2, 0.2),
            translation_y_range=NumericRange(-0.1, -0.1),
            shared_yaw_range=YawRange(math.pi / 4.0, math.pi / 4.0),
            local_jitter_range=(NumericRange(0.0, 0.0), NumericRange(0.01, 0.01)),
            rng=rng,
        )

        assert transform.translation == pytest.approx((0.2, -0.1))
        assert transform.shared_yaw == pytest.approx(math.pi / 4.0)
        assert transform.local_jitter["obj_a"] == pytest.approx((0.0, 0.01))
        assert transform.local_jitter["obj_b"] == pytest.approx((0.0, 0.01))


# ---------------------------------------------------------------------------
# perturbation_policy.py — completely untested module
# ---------------------------------------------------------------------------


class TestPerturbationPolicyYawBounds:
    """Tests for yaw_bounds() in perturbation_policy.py."""

    def test_bowl_class_gets_bowl_span(self):
        """Bowl asset class must use _YAW_SPAN_BY_CLASS['bowl'] = 0.55."""
        from libero_infinity.perturbation_policy import yaw_bounds

        lo, hi = yaw_bounds(canonical_yaw=0.0, asset_class="akita_black_bowl")
        # Bowl span = 0.55 (matched via substring 'bowl' in name)
        assert hi - lo == pytest.approx(2 * 0.55, abs=0.01)
        assert lo == pytest.approx(-0.55, abs=0.01)
        assert hi == pytest.approx(0.55, abs=0.01)

    def test_canonical_yaw_shifts_centre(self):
        """Non-zero canonical_yaw must shift the interval centre."""
        from libero_infinity.perturbation_policy import yaw_bounds

        centre = math.pi / 2.0
        lo, hi = yaw_bounds(canonical_yaw=centre, asset_class="mug")
        midpoint = (lo + hi) / 2.0
        # mug span = 0.65, so the centre should be at math.pi/2
        assert midpoint == pytest.approx(centre, abs=1e-6)
        assert hi - lo == pytest.approx(2 * 0.65, abs=0.01)

    def test_none_canonical_yaw_defaults_to_zero_centre(self):
        """canonical_yaw=None must default to 0.0 centre."""
        from libero_infinity.perturbation_policy import yaw_bounds

        lo, hi = yaw_bounds(canonical_yaw=None, asset_class="plate")
        midpoint = (lo + hi) / 2.0
        assert midpoint == pytest.approx(0.0, abs=1e-6)

    def test_round_object_uses_wider_span_than_non_round(self):
        """Round objects (bowl, plate) should get a wider yaw span than non-round ones."""
        from libero_infinity.perturbation_policy import yaw_bounds

        lo_bowl, hi_bowl = yaw_bounds(canonical_yaw=0.0, asset_class="akita_black_bowl")
        lo_box, hi_box = yaw_bounds(canonical_yaw=0.0, asset_class="alphabet_soup_can")
        span_bowl = hi_bowl - lo_bowl
        span_box = hi_box - lo_box
        assert span_bowl > span_box, f"Bowl should have wider yaw span than soup can: {span_bowl:.3f} vs {span_box:.3f}"  # fmt: skip  # noqa: E501

    def test_support_class_overrides_asset_class_for_yaw(self):
        """When support_class matches a key, it should override asset_class span."""
        from libero_infinity.perturbation_policy import _YAW_SPAN_BY_CLASS, yaw_bounds

        # cabinet support constrains yaw tightly
        cabinet_span = _YAW_SPAN_BY_CLASS.get("cabinet", 0.12)
        lo, hi = yaw_bounds(
            canonical_yaw=0.0,
            asset_class="akita_black_bowl",
            support_class="wooden_cabinet",
        )
        # wooden_cabinet contains "cabinet" keyword → cabinet span used
        assert hi - lo == pytest.approx(2 * cabinet_span, abs=0.01)


class TestPerturbationPolicySupportOffsetBounds:
    """Tests for support_offset_bounds() in perturbation_policy.py."""

    def test_basic_geometry_plate_support(self):
        """Plate support with default child dims should produce non-zero offset bounds."""
        from libero_infinity.perturbation_policy import support_offset_bounds

        support_dims = (0.25, 0.20, 0.02)  # plate footprint
        child_dims = (0.10, 0.10, 0.08)  # small object

        ox, oy = support_offset_bounds(
            support_dims=support_dims,
            child_dims=child_dims,
            support_class="plate",
            region_name=None,
            contained=False,
        )
        # base_x = (0.25 - 0.10) / 2 = 0.075; scale = 0.55 → ox = 0.075 * 0.55
        assert ox == pytest.approx(0.075 * 0.55, abs=1e-6)
        assert oy == pytest.approx(0.05 * 0.55, abs=1e-6)

    def test_contained_flag_shrinks_offsets(self):
        """contained=True must shrink both offsets by factor 0.85."""
        from libero_infinity.perturbation_policy import support_offset_bounds

        support_dims = (0.25, 0.20, 0.02)
        child_dims = (0.10, 0.10, 0.08)

        ox_free, oy_free = support_offset_bounds(
            support_dims=support_dims,
            child_dims=child_dims,
            support_class="plate",
            region_name=None,
            contained=False,
        )
        ox_cont, oy_cont = support_offset_bounds(
            support_dims=support_dims,
            child_dims=child_dims,
            support_class="plate",
            region_name=None,
            contained=True,
        )
        assert ox_cont == pytest.approx(ox_free * 0.85, abs=1e-6)
        assert oy_cont == pytest.approx(oy_free * 0.85, abs=1e-6)

    def test_cook_region_keyword_overrides_class_scale(self):
        """region_name containing 'cook_region' must use cook_surface scale (0.72, 0.72)."""
        from libero_infinity.perturbation_policy import support_offset_bounds

        support_dims = (0.24, 0.18, 0.08)
        child_dims = (0.08, 0.08, 0.05)

        ox_cook, _ = support_offset_bounds(
            support_dims=support_dims,
            child_dims=child_dims,
            support_class="flat_stove",
            region_name="cook_region",
            contained=False,
        )
        ox_generic, _ = support_offset_bounds(
            support_dims=support_dims,
            child_dims=child_dims,
            support_class=None,
            region_name=None,
            contained=False,
        )
        # cook_region scale (0.72) > default scale (0.50) → cook should be wider
        assert ox_cook > ox_generic, f"cook_region should produce wider offset ({ox_cook:.4f}) than generic ({ox_generic:.4f})"  # fmt: skip  # noqa: E501


class TestPerturbationPolicyCoordinatedGroupOffset:
    """Tests for coordinated_group_offset() in perturbation_policy.py."""

    def test_single_member_returns_zero(self):
        """With member_count=1, offset must be (0.0, 0.0) — no coordination needed."""
        from libero_infinity.perturbation_policy import coordinated_group_offset

        ox, oy = coordinated_group_offset(
            member_count=1,
            support_dims=(0.8, 0.6, 0.05),
        )
        assert ox == pytest.approx(0.0)
        assert oy == pytest.approx(0.0)

    def test_two_members_returns_nonzero(self):
        """With member_count=2, offset must be non-zero."""
        from libero_infinity.perturbation_policy import coordinated_group_offset

        ox, oy = coordinated_group_offset(
            member_count=2,
            support_dims=(0.8, 0.6, 0.05),
        )
        assert ox > 0.0
        assert oy > 0.0

    def test_more_members_does_not_decrease_offset(self):
        """Offset should not decrease when more members are added (scale caps at 0.30)."""
        from libero_infinity.perturbation_policy import coordinated_group_offset

        dims = (0.8, 0.6, 0.05)
        ox_2, _ = coordinated_group_offset(member_count=2, support_dims=dims)
        ox_5, _ = coordinated_group_offset(member_count=5, support_dims=dims)
        # cap at 0.30 prevents increase beyond that, but should not decrease
        assert ox_5 >= ox_2, f"Offset for 5 members ({ox_5:.4f}) < offset for 2 members ({ox_2:.4f})"  # fmt: skip  # noqa: E501

    def test_scale_caps_at_30_percent(self):
        """Scale must be capped at 0.30 regardless of member_count."""
        from libero_infinity.perturbation_policy import coordinated_group_offset

        dims = (0.8, 0.6, 0.05)
        # At member_count=3: scale = min(0.18 + 0.12, 0.30) = 0.30 already
        ox_3, oy_3 = coordinated_group_offset(member_count=3, support_dims=dims)
        # At member_count=100: scale = min(0.18 + 4.00, 0.30) = 0.30 → same
        ox_100, oy_100 = coordinated_group_offset(member_count=100, support_dims=dims)
        assert ox_3 == pytest.approx(ox_100, abs=1e-6)
        assert oy_3 == pytest.approx(oy_100, abs=1e-6)
