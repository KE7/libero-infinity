import pathlib

from conftest import BDDL_DIR, BOWL_BDDL, OPEN_DRAWER_BDDL, STOVE_BDDL


def _find_bddl(glob_pattern: str) -> pathlib.Path:
    matches = list(BDDL_DIR.glob(glob_pattern))
    assert matches, f"Missing BDDL matching {glob_pattern!r}"
    return matches[0]


PUSH_PLATE_BDDL = _find_bddl("**/push_the_plate_to_the_front_of_the_stove.bddl")
TURN_OFF_STOVE_BDDL = _find_bddl("**/*turn_off_the_stove.bddl")
MICROWAVE_CLOSE_BDDL = _find_bddl(
    "**/*put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl"
)
STACK_TRAY_BDDL = _find_bddl(
    "**/*stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray.bddl"
)
SHARED_STOVE_BDDL = _find_bddl("**/*put_both_moka_pots_on_the_stove.bddl")
RIGHT_OF_PLATE_BDDL = _find_bddl("**/*put_the_white_bowl_to_the_right_of_the_plate.bddl")


class TestTaskSemantics:
    def test_task_config_preserves_init_text_region_refs_and_yaw(self):
        from libero_infinity.task_config import TaskConfig

        cfg = TaskConfig.from_bddl(BOWL_BDDL)

        assert "(On akita_black_bowl_1 main_table_akita_black_bowl_region)" in cfg.init_text
        assert "main_table_cabinet_region" in cfg.region_refs
        assert cfg.region_refs["main_table_cabinet_region"].yaw_min == 3.141592653589793
        assert cfg.region_refs["main_table_cabinet_region"].yaw_max == 3.141592653589793

    def test_atomic_goal_predicates_cover_supported_types(self):
        from libero_infinity.task_config import TaskConfig

        bowl_cfg = TaskConfig.from_bddl(BOWL_BDDL)
        in_cfg = TaskConfig.from_bddl(
            _find_bddl("**/open_the_top_drawer_and_put_the_bowl_inside.bddl")
        )
        open_cfg = TaskConfig.from_bddl(OPEN_DRAWER_BDDL)
        close_cfg = TaskConfig.from_bddl(MICROWAVE_CLOSE_BDDL)
        turnon_cfg = TaskConfig.from_bddl(STOVE_BDDL)
        turnoff_cfg = TaskConfig.from_bddl(TURN_OFF_STOVE_BDDL)

        on_atom = bowl_cfg.semantics.goal_predicates[0]
        assert on_atom.predicate == "On"
        assert on_atom.primary_name == "akita_black_bowl_1"
        assert on_atom.target_name == "plate_1"
        assert on_atom.target_kind == "object"

        in_atom = in_cfg.semantics.goal_predicates[0]
        assert in_atom.predicate == "In"
        assert in_atom.target_name == "wooden_cabinet_1_top_region"
        assert in_atom.target_kind == "region"
        assert in_atom.support_instance_name == "wooden_cabinet_1"
        assert in_atom.support_region_name == "top_region"

        open_atom = open_cfg.semantics.goal_predicates[0]
        assert open_atom.predicate == "Open"
        assert open_atom.primary_name == "wooden_cabinet_1_middle_region"
        assert open_atom.primary_kind == "region"
        assert open_atom.support_instance_name == "wooden_cabinet_1"

        close_atom = close_cfg.semantics.goal_predicates[1]
        assert close_atom.predicate == "Close"
        assert close_atom.primary_name == "microwave_1"
        assert close_atom.primary_kind == "fixture"

        turnon_atom = turnon_cfg.semantics.goal_predicates[0]
        assert turnon_atom.predicate == "Turnon"
        assert turnon_atom.primary_name == "flat_stove_1"

        turnoff_atom = turnoff_cfg.semantics.goal_predicates[0]
        assert turnoff_atom.predicate == "Turnoff"
        assert turnoff_atom.primary_name == "flat_stove_1"

    def test_bounded_goal_region_exclusions_are_derived_from_goal_regions(self):
        from libero_infinity.task_config import TaskConfig

        cfg = TaskConfig.from_bddl(PUSH_PLATE_BDDL)
        exclusions = cfg.semantics.goal_region_exclusions

        assert len(exclusions) == 1
        exclusion = exclusions[0]
        assert exclusion.object_name == "plate_1"
        assert exclusion.region_full_name == "main_table_stove_front_region"
        assert exclusion.support_instance_name == "main_table"
        assert exclusion.bounds == (
            -0.09,
            -0.010000000000000002,
            0.16999999999999998,
            0.25,
        )

    def test_support_graph_labels_cover_root_fixture_fixture_container_and_object_stack(
        self,
    ):
        from libero_infinity.task_config import TaskConfig

        bowl_cfg = TaskConfig.from_bddl(BOWL_BDDL)
        contained_cfg = TaskConfig.from_bddl(
            _find_bddl(
                "**/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate.bddl"
            )
        )
        stacked_cfg = TaskConfig.from_bddl(STACK_TRAY_BDDL)

        bowl_edge = next(
            edge
            for edge in bowl_cfg.semantics.init_support_graph
            if edge.child_name == "akita_black_bowl_1"
        )
        assert bowl_edge.relation == "on_root_region"
        assert bowl_edge.parent_name == "main_table"

        contained_edge = next(
            edge
            for edge in contained_cfg.semantics.init_support_graph
            if edge.child_name == "akita_black_bowl_1"
        )
        assert contained_edge.relation == "in_fixture_region"
        assert contained_edge.parent_name == "wooden_cabinet_1"
        assert contained_edge.region_full_name == "wooden_cabinet_1_top_region"

        stacked_goal_edge = next(
            edge
            for edge in stacked_cfg.semantics.goal_support_graph
            if edge.child_name == "akita_black_bowl_1"
        )
        tray_goal_edge = next(
            edge
            for edge in stacked_cfg.semantics.goal_support_graph
            if edge.child_name == "akita_black_bowl_2"
        )
        assert stacked_goal_edge.relation == "on_object"
        assert stacked_goal_edge.parent_name == "akita_black_bowl_2"
        assert tray_goal_edge.relation == "in_object_region"
        assert tray_goal_edge.parent_name == "wooden_tray_1"

    def test_articulatable_fixture_metadata_covers_drawers_and_microwaves(self):
        from libero_infinity.task_config import TaskConfig

        drawer_cfg = TaskConfig.from_bddl(OPEN_DRAWER_BDDL)
        microwave_cfg = TaskConfig.from_bddl(MICROWAVE_CLOSE_BDDL)

        drawer_meta = drawer_cfg.semantics.articulatable_fixtures[0]
        assert drawer_meta.fixture_name == "wooden_cabinet_1"
        assert drawer_meta.family == "cabinet"
        assert drawer_meta.articulation_kind == "drawer"
        assert drawer_meta.control_target_name == "wooden_cabinet_1_middle_region"
        assert drawer_meta.compartment_name == "middle"
        assert drawer_meta.goal_state == "open"

        microwave_meta = microwave_cfg.semantics.articulatable_fixtures[0]
        assert microwave_meta.fixture_name == "microwave_1"
        assert microwave_meta.family == "microwave"
        assert microwave_meta.articulation_kind == "door"
        assert microwave_meta.control_target_name == "microwave_1"
        assert microwave_meta.compartment_name == "heating"
        assert microwave_meta.init_state == "open"
        assert microwave_meta.goal_state == "closed"

    def test_visibility_targets_include_goal_objects_supports_and_regions(self):
        from libero_infinity.task_config import TaskConfig

        cfg = TaskConfig.from_bddl(MICROWAVE_CLOSE_BDDL)
        names = {(target.name, target.kind) for target in cfg.semantics.visibility_targets}

        assert ("white_yellow_mug_1", "object") in names
        assert ("microwave_1_heating_region", "region") in names
        assert ("microwave_1", "fixture") in names

    def test_yaw_hints_include_init_and_goal_regions(self):
        from libero_infinity.task_config import TaskConfig

        bowl_cfg = TaskConfig.from_bddl(BOWL_BDDL)
        goal_cfg = TaskConfig.from_bddl(RIGHT_OF_PLATE_BDDL)

        bowl_hint = next(
            hint for hint in bowl_cfg.semantics.yaw_hints if hint.entity_name == "wooden_cabinet_1"
        )
        assert bowl_hint.phase == "init"
        assert bowl_hint.region_full_name == "main_table_cabinet_region"
        assert bowl_hint.yaw_range == (3.141592653589793, 3.141592653589793)

        goal_hint = next(
            hint for hint in goal_cfg.semantics.yaw_hints if hint.entity_name == "white_bowl_1"
        )
        assert goal_hint.phase == "goal"
        assert goal_hint.region_full_name == "kitchen_table_plate_right_region"
        assert goal_hint.yaw_range == (0.0, 0.0)

    def test_coordination_groups_cover_direct_and_transitive_shared_support_layouts(
        self,
    ):
        from libero_infinity.task_config import TaskConfig

        shared_cfg = TaskConfig.from_bddl(SHARED_STOVE_BDDL)
        stacked_cfg = TaskConfig.from_bddl(STACK_TRAY_BDDL)

        shared_group = shared_cfg.semantics.coordination_groups[0]
        assert shared_group.root_support_name == "flat_stove_1_cook_region"
        assert shared_group.member_names == ("moka_pot_1", "moka_pot_2")

        stacked_group = stacked_cfg.semantics.coordination_groups[0]
        assert stacked_group.root_support_name == "wooden_tray_1_contain_region"
        assert stacked_group.member_names == (
            "akita_black_bowl_1",
            "akita_black_bowl_2",
        )
