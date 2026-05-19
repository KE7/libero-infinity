# BDDL/site names in synthetic graph tuples are unavoidably long; relax E501.
# ruff: noqa: E501
"""Focused regression tests for the object/language rewrite bug-fix bundle.

Each test targets one specific bug:
  1. ``substitute_multi`` chained class-swap collision
  2. Missing fixture-container dims collapsing variant pool
  3. Same-class multi-instance overwrite in ``bddl_for_scene``
  4. ``generate_cf_bddls`` rewriting only the first goal predicate
  5. Container-landmark embedded in region names producing nonsense language
  6. ``obj_of_interest`` rewrite substring collision
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

from conftest import BDDL_DIR

from libero_infinity.bddl_preprocessor import (
    bddl_for_scene,
    generate_cf_bddls,
    parse_object_classes,
    substitute_multi,
    substitute_per_instance,
)

SAMPLE_BDDL = (
    BDDL_DIR / "libero_object" / "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
)


# ---------------------------------------------------------------------------
# Bug 1: substitute_multi class-swap collision
# ---------------------------------------------------------------------------


def test_substitute_multi_avoids_chained_collision() -> None:
    """alphabet_soup→tomato_sauce alongside tomato_sauce→popcorn must not
    chain through the first rewrite (alphabet_soup_1 should land on
    tomato_sauce, NOT popcorn).
    """
    bddl_text = SAMPLE_BDDL.read_text()
    patched = substitute_multi(
        bddl_text,
        {"alphabet_soup": "tomato_sauce", "tomato_sauce": "popcorn"},
    )
    classes = parse_object_classes(patched)
    msg = "alphabet_soup_1 must remain at its first replacement, not chain to popcorn"
    assert classes["alphabet_soup_1"] == "tomato_sauce", msg
    assert classes["tomato_sauce_1"] == "popcorn"


# ---------------------------------------------------------------------------
# Bug 2: fixture-container variant filter collapse
# ---------------------------------------------------------------------------


def test_plan_object_keeps_full_variant_pool_for_fixture_container() -> None:
    """An object marked contained_in a FixtureNode (e.g. microwave) must NOT
    have its variant pool clipped by the missing OBJECT_DIMENSIONS fallback.
    """
    from libero_infinity.ir.nodes import (
        ArticulationModel,
        FixtureNode,
        ObjectNode,
        PlanDiagnostics,
        SceneEdge,
    )
    from libero_infinity.ir.scene_graph import SemanticSceneGraph
    from libero_infinity.planner.axes import plan_object

    graph = SemanticSceneGraph(
        task_language="put mug in microwave",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    microwave = FixtureNode(
        node_id="microwave_1",
        node_type="fixture",
        instance_name="microwave_1",
        object_class="microwave",
        is_articulatable=True,
    )
    mug = ObjectNode(
        node_id="white_yellow_mug_1",
        node_type="object",
        instance_name="white_yellow_mug_1",
        object_class="white_yellow_mug",
        contained=True,
    )
    graph.add_node(microwave)
    graph.add_node(mug)
    graph.add_edge(
        SceneEdge(src_id="white_yellow_mug_1", dst_id="microwave_1", label="contained_in")
    )

    diag = PlanDiagnostics()
    plan = plan_object(graph, frozenset(["object"]), diag)

    # white_yellow_mug has 3 variants in the registry; with the bug they would
    # collapse to 1 (the canonical class) because the (0.08, 0.08, 0.06)
    # fallback interior is smaller than the mug. We just need >= 2 distinct
    # variants surfaced for the substitution choice to be meaningful.
    msg_present = "mug should still get a variant pool even when its container is a fixture"
    assert "white_yellow_mug_1" in plan, msg_present
    pool = plan["white_yellow_mug_1"]
    assert len(pool) >= 2, f"variant pool collapsed to {pool}"


# ---------------------------------------------------------------------------
# Bug 3: same-class multi-instance overwrite in bddl_for_scene
# ---------------------------------------------------------------------------


def test_bddl_for_scene_per_instance_overwrite() -> None:
    """When two instances share an original class but receive *different*
    asset replacements, both replacements must survive — neither instance can
    silently inherit the other's class.
    """
    # Synthesise a tiny BDDL with two same-class instances.
    bddl_text = """(define (problem T)
  (:language do something)
  (:regions
    (bin_region
        (:target floor)
        (:ranges ( (0 0 0 0) ))
    )
  )
  (:fixtures
    floor - floor
  )
  (:objects
    butter_1 butter_2 - butter
    plate_1 - plate
  )
  (:obj_of_interest
    butter_1
    butter_2
  )
  (:init
    (On butter_1 floor_bin_region)
    (On butter_2 floor_bin_region)
    (On plate_1 floor_bin_region)
  )
  (:goal
    (And (On butter_1 plate_1))
  )
)
"""
    src = pathlib.Path("/tmp/_libinf_per_instance_test.bddl")
    src.write_text(bddl_text)
    try:
        scene = SimpleNamespace(
            params={},
            objects=[
                SimpleNamespace(libero_name="butter_1", asset_class="cream_cheese"),
                SimpleNamespace(libero_name="butter_2", asset_class="popcorn"),
            ],
        )
        orig = parse_object_classes(bddl_text)
        with bddl_for_scene(scene, str(src), orig) as tmp:
            patched = parse_object_classes(pathlib.Path(tmp).read_text())
        assert patched["butter_1"] == "cream_cheese"
        assert patched["butter_2"] == "popcorn"
        # The plate must survive untouched.
        assert patched["plate_1"] == "plate"
    finally:
        src.unlink(missing_ok=True)


def test_substitute_per_instance_splits_shared_class_line() -> None:
    """Direct unit test for the per-instance helper."""
    bddl_text = """(define (problem T)
  (:objects
    butter_1 butter_2 butter_3 - butter
  )
  (:goal
    (And (On butter_1 floor))
  )
)
"""
    patched = substitute_per_instance(
        bddl_text, {"butter_1": "cream_cheese", "butter_3": "popcorn"}
    )
    classes = parse_object_classes(patched)
    assert classes["butter_1"] == "cream_cheese"
    assert classes["butter_2"] == "butter"
    assert classes["butter_3"] == "popcorn"


# ---------------------------------------------------------------------------
# Bug 4: generate_cf_bddls multi-predicate goals
# ---------------------------------------------------------------------------


def test_generate_cf_bddls_skips_multi_predicate_goal() -> None:
    """When a goal has more than one On/In predicate, CF generation must
    refuse rather than rewrite only the first.
    """
    bddl_text = """(define (problem T)
  (:language put soup and pudding away)
  (:regions
    (bin_region (:target floor) (:ranges ( (0 0 0 0) )))
  )
  (:fixtures
    floor - floor
  )
  (:objects
    alphabet_soup_1 - alphabet_soup
    chocolate_pudding_1 - chocolate_pudding
    basket_1 - basket
    plate_1 - plate
    cream_cheese_1 - cream_cheese
  )
  (:obj_of_interest
    alphabet_soup_1
    chocolate_pudding_1
  )
  (:init
  )
  (:goal
    (And (On alphabet_soup_1 plate_1) (In chocolate_pudding_1 basket_1_contain_region))
  )
)
"""
    variants = generate_cf_bddls(bddl_text)
    msg = (
        "multi-predicate goals must not generate CF variants — "
        "the language can only describe one swap and the other predicate "
        "would be left referring to the original source object"
    )
    assert variants == [], msg


# ---------------------------------------------------------------------------
# Bug 5: container/landmark region phrase extraction
# ---------------------------------------------------------------------------


def test_generate_cf_bddls_uses_container_landmark_in_language() -> None:
    """A region like ``study_table_desk_caddy_front_left_contain_region``
    must produce language referring to the desk caddy, not the study table.
    """
    bddl_text = """(define (problem T)
  (:language put the book on the desk caddy)
  (:regions
    (desk_caddy_front_left_contain_region (:target study_table) (:ranges ( (0 0 0 0) )))
  )
  (:fixtures
    study_table - study_table
  )
  (:objects
    black_book_1 - black_book
    yellow_book_1 - yellow_book
    porcelain_mug_1 - porcelain_mug
  )
  (:obj_of_interest
    black_book_1
  )
  (:init
  )
  (:goal
    (And (On black_book_1 study_table_desk_caddy_front_left_contain_region))
  )
)
"""
    variants = generate_cf_bddls(bddl_text)
    assert variants, "expected at least one CF variant"
    # Every variant's language should mention the desk caddy, not the study
    # table.
    for _suffix, cf_text in variants:
        landmark_msg = f"variant language lost the caddy landmark: {cf_text!r}"
        assert "desk caddy" in cf_text or "caddy" in cf_text, landmark_msg
        # Concretely, the bug surfaced as "on the study table"
        # in the language string.
        # Find the (:language ...) line and assert.
        lang_line = next((line for line in cf_text.splitlines() if "(:language" in line), "")
        fallback_msg = f"variant language degraded to study-table fallback: {lang_line!r}"
        assert "study table" not in lang_line, fallback_msg


# ---------------------------------------------------------------------------
# Bug 6: bounded obj_of_interest rewrite
# ---------------------------------------------------------------------------


def test_generate_cf_bddls_obj_of_interest_bounded_rewrite() -> None:
    """When obj_of_interest contains an instance whose name is a substring of
    another (``bowl_1`` vs. ``bowl_10``), the CF rewrite must only replace the
    exact source instance, not its substring siblings.
    """
    bddl_text = """(define (problem T)
  (:language put the bowl on the plate)
  (:regions
    (region_a (:target floor) (:ranges ( (0 0 0 0) )))
  )
  (:fixtures
    floor - floor
  )
  (:objects
    akita_black_bowl_1 akita_black_bowl_10 - akita_black_bowl
    plate_1 - plate
    cream_cheese_1 - cream_cheese
  )
  (:obj_of_interest
    akita_black_bowl_1
    akita_black_bowl_10
    plate_1
  )
  (:init
  )
  (:goal
    (And (On akita_black_bowl_1 plate_1))
  )
)
"""
    variants = generate_cf_bddls(bddl_text)
    assert variants, "expected at least one CF variant"
    for suffix, cf_text in variants:
        # akita_black_bowl_10 must remain present, untouched.
        clobber_msg = f"substring sibling akita_black_bowl_10 was clobbered in {suffix}"
        assert "akita_black_bowl_10" in cf_text, clobber_msg
        # The block following (:obj_of_interest must still list both
        # akita_black_bowl_10 and plate_1 — the source (akita_black_bowl_1)
        # was replaced by cf_inst.
        ooi_idx = cf_text.find("(:obj_of_interest")
        assert ooi_idx != -1
        end_idx = cf_text.find(")", ooi_idx)
        ooi_block = cf_text[ooi_idx:end_idx]
        ooi_msg = f"obj_of_interest dropped akita_black_bowl_10: {ooi_block!r}"
        assert "akita_black_bowl_10" in ooi_block, ooi_msg


# ---------------------------------------------------------------------------
# Bug 7: container variant pool admits classes without contain_region site
#
# Root cause for the ``*_contain_region`` family failures in Stage 3 Run 2b
# (see ``rca/stage3_run2b_contain_region_family.md``).  The basket / wooden
# tray / desk caddy variant pools in ``asset_variants.json`` include classes
# that lack a MuJoCo ``contain_region`` site (e.g. ``white_storage_box``,
# ``chefmate_8_frypan``). When the object-axis planner picks one of those
# as a substitute, LIBERO's ``_load_sites_in_arena`` cannot register
# ``<inst>_contain_region`` in ``object_states_dict`` — and the BDDL goal
# predicate ``(In x basket_1_contain_region)`` then raises ``KeyError`` on
# the first ``check_success()`` after settle.  The planner must therefore
# filter the variant pool of any node whose ``_contain_region`` is required
# by an incoming ``contained_in`` edge.
# ---------------------------------------------------------------------------


def test_class_provides_contain_region_probe() -> None:
    """The MJCF probe must correctly classify known container vs non-container
    LIBERO asset classes."""
    from libero_infinity.asset_registry import class_provides_contain_region

    # Container classes — MJCF declares ``name="contain_region"``.
    assert class_provides_contain_region("basket")
    assert class_provides_contain_region("wooden_tray")
    assert class_provides_contain_region("desk_caddy")

    # Non-container classes — present in the basket/tray/caddy variant pools
    # but their MJCF has no ``contain_region`` site.
    assert not class_provides_contain_region("white_storage_box")
    assert not class_provides_contain_region("chefmate_8_frypan")
    # Random graspables should also be False.
    assert not class_provides_contain_region("alphabet_soup")


def test_plan_object_filters_container_variants_lacking_contain_region() -> None:
    """Synthetic-graph variant of the container-affordance filter test.

    Scope: pins ``plan_object``'s filter logic given the production
    ``goal_target`` edge shape (``dst_id == "<node_id>_<contain_site>"``).
    Originally constructed against an inert ``contained_in`` edge per PR #7;
    repaired in ``fix/contain-region-filter-real-graph`` to mirror the real
    graph builder. The real-BDDL regression lives in
    ``tests/test_planner_contain_region_real_graph.py``; this one keeps a
    self-contained unit-level pin on the asset-registry + planner contract.
    """
    from libero_infinity.ir.nodes import (
        ArticulationModel,
        ObjectNode,
        PlanDiagnostics,
        SceneEdge,
    )
    from libero_infinity.ir.scene_graph import SemanticSceneGraph
    from libero_infinity.planner.axes import plan_object

    graph = SemanticSceneGraph(
        task_language="put both items in the basket",
        bddl_path="<test>",
        articulation_model=ArticulationModel.canonical(),
    )
    basket = ObjectNode(
        node_id="basket_1",
        node_type="object",
        instance_name="basket_1",
        object_class="basket",
    )
    soup = ObjectNode(
        node_id="alphabet_soup_1",
        node_type="object",
        instance_name="alphabet_soup_1",
        object_class="alphabet_soup",
        contained=True,
    )
    cheese = ObjectNode(
        node_id="cream_cheese_1",
        node_type="object",
        instance_name="cream_cheese_1",
        object_class="cream_cheese",
        contained=True,
    )
    graph.add_node(basket)
    graph.add_node(soup)
    graph.add_node(cheese)
    # Real graph shape: BDDL ``(In x basket_1_contain_region)`` is emitted as
    # a ``goal_target`` edge whose ``dst_id`` is the full site name. See
    # ``ir/graph_builder.py::build_semantic_scene_graph``.
    graph.add_edge(
        SceneEdge(src_id="alphabet_soup_1", dst_id="basket_1_contain_region", label="goal_target")
    )
    graph.add_edge(
        SceneEdge(src_id="cream_cheese_1", dst_id="basket_1_contain_region", label="goal_target")
    )

    diag = PlanDiagnostics()
    plan = plan_object(graph, frozenset(["object"]), diag)

    # basket_1 must have a pool; every variant in it must preserve contain_region.
    # If it had been pinned to the canonical singleton it would be excluded from
    # the plan (the planner skips no-choice nodes); the basket pool has
    # ["basket", "wooden_tray", "white_storage_box"] and the two container
    # classes survive filtering, so the pool must be present and exclude the
    # non-container.
    from libero_infinity.asset_registry import class_provides_contain_region

    assert "basket_1" in plan, "basket should still have a substitution choice"
    pool = plan["basket_1"]
    _msg_filtered = f"lossy variant lacking contain_region site must be filtered: got {pool!r}"
    assert "white_storage_box" not in pool, _msg_filtered
    for v in pool:
        _msg_variant = f"variant {v!r} kept in container pool despite missing contain_region"
        assert class_provides_contain_region(v), _msg_variant


def test_plan_object_pins_container_when_no_safe_variants() -> None:
    """If no variant of a container preserves contain_region, the planner
    must pin to the canonical class (drop it from the plan, since a 1-class
    pool is a no-op substitution)."""
    # Realistic scenario: temporarily narrow basket's variant pool to only
    # non-container alternatives. After the planner's safety filter, the
    # only surviving variant must be the canonical class (basket itself),
    # which collapses the pool to a singleton and drops basket_1 from the
    # plan dict (plan_object's existing no-choice filter).
    import libero_infinity.asset_registry as ar
    from libero_infinity.ir.nodes import (
        ArticulationModel,
        ObjectNode,
        PlanDiagnostics,
        SceneEdge,
    )
    from libero_infinity.ir.scene_graph import SemanticSceneGraph
    from libero_infinity.planner.axes import plan_object

    container_class = "basket"
    saved = ar.ASSET_VARIANTS[container_class]
    ar.ASSET_VARIANTS[container_class] = [
        container_class,
        "white_storage_box",
        "chefmate_8_frypan",
    ]
    try:
        graph = SemanticSceneGraph(
            task_language="put item in basket",
            bddl_path="<test>",
            articulation_model=ArticulationModel.canonical(),
        )
        cont = ObjectNode(
            node_id="basket_1",
            node_type="object",
            instance_name="basket_1",
            object_class=container_class,
        )
        item = ObjectNode(
            node_id="item_1",
            node_type="object",
            instance_name="item_1",
            object_class="alphabet_soup",
            contained=True,
        )
        graph.add_node(cont)
        graph.add_node(item)
        # Real graph shape (see note in the sibling test above).
        graph.add_edge(
            SceneEdge(src_id="item_1", dst_id="basket_1_contain_region", label="goal_target")
        )

        diag = PlanDiagnostics()
        plan = plan_object(graph, frozenset(["object"]), diag)
        # basket_1 must NOT appear — no contain_region-preserving variants
        # other than canonical, so pinning collapses pool to singleton.
        assert "basket_1" not in plan, (
            "container with no contain_region-preserving variants must pin "
            f"to canonical and be dropped from plan, got {plan!r}"
        )
    finally:
        ar.ASSET_VARIANTS[container_class] = saved
