"""Tests for the new perturbation features:

1. ``generate_task_perturbed_bddls`` — destination swaps, predicate
   negations, compositional and color-swap task variants.
2. ``swap_arena`` — table-arena rewrite.
3. ``plan_sensor_noise`` + renderer + simulator transform.
4. ``language_paraphrase.generate_paraphrased_bddls`` — graceful
   no-op without litellm / API key.
"""

from __future__ import annotations

from conftest import BDDL_DIR

from libero_infinity.bddl_preprocessor import (
    _extract_block,
    _parse_language,
    generate_task_perturbed_bddls,
    parse_object_classes,
    swap_arena,
)

_BOWL_PLATE = (BDDL_DIR / "libero_goal" / "put_the_bowl_on_the_plate.bddl").read_text()
_STOVE = (BDDL_DIR / "libero_goal" / "turn_on_the_stove.bddl").read_text()


# ---------------------------------------------------------------------------
# Task perturbations
# ---------------------------------------------------------------------------


def test_task_perturbation_emits_destination_swap() -> None:
    variants = generate_task_perturbed_bddls(
        _BOWL_PLATE,
        include_destination_swaps=True,
        include_predicate_negations=False,
        include_compositional=False,
        include_color_swaps=False,
    )
    assert variants, "expected destination-swap variants on the bowl-on-plate task"
    suffixes = [s for s, _ in variants]
    assert any(s.startswith("_task_dest_") for s in suffixes)
    # The new dest must appear in the (:goal ...) predicate and the
    # language string must mention it.
    suffix, cf_text = variants[0]
    new_dest_class = suffix.removeprefix("_task_dest_")
    goal = _extract_block(cf_text, "goal") or ""
    lang = _parse_language(cf_text).lower()
    assert new_dest_class in goal
    assert new_dest_class.replace("_", " ") in lang


def test_task_perturbation_emits_predicate_negation_for_stove() -> None:
    variants = generate_task_perturbed_bddls(
        _STOVE,
        include_destination_swaps=False,
        include_predicate_negations=True,
        include_compositional=False,
        include_color_swaps=False,
    )
    suffixes = [s for s, _ in variants]
    assert any("_task_neg_" in s for s in suffixes)
    # The negated language should say "Turn off" since original is Turnon.
    cf_text = next(t for s, t in variants if s == "_task_neg_turnoff")
    lang = _parse_language(cf_text).lower()
    assert "turn off" in lang
    goal = _extract_block(cf_text, "goal") or ""
    assert "Turnoff" in goal
    assert "Turnon" not in goal


def test_task_perturbation_emits_compositional_variant() -> None:
    variants = generate_task_perturbed_bddls(
        _BOWL_PLATE,
        include_destination_swaps=False,
        include_predicate_negations=False,
        include_compositional=True,
        include_color_swaps=False,
    )
    assert variants
    suffix, cf_text = variants[0]
    assert suffix.startswith("_task_compose_")
    lang = _parse_language(cf_text).lower()
    assert "and also" in lang


def test_task_perturbation_emits_color_swap_for_known_class() -> None:
    """``akita_black_bowl`` has ``white_bowl`` as a registered visual variant."""
    variants = generate_task_perturbed_bddls(
        _BOWL_PLATE,
        include_destination_swaps=False,
        include_predicate_negations=False,
        include_compositional=False,
        include_color_swaps=True,
    )
    suffixes = [s for s, _ in variants]
    assert any(s == "_task_color_white_bowl" for s in suffixes), suffixes
    # Color swap rewrites the (:objects ...) declaration but keeps the
    # instance name the same.
    cf_text = next(t for s, t in variants if s == "_task_color_white_bowl")
    classes = parse_object_classes(cf_text)
    assert classes.get("akita_black_bowl_1") == "white_bowl"


# ---------------------------------------------------------------------------
# Arena swap
# ---------------------------------------------------------------------------


def test_swap_arena_rewrites_workspace_and_region_targets() -> None:
    """Arena swap renames the workspace fixture instance + class
    everywhere — the (:fixtures ...) declaration, all region targets,
    and any init/goal predicates that reference the old name."""
    out = swap_arena(_BOWL_PLATE, "kitchen_table")
    assert out is not None, "kitchen_table is geometrically compatible with main_table layout"
    assert "kitchen_table - kitchen_table" in out or "kitchen_table_1 - kitchen_table" in out
    assert "main_table - table" not in out
    # Region targets that previously pointed at main_table now point at
    # the new kitchen_table instance.
    assert "(:target main_table)" not in out


def test_swap_arena_returns_none_for_no_op_swap() -> None:
    # main_table is class "table"; swapping to "table" is a no-op.
    assert swap_arena(_BOWL_PLATE, "table") is None


def test_swap_arena_returns_none_for_incompatible_geometry() -> None:
    """A region that extends past the target arena's half-extents must
    abort the swap rather than silently produce off-table placements."""
    # Inject a region that would clip on living_room_table (whose x_half
    # is only 0.35). Insert ranges (-0.40, 0, 0.40, 0) — out of bounds.
    bddl = _BOWL_PLATE.replace(
        "(:goal",
        "(:regions (oversized (:target main_table) (:ranges ((-0.40 0 0.40 0)))))\n  (:goal",
        1,
    )
    out = swap_arena(bddl, "living_room_table")
    assert out is None


# ---------------------------------------------------------------------------
# Sensor noise
# ---------------------------------------------------------------------------


def test_plan_sensor_noise_returns_plan_when_axis_requested() -> None:
    from libero_infinity.ir.nodes import ArticulationModel, PlanDiagnostics
    from libero_infinity.ir.scene_graph import SemanticSceneGraph
    from libero_infinity.planner.axes import plan_sensor_noise

    graph = SemanticSceneGraph(articulation_model=ArticulationModel.canonical())
    diag = PlanDiagnostics()
    plan = plan_sensor_noise(graph, frozenset(["sensor_noise"]), diag)
    assert plan is not None
    assert "gaussian_noise" in plan.kinds
    assert plan.severity_lo == 1
    assert plan.severity_hi == 5


def test_plan_sensor_noise_returns_none_when_axis_not_requested() -> None:
    from libero_infinity.ir.nodes import ArticulationModel, PlanDiagnostics
    from libero_infinity.ir.scene_graph import SemanticSceneGraph
    from libero_infinity.planner.axes import plan_sensor_noise

    graph = SemanticSceneGraph(articulation_model=ArticulationModel.canonical())
    diag = PlanDiagnostics()
    assert plan_sensor_noise(graph, frozenset(["position"]), diag) is None


def test_renderer_emits_sensor_noise_params() -> None:
    from libero_infinity.compiler import generate_scenic
    from libero_infinity.task_config import TaskConfig

    cfg = TaskConfig.from_bddl(str(BDDL_DIR / "libero_goal" / "put_the_bowl_on_the_plate.bddl"))
    program = generate_scenic(cfg, perturbation="sensor_noise")
    assert "param sensor_noise_kind = Uniform(" in program
    assert "param sensor_noise_severity = DiscreteRange(1, 5)" in program


def test_apply_image_corruption_preserves_shape_and_dtype() -> None:
    import numpy as np

    from libero_infinity.simulator import _apply_image_corruption

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    for kind in (
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "gaussian_blur",
        "motion_blur",
        "defocus_blur",
        "jpeg_compression",
        "brightness_jitter",
        "contrast_jitter",
        "saturation_jitter",
    ):
        out = _apply_image_corruption(img, kind, severity=3)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


def test_apply_image_corruption_passes_through_unknown_kind() -> None:
    import numpy as np

    from libero_infinity.simulator import _apply_image_corruption

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = _apply_image_corruption(img, "not_a_real_kind", severity=1)
    np.testing.assert_array_equal(out, img)


# ---------------------------------------------------------------------------
# LLM paraphrasing — graceful fallback when litellm is missing or fails
# ---------------------------------------------------------------------------


def test_paraphrase_returns_empty_when_litellm_missing(monkeypatch) -> None:
    """If litellm is not installed (or any other ImportError), the
    helper returns ``[]`` rather than raising."""
    import sys

    from libero_infinity import language_paraphrase

    # Hide the real ``litellm`` module so the import inside the helper
    # raises ImportError.
    monkeypatch.setitem(sys.modules, "litellm", None)
    assert language_paraphrase.generate_paraphrased_bddls(_BOWL_PLATE) == []


def test_paraphrase_returns_empty_for_zero_variants() -> None:
    from libero_infinity import language_paraphrase

    assert language_paraphrase.generate_paraphrased_bddls(_BOWL_PLATE, n_variants=0) == []


def test_paraphrase_extracts_json_array_from_fenced_response() -> None:
    """The helper handles the common LLM reply shapes (raw array,
    ```json fenced```, or array embedded in surrounding prose)."""
    from libero_infinity.language_paraphrase import _extract_json_array

    raw_array = '["foo", "bar", "baz"]'
    assert _extract_json_array(raw_array) == ["foo", "bar", "baz"]
    fenced = '```json\n["foo", "bar"]\n```'
    assert _extract_json_array(fenced) == ["foo", "bar"]
    embedded = 'sure thing — here you go:\n["a", "b"]\nlet me know if'
    assert _extract_json_array(embedded) == ["a", "b"]
    assert _extract_json_array("not json at all") == []
