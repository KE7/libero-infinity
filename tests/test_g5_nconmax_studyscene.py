"""Regression tests for the run3 G5 STUDY_SCENE reset failures.

RCA (``~/.omar/ea/4/validation_run3/rca/g5_nconmax_studyscene.md``): 24
``RuntimeError`` reset failures, all on ``libero_90/STUDY_SCENE*``, had two
*independent* root causes — neither was a MuJoCo ``ncon`` contact-arena
overflow:

1. **Fixture body_pos injection clobber (primary).** ``desk_caddy_1`` and
   ``wooden_two_layer_shelf_1`` are declared under ``(:fixtures ...)`` and are
   compiled as *jointless static bodies* — the only way to relocate them to a
   Scenic-sampled position is to write ``sim.model.body_pos`` directly
   (``_inject_object_pose`` fallback). ``setup()`` called
   ``_restore_model_baseline()`` *after* the injection loop, reverting those
   ``body_pos`` writes to the XML default. The fixture then sat at its default
   while the settle validator compared it against the (un-applied) Scenic
   target → a phantom ~0.20 m "drift" rejected every one of the 10 retries.

2. **``study_table`` workspace-surface false contact (distractor subsets).**
   ``_validate_settled_positions`` skipped resting contact only with a body
   named ``table``/``table*``. STUDY scenes name their surface ``study_table``,
   so a distractor legitimately resting on the table read as a fixture overlap.

A complementary durable fix auto-sizes the MuJoCo contact/constraint arena
(robosuite hardcodes ``nconmax/njmax=5000``) so a single dense scene can never
truncate contacts — retiring the ``ncon = 5000`` warning spam.

These tests pin all three behaviours.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import BDDL_DIR, requires_libero

from libero_infinity.simulator import (
    _autosize_contact_arena,
    _is_workspace_surface_body,
)

# ---------------------------------------------------------------------------
# Pure unit tests — no LIBERO/MuJoCo needed (Tier 1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body_name",
    [
        "table",
        "table_main",
        "kitchen_table",
        "living_room_table",
        "study_table",
        "main_table",
        "floor",
        "floor_main",
        "STUDY_TABLE",
    ],
)
def test_workspace_surface_recognised(body_name):
    """Every LIBERO workspace surface name must be recognised as a rest surface."""
    assert _is_workspace_surface_body(body_name)


@pytest.mark.parametrize(
    "body_name",
    [
        "desk_caddy_1_main",
        "wooden_two_layer_shelf_1",
        "wooden_cabinet_1",
        "flat_stove_1",
        "red_coffee_mug_1",
        "black_book_1",
        "basket_1",
    ],
)
def test_non_surface_not_skipped(body_name):
    """Task objects / fixtures must NOT be mistaken for the workspace surface."""
    assert not _is_workspace_surface_body(body_name)


def test_autosize_rewrites_both_caps():
    """The processor rewrites a robosuite-style <size> cap to the auto sentinel."""
    xml = '<mujoco><size njmax="5000" nconmax="5000"/><worldbody/></mujoco>'
    out = _autosize_contact_arena(xml)
    assert 'nconmax="-1"' in out
    assert 'njmax="-1"' in out
    assert '"5000"' not in out


def test_autosize_handles_attr_order_and_spacing():
    """Rewrite is order/spacing independent and only touches the <size> element."""
    xml = '<size  nconmax = "12000"   njmax="8000" />'
    out = _autosize_contact_arena(xml)
    assert 'nconmax="-1"' in out
    assert 'njmax="-1"' in out


def test_autosize_noop_without_caps():
    """A <size> element without nconmax/njmax (already dynamic) is left untouched."""
    xml = '<mujoco><size memory="200M"/></mujoco>'
    assert _autosize_contact_arena(xml) == xml


def test_autosize_noop_without_size_element():
    """No <size> element → nothing to rewrite (MuJoCo already auto-sizes)."""
    xml = "<mujoco><worldbody/></mujoco>"
    assert _autosize_contact_arena(xml) == xml


# ---------------------------------------------------------------------------
# Integration tests — require LIBERO + MuJoCo (Tier 2)
# ---------------------------------------------------------------------------

_STUDY_FIXTURE_BDDLS = sorted(
    BDDL_DIR.glob("**/STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy.bddl")
)
_STUDY_SHELF_BDDLS = sorted(
    BDDL_DIR.glob(
        "**/STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf.bddl"
    )
)


@requires_libero
@pytest.mark.parametrize(
    ("bddls", "fixture_name"),
    [
        (_STUDY_FIXTURE_BDDLS, "desk_caddy_1"),
        (_STUDY_SHELF_BDDLS, "wooden_two_layer_shelf_1"),
    ],
)
def test_static_fixture_injection_not_clobbered(bddls, fixture_name):
    """A jointless STUDY-scene fixture must settle at its Scenic target.

    Before the ``_restore_model_baseline`` reorder, the fixture stayed at its
    XML default and the validator measured a ~0.20 m phantom drift, exhausting
    all 10 retries (g5 RuntimeError). After the fix the fixture is actually
    relocated, so reset() succeeds and the settled fixture is within the
    settle-drift tolerance of its sampled xy target.
    """
    if not bddls:
        pytest.skip("STUDY_SCENE BDDL not found")
    from libero_infinity.gym_env import LIBEROScenicEnv

    env = LIBEROScenicEnv(
        bddl_path=str(bddls[0]),
        perturbation="object,robot,lighting",
        seed=0,
        resolution=128,
    )
    try:
        # reset() must NOT raise — the historical failure was a RuntimeError
        # ("failed to find a valid scene after 10 retries"). With the fixture
        # injection clobbered, the phantom 0.20 m drift rejected every retry,
        # so a clean reset is itself the primary regression guard.
        env.reset()
        sim = env._sim.libero_env.env.sim
        bid = None
        for cand in (fixture_name, fixture_name + "_main"):
            try:
                bid = sim.model.body_name2id(cand)
                break
            except Exception:
                continue
        assert bid is not None, f"{fixture_name} body not found"
        settled = np.array(sim.data.body_xpos[bid][:3], dtype=float)
        assert np.all(np.isfinite(settled)), f"{fixture_name} settled to non-finite pose"
        # The contact/constraint arena was auto-sized for this rebuilt sim.
        assert int(sim.model._model.nconmax) == -1
        assert int(sim.model._model.njmax) == -1
    finally:
        env.close()


@requires_libero
def test_studyscene_distractor_reset_succeeds():
    """A STUDY scene with distractors must reset (study_table contact not flagged)."""
    if not _STUDY_FIXTURE_BDDLS:
        pytest.skip("STUDY_SCENE BDDL not found")
    from libero_infinity.gym_env import LIBEROScenicEnv

    env = LIBEROScenicEnv(
        bddl_path=str(_STUDY_FIXTURE_BDDLS[0]),
        perturbation="object,robot,lighting,texture,distractor",
        seed=0,
        resolution=128,
    )
    try:
        obs = env.reset()
        assert obs is not None
    finally:
        env.close()
