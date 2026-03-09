"""Unit tests for libero_infinity.grounding — no MuJoCo / LIBERO required.

Tests use lightweight mock objects that replicate the MuJoCo sim contact API:
    mj_sim.data.ncon
    mj_sim.data.contact[i].geom1 / .geom2
    mj_sim.model.geom_id2name(id)
    mj_sim.model.geom_bodyid[id]
    mj_sim.model.body_id2name(id)

Scenarios tested
─────────────────
1. is_faithful=True  — gripper first contacts the target object
2. is_faithful=False — gripper first contacts a non-target object
3. no_contact        — gripper never touches anything (policy froze)
4. aggregate_grounding() — aggregation over multiple GroundingResult objects
"""

from __future__ import annotations

import pytest

from libero_infinity.grounding import (
    GroundingResult,
    GroundingTracker,
    aggregate_grounding,
)

# ---------------------------------------------------------------------------
# Mock MuJoCo sim infrastructure
# ---------------------------------------------------------------------------


class _MockContact:
    """Minimal mjContact-like struct (geom1, geom2 integer IDs)."""

    def __init__(self, geom1: int, geom2: int):
        self.geom1 = geom1
        self.geom2 = geom2


class _MockModel:
    """Minimal MuJoCo model with geom/body lookups."""

    def __init__(
        self,
        geom_names: dict[int, str],
        geom_to_body: dict[int, int],
        body_names: dict[int, str],
    ):
        self._geom_names = geom_names
        self._geom_to_body = geom_to_body
        self._body_names = body_names
        # Expose geom_bodyid as an array-like (supports [] indexing)
        self.geom_bodyid = geom_to_body

    def geom_id2name(self, geom_id: int) -> str:
        if geom_id not in self._geom_names:
            raise ValueError(f"Unknown geom id: {geom_id}")
        return self._geom_names[geom_id]

    def body_id2name(self, body_id: int) -> str:
        if body_id not in self._body_names:
            raise ValueError(f"Unknown body id: {body_id}")
        return self._body_names[body_id]


class _MockData:
    """Minimal MuJoCo data with contact list."""

    def __init__(self, contacts: list[_MockContact]):
        self._contacts = contacts

    @property
    def ncon(self) -> int:
        return len(self._contacts)

    @property
    def contact(self):
        return self._contacts


class _MockSim:
    """Minimal MuJoCo sim (model + data)."""

    def __init__(self, model: _MockModel, data: _MockData):
        self.model = model
        self.data = data


# ---------------------------------------------------------------------------
# Geom / body layout for tests
#
# IDs:
#   0 — gripper0_finger1_collision  (body 0: "gripper0_body")
#   1 — gripper0_finger2_collision  (body 0: "gripper0_body")
#   2 — akita_black_bowl_collision  (body 1: "akita_black_bowl_1_main")
#   3 — red_mug_collision           (body 2: "red_mug_1_main")
# ---------------------------------------------------------------------------

_GEOM_NAMES = {
    0: "gripper0_finger1_collision",
    1: "gripper0_finger2_collision",
    2: "akita_black_bowl_collision",
    3: "red_mug_collision",
}
_GEOM_TO_BODY = {
    0: 0,  # gripper finger 1 → gripper body
    1: 0,  # gripper finger 2 → gripper body
    2: 1,  # bowl geom → bowl body
    3: 2,  # mug geom  → mug body
}
_BODY_NAMES = {
    0: "gripper0_body",
    1: "akita_black_bowl_1_main",
    2: "red_mug_1_main",
}

_MODEL = _MockModel(_GEOM_NAMES, _GEOM_TO_BODY, _BODY_NAMES)


def _make_sim(contacts: list[_MockContact]) -> _MockSim:
    """Construct a mock sim with the shared model and given contact list."""
    return _MockSim(_MODEL, _MockData(contacts))


def _make_empty_sim() -> _MockSim:
    """Sim with no active contacts."""
    return _make_sim([])


def _make_gripper_bowl_contact() -> _MockSim:
    """Sim where gripper finger 1 contacts the bowl."""
    return _make_sim([_MockContact(geom1=0, geom2=2)])  # finger1 ↔ bowl


def _make_gripper_mug_contact() -> _MockSim:
    """Sim where gripper finger 2 contacts the mug."""
    return _make_sim([_MockContact(geom1=1, geom2=3)])  # finger2 ↔ mug


# ---------------------------------------------------------------------------
# Scenario 1: is_faithful=True
# ---------------------------------------------------------------------------


class TestFaithfulEpisode:
    """Gripper first contacts the language-specified target → is_faithful=True."""

    def test_is_faithful_after_single_step(self):
        """One step with bowl contact → faithful result."""
        tracker = GroundingTracker(
            target_object_name="akita_black_bowl",
            episode_id="ep_faithful",
        )
        tracker.step(_make_gripper_bowl_contact())
        result = tracker.result(task_success=True)

        assert result.is_faithful is True
        assert result.first_contact_object == "akita_black_bowl"
        assert result.contact_timestep == 0
        assert result.task_success is True
        assert result.episode_id == "ep_faithful"
        assert result.target_object == "akita_black_bowl"

    def test_is_faithful_after_several_empty_then_contact(self):
        """Several no-contact steps before first contact → faithful, correct timestep."""
        tracker = GroundingTracker(target_object_name="akita_black_bowl")

        # 3 steps with no contacts
        for _ in range(3):
            tracker.step(_make_empty_sim())

        # Contact at step index 3
        tracker.step(_make_gripper_bowl_contact())
        result = tracker.result(task_success=False)

        assert result.is_faithful is True
        assert result.contact_timestep == 3

    def test_first_contact_wins_even_if_later_contact_differs(self):
        """If bowl touched first, mug contact later must not override the result."""
        tracker = GroundingTracker(target_object_name="akita_black_bowl")

        tracker.step(_make_gripper_bowl_contact())  # step 0 — target
        tracker.step(_make_gripper_mug_contact())  # step 1 — non-target, ignored
        result = tracker.result(task_success=True)

        assert result.is_faithful is True
        assert result.first_contact_object == "akita_black_bowl"
        assert result.contact_timestep == 0

    def test_geom_order_reversed_still_faithful(self):
        """Contact pair (bowl_geom, gripper_geom) — reversed order still detected."""
        sim = _make_sim([_MockContact(geom1=2, geom2=0)])  # bowl ↔ finger1
        tracker = GroundingTracker(target_object_name="akita_black_bowl")
        tracker.step(sim)
        result = tracker.result(task_success=True)

        assert result.is_faithful is True


# ---------------------------------------------------------------------------
# Scenario 2: is_faithful=False
# ---------------------------------------------------------------------------


class TestBiasedEpisode:
    """Gripper first contacts a non-target object → is_faithful=False."""

    def test_is_faithful_false_when_gripper_contacts_wrong_object(self):
        """Target is bowl but gripper grabs mug first → biased."""
        tracker = GroundingTracker(
            target_object_name="akita_black_bowl",
            episode_id="ep_biased",
        )
        tracker.step(_make_gripper_mug_contact())
        result = tracker.result(task_success=True)

        assert result.is_faithful is False
        assert result.first_contact_object == "red_mug"
        assert result.contact_timestep == 0

    def test_success_does_not_imply_faithful(self):
        """Task succeeded via shortcut: grabs wrong object, still completes."""
        tracker = GroundingTracker(target_object_name="akita_black_bowl")
        tracker.step(_make_gripper_mug_contact())
        result = tracker.result(task_success=True)

        assert result.is_faithful is False
        assert result.task_success is True

    def test_biased_failed_episode(self):
        """Biased (wrong first contact) AND task failed."""
        tracker = GroundingTracker(target_object_name="akita_black_bowl")
        tracker.step(_make_gripper_mug_contact())
        result = tracker.result(task_success=False)

        assert result.is_faithful is False
        assert result.task_success is False


# ---------------------------------------------------------------------------
# Scenario 3: no_contact
# ---------------------------------------------------------------------------


class TestNoContactEpisode:
    """Gripper never touches anything (policy froze / episode ends early)."""

    def test_no_contact_result(self):
        """All steps empty → first_contact_object=None, is_faithful=False."""
        tracker = GroundingTracker(
            target_object_name="akita_black_bowl",
            episode_id="ep_nocontact",
        )
        for _ in range(10):
            tracker.step(_make_empty_sim())
        result = tracker.result(task_success=False)

        assert result.first_contact_object is None
        assert result.is_faithful is False
        assert result.contact_timestep is None
        assert result.task_success is False

    def test_no_contact_zero_steps(self):
        """Episode ends immediately without any steps."""
        tracker = GroundingTracker(target_object_name="akita_black_bowl")
        result = tracker.result(task_success=False)

        assert result.first_contact_object is None
        assert result.is_faithful is False

    def test_empty_target_name_always_biased(self):
        """If target_object_name is empty string, is_faithful must be False even on contact."""
        tracker = GroundingTracker(target_object_name="")
        tracker.step(_make_gripper_bowl_contact())
        result = tracker.result(task_success=True)

        # Empty target → cannot be faithful (no defined target to match)
        assert result.is_faithful is False


# ---------------------------------------------------------------------------
# Scenario 4: aggregate_grounding()
# ---------------------------------------------------------------------------


class TestAggregateGrounding:
    """aggregate_grounding() correctly computes summary statistics."""

    def _make_result(
        self,
        *,
        is_faithful: bool,
        task_success: bool,
        first_contact: str | None = "akita_black_bowl",
    ) -> GroundingResult:
        return GroundingResult(
            episode_id="ep_x",
            target_object="akita_black_bowl",
            first_contact_object=first_contact,
            is_faithful=is_faithful,
            contact_timestep=5 if first_contact else None,
            task_success=task_success,
        )

    def test_empty_results_returns_empty_dict(self):
        assert aggregate_grounding([]) == {}

    def test_all_faithful_all_success(self):
        results = [self._make_result(is_faithful=True, task_success=True) for _ in range(4)]
        metrics = aggregate_grounding(results)

        assert metrics["n_episodes"] == 4
        assert metrics["grounding_rate"] == pytest.approx(1.0)
        assert metrics["success_rate"] == pytest.approx(1.0)
        assert metrics["faithful_success_rate"] == pytest.approx(1.0)
        assert metrics["biased_rate"] == pytest.approx(0.0)
        assert metrics["no_contact_rate"] == pytest.approx(0.0)

    def test_all_biased_all_success(self):
        """High success rate + low grounding rate = visual shortcut scenario."""
        results = [
            self._make_result(is_faithful=False, task_success=True, first_contact="red_mug")
            for _ in range(4)
        ]
        metrics = aggregate_grounding(results)

        assert metrics["grounding_rate"] == pytest.approx(0.0)
        assert metrics["success_rate"] == pytest.approx(1.0)
        assert metrics["faithful_success_rate"] == pytest.approx(0.0)
        assert metrics["biased_rate"] == pytest.approx(1.0)

    def test_mixed_episodes(self):
        """3 faithful-success, 1 biased-success, 1 no-contact-failure."""
        results = [
            self._make_result(is_faithful=True, task_success=True),  # faithful + success
            self._make_result(is_faithful=True, task_success=True),  # faithful + success
            self._make_result(is_faithful=True, task_success=True),  # faithful + success
            self._make_result(is_faithful=False, task_success=True, first_contact="red_mug"),
            self._make_result(is_faithful=False, task_success=False, first_contact=None),
        ]
        metrics = aggregate_grounding(results)

        assert metrics["n_episodes"] == 5
        assert metrics["grounding_rate"] == pytest.approx(3 / 5)
        assert metrics["success_rate"] == pytest.approx(4 / 5)
        assert metrics["faithful_success_rate"] == pytest.approx(3 / 5)
        # biased = touched wrong object (1 episode); no-contact is separate (1 episode)
        assert metrics["biased_rate"] == pytest.approx(1 / 5)
        assert metrics["no_contact_rate"] == pytest.approx(1 / 5)

    def test_all_no_contact(self):
        results = [
            self._make_result(is_faithful=False, task_success=False, first_contact=None)
            for _ in range(3)
        ]
        metrics = aggregate_grounding(results)

        assert metrics["no_contact_rate"] == pytest.approx(1.0)
        assert metrics["grounding_rate"] == pytest.approx(0.0)

    def test_three_rates_sum_to_one(self):
        """grounding_rate + biased_rate + no_contact_rate == 1.0 always."""
        results = [
            self._make_result(is_faithful=True, task_success=True),
            self._make_result(is_faithful=False, task_success=False, first_contact="red_mug"),
            self._make_result(is_faithful=False, task_success=False, first_contact=None),
        ]
        metrics = aggregate_grounding(results)

        # Each episode falls into exactly one of the three buckets
        assert (
            metrics["grounding_rate"] + metrics["biased_rate"] + metrics["no_contact_rate"]
        ) == pytest.approx(1.0)
        # With 1 faithful, 1 biased, 1 no-contact — all equal thirds
        assert metrics["grounding_rate"] == pytest.approx(1 / 3)
        assert metrics["biased_rate"] == pytest.approx(1 / 3)
        assert metrics["no_contact_rate"] == pytest.approx(1 / 3)

    def test_biased_rate_excludes_no_contact(self):
        """A policy that never moves: biased_rate=0, no_contact_rate=1."""
        results = [
            self._make_result(is_faithful=False, task_success=False, first_contact=None)
            for _ in range(5)
        ]
        metrics = aggregate_grounding(results)

        # Robot never moved — not "biased", just frozen
        assert metrics["biased_rate"] == pytest.approx(0.0)
        assert metrics["no_contact_rate"] == pytest.approx(1.0)
        assert metrics["grounding_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Scenario 5: counterfactual (CF) grounding
#
# The CF scenario is the core of the language-ablation pattern:
#   - Scene contains the training-canonical object (bowl) AND a second object (mug)
#   - Instruction names the NON-canonical object as target ("put the mug on the plate")
#   - A biased policy ignores the instruction and defaults to the canonical object
#   - A faithful policy follows the instruction and grabs the mug
#
# All mechanics are already covered by Scenarios 1-2, but these tests
# explicitly frame the CF failure mode so regressions are caught.
# ---------------------------------------------------------------------------


class TestCounterfactualGrounding:
    """Grounding tracker with a non-canonical instruction target in a two-object scene.

    Scene: bowl (canonical training object) + mug (alternative target).
    Instruction: "put the red_mug on the plate" — the non-default object.
    """

    def test_cf_faithful_policy_grabs_instructed_object(self):
        """Policy follows the CF instruction and grabs the mug → faithful."""
        tracker = GroundingTracker(target_object_name="red_mug")
        tracker.step(_make_gripper_mug_contact())  # correctly grabs mug
        result = tracker.result(task_success=True)

        assert result.is_faithful is True
        assert result.first_contact_object == "red_mug"

    def test_cf_biased_policy_defaults_to_canonical_object(self):
        """Policy ignores CF instruction and grabs bowl (training default) → biased."""
        tracker = GroundingTracker(target_object_name="red_mug")
        tracker.step(_make_gripper_bowl_contact())  # grabs canonical bowl instead
        result = tracker.result(task_success=False)

        assert result.is_faithful is False
        assert result.first_contact_object == "akita_black_bowl"

    def test_cf_biased_policy_can_still_succeed_via_shortcut(self):
        """Biased policy grabs the wrong object first but task still completes.

        This replicates the VLA-VA Table I finding: a model can score high on
        task success while systematically ignoring language instructions.
        """
        tracker = GroundingTracker(target_object_name="red_mug")
        tracker.step(_make_gripper_bowl_contact())
        result = tracker.result(task_success=True)  # success via visual shortcut

        assert result.is_faithful is False
        assert result.task_success is True

    def test_cf_aggregate_reveals_language_following_gap(self):
        """Aggregate over episodes: grounding_rate << success_rate exposes the gap.

        A policy that achieves 100% success but 0% grounding is purely vision-driven.
        The gap (success - grounding) is the language-contribution metric from the
        README ablation pattern.
        """
        # 10 episodes: instruction says "mug", robot always grabs bowl, always succeeds
        results = [
            GroundingResult(
                episode_id=f"ep_{i}",
                target_object="red_mug",
                first_contact_object="akita_black_bowl",  # wrong object
                is_faithful=False,
                contact_timestep=5,
                task_success=True,  # but task "succeeds" anyway
            )
            for i in range(10)
        ]
        metrics = aggregate_grounding(results)

        assert metrics["success_rate"] == pytest.approx(1.0)
        assert metrics["grounding_rate"] == pytest.approx(0.0)
        assert metrics["biased_rate"] == pytest.approx(1.0)
        # language_contribution = success_rate - grounding_rate = 1.0
        # (model is entirely vision-driven)
        assert (metrics["success_rate"] - metrics["grounding_rate"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Body-name normalisation edge cases
# ---------------------------------------------------------------------------


class TestBodyNameNormalisation:
    """Verify that grounding tracker handles varied LIBERO body-name patterns."""

    def _geom_names_for(self, body_name: str) -> dict[int, str]:
        """Build geom/body maps for a custom body name."""
        return {
            0: "gripper0_finger1_collision",
            1: f"{body_name}_collision",
        }

    def _make_sim_with_body(self, body_name: str) -> _MockSim:
        geom_names = {
            0: "gripper0_finger1_collision",
            1: "custom_geom_collision",
        }
        geom_to_body = {0: 0, 1: 1}
        body_names = {0: "gripper0_body", 1: body_name}
        model = _MockModel(geom_names, geom_to_body, body_names)
        data = _MockData([_MockContact(geom1=0, geom2=1)])
        return _MockSim(model, data)

    @pytest.mark.parametrize(
        "body_name, target, expect_faithful",
        [
            ("akita_black_bowl_1_main", "akita_black_bowl", True),
            ("akita_black_bowl_1", "akita_black_bowl", True),
            ("red_mug_1_main", "akita_black_bowl", False),
            ("red_mug_2", "red_mug", True),
            ("target_1_main", "target", True),
        ],
    )
    def test_normalisation(self, body_name: str, target: str, expect_faithful: bool):
        sim = self._make_sim_with_body(body_name)
        tracker = GroundingTracker(target_object_name=target)
        tracker.step(sim)
        result = tracker.result(task_success=False)
        assert result.is_faithful is expect_faithful, (
            f"body_name={body_name!r}, target={target!r}: "
            f"expected is_faithful={expect_faithful}, got {result.is_faithful}. "
            f"first_contact_object={result.first_contact_object!r}"
        )
