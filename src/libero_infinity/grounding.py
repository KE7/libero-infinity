"""Grounding rate tracker for LIBERO episodes.

Grounding rate measures whether the robot's gripper first contacts the
*language-specified* target object rather than a training-canonical distractor.

Distinct from success rate:
  - Faithful episode: gripper first touches the target object
  - Biased episode: gripper first touches a non-target (usually
    training-canonical) object — success rate may stay high via shortcuts

Based on Section 1 of docs/eval_plan.md (VLA-VA paper, arXiv 2602.17659).

Typical usage in an eval loop
──────────────────────────────
    target = scene.params.get("target_object_name", "")
    tracker = GroundingTracker(target_object_name=target, episode_id="ep_0000")

    obs = env.reset()
    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        tracker.step(mj_sim)       # mj_sim = libero_env.env.sim
        if done:
            break

    result = tracker.result(task_success=env.check_success())

MuJoCo contact API used
────────────────────────
    mj_sim.data.ncon              — number of active contacts
    mj_sim.data.contact[i].geom1 — geom ID of first geom in contact pair
    mj_sim.data.contact[i].geom2 — geom ID of second geom in contact pair
    mj_sim.model.geom_id2name(id) — geom ID → geom name string
    mj_sim.model.geom_bodyid[id]  — geom ID → body ID (int array)
    mj_sim.model.body_id2name(id) — body ID → body name string
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Gripper collision geoms for the Franka Panda in LIBERO
GRIPPER_GEOMS: frozenset[str] = frozenset(
    {
        "gripper0_finger1_collision",
        "gripper0_finger2_collision",
    }
)

# Matches trailing instance number(s) and optional "_main" suffix.
# Examples: "akita_black_bowl_1_main" → "akita_black_bowl"
#           "red_mug_1"               → "red_mug"
#           "distractor_0_main"       → "distractor"
_BODY_SUFFIX_RE = re.compile(r"(_\d+)+(_main)?$")


def _normalize_body_name(body_name: str) -> str:
    """Strip trailing instance number(s) and '_main' from a MuJoCo body name.

    Args:
        body_name: Raw body name from the MuJoCo model, e.g. "akita_black_bowl_1_main".

    Returns:
        Normalised asset-class-like string, e.g. "akita_black_bowl".
    """
    return _BODY_SUFFIX_RE.sub("", body_name)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GroundingResult:
    """Outcome of a single episode with respect to language grounding.

    Attributes:
        episode_id: Unique string identifier for the episode.
        target_object: Language-specified target object (from scene params).
        first_contact_object: Normalised name of the first object the
            gripper touched, or None if no gripper contact occurred.
        is_faithful: True iff the gripper first touched the target object.
        contact_timestep: Simulation step at which first contact was detected.
        task_success: Whether the task was completed successfully.
    """

    episode_id: str
    target_object: str
    first_contact_object: Optional[str]
    is_faithful: bool
    contact_timestep: Optional[int]
    task_success: bool


@dataclass
class GroundingTracker:
    """Per-episode grounding tracker; call step() each sim step.

    Attributes:
        target_object_name: The language-specified target object, e.g.
            ``"akita_black_bowl"``.  Matched against normalised MuJoCo body
            names via substring containment (case-insensitive).
        episode_id: Optional identifier for logging.
    """

    target_object_name: str
    episode_id: str = ""

    _first_contact_object: Optional[str] = field(default=None, init=False, repr=False)
    _first_contact_step: Optional[int] = field(default=None, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)

    def step(self, mj_sim: Any) -> None:
        """Detect gripper–object contacts at this timestep.

        Call this once per ``env.step()`` with the MuJoCo sim object.
        After the first contact is recorded, subsequent calls are cheap
        no-ops (apart from incrementing the step counter).

        Args:
            mj_sim: The MuJoCo sim object accessible via
                ``libero_env.env.sim`` in a :class:`LIBEROSimulation`.
        """
        if self._first_contact_object is not None:
            # First contact already recorded; skip expensive contact scan.
            self._step += 1
            return

        for i in range(mj_sim.data.ncon):
            contact = mj_sim.data.contact[i]
            g1_id: int = contact.geom1
            g2_id: int = contact.geom2

            try:
                g1_name: str = mj_sim.model.geom_id2name(g1_id)
                g2_name: str = mj_sim.model.geom_id2name(g2_id)
            except Exception:
                # Geom IDs outside model range — skip silently.
                continue

            if g1_name in GRIPPER_GEOMS or g2_name in GRIPPER_GEOMS:
                # The non-gripper geom belongs to the contacted object.
                other_geom_id = g2_id if g1_name in GRIPPER_GEOMS else g1_id
                try:
                    body_id = mj_sim.model.geom_bodyid[other_geom_id]
                    raw_body_name: str = mj_sim.model.body_id2name(body_id)
                except Exception:
                    continue

                self._first_contact_object = _normalize_body_name(raw_body_name)
                self._first_contact_step = self._step
                break  # Only first contact matters.

        self._step += 1

    def result(self, task_success: bool) -> GroundingResult:
        """Finalise the episode and return a :class:`GroundingResult`.

        Args:
            task_success: Whether the LIBERO task was completed.

        Returns:
            :class:`GroundingResult` capturing language-grounding outcome.
        """
        # Faithful if gripper contacted the target (case-insensitive substring match).
        target_lower = self.target_object_name.lower()
        first_lower = (self._first_contact_object or "").lower()
        is_faithful: bool = bool(
            self._first_contact_object is not None
            and target_lower
            and (target_lower in first_lower or first_lower in target_lower)
        )
        return GroundingResult(
            episode_id=self.episode_id,
            target_object=self.target_object_name,
            first_contact_object=self._first_contact_object,
            is_faithful=is_faithful,
            contact_timestep=self._first_contact_step,
            task_success=task_success,
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_grounding(results: list[GroundingResult]) -> dict:
    """Aggregate per-episode grounding results into summary metrics.

    Args:
        results: List of :class:`GroundingResult` from one or more episodes.

    Returns:
        Dict with the following keys (empty dict if *results* is empty):

        * ``n_episodes``: total episode count
        * ``grounding_rate``: fraction where gripper first touched the target
        * ``success_rate``: fraction that completed the task
        * ``faithful_success_rate``: fraction that were both faithful *and* successful
        * ``biased_rate``: fraction where gripper first touched a *non-target*
          object (excludes no-contact episodes where the gripper never moved)
        * ``no_contact_rate``: fraction where gripper never touched any object

        The three mutually-exclusive outcomes satisfy:
        ``grounding_rate + biased_rate + no_contact_rate == 1.0``
    """
    n = len(results)
    if n == 0:
        return {}

    n_faithful = sum(1 for r in results if r.is_faithful)
    n_successful = sum(1 for r in results if r.task_success)
    n_faithful_success = sum(1 for r in results if r.is_faithful and r.task_success)
    n_no_contact = sum(1 for r in results if r.first_contact_object is None)
    # Biased = touched an object, but it was the wrong one (not no-contact)
    n_biased = n - n_faithful - n_no_contact

    return {
        "n_episodes": n,
        "grounding_rate": n_faithful / n,
        "success_rate": n_successful / n,
        "faithful_success_rate": n_faithful_success / n,
        "biased_rate": n_biased / n,
        "no_contact_rate": n_no_contact / n,
    }
