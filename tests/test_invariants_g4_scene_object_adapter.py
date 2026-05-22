"""Regression test for the G4 family-B/C/D scene-object adapter.

Background
----------
Scenic ``LIBEROObject`` / ``LIBEROFixture`` instances expose their BDDL
instance name as the declared property ``libero_name`` — there is **no**
``name`` property. The G4 family-B/C/D invariants (domain / consistency /
affordance) used to read ``getattr(o, "name", ...)``, so every scene object
keyed as ``""`` / ``"?"`` and the whole B/C/D family scored uniformly
false-negative (see RCA stage1_g4_bcd_scene_object_adapter).

This test constructs a small known scene whose objects use the **Scenic**
identity convention (``libero_name`` + ``asset_class``, no ``name``) and runs
the combined ``g4_domain_consistency_hook``. It asserts:

* objects are identified by their real instance name (no ``""`` / ``"?"`` keys);
* on a nominal scene the per-object B/C/D checks pass (reflect ground truth);
* on a deliberately corrupted scene the per-object checks fail.

Against the pre-fix code every per-object check is false and the consistency /
affordance keys collapse onto ``"?"`` — so the nominal-scene assertions below
fail. After the adapter fix they pass.
"""

from __future__ import annotations

from libero_infinity.validation.invariants import g4_domain_consistency_hook

# ---------------------------------------------------------------------------
# Scenic-style scene doubles — identity via ``libero_name`` (NOT ``name``).
# ---------------------------------------------------------------------------


class LIBEROObject:
    """Movable task object — mirrors the Scenic class' identity surface."""

    def __init__(self, libero_name, asset_class, position):
        self.libero_name = libero_name
        self.asset_class = asset_class
        self.position = position
        self.height = 0.06


class LIBEROFixture:
    """Non-movable fixture — must be excluded from B/C/D scoring."""

    def __init__(self, libero_name, position):
        self.libero_name = libero_name
        self.asset_class = ""  # fixtures carry no sampled asset class
        self.position = position
        self.height = 0.40


class _Scene:
    def __init__(self, objects):
        self.objects = objects
        self.params = {}


class _Env:
    """Duck-typed LIBERO env exposing ``get_object_state``."""

    def __init__(self, states):
        self._states = states

    def get_object_state(self, name):
        return self._states.get(name)

    def check_success(self):  # goal_false_at_reset -> passed True
        return False


class _BDDLObj:
    def __init__(self, instance_name, object_class):
        self.instance_name = instance_name
        self.object_class = object_class


class _BDDL:
    def __init__(self, movable):
        self.movable_objects = movable
        self.init_text = ""
        self.goal_text = ""


_REGISTRY = {"akita_black_bowl", "plate"}
_GRASP_POINTS = {"akita_black_bowl": (0.0, 0.0, 0.02), "plate": (0.0, 0.0, 0.01)}


def _nominal_scene():
    return _Scene(
        [
            LIBEROObject("akita_black_bowl_1", "akita_black_bowl", (0.10, 0.20, 0.90)),
            LIBEROObject("plate_1", "plate", (-0.10, 0.05, 0.90)),
            LIBEROFixture("wooden_cabinet_1", (0.03, -0.24, 0.90)),
        ]
    )


def _nominal_env():
    return _Env(
        {
            "akita_black_bowl_1": {
                "position": (0.10, 0.20, 0.90),
                "orientation": None,
                "class": "akita_black_bowl",
            },
            "plate_1": {
                "position": (-0.10, 0.05, 0.90),
                "orientation": None,
                "class": "plate",
            },
        }
    )


def _bddl():
    return _BDDL(
        [
            _BDDLObj("akita_black_bowl_1", "akita_black_bowl"),
            _BDDLObj("plate_1", "plate"),
        ]
    )


# ---------------------------------------------------------------------------
# Pass case — real identities, ground-truth-consistent scene.
# ---------------------------------------------------------------------------


def test_g4_bcd_hook_identifies_objects_and_passes_on_nominal_scene():
    flat = g4_domain_consistency_hook(
        _nominal_scene(),
        _nominal_env(),
        _bddl(),
        registry=_REGISTRY,
        grasp_points=_GRASP_POINTS,
    )

    # --- objects keyed by their REAL libero_name, never "" / "?" -----------
    con_keys = [k for k in flat if k.startswith("consistency:")]
    aff_keys = [k for k in flat if k.startswith("affordance:")]
    assert con_keys, "expected per-object consistency results"
    assert aff_keys, "expected per-object affordance results"
    for k in con_keys + aff_keys:
        obj_name = k.rsplit(":", 1)[1]
        assert obj_name not in ("", "?"), f"unresolved object identity in key {k!r}"

    # exactly the two movable objects are scored — the fixture is excluded.
    scored = {k.rsplit(":", 1)[1] for k in con_keys}
    assert scored == {"akita_black_bowl_1", "plate_1"}, scored
    assert "wooden_cabinet_1" not in scored

    # --- domain family B reflects ground truth ----------------------------
    assert flat["domain:bddl_objects_present"].passed is True
    assert flat["domain:assets_in_registry"].passed is True

    # --- consistency family C: real per-object pass -----------------------
    for nm in ("akita_black_bowl_1", "plate_1"):
        assert flat[f"consistency:pose_tolerance:{nm}"].passed is True
        assert flat[f"consistency:class_match:{nm}"].passed is True

    # --- affordance family D: grasp clearance pass ------------------------
    for nm in ("akita_black_bowl_1", "plate_1"):
        assert flat[f"affordance:aabb_clear_around_grasp:{nm}"].passed is True


# ---------------------------------------------------------------------------
# Fail case — deliberate ground-truth violations must be detected.
# ---------------------------------------------------------------------------


def test_g4_bcd_hook_detects_deliberate_violations():
    # Scene: bowl asset swapped to an unregistered class; plate omitted so a
    # BDDL object is genuinely missing from the scene.
    scene = _Scene(
        [
            LIBEROObject("akita_black_bowl_1", "not_a_real_asset", (0.10, 0.20, 0.90)),
        ]
    )
    # Env: bowl is far displaced and has a mismatched class.
    env = _Env(
        {
            "akita_black_bowl_1": {
                "position": (0.50, 0.20, 0.90),  # 0.40 m off — beyond pos_tol
                "orientation": None,
                "class": "white_bowl",  # class mismatch
            },
        }
    )

    flat = g4_domain_consistency_hook(
        scene, env, _bddl(), registry=_REGISTRY, grasp_points=_GRASP_POINTS
    )

    # bowl still resolves to its real name (adapter works even on bad data).
    assert "consistency:pose_tolerance:akita_black_bowl_1" in flat

    # domain: plate_1 missing from scene, bowl class not in registry.
    assert flat["domain:bddl_objects_present"].passed is False
    assert flat["domain:assets_in_registry"].passed is False

    # consistency: displaced pose + class mismatch both caught.
    assert flat["consistency:pose_tolerance:akita_black_bowl_1"].passed is False
    assert flat["consistency:class_match:akita_black_bowl_1"].passed is False


# ---------------------------------------------------------------------------
# Producer (Defect 2): the renderer emits `with asset_class` on every object.
# ---------------------------------------------------------------------------


def test_renderer_emits_asset_class_on_every_object():
    """The compiler must emit ``with asset_class`` on every generated object,
    even when the ``object`` perturbation axis is inactive — otherwise sampled
    scene objects carry no class string and G4 family-B/C/D cannot validate the
    instantiated asset. Renders a real scene with only the ``position`` axis
    active (so no asset substitution happens) and checks every task-object
    declaration."""
    import glob

    from libero_infinity.ir.graph_builder import build_semantic_scene_graph
    from libero_infinity.planner.composition import plan_perturbations
    from libero_infinity.renderer.scenic_renderer import render_scenic
    from libero_infinity.task_config import TaskConfig

    bddl = glob.glob("src/libero_infinity/data/libero_runtime/bddl_files/**/*.bddl", recursive=True)
    if not bddl:
        import pytest

        pytest.skip("No BDDL files found")
    cfg = TaskConfig.from_bddl(bddl[0])
    graph = build_semantic_scene_graph(cfg)
    plan = plan_perturbations(graph, "position")  # object axis INACTIVE
    src = render_scenic(plan, graph)

    obj_lines = [ln for ln in src.splitlines() if "new LIBEROObject" in ln]
    assert obj_lines, "expected at least one LIBEROObject declaration"
    for ln in obj_lines:
        assert "with asset_class" in ln, f"object missing asset_class: {ln!r}"
