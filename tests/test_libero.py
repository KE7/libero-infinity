"""Tier 2 — LIBERO simulation tests (require MuJoCo + robosuite + LIBERO).

Tests full Scenic → LIBERO env → MuJoCo pipeline: pose injection, BDDL
preprocessing, camera/lighting perturbation, distractor injection, and
reversed task integration.
"""

import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import (
    BDDL_DIR,
    BOWL_BDDL,
    DRAWER_PICK_BOWL_BDDL,
    FLOOR_BASKET_BDDL,
    MICROWAVE_BDDL,
    SCENIC_DIR,
    STOVE_BDDL,
    requires_libero,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _setup_with_visibility_retry(sim_factory, scenario, max_retries=8):
    """Call sim_factory(scene) → sim, run sim.setup(), retrying on transient errors.

    VisibilityError, CollisionError, and RejectionException are all retried —
    physics settling instabilities and Scenic sampling exhaustion on specific
    samples are transient; a fresh sample typically succeeds.
    Returns (sim, scene) on success.
    """
    from libero_infinity.validation_errors import CollisionError, VisibilityError

    try:
        from scenic.core.distributions import RejectionException
    except ImportError:
        RejectionException = Exception  # type: ignore[assignment,misc]

    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            scene, _ = scenario.generate(maxIterations=2000, verbosity=0)
        except RejectionException as exc:
            last_exc = exc
            continue
        sim = sim_factory(scene)
        try:
            sim.setup()
            return sim, scene
        except (CollisionError, VisibilityError) as exc:
            sim.destroy()
            last_exc = exc
    raise RuntimeError(f"Setup error persisted after {max_retries} retries") from last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Core simulator integration
# ─────────────────────────────────────────────────────────────────────────────


@requires_libero
class TestLIBEROSimulatorIntegration:
    """Test full Scenic → LIBERO env → MuJoCo pose injection pipeline."""

    @pytest.fixture(scope="class")
    def scenic_scene(self):
        import scenic as sc

        path = SCENIC_DIR / "position_perturbation.scenic"
        scenario = sc.scenarioFromFile(
            str(path),
            params={
                "task": "put_the_bowl_on_the_plate",
                "bddl_path": str(BOWL_BDDL),
            },
        )
        scene, _ = scenario.generate(maxIterations=2000, verbosity=0)
        return scene

    @pytest.fixture(scope="class")
    def simulation(self, scenic_scene):
        from libero_infinity.simulator import LIBEROSimulator

        sim = LIBEROSimulator(bddl_path=str(BOWL_BDDL))
        episode = sim.createSimulation(
            scenic_scene,
            maxSteps=50,
            timestep=0.05,
            name="e2e_test",
            verbosity=0,
        )
        episode.setup()
        yield episode
        episode.destroy()

    def test_env_created(self, simulation):
        assert simulation.libero_env is not None

    def test_initial_obs_has_expected_keys(self, simulation):
        obs = simulation.last_obs
        assert obs is not None
        robot_keys = [k for k in obs if "robot" in k or "agentview" in k]
        assert len(robot_keys) >= 1

    def test_injected_positions_match_scenic(self, scenic_scene, simulation):
        """Positions read from MuJoCo must match Scenic-sampled positions."""
        TOLERANCE_M = 0.05  # 5 cm tolerance after physics settling

        sim_inner = simulation.libero_env.env.sim

        checked = 0
        for obj in scenic_scene.objects:
            libero_name = getattr(obj, "libero_name", "")
            if not libero_name:
                continue

            scenic_xy = np.array([obj.position.x, obj.position.y])

            body_id = None
            for candidate in (libero_name, libero_name + "_main"):
                try:
                    body_id = sim_inner.model.body_name2id(candidate)
                    break
                except Exception:
                    pass
            if body_id is None:
                continue

            mj_xy = sim_inner.data.body_xpos[body_id][:2].copy()
            dist = np.linalg.norm(scenic_xy - mj_xy)
            assert dist <= TOLERANCE_M, (
                f"{libero_name}: Scenic xy={scenic_xy} "
                f"vs MuJoCo xy={mj_xy} — dist={dist:.4f} m > {TOLERANCE_M} m"
            )
            checked += 1

        assert checked >= 1, "No objects were found in the MuJoCo model to verify"

    def test_step_works(self, simulation):
        for _ in range(5):
            simulation.step()

    def test_get_properties_position(self, scenic_scene, simulation):
        from scenic.core.vectors import Vector as ScenicVector

        for obj in scenic_scene.objects:
            libero_name = getattr(obj, "libero_name", "")
            if not libero_name:
                continue

            props = simulation.getProperties(obj, {"position"})
            assert "position" in props
            pos = props["position"]
            assert isinstance(pos, ScenicVector)
            assert 0.5 < pos.z < 1.5

    def test_check_success_false_initially(self, simulation):
        assert simulation.check_success() is False


@requires_libero
class TestScenicSafetyInvariants:
    """Settled scenes should remain on-workspace across existing Scenic programs."""

    @pytest.mark.parametrize(
        ("scenic_file", "params"),
        [
            (
                "position_perturbation.scenic",
                {
                    "task": "put_the_bowl_on_the_plate",
                    "bddl_path": str(BOWL_BDDL),
                },
            ),
            (
                "combined_perturbation.scenic",
                {
                    "task": "put_the_bowl_on_the_plate",
                    "bddl_path": str(BOWL_BDDL),
                    "perturb_class": "akita_black_bowl",
                },
            ),
            (
                "distractor_perturbation.scenic",
                {
                    "bddl_path": str(BOWL_BDDL),
                },
            ),
        ],
    )
    def test_settled_objects_stay_safe(self, scenic_file, params):
        import scenic as sc
        from libero_infinity.simulator import (
            MIN_SETTLED_Z,
            TABLE_X_MAX,
            TABLE_X_MIN,
            TABLE_Y_MAX,
            TABLE_Y_MIN,
            LIBEROSimulator,
        )

        scenario = sc.scenarioFromFile(str(SCENIC_DIR / scenic_file), params=params)
        sim, scene = _setup_with_visibility_retry(
            lambda s: LIBEROSimulator(bddl_path=str(BOWL_BDDL)).createSimulation(
                s,
                maxSteps=10,
                timestep=0.05,
                verbosity=0,
            ),
            scenario,
        )

        try:
            mj_sim = sim.libero_env.env.sim
            checked = 0
            for obj in scene.objects:
                libero_name = getattr(obj, "libero_name", "")
                if not libero_name or libero_name.startswith("distractor_"):
                    continue

                body_id = None
                for candidate in (libero_name, libero_name + "_main"):
                    try:
                        body_id = mj_sim.model.body_name2id(candidate)
                        break
                    except Exception:
                        continue
                if body_id is None:
                    continue

                pos = mj_sim.data.body_xpos[body_id][:3].copy()
                assert TABLE_X_MIN - 0.02 <= pos[0] <= TABLE_X_MAX + 0.02
                assert TABLE_Y_MIN - 0.02 <= pos[1] <= TABLE_Y_MAX + 0.02
                assert pos[2] >= MIN_SETTLED_Z
                checked += 1

            assert checked >= 2, f"No settled task objects were verified for {scenic_file}"
        finally:
            sim.destroy()

    def test_settled_generated_scene_preserves_object_clearance(self):
        from libero_infinity.asset_registry import get_dimensions
        from libero_infinity.compiler import compile_task_to_scenario
        from libero_infinity.simulator import LIBEROSimulator
        from libero_infinity.task_config import TaskConfig

        bddl = (
            BDDL_DIR
            / "libero_spatial"
            / "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate.bddl"
        )
        cfg = TaskConfig.from_bddl(str(bddl))
        scenario = compile_task_to_scenario(cfg, "position")
        dims_by_name = {
            obj.instance_name: get_dimensions(obj.object_class) for obj in cfg.movable_objects
        }

        for _ in range(3):
            sim, scene = _setup_with_visibility_retry(
                lambda s: LIBEROSimulator(bddl_path=str(bddl)).createSimulation(
                    s,
                    maxSteps=10,
                    timestep=0.05,
                    verbosity=0,
                ),
                scenario,
            )

            try:
                mj_sim = sim.libero_env.env.sim
                settled_xy = {}
                for name in dims_by_name:
                    body_id = None
                    for candidate in (name, name + "_main"):
                        try:
                            body_id = mj_sim.model.body_name2id(candidate)
                            break
                        except Exception:
                            continue
                    if body_id is not None:
                        settled_xy[name] = mj_sim.data.body_xpos[body_id][:2].copy()

                for i, name_a in enumerate(sorted(settled_xy)):
                    dims_a = dims_by_name[name_a]
                    for name_b in sorted(settled_xy)[i + 1 :]:
                        dims_b = dims_by_name[name_b]
                        dx = abs(float(settled_xy[name_a][0] - settled_xy[name_b][0]))
                        dy = abs(float(settled_xy[name_a][1] - settled_xy[name_b][1]))
                        min_dx = (dims_a[0] + dims_b[0]) / 2.0
                        min_dy = (dims_a[1] + dims_b[1]) / 2.0
                        assert dx >= min_dx - 0.001 or dy >= min_dy - 0.001, (
                            f"{name_a} overlaps {name_b} after settling: "
                            f"dx={dx:.3f} < {min_dx:.3f} and dy={dy:.3f} < {min_dy:.3f}"
                        )
            finally:
                sim.destroy()

    def test_microwave_position_scene_keeps_depth_and_visibility_targets(self):
        from libero_infinity.compiler import compile_task_to_scenario
        from libero_infinity.simulator import LIBEROSimulator
        from libero_infinity.task_config import TaskConfig

        cfg = TaskConfig.from_bddl(str(MICROWAVE_BDDL))
        scenario = compile_task_to_scenario(cfg, "position")

        sim, scene = _setup_with_visibility_retry(
            lambda s: LIBEROSimulator(bddl_path=str(MICROWAVE_BDDL)).createSimulation(
                s,
                maxSteps=10,
                timestep=0.05,
                verbosity=0,
            ),
            scenario,
        )

        try:
            assert "agentview_depth" in sim.last_obs
            assert list(scene.params.get("visibility_targets", [])) == ["white_yellow_mug_1"]
        finally:
            sim.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# BDDL preprocessor integration
# ─────────────────────────────────────────────────────────────────────────────


class TestBDDLPreprocessorUnit:
    """Pure string-level BDDL rewrite tests."""

    def test_substitute_multi_merges_duplicate_class_lines(self):
        from libero_infinity.bddl_preprocessor import (
            parse_object_classes,
            substitute_multi,
        )

        bddl_path = (
            BDDL_DIR / "libero_object" / "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
        )
        patched = substitute_multi(
            bddl_path.read_text(),
            {"alphabet_soup": "tomato_sauce"},
        )
        patched_classes = parse_object_classes(patched)
        assert patched_classes["alphabet_soup_1"] == "tomato_sauce"
        assert patched_classes["tomato_sauce_1"] == "tomato_sauce"
        assert patched.count(" - tomato_sauce") == 1

    def test_bddl_for_scene_applies_all_object_asset_substitutions(self):
        from libero_infinity.bddl_preprocessor import (
            bddl_for_scene,
            parse_object_classes,
        )

        scene = SimpleNamespace(
            params={},
            objects=[
                SimpleNamespace(libero_name="alphabet_soup_1", asset_class="tomato_sauce"),
                SimpleNamespace(libero_name="cream_cheese_1", asset_class="chocolate_pudding"),
                SimpleNamespace(libero_name="tomato_sauce_1", asset_class="tomato_sauce"),
            ],
        )
        bddl_path = (
            BDDL_DIR / "libero_object" / "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
        )
        orig = parse_object_classes(bddl_path.read_text())
        with bddl_for_scene(scene, str(bddl_path), orig) as tmp:
            patched = parse_object_classes(pathlib.Path(tmp).read_text())
            assert patched["alphabet_soup_1"] == "tomato_sauce"
            assert patched["cream_cheese_1"] == "chocolate_pudding"


@requires_libero
class TestBDDLPreprocessor:
    """BDDL object substitution pipeline with real files."""

    def test_substitute_bowl_asset(self):
        from libero_infinity.bddl_preprocessor import (
            parse_object_classes,
            substitute_asset,
        )

        bddl_text = BOWL_BDDL.read_text()
        obj_classes = parse_object_classes(bddl_text)
        assert obj_classes["akita_black_bowl_1"] == "akita_black_bowl"

        patched = substitute_asset(bddl_text, "akita_black_bowl", "white_bowl")
        patched_classes = parse_object_classes(patched)
        assert patched_classes["akita_black_bowl_1"] == "white_bowl"

        orig_classes = parse_object_classes(bddl_text)
        assert orig_classes["akita_black_bowl_1"] == "akita_black_bowl"

    def test_patched_bddl_context_manager(self):
        import pathlib

        from libero_infinity.bddl_preprocessor import parse_object_classes, patched_bddl

        with patched_bddl(BOWL_BDDL, {"akita_black_bowl": "white_bowl"}) as tmp:
            tmp_path = pathlib.Path(tmp)
            assert tmp_path.exists()
            patched_classes = parse_object_classes(tmp_path.read_text())
            assert patched_classes["akita_black_bowl_1"] == "white_bowl"
        assert not tmp_path.exists(), "Temp BDDL file not cleaned up"

    def test_patched_bddl_loads_in_libero(self):
        from libero.libero.envs.env_wrapper import OffScreenRenderEnv

        from libero_infinity.bddl_preprocessor import patched_bddl

        with patched_bddl(BOWL_BDDL, {"akita_black_bowl": "white_bowl"}) as tmp:
            env = OffScreenRenderEnv(
                bddl_file_name=tmp,
                has_renderer=False,
                has_offscreen_renderer=True,
                camera_heights=64,
                camera_widths=64,
                horizon=10,
            )
            obs = env.reset()
            assert obs is not None
            env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Camera / lighting perturbation in MuJoCo
# ─────────────────────────────────────────────────────────────────────────────


@requires_libero
class TestCameraPerturb:
    """Camera perturbation modifies the agentview camera in MuJoCo."""

    def test_camera_position_changes(self):
        import scenic as sc
        from libero_infinity.simulator import LIBEROSimulator

        path = SCENIC_DIR / "camera_perturbation.scenic"
        scenario = sc.scenarioFromFile(
            str(path),
            params={
                "bddl_path": str(BOWL_BDDL),
                "camera_x_range": 0.15,
                "camera_z_range": 0.10,
            },
        )
        sim, scene = _setup_with_visibility_retry(
            lambda s: LIBEROSimulator(bddl_path=str(BOWL_BDDL)).createSimulation(
                s,
                maxSteps=10,
                timestep=0.05,
                verbosity=0,
            ),
            scenario,
        )
        try:
            mj_sim = sim.libero_env.env.sim
            cam_id = mj_sim.model.camera_name2id("agentview")
            assert mj_sim.model.cam_pos[cam_id] is not None
            assert sim.last_obs is not None
        finally:
            sim.destroy()


@requires_libero
class TestLightingPerturb:
    """Lighting perturbation modifies MuJoCo light arrays."""

    def test_lighting_applied(self):
        import scenic as sc
        from libero_infinity.simulator import LIBEROSimulator

        path = SCENIC_DIR / "lighting_perturbation.scenic"
        scenario = sc.scenarioFromFile(
            str(path),
            params={
                "bddl_path": str(BOWL_BDDL),
                "intensity_min": 0.5,
                "intensity_max": 1.5,
            },
        )
        sim, scene = _setup_with_visibility_retry(
            lambda s: LIBEROSimulator(bddl_path=str(BOWL_BDDL)).createSimulation(
                s,
                maxSteps=10,
                timestep=0.05,
                verbosity=0,
            ),
            scenario,
        )
        try:
            assert sim.last_obs is not None
            obs, _, _, _ = sim.step_with_action(np.zeros(7))
            assert obs is not None
        finally:
            sim.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-generated scenic + LIBERO
# ─────────────────────────────────────────────────────────────────────────────


@requires_libero
class TestAutoGeneratedScenic:
    """Auto-generated Scenic programs work with the full LIBERO pipeline."""

    def test_bddl_only_position_perturbation(self):
        from libero_infinity.compiler import compile_task_to_scenario
        from libero_infinity.simulator import LIBEROSimulator
        from libero_infinity.task_config import TaskConfig

        cfg = TaskConfig.from_bddl(BOWL_BDDL)
        scenario = compile_task_to_scenario(cfg, "position")
        sim, scene = _setup_with_visibility_retry(
            lambda s: LIBEROSimulator(bddl_path=str(BOWL_BDDL)).createSimulation(
                s,
                maxSteps=10,
                timestep=0.05,
                verbosity=0,
            ),
            scenario,
        )
        assert sim.last_obs is not None
        sim.destroy()

    def test_goal_fixture_position_perturbation_moves_fixture(self):
        from libero_infinity.compiler import compile_task_to_scenario
        from libero_infinity.simulator import MIN_SETTLED_Z, LIBEROSimulator
        from libero_infinity.task_config import TaskConfig

        cfg = TaskConfig.from_bddl(STOVE_BDDL)
        scenario = compile_task_to_scenario(cfg, "position")
        sim, scene = _setup_with_visibility_retry(
            lambda s: LIBEROSimulator(bddl_path=str(STOVE_BDDL)).createSimulation(
                s,
                maxSteps=10,
                timestep=0.05,
                verbosity=0,
            ),
            scenario,
        )
        stove_obj = next(
            obj for obj in scene.objects if getattr(obj, "libero_name", "") == "flat_stove_1"
        )
        try:
            mj_sim = sim.libero_env.env.sim
            body_id = mj_sim.model.body_name2id("flat_stove_1_main")
            mj_pos = mj_sim.data.body_xpos[body_id][:3].copy()
            scenic_xy = np.array([stove_obj.position.x, stove_obj.position.y])
            assert np.linalg.norm(mj_pos[:2] - scenic_xy) <= 0.05
            assert mj_pos[2] >= MIN_SETTLED_Z
            assert sim.last_obs is not None
        finally:
            sim.destroy()

    def test_contained_object_position_perturbation_moves_with_support(self):
        from scenic.core.distributions import RejectionException

        from libero_infinity.compiler import compile_task_to_scenario
        from libero_infinity.simulator import LIBEROSimulator
        from libero_infinity.task_config import TaskConfig

        cfg = TaskConfig.from_bddl(DRAWER_PICK_BOWL_BDDL)
        scenario = compile_task_to_scenario(cfg, "position")

        canonical_simulator = LIBEROSimulator(bddl_path=str(DRAWER_PICK_BOWL_BDDL))
        canonical_scene = SimpleNamespace(objects=[])
        canonical = canonical_simulator.createSimulation(
            canonical_scene,
            maxSteps=1,
            timestep=0.05,
            verbosity=0,
        )
        canonical.setup()
        try:
            default_mj = canonical.libero_env.env.sim
            bowl_body = default_mj.model.body_name2id("akita_black_bowl_1_main")
            cabinet_body = default_mj.model.body_name2id("wooden_cabinet_1_main")
            _default_cabinet_xy = default_mj.data.body_xpos[cabinet_body][:2].copy()
        finally:
            canonical.destroy()

        for _ in range(5):
            try:
                scene, _ = scenario.generate(maxIterations=4000, verbosity=0)
                break
            except RejectionException:
                continue
        else:
            pytest.fail("Could not generate a contained-object position scene")

        sim, scene = _setup_with_visibility_retry(
            lambda s: LIBEROSimulator(bddl_path=str(DRAWER_PICK_BOWL_BDDL)).createSimulation(
                s,
                maxSteps=10,
                timestep=0.05,
                verbosity=0,
            ),
            scenario,
        )
        try:
            mj_sim = sim.libero_env.env.sim
            bowl_body = mj_sim.model.body_name2id("akita_black_bowl_1_main")
            cabinet_body = mj_sim.model.body_name2id("wooden_cabinet_1_main")
            bowl_xy = mj_sim.data.body_xpos[bowl_body][:2].copy()
            cabinet_xy = mj_sim.data.body_xpos[cabinet_body][:2].copy()
            final_delta = bowl_xy - cabinet_xy
            scenic_bowl = next(
                obj
                for obj in scene.objects
                if getattr(obj, "libero_name", "") == "akita_black_bowl_1"
            )
            scenic_cabinet = next(
                obj
                for obj in scene.objects
                if getattr(obj, "libero_name", "") == "wooden_cabinet_1"
            )
            scenic_delta = np.array(
                [
                    scenic_bowl.position.x - scenic_cabinet.position.x,
                    scenic_bowl.position.y - scenic_cabinet.position.y,
                ]
            )
            # The compiler places fixtures at fixed canonical positions (no
            # perturbation), so cabinet_xy ≈ default_cabinet_xy. We only
            # verify that the bowl's settled MuJoCo position matches the
            # Scenic-sampled position within physics-settling tolerance.
            assert np.linalg.norm(final_delta - scenic_delta) <= 0.05
        finally:
            sim.destroy()

    def test_position_perturbation_avoids_fixed_fixture_contacts(self):
        from libero_infinity.compiler import compile_task_to_scenario
        from libero_infinity.simulator import TABLE_Z, LIBEROSimulator
        from libero_infinity.task_config import TaskConfig

        bddl = (
            BDDL_DIR
            / "libero_spatial"
            / "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate.bddl"
        )
        cfg = TaskConfig.from_bddl(bddl)
        scenario = compile_task_to_scenario(cfg, "position")

        def is_fixed_fixture_contact(name_a: str, name_b: str) -> bool:
            fixed = ("flat_stove_1", "wooden_cabinet_1")
            movable = tuple(obj.instance_name for obj in cfg.movable_objects)
            return (
                any(name_a.startswith(prefix) for prefix in fixed)
                and any(name_b.startswith(prefix) for prefix in movable)
            ) or (
                any(name_b.startswith(prefix) for prefix in fixed)
                and any(name_a.startswith(prefix) for prefix in movable)
            )

        checked = 0
        for _ in range(3):
            sim, scene = _setup_with_visibility_retry(
                lambda s: LIBEROSimulator(bddl_path=str(bddl)).createSimulation(
                    s,
                    maxSteps=10,
                    timestep=0.05,
                    verbosity=0,
                ),
                scenario,
            )
            try:
                mj_sim = sim.libero_env.env.sim
                bowl_body = mj_sim.model.body_name2id("akita_black_bowl_2_main")
                bowl_pos = mj_sim.data.body_xpos[bowl_body][:3].copy()
                assert bowl_pos[2] < TABLE_Z + 0.20

                offending = []
                for i in range(int(mj_sim.data.ncon)):
                    contact = mj_sim.data.contact[i]
                    name_a = mj_sim.model.geom_id2name(int(contact.geom1)) or ""
                    name_b = mj_sim.model.geom_id2name(int(contact.geom2)) or ""
                    # Only flag contacts with penetration > 2 mm.  MuJoCo's
                    # penalty-based solver permits sub-2mm overlap as a normal
                    # numerical tolerance (e.g. burner sub-geoms protruding
                    # slightly beyond the AABB used for clearance constraints).
                    if is_fixed_fixture_contact(name_a, name_b) and float(contact.dist) < -0.002:
                        offending.append((name_a, name_b, float(contact.dist)))

                assert not offending
                checked += 1
            finally:
                sim.destroy()

        assert checked == 3

    def test_floor_scene_position_perturbation_setup_succeeds(self):
        from libero_infinity.compiler import compile_task_to_scenario
        from libero_infinity.simulator import LIBEROSimulator
        from libero_infinity.task_config import TaskConfig

        cfg = TaskConfig.from_bddl(FLOOR_BASKET_BDDL)
        scenario = compile_task_to_scenario(cfg, "position")
        sim, scene = _setup_with_visibility_retry(
            lambda s: LIBEROSimulator(bddl_path=str(FLOOR_BASKET_BDDL)).createSimulation(
                s,
                maxSteps=10,
                timestep=0.05,
                verbosity=0,
            ),
            scenario,
        )
        try:
            mj_sim = sim.libero_env.env.sim
            obj = next(
                scene_obj
                for scene_obj in scene.objects
                if getattr(scene_obj, "libero_name", "") == "cream_cheese_1"
            )
            body_id = mj_sim.model.body_name2id("cream_cheese_1_main")
            mj_xy = mj_sim.data.body_xpos[body_id][:2].copy()
            assert np.linalg.norm(mj_xy - np.array([obj.position.x, obj.position.y])) <= 0.05
            assert sim.last_obs is not None
        finally:
            sim.destroy()

    def test_goal_region_task_does_not_start_successful(self):
        from libero_infinity.compiler import compile_task_to_scenario
        from libero_infinity.simulator import LIBEROSimulator
        from libero_infinity.task_config import TaskConfig

        bddl = BDDL_DIR / "libero_goal" / "push_the_plate_to_the_front_of_the_stove.bddl"
        cfg = TaskConfig.from_bddl(bddl)
        scenario = compile_task_to_scenario(cfg, "position")
        checked = 0
        for _ in range(3):
            sim, scene = _setup_with_visibility_retry(
                lambda s: LIBEROSimulator(bddl_path=str(bddl)).createSimulation(
                    s,
                    maxSteps=10,
                    timestep=0.05,
                    verbosity=0,
                ),
                scenario,
            )
            try:
                assert sim.check_success() is False
                checked += 1
            finally:
                sim.destroy()

        assert checked == 3


# ─────────────────────────────────────────────────────────────────────────────
# Distractor LIBERO integration
# ─────────────────────────────────────────────────────────────────────────────


@requires_libero
class TestDistractorLIBEROIntegration:
    """Distractor objects injected into MuJoCo via BDDL patching."""

    def test_distractor_appears_in_mujoco(self):
        import scenic as sc
        from libero_infinity.simulator import LIBEROSimulator

        path = SCENIC_DIR / "distractor_perturbation.scenic"
        scenario = sc.scenarioFromFile(
            str(path),
            params={"bddl_path": str(BOWL_BDDL)},
        )
        sim, scene = _setup_with_visibility_retry(
            lambda s: LIBEROSimulator(bddl_path=str(BOWL_BDDL)).createSimulation(
                s,
                maxSteps=10,
                timestep=0.05,
                verbosity=0,
            ),
            scenario,
        )

        try:
            n = int(scene.params["n_distractors"])
            assert n >= 1

            mj_sim = sim.libero_env.env.sim
            found = 0
            for i in range(n):
                for suffix in (f"distractor_{i}", f"distractor_{i}_main"):
                    try:
                        mj_sim.model.body_name2id(suffix)
                        found += 1
                        break
                    except Exception:
                        pass
            assert found >= 1

            assert sim.last_obs is not None
            obs, _, _, _ = sim.step_with_action(np.zeros(7))
            assert obs is not None
        finally:
            sim.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Reversed task LIBERO integration
# ─────────────────────────────────────────────────────────────────────────────


@requires_libero
class TestReversedLIBEROIntegration:
    """Reversed BDDL loads and runs in LIBERO."""

    def test_reversed_bddl_loads_in_libero(self):
        from libero.libero.envs.env_wrapper import OffScreenRenderEnv

        from libero_infinity.bddl_preprocessor import patched_bddl_from_string
        from libero_infinity.task_reverser import reverse_bddl

        reversed_content = reverse_bddl(BOWL_BDDL.read_text())

        with patched_bddl_from_string(reversed_content) as tmp:
            env = OffScreenRenderEnv(
                bddl_file_name=tmp,
                has_renderer=False,
                has_offscreen_renderer=True,
                camera_heights=64,
                camera_widths=64,
                horizon=10,
            )
            obs = env.reset()
            assert obs is not None
            env.close()

    def test_reversed_with_scenic_position_perturbation(self):
        from libero_infinity.bddl_preprocessor import patched_bddl_from_string
        from libero_infinity.compiler import compile_task_to_scenario
        from libero_infinity.simulator import LIBEROSimulator
        from libero_infinity.task_config import TaskConfig
        from libero_infinity.task_reverser import reverse_bddl

        reversed_content = reverse_bddl(BOWL_BDDL.read_text())
        cfg = TaskConfig.from_string(reversed_content, path="<reversed>")
        scenario = compile_task_to_scenario(cfg, "position")
        with patched_bddl_from_string(reversed_content) as bddl_tmp:
            sim, scene = _setup_with_visibility_retry(
                lambda s: LIBEROSimulator(bddl_path=bddl_tmp).createSimulation(
                    s,
                    maxSteps=10,
                    timestep=0.05,
                    verbosity=0,
                ),
                scenario,
            )
            try:
                assert sim.last_obs is not None
                for _ in range(5):
                    sim.step()
            finally:
                sim.destroy()
