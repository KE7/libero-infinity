"""Comprehensive perturbation sampling verification.

For every single perturbation axis and key combinations, this test:
1. Compiles a Scenic scenario for the perturbation type
2. Samples 10 scenes from that scenario
3. Verifies that sampled values VARY across the 10 samples (not identical)
4. Verifies that sampled values DIFFER from canonical BDDL init positions

This ensures Scenic is actually being used to produce diverse perturbations
and that the perturbation pipeline is not silently returning defaults.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import BDDL_DIR, BOWL_BDDL

from libero_infinity.compiler import compile_task_to_scenario
from libero_infinity.task_config import TaskConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES = 10
MAX_ITERATIONS = 4000  # high enough for constrained scenarios

# The 8 individual axes
SINGLE_AXES = [
    "position",
    "object",
    "camera",
    "lighting",
    "texture",
    "distractor",
    "background",
    "articulation",
]

# Key combinations
COMBO_AXES = [
    "combined",  # position + object + camera + lighting + distractor + background
    "full",      # all 8 axes
    "position,camera",
    "position,object",
    "position,lighting",
    "position,distractor",
    "position,background",
    "position,texture",
    "position,articulation",
    "camera,lighting",
    "object,distractor",
    "position,object,camera",
    "position,object,camera,lighting",
    "position,object,camera,lighting,distractor",
]

ALL_PERTURBATION_SPECS = SINGLE_AXES + COMBO_AXES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_positions(scene) -> dict[str, tuple[float, float]]:
    """Extract {libero_name: (x, y)} from a Scenic scene."""
    positions = {}
    for obj in scene.objects:
        name = getattr(obj, "libero_name", "")
        if name:
            positions[name] = (float(obj.position.x), float(obj.position.y))
    return positions


def _extract_asset_classes(scene) -> dict[str, str]:
    """Extract {libero_name: asset_class} for objects that have a non-empty asset_class."""
    classes = {}
    for obj in scene.objects:
        name = getattr(obj, "libero_name", "")
        ac = getattr(obj, "asset_class", "")
        if name and ac:
            classes[name] = ac
    return classes


def _canonical_positions(cfg: TaskConfig) -> dict[str, tuple[float, float]]:
    """Extract canonical init positions from the TaskConfig."""
    positions = {}
    for obj in cfg.movable_objects:
        if obj.init_x is not None and obj.init_y is not None:
            positions[obj.instance_name] = (obj.init_x, obj.init_y)
    return positions


def _sample_scenes(scenario, n: int = N_SAMPLES):
    """Generate n scenes, tolerating occasional RejectionExceptions."""
    try:
        from scenic.core.distributions import RejectionException
    except ImportError:
        RejectionException = Exception  # type: ignore[assignment,misc]

    scenes = []
    attempts = 0
    max_attempts = n * 3  # Allow some retries
    while len(scenes) < n and attempts < max_attempts:
        attempts += 1
        try:
            scene, _ = scenario.generate(maxIterations=MAX_ITERATIONS, verbosity=0)
            scenes.append(scene)
        except RejectionException:
            continue
    return scenes


def _has_position_axis(spec: str) -> bool:
    """Check if a perturbation spec includes the position axis."""
    from libero_infinity.planner.composition import parse_axes

    return "position" in parse_axes(spec)


def _has_object_axis(spec: str) -> bool:
    from libero_infinity.planner.composition import parse_axes

    return "object" in parse_axes(spec)


def _has_camera_axis(spec: str) -> bool:
    from libero_infinity.planner.composition import parse_axes

    return "camera" in parse_axes(spec)


def _has_lighting_axis(spec: str) -> bool:
    from libero_infinity.planner.composition import parse_axes

    return "lighting" in parse_axes(spec)


def _has_distractor_axis(spec: str) -> bool:
    from libero_infinity.planner.composition import parse_axes

    return "distractor" in parse_axes(spec)


def _has_background_axis(spec: str) -> bool:
    from libero_infinity.planner.composition import parse_axes

    return "background" in parse_axes(spec)


# ---------------------------------------------------------------------------
# Tests — single axes
# ---------------------------------------------------------------------------


class TestPositionPerturbationSampling:
    """Position axis: sampled (x, y) must vary and differ from canonical."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return TaskConfig.from_bddl(BOWL_BDDL)

    @pytest.fixture(scope="class")
    def scenes(self, cfg):
        scenario = compile_task_to_scenario(cfg, "position")
        return _sample_scenes(scenario)

    def test_10_samples_generated(self, scenes):
        assert len(scenes) == N_SAMPLES

    def test_positions_vary_across_samples(self, scenes):
        """At least one object's position must differ across 10 samples."""
        all_positions = [_extract_positions(s) for s in scenes]
        common_names = set.intersection(*(set(p.keys()) for p in all_positions))
        assert len(common_names) > 0, "No common objects found across samples"

        varied = False
        for name in common_names:
            xs = [p[name][0] for p in all_positions]
            ys = [p[name][1] for p in all_positions]
            if np.std(xs) > 1e-6 or np.std(ys) > 1e-6:
                varied = True
                break
        assert varied, "All 10 position samples are identical — Scenic not perturbing"

    def test_positions_differ_from_canonical(self, cfg, scenes):
        """At least one sample must deviate from canonical BDDL init position."""
        canonical = _canonical_positions(cfg)
        all_positions = [_extract_positions(s) for s in scenes]

        any_deviated = False
        for name, (cx, cy) in canonical.items():
            for pos_dict in all_positions:
                if name in pos_dict:
                    sx, sy = pos_dict[name]
                    dist = np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2)
                    if dist > 0.005:
                        any_deviated = True
                        break
            if any_deviated:
                break

        assert any_deviated, (
            "No sample deviated from canonical positions — perturbation not applied"
        )


class TestObjectPerturbationSampling:
    """Object axis: sampled asset classes must show diversity."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return TaskConfig.from_bddl(BOWL_BDDL)

    @pytest.fixture(scope="class")
    def scenes(self, cfg):
        scenario = compile_task_to_scenario(cfg, "object")
        return _sample_scenes(scenario)

    def test_10_samples_generated(self, scenes):
        assert len(scenes) == N_SAMPLES

    def test_asset_classes_vary(self, scenes):
        """At least one object should have >1 unique asset class across samples."""
        all_classes = [_extract_asset_classes(s) for s in scenes]
        # Filter to objects that have asset_class set
        all_keys = set()
        for c in all_classes:
            all_keys.update(c.keys())

        varied = False
        for name in all_keys:
            unique_classes = {c[name] for c in all_classes if name in c}
            if len(unique_classes) > 1:
                varied = True
                break
        assert varied, "All 10 object samples use the same asset class — no object perturbation"


class TestCameraPerturbationSampling:
    """Camera axis: cam_azimuth/elevation/distance must vary across samples."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return TaskConfig.from_bddl(BOWL_BDDL)

    @pytest.fixture(scope="class")
    def scenes(self, cfg):
        scenario = compile_task_to_scenario(cfg, "camera")
        return _sample_scenes(scenario)

    def test_10_samples_generated(self, scenes):
        assert len(scenes) == N_SAMPLES

    def test_camera_params_vary(self, scenes):
        """cam_azimuth, cam_elevation, cam_distance should vary."""
        for param_name in ("cam_azimuth", "cam_elevation", "cam_distance"):
            vals = [float(s.params[param_name]) for s in scenes if param_name in s.params]
            if vals:
                assert np.std(vals) > 1e-6, (
                    f"{param_name} identical across {len(vals)} samples: {vals[:3]}"
                )
                return  # At least one varying param found

        pytest.fail("No camera params (cam_azimuth/elevation/distance) found in scenes")


class TestLightingPerturbationSampling:
    """Lighting axis: intensity/ambient must vary."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return TaskConfig.from_bddl(BOWL_BDDL)

    @pytest.fixture(scope="class")
    def scenes(self, cfg):
        scenario = compile_task_to_scenario(cfg, "lighting")
        return _sample_scenes(scenario)

    def test_10_samples_generated(self, scenes):
        assert len(scenes) == N_SAMPLES

    def test_lighting_params_vary(self, scenes):
        """Light intensity or ambient should vary across samples."""
        for param_name in ("light_intensity", "intensity", "light_ambient", "ambient"):
            vals = [float(s.params[param_name]) for s in scenes if param_name in s.params]
            if len(vals) >= 2:
                assert np.std(vals) > 1e-6, (
                    f"{param_name} identical across samples: {vals[:3]}"
                )
                return
        pytest.fail("No lighting params found in scenes")


class TestTexturePerturbationSampling:
    """Texture axis: table_texture param must be set."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return TaskConfig.from_bddl(BOWL_BDDL)

    @pytest.fixture(scope="class")
    def scenes(self, cfg):
        scenario = compile_task_to_scenario(cfg, "texture")
        return _sample_scenes(scenario)

    def test_10_samples_generated(self, scenes):
        assert len(scenes) == N_SAMPLES

    def test_texture_param_present(self, scenes):
        """table_texture param should be set in all scenes."""
        for i, s in enumerate(scenes):
            assert "table_texture" in s.params, f"Sample {i}: no table_texture param"


class TestDistractorPerturbationSampling:
    """Distractor axis: distractors must appear with varying positions."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return TaskConfig.from_bddl(BOWL_BDDL)

    @pytest.fixture(scope="class")
    def scenes(self, cfg):
        scenario = compile_task_to_scenario(cfg, "distractor")
        return _sample_scenes(scenario)

    def test_10_samples_generated(self, scenes):
        assert len(scenes) == N_SAMPLES

    def test_distractors_present(self, scenes):
        """Every sample should have at least 1 distractor."""
        for i, s in enumerate(scenes):
            n = int(s.params.get("n_distractors", 0))
            assert n >= 1, f"Sample {i} has 0 distractors"

    def test_distractor_positions_vary(self, scenes):
        """Distractor positions should differ across samples."""
        all_d0_positions = []
        for s in scenes:
            for obj in s.objects:
                name = getattr(obj, "libero_name", "")
                if name == "distractor_0":
                    all_d0_positions.append((float(obj.position.x), float(obj.position.y)))
                    break
        if len(all_d0_positions) >= 2:
            xs = [p[0] for p in all_d0_positions]
            ys = [p[1] for p in all_d0_positions]
            assert np.std(xs) > 1e-6 or np.std(ys) > 1e-6, (
                "distractor_0 position identical across all samples"
            )


class TestBackgroundPerturbationSampling:
    """Background axis: wall/floor textures must vary."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return TaskConfig.from_bddl(BOWL_BDDL)

    @pytest.fixture(scope="class")
    def scenes(self, cfg):
        scenario = compile_task_to_scenario(cfg, "background")
        return _sample_scenes(scenario)

    def test_10_samples_generated(self, scenes):
        assert len(scenes) == N_SAMPLES

    def test_background_params_vary(self, scenes):
        """wall_texture or floor_texture should vary across samples."""
        for param_name in ("wall_texture", "floor_texture"):
            vals = [str(s.params[param_name]) for s in scenes if param_name in s.params]
            if len(vals) >= 2:
                unique = set(vals)
                if len(unique) > 1:
                    return  # Found variation
        pytest.fail("No background param variation found (wall_texture, floor_texture)")


class TestArticulationPerturbationSampling:
    """Articulation axis: fixture joint values should vary."""

    @pytest.fixture(scope="class")
    def cfg(self):
        bddl = BDDL_DIR / "libero_goal" / "open_the_middle_drawer_of_the_cabinet.bddl"
        if not bddl.exists():
            pytest.skip("Drawer BDDL not found")
        return TaskConfig.from_bddl(bddl)

    @pytest.fixture(scope="class")
    def scenes(self, cfg):
        scenario = compile_task_to_scenario(cfg, "articulation")
        return _sample_scenes(scenario)

    def test_10_samples_generated(self, scenes):
        assert len(scenes) == N_SAMPLES

    def test_articulation_joint_values_vary(self, scenes):
        """Articulation joint values (articulation_*) should vary."""
        # Find articulation params (e.g. articulation_wooden_cabinet_1)
        art_keys = set()
        for s in scenes:
            for k in s.params:
                if k.startswith("articulation_") and not k.endswith("_state"):
                    art_keys.add(k)

        assert len(art_keys) > 0, "No articulation params found in scenes"

        for key in art_keys:
            vals = [float(s.params[key]) for s in scenes if key in s.params]
            if len(vals) >= 2 and np.std(vals) > 1e-6:
                return  # Found variation
        pytest.fail(f"Articulation joint values identical across samples: {art_keys}")


# ---------------------------------------------------------------------------
# Tests — preset combinations
# ---------------------------------------------------------------------------


class TestCombinedPerturbationSampling:
    """combined preset: position + object + camera + lighting + distractor + background."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return TaskConfig.from_bddl(BOWL_BDDL)

    @pytest.fixture(scope="class")
    def scenes(self, cfg):
        scenario = compile_task_to_scenario(cfg, "combined")
        return _sample_scenes(scenario)

    def test_at_least_5_samples(self, scenes):
        """Combined has many constraints; allow partial generation."""
        assert len(scenes) >= 5, f"Only {len(scenes)} samples generated for combined"

    def test_positions_vary(self, scenes):
        all_positions = [_extract_positions(s) for s in scenes]
        common_names = set.intersection(*(set(p.keys()) for p in all_positions))
        varied = any(
            np.std([p[name][0] for p in all_positions]) > 1e-6
            for name in common_names
        )
        assert varied, "combined: positions don't vary"

    def test_assets_vary(self, scenes):
        all_classes = [_extract_asset_classes(s) for s in scenes]
        all_keys = set()
        for c in all_classes:
            all_keys.update(c.keys())
        varied = any(
            len({c[name] for c in all_classes if name in c}) > 1
            for name in all_keys
        )
        assert varied, "combined: asset classes don't vary"

    def test_positions_differ_from_canonical(self, cfg, scenes):
        canonical = _canonical_positions(cfg)
        all_positions = [_extract_positions(s) for s in scenes]
        any_deviated = False
        for name, (cx, cy) in canonical.items():
            for pos_dict in all_positions:
                if name in pos_dict:
                    sx, sy = pos_dict[name]
                    if np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2) > 0.005:
                        any_deviated = True
                        break
            if any_deviated:
                break
        assert any_deviated, "combined: no sample deviated from canonical positions"


class TestFullPerturbationSampling:
    """full preset: all 8 axes active simultaneously."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return TaskConfig.from_bddl(BOWL_BDDL)

    @pytest.fixture(scope="class")
    def scenes(self, cfg):
        scenario = compile_task_to_scenario(cfg, "full")
        return _sample_scenes(scenario)

    def test_at_least_5_samples(self, scenes):
        """Full has many constraints; allow partial generation."""
        assert len(scenes) >= 5, f"Only {len(scenes)} samples generated for full"

    def test_positions_vary(self, scenes):
        if len(scenes) < 2:
            pytest.skip("Not enough samples")
        all_positions = [_extract_positions(s) for s in scenes]
        common_names = set.intersection(*(set(p.keys()) for p in all_positions))
        varied = any(
            np.std([p[name][0] for p in all_positions]) > 1e-6
            for name in common_names
        )
        assert varied, "full: positions don't vary"

    def test_positions_differ_from_canonical(self, cfg, scenes):
        if len(scenes) < 2:
            pytest.skip("Not enough samples")
        canonical = _canonical_positions(cfg)
        all_positions = [_extract_positions(s) for s in scenes]
        any_deviated = False
        for name, (cx, cy) in canonical.items():
            for pos_dict in all_positions:
                if name in pos_dict:
                    sx, sy = pos_dict[name]
                    if np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2) > 0.005:
                        any_deviated = True
                        break
            if any_deviated:
                break
        assert any_deviated, "full: no sample deviated from canonical positions"


# ---------------------------------------------------------------------------
# Parametrized: every spec compiles and generates diverse samples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ALL_PERTURBATION_SPECS, ids=ALL_PERTURBATION_SPECS)
def test_spec_compiles_and_generates(spec):
    """Every perturbation spec must compile and generate at least 5 scenes."""
    cfg = TaskConfig.from_bddl(BOWL_BDDL)
    scenario = compile_task_to_scenario(cfg, spec)
    scenes = _sample_scenes(scenario, n=N_SAMPLES)
    assert len(scenes) >= 5, f"{spec}: only {len(scenes)}/{N_SAMPLES} samples generated"


@pytest.mark.parametrize("spec", ALL_PERTURBATION_SPECS, ids=ALL_PERTURBATION_SPECS)
def test_spec_shows_variation(spec):
    """Every perturbation spec must show variance across sampled scenes."""
    cfg = TaskConfig.from_bddl(BOWL_BDDL)
    scenario = compile_task_to_scenario(cfg, spec)
    scenes = _sample_scenes(scenario, n=N_SAMPLES)
    assert len(scenes) >= 2, f"{spec}: need at least 2 samples to check variation"

    # Check positions vary (if position axis is active)
    all_positions = [_extract_positions(s) for s in scenes]
    common_names = set.intersection(*(set(p.keys()) for p in all_positions)) if all_positions else set()
    position_varied = any(
        np.std([p[name][0] for p in all_positions]) > 1e-6
        or np.std([p[name][1] for p in all_positions]) > 1e-6
        for name in common_names
    ) if common_names else False

    # Check asset classes vary (if object axis is active)
    all_classes = [_extract_asset_classes(s) for s in scenes]
    all_class_keys = set()
    for c in all_classes:
        all_class_keys.update(c.keys())
    asset_varied = any(
        len({c[name] for c in all_classes if name in c}) > 1
        for name in all_class_keys
    ) if all_class_keys else False

    # Check numeric scene params vary (camera, lighting, articulation joints)
    param_varied = False
    numeric_params = set()
    for s in scenes:
        for k, v in s.params.items():
            if isinstance(v, (int, float)) and k not in ("n_distractors", "ood_margin"):
                numeric_params.add(k)
    for key in numeric_params:
        vals = [float(s.params[key]) for s in scenes if key in s.params]
        if len(vals) >= 2 and np.std(vals) > 1e-6:
            param_varied = True
            break

    # Check string scene params vary (background textures, etc.)
    string_params_varied = False
    for k in ("wall_texture", "floor_texture"):
        vals = [str(s.params[k]) for s in scenes if k in s.params]
        if len(vals) >= 2 and len(set(vals)) > 1:
            string_params_varied = True
            break

    # Check distractors present
    distractor_present = any(
        int(s.params.get("n_distractors", 0)) > 0 for s in scenes
    )

    assert position_varied or asset_varied or param_varied or string_params_varied or distractor_present, (
        f"{spec}: all {len(scenes)} samples appear identical — "
        "no variation in positions, assets, params, or distractors"
    )


@pytest.mark.parametrize("spec", [s for s in ALL_PERTURBATION_SPECS if "position" in s or s in ("combined", "full")],
                         ids=[s for s in ALL_PERTURBATION_SPECS if "position" in s or s in ("combined", "full")])
def test_spec_differs_from_canonical(spec):
    """Specs with position axis must produce positions different from canonical."""
    cfg = TaskConfig.from_bddl(BOWL_BDDL)

    if not _has_position_axis(spec):
        pytest.skip(f"{spec} does not include position axis")

    scenario = compile_task_to_scenario(cfg, spec)
    scenes = _sample_scenes(scenario, n=N_SAMPLES)
    assert len(scenes) >= 2, f"{spec}: need at least 2 samples"

    canonical = _canonical_positions(cfg)
    all_positions = [_extract_positions(s) for s in scenes]

    any_deviated = False
    for name, (cx, cy) in canonical.items():
        for pos_dict in all_positions:
            if name in pos_dict:
                sx, sy = pos_dict[name]
                if np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2) > 0.005:
                    any_deviated = True
                    break
        if any_deviated:
            break

    assert any_deviated, (
        f"{spec}: no sample deviated from canonical positions — perturbation not applied"
    )
