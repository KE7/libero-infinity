# Python API Reference

[Back to main README](../README.md)

---

## `libero_infinity.simulator`

### `LIBEROSimulator`

```python
from libero_infinity.simulator import LIBEROSimulator

sim = LIBEROSimulator(bddl_path="path/to/task.bddl", env_kwargs={...})
episode = sim.createSimulation(scene, maxSteps=300, timestep=0.05)
episode.setup()      # init env, inject poses, apply perturbations
# ... drive with step_with_action() ...
episode.destroy()
```

### `LIBEROSimulation` key methods

| Method | Description |
|--------|-------------|
| `setup()` | Init LIBERO env, inject Scenic poses, apply camera/lighting/texture perturbations, `mj_forward` |
| `step()` | Advance physics with zero action (one control timestep) |
| `step_with_action(action)` | Drive with real policy action; returns `(obs, reward, done, info)` |
| `check_success()` | Query LIBERO's task predicate |
| `getProperties(obj, props)` | Read `position`/`orientation`/`velocity`/`speed` from MuJoCo |
| `destroy()` | Close env, clean up temp files |
| `last_obs` | Most recent observation dict |
| `mj_handles` | Raw MuJoCo `(model, data)` handles |

---

## `libero_infinity.eval`

### `evaluate()`

```python
from libero_infinity.eval import evaluate

results = evaluate(
    scenic_path="scenic/combined_perturbation.scenic",
    bddl_path="path/to/task.bddl",
    policy=my_policy_fn,     # callable: obs_dict -> action_array
    n_scenes=200,
    max_steps=300,
    scenic_params={...},     # override Scenic globalParameters
    env_kwargs={...},        # forward to OffScreenRenderEnv
    verbose=True,
    seed=42,
    render_live="cv2",       # None | "cv2" | "viewer"
    camera="agentview",
)
```

### `evaluate_adversarial()`

```python
from libero_infinity.eval import evaluate_adversarial

results = evaluate_adversarial(
    scenic_path="scenic/verifai_position.scenic",
    bddl_path="path/to/task.bddl",
    policy=my_policy_fn,
    n_samples=200,
    max_steps=300,
    verbose=True,
)
```

### `EvalResults`

| Field | Type | Description |
|-------|------|-------------|
| `scenic_path` | `str` | Path to the Scenic program used |
| `bddl_path` | `str` | Path to the BDDL task file |
| `n_scenes` | `int` | Total number of scenes evaluated |
| `n_success` | `int` | Number of successful episodes |
| `success_rate` | `float` | Fraction of successful episodes |
| `ci_95` | `float` | 95% Wilson confidence interval half-width |
| `episodes` | `list[EpisodeResult]` | Per-episode data |
| `summary()` | method | Human-readable summary string |
| `to_json()` | method | JSON serialization |

### `EpisodeResult`

| Field | Type | Description |
|-------|------|-------------|
| `scene_index` | `int` | Scene number |
| `success` | `bool` | Whether the task was completed |
| `steps` | `int` | Number of steps taken |
| `n_scenic_rejections` | `int` | Scenic rejection samples before acceptance |
| `scenic_params` | `dict` | Global parameters from the scene |
| `object_positions` | `dict[str, list]` | `libero_name → [x, y, z]` |
| `object_classes` | `dict[str, str]` | `libero_name → asset_class` |
| `elapsed_s` | `float` | Wall-clock time for this episode |

---

## `libero_infinity.gym_env`

### `LIBEROScenicEnv`

Standard `gym.Env` wrapper for RL/VLA training. See [Gym wrapper docs](gym-wrapper.md).

```python
from libero_infinity.gym_env import LIBEROScenicEnv

env = LIBEROScenicEnv(
    bddl_path="path/to/task.bddl",
    perturbation="position",
)
obs = env.reset()
obs, reward, done, info = env.step(action)
env.close()
```

### `make_vec_env()`

Create parallel vectorized environments.

```python
from libero_infinity.gym_env import make_vec_env

vec_env = make_vec_env(
    bddl_path="path/to/task.bddl",
    n_envs=4,
    perturbation="combined",
)
```

---

## `libero_infinity.task_config`

### `TaskConfig`

```python
from libero_infinity.task_config import TaskConfig

cfg = TaskConfig.from_bddl("path/to/any_task.bddl")
cfg.movable_objects   # [ObjectInfo(instance_name, object_class, init_x, init_y)]
cfg.fixtures          # [FixtureInfo(instance_name, fixture_class)]
cfg.regions           # {name: RegionInfo(target, x_min, x_max, y_min, y_max)}
cfg.language          # "put the bowl on the plate"
cfg.perturbable_classes  # {"akita_black_bowl", "plate"} — classes with OOD variants
```

### `ObjectInfo`

| Field | Type | Description |
|-------|------|-------------|
| `instance_name` | `str` | BDDL instance name (e.g. `"akita_black_bowl_1"`) |
| `object_class` | `str` | Asset class (e.g. `"akita_black_bowl"`) |
| `region_name` | `str \| None` | Placement region name |
| `init_x` | `float \| None` | Centre of placement region (x) |
| `init_y` | `float \| None` | Centre of placement region (y) |
| `stacked_on` | `str \| None` | Instance name of parent object for stacking |

---

## `libero_infinity.compiler` (scenic generation)

```python
from libero_infinity.compiler import generate_scenic, generate_scenic_file

# Generate as string
code = generate_scenic(cfg, perturbation="full", min_clearance=0.10)

# Generate and write to file (returns path)
path = generate_scenic_file(cfg, perturbation="combined")
```

### `generate_scenic()` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cfg` | `TaskConfig` | required | Parsed BDDL task config |
| `perturbation` | `str` | `"position"` | Perturbation axis |
| `min_clearance` | `float` | `0.10` | Minimum clearance between objects (m) |
| `ood_margin` | `float` | `0.15` | Preferred OOD distance (m) |
| `workspace_margin` | `float` | `0.05` | Inset from workspace edges (m) |
| `max_distractors` | `int` | `3` | Max distractor slots |
| `min_distractors` | `int` | `1` | Min active distractors |
| `distractor_pool` | `list[str] \| None` | `None` | Custom distractor classes |
| `distractor_clearance` | `float` | `0.08` | Min distractor clearance (m) |

---

## `libero_infinity.task_reverser`

```python
from libero_infinity.task_reverser import reverse_bddl

# Reverse a BDDL task: swap init and goal conditions
reversed_content = reverse_bddl(
    bddl_content,                   # full BDDL text
    return_region_margin=0.05,      # half-width of return region (metres)
)
# Returns modified BDDL string with reversed task semantics
```

---

## `libero_infinity.bddl_preprocessor`

```python
from libero_infinity.bddl_preprocessor import (
    substitute_asset,           # (bddl_str, orig_class, repl_class) -> str
    substitute_multi,           # (bddl_str, {orig: repl, ...}) -> str
    patched_bddl,               # context manager: yields path to temp file (from file)
    patched_bddl_from_string,   # context manager: yields path to temp file (from string)
    parse_object_classes,       # (bddl_str) -> {instance: class}
    add_distractor_objects,     # (bddl_str, [(name, class)]) -> str
)
```

---

## `libero_infinity.asset_registry`

```python
from libero_infinity.asset_registry import (
    ASSET_VARIANTS,        # {class: [variants]}  — loaded from JSON
    OBJECT_DIMENSIONS,     # {class: [w, l, h]}   — loaded from JSON
    get_variants,          # (class, include_canonical=True) -> list
    has_variants,          # (class) -> bool
    get_dimensions,        # (class) -> (w, l, h)
)
```

---

## `libero_infinity.perturbation_audit`

Utilities for auditing generated perturbations — measuring constraint structure, computing object/anchor displacements, and scoring visible-change magnitude.

### `analyze_generated_constraints()`

```python
from libero_infinity.perturbation_audit import analyze_generated_constraints

audit = analyze_generated_constraints(scenic_code)   # scenic_code: str
audit.hard_require_total    # int — total hard require statements
audit.soft_require_total    # int — total soft require[weight] statements
audit.soft_ood_bias         # int — soft OOD-bias constraints
audit.temporal_operators    # tuple[str, ...] — e.g. ("always", "eventually")
```

### `ConstraintAudit`

| Field | Type | Description |
|-------|------|-------------|
| `hard_require_total` | `int` | Total hard `require` statements |
| `soft_require_total` | `int` | Total soft `require[weight]` statements |
| `hard_axis_clearance` | `int` | Hard axis-clearance constraints (`abs(position.x/y)`) |
| `hard_distance_clearance` | `int` | Hard distance clearance constraints |
| `soft_ood_bias` | `int` | Soft OOD-distance bias constraints |
| `temporal_require_total` | `int` | Requires containing temporal operators |
| `temporal_operators` | `tuple[str, ...]` | Temporal operators found (`always`, `eventually`, etc.) |

### `score_visible_change()`

```python
from libero_infinity.perturbation_audit import score_visible_change, VisibleChangeScoreConfig

score = score_visible_change(
    frame_a,                        # np.ndarray — canonical render (H×W×3, uint8)
    frame_b,                        # np.ndarray — perturbed render
    anchor_records,                 # Sequence[AnchorPixelRecord], optional
    config=VisibleChangeScoreConfig(),
)
score.material_visible_change   # bool — deterministic pass/fail
score.combined_score            # float [0, 1]
score.should_run_vlm_check      # bool — borderline → send to VLM
```

### `VisibleChangeScore`

| Field | Type | Description |
|-------|------|-------------|
| `rgb_mean_delta` | `float` | Mean absolute per-pixel RGB difference in [0, 1] |
| `rgb_score` | `float` | Normalised RGB component score |
| `anchor_summary` | `AnchorPixelSummary` | Aggregated anchor visibility/displacement stats |
| `anchor_displacement_score` | `float` | Normalised anchor motion score |
| `anchor_visibility_score` | `float` | Mean perturbed visibility fraction |
| `combined_score` | `float` | Weighted combination of the three scores |
| `material_rgb_change` | `bool` | RGB delta exceeds material threshold |
| `material_anchor_motion` | `bool` | Anchor displacement exceeds motion threshold |
| `anchor_visibility_ok` | `bool` | Perturbed anchors sufficiently visible |
| `material_visible_change` | `bool` | Overall pass/fail verdict |
| `should_run_vlm_check` | `bool` | Whether VLM secondary check is warranted |

### `VisibleChangeScoreConfig`

Configuration dataclass for `score_visible_change()`.

| Field | Default | Description |
|-------|---------|-------------|
| `rgb_delta_material_threshold` | `0.015` | Minimum mean RGB delta to count as material |
| `anchor_displacement_reference_px` | `12.0` | Reference displacement (score = 1.0 at this value) |
| `anchor_motion_material_threshold_px` | `6.0` | Minimum anchor motion to count as material |
| `minimum_perturbed_visibility_fraction` | `0.5` | Required visible anchor fraction |
| `minimum_perturbed_in_frame_fraction` | `0.5` | Required in-frame anchor fraction |
| `rgb_weight` | `0.55` | Weight of RGB component in combined score |
| `anchor_displacement_weight` | `0.30` | Weight of anchor motion component |
| `anchor_visibility_weight` | `0.15` | Weight of anchor visibility component |
| `combined_material_threshold` | `0.35` | Threshold for `material_visible_change` |
| `vlm_ambiguity_lower` | `0.25` | Lower bound of VLM-check ambiguity zone |
| `vlm_ambiguity_upper` | `0.75` | Upper bound of VLM-check ambiguity zone |

### Other public functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `canonical_xy_for_object` | `(cfg, obj) -> tuple[float, float] \| None` | Canonical XY anchor for a movable object |
| `moving_support_names` | `(cfg) -> tuple[set, set, dict]` | Movable support fixture/object names and parent map |
| `object_displacements` | `(cfg, scene_objects) -> dict[str, float]` | XY displacement of each movable object from its canonical pose |
| `support_displacements` | `(cfg, scene_objects) -> dict[str, float]` | XY displacement of movable supports from their canonical pose |
| `mean_absolute_image_delta` | `(frame_a, frame_b) -> float` | Mean absolute RGB delta in [0, 1] |
| `parse_anchor_pixel_records` | `(payload) -> list[AnchorPixelRecord]` | Parse JSON-like anchor payloads into typed records |
| `summarize_anchor_pixel_records` | `(records, frame_shape) -> AnchorPixelSummary` | Aggregate anchor visibility and displacement statistics |
| `summarize_numeric` | `(values) -> NumericSummary` | Basic summary statistics (count, mean, median, p10, p90, min, max) |

### `GeminiVisibilityChecker`

Optional LiteLLM helper for curated VLM-based visibility / occlusion audits. Requires `litellm` to be installed.

```python
from libero_infinity.perturbation_audit import GeminiVisibilityChecker

checker = GeminiVisibilityChecker(
    model="vertex/gemini-3-flash-preview",
    project="my-gcp-project",    # optional; resolved from env/ADC if omitted
    location="global",
)
text = checker.describe_visibility(
    prompt="Is the bowl visible and unoccluded?",
    image_bytes=png_bytes,
)
```

---

## `libero_infinity.task_semantics`

Typed task semantics derived from a `TaskConfig` — goal predicates, support graphs, articulatable fixtures, visibility targets, yaw hints, and object coordination groups.

### `derive_task_semantics()`

```python
from libero_infinity.task_semantics import derive_task_semantics

sem = derive_task_semantics(cfg)       # cfg: TaskConfig
sem.goal_predicates                    # tuple[AtomicGoalPredicate, ...]
sem.goal_region_exclusions             # tuple[GoalRegionExclusion, ...]
sem.init_support_graph                 # tuple[SupportGraphEdge, ...]
sem.goal_support_graph                 # tuple[SupportGraphEdge, ...]
sem.articulatable_fixtures             # tuple[ArticulatableFixtureSemantics, ...]
sem.visibility_targets                 # tuple[VisibilityTarget, ...]
sem.yaw_hints                          # tuple[YawHint, ...]
sem.coordination_groups                # tuple[CoordinationGroup, ...]
```

### `TaskSemantics`

| Field | Type | Description |
|-------|------|-------------|
| `goal_predicates` | `tuple[AtomicGoalPredicate, ...]` | Parsed goal predicates with typed target decomposition |
| `goal_region_exclusions` | `tuple[GoalRegionExclusion, ...]` | Bounded goal regions excluded from reset placement |
| `init_support_graph` | `tuple[SupportGraphEdge, ...]` | Support/containment graph for the initial configuration |
| `goal_support_graph` | `tuple[SupportGraphEdge, ...]` | Support/containment graph for the goal configuration |
| `articulatable_fixtures` | `tuple[ArticulatableFixtureSemantics, ...]` | Metadata for task-relevant articulatable fixtures |
| `visibility_targets` | `tuple[VisibilityTarget, ...]` | Entities that must remain visible during task execution |
| `yaw_hints` | `tuple[YawHint, ...]` | Yaw priors sourced from BDDL region annotations |
| `coordination_groups` | `tuple[CoordinationGroup, ...]` | Objects sharing the same resolved goal support layout |

### `AtomicGoalPredicate`

| Field | Type | Description |
|-------|------|-------------|
| `predicate` | `GoalPredicateType` | `"On"`, `"In"`, `"Open"`, `"Close"`, `"Turnon"`, or `"Turnoff"` |
| `raw_arguments` | `tuple[str, ...]` | Raw argument strings from the BDDL |
| `primary_name` | `str` | Primary entity name (object being placed or actuated) |
| `primary_kind` | `EntityKind` | `"object"`, `"fixture"`, `"region"`, or `"unknown"` |
| `target_name` | `str \| None` | Target / destination entity name |
| `target_kind` | `EntityKind \| None` | Kind of the target entity |
| `support_instance_name` | `str \| None` | Resolved support fixture or object instance |
| `support_region_name` | `str \| None` | Resolved region name within the support |

### `ArticulatableFixtureSemantics`

| Field | Type | Description |
|-------|------|-------------|
| `fixture_name` | `str` | BDDL instance name |
| `fixture_class` | `str` | Asset class (e.g. `"microwave"`) |
| `family` | `ArticulationFamily` | `"cabinet"` or `"microwave"` |
| `articulation_kind` | `ArticulationKind` | `"drawer"` or `"door"` |
| `control_target_name` | `str` | Name of the controlled joint target |
| `compartment_name` | `str \| None` | Sub-compartment label (`"top"`, `"middle"`, `"bottom"`, `"heating"`) |
| `init_state` | `ArticulationState` | `"open"`, `"closed"`, or `"unknown"` |
| `goal_state` | `ArticulationState` | `"open"`, `"closed"`, or `"unknown"` |

---

## `libero_infinity.vision_validation`

Secondary VLM-based ambiguity checks for perturbation audits. Requires `litellm[google]` and Google Cloud credentials.

### `run_curated_ambiguity_check()`

```python
from libero_infinity.vision_validation import run_curated_ambiguity_check

result = run_curated_ambiguity_check(
    task_instruction="put the bowl on the plate",
    visible_change=score,            # VisibleChangeScore from perturbation_audit
    canonical_image=frame_a,         # np.ndarray | bytes | Path | str
    perturbed_image=frame_b,
    model="vertex_ai/gemini-3-flash-preview",
    project=None,                    # resolved from env / ADC if omitted
    location=None,                   # defaults to "global" for Gemini 3
    timeout=60,
    temperature=0.0,
)
result.decision    # "clear" | "ambiguous" | "not_visible"
result.confidence  # float [0, 1] or None
result.reasoning   # model explanation string
```

### `VisionValidationResult`

| Field | Type | Description |
|-------|------|-------------|
| `decision` | `str` | `"clear"`, `"ambiguous"`, `"not_visible"`, or error codes `"parse_error"` / `"request_error"` |
| `confidence` | `float \| None` | Model confidence in [0, 1], or `None` on parse error |
| `reasoning` | `str` | Free-text reasoning from the model |
| `raw_response` | `str` | Verbatim model response before parsing |
| `model` | `str` | Model identifier used |
| `project` | `str` | GCP project used |
| `location` | `str` | Vertex AI location used |

### Other public functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_ambiguity_messages` | `(*, task_instruction, visible_change, canonical_image, perturbed_image) -> list[dict]` | Build multimodal LiteLLM messages anchored by the deterministic score |
| `parse_vision_validation_response` | `(response, *, model, project, location) -> VisionValidationResult` | Parse JSON-ish VLM output into a stable result shape |
| `resolve_vertex_project` | `(project=None) -> str` | Resolve GCP project from argument, env vars, ADC, or `gcloud` |
| `resolve_vertex_location` | `(location=None, *, model) -> str` | Resolve Vertex location (defaults `"global"` for Gemini 3 preview models) |

---

## `libero_infinity.scene_semantics`

> **Internal module** — not exported in `libero_infinity.__all__`. Prefer `libero_infinity.task_semantics` for the richer typed API.

Low-level helpers for extracting structured task semantics from a `TaskConfig`, including flat predicate parsing and articulation band selection.

```python
from libero_infinity.scene_semantics import (
    parse_atomic_predicates,
    goal_predicates,
    init_predicates,
    task_relevant_object_names,
    coordination_groups,
    articulated_fixture_specs,
    support_contains_articulated_compartment,
)
```

### Public functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `parse_atomic_predicates` | `(text) -> list[AtomicPredicate]` | Extract flat `(predicate, arg1, arg2?)` atoms from a BDDL block |
| `goal_predicates` | `(cfg) -> list[AtomicPredicate]` | Predicates from the goal block |
| `init_predicates` | `(cfg) -> list[AtomicPredicate]` | Predicates from the init block |
| `task_relevant_object_names` | `(cfg) -> set[str]` | Movable objects the policy must be able to see |
| `coordination_groups` | `(cfg) -> dict[str, list[str]]` | Objects grouped by shared support / container |
| `articulated_fixture_specs` | `(cfg) -> dict[str, ArticulationSpec]` | Safe articulation bands keyed by fixture instance name |
| `support_contains_articulated_compartment` | `(cfg, fixture_name) -> bool` | Whether a fixture has task objects inside a moving compartment |

### `ArticulationSpec`

| Field | Type | Description |
|-------|------|-------------|
| `fixture_name` | `str` | BDDL instance name |
| `state_kind` | `str` | `"Open"`, `"Close"`, `"Turnon"`, or `"Turnoff"` |
| `lo` | `float` | Lower joint-angle bound (radians) |
| `hi` | `float` | Upper joint-angle bound (radians) |
| `reason` | `str` | Why this band was chosen (e.g. `"goal_open"`, `"canonical_init_band"`) |

---

## `libero_infinity.perturbation_policy`

> **Internal module** — not exported in `libero_infinity.__all__`. Low-level helpers for programmatic (non-Scenic) position perturbation policies.

Implements support-aware position envelopes, coordinated group transforms, and yaw-range inference for use outside the Scenic generator.

```python
from libero_infinity.perturbation_policy import (
    NumericRange, YawRange, LocalEnvelope, GroupTransform,
    parse_region_yaw_ranges_from_text,
    parse_region_yaw_ranges_from_file,
    resolve_object_yaw_ranges,
    infer_support_type,
    support_local_envelope,
    coordination_groups,
    sample_group_transform,
    apply_group_transform,
)
```

### Key types

| Class | Description |
|-------|-------------|
| `NumericRange(minimum, maximum)` | Closed interval with `.span` property and `.sample(rng) -> float` |
| `YawRange(minimum, maximum)` | Yaw interval in radians (subclass of `NumericRange`) |
| `LocalEnvelope(x_half_extent, y_half_extent, support_type)` | Support-relative local perturbation envelope |
| `GroupTransform(translation, shared_yaw, local_jitter)` | Shared translation + yaw with per-object jitter |

### Key functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `infer_support_type` | `(*, support_class, region_name, contained) -> str` | Classify support as `"contained"`, `"cook_surface"`, `"shelf_surface"`, `"workspace"`, or `"object_surface"` |
| `support_local_envelope` | `(*, support_dims, child_dims, support_class, region_name, contained, clearance_margin) -> LocalEnvelope` | Compute support-relative xy half-extents |
| `coordination_groups` | `(cfg) -> dict[str, tuple[ObjectInfo, ...]]` | Group movable objects by shared support or root workspace |
| `sample_group_transform` | `(object_names, *, translation_x_range, translation_y_range, rng, shared_yaw_range, local_jitter_range) -> GroupTransform` | Sample a coordinated transform for a group of objects |
| `apply_group_transform` | `(canonical_positions, *, anchor_xy, transform) -> dict[str, tuple[float, float]]` | Apply shared translation/yaw plus per-object jitter to XY positions |
| `resolve_object_yaw_ranges` | `(cfg, *, default) -> dict[str, YawRange \| None]` | Infer yaw range per movable object from BDDL region metadata |
| `parse_region_yaw_ranges_from_text` | `(bddl_text) -> dict[str, YawRange]` | Extract per-region yaw ranges from BDDL text |
| `parse_region_yaw_ranges_from_file` | `(path) -> dict[str, YawRange]` | Load a BDDL file and extract its region yaw ranges |

---

## `libero_infinity.perturbation_policy_helpers`

> **Internal module** — not exported in `libero_infinity.__all__`. Geometry helpers for computing support offset bounds and yaw intervals; used by the position perturbation policy.

```python
from libero_infinity.perturbation_policy_helpers import (
    support_offset_bounds,
    yaw_bounds,
    coordinated_group_offset,
)
```

### `support_offset_bounds()`

```python
x_max, y_max = support_offset_bounds(
    support_dims=(0.30, 0.20, 0.02),   # (width, length, height) in metres
    child_dims=(0.10, 0.08, 0.06),
    support_class="plate",
    region_name=None,
    contained=False,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `support_dims` | `tuple[float, float, float]` | Support (width, length, height) in metres |
| `child_dims` | `tuple[float, float, float]` | Child object (width, length, height) in metres |
| `support_class` | `str \| None` | Asset class of the support (e.g. `"plate"`, `"tray"`) |
| `region_name` | `str \| None` | Named region within the support (e.g. `"cook_region"`) |
| `contained` | `bool` | Whether the child is inside the support (applies extra scale reduction) |

Returns `(x_half_extent, y_half_extent)` — maximum local offsets in metres.

### `yaw_bounds()`

```python
lo, hi = yaw_bounds(canonical_yaw=0.0, asset_class="akita_black_bowl")
```

Returns a `(lo, hi)` yaw interval in radians around `canonical_yaw`, sized by class keyword matching. Returns `None` only when a fully unknown class has no round-object heuristic match.

### `coordinated_group_offset()`

```python
dx, dy = coordinated_group_offset(member_count=3, support_dims=(0.30, 0.20, 0.02))
```

Returns the shared `(x, y)` translation range for multiple objects on the same support (grows with `member_count`, capped at 30 % of support size).

---

## `libero_infinity.validation_errors`

Typed exception hierarchy for scenario post-settle validation, with recovery strategy metadata.

```python
from libero_infinity.validation_errors import (
    ScenarioValidationError,
    CollisionError,
    VisibilityError,
    InfeasibleScenarioError,
    RECOVERY_STRATEGY,
    MAX_VISIBILITY_RETRIES,     # = 10
)
```

### Exception classes

| Class | Inherits | Recovery strategy | Description |
|-------|----------|-------------------|-------------|
| `ScenarioValidationError` | `RuntimeError` | — | Base class for all post-settle validation errors |
| `CollisionError` | `ScenarioValidationError` | `"propagate_immediately"` | Objects in collision after MuJoCo settling; carries `.object_names: list[str]` |
| `VisibilityError` | `ScenarioValidationError` | `"resample_scenario"` | Task-relevant objects invisible after settling; carries `.invisible_names: list[str]` |
| `InfeasibleScenarioError` | `Exception` | — | All `MAX_VISIBILITY_RETRIES` attempts exhausted; carries `.n_resample` and `.n_replan` |

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `MAX_VISIBILITY_RETRIES` | `10` | Maximum re-sample attempts for a `VisibilityError` |
| `RECOVERY_STRATEGY` | `dict[type, str]` | Maps each typed error class to its recovery strategy name |

---

## `libero_infinity.ir`

Intermediate Representation (IR) — a typed semantic scene graph built from a parsed `TaskConfig`, independent of any Scenic syntax. Used internally by the planner to compute axis-specific perturbation plans.

```python
from libero_infinity.ir import (
    build_semantic_scene_graph,    # main entry point: TaskConfig -> SemanticSceneGraph
    SemanticSceneGraph, SemanticError,
    # Node types
    SceneNode, WorkspaceNode, FixtureNode, MovableSupportNode,
    ObjectNode, RegionNode, CameraNode, LightNode, DistractorSlotNode,
    # Edge
    SceneEdge,
    # Models / diagnostics
    ArticulationModel, PlanDiagnostics,
)
```

### `build_semantic_scene_graph()`

```python
from libero_infinity.ir import build_semantic_scene_graph

graph = build_semantic_scene_graph(cfg)   # cfg: TaskConfig
graph.nodes          # dict[str, SceneNode]
graph.edges          # list[SceneEdge]
graph.task_language  # "put the bowl on the plate"
graph.bddl_path      # original BDDL path
```

Converts a flat `TaskConfig` into a typed, validated scene graph with explicit support / containment / articulation edges. Calls `validate_dag()` before returning; raises `SemanticError` on a cycle.

### `SemanticSceneGraph` methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_node` | `(node) -> None` | Register a node (keyed by `node_id`) |
| `add_edge` | `(edge) -> None` | Append a directed edge |
| `get_node` | `(node_id) -> SceneNode \| None` | Look up a node by id |
| `edges_from` | `(node_id) -> list[SceneEdge]` | Outgoing edges from a node |
| `edges_to` | `(node_id) -> list[SceneEdge]` | Incoming edges to a node |
| `edges_by_label` | `(label) -> list[SceneEdge]` | All edges with a given label |
| `validate_dag` | `() -> None` | Topological-sort check; raises `SemanticError` on cycle |

### Node types

| Class | `node_type` | Key extra fields |
|-------|-------------|------------------|
| `WorkspaceNode` | `"workspace"` | `surface_bounds` |
| `FixtureNode` | `"fixture"` | `placement_target`, `init_x/y/yaw`, `is_articulatable` |
| `MovableSupportNode` | `"movable_support"` | `placement_target`, `init_x/y/yaw`, `stacked_on` |
| `ObjectNode` | `"object"` | `placement_target`, `init_x/y/yaw`, `stacked_on`, `contained` |
| `RegionNode` | `"region"` | `target`, `x_min/max`, `y_min/max`, `yaw_min/max` |
| `CameraNode` | `"camera"` | — |
| `LightNode` | `"light"` | — |
| `DistractorSlotNode` | `"distractor_slot"` | `slot_index` (0–4) |

All nodes share the base `SceneNode` fields: `node_id`, `node_type`, `instance_name`, `object_class`, `metadata`.

### Edge labels

| Label | Direction | Meaning |
|-------|-----------|---------|
| `"supported_by"` | object → fixture/workspace | Object rests on a support surface |
| `"contained_in"` | object → fixture/object | Object is inside a container |
| `"stacked_on"` | object → object | Object stacked on another movable object |
| `"anchored_to"` | fixture → workspace | Non-movable fixture is rooted to the workspace |
| `"articulated_by"` | fixture → itself | Articulatable fixture self-loop marker |
| `"must_remain_visible_with"` | object → camera | Object must stay in the camera frustum |
| `"goal_target"` | object → region/fixture | Object's goal placement target |

### `ArticulationModel`

Encodes knowledge about articulatable fixture families and joint-angle ranges.

```python
from libero_infinity.ir import ArticulationModel

model = ArticulationModel.canonical()    # built-in LIBERO fixtures
model.is_articulatable("microwave")      # True
model.get_range("microwave", "Open")    # (-2.094, -1.3)
model.get_family("white_cabinet")       # ("cabinet", "drawer")
```

### `PlanDiagnostics`

Tracks which perturbation axes were constrained, narrowed, or dropped during planning.

| Method | Description |
|--------|-------------|
| `drop_axis(axis, reason)` | Mark axis as completely dropped |
| `narrow_axis(axis, reason)` | Mark axis as narrowed (range reduced but still active) |
| `constrain_axis(axis, reason)` | Mark axis as constrained to a specific value |

Fields: `constrained_axes`, `narrowed_axes`, `dropped_axes`, `reasons` (dict), `warnings`.

---

## `libero_infinity.planner`

Full perturbation planning pipeline: composes independent per-axis plans into a single `PerturbationPlan`, then applies a single-pass cross-axis validation step.

```python
from libero_infinity.planner import (
    plan_perturbations,    # main entry point
    parse_axes,            # expand preset names to frozenset
    plan_position,         # position-axis sub-planner
    plan_lighting,         # lighting-axis sub-planner
    plan_texture,          # texture-axis sub-planner
    # Result types
    PerturbationPlan, PositionPlan, AxisEnvelope,
    ArticulationPlan, CameraPlan, LightingPlan, TexturePlan,
    InfeasiblePerturbationError,
)
```

### `plan_perturbations()`

```python
from libero_infinity.planner import plan_perturbations
from libero_infinity.ir import build_semantic_scene_graph

graph = build_semantic_scene_graph(cfg)
plan = plan_perturbations(graph, "combined")

plan.position_plans          # dict[str, PositionPlan]
plan.object_substitutions    # dict[str, list[str]]
plan.articulation_plans      # dict[str, ArticulationPlan]
plan.camera_plan             # CameraPlan | None
plan.lighting_plan           # LightingPlan | None
plan.texture_plan            # TexturePlan | None
plan.background_plan         # BackgroundPlan | None
plan.distractor_budget       # int
plan.distractor_classes      # list[str]
plan.active_axes             # frozenset[str]
plan.diagnostics             # PlanDiagnostics
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `SemanticSceneGraph` | Scene graph from `build_semantic_scene_graph()` |
| `request` | `str \| frozenset[str]` | Axis specification; presets or comma-separated axis names |

**Axis presets:**

| Preset | Axes included |
|--------|---------------|
| `"combined"` | `position`, `object`, `camera`, `lighting`, `distractor`, `background` |
| `"full"` | All of `"combined"` plus `texture` and `articulation` |

### `parse_axes()`

```python
from libero_infinity.planner import parse_axes

parse_axes("position,camera")   # frozenset({"position", "camera"})
parse_axes("combined")          # expands to the full combined preset
```

### Per-axis sub-planners

Each sub-planner has the signature `(graph, request_axes, diagnostics) -> result`:

| Function | Module | Returns | Description |
|----------|--------|---------|-------------|
| `plan_position` | `planner.position` | `dict[str, PositionPlan]` | Per-object position envelopes from the scene graph |
| `plan_object` | `planner.axes` | `dict[str, list[str]]` | Variant substitution pools per object |
| `plan_articulation` | `planner.axes` | `dict[str, ArticulationPlan]` | Initial articulation states (always runs for goal-reachability) |
| `plan_camera` | `planner.axes` | `CameraPlan \| None` | Camera perturbation envelope |
| `plan_lighting` | `planner.axes` | `LightingPlan \| None` | Lighting intensity / ambient / position ranges |
| `plan_texture` | `planner.axes` | `TexturePlan \| None` | Table surface texture (`"random"` or named) |
| `plan_background` | `planner.axes` | `BackgroundPlan \| None` | Wall/floor texture candidate pool |
| `plan_distractor` | `planner.axes` | `tuple[int, list[str]]` | `(budget, class_pool)` for distractor objects |

### `PositionPlan`

| Field | Type | Description |
|-------|------|-------------|
| `object_name` | `str` | BDDL instance name |
| `x_envelope` | `AxisEnvelope` | Workspace x-axis perturbation range (`lo`, `hi`, `axis`) |
| `y_envelope` | `AxisEnvelope` | Workspace y-axis perturbation range |
| `support_name` | `str` | Node id of the support surface |
| `use_relative_positioning` | `bool` | `True` for stacked or contained objects |
| `yaw_lo` | `float` | Minimum yaw in radians (default −π) |
| `yaw_hi` | `float` | Maximum yaw in radians (default +π) |
| `exclusion_zones` | `list[tuple[float,float,float,float]]` | Axis-aligned boxes `(x0, y0, x1, y1)` where the task is pre-solved |
| `exclusion_min_distance` | `float \| None` | Distance-based exclusion fallback |

### `ArticulationPlan`

| Field | Type | Description |
|-------|------|-------------|
| `fixture_name` | `str` | BDDL instance name |
| `state_kind` | `str` | `"Open"`, `"Close"`, `"Turnon"`, or `"Turnoff"` |
| `lo` | `float` | Lower joint-angle bound (radians) |
| `hi` | `float` | Upper joint-angle bound (radians) |
| `reason` | `str` | Why this band was chosen |
| `goal_reachability_ok` | `bool` | Whether the goal remains reachable from this init state |

### `InfeasiblePerturbationError`

Raised when a perturbation envelope collapses to zero volume. Carries `.diagnostics: PlanDiagnostics | None`.
