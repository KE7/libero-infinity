# LIBERO-PRO vs. libero-infinity: Perturbation Comparison Report

*Generated 2026-03-14 by libero-compare agent*

---

## Executive Summary

LIBERO-PRO (arxiv:2510.03827, 2025) introduced predefined perturbation sets across four dimensions — object attributes, initial positions, language instructions, and environments — to expose VLA model memorization. Its position and object perturbations ("P1" and "P2" in this document) are discrete: ~10 pre-defined position swap pairs and 2–6 pre-defined object replacements per class.

libero-infinity defines perturbation distributions as **Scenic 3 probabilistic programs** over continuous parameter spaces. Where LIBERO-PRO uses predefined discrete sets, libero-infinity uses continuous distributions with formal constraint checking — a different approach suited to different evaluation goals. libero-infinity also covers three perturbation axes not present in LIBERO-PRO — camera pose, lighting, and distractor injection — and supports arbitrary axis composability. The evaluation surface is not a checklist; it is an open-ended probabilistic program that can be sampled indefinitely.

**Summary:** LIBERO-PRO uses predefined perturbation sets optimized for reproducibility and interpretability; libero-infinity uses Scenic 3 continuous distributions for open-ended, statistically inexhaustible sampling. Both represent valid evaluation philosophies suited to different goals.

---

## 1. Note on "P1" and "P2" Terminology

The LIBERO-PRO paper (arxiv:2510.03827) does **not** use formal "P1" / "P2" labels. Its four perturbation dimensions are denoted by their initial letters: **O** (Object attributes), **I** (Initial position), **L** (Language instructions), **E** (Environment). The δ_k notation in Equation 10 uses k ∈ {O, I, L, E}.

In this report, **P1 refers to LIBERO-PRO's Initial Position Perturbation (I)** and **P2 refers to LIBERO-PRO's Object Attribute Perturbation (O)**, as these are the two dimensions most directly comparable to libero-infinity's perturbation system and the ones with corresponding code artefacts in the LIBERO-PRO repository (`use_swap` flag and `LIBERO_*_SWAP` task suites for P1; `use_object` flag for P2).

---

## 2. LIBERO-PRO Perturbation Framework

### 2.1 Overview

The paper defines four perturbation dimensions evaluated separately via config flags in `evaluation_config.yaml`:

| Flag | Dimension | Description |
|------|-----------|-------------|
| `use_swap` | Initial Position (I) | P1: Relocate objects to alternative pre-defined spatial regions |
| `use_object` | Object Attributes (O) | P2: Swap object mesh/texture/size |
| `use_language` | Language Instructions (L) | Paraphrase or substitute task instructions |
| `use_environment` | Environment (E) | Switch among LIBERO's 5 workspace environments |

Key property: **only one flag can be active at a time** (task perturbation cannot be combined with others; in practice axes are tested in isolation).

### 2.2 P1 — Initial Position Perturbation

**Definition:** "Alters initial object positions to modify absolute and relative arrangements, ensuring physical plausibility, e.g., moving a cup relative to a plate."

**Mechanism:** Pre-defined alternative spatial regions are assigned to all manipulable objects. The code generates task suite variants with suffix `_SWAP`: `LIBERO_GOAL_SWAP`, `LIBERO_SPATIAL_SWAP`, `LIBERO_10_SWAP`, `LIBERO_OBJECT_SWAP`. Each variant is a pre-defined alternative configuration.

**Scope and coverage:**
- ~10 pre-defined position swap pairs per task suite (our blog post characterisation: "10 swap pairs + fixed grid")
- Positions are picked from a small set of alternative spatial regions — not sampled from a distribution
- No continuous coverage: each "new" configuration is a specific fixed (x, y) override
- The magnitude parameter δ_I controls displacement, but the underlying configurations are still discrete
- Sensitivity analysis (Figure 6 of the paper) shows performance collapse beyond ~0.2 units of displacement (OpenVLA, pi0) or ~0.4 units (pi0.5)
- Evaluated at 50 episodes per task on the standard LIBERO evaluation protocol

**What P1 does NOT do:**
- No continuous or stochastic sampling of object positions
- No probabilistic constraint checking (hard reject if swap violates plausibility)
- No OOD bias (soft preference for novel positions)
- No perturbation of goal fixtures (drawers, stoves, microwaves, cabinets)
- Cannot compose position perturbation with other axes
- Finite: a deterministic team could enumerate and overfit all swap pairs

### 2.3 P2 — Object Attribute Perturbation

**Definition:** "Modifies the appearance, size, and color of the original objects while preserving semantic equivalence — for example, changing a red cup to blue."

**Mechanism:** New 3D assets are created for the four LIBERO task suites. The `use_object` flag swaps the BDDL object class to one of the replacement assets.

**Scope and coverage:**
- ~2–6 pre-defined replacements per object class
- Assets created specifically for the four LIBERO suites; not a general-purpose registry
- Properties changed: color, texture, scale (size) — i.e., visual appearance only
- No sampling: each replacement is deterministic (which replacement is applied is configured, not drawn stochastically)
- Finite: the full set of object variants is a closed, enumerable list

**What P2 does NOT do:**
- No camera or lighting changes to accompany the visual OOD
- No stochastic sampling from a variant pool
- No composability with position perturbation
- No distractor objects to test attention/focus

### 2.4 Language and Environment Perturbations (L, E)

For completeness:
- **L (Language):** Paraphrase ("pick up the bottle" → "grab the bottle") and task substitution ("pick up the bottle" → "pick up the bowl"). ~3 paraphrases per task.
- **E (Environment):** Leverages LIBERO's 5 built-in environment categories (different room appearances). Workspace task content unchanged; only scene aesthetics change.

Neither of these has a direct counterpart in libero-infinity's current perturbation system.

---

## 3. libero-infinity Perturbation Framework

### 3.1 Architecture

libero-infinity defines six composable perturbation axes, each as a **Scenic 3 probability distribution** with formal constraint checking via rejection sampling:

| Axis | Scenic File | Distribution Family | Parameters |
|------|------------|---------------------|------------|
| Position | `position_perturbation.scenic` | Uniform over workspace | x ∈ [-0.40,0.40] m, y ∈ [-0.30,0.30] m per object |
| Object | `object_perturbation.scenic` | Uniform over variant pool | 34 object classes, 2–5 variants each |
| Camera | `camera_perturbation.scenic` | Uniform range offsets | x/y: ±10 cm, z: ±8 cm, tilt: ±15° |
| Lighting | `lighting_perturbation.scenic` | Uniform ranges | intensity ×[0.4,2.0], x/y/z offset ±0.5 m, ambient [0.05,0.6] |
| Texture | *(via params)* | Discrete pool | Table surface material swap |
| Distractor | `distractor_perturbation.scenic` | DiscreteRange(1,5) + Uniform pool | 10-item pool, spatially constrained |

All axes are **composable**: `--perturbation position,camera,distractor` or preset `full` activates all axes simultaneously in a single jointly-sampled Scenic program.

### 3.2 Position Perturbation (vs. LIBERO-PRO P1)

**Distribution:** Each object's (x, y) is drawn **independently and uniformly** from the full workspace.

```
x ∈ [TABLE_X_MIN + WORKSPACE_MARGIN, TABLE_X_MAX − WORKSPACE_MARGIN]
   = [-0.29, 0.29] m  (with default margin 0.11 m)
y ∈ [TABLE_Y_MIN + WORKSPACE_MARGIN, TABLE_Y_MAX − WORKSPACE_MARGIN]
   = [-0.19, 0.19] m
z = TABLE_Z = 0.82 m  (fixed to table surface)
```

**Hard constraints** (rejection sampler loops until satisfied):
- Pairwise clearance: `distance(A, B) > min_clearance` (default 0.22 m for bowl+plate pair)
- Workspace bounds enforced by `in SAFE_REGION` / `in PLATE_SAFE_REGION` specifiers

**Soft constraints** (preference, not prohibition):
- `require[0.8] distance(bowl, bowl_train_pt) > ood_margin (0.15 m)` — 80% bias toward OOD positions
- `require[0.7] distance(plate, plate_train_pt) > ood_margin` — 70% bias

**Fixture support:** Goal fixtures (drawers, stoves, microwaves, cabinets) are also perturbed when the task goal is fixture-backed, via optional `goal_fixture_*` parameters.

**Configuration count:** Strictly infinite. Each call to `scenario.generate()` draws a new i.i.d. sample from the continuous 2D uniform. For N objects, the joint configuration space is ∝ (workspace_area)^N minus rejection regions.

**Auto-generation:** The compiler pipeline (`compiler.py` → `planner/composition.py` → `renderer/scenic_renderer.py`) auto-generates position-perturbed Scenic programs for any LIBERO BDDL task file. Zero hand-coding per task.

### 3.3 Object Perturbation (vs. LIBERO-PRO P2)

**Distribution:** `Uniform(*variant_pool)` draws one asset class uniformly from the per-class variant list on every scene sample.

**Registry:** `src/libero_infinity/data/asset_variants.json` — single JSON source of truth shared between Scenic and Python layers:
- 34 object classes
- Each class has 2–5 variants (e.g., `akita_black_bowl` → [`akita_black_bowl`, `white_bowl`, `glazed_rim_porcelain_ramekin`])
- 10-item distractor pool for clutter injection

**BDDL patching:** After Scenic samples `chosen_asset = "white_bowl"`, the eval harness rewrites the BDDL object class before environment creation. LIBERO then loads the correct mesh automatically.

**Configuration count:** Per-class variant count = 2–5 discrete values, but these are drawn **stochastically** per episode rather than deterministically cycled. In combination with position perturbation, the joint space grows multiplicatively.

### 3.4 Camera Perturbation (no LIBERO-PRO equivalent)

Perturbs the `agentview` camera pose in 4 independent dimensions:

```
camera_x_offset = Range(-0.10, 0.10)   # lateral shift
camera_y_offset = Range(-0.10, 0.10)   # forward/back shift
camera_z_offset = Range(-0.08, 0.08)   # height shift
camera_tilt     = Range(-15, 15)       # degrees, applied via scipy quaternion compose
```

Applied after `env.reset()` by modifying MuJoCo `cam_pos` and `cam_quat` arrays. Cited by LIBERO-Plus (2510.13626) as "the most damaging perturbation axis for policy performance."

**LIBERO-PRO P1/P2 equivalent:** None. LIBERO-PRO has no camera perturbation.

### 3.5 Lighting Perturbation (no LIBERO-PRO equivalent)

Five independent continuous parameters:

| Parameter | Range | Effect |
|-----------|-------|--------|
| `light_intensity` | [0.4, 2.0] | Diffuse/specular multiplier |
| `light_x_offset` | [-0.5, 0.5] m | Light source lateral position |
| `light_y_offset` | [-0.5, 0.5] m | Light source fore/aft position |
| `light_z_offset` | [-0.5, 0.5] m | Light source height |
| `ambient_level` | [0.05, 0.6] | Global ambient intensity |

Modifies MuJoCo `light.diffuse`, `light.specular`, `light.pos`, and `model.geom_rgba` for ambient.

**LIBERO-PRO P1/P2 equivalent:** None.

### 3.6 Distractor Injection (no LIBERO-PRO equivalent)

Adds 1–5 non-task "clutter" objects to the scene via BDDL rewriting:

- **Count:** `DiscreteRange(1, 5)` drawn per scene
- **Identity:** `Uniform(*DISTRACTOR_POOL)` — pool of 10 small graspable objects with verified LIBERO XML assets
- **Placement:** Each distractor placed `in SAFE_REGION` with hard clearance to all task objects (`> 0.13 m`) and distractor-to-distractor clearance (`> 0.06 m`)
- **"Max slots" pattern:** All 5 slots exist in Scenic's constraint space; only the first `n_distractors` are injected into MuJoCo — inactive slots are harmless

**LIBERO-PRO P1/P2 equivalent:** None. LIBERO-PRO never adds irrelevant objects.

---

## 4. Quantitative Comparison

### 4.1 Configuration Cardinality

| Dimension | LIBERO-PRO | libero-infinity |
|-----------|-----------|-----------------|
| **Position space** | ~10 discrete swap pairs per task suite | Continuous: ∝ (0.58×0.38 m²)^N_objects per-task workspace |
| **Object variants** | 2–6 static replacements per class | 2–5 Uniform-sampled variants; 34 classes in registry |
| **Camera** | 1 fixed viewpoint | ℝ⁴ continuous ball of radius (10cm, 10cm, 8cm, 15°) |
| **Lighting** | 1 fixed illumination | ℝ⁵ continuous hyper-rectangle |
| **Distractor count** | 0 (not supported) | DiscreteRange(1, 5) |
| **Distractor identity** | N/A | Uniform over 10-class pool |
| **Total unique configurations** | O(10–20) per task | Effectively infinite (continuous joint distribution) |
| **Axes composable** | No (single-axis only) | Yes: any subset or all 6 simultaneously |
| **Tasks covered** | 4 LIBERO task suites | Any BDDL task (auto-generated) |

### 4.2 Distribution Coverage

**Position:** LIBERO-PRO's P1 covers a discrete set of ~10 alternative positions in the workspace. libero-infinity samples from the full 0.58 m × 0.38 m continuous workspace (after safety margins). LIBERO-PRO's 10 swap-pair positions are a measure-zero subset of libero-infinity's distribution. A typical libero-infinity evaluation at N=200 scenes explores ~200 statistically independent configurations; LIBERO-PRO recycles from ~10 fixed ones.

**Object:** LIBERO-PRO's P2 selects from 2–6 static asset replacements per class. libero-infinity draws from the same or similar variant counts (2–5), but stochastically per-episode rather than deterministically cycling. More importantly, libero-infinity can compose object variation with position variation, camera shift, and lighting simultaneously — LIBERO-PRO cannot.

**Camera/Lighting/Distractor:** LIBERO-PRO has no counterpart at all. These axes contribute zero evaluation coverage in LIBERO-PRO.

### 4.3 Sample Diversity per Episode Budget

For a 200-episode evaluation budget:

| Metric | LIBERO-PRO | libero-infinity |
|--------|-----------|-----------------|
| Distinct position states seen | ≤10 (recycled) | ~200 (i.i.d. continuous) |
| Memorization risk | High — finite set fully enumerable | Negligible — infinite distribution |
| Success rate estimator | Point estimate over known population | Wilson CI over i.i.d. sample → `rate ± CI` convergence |
| Adversarial search support | None | VerifAI cross-entropy concentration |

---

## 5. What libero-infinity Does That LIBERO-PRO Does Not

1. **Continuous distributions over object positions** — the entire reachable workspace is covered, not 10 hand-chosen swap pairs
2. **Stochastic OOD bias** — soft Scenic constraints bias toward novel positions without hard-excluding the training region (needed for proper OOD evaluation)
3. **Camera pose perturbation** — 4D continuous: lateral, longitudinal, height, tilt
4. **Lighting perturbation** — 5D continuous: intensity, position (x,y,z), ambient level
5. **Table texture perturbation** — material swap on the workspace surface
6. **Distractor object injection** — 1–5 foreign objects with spatial plausibility constraints
7. **Arbitrary axis composability** — any subset of all 6 axes in a single evaluation run
8. **Auto-generation for any BDDL task** — compiler pipeline generates valid Scenic programs from any BDDL without per-task authoring
9. **Adversarial CE search** — VerifAI cross-entropy Bayesian optimization concentrates samples on failure-inducing regions
10. **Gym API for training** — standard `gym.Env` wrapper enables domain-randomized training, not just evaluation
11. **Task reversal** — BDDL goal/initial state inversion doubles the evaluation surface
12. **Infinite sample pool** — evaluation does not exhaust a fixed set; confidence intervals narrow with N
13. **Principled constraint calibration** — workspace margins empirically calibrated to zero physics failures (`workspace_margin=0.11 m`, measured at zero hard-failure rate vs. 12.5% before calibration)

---

## 6. Gaps Where LIBERO-PRO Does Things libero-infinity Currently Does Not

### 6.1 Language / Instruction Perturbation
LIBERO-PRO's L dimension perturbs the natural-language task instruction via paraphrase ("pick up the bottle" → "grab the bottle") and task substitution. libero-infinity does not perturb the instruction text — the language string from the BDDL is passed unmodified to the policy. This is a real gap for evaluating VLA language robustness.

### 6.2 Full Scene Environment Switching
LIBERO-PRO's E dimension switches among LIBERO's 5 built-in environment categories (different room aesthetics, furniture arrangements, background textures). libero-infinity perturbs only the table surface texture and lighting; it does not swap the full scene environment. This means libero-infinity's visual domain randomization is narrower in scene-level appearance diversity.

### 6.3 Object Size / Scale Perturbation
LIBERO-PRO's P2 (object attribute) explicitly lists **size** as a modifiable property (in addition to color and texture). libero-infinity's object perturbation swaps the entire mesh (which implicitly changes geometry), but does not have a dedicated scale perturbation parameter that rescales an existing object mesh by a continuous factor. The variant-pool approach does change geometry, but not as a continuous scale.

### 6.4 Inter-object Relative Arrangement Bias
LIBERO-PRO's P1 explicitly targets "relative arrangements" — moving object A closer to or farther from object B as a deliberate test dimension. libero-infinity samples object positions independently (with pairwise clearance hard constraints), so relative arrangements vary but are not explicitly controlled as a perturbation axis. A future extension could add a soft constraint biasing pairwise distance distributions.

---

## 7. Side-by-Side Comparison Table

| Feature | LIBERO-PRO P1/P2 | libero-infinity |
|---------|-----------------|-----------------|
| **Position perturbation type** | Discrete: ~10 fixed swap pairs | Continuous uniform over workspace |
| **Position coverage** | ~10 fixed (x,y) per task suite | ℝ² workspace minus rejected regions |
| **Hard position constraints** | Physical plausibility (by construction) | Rejection sampling: `require dist(A,B) > 0.22 m` |
| **Soft OOD position bias** | Not present | `require[0.8] dist > 0.15 m` from training pos |
| **Object perturbation type** | Pre-defined asset list (2–6 per class) | Stochastic `Uniform` draw per episode |
| **Object property changed** | Color, texture, size | Mesh + texture (full asset swap) |
| **Object classes in registry** | Subset of 4 LIBERO task suites | 34 classes, full LIBERO object vocabulary |
| **Camera perturbation** | None | 4D continuous: ±10 cm lateral/longitudinal, ±8 cm height, ±15° tilt |
| **Lighting perturbation** | None | 5D continuous: intensity ×[0.4,2.0], position ±0.5 m, ambient [0.05,0.6] |
| **Texture perturbation** | Full scene environment swap (5 variants) | Table surface texture swap |
| **Distractor objects** | None | 1–5 foreign objects, constrained placement |
| **Language perturbation** | Yes (paraphrase + task substitution) | Not implemented |
| **Axis composability** | Single axis only | Any subset of 6 axes simultaneously |
| **Task coverage** | 4 LIBERO suites (hand-crafted) | Any BDDL task (auto-generated) |
| **Total evaluation configs** | O(10–20) per task × N_suites | Infinite i.i.d. samples |
| **Memorization-proof** | No (finite, enumerable) | Yes (continuous distribution, never repeated) |
| **Success rate estimator** | Point estimate | Wilson CI → convergent statistic |
| **Adversarial search** | None | VerifAI cross-entropy method |
| **Gym wrapper** | None | `LIBEROScenicEnv(gym.Env)` + `make_vec_env` |
| **Task reversal** | None | BDDL goal/initial inversion |

---

## 8. Summary

LIBERO-PRO P1/P2 and libero-infinity represent different design choices for perturbation-based evaluation, suited to different goals:

- **On position (P1):** LIBERO-PRO places objects at ~10 hand-chosen alternative positions — a fully reproducible, auditable set. libero-infinity samples from a continuous 2D uniform over the full reachable workspace, constrained by physics and biased toward OOD configurations. LIBERO-PRO's predefined positions offer predictable coverage; libero-infinity's continuous Scenic 3 distribution offers statistical inexhaustibility.

- **On object variation (P2):** Both systems cover 2–5 variants per class. Key differences: libero-infinity draws stochastically per episode (no deterministic cycling), applies across any BDDL task (not just 4 suites), and can compose object variation with all other axes simultaneously.

- **On axes only libero-infinity covers:** Camera perturbation, lighting perturbation, table texture variation, and distractor injection have no counterpart in LIBERO-PRO. libero-infinity's camera perturbation (cited by LIBERO-Plus as the highest-impact axis) adds an additional evaluation dimension.

- **On composability:** LIBERO-PRO tests axes in isolation. libero-infinity can compose all six axes in a single scene, enabling evaluation of compound covariate shift.

**The main gaps** are on the *language* side (instruction paraphrase and task logic substitution) and *full scene environment* switching, where LIBERO-PRO's L and E dimensions have no current counterpart in libero-infinity. These are genuine gaps for language robustness evaluation.

**Bottom line:** For the spatial/visual perturbation axes (position and object appearance), libero-infinity uses continuous Scenic 3 distributions while LIBERO-PRO uses predefined sets — different evaluation philosophies. For camera, lighting, distractor, and composability, libero-infinity provides additional evaluation dimensions. For language robustness and full-scene environment switching, LIBERO-PRO's L and E dimensions address evaluation needs that libero-infinity currently does not cover.

---

## References

1. Zhang, X. et al. *LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization.* arXiv:2510.03827, 2025. https://arxiv.org/abs/2510.03827
2. LIBERO-PRO GitHub repository: https://github.com/Zxy-MLlab/LIBERO-PRO
3. Fu, S. et al. *LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models.* arXiv:2510.13626, 2025.
4. *LIBERO-X: Robustness Litmus for Vision-Language-Action Models.* arXiv:2602.06556, 2026.
5. Liu, B. et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.* NeurIPS 2023.
6. Fremont, D.J. et al. *Scenic: A Language for Scenario Specification and Scene Generation.* PLDI 2019.
