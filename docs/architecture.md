# Architecture

[Back to main README](../README.md)

## System Overview

<p align="center">
  <img src="../assets/architecture_pipeline.png" width="85%" alt="LIBERO-Infinity Architecture">
</p>

### Detailed Flow

```
                          ┌─────────────────────┐
                          │   BDDL task file     │
                          │  (any BDDL task)     │
                          └────────┬────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
             ┌────────────┐ ┌──────────┐  ┌────────────────┐
             │  --reverse │ │ TaskConfig│  │  BDDL          │
             │  (optional)│ │  .from_   │  │  Preprocessor  │
             │  task_     │ │  bddl()  │  │  (asset sub,   │
             │  reverser  │ │          │  │   distractors) │
             └─────┬──────┘ └────┬─────┘  └───────┬────────┘
                   │             │                │
                   │             ▼                │
                   │      ┌────────────┐          │
                   │      │  Scenic    │          │
                   │      │  Generator │          │
                   │      └─────┬──────┘          │
                   │            │                 │
                   ▼            ▼                 │
            ┌───────────────────────────────┐     │
            │  .scenic program (generated   │     │
            │  or hand-written)             │     │
            └──────────────┬────────────────┘     │
                           │ model libero_model   │
                           ▼                      │
            ┌──────────────────────────────┐      │
            │  Scenic 3 constraint solver  │      │
            │  (rejection sampling)        │      │
            └──────────────┬───────────────┘      │
                           │ scene.objects,        │
                           │ scene.params           │
                           ▼                      ▼
            ┌─────────────────────────────────────────────┐
            │         Evaluation Harness (eval.py)         │
            │                                              │
            │  For each scene:                             │
            │    1. Resolve BDDL (asset sub / reversal)    │
            │    2. Create LIBEROSimulation                │
            │    3. setup() → inject poses + perturbations │
            │    4. step_with_action() × N (policy loop)   │
            │    5. check_success() → EpisodeResult        │
            │                                              │
            │  Modes: standard (i.i.d.) | adversarial (CE) │
            └──────────────────┬──────────────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │  LIBERO / MuJoCo / robosuite         │
            │  OffScreenRenderEnv → physics + obs  │
            └──────────────────────────────────────┘
```

## Layered Scenic architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  Layer 3  — Scenic perturbation programs  (scenic/*.scenic)       │
│                                                                   │
│  Hand-written:                                                    │
│    position_perturbation.scenic   — spatial distribution          │
│    object_perturbation.scenic     — asset identity distribution   │
│    combined_perturbation.scenic   — joint position + identity     │
│    camera_perturbation.scenic     — viewpoint perturbation        │
│    lighting_perturbation.scenic   — illumination perturbation     │
│    verifai_position.scenic        — adversarial CE sampling       │
│                                                                   │
│  Auto-generated from any BDDL (written to scenic/generated/):    │
│    compiler.py → _gen_<task>_<mode>.scenic      │
│                                                                   │
│  Express WHAT varies and WHAT constraints must hold.              │
│  Scenic's rejection sampler produces valid samples.               │
└────────────────────┬──────────────────────────────────────────────┘
                     │  model libero_model
                     ▼
┌───────────────────────────────────────────────────────────────────┐
│  Layer 2  — World model  (scenic/libero_model.scenic)             │
│                                                                   │
│  LIBEROObject class, LIBEROFixture class                          │
│  Table geometry constants (TABLE_Z, TABLE_X_MIN, ...)             │
│  ASSET_VARIANTS + OBJECT_DIMENSIONS loaded from JSON              │
└────────────────────┬──────────────────────────────────────────────┘
                     │  Python driver
                     ▼
┌───────────────────────────────────────────────────────────────────┐
│  Layer 1  — LIBERO simulator bridge  (simulator.py)               │
│                                                                   │
│  LIBEROSimulator  (subclasses scenic.core.simulators.Simulator)   │
│  LIBEROSimulation (subclasses scenic.core.simulators.Simulation)  │
│                                                                   │
│  Injects sampled poses via set_joint_qpos.                        │
│  Applies camera/lighting/texture perturbation via MuJoCo model.   │
│  Provides step_with_action() for policy-driven evaluation.        │
└───────────────────────────────────────────────────────────────────┘
```

## Multi-task support

LIBERO-Infinity works with **any** LIBERO BDDL task file without hand-written
Scenic programs. The pipeline:

1. `task_config.py` parses the BDDL to extract movable objects, fixtures, regions, and initial positions
2. `compiler.py` emits a valid `.scenic` program with the requested perturbation axes
3. The generated program is compiled and sampled like any hand-written one

```bash
# No --scenic needed — auto-generates from BDDL
libero-eval --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/any_task.bddl \
  --perturbation full --n-scenes 100 --verbose
```

## Key design decisions

### `allowCollisions: True` on all LIBEROObjects

Scenic's default collision checking uses FCL, which requires the full FCL C library
on ARM64. Instead, clearance is enforced by explicit `require distance from A to B > threshold`
constraints, which directly correspond to the physics the simulator enforces.

### `at Vector(Range(...), Range(...), Z)` not `in BoxRegion`

`BoxRegion` triggers Scenic's FCL-based containment check. Using `Range` distributions
directly in the `at` specifier achieves identical uniform sampling without FCL.

### Extract `globalParameters` to locals before `require`

Scenic evaluates `require` expressions as deferred closures. Accessing
`globalParameters.my_param` inside a closure can fail at runtime. Capture values first:

```scenic
_val = globalParameters.my_param   # captured at compile time
require some_expr > _val           # closure sees a normal Python variable
```

### `LIBEROSimulation` does not call `super().__init__()`

Scenic's `Simulation.__init__` runs the entire simulation loop eagerly and
re-evaluates all `require[p]` statements at each physics step. This converts
soft sampling constraints (e.g. 80% OOD bias) into hard runtime rejects. We
manually initialise minimal Scenic state and use a lazy lifecycle instead.

### Single JSON source of truth for asset variants

`src/libero_infinity/data/asset_variants.json` is loaded by both `asset_registry.py`
(Python side) and `libero_model.scenic` (Scenic side). No duplication.

### MuJoCo body naming: `{instance}_main`

LIBERO/robosuite appends `_main` to BDDL instance names. `_inject_object_pose()`
and `getProperties()` try both names automatically.

### Object centre vs table surface (z injection)

Scenic places object centres at `TABLE_Z` (table surface). MuJoCo free-joint
`qpos` locates the centre of mass, which must be above the surface by `height/2`.
`_inject_object_pose()` applies this correction automatically.

### BDDL patching vs. XML patching

Object substitution is done at the BDDL text level rather than by patching MuJoCo
XML directly. LIBERO's env constructor re-parses the BDDL and loads the correct
XML mesh automatically.

## File map

```
libero-infinity/
│
├── pyproject.toml
├── vendor/
│   └── python_fcl-0.7.0.10-...-none-any.whl   ARM64 FCL stub
│
├── vendor/libero/                       Vendored LIBERO runtime (KE7/Libero fork)
│   └── libero/libero/
│       ├── bddl_files/                  Task BDDL files (full LIBERO task suite)
│       └── envs/                        MuJoCo environments + wrappers
│
├── scenic/
│   ├── libero_model.scenic              Layer 2: world vocabulary
│   │                                   LIBEROObject / LIBEROFixture classes
│   │                                   Table geometry constants
│   │                                   ASSET_VARIANTS + OBJECT_DIMENSIONS (from JSON)
│   │
│   ├── position_perturbation.scenic     Spatial distribution (uniform workspace)
│   ├── object_perturbation.scenic       Asset identity distribution
│   ├── combined_perturbation.scenic     Joint position + identity
│   ├── camera_perturbation.scenic       Viewpoint perturbation
│   ├── lighting_perturbation.scenic     Illumination perturbation
│   ├── verifai_position.scenic          Adversarial CE sampling variant
│   ├── distractor_perturbation.scenic   Random clutter objects (1-N)
│   └── generated/                       Auto-generated programs (gitignored)
│       └── _gen_*.scenic                One per task × perturbation axis
│
├── scripts/
│   └── generate_reversed_bddls.py      Batch BDDL reversal for a directory
│
├── src/libero_infinity/
│   ├── __init__.py
│   ├── simulator.py                     Layer 1: Scenic <-> MuJoCo bridge
│   │                                   LIBEROSimulator / LIBEROSimulation
│   │                                   Position injection + env perturbations
│   │
│   ├── eval.py                          Evaluation harness + CLI
│   │                                   Standard (i.i.d.) + adversarial (CE) modes
│   │                                   BDDL-only auto-generation + --reverse support
│   │
│   ├── gym_env.py                       Gym wrapper for RL/VLA training
│   │                                   LIBEROScenicEnv(gym.Env) + make_vec_env
│   │
│   ├── task_config.py                   BDDL parser -> TaskConfig
│   │                                   Extracts objects, fixtures, regions, positions
│   │                                   Handles stacking deps (stacked_on field)
│   │
│   ├── compiler.py                        Dynamic .scenic program generation
│   │                                   7 perturbation modes, composable
│   │                                   Relative positioning for stacked objects
│   │
│   ├── task_reverser.py                 BDDL task reversal: forward -> backward
│   │                                   On/In/Open/Close/Turnon/Turnoff rules
│   │                                   Language construction from goal predicates
│   │
│   ├── bddl_preprocessor.py             Shared BDDL parsing utilities
│   │                                   Asset substitution + distractor injection
│   │                                   _find_closing_paren, _extract_block,
│   │                                   _parse_language, _parse_declarations
│   │
│   ├── asset_registry.py                OOD variant registry (loads from JSON)
│   │
│   └── data/
│       ├── __init__.py
│       └── asset_variants.json          Single source of truth: 34 object classes
│                                        with variant pools, 35 dimension entries
│
├── docs/
│   ├── scenic_perturbations.md          All 6 perturbation axes in detail
│   ├── evaluation_pipeline.md           Eval harness, CLI, adversarial mode
│   ├── architecture.md                  System diagrams, design decisions, file map
│   ├── api-reference.md                 Python API for all modules
│   ├── gym-wrapper.md                   Gym env for RL/VLA training loops
│   ├── task-reversal.md                 Backward evaluation scenarios
│   ├── installation.md                  Detailed setup, platform support
│   ├── observations-actions.md          Obs/action schema, policy interface
│   ├── contributing.md                  Contribution guidelines
│   └── internal/                        Development artifacts
│
└── tests/
    ├── conftest.py                      Shared fixtures, helpers, skip markers
    ├── test_scenic.py                   Tier 1: Scenic-only tests (no LIBERO)
    ├── test_libero.py                   Tier 2: LIBERO simulation integration
    ├── test_gym.py                      Tier 3: Gym wrapper tests
    └── test_e2e.py                      Backward-compat shim (re-exports above)
```
