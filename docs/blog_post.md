# LIBERO-Infinity: When Robotics Benchmarks Meet Probabilistic Programs

## TL;DR

Current robotics benchmarks evaluate policies on fixed test sets that clever models can memorize. **LIBERO-Infinity** brings Scenic 3 — UC Berkeley's probabilistic programming language for autonomous systems — to robot manipulation evaluation, offering genuine probability distributions instead of fixed perturbation checklists. The result: an open-ended evaluation framework that can generate *infinite* variations of any LIBERO manipulation task, turning success rate from a checklist metric into a population statistic. To our knowledge, this is the first system to integrate a probabilistic programming language with a robotics manipulation benchmark.

---

## The Problem: Benchmarks You Can Memorize

Robot learning has a generalization problem, and our benchmarks are partly to blame.

Consider a typical evaluation pipeline: train a vision-language-action (VLA) model on demonstrations, then test it on a held-out set of scenes. If the test set is finite and fixed -- say, 10-20 pre-configured initial states per task -- a sufficiently expressive model can simply memorize the evaluation distribution. High benchmark scores then reflect rote recall, not genuine task understanding.

This is not hypothetical. [LIBERO-PRO](https://arxiv.org/html/2510.03827v1) demonstrated exactly this: models achieving **>90% accuracy** on standard LIBERO evaluation **collapsed to 0%** under systematic perturbations to object appearance, position, and instructions. The models had memorized the training environments, not learned manipulation.

The [COLOSSEUM benchmark](https://arxiv.org/abs/2402.08191) (NVIDIA, 2024) took a step further, defining 14 perturbation axes across 20 tasks and showing 30-75% performance degradation under perturbation. But even COLOSSEUM uses scripted, enumerated variations -- a fixed set that a future model could, in principle, overfit.

The core issue: **finite test sets are fundamentally memorizable**. What we need are *distributions* over test scenarios -- distributions that can be sampled indefinitely, that encode physical constraints formally, and that make success rate converge to a genuine population statistic as sample count grows.

---

## Background

### LIBERO: The Foundation

[LIBERO](https://arxiv.org/abs/2306.03310) (NeurIPS 2023, Datasets & Benchmarks track) is a benchmark of **language-conditioned manipulation tasks** built on robosuite and MuJoCo. Tasks are grouped into suites that test different knowledge transfer dimensions:

- **LIBERO-Spatial**: Same objects, different spatial relations
- **LIBERO-Object**: Different objects, same procedural context
- **LIBERO-Goal**: Same objects, different goals
- **LIBERO-100/Long**: Entangled tasks mixing all dimensions

Each task is defined by a BDDL (Behavior Description Definition Language) file specifying objects, their initial placement regions, and goal predicates. LIBERO provides high-quality human-teleoperated demonstrations and has become a standard testbed for VLA models like RT-1-X, pi-0, and SpatialVLA.

### LIBERO-PRO: Exposing the Memorization Gap

[LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) extends LIBERO with systematic perturbations -- swapping object meshes, shifting initial positions, paraphrasing instructions, and changing environments. Its key contribution is proving that current VLA models rely on memorized action sequences rather than genuine perception.

LIBERO-PRO's perturbations use a predefined set: ~10 position swap pairs, 2-6 object replacements per class, 3 instruction paraphrases. Each axis is tested in isolation.

### Scenic: Probabilistic Programming for Cyber-Physical Systems

[Scenic](https://github.com/BerkeleyLearnVerify/Scenic) is a probabilistic programming language developed at UC Berkeley for modeling the environments of cyber-physical systems. Originally designed for autonomous driving (with deep integration into CARLA and other simulators), Scenic allows users to write programs that define *distributions over scenes* rather than single scenes.

A Scenic program composes three elements:

1. **Distributions** -- `Range(lo, hi)`, `Uniform(a, b, c)`, `DiscreteRange(1, 5)` -- that define what varies
2. **Hard constraints** -- `require expr` -- that Scenic's rejection sampler must satisfy
3. **Soft constraints** -- `require[p] expr` -- that bias the distribution without forbidding regions

Scenic has been used by Boeing, Meta, Toyota, and Deutsche Bahn for testing autonomous systems. Its companion tool [VerifAI](https://github.com/BerkeleyLearnVerify/VerifAI) provides falsification-guided adversarial search over the Scenic distribution.

What makes Scenic compelling for benchmarks is that it gives you a *formal, declarative specification of the test distribution* -- not a bag of scripts, but a constraint-checked probabilistic program that is correct by construction.

---

## The Contribution: Blending Benchmarks with Probabilistic Programs

LIBERO-Infinity is a new evaluation framework built by combining Scenic 3 (UC Berkeley probabilistic programming) with LIBERO's simulation environment and task suite. Scenic 3 programs define genuine probability distributions over scene configurations — something that didn't exist before in robot manipulation evaluation. Each sample is drawn i.i.d., enforced by constraint solving, and never repeated.

| Dimension | LIBERO-PRO | LIBERO-Infinity |
|-----------|-----------|-----------------|
| **Evaluation set** | Finite (~10-20 states/task) | Open-ended: infinite i.i.d. samples |
| **Position** | 10 swap pairs over a discrete grid | Continuous uniform over workspace |
| **Object identity** | 2-6 pre-defined replacements | 34 object classes with variant pools, sampled uniformly |
| **Camera / Lighting** | Fixed | Range-based perturbation per scene |
| **Composability** | Each axis in isolation | Any subset via `--perturbation a,b,c` |
| **Adversarial search** | Not supported | Cross-entropy Bayesian optimization |
| **Gym API** | None | Standard `gym.Env` + `make_vec_env` |
| **Task coverage** | Subset of LIBERO | Any LIBERO BDDL task via auto-gen |

The key insight is that Scenic's constraint solver does exactly what a good benchmark needs: it defines *what is physically plausible* (hard constraints on clearance, reachability, collision avoidance) and *what is interesting* (soft constraints biasing toward out-of-distribution configurations), then samples from the resulting distribution.

---

## How It Works: A Layered Scenic Architecture

LIBERO-Infinity is organized into three layers, each building on the one below:

```
Layer 3 — Perturbation programs  (scenic/*.scenic)
  Express WHAT varies and WHAT constraints must hold.
      │  model libero_model
      ▼
Layer 2 — World model  (scenic/libero_model.scenic)
  LIBEROObject/LIBEROFixture classes, table geometry,
  asset variant registry.
      │  Python driver
      ▼
Layer 1 — Simulator bridge  (src/libero_infinity/simulator.py)
  Scenic ↔ MuJoCo bridge. Injects sampled poses,
  applies camera/lighting/texture perturbations.
```

### Layer 2: The World Model

The `libero_model.scenic` file defines a shared vocabulary for all perturbation programs:

```scenic
# Table workspace constants (from LIBERO arena XML)
TABLE_Z      = 0.82      # table surface height (metres)
TABLE_X_MIN  = -0.40
TABLE_X_MAX  =  0.40
TABLE_Y_MIN  = -0.30
TABLE_Y_MAX  =  0.30

# Scenic object classes
class LIBEROObject(Object):
    libero_name:     ""       # BDDL instance name
    asset_class:     ""       # Object type (for mesh swapping)
    graspable:       True
    allowCollisions: True     # Clearance via explicit constraints, not FCL
    width:           0.08
    length:          0.08
    height:          0.06
```

Asset variants (34 object classes, 39 bounding box entries) are loaded from a single JSON source of truth shared between the Scenic and Python layers.

### Layer 3: Perturbation Programs

Here is what a position perturbation looks like in Scenic -- the full program for randomizing object placement:

```scenic
model libero_model

param min_clearance = 0.12
param ood_margin    = 0.15

# Objects placed uniformly on the workspace
bowl = new LIBEROObject with libero_name "akita_black_bowl_1",
                         with asset_class "akita_black_bowl",
                         at Vector(Range(TABLE_X_MIN, TABLE_X_MAX),
                                   Range(TABLE_Y_MIN, TABLE_Y_MAX),
                                   TABLE_Z)

plate = new LIBEROObject with libero_name "plate_1",
                          with asset_class "plate",
                          at Vector(Range(TABLE_X_MIN, TABLE_X_MAX),
                                    Range(TABLE_Y_MIN, TABLE_Y_MAX),
                                    TABLE_Z)

# Hard constraints: physical plausibility
require (distance from bowl to plate) > _min_clearance

# Soft constraints: prefer out-of-distribution positions
require[0.8] distance from bowl to bowl_train_pt > _ood_margin
```

LIBERO-PRO uses a YAML file listing 10 specific (x, y) swap pairs — a useful predefined set. LIBERO-Infinity instead uses a continuous distribution over the full workspace, constrained by physics and biased toward novel configurations: a different tool for different evaluation goals.

### Layer 1: The Simulator Bridge

The `LIBEROSimulation` class implements Scenic's simulator interface to bridge sampled scenes into MuJoCo:

1. **`setup()`**: Creates a LIBERO environment, calls `env.reset()` to load the BDDL scene, then *overrides* each object's joint position with the Scenic-sampled coordinates. Applies camera, lighting, and texture perturbations to the MuJoCo model. Runs 50 settling steps so objects rest naturally.

2. **`step_with_action()`**: Passes real policy actions through to MuJoCo for evaluation.

3. **`getProperties()`**: Reads back live MuJoCo state for Scenic's temporal monitors.

### Auto-Generation: Any Task, Zero Setup

You don't need hand-written Scenic programs. The `compiler.py` module parses any BDDL file and emits a valid `.scenic` program with the requested perturbation axes:

```bash
# Evaluate any LIBERO BDDL task -- Scenic program auto-generated
libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/any_task.bddl \
  --perturbation full --n-scenes 200 --verbose
```

The generator handles stacking dependencies (e.g., a bowl stacked on a plate inherits its parent's perturbed position), fixture detection, and adaptive distractor slot counts.

---

## Six Composable Perturbation Axes

LIBERO-Infinity defines six perturbation axes, each as a Scenic distribution:

| Axis | What Varies | Scenic Distribution |
|------|-------------|---------------------|
| **Position** | Object (x, y) placement | `Range` over reachable workspace with clearance constraints |
| **Object** | Visual identity (mesh + texture) | `Uniform` over OOD asset variants per class |
| **Camera** | Viewpoint position and angle | Range-based offsets + tilt on agentview camera |
| **Lighting** | Scene illumination | Intensity multiplier, position offsets, ambient level |
| **Texture** | Table surface material | Material swap on table geometry |
| **Distractor** | Scene clutter | 1-5 non-task objects via BDDL rewriting with spatial constraints |

The key feature is **arbitrary composability**. Any subset of axes can be activated with a comma-separated flag:

```bash
--perturbation position,camera              # two axes
--perturbation object,lighting,distractor   # three axes
--perturbation full                         # all axes
```

This means evaluation can be tailored: test position robustness alone, combine visual perturbations, or stress-test with everything at once.

### Adversarial Search

Beyond i.i.d. sampling, LIBERO-Infinity integrates with VerifAI's cross-entropy method for **adversarial falsification**. Instead of uniform sampling, the cross-entropy sampler concentrates on failure-inducing regions of the distribution:

```scenic
# VerifaiRange enables cross-entropy concentration
bowl = new LIBEROObject at Vector(
    VerifaiRange(TABLE_X_MIN + 0.05, TABLE_X_MAX - 0.05),
    VerifaiRange(TABLE_Y_MIN + 0.05, TABLE_Y_MAX - 0.05),
    TABLE_Z)
```

After each episode, feedback (success/failure) is passed back to the sampler, which adaptively narrows the distribution toward worst-case configurations. This finds the specific object arrangements that break a policy -- invaluable for debugging and hardening.

---

## What This Enables

### From Checklists to Statistics

With finite test sets, success rate is a point estimate over a known population. With LIBERO-Infinity, success rate becomes a **population statistic with a confidence interval**:

```
Success rate: 73.5% +/- 6.1% (147/200 scenes)
```

The 95% Wilson confidence interval narrows as you sample more scenes. There is no ceiling -- you can always draw more samples for tighter estimates. This is fundamentally different from "passed 18/20 pre-defined tests."

### Domain-Randomized Training via Gym API

LIBERO-Infinity isn't just for evaluation -- it provides a standard `gym.Env` wrapper for training under domain randomization:

```python
from libero_infinity.gym_env import LIBEROScenicEnv, make_vec_env

# Single environment with full perturbation
env = LIBEROScenicEnv(
    bddl_path="path/to/task.bddl",
    perturbation="full",
    resolution=256,
)

# Parallel rollouts for RL training
vec_env = make_vec_env("path/to/task.bddl", n_envs=8, perturbation="full")
```

Every `reset()` samples a fresh scene from the Scenic distribution -- new object positions, new visual appearances, new camera angles. This is principled domain randomization backed by formal constraints rather than ad-hoc random ranges scattered across training scripts.

### Task Reversal

LIBERO-Infinity can **reverse any task** -- the goal becomes the initial state and vice versa. "Put the bowl on the plate" becomes "take the bowl off the plate." This doubles the evaluation surface without new task authoring, and tests whether models understand task semantics or just replay trajectories.

---

## Is This Really the First?

We claim LIBERO-Infinity is the first to blend a robotics manipulation benchmark with a probabilistic programming language. Let's examine the landscape:

- **LIBERO-PRO** (2025): Predefined perturbation sets. No formal distribution specification.
- **THE COLOSSEUM** (NVIDIA, 2024): 14 perturbation axes, but scripted via Python functions -- not a probabilistic programming language. Perturbations are enumerated, not sampled from constraint-checked distributions.
- **Domain randomization** (Tobin et al., 2017 and descendants): Random ranges in training code, but no formal constraint language. No composable specification. No rejection sampling for physical plausibility.
- **Scenic applications**: Extensive use in autonomous driving (CARLA), aviation, and maritime. A [Mars rover example](https://scenic-lang.org/cvpr24/) exists in the Scenic documentation. But no prior integration with a tabletop manipulation benchmark like LIBERO.

The gap we fill: **formal, constraint-checked, probabilistically-specified evaluation distributions for robotic manipulation**. Scenic provides the language; LIBERO provides the tasks; LIBERO-Infinity provides the bridge.

---

## Principled Constraint Calibration via Autonomous Multi-Agent Teams

### The Problem: Thresholds Set Without Data

LIBERO-Infinity's Scenic programs use `require` constraints to enforce physical plausibility: minimum clearance between objects, inset margins from workspace edges, clearance around distractors. These thresholds were originally set by hand — reasonable-looking numbers with no empirical grounding. The original values were `min_clearance = 0.10`, `workspace_margin = 0.05`, `distractor_clearance = 0.08`.

The danger of uncalibrated thresholds is subtle but serious. After a Scenic sample is injected into MuJoCo, objects don't stay exactly where Scenic placed them. Rigid bodies get nudged by contact forces, gravity, and the constraint solver during the settling phase. Near workspace edges, this drift can push objects off the table entirely — a hard failure that crashes the episode. With `workspace_margin = 0.05`, our adversarial scene selector was consistently placing objects just inside the nominal boundary, and the physics would knock them over the edge: **12.5% hard failure rate** at baseline, with maximum XY drift reaching 0.83 m in the worst case. These failures are silent — they don't raise exceptions in normal evaluation loops; they just corrupt the training data.

Without principled thresholds, we had no guarantee that any given Scenic sample would produce a valid, stable scene. And with the full LIBERO task suite, each with multiple perturbation axes, manually tuning thresholds by trial-and-error was not tractable.

### How We Used OMAR to Calibrate

We used [OMAR](https://github.com/lsk567/omar) (One-Man Army), our multi-agent orchestration system, to run the calibration autonomously. The EA dispatched a dedicated `drift-calib` agent with a full implementation spec. The agent worked without human intervention from start to finish:

1. Read `compiler.py` and identified all threshold constants and their roles.
2. Built an adversarial scene selector targeting worst-case placements — edge-adjacent positions, large objects, densely packed distractor configurations.
3. Implemented a parallel calibration loop using Python `multiprocessing` with 3 workers, evaluating 8 scenes per binary-search step across all perturbation axes.
4. Ran real MuJoCo physics simulations (not stubs — `mujoco_available: true`, `stub_mode: false`) and measured per-object XY drift after settling.
5. Binary-searched each parameter independently to find the minimum safe value: the smallest threshold that drives hard failures to zero while keeping 95th-percentile drift within a 5 cm budget.
6. Committed both `scripts/calibrate_drift.py` and `calibration_results.json` to the repository.

Total wall time: approximately 34 minutes. Zero human interventions.

### What the Calibration Found

| Parameter | Old Value | Recommended | Hard Failure Rate (before → after) | 95th-pct Drift (before → after) |
|---|---|---|---|---|
| `workspace_margin` | 0.05 m | **0.11 m** | **12.5% → 0.0%** | 12.0 mm → 8.9 mm |
| `min_clearance` | 0.10 m | 0.096 m | 0.0% → 0.0% | 15.4 mm → 10.3 mm |
| `distractor_clearance` | 0.08 m | 0.034 m | 0.0% → 0.0% | 21.4 mm → 21.4 mm |

`workspace_margin` was the critical fix. At 0.05 m, objects placed near the nominal workspace boundary had a 1-in-8 chance of falling off the table after MuJoCo settling — the contact solver would give them a final nudge over the edge. The binary search converged at 0.1114 m (rounded to 0.11 m); below 0.1097 m, hard failures reappeared reliably. `min_clearance` and `distractor_clearance` were already conservative enough — the calibration confirmed they could actually be tightened, though we leave them unchanged for now to preserve the existing margin of safety.

The calibration ran 64 scenes to characterize `workspace_margin` (6 binary-search steps × 8 scenes + baseline + final validation), using adversarial placement: objects pushed toward table edges, maximum-footprint objects, and 5-distractor configurations. The 3-worker parallelism kept per-step latency under 30 seconds even with real physics settling.

### Why This Matters

Correct thresholds give us three things:

**Zero hard failures.** With `workspace_margin = 0.11`, every Scenic sample that passes constraint solving will also produce a stable, settled MuJoCo scene. Training data is clean by construction, not by luck.

**Reproducible validity guarantees.** The `require` constraints in generated Scenic programs now have a direct empirical backing: we know the margin was chosen to eliminate physics failures on adversarial configurations. When a sample passes rejection sampling, it reflects a physically realizable scene.

**A template for future calibration.** As we add new object classes, new fixture geometries, or new perturbation axes, `calibrate_drift.py` gives us a repeatable methodology. Run it, read the JSON, update the constant — an afternoon of compute, not weeks of manual tuning.

More broadly, this demonstrates what OMAR is for. We gave the system a loosely-specified engineering task — "calibrate these thresholds" — with no implementation details, and it returned production-ready artifacts: a working calibration script, a results file with full search history, and a committed fix. The agent made real architectural decisions (adversarial vs. random scene selection, binary search vs. grid search, parallel vs. serial evaluation) and executed them correctly against a live physics simulator. That's the kind of autonomous engineering leverage that makes a small team move at scale.

---

## Built with OMAR

LIBERO-Infinity was developed using [OMAR](https://github.com/lsk567/omar) (One-Man Army), a TUI dashboard for managing AI coding agents built on tmux. OMAR enables a single developer to orchestrate multiple AI agents working in parallel -- one exploring the LIBERO codebase, another writing Scenic programs, another building the simulator bridge, another writing tests.

The project's layered architecture (world model, perturbation programs, simulator bridge, evaluation harness, gym wrapper, task reversal, documentation, 90+ tests) was developed by a small team amplified by OMAR's multi-agent workflow. When building systems that span multiple frameworks (Scenic's probabilistic programming runtime, MuJoCo physics, LIBERO's environment stack, gym's API), the ability to have specialized agents working in parallel on independent modules -- each with full context on their piece -- is a force multiplier.

---

## What's Next

LIBERO-Infinity opens several directions:

- **Behavioral perturbations**: Scenic 3 supports temporal behaviors (`do action for N steps`). Future work could define distributions over *dynamic* perturbations -- objects that move during execution, lighting that changes mid-episode, distractor objects that enter the scene.

- **Scenic-guided curriculum learning**: Use the soft constraint mechanism to gradually shift the training distribution from easy (near-canonical) to hard (far-OOD) configurations, with the curriculum defined declaratively in Scenic.

- **Multi-task compositional evaluation**: Scenic's `compose` blocks allow chaining sub-scenarios. This could enable evaluation of multi-step tasks where each phase has its own perturbation distribution.

- **Real-world transfer calibration**: Pair Scenic-sampled sim perturbations with real-world measurements to calibrate the sim-to-real gap for specific perturbation axes, following the COLOSSEUM's approach of correlating sim and real perturbation sensitivity.

- **Community-contributed perturbation programs**: The `.scenic` file format is human-readable and version-controllable. Researchers can contribute new perturbation programs (new constraint sets, new distribution families, new object pools) as standalone files.

---

## Get Started

```bash
git clone https://github.com/KE7/libero-infinity.git && cd libero-infinity
make install-full   # creates venv, installs deps, bootstraps HF assets
make test           # 90 tests, headless

# Run evaluation with full perturbation on any task
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl \
  --perturbation full --n-scenes 200 --verbose
```

---

## References

1. Liu, B., Zhu, Y., et al. [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310). NeurIPS 2023, Datasets & Benchmarks.

2. Zhang, X., et al. [LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization](https://arxiv.org/html/2510.03827v1). 2025.

3. Fremont, D.J., Dreossi, T., et al. [Scenic: A Language for Scenario Specification and Scene Generation](https://arxiv.org/abs/1809.09310). PLDI 2019.

4. Fremont, D.J., et al. [Scenic: A Language for Scenario Specification and Data Generation](https://link.springer.com/article/10.1007/s10994-021-06120-5). Machine Learning, 2022.

5. Pumacay, I., et al. [THE COLOSSEUM: A Benchmark for Evaluating Generalization for Robotic Manipulation](https://arxiv.org/abs/2402.08191). 2024.

6. Scenic 3.0 and VerifAI: [github.com/BerkeleyLearnVerify/Scenic](https://github.com/BerkeleyLearnVerify/Scenic)

7. OMAR: [github.com/lsk567/omar](https://github.com/lsk567/omar) -- TUI dashboard for multi-agent development.
