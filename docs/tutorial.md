# LIBERO-Infinity: New-User Tutorial

[Back to main README](../README.md)

> **What you will build:** By the end of this tutorial you will have run a
> 10-episode evaluation of a VLA policy under **position + camera perturbation**,
> produced a JSON results file, and know how to extend the evaluation to any of
> LIBERO's BDDL tasks using the CLI or the Python Gym API.

---

## Table of Contents

1. [What you will build](#1-what-you-will-build)
2. [Prerequisites](#2-prerequisites)
3. [Your first evaluation (CLI)](#3-your-first-evaluation-cli)
4. [Adding more perturbation axes](#4-adding-more-perturbation-axes)
5. [Using the Gym wrapper (Python API)](#5-using-the-gym-wrapper-python-api)
6. [Evaluating your own VLA](#6-evaluating-your-own-vla)
7. [Understanding the results](#7-understanding-the-results)
8. [Next steps](#8-next-steps)

---

## 1. What you will build

LIBERO-Infinity wraps any LIBERO task in a Scenic 3 probabilistic program that
generates an unlimited stream of i.i.d. test scenes. Each call to `reset()`
draws a fresh scene: different object positions, different object meshes,
different camera angle — whatever axes you activate.

By the end of this tutorial you will:

- **Pick a task** from the included BDDL library (we'll use *"Put the bowl on
  the plate"*)
- **Run a 10-episode evaluation** with `--perturbation position` from the CLI
  and save a JSON results file
- **Switch to combined perturbation** (position + object identity) and see how
  the scene distribution changes
- **Write a short Python script** that uses `LIBEROScenicEnv` to drive a
  custom policy in a training loop
- **Understand what the numbers mean** — success rate, Wilson confidence
  intervals, and what the BDDL goal checker actually checks

---

## 2. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.11+ | Required by Scenic 3 |
| **uv** | latest | Fast package manager — replaces pip+venv |
| **MuJoCo** | 2.x | Installed automatically by `uv sync` |
| **EGL or OSMesa** | — | Headless OpenGL for MuJoCo rendering |
| **GPU (recommended)** | — | For VLA inference; simulation itself is CPU-only |

> **macOS users:** MuJoCo renders natively on Apple Silicon — omit the
> `MUJOCO_GL=egl` prefix shown throughout this tutorial.

If you haven't installed LIBERO-Infinity yet, follow
[docs/installation.md](installation.md) first and come back here. The quick
path is:

```bash
git clone https://github.com/KE7/libero-infinity.git && cd libero-infinity
make install-full   # creates venv, installs deps, downloads HF assets
make test           # smoke-test: should pass ~10 tests
```

All commands below assume you are in the `libero-infinity/` repo root and have
activated the virtual environment:

```bash
source .venv/bin/activate   # or: uv run <command>  (works without activation)
```

---

## 3. Your first evaluation (CLI)

### Step 1 — Browse the available tasks

LIBERO-Infinity ships the full LIBERO BDDL task suite organised into four suites. List the
tasks in the `libero_goal` suite (10 tasks):

```bash
ls src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/
```

You should see:

```
open_the_middle_drawer_of_the_cabinet.bddl
open_the_top_drawer_and_put_the_bowl_inside.bddl
push_the_plate_to_the_front_of_the_stove.bddl
put_the_bowl_on_the_plate.bddl
put_the_bowl_on_the_stove.bddl
put_the_bowl_on_top_of_the_cabinet.bddl
put_the_cream_cheese_in_the_bowl.bddl
put_the_wine_bottle_on_the_rack.bddl
put_the_wine_bottle_on_top_of_the_cabinet.bddl
turn_on_the_stove.bddl
```

To see all tasks across all suites:

```bash
ls src/libero_infinity/data/libero_runtime/bddl_files/libero_{goal,spatial,object,90,10}/
```

To inspect the natural-language instruction for any task:

```python
from libero_infinity.task_config import TaskConfig

cfg = TaskConfig.from_bddl(
    "src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/"
    "put_the_bowl_on_the_plate.bddl"
)
print(cfg.language)
# → "Put the bowl on the plate"
```

### Step 2 — Run a 10-episode evaluation

We will evaluate on `put_the_bowl_on_the_plate` with position perturbation
(object positions drawn uniformly over the reachable workspace) and save
results to a JSON file.

```bash
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl \
  --perturbation position \
  --n-scenes 10 \
  --output results_position.json \
  --verbose
```

> **What `--perturbation position` does:** Each of the 10 scenes draws the bowl
> and plate positions independently and uniformly over the tabletop workspace
> (x ∈ [-0.40, 0.40] m, y ∈ [-0.30, 0.30] m), subject to a minimum pairwise
> clearance of 0.12 m and a soft anti-canonical bias (positions near the
> training pose are down-weighted 20–30%).

> **Note:** `libero-eval` requires a running robo-eval server — there is no
> built-in default policy. To connect a VLA, see
> `examples/05_robo_eval_cli.sh` or `examples/04_custom_vla.py`.

### Step 3 — Read the verbose output

With `--verbose` each episode prints a one-liner:

```
INFO  Auto-generated Scenic program: scenic/_gen_put_the_bowl_on_the_plate_a1b2c3d4_position.scenic
INFO  Compiling Scenic scenario: ...

[   1/10] ✗ steps=300  iters=  3  t= 8.2s  positions={'akita_black_bowl_1': [0.231, -0.187, 0.82], 'plate_1': [-0.118, 0.094, 0.82]}
[   2/10] ✗ steps=300  iters=  7  t= 8.5s  positions={'akita_black_bowl_1': [-0.302, 0.211, 0.82], 'plate_1': [0.097, -0.052, 0.82]}
...
[  10/10] ✗ steps=300  iters=  2  t= 8.1s  positions={'akita_black_bowl_1': [0.015, -0.265, 0.82], 'plate_1': [0.284, 0.178, 0.82]}

Success rate: 0.0% ± 0.0% (0/10 scenes)
Results written to results_position.json
```

Column meanings:

| Column | Meaning |
|--------|---------|
| `✓` / `✗` | Task succeeded / failed |
| `steps=300` | Episode ran to the full 300-step horizon |
| `iters=3` | Scenic needed 3 rejection-sampling attempts to find a valid scene |
| `t=8.2s` | Wall-clock time for this episode |
| `positions={...}` | Sampled (x, y, z) for each movable object |

### Step 4 — Inspect the JSON results file

```bash
cat results_position.json
```

The top-level structure is:

```json
{
  "scenic_path": "/abs/path/to/_gen_...position.scenic",
  "bddl_path":   "/abs/path/to/put_the_bowl_on_the_plate.bddl",
  "n_scenes":    10,
  "n_success":   0,
  "success_rate": 0.0,
  "ci_95":       0.0,
  "episodes": [
    {
      "scene_index":        0,
      "success":            false,
      "steps":              300,
      "n_scenic_rejections": 2,
      "scenic_params":      {},
      "object_positions":   {"akita_black_bowl_1": [0.231, -0.187, 0.82],
                             "plate_1":            [-0.118, 0.094, 0.82]},
      "object_classes":     {"akita_black_bowl_1": "akita_black_bowl",
                             "plate_1": "plate"},
      "elapsed_s":          8.2
    },
    ...
  ]
}
```

Key fields:

| Field | Description |
|-------|-------------|
| `success_rate` | Fraction of episodes where the BDDL goal was achieved |
| `ci_95` | 95% Wilson confidence interval half-width (e.g. `0.061` means ±6.1%) |
| `n_scenic_rejections` | How many Scenic samples were discarded before a valid scene was found |
| `scenic_params` | Numeric scene parameters sampled by Scenic (camera offsets, light intensity, etc.) |
| `object_positions` | (x, y, z) of each movable object after settling |
| `object_classes` | Asset class used for each object (may differ from BDDL default under object perturbation) |

> **Tip:** Convert to CSV for quick analysis with:
> ```python
> import json, csv
> data = json.load(open("results_position.json"))
> with open("results.csv", "w", newline="") as f:
>     w = csv.DictWriter(f, fieldnames=["scene_index","success","steps","elapsed_s"])
>     w.writeheader()
>     w.writerows(data["episodes"])
> ```

---

## 4. Adding more perturbation axes

LIBERO-Infinity has **six composable perturbation axes**. You can activate any
subset with a comma-separated list, or use a named preset.

### Quick reference

| Axis | What changes each episode | One-word summary |
|------|--------------------------|-----------------|
| `position` | Object (x, y) placement — uniform over workspace | *Where* |
| `object` | Object mesh + texture — swapped from a pool of 34 asset classes | *What* |
| `camera` | Agentview camera position (±10 cm) and tilt (±15°) | *View* |
| `lighting` | Diffuse/specular intensity (0.4×–2.0×) and ambient level | *Light* |
| `texture` | Table surface material | *Surface* |
| `distractor` | 1–5 random non-task objects added to the scene | *Clutter* |

Named presets:

| Preset | Equivalent to |
|--------|--------------|
| `combined` | `position,object` |
| `full` | All six axes |

### Switching to `position,camera`

This is the combination mentioned in the tutorial goal — it randomises both
where objects sit and from what angle the robot sees them:

```bash
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl \
  --perturbation position,camera \
  --n-scenes 10 \
  --output results_position_camera.json \
  --verbose
```

The verbose output now includes camera offset parameters in `scenic_params`:

```
[   1/10] ✗ steps=300  iters=  4  t= 8.6s  positions={'akita_black_bowl_1': [0.183, 0.072, 0.82], ...}
```

And in the JSON:

```json
"scenic_params": {
  "camera_x_offset":  0.073,
  "camera_y_offset": -0.041,
  "camera_z_offset":  0.018,
  "camera_tilt":      7.3
}
```

### Switching to `combined` (position + object)

The `combined` preset simultaneously randomises object positions *and* swaps
the movable object for a different mesh from the asset registry:

```bash
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl \
  --perturbation combined \
  --n-scenes 10 \
  --output results_combined.json \
  --verbose
```

Now `object_classes` in the JSON will vary across episodes — the bowl might
become a `white_bowl`, `red_bowl`, or `ikea_cup` instead of the canonical
`akita_black_bowl`:

```json
"object_classes": {
  "akita_black_bowl_1": "white_bowl",
  "plate_1": "plate"
}
```

### Activating all axes at once

```bash
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl \
  --perturbation full \
  --n-scenes 10 \
  --verbose
```

For a real evaluation benchmark (as in the README comparison table), use at
least 50 scenes per task — 10 gives high variance but is fine for a smoke test.

---

## 5. Using the Gym wrapper (Python API)

`LIBEROScenicEnv` is a standard `gym.Env` (gym 0.25 API). Every `reset()` call
samples a fresh perturbed scene — objects land somewhere new, possibly with a
different mesh and camera angle — and returns the first observation dict. Use
this for training loops, RL, or any code that expects a gym interface.

### Complete working example

```python
import numpy as np
from libero_infinity.gym_env import LIBEROScenicEnv
from libero_infinity.task_config import TaskConfig

# ── 1. Point at a BDDL task ──────────────────────────────────────────────────
BDDL = (
    "src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/"
    "put_the_bowl_on_the_plate.bddl"
)

# Read the natural-language instruction (needed by VLA models)
cfg = TaskConfig.from_bddl(BDDL)
print("Task:", cfg.language)
# → "Put the bowl on the plate"

# ── 2. Create the environment ─────────────────────────────────────────────────
# perturbation="position" randomises (x,y) of the bowl and plate each reset.
# resolution=256 sets camera image size to 256×256 px.
env = LIBEROScenicEnv(
    bddl_path=BDDL,
    perturbation="position",   # try "combined" or "position,camera" too
    resolution=256,
    max_steps=300,
)

# ── 3. Observation and action spaces ─────────────────────────────────────────
# Action space is always: Box(shape=(7,), low=-1, high=1)
# Obs space is a Dict built lazily after first reset() — inspect it after.
print("Action space:", env.action_space)
# → Box(-1.0, 1.0, (7,), float32)

# ── 4. Run three episodes ─────────────────────────────────────────────────────
for episode in range(3):
    # reset() samples a new scene and returns the first observation.
    obs = env.reset()

    # obs is a dict — inspect the keys once:
    if episode == 0:
        print("Observation keys:", list(obs.keys()))
        # → ['agentview_image', 'robot0_eye_in_hand_image',
        #    'robot0_joint_pos', 'robot0_eef_pos', 'robot0_eef_quat',
        #    'robot0_gripper_qpos', 'robot0_proprio-state',
        #    'akita_black_bowl_1_pos', 'plate_1_pos', 'object-state', ...]

    total_reward = 0.0

    for step in range(env.max_steps):
        # ── Your policy goes here ──────────────────────────────────────────
        # Replace this zero-action stub with a real VLA call (Section 6).
        action = np.zeros(7, dtype=np.float32)
        # action format: [dx, dy, dz, dax, day, daz, gripper]
        # values in [-1, 1]; gripper: -1 = open, +1 = close
        # ──────────────────────────────────────────────────────────────────

        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

    print(
        f"Episode {episode + 1}: "
        f"success={info['success']}  steps={info['steps']}  reward={total_reward:.1f}"
    )

env.close()
```

Expected output (zero policy, no successes):

```
Task: Put the bowl on the plate
Action space: Box(-1.0, 1.0, (7,), float32)
Observation keys: ['agentview_image', 'robot0_eye_in_hand_image', ...]
Episode 1: success=False  steps=300  reward=0.0
Episode 2: success=False  steps=300  reward=0.0
Episode 3: success=False  steps=300  reward=0.0
```

### Reading key observation fields

| Key | Shape | What it is |
|-----|-------|------------|
| `agentview_image` | `(H, W, 3)` uint8 | Third-person RGB camera; **flip rows** for standard display: `img[::-1]` |
| `robot0_eye_in_hand_image` | `(H, W, 3)` uint8 | Wrist-mounted camera |
| `robot0_eef_pos` | `(3,)` float | End-effector (x, y, z) in world frame |
| `robot0_eef_quat` | `(4,)` float | End-effector orientation `[x, y, z, w]` |
| `robot0_gripper_qpos` | `(2,)` float | Gripper finger joint positions |
| `robot0_proprio-state` | `(39,)` float | Concatenated proprioception (handy for state-based models) |
| `{obj}_pos` | `(3,)` float | Object position, e.g. `akita_black_bowl_1_pos` |

### Reading Scenic scene parameters

Scene parameters (camera offsets, sampled light levels, etc.) live in the
`EpisodeResult` returned by `evaluate()`, not in the `obs` dict. When using
the Gym wrapper, capture them via your own bookkeeping after each `reset()`, or
switch to the `evaluate()` API for richer per-episode metadata:

```python
from libero_infinity.eval import evaluate
from libero_infinity.compiler import generate_scenic_file
from libero_infinity.task_config import TaskConfig

cfg = TaskConfig.from_bddl(BDDL)
scenic_path = generate_scenic_file(cfg, perturbation="position,camera")

results = evaluate(
    scenic_path=scenic_path,
    bddl_path=BDDL,
    policy=my_policy_fn,   # (obs_dict) -> np.ndarray (7,)
    n_scenes=10,
    verbose=True,
)

for ep in results.episodes:
    print(ep.scene_index, ep.success, ep.scenic_params)
    # ep.scenic_params contains: camera_x_offset, camera_y_offset, etc.
    print(ep.object_positions)
    # ep.object_positions: {'akita_black_bowl_1': [x, y, z], ...}
```

### Parallel rollouts

For faster data collection, run multiple environments in parallel:

```python
from libero_infinity.gym_env import make_vec_env
import numpy as np

vec_env = make_vec_env(
    bddl_path=BDDL,
    n_envs=4,              # 4 parallel environments
    perturbation="combined",
    resolution=128,
)

obs = vec_env.reset()                         # shape: (4, ...)
actions = np.zeros((4, 7), dtype=np.float32)
obs, rewards, dones, infos = vec_env.step(actions)
vec_env.close()
```

---

## 6. Evaluating your own VLA

### The policy interface

LIBERO-Infinity expects your VLA to expose a single Python callable:

```python
def my_policy(obs: dict[str, np.ndarray]) -> np.ndarray:
    """
    Args:
        obs: Observation dict (see Section 5 for key list).

    Returns:
        action: np.ndarray of shape (7,), values in [-1, 1].
                Format: [dx, dy, dz, dax, day, daz, gripper]
                        (OSC_POSE end-effector delta controller)
    """
    ...
```

That's the entire contract. No HTTP server, no special protocol — just a Python
function that takes an obs dict and returns a 7D action array. The task
instruction is **not** in the obs dict; retrieve it from `TaskConfig.language`
and capture it in a closure:

```python
from libero_infinity.task_config import TaskConfig

cfg = TaskConfig.from_bddl(BDDL)

def make_policy(instruction: str):
    """Wrap a VLA model into a policy closure."""
    def policy(obs: dict) -> np.ndarray:
        # Flip the camera image from OpenGL (bottom-left) to standard (top-left)
        image = obs["agentview_image"][::-1]   # shape (H, W, 3)

        # ── Replace this with your model's inference call ──────────────────
        action = my_vla_model.predict(image, instruction)   # (7,)
        # ──────────────────────────────────────────────────────────────────

        return np.array(action, dtype=np.float64)
    return policy

policy = make_policy(cfg.language)
```

### Plug it into `evaluate()`

```python
import glob
from libero_infinity.eval import evaluate
from libero_infinity.compiler import generate_scenic_file
from libero_infinity.task_config import TaskConfig

# Evaluate every task in libero_spatial (10 tasks, 50 scenes each)
bddl_dir = "src/libero_infinity/data/libero_runtime/bddl_files/libero_spatial"
for bddl_path in sorted(glob.glob(f"{bddl_dir}/*.bddl")):
    cfg = TaskConfig.from_bddl(bddl_path)
    policy = make_policy(cfg.language)

    scenic_path = generate_scenic_file(cfg, perturbation="combined")

    results = evaluate(
        scenic_path=scenic_path,
        bddl_path=bddl_path,
        policy=policy,
        n_scenes=50,
        max_steps=300,
        env_kwargs={
            "camera_heights": 224,   # match your model's training resolution
            "camera_widths":  224,
        },
        verbose=True,
    )
    print(f"[{cfg.language[:50]}] {results.summary()}")
    # → "[Pick up the black bowl next to the plate and place] Success rate: 61.0% ± 13.5% (..."
```

### Image orientation

Most VLA models expect images with the origin at the **top-left**. MuJoCo
renders with the origin at the **bottom-left** (OpenGL convention). Always flip:

```python
image = obs["agentview_image"][::-1]   # vertical flip — almost always correct
```

Some models trained with lerobot's `LiberoProcessorStep` expect a full 180°
rotation instead. Check your model's preprocessing docs.

### Action chunks

Models that output multiple actions per inference call (e.g. pi0.5 outputs 50
at once) need a queue wrapper:

```python
def make_chunked_policy(instruction: str, chunk_size: int = 50):
    action_queue = []

    def policy(obs: dict) -> np.ndarray:
        nonlocal action_queue
        if not action_queue:
            # Query model for next chunk
            actions = my_vla_model.predict_chunk(obs, instruction)  # (chunk_size, 7)
            action_queue = list(actions)
        return action_queue.pop(0)

    return policy
```

### Hardware notes

| Component | Hardware needed |
|-----------|----------------|
| MuJoCo physics + rendering | CPU (EGL uses GPU for offscreen render but no CUDA) |
| VLA inference (7B+ models) | GPU with ≥16 GB VRAM recommended |
| Scenic constraint solving | CPU-only |

For a 200-scene run with a 7B VLA on GPU, expect roughly 5–10 minutes.

---

## 7. Understanding the results

### The summary line

```
Success rate: 61.0% ± 13.5% (31/50 scenes)
```

| Part | Meaning |
|------|---------|
| `61.0%` | Fraction of episodes where the task goal was satisfied at episode end |
| `± 13.5%` | **95% Wilson confidence interval** half-width |
| `(31/50)` | Raw counts |

### What counts as success?

An episode is a **success** if LIBERO's built-in BDDL goal checker evaluates
the goal predicate as `True` at the end of the episode (or at the step the
policy finishes).

For `put_the_bowl_on_the_plate`, the goal predicate is:

```
(And (On akita_black_bowl_1 plate_1))
```

"On" is checked by LIBERO's geometry utilities — it passes when the bowl's
centre of mass is within the plate's contact region and the bowl is in stable
contact with the plate surface. The robot doesn't need to release the bowl;
the check fires the moment the physical state satisfies the predicate.

Other predicates you'll encounter: `InTopDrawerOf`, `InContainer`,
`TurnedOn`, `Open`, `Stacked`.

### Wilson confidence intervals

LIBERO-Infinity reports **Wilson score 95% confidence intervals** rather than
normal-approximation ("Wald") intervals. Wilson intervals are accurate even
when the sample size is small or the success rate is near 0% or 100% — exactly
the regime where evaluating on 10–50 scenes lives.

Rule of thumb: a ±5% CI requires roughly **100 scenes**; ±10% requires ~25.

```python
# The CI half-width is available directly:
print(f"{results.success_rate:.1%} ± {results.ci_95:.1%}")
```

### What to compare against

| Baseline | Meaning |
|----------|---------|
| Standard LIBERO (no perturbation) | Fixed initial states from the original benchmark dataset |
| `--perturbation position` | Continuous uniform position randomisation |
| `--perturbation combined` | Position + object identity (recommended default) |
| `--perturbation full` | All six axes — the hardest test |

A significant gap between "no perturbation" and "combined" means the policy has
memorised scene layout rather than truly generalising. This is the key insight
LIBERO-Infinity is designed to surface.

### Adversarial mode

Once you have a baseline success rate, you can search for *worst-case* scenes:

```bash
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl \
  --mode adversarial \
  --n-scenes 100 \
  --verbose
```

This uses cross-entropy Bayesian optimisation (via VerifAI) to iteratively
concentrate sampling on failure-inducing scene configurations.

---

## 8. Next steps

You now have the fundamentals. Here is where to go next:

| Document | What you'll learn |
|----------|-----------------|
| [Scenic Perturbations](scenic_perturbations.md) | Detailed parameters for all six perturbation axes, constraint tuning, and adding new asset variants |
| [Gym Wrapper](gym-wrapper.md) | Parallel rollouts with `make_vec_env`, RL training integration, curriculum learning with task reversal |
| [Evaluation Pipeline](evaluation_pipeline.md) | Full CLI flag reference, adversarial search, live rendering, and technical details (control frequency, coordinate system) |
| [Observations & Actions](observations-actions.md) | Complete obs dict schema, action dimensions, proprioception keys, and concrete policy examples |
| [Architecture](architecture.md) | How the Scenic → BDDL → MuJoCo pipeline fits together; layered Scenic design; file map |
| [API Reference](api-reference.md) | Full Python API for `evaluate()`, `LIBEROScenicEnv`, `TaskConfig`, `compile_task_to_scenic()`, and every other module |
| [Task Reversal](task-reversal.md) | Flip any task backward — the goal state becomes the initial configuration |

### Common next experiments

```bash
# Evaluate a different suite (libero_spatial — 10 tasks with spatial language)
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate.bddl \
  --perturbation combined --n-scenes 50 --verbose

# Add clutter objects (1-5 distractors per scene)
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl \
  --perturbation position,distractor --n-scenes 50 --verbose

# Reverse the task: bowl starts on the plate, must be placed back on the table
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl \
  --reverse --perturbation position --n-scenes 50 --verbose

# Watch the robot live (requires a display)
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl \
  --perturbation combined --n-scenes 3 --watch cv2 --verbose
```

---

*Have questions or found a bug? See [contributing.md](contributing.md).*
