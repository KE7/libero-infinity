# Evaluation Pipeline

[Back to main README](../README.md)

The evaluation harness (`src/libero_infinity/eval.py`) drives the full pipeline:
Scenic scene sampling, BDDL resolution, LIBERO environment creation, policy rollout,
and metric aggregation. It supports two evaluation modes and provides both a Python
API and a CLI.

---

## Pipeline Overview

<p align="center">
  <img src="../assets/architecture_pipeline.png" width="85%" alt="Evaluation Pipeline">
</p>

For each evaluation run:

1. **Parse** the BDDL task file to extract objects, regions, and constraints
2. **Generate** a Scenic program (or use a hand-written one) with the requested perturbation axes
3. **Sample** a scene from the Scenic distribution via rejection sampling
4. **Resolve** the BDDL (apply asset substitutions, distractors, or task reversal)
5. **Create** a LIBERO simulation with the sampled poses and perturbations
6. **Evaluate** the policy over the episode (up to `max_steps` actions)
7. **Record** the result (success/failure, steps, positions, timing)
8. **Report** aggregate statistics with 95% Wilson confidence intervals

---

## Standard Evaluation (i.i.d. Sampling)

Samples N scenes independently from the Scenic program and evaluates the policy on
each. Reports success rate with 95% Wilson confidence interval.

### CLI

The CLI requires a running robo-eval server — there is no built-in default
policy. To connect a VLA, see `examples/05_robo_eval_cli.sh`. To evaluate
with a custom policy callable, use the Python API (see
[End-to-End VLA Evaluation Example](#end-to-end-vla-evaluation-example) below).

```bash
libero-eval \
  --bddl path/to/task.bddl \
  --perturbation combined \
  --n-scenes 200 \
  --max-steps 300 \
  --output results.json \
  --verbose
```

### Python API

```python
from libero_infinity.eval import evaluate

results = evaluate(
    scenic_path="scenic/combined_perturbation.scenic",
    bddl_path="path/to/task.bddl",
    policy=my_policy_fn,          # callable: obs_dict -> np.ndarray (action)
    n_scenes=200,
    max_steps=300,
    scenic_params={
        "perturb_class": "akita_black_bowl",
        "min_clearance": 0.12,
        "include_canonical": False,
    },
    verbose=True,
    seed=42,
)

print(results.summary())
# -> "Success rate: 73.5% +/- 6.1% (147/200 scenes)"
```

### Output

Each `EpisodeResult` contains: `success`, `steps`, `n_scenic_rejections`,
`object_positions`, `object_classes`, `elapsed_s`, `scenic_params`.

---

## End-to-End VLA Evaluation Example

The Python API is the primary way to evaluate real VLA policies. The CLI uses a
to evaluate a real model, pass your policy callable to `evaluate()` directly.

Here is a complete, copy-paste-ready example using a HuggingFace VLA model:

```python
import glob
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from libero_infinity.eval import evaluate
from libero_infinity.task_config import TaskConfig
from libero_infinity.compiler import generate_scenic_file

# --- 1. Load your VLA model ---
model_id = "openvla/openvla-7b-finetuned-libero-spatial"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

# --- 2. Define a policy callable: obs_dict -> action (7,) ---
#
# The policy receives the obs dict from the simulator and must return a 7D
# action array in [-1, 1]:
#   [dx, dy, dz, dax, day, daz, gripper]  (OSC_POSE controller)
#
# Task instructions are NOT in the obs dict — get them from TaskConfig.language
# and capture via a closure.

def make_policy(instruction: str):
    """Create a policy closure that captures the task instruction."""
    def policy(obs: dict) -> np.ndarray:
        # Third-person camera image (flip from OpenGL bottom-left origin)
        image = obs["agentview_image"][::-1]

        # Run VLA inference (adapt to your model's API)
        inputs = processor(image, instruction).to(model.device)
        action = model.predict_action(**inputs)  # shape: (7,)
        return np.array(action, dtype=np.float64)
    return policy

# --- 3. Set image resolution to match your model ---
#
# Pass camera_heights and camera_widths in env_kwargs.
# Common values: 224 (OpenVLA, SpatialVLA), 256 (pi0.5, SmolVLA).
env_kwargs = {"camera_heights": 224, "camera_widths": 224}

# --- 4. Iterate over all tasks in a benchmark suite ---
bddl_dir = "src/libero_infinity/data/libero_runtime/bddl_files/libero_spatial"
bddl_files = sorted(glob.glob(f"{bddl_dir}/*.bddl"))

for bddl_path in bddl_files:
    # Parse the task to get the language instruction
    cfg = TaskConfig.from_bddl(bddl_path)
    print(f"Task: {cfg.language}")

    # Create the policy with the task instruction
    policy = make_policy(cfg.language)

    # Auto-generate a Scenic perturbation program from the BDDL
    scenic_path = generate_scenic_file(cfg, perturbation="combined")

    # Evaluate on 50 perturbed scenes
    results = evaluate(
        scenic_path=scenic_path,
        bddl_path=bddl_path,
        policy=policy,
        n_scenes=50,
        max_steps=300,
        env_kwargs=env_kwargs,
        verbose=True,
    )
    print(results.summary())
```

### Handling Action Chunks

Some VLA models (e.g., pi0.5) output multiple actions per inference call
(action chunks). Execute them sequentially before querying the model again:

```python
def make_chunked_policy(instruction: str, chunk_size: int = 50):
    """Policy wrapper for models that output action chunks."""
    action_queue = []

    def policy(obs: dict) -> np.ndarray:
        nonlocal action_queue
        if not action_queue:
            # Query the model for a new chunk of actions
            actions = my_vla_model.predict(obs, instruction)  # shape: (chunk_size, 7)
            action_queue = list(actions)
        return action_queue.pop(0)
    return policy
```

---

## Adversarial Evaluation (Cross-Entropy Search)

Uses Scenic's feedback-driven sampling to find failure-inducing scenes. After each
episode, the harness passes `feedback = 0.0` (success) or `1.0` (failure) back to
`scenario.generate()`. When the Scenic program uses `VerifaiRange` instead of `Range`,
the cross-entropy sampler concentrates on high-failure regions over iterations.

### CLI

```bash
libero-eval \
  --bddl path/to/task.bddl \
  --mode adversarial \
  --n-scenes 200 \
  --verbose
```

### Python API

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

# Results sorted by failure (worst-case first)
worst = results.episodes[0]
print(f"Worst-case positions: {worst.object_positions}")
```

Even without VerifAI installed, `evaluate_adversarial()` works as a standard
evaluation loop with worst-case episode tracking.

---

## CLI Reference

### All Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--bddl` | *required* | Path to BDDL task file |
| `--scenic` | auto-gen | Path to .scenic program (auto-generated from BDDL if omitted) |
| `--perturbation` | `position` | Perturbation axes: `position`, `object`, `combined`, `camera`, `lighting`, `distractor`, `full` |
| `--n-scenes` | `100` | Number of sampled scenes |
| `--max-steps` | `300` | Episode horizon |
| `--mode` | `standard` | `standard` (i.i.d.) or `adversarial` (CE search) |
| `--seed` | None | RNG seed for reproducibility |
| `--output` | None | JSON output path |
| `--watch` | None | `cv2` (OpenCV window) or `viewer` (MuJoCo GUI) |
| `--camera` | `agentview` | Camera for `--watch cv2` display |
| `--resolution` | `512` | Camera image resolution in pixels |
| `--reverse` | False | Reverse the task (backward evaluation) |
| `--max-distractors` | `5` | Max distractor objects |
| `--min-distractors` | `1` | Min distractor objects |
| `--verbose` | False | Print per-episode summaries |

### Auto-Generation from BDDL

When `--scenic` is not specified, the harness auto-generates a Scenic program
from the BDDL:

```bash
# No --scenic needed
libero-eval --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/any_task.bddl \
  --perturbation full --n-scenes 100 --verbose
```

This works with any LIBERO BDDL task file.

---

## Live Rendering

### OpenCV Window (`--watch cv2`)

Streams the agent camera to a lightweight window. Press `q` to skip to the next episode.

```bash
libero-eval --bddl path/to/task.bddl --n-scenes 5 --watch cv2 --verbose
```

Requires a display (`DISPLAY` environment variable set).

### MuJoCo Interactive Viewer (`--watch viewer`)

Opens the full MuJoCo GUI with orbit, pan, zoom, and contact-force visualization.
Close the window to advance.

```bash
libero-eval --bddl path/to/task.bddl --n-scenes 1 --watch viewer
```

Requires GLFW and X11.

> **Note:** On headless servers, only headless mode (`MUJOCO_GL=egl`) is available. The cv2 and viewer modes are interactive debugging tools for machines with displays. See [docs/internal/scenic_test_results.md](internal/scenic_test_results.md) for render mode compatibility.

---

## Task Reversal

Add `--reverse` to any evaluation command to flip the task backward:

```bash
libero-eval --bddl path/to/put_the_bowl_on_the_plate.bddl \
  --reverse --perturbation position --n-scenes 100 --verbose
```

The bowl starts on the plate and must be placed back on the table. See [task-reversal.md](task-reversal.md) for details.

---

## Extending to New Tasks

### Option 1: Auto-generate from BDDL (recommended)

No Scenic program needed. Just point at the BDDL:

```bash
libero-eval --bddl src/libero_infinity/data/libero_runtime/bddl_files/my_new_task.bddl --perturbation combined --n-scenes 100
```

### Option 2: Add OOD variants for new object classes

Edit `src/libero_infinity/data/asset_variants.json` to add variant lists and
dimensions for new classes.

### Option 3: Hand-write a Scenic program

For custom constraints beyond what the generator produces, write a `.scenic` file:

```scenic
model libero_model

param bddl_path = ""

my_obj = new LIBEROObject with libero_name "my_obj_1",
                           with asset_class "my_class",
                           with width 0.10, with length 0.10, with height 0.06,
                           at Vector(Range(TABLE_X_MIN + 0.05, TABLE_X_MAX - 0.05),
                                     Range(TABLE_Y_MIN + 0.05, TABLE_Y_MAX - 0.05),
                                     TABLE_Z)
```

---

## Technical Details

This section documents implementation details that are important when integrating
a VLA policy with LIBERO-Infinity.

### Control Frequency

The simulator runs at **20 Hz** (50 ms per step). This is set in `simulator.py`
via `control_freq=20` when creating the LIBERO environment. Each call to
`step_with_action()` or `env.step()` advances the simulation by one control step.
With a `max_steps=300` episode horizon, each episode lasts up to 15 seconds of
simulated time.

### Image Orientation

Camera images follow **OpenGL convention** with the origin at the bottom-left corner.
Most VLA models expect top-left origin (standard image convention). Flip vertically
before passing to your model:

```python
image = obs["agentview_image"][::-1]  # flip rows (vertical flip)
```

Some models (e.g., those trained with lerobot's `LiberoProcessorStep`) expect a
full 180-degree rotation (`[::-1, ::-1]`, flipping both height and width). Check
your model's training preprocessing to determine the correct convention.

### State Encoding

The observation dict provides the end-effector orientation as a quaternion
(`robot0_eef_quat`, shape `(4,)`, `[x, y, z, w]` order). Many VLA models expect
axis-angle representation instead. Convert with scipy:

```python
from scipy.spatial.transform import Rotation
quat = obs["robot0_eef_quat"]  # [x, y, z, w]
axisangle = Rotation.from_quat(quat).as_rotvec()  # shape (3,)
```

A typical 8-dimensional state vector for VLA models is:
`[eef_pos(3), axisangle(3), gripper_qpos(2)]`.

### Warmup / Settling Steps

After injecting Scenic-sampled object positions, the simulator runs **50 physics
settling steps** (via `mj_step`) to let objects come to rest on the table surface.
Velocities are then zeroed so the policy starts from a quiescent state.

This differs from standard LIBERO evaluation, which runs **10 no-op steps** using
the environment's `step()` method (which includes control). The settling steps in
LIBERO-Infinity are raw physics steps (no controller), ensuring objects reach stable
resting poses after position injection.

### GPU and Hardware Requirements

| Component | Hardware | Notes |
|-----------|----------|-------|
| MuJoCo simulation | **CPU only** | No GPU required for physics or rendering (EGL uses the GPU for offscreen rendering but does not require CUDA) |
| VLA inference | **GPU recommended** | Most VLA models (7B+ parameters) require a GPU with 16+ GB VRAM |
| Scenic sampling | **CPU only** | Constraint solving is CPU-bound |

For a typical evaluation run (200 scenes, single policy), expect:
- ~5 minutes with a fast VLA on GPU
- ~2 hours with a slow VLA or CPU-only inference

Set `MUJOCO_GL=egl` for headless GPU-accelerated rendering on Linux servers.

### Coordinate System

The LIBERO/MuJoCo world frame:
- **+x** = forward (away from robot base)
- **+y** = left
- **+z** = up
- Table surface at z ≈ 0.82 m

Actions are 7D end-effector deltas in this frame:
`[dx, dy, dz, dax, day, daz, gripper]` with values in `[-1, 1]`.

---

## EvalResults API

| Field | Type | Description |
|-------|------|-------------|
| `scenic_path` | `str` | Path to the Scenic program used |
| `bddl_path` | `str` | Path to the BDDL task file |
| `n_scenes` | `int` | Total scenes evaluated |
| `n_success` | `int` | Successful episodes |
| `success_rate` | `float` | Fraction of successes |
| `ci_95` | `float` | 95% Wilson CI half-width |
| `episodes` | `list[EpisodeResult]` | Per-episode data |
| `summary()` | method | Human-readable summary |
| `to_json()` | method | JSON serialization |

See [api-reference.md](api-reference.md) for the complete Python API.
