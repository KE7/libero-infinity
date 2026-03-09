# Observation and Action Spaces

[Back to main README](../README.md)

Understanding these is essential for plugging in your own VLA or RL policy.

---

## Observation dict

The `obs` dict returned by `env.reset()` and `sim.step_with_action(action)` contains:

### Visual observations

Configurable resolution, default 128x128 (set via `--resolution N` or `camera_heights`/`camera_widths`
in env_kwargs).

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `agentview_image` | `(H, W, 3)` | `uint8` | Third-person RGB camera (above and behind robot) |
| `robot0_eye_in_hand_image` | `(H, W, 3)` | `uint8` | Wrist-mounted RGB camera |

Images follow OpenGL convention (origin bottom-left). Flip vertically for standard display:
`frame[::-1]`.

### Robot proprioception

Always present, independent of task.

| Key | Shape | Description |
|-----|-------|-------------|
| `robot0_joint_pos` | `(7,)` | Joint angles (radians) |
| `robot0_joint_vel` | `(7,)` | Joint velocities |
| `robot0_joint_pos_cos` | `(7,)` | cos(joint angles) |
| `robot0_joint_pos_sin` | `(7,)` | sin(joint angles) |
| `robot0_eef_pos` | `(3,)` | End-effector position (x, y, z) |
| `robot0_eef_quat` | `(4,)` | End-effector orientation (quaternion) |
| `robot0_gripper_qpos` | `(2,)` | Gripper finger joint positions |
| `robot0_gripper_qvel` | `(2,)` | Gripper finger joint velocities |
| `robot0_proprio-state` | `(39,)` | Concatenated proprioceptive state |

### Object state

Varies per task — one group per object in the BDDL.

| Key pattern | Shape | Description |
|-------------|-------|-------------|
| `{obj}_pos` | `(3,)` | Object position (x, y, z) |
| `{obj}_quat` | `(4,)` | Object orientation (quaternion) |
| `{obj}_to_robot0_eef_pos` | `(3,)` | Relative position to end-effector |
| `{obj}_to_robot0_eef_quat` | `(4,)` | Relative orientation to end-effector |
| `object-state` | `(N,)` | Concatenated object state (size varies by task) |

For example, `put_the_bowl_on_the_plate.bddl` has object keys for `akita_black_bowl_1`,
`plate_1`, `cream_cheese_1`, and `wine_bottle_1`.

---

## Action space

| Property | Value |
|----------|-------|
| Shape | `(7,)` |
| Range | `[-1.0, 1.0]` per dimension |
| Controller | OSC_POSE (Operational Space Controller) |
| Robot | Panda 7-DOF arm with parallel-jaw gripper |

### Action dimensions

| Index | Description |
|-------|-------------|
| 0-2 | End-effector position delta (dx, dy, dz) |
| 3-5 | End-effector orientation delta (axis-angle) |
| 6 | Gripper action (-1 = open, +1 = close) |

---

## Policy interface

The `evaluate()` function and `LIBEROScenicEnv` Gym wrapper both expect a policy
with this signature:

```python
def my_policy(obs: dict[str, np.ndarray]) -> np.ndarray:
    """
    Args:
        obs: Observation dict with keys described above.

    Returns:
        Action array of shape (7,) with values in [-1, 1].
    """
    # Example: VLA inference
    image = obs["agentview_image"][::-1]  # flip to standard orientation
    proprio = obs["robot0_proprio-state"]
    action = my_model(image, proprio)     # your model here
    return action
```

### Task instructions

VLA policies typically need a natural language task instruction (e.g. "put the bowl on the plate"). This is NOT included in the obs dict. To access it:

```python
from libero_infinity.task_config import TaskConfig

cfg = TaskConfig.from_bddl("path/to/task.bddl")
task_instruction = cfg.language  # e.g. "put the bowl on the plate"

def make_policy(instruction: str):
    def policy(obs):
        image = obs["agentview_image"][::-1]
        action = my_vla_model(image, instruction)
        return action
    return policy

results = evaluate(
    scenic_path=...,
    bddl_path="path/to/task.bddl",
    policy=make_policy(task_instruction),
    n_scenes=100,
)
```

### Concrete example: heuristic policy

```python
import numpy as np
from libero_infinity.eval import evaluate

def reach_forward_policy(obs):
    """Move end-effector forward and down, then close gripper."""
    action = np.zeros(7)
    action[0] = 0.3   # move forward (x)
    action[2] = -0.1   # move down (z)
    action[6] = 1.0    # close gripper
    return action

results = evaluate(
    scenic_path=None,  # uses --bddl auto-generation
    bddl_path="src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl",
    policy=reach_forward_policy,
    n_scenes=10,
    max_steps=300,
    verbose=True,
)
print(results.summary())
```

### Using the Gym wrapper

```python
from libero_infinity.gym_env import LIBEROScenicEnv

env = LIBEROScenicEnv(
    bddl_path="path/to/task.bddl",
    perturbation="position",
    resolution=256,
)

obs = env.reset()
for _ in range(300):
    action = my_policy(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
env.close()
```
