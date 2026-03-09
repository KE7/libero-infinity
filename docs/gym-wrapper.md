# Gym Wrapper for RL/VLA Training

[Back to main README](../README.md)

**Module**: `src/libero_infinity/gym_env.py`

`LIBEROScenicEnv` wraps the full Scenic perturbation pipeline into a standard
`gym.Env` (gym 0.25 API). Each `reset()` samples a new perturbed scene —
randomising object positions, assets, camera, lighting, etc. — and creates
a fresh LIBERO simulation. This gives RL and VLA training loops domain
randomization for free.

---

## Quick start

```python
from libero_infinity.gym_env import LIBEROScenicEnv

env = LIBEROScenicEnv(
    bddl_path="src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/"
              "put_the_bowl_on_the_plate.bddl",
    perturbation="combined",   # position + object identity
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

---

## Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bddl_path` | `str` | required | Path to BDDL task file |
| `perturbation` | `str` | `"position"` | `"position"`, `"object"`, `"combined"`, `"camera"`, `"lighting"`, `"distractor"`, `"full"` |
| `scenic_path` | `str \| None` | `None` | Hand-written .scenic file (auto-generated if None) |
| `resolution` | `int` | `128` | Camera image resolution |
| `max_steps` | `int` | `300` | Episode horizon |
| `seed` | `int \| None` | `None` | RNG seed (applied once on first reset) |
| `reverse` | `bool` | `False` | Reverse the task (backward evaluation) |
| `scenic_params` | `dict` | `{}` | Override Scenic globalParameters |
| `env_kwargs` | `dict` | `{}` | Extra kwargs for OffScreenRenderEnv |
| `scenic_generate_kwargs` | `dict` | `{}` | Extra kwargs for `generate_scenic()` |

---

## Action and observation spaces

### Action space

`gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=float32)`

7D continuous: `[dx, dy, dz, droll, dpitch, dyaw, gripper]`.
See [observations-actions.md](observations-actions.md) for details.

### Observation space

`gym.spaces.Dict` constructed lazily from the first observation.
Contains all visual, proprioceptive, and object-state keys described in
[observations-actions.md](observations-actions.md).

---

## Reward

Binary sparse reward:
- `1.0` when the LIBERO task predicate is satisfied (e.g., bowl is on the plate)
- `0.0` otherwise

The episode terminates when either:
1. The task is completed (`success=True`)
2. The step count reaches `max_steps`

For shaped rewards, access `info["success"]` and the raw observation dict
to compute custom reward functions.

---

## Parallel rollouts

### `make_vec_env()`

Creates multiple `LIBEROScenicEnv` instances wrapped in gym's vectorized
environment API. Uses `AsyncVectorEnv` (subprocess-based) by default for
true parallelism.

```python
from libero_infinity.gym_env import make_vec_env

vec_env = make_vec_env(
    bddl_path="path/to/task.bddl",
    n_envs=4,
    perturbation="combined",
    resolution=128,
)

obs = vec_env.reset()                       # batched reset
actions = np.stack([policy(o) for o in obs])  # (4, 7)
obs, rewards, dones, infos = vec_env.step(actions)
vec_env.close()
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bddl_path` | `str` | required | Shared BDDL task file |
| `n_envs` | `int` | `4` | Number of parallel environments |
| `perturbation` | `str` | `"position"` | Perturbation mode |
| `resolution` | `int` | `128` | Camera resolution |
| `max_steps` | `int` | `300` | Episode horizon |
| `use_subprocess` | `bool` | `True` | `True` = `AsyncVectorEnv`, `False` = `SyncVectorEnv` |

### Choosing the right parallelism

| Mode | When to use |
|------|-------------|
| `AsyncVectorEnv` (`use_subprocess=True`) | Production training — each env runs in its own process, giving true parallel MuJoCo stepping |
| `SyncVectorEnv` (`use_subprocess=False`) | Debugging — sequential execution, easier to inspect errors |

### With Stable-Baselines3

```python
from stable_baselines3 import PPO
from libero_infinity.gym_env import make_vec_env

vec_env = make_vec_env(
    bddl_path="path/to/task.bddl",
    n_envs=8,
    perturbation="full",
    resolution=128,
    use_subprocess=True,
)

model = PPO("MultiInputPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100_000)
```

---

## Rendering

```python
env = LIBEROScenicEnv(bddl_path="path/to/task.bddl", resolution=512)
obs = env.reset()

# Get the current frame
frame = env.render(mode="rgb_array")  # (512, 512, 3) uint8

# Display with OpenCV
import cv2
cv2.imshow("LIBERO", frame[::-1, :, ::-1])  # flip + RGB→BGR
cv2.waitKey(0)
```

---

## Lifecycle

1. **Constructor**: Compiles the Scenic scenario (one-time cost ~2-5s)
2. **`reset()`**: Generates a new scene, creates LIBERO env, injects poses (~0.5-1s)
3. **`step(action)`**: Advances MuJoCo physics (~5ms per step)
4. **`close()`**: Destroys simulation, cleans up temp files

The Scenic scenario is compiled once and reused across all episodes. Only
scene generation and LIBERO env creation happen per-episode in `reset()`.

---

## Integration patterns

### Domain randomization for VLA fine-tuning

```python
env = LIBEROScenicEnv(
    bddl_path="path/to/task.bddl",
    perturbation="full",         # all axes randomized
    resolution=256,              # match your VLA input resolution
    max_steps=400,
)

# Collect demonstration data with randomized scenes
for episode in range(1000):
    obs = env.reset()
    trajectory = []
    for step in range(400):
        action = expert_policy(obs)
        obs, reward, done, info = env.step(action)
        trajectory.append((obs, action, reward))
        if done:
            break
    save_trajectory(trajectory)
```

### Curriculum learning with reversed tasks

```python
# Start with forward tasks (easier)
forward_env = LIBEROScenicEnv(
    bddl_path="path/to/task.bddl",
    perturbation="position",
    reverse=False,
)

# Progress to reversed tasks (harder — novel initial configs)
backward_env = LIBEROScenicEnv(
    bddl_path="path/to/task.bddl",
    perturbation="position",
    reverse=True,
)
```
