# LIBERO-Infinity Examples

Standalone, runnable examples demonstrating the key APIs of LIBERO-Infinity.

## Prerequisites

Install the package first (from the repo root):

```bash
make install-full   # or: uv sync --extra simulation --extra dev
```

All examples use `MUJOCO_GL=egl` for headless rendering (default on Linux servers).
On macOS, omit the variable. Use `MUJOCO_GL=osmesa` if EGL is unavailable.

---

## Examples

| File | Description |
|------|-------------|
| [`01_basic_eval.py`](01_basic_eval.py) | Minimal evaluation loop using `evaluate()` — loads a BDDL task, runs 3 episodes with a random policy, prints per-episode success and Scenic scene params |
| [`02_gym_wrapper.py`](02_gym_wrapper.py) | Standard Gym API usage — shows `reset()` / `step()` loop, how to read obs keys (RGB image, proprioception, object state), and how to access Scenic scene params from `info` |
| [`03_perturbation_axes.py`](03_perturbation_axes.py) | Walks through all 6 perturbation axes — resets the environment twice per axis and prints the sampled Scenic params to show what each axis actually randomizes |
| [`04_custom_vla.py`](04_custom_vla.py) | Template for plugging in a custom VLA — provides a `MockVLA` with the robo-eval `/predict` interface, runs a full eval loop, with clear TODOs showing where to swap in a real model |
| [`05_robo_eval_cli.sh`](05_robo_eval_cli.sh) | **Recommended for real VLAs** — shell script showing how to run `robo-eval` (from liten-vla) with LIBERO-Infinity; includes the Pi0.5 integration test command and commented variations for different perturbation axes, episode counts, and tasks |

---

## Quick run

```bash
cd /path/to/libero-infinity

# 01 — basic evaluation (random policy, 3 episodes)
MUJOCO_GL=egl python examples/01_basic_eval.py

# 02 — gym wrapper demonstration
MUJOCO_GL=egl python examples/02_gym_wrapper.py

# 03 — perturbation axes survey
MUJOCO_GL=egl python examples/03_perturbation_axes.py

# 04 — custom VLA template (mock model, no GPU needed)
MUJOCO_GL=egl python examples/04_custom_vla.py

# 05 — robo-eval CLI (requires liten-vla + Pi0.5 weights)
bash examples/05_robo_eval_cli.sh
```

---

## Notes

- All examples resolve BDDL paths relative to the repo root. Run them from the
  repo root, or set the `LIBERO_ROOT` environment variable.
- Each episode takes ~5–15 s to set up (Scenic compilation + MuJoCo reset).
  Subsequent episodes in the same process are faster (~0.5–1 s per reset).
- See [`docs/observations-actions.md`](../docs/observations-actions.md) for the
  full obs/action schema, and [`docs/gym-wrapper.md`](../docs/gym-wrapper.md)
  for Gym API details.
