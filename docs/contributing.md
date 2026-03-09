# Contributing to LIBERO-Infinity

[Back to main README](../README.md)

Thank you for your interest in contributing to LIBERO-Infinity! This guide will help you
get started.

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/libero-infinity.git
   cd libero-infinity
   ```
3. **Install [uv](https://docs.astral.sh/uv/)** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Fallback: pip install uv
   ```
4. **Install** the development environment:
   ```bash
   make install-full
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/my-feature
   ```

---

## Development Setup

```bash
# Install all dependencies including dev tools
uv sync --extra simulation --extra dev

# Run linter
uv run ruff check src/ tests/

# Run formatter
uv run ruff format src/ tests/

# Run the full test suite (requires GPU/EGL for simulation tests)
make test
# Equivalent: MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m pytest tests/ -v

# Run only Tier 1 (no-GPU) tests — fast iteration for contributors without a GPU
make test-fast
# Equivalent: MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m pytest \
#   tests/test_scenic.py tests/test_perturbation_policy.py tests/test_planner.py -v

# Run specific test file directly
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m pytest tests/test_scenic.py -v
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m pytest tests/test_libero.py -v
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m pytest tests/test_gym.py -v

# Full build verification
MUJOCO_GL=egl uv run python scripts/verify_build.py
```

---

## Code Style

- **Python 3.11+** features are welcome
- **Ruff** for linting and formatting (configured in `pyproject.toml`)
- Target line length: **100 characters**
- Lint rules: E, F, W, I (pyflakes, pycodestyle, isort)

---

## Areas for Contribution

### New Perturbation Axes

#### IR / Planner / Renderer Pipeline

The compiler is a four-stage pipeline: BDDL text → `TaskConfig` → `SemanticSceneGraph` (IR)
→ `PerturbationPlan` → Scenic program string.  The **planner** (`plan_<axis>()` functions in
`planner/axes.py`) is a pure function of the scene graph that computes an axis-specific plan
struct (e.g. `BackgroundPlan`) encoding *what* to perturb and *within what envelope*, with zero
rendering logic.  The **renderer** (`scenic_renderer.py`) is a pure function of the plan that
emits Scenic 3 `param` statements (e.g. `param wall_texture = Uniform(...)`) — it contains no
task-semantic conditionals.  The generated `.scenic` file is compiled by Scenic, which samples
concrete parameter values (e.g. a specific texture name); those values are then read by
`LIBEROSimulation` (the MuJoCo bridge) and applied to the live simulation after `env.reset()`.
This strict layering means task-semantic decisions live only in the IR builder/planner, while
the renderer and Scenic files remain fully task-agnostic.

#### Files to Touch

Adding a new perturbation axis (example: `background`) requires changes to **all** of the
following files:

1. **`src/libero_infinity/planner/types.py`** — Add a new `@dataclass` plan struct (e.g.
   `BackgroundPlan`) that encodes the perturbation envelope for the new axis (fields, defaults,
   and docstring explaining each field's semantics).

2. **`src/libero_infinity/planner/axes.py`** — Add a `plan_<axis>()` function that takes
   `(graph, request_axes, diagnostics)` and returns the new plan struct (or `None` when the
   axis is not requested).  Add any axis-specific constants (e.g. texture lists, budget caps)
   at module level above the function.

3. **`src/libero_infinity/planner/composition.py`** — Wire the new axis into `AXIS_PRESETS`
   (add `"<axis>"` to the `"combined"` and/or `"full"` frozensets as appropriate) and call
   `plan_<axis>()` inside `plan_perturbations()`, storing the result on `PerturbationPlan`.

4. **`src/libero_infinity/planner/__init__.py`** — Export the new plan struct and planner
   function by adding them to the `from ... import (...)` blocks and to `__all__`.

5. **`src/libero_infinity/renderer/scenic_renderer.py`** — Add a `_render_<axis>()` fragment
   function (pure function of `plan` + `graph`) that emits the Scenic `param` statements for
   the new axis, and call it from `render_scenic()` in the correct position in the `fragments`
   list.

6. **`src/libero_infinity/compiler.py`** — If the new axis introduces params that require
   special handling during scene compilation (e.g. file-path resolution, extra imports), add
   that logic here.  For most axes no change is needed beyond wiring the planner; verify that
   `compile_task_to_scenario()` correctly propagates the new params.

7. **`scenic/libero_model.scenic`** — Add any new world-level constants or vocabulary needed
   by the `.scenic` file (e.g. `LIBERO_BACKGROUND_TEXTURES` list, new `BoxRegion` definitions,
   or helper variables that all perturbation programs share).

8. **`scenic/<axis>_perturbation.scenic`** — Create a new standalone Scenic program for the
   axis (use `background_perturbation.scenic` as a template): `model libero_model`, global
   `param` declarations, sampling expressions (e.g. `Uniform(*_candidates)`), and an `ego`
   placeholder object required by Scenic.

9. **`tests/`** — Add unit tests at the appropriate tier: a Tier 1 test in `test_planner.py`
   that calls `plan_<axis>()` directly and asserts the plan struct fields, and a Tier 1 test in
   `test_scenic.py` that compiles a task with the new axis active and checks that the expected
   `param` keys appear in the generated Scenic output.

### New Object Variants

Expand the OOD asset registry:

1. Verify the new asset exists in LIBERO's MuJoCo XML files
2. Add the variant to `src/libero_infinity/data/asset_variants.json`
3. Add bounding box dimensions for clearance calculations

### Benchmark Results

Help populate the evaluation tables:

1. Run `libero-eval` with different policies and perturbation modes
2. Report results as success rate +/- 95% Wilson CI over 200+ scenes
3. Include configuration details (policy, perturbation, n_scenes, seed)

### Documentation

- Improve existing docs with clearer examples
- Add tutorials for common workflows
- Create video demonstrations of the evaluation pipeline

### Bug Reports

Include:
- Python version, OS, GPU (if applicable)
- Steps to reproduce
- Full error traceback
- Expected vs. actual behavior

---

## Pull Request Process

1. Ensure all tests pass: `make test` (or at minimum `make test-fast` for no-GPU environments)
2. Ensure code passes linting: `uv run ruff check src/ tests/`
3. Update documentation if your change affects user-facing behavior
4. Write a clear PR description explaining the change and its motivation
5. Reference any related issues

---

## Test Tiers

| Tier | Files | Requires | Speed | `make` target |
|------|-------|----------|-------|---------------|
| 1 | `test_scenic.py`, `test_perturbation_policy.py`, `test_planner.py` | Scenic 3 only — no GPU | Fast (~10–30s) | `make test-fast` |
| 2 | `test_libero.py` | Scenic + LIBERO + MuJoCo | Medium (~60s) | `make test` |
| 3 | `test_gym.py` | Full stack | Medium (~30s) | `make test` |

Use `make test-fast` when working on Scenic programs, perturbation logic, or the planner —
it runs without a GPU and completes in under a minute.  Use `make test` (full suite) before
opening a pull request to confirm nothing is broken end-to-end.

When adding features, write tests at the appropriate tier. Tier 1 tests run without
LIBERO and are the fastest to iterate on.

---

## Project Structure

```
src/libero_infinity/
  eval.py              # Evaluation harness + CLI entry point
  simulator.py         # Scenic <-> MuJoCo bridge
  gym_env.py           # Gym wrapper
  task_config.py       # BDDL parser
  compiler.py           # Dynamic Scenic program generation
  task_reverser.py     # Task reversal (backward evaluation)
  bddl_preprocessor.py # BDDL parsing + asset substitution
  asset_registry.py    # OOD variant registry
  data/
    asset_variants.json # Single source of truth for all variants

scenic/
  libero_model.scenic           # World vocabulary (Layer 2)
  position_perturbation.scenic  # Hand-written perturbation programs (Layer 3)
  object_perturbation.scenic
  combined_perturbation.scenic
  camera_perturbation.scenic
  lighting_perturbation.scenic
  distractor_perturbation.scenic
  verifai_position.scenic

tests/
  test_scenic.py   # Tier 1: Scenic-only
  test_libero.py   # Tier 2: Simulation integration
  test_gym.py      # Tier 3: Gym wrapper
```

---

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](../LICENSE).
