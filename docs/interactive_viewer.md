# Interactive MuJoCo Viewer

[Back to main README](../README.md)

`scripts/viewer.py` opens the **MuJoCo passive viewer** window for any LIBERO
task, sampling a fresh Scenic-perturbed scene on every key press.  It is the
fastest way to visually inspect what a perturbation distribution *looks like*
in practice.

---

## Examples

```bash
# Activate the virtual environment first
source .venv/bin/activate      # or: eval "$(uv venv --activate)"

# 1) Auto-generate Scenic from the BDDL (most common)
python scripts/viewer.py \
    --suite libero_spatial \
    --bddl pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate \
    --perturbation position

# 2) Auto-generate Scenic for a different perturbation axis
python scripts/viewer.py \
    --suite libero_spatial \
    --bddl pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate \
    --perturbation combined

# 3) Use a hand-written Scenic program from scenic/
python scripts/viewer.py \
    --suite libero_spatial \
    --bddl pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate \
    --scenic position_perturbation

# 4) Use an explicit Scenic file path
python scripts/viewer.py \
    --suite libero_spatial \
    --bddl pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate \
    --scenic scenic/position_perturbation.scenic

# 5) Use an explicit BDDL path instead of a task name
python scripts/viewer.py \
    --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl \
    --perturbation full
```

Once the window opens:

| Key | Action |
|-----|--------|
| **R** / **Space** / **Enter** | Sample a **new scene** from the Scenic distribution and reload the viewer |
| **Q** | Quit |

Closing the window also quits.

### Camera controls

MuJoCo's passive viewer uses the standard mouse navigation controls:

| Control | Action |
|---------|--------|
| **Left mouse drag** | Rotate / orbit the camera around the scene |
| **Right mouse drag** | Pan the camera |
| **Mouse wheel** or **middle mouse drag** | Zoom in / out |

If your mouse/trackpad does not provide a right or middle button, use your OS /
terminal emulator's secondary-click emulation.

---

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--bddl` | ✓ | Task name (stem of the `.bddl` file, no extension) **or** a full path to a `.bddl` file. |
| `--suite` | recommended | Suite to search in (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`). Required when the task name is ambiguous across suites. |
| `--perturbation` | | Perturbation axis for auto-generated Scenic (default: `position`). See [Perturbation choices](#perturbation-choices). |
| `--scenic` | | Optional Scenic program to use instead of auto-generating one from the BDDL. Accepts an explicit path or a hand-written file name from `scenic/`. |

If `--scenic` is omitted, the viewer uses an auto-generated Scenic program.
If `--scenic` is provided, the viewer loads that program directly and ignores `--perturbation` (with a warning if both are passed).

### Perturbation choices

| Value | What varies |
|-------|-------------|
| `position` | Object (x, y) placement — uniform over the workspace |
| `object` | Asset identity (mesh + texture) swapped from the variant registry |
| `distractor` | 1–3 random clutter objects added to the scene |
| `combined` | Position **+** object identity (simultaneously) |
| `camera` | Agentview camera position and tilt angle |
| `lighting` | Light intensity, position, and ambient level |
| `full` | All axes at once |

---

## Make target

```makefile
make view BDDL=<task_name> [SUITE=<suite>] [PERTURBATION=<type>]
```

All three variables are forwarded directly to `scripts/viewer.py`.  `SUITE` and
`PERTURBATION` are optional (defaults: auto-detect suite, `position`).
The make target is only for generated Scenic programs; if you want to choose a
specific hand-written Scenic file, call `scripts/viewer.py` directly with
`--scenic`.

```bash
# Generated Scenic with combined perturbation
make view \
    BDDL=pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate \
    SUITE=libero_spatial \
    PERTURBATION=combined

# Generated Scenic with an explicit BDDL path
make view \
    BDDL=src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl \
    PERTURBATION=full
```

---

## Display requirements

The MuJoCo passive viewer uses GLFW and therefore requires an active display.

| Environment | Setup |
|-------------|-------|
| **Linux desktop / VNC** | Should work out-of-the-box. If not: `export DISPLAY=:0` |
| **SSH with X11 forwarding** | `ssh -X user@host` then `export DISPLAY=localhost:10.0` |
| **Headless server** | Use headless mode instead — see below |
| **macOS** | Use `mjpython` instead of `python`: `mjpython scripts/viewer.py ...` |

### Headless alternative

If no display is available, stream rendered frames to an OpenCV window instead:

```bash
MUJOCO_GL=egl libero-eval \
    --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_spatial/<task>.bddl \
    --perturbation position \
    --n-scenes 10 \
    --watch cv2
```

---

## MuJoCo rendering backend

The `MUJOCO_GL` environment variable controls *off-screen* rendering inside the
LIBERO simulation (cameras, observations).  The passive viewer always uses GLFW
for the window regardless of `MUJOCO_GL`.

| `MUJOCO_GL` | When to use |
|-------------|-------------|
| `egl` | Headless servers with EGL (most common for evaluation) |
| `osmesa` | If EGL is not available |
| *(unset)* | macOS with native OpenGL |

```bash
# Headless server with EGL + local X display for the window
MUJOCO_GL=egl DISPLAY=:0 python scripts/viewer.py \
    --suite libero_spatial \
    --bddl pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate \
    --perturbation position
```

---

## How it works

1. **Compile** — if `--scenic` is omitted, `compiler.py` generates a
   `.scenic` program for the chosen task + perturbation; otherwise the viewer
   compiles the Scenic file you passed (~2–5 s).
2. **Sample** — on startup (and on each R/Space/Enter press), the Scenic
   rejection sampler draws a valid scene from the distribution.
3. **Inject** — `LIBEROSimulation.setup()` resets the LIBERO env, overrides
   object joint `qpos` with the sampled positions/orientations, runs 50 MuJoCo
   settling steps, and applies any camera/lighting/texture perturbations.
4. **Display** — the raw MuJoCo `(model, data)` handles are passed to
   `mujoco.viewer.launch_passive()`.  The viewer thread renders the scene;
   the main thread advances physics with zero-action steps and calls
   `handle.sync()` to push each new state to the window.
5. **Resample** — pressing R/Space/Enter sets a flag in the key callback; the
   main loop tears down the current simulation, samples a fresh scene, and
   reloads the viewer (via `handle.load()` on mujoco >= 3.x, or by relaunching
   the window on older versions).

---

## Troubleshooting

### `mujoco.viewer could not be imported`

Install mujoco 3.x:

```bash
uv add mujoco        # or: pip install mujoco
```

### `GLFW: The DISPLAY environment variable is missing`

Set `DISPLAY` before running:

```bash
export DISPLAY=:0
python scripts/viewer.py ...
```

### `EGL is not available`

Install EGL or switch to OSMesa:

```bash
sudo apt-get install libegl1-mesa-dev    # Ubuntu/Debian
# or
export MUJOCO_GL=osmesa
python scripts/viewer.py ...
```

### Window opens but is black / empty

This can happen when the settling loop fails silently.  Add `--verbose` to
`libero-eval` to diagnose, or try a different perturbation axis.

### macOS: viewer crashes immediately

Use `mjpython` as required by MuJoCo on macOS:

```bash
mjpython scripts/viewer.py --suite libero_spatial \
    --bddl pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
```
