#!/usr/bin/env python3
"""Generate worst-case perturbation gallery for README.

For each perturbation axis, we read the Scenic program, identify the extreme
end of the sampling distribution (worst case), then render that scene directly.
This makes the differences visually obvious to humans reading the README.

Run from project root:
    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python scripts/generate_gallery.py
"""
import os
import pathlib
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from libero_infinity.runtime import get_bddl_dir

ROOT = pathlib.Path(__file__).parent.parent
BDDL = str(get_bddl_dir() / "libero_goal" / "put_the_bowl_on_the_plate.bddl")
OUT = ROOT / "assets/perturbation_gallery.png"
RES = 320  # render resolution per panel
_ZERO_ACTION: np.ndarray | None = None


# ── helpers ──────────────────────────────────────────────────────────────────

def make_env(bddl_path: str):
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        camera_names=["agentview"],
        camera_heights=RES,
        camera_widths=RES,
        control_freq=20,
        horizon=200,
        ignore_done=False,
        hard_reset=True,
    )
    env.reset()
    return env


def get_frame(env) -> np.ndarray:
    """Render the current sim state via env.step(zero).

    _get_observations() returns the cached result from the last env.step() or
    env.reset() call — it does NOT re-render.  Driving one zero-action step
    flushes the physics state through the full render pipeline and returns a
    fresh image.
    """
    global _ZERO_ACTION
    if _ZERO_ACTION is None or _ZERO_ACTION.shape[0] != env.env.action_spec[0].shape[0]:
        _ZERO_ACTION = np.zeros(env.env.action_spec[0].shape[0])
    obs, _, _, _ = env.step(_ZERO_ACTION)
    return obs["agentview_image"][::-1].copy()


def settle(env, steps: int = 40):
    """Zero velocities and run physics steps so injected objects come to rest."""
    import mujoco
    m = env.env.sim.model._model
    d = env.env.sim.data._data
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(steps):
        mujoco.mj_step(m, d)
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def set_joint_xy(env, joint_name: str, x: float, y: float):
    """Teleport object to (x, y) while keeping original z and orientation."""
    sim_data = env.env.sim.data
    try:
        qpos = sim_data.get_joint_qpos(joint_name).copy()
        qpos[0] = x
        qpos[1] = y
        sim_data.set_joint_qpos(joint_name, qpos)
    except Exception as e:
        print(f"    WARNING: could not set {joint_name}: {e}")


def mj_forward(env):
    import mujoco
    mujoco.mj_forward(env.env.sim.model._model, env.env.sim.data._data)


def step_model_change(env):
    """After directly modifying model params (camera, lights), call mj_forward
    then one env.step so the render pipeline sees the updated model state."""
    mj_forward(env)
    # We don't need the return value; get_frame() will call step() again
    # and capture the rendered obs from that step.  But the model params set
    # above (cam_pos, cam_quat, light params) persist across steps because they
    # live on the *model* not the *data*, so they'll be visible in get_frame().
    pass


# ── panel renderers ───────────────────────────────────────────────────────────

def panel_default() -> np.ndarray:
    """Canonical scene — standard env.reset(), no perturbation."""
    env = make_env(BDDL)
    img = get_frame(env)
    env.close()
    return img


def panel_position() -> np.ndarray:
    """Worst-case position: objects at far corners of workspace.

    From position_perturbation.scenic:
        bowl  in SAFE_REGION  (x∈[-0.35, 0.35], y∈[-0.25, 0.25])
        plate in PLATE_SAFE_REGION
        require[0.8] distance from bowl to bowl_train_pt > 0.15  (OOD bias)

    Training positions: bowl≈(0.12,-0.05), plate stays canonical.
    Worst-case: push only the bowl (the task object) to the opposite corner.
    """
    env = make_env(BDDL)
    # Move only the bowl — the target object — to the far opposite corner
    set_joint_xy(env, "akita_black_bowl_1_joint0", x=-0.30, y=0.22)
    settle(env)
    img = get_frame(env)
    env.close()
    return img


def panel_object() -> np.ndarray:
    """Worst-case object: most visually different variant.

    From object_perturbation.scenic:
        chosen_asset = Uniform(*ASSET_VARIANTS["akita_black_bowl"])
        # → ["akita_black_bowl", "white_bowl", "glazed_rim_porcelain_ramekin"]

    Worst-case: glazed_rim_porcelain_ramekin — completely different shape & color.
    """
    from libero_infinity.bddl_preprocessor import substitute_asset
    bddl_text = pathlib.Path(BDDL).read_text()
    patched = substitute_asset(bddl_text, "akita_black_bowl", "glazed_rim_porcelain_ramekin")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bddl", delete=False, prefix="libero_gal_") as f:
        f.write(patched)
        tmp = f.name
    try:
        env = make_env(tmp)
        img = get_frame(env)
        env.close()
    finally:
        os.unlink(tmp)
    return img


def panel_combined() -> np.ndarray:
    """Worst-case combined: far-corner positions + different object class.

    From combined_perturbation.scenic:
        chosen_asset = Uniform(*_variants)                    # object axis
        bowl = new LIBEROObject ... in SAFE_REGION             # position axis
        require (distance from bowl to plate) > _min_clearance
    """
    from libero_infinity.bddl_preprocessor import substitute_asset
    bddl_text = pathlib.Path(BDDL).read_text()
    patched = substitute_asset(bddl_text, "akita_black_bowl", "glazed_rim_porcelain_ramekin")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bddl", delete=False, prefix="libero_gal_") as f:
        f.write(patched)
        tmp = f.name
    try:
        env = make_env(tmp)
        set_joint_xy(env, "akita_black_bowl_1_joint0", x=-0.30, y=0.22)
        set_joint_xy(env, "plate_1_joint0",            x=0.28,  y=-0.20)
        settle(env)
        img = get_frame(env)
        env.close()
    finally:
        os.unlink(tmp)
    return img


def panel_camera() -> np.ndarray:
    """Worst-case camera: all offsets and tilt at maximum magnitude.

    From camera_perturbation.scenic:
        param camera_x_range = 0.10     → Range(-0.10, 0.10)
        param camera_z_range = 0.08     → Range(-0.08, 0.08)
        param camera_tilt_range = 15.0  → Range(-15.0, 15.0)

    Worst-case: dx=+0.10, dy=+0.10, dz=-0.08 (lower), tilt=+15°.
    """
    from scipy.spatial.transform import Rotation as R
    env = make_env(BDDL)
    sim = env.env.sim
    cam_id = sim.model.camera_name2id("agentview")

    # Max position offsets
    sim.model.cam_pos[cam_id][0] += 0.10   # forward
    sim.model.cam_pos[cam_id][1] += 0.10   # left
    sim.model.cam_pos[cam_id][2] -= 0.08   # lower → see scene from below-ish

    # Max tilt: +15° around camera local x-axis
    # MuJoCo stores camera quaternion as (w, x, y, z)
    cq = sim.model.cam_quat[cam_id].copy()
    r_current = R.from_quat([cq[1], cq[2], cq[3], cq[0]])  # (x,y,z,w)
    r_tilt = R.from_euler("x", 15.0, degrees=True)
    r_new = r_current * r_tilt
    q = r_new.as_quat()  # (x,y,z,w)
    sim.model.cam_quat[cam_id] = [q[3], q[0], q[1], q[2]]  # back to (w,x,y,z)

    img = get_frame(env)
    env.close()
    return img


def panel_lighting() -> np.ndarray:
    """Worst-case lighting: minimum intensity and ambient (very dark scene).

    From lighting_perturbation.scenic:
        param intensity_min = 0.4   → Range(0.4, 2.0)
        param ambient_min   = 0.05  → Range(0.05, 0.6)

    Worst-case for a policy: darkest setting — objects barely visible.
    """
    env = make_env(BDDL)
    sim = env.env.sim
    # Multiply all light diffuse/specular by 0.4 (minimum intensity)
    for i in range(sim.model.nlight):
        sim.model.light_diffuse[i]  *= 0.4
        sim.model.light_specular[i] *= 0.4
    # Near-zero ambient: scene barely lit by indirect light
    sim.model.vis.headlight.ambient[:] = 0.05
    img = get_frame(env)
    env.close()
    return img


# ── panel metadata ────────────────────────────────────────────────────────────

PANELS = [
    # (render_fn, bold_title, scenic_code_line, title_color)
    (
        panel_default,
        "Default (canonical)",
        "env.reset()  # standard LIBERO initial state",
        "#444444",
    ),
    (
        panel_position,
        "Position Perturbation",
        "bowl = new LIBEROObject in SAFE_REGION",
        "#1976D2",
    ),
    (
        panel_object,
        "Object Perturbation",
        'Uniform(*ASSET_VARIANTS["akita_black_bowl"])',
        "#E65100",
    ),
    (
        panel_combined,
        "Combined (Pos + Obj)",
        "bowl in SAFE_REGION, chosen_asset = Uniform(*_variants)",
        "#C62828",
    ),
    (
        panel_camera,
        "Camera Perturbation",
        "param camera_tilt = Range(-15.0, 15.0)",
        "#B71C1C",
    ),
    (
        panel_lighting,
        "Lighting Perturbation",
        "param light_intensity = Range(0.4, 2.0)",
        "#6A1B9A",
    ),
]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating worst-case perturbation gallery...")

    images = []
    for render_fn, title, code, color in PANELS:
        print(f"  [{title}] ...")
        try:
            img = render_fn()
            images.append(img)
            print(f"    OK — shape {img.shape}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    FAILED — using placeholder")
            images.append(np.zeros((RES, RES, 3), dtype=np.uint8))

    # ── compose 3×2 gallery ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle("Libero-∞  Perturbation Axes", fontsize=15, fontweight="bold", y=0.995)

    for ax, img, (_, title, code, color) in zip(axes.flat, images, PANELS):
        ax.imshow(img)
        ax.axis("off")
        # Line 1: bold colored title
        ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=5)
        # Line 2: monospace Scenic code below the image
        ax.text(
            0.5, -0.02,
            code,
            transform=ax.transAxes,
            fontsize=7.5,
            ha="center",
            va="top",
            fontfamily="monospace",
            color="#555555",
        )

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.subplots_adjust(hspace=0.25)
    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(str(OUT), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
