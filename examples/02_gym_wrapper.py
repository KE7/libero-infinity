"""Standard Gym API usage with LIBEROScenicEnv.

Demonstrates:
  - Creating a LIBEROScenicEnv and stepping through a full episode
  - Reading observation keys: RGB images, proprioception, object state
  - Accessing Scenic scene parameters from the info dict (via the scene stored
    on the simulation object)
  - Compatibility notes for stable-baselines3 and other RL libraries

Run from the repo root:

    MUJOCO_GL=egl python examples/02_gym_wrapper.py

MuJoCo rendering backends:
  MUJOCO_GL=egl     — EGL (default on headless Linux servers)
  MUJOCO_GL=osmesa  — software renderer (fallback if EGL is unavailable)
  (unset)           — macOS / desktop Linux with a display
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(os.environ.get("LIBERO_ROOT", pathlib.Path(__file__).parent.parent))
_BDDL_PATH = (
    _REPO_ROOT
    / "src/libero_infinity/data/libero_runtime/bddl_files/libero_goal"
    / "put_the_bowl_on_the_plate.bddl"
)


def _check_deps() -> None:
    """Exit gracefully if required packages are missing."""
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print(
            "ERROR: mujoco is not installed.\n"
            "Install it with:  uv pip install mujoco\n"
            "Then re-run:      MUJOCO_GL=egl python examples/02_gym_wrapper.py"
        )
        sys.exit(1)

    if not _BDDL_PATH.exists():
        print(f"ERROR: BDDL file not found: {_BDDL_PATH}")
        print("Run from the libero-infinity repo root or set LIBERO_ROOT.")
        sys.exit(1)


def print_obs_summary(obs: dict, label: str = "obs") -> None:
    """Pretty-print the keys and shapes in an observation dict."""
    print(f"\n{label} keys ({len(obs)} total):")

    # Group by category for readability
    image_keys = [k for k in obs if k.endswith("_image")]
    proprio_keys = [k for k in obs if k.startswith("robot0_") and not k.endswith("_image")]
    object_keys = [k for k in obs if not k.startswith("robot0_") and not k.endswith("_image")]

    if image_keys:
        print("  [visual]")
        for k in sorted(image_keys):
            v = obs[k]
            print(f"    {k:45s} shape={v.shape}  dtype={v.dtype}")

    if proprio_keys:
        print("  [proprioception]")
        for k in sorted(proprio_keys):
            v = obs[k]
            print(f"    {k:45s} shape={v.shape}  dtype={v.dtype}")

    if object_keys:
        print("  [object state]")
        for k in sorted(object_keys):
            v = obs[k]
            print(f"    {k:45s} shape={v.shape}  dtype={v.dtype}")


def main() -> None:
    _check_deps()

    print("=== LIBERO-Infinity: Gym Wrapper Example ===\n")
    print(f"Task BDDL : {_BDDL_PATH.name}")
    print("Perturbation: combined  (position + object identity)\n")

    # ------------------------------------------------------------------
    # Step 1: Create the environment
    # ------------------------------------------------------------------
    # LIBEROScenicEnv is a standard gym.Env. The constructor compiles the
    # Scenic scenario once (takes ~2-5 s) and stores it for reuse across
    # all calls to reset().
    #
    # Key parameters:
    #   bddl_path    — path to the BDDL task file
    #   perturbation — which axes to randomise on every reset()
    #   resolution   — camera image size in pixels (H = W = resolution)
    #   max_steps    — episode horizon (done=True after this many steps)
    from libero_infinity.gym_env import LIBEROScenicEnv

    env = LIBEROScenicEnv(
        bddl_path=str(_BDDL_PATH),
        perturbation="combined",   # position + object identity
        resolution=128,            # 128×128 RGB images
        max_steps=50,              # short horizon for this demo
        seed=0,
    )

    print(f"Action space : {env.action_space}")
    # Observation space is populated lazily on first reset() — it will be
    # printed after the call below.

    # ------------------------------------------------------------------
    # Step 2: Reset — sample a new perturbed scene
    # ------------------------------------------------------------------
    # reset() samples one scene from the Scenic program, injects the sampled
    # poses into MuJoCo, runs physics for a few settling steps, and returns
    # the initial observation dict.
    print("\nCalling env.reset() …")
    obs = env.reset()

    print(f"\nObservation space (built from first obs): {env.observation_space}")
    print_obs_summary(obs, label="Initial obs")

    # ------------------------------------------------------------------
    # Step 3: Read specific observation keys
    # ------------------------------------------------------------------
    # Visual observations — shape (H, W, 3), dtype uint8
    # Images follow OpenGL convention (origin bottom-left).
    # Flip with frame[::-1] before displaying or feeding to a vision model.
    agentview = obs["agentview_image"]        # third-person camera
    wrist_cam = obs["robot0_eye_in_hand_image"]  # wrist-mounted camera
    print(f"\nAgentview image : shape={agentview.shape}  dtype={agentview.dtype}")
    print(f"Wrist camera    : shape={wrist_cam.shape}  dtype={wrist_cam.dtype}")
    print("(Flip vertically with frame[::-1] for standard top-left-origin display)")

    # Proprioceptive state — always present, independent of task
    eef_pos = obs["robot0_eef_pos"]          # (3,) end-effector position
    proprio = obs["robot0_proprio-state"]    # (39,) concatenated proprio
    print(f"\nEnd-effector position  : {eef_pos}")
    print(f"Concatenated proprio   : shape={proprio.shape}")

    # Task instruction is NOT in obs — retrieve it from TaskConfig
    from libero_infinity.task_config import TaskConfig
    cfg = TaskConfig.from_bddl(str(_BDDL_PATH))
    print(f"\nTask instruction (from TaskConfig): \"{cfg.language}\"")
    print("Pass this string to your VLA as the language conditioning input.")

    # ------------------------------------------------------------------
    # Step 4: Run a short episode with a zero-action policy
    # ------------------------------------------------------------------
    print("\n--- Running 50-step episode (zero-action policy) ---")
    total_reward = 0.0
    for step in range(50):
        # Zero action: robot stays still.  Replace with your policy here.
        action = np.zeros(7, dtype=np.float32)

        # step() returns the gym 0.25 4-tuple:
        #   obs    — next observation dict
        #   reward — 1.0 on task completion, 0.0 otherwise
        #   done   — True when episode ends (success or horizon)
        #   info   — {"success": bool, "steps": int}
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if step % 10 == 0 or done:
            print(
                f"  step={step:3d}  reward={reward:.1f}  "
                f"done={done}  success={info['success']}"
            )

        if done:
            break

    print(f"\nEpisode finished.  total_reward={total_reward:.1f}")

    # ------------------------------------------------------------------
    # Step 5: Second reset — shows a different perturbed scene
    # ------------------------------------------------------------------
    print("\n--- Resetting for a new scene (perturbation resampled) ---")
    obs2 = env.reset()

    # The object positions in obs2 will differ from obs because each reset()
    # independently samples a new scene from the Scenic distribution.
    eef_pos2 = obs2["robot0_eef_pos"]
    print(f"New EEF position after reset: {eef_pos2}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    env.close()
    print("\nenv.close() — simulation destroyed, temp files cleaned up.")

    # ------------------------------------------------------------------
    # Stable-Baselines3 compatibility note
    # ------------------------------------------------------------------
    print(
        "\n--- stable-baselines3 / standard RL library compatibility ---\n"
        "LIBEROScenicEnv follows the gym 0.25 API (4-tuple step returns).\n"
        "Use make_vec_env() for vectorised rollouts:\n\n"
        "    from libero_infinity.gym_env import make_vec_env\n"
        "    from stable_baselines3 import PPO\n\n"
        "    vec_env = make_vec_env(\n"
        f"        bddl_path=\"{_BDDL_PATH}\",\n"
        "        n_envs=4,\n"
        "        perturbation=\"combined\",\n"
        "        use_subprocess=True,  # AsyncVectorEnv for true parallelism\n"
        "    )\n"
        "    model = PPO(\"MultiInputPolicy\", vec_env, verbose=1)\n"
        "    model.learn(total_timesteps=100_000)"
    )


if __name__ == "__main__":
    main()
