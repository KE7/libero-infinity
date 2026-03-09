"""Template for plugging in a custom VLA (Vision-Language-Action model).

Demonstrates:
  - The policy interface expected by evaluate() and LIBEROScenicEnv
  - A MockVLA class that mimics the robo-eval /predict HTTP contract
  - A full evaluation loop using the mock model
  - Clear TODOs showing exactly where to swap in a real VLA

The robo-eval /predict contract (HTTP POST):
  Request  JSON: {"image": <base64-encoded RGB JPEG>, "instruction": "<str>"}
  Response JSON: {"action": [a0, a1, a2, a3, a4, a5, a6]}
  Action format: 7D float in [-1, 1] — [dx, dy, dz, droll, dpitch, dyaw, gripper]

Run from the repo root:

    MUJOCO_GL=egl python examples/04_custom_vla.py

MuJoCo rendering backends:
  MUJOCO_GL=egl     — EGL (default on headless Linux servers)
  MUJOCO_GL=osmesa  — software renderer (fallback if EGL is unavailable)
  (unset)           — macOS / desktop Linux with a display
"""

from __future__ import annotations

import base64
import os
import pathlib
import sys
from typing import Callable

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


# ---------------------------------------------------------------------------
# MockVLA — mimics the robo-eval /predict interface
# ---------------------------------------------------------------------------
# The robo-eval /predict contract:
#
#   POST http://<host>/predict
#   Content-Type: application/json
#
#   Request body:
#     {
#       "image":       "<base64-encoded RGB JPEG string>",
#       "instruction": "put the bowl on the plate"
#     }
#
#   Response body:
#     {
#       "action": [dx, dy, dz, droll, dpitch, dyaw, gripper]
#                  // 7 floats in [-1, 1]
#     }
#
# The MockVLA below replicates this interface in-process so the example can
# run without a real model server.  Replace the body of predict() with an
# actual HTTP call (or direct model inference) to use your real VLA.


class MockVLA:
    """In-process mock that exposes the robo-eval /predict interface.

    Replace this class (or its predict() method) with a real VLA client.
    See the TODOs below for the exact substitution points.
    """

    def __init__(self, instruction: str) -> None:
        """
        TODO (1/3): Replace with real model initialisation.

        For a remote robo-eval server:
            self.endpoint = "http://localhost:8080/predict"
            self.instruction = instruction

        For a local HuggingFace model:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                "openvla/openvla-7b-finetuned-libero-goal", trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b-finetuned-libero-goal",
                torch_dtype="auto", device_map="auto", trust_remote_code=True,
            )
        """
        self.instruction = instruction
        print(f"  [MockVLA] Initialised with instruction: \"{instruction}\"")

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run inference and return a 7D action.

        Parameters
        ----------
        image_rgb : np.ndarray
            RGB image of shape (H, W, 3), dtype uint8, origin top-left.
            (Flip from OpenGL convention with image[::-1] before passing here.)

        Returns
        -------
        np.ndarray
            Shape (7,), values in [-1, 1]:
            [dx, dy, dz, droll, dpitch, dyaw, gripper]

        TODO (2/3): Replace this method body with real inference.

        For a remote robo-eval server:
            import requests, io
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(image_rgb).save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            resp = requests.post(
                self.endpoint,
                json={"image": b64, "instruction": self.instruction},
                timeout=5.0,
            )
            resp.raise_for_status()
            return np.array(resp.json()["action"], dtype=np.float64)

        For a local HuggingFace OpenVLA model:
            from PIL import Image
            inputs = self.processor(
                Image.fromarray(image_rgb), self.instruction
            ).to(self.model.device)
            action = self.model.predict_action(**inputs)
            return np.array(action, dtype=np.float64)
        """
        # --- Mock: return a small constant action (robot gently moves forward) ---
        action = np.zeros(7, dtype=np.float64)
        action[0] = 0.05   # small +x translation
        action[6] = 1.0    # gripper closed
        return action

    @staticmethod
    def encode_image(image_bgr_or_rgb: np.ndarray) -> str:
        """Utility: encode a numpy image to a base64 JPEG string.

        Useful if your model server expects the robo-eval /predict format.
        """
        try:
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(image_bgr_or_rgb).save(buf, format="JPEG", quality=95)
            return base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            raise ImportError(
                "Pillow is required for image encoding: uv pip install pillow"
            )


# ---------------------------------------------------------------------------
# Policy factory — wraps the VLA to match the libero-infinity policy signature
# ---------------------------------------------------------------------------

def make_vla_policy(vla: MockVLA) -> Callable[[dict], np.ndarray]:
    """Wrap a VLA into the policy signature expected by evaluate().

    The policy callable must have signature:
        policy(obs: dict[str, np.ndarray]) -> np.ndarray  # shape (7,)

    This wrapper:
      1. Extracts the agentview RGB image from obs
      2. Flips it from OpenGL (bottom-left origin) to standard (top-left) convention
      3. Calls vla.predict() and returns the 7D action

    TODO (3/3): If your VLA also needs proprioception, add it here.
    """
    def policy(obs: dict) -> np.ndarray:
        # agentview_image is (H, W, 3) uint8 with origin bottom-left (OpenGL).
        # Flip vertically before feeding to vision models.
        image = obs["agentview_image"][::-1]  # → standard top-left origin

        # Optional: also pass proprioceptive state to the model
        # proprio = obs["robot0_proprio-state"]  # shape (39,)

        return vla.predict(image)

    return policy


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def _check_deps() -> None:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print(
            "ERROR: mujoco is not installed.\n"
            "Install it with:  uv pip install mujoco\n"
            "Then re-run:      MUJOCO_GL=egl python examples/04_custom_vla.py"
        )
        sys.exit(1)

    if not _BDDL_PATH.exists():
        print(f"ERROR: BDDL file not found: {_BDDL_PATH}")
        print("Run from the libero-infinity repo root or set LIBERO_ROOT.")
        sys.exit(1)


def main() -> None:
    _check_deps()

    print("=== LIBERO-Infinity: Custom VLA Template ===\n")
    print(f"Task BDDL : {_BDDL_PATH.name}")
    print("Perturbation: combined  (position + object identity)")
    print("Policy      : MockVLA   (replace with your real model)\n")

    # --- Step 1: Get the task instruction ---
    # VLA models need the natural language instruction as input.
    # TaskConfig parses the BDDL to extract it.
    from libero_infinity.task_config import TaskConfig

    cfg = TaskConfig.from_bddl(str(_BDDL_PATH))
    instruction = cfg.language
    print(f"Task instruction: \"{instruction}\"\n")

    # --- Step 2: Initialise the VLA ---
    # TODO: Replace MockVLA with your real model class.
    vla = MockVLA(instruction=instruction)

    # --- Step 3: Wrap the VLA as a policy callable ---
    policy = make_vla_policy(vla)

    # --- Step 4: Auto-generate a Scenic program and run evaluate() ---
    from libero_infinity.compiler import generate_scenic_file
    from libero_infinity.eval import evaluate

    scenic_path = generate_scenic_file(cfg, perturbation="combined")
    print(f"Generated Scenic program: {scenic_path}\n")

    print("Running 3 evaluation episodes …\n")
    results = evaluate(
        scenic_path=scenic_path,
        bddl_path=str(_BDDL_PATH),
        policy=policy,
        n_scenes=3,
        max_steps=50,      # short horizon for the demo
        verbose=True,      # print per-episode progress
        seed=7,
    )

    print(f"\n{results.summary()}")
    print(
        "\nNote: The MockVLA just moves the arm slightly forward, so 0% success\n"
        "is expected.  Replace MockVLA with your real model to measure actual\n"
        "performance.\n"
    )

    # --- Step 5: Inspect per-episode data ---
    print("Per-episode Scenic params (sampled scene diversity):")
    for ep in results.episodes:
        print(f"  Episode {ep.scene_index + 1}: success={ep.success}  "
              f"params={ep.scenic_params}")

    # Cleanup
    pathlib.Path(scenic_path).unlink(missing_ok=True)

    # --- Action chunk example (e.g. pi0.5 outputs 50 actions at once) ---
    print(
        "\n--- Action chunk VLA (e.g. pi0.5 outputs 50 actions per query) ---\n"
        "\n"
        "def make_chunked_policy(vla, chunk_size=50):\n"
        "    queue = []\n"
        "    def policy(obs):\n"
        "        nonlocal queue\n"
        "        if not queue:\n"
        "            image = obs['agentview_image'][::-1]\n"
        "            # vla.predict_chunk returns shape (chunk_size, 7)\n"
        "            queue = list(vla.predict_chunk(image))\n"
        "        return queue.pop(0)\n"
        "    return policy\n"
    )


if __name__ == "__main__":
    main()
