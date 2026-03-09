"""Tier 3 — Gym wrapper tests (require LIBERO).

Tests for LIBEROScenicEnv(gym.Env) and make_vec_env().
"""

import numpy as np
from conftest import BOWL_BDDL, requires_libero


@requires_libero
class TestGymWrapper:
    """Tests for the LIBEROScenicEnv gym wrapper."""

    BDDL = str(BOWL_BDDL)

    def test_gym_env_creates(self):
        from libero_infinity.gym_env import LIBEROScenicEnv

        env = LIBEROScenicEnv(
            bddl_path=self.BDDL,
            perturbation="position",
            max_steps=5,
        )
        assert env.action_space.shape == (7,)
        env.close()

    def test_gym_reset_returns_obs(self):
        from libero_infinity.gym_env import LIBEROScenicEnv

        env = LIBEROScenicEnv(
            bddl_path=self.BDDL,
            perturbation="position",
            max_steps=5,
        )
        obs = env.reset()
        assert isinstance(obs, dict)
        assert "agentview_image" in obs
        assert "robot0_joint_pos" in obs
        assert obs["agentview_image"].dtype == np.uint8
        env.close()

    def test_gym_step_returns_4_tuple(self):
        from libero_infinity.gym_env import LIBEROScenicEnv

        env = LIBEROScenicEnv(
            bddl_path=self.BDDL,
            perturbation="position",
            max_steps=5,
        )
        env.reset()
        obs, reward, done, info = env.step(np.zeros(7))
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert not done  # first step should not be terminal
        assert "success" in info
        assert "steps" in info
        env.close()

    def test_gym_render_rgb_array(self):
        from libero_infinity.gym_env import LIBEROScenicEnv

        env = LIBEROScenicEnv(
            bddl_path=self.BDDL,
            perturbation="position",
            resolution=64,
            max_steps=5,
        )
        env.reset()
        frame = env.render(mode="rgb_array")
        assert frame is not None
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint8
        env.close()

    def test_gym_double_reset(self):
        from libero_infinity.gym_env import LIBEROScenicEnv

        env = LIBEROScenicEnv(
            bddl_path=self.BDDL,
            perturbation="position",
            max_steps=5,
        )
        obs1 = env.reset()
        pos1 = obs1["akita_black_bowl_1_pos"].copy()

        obs2 = env.reset()
        pos2 = obs2["akita_black_bowl_1_pos"].copy()

        assert not np.allclose(pos1, pos2, atol=1e-6), f"Two resets produced identical positions: {pos1}"  # fmt: skip  # noqa: E501
        env.close()

    def test_gym_horizon_terminates(self):
        from libero_infinity.gym_env import LIBEROScenicEnv

        env = LIBEROScenicEnv(
            bddl_path=self.BDDL,
            perturbation="position",
            max_steps=3,
        )
        env.reset()
        for i in range(3):
            obs, reward, done, info = env.step(np.zeros(7))
        assert done is True
        assert info["steps"] == 3
        env.close()

    def test_gym_reversed_task(self):
        from libero_infinity.gym_env import LIBEROScenicEnv

        env = LIBEROScenicEnv(
            bddl_path=self.BDDL,
            perturbation="position",
            reverse=True,
            max_steps=5,
        )
        obs = env.reset()
        assert isinstance(obs, dict)
        assert "agentview_image" in obs
        obs, reward, done, info = env.step(np.zeros(7))
        assert isinstance(obs, dict)
        env.close()

    def test_gym_obs_space_built(self):
        from libero_infinity.gym_env import LIBEROScenicEnv

        env = LIBEROScenicEnv(
            bddl_path=self.BDDL,
            perturbation="position",
            resolution=64,
            max_steps=5,
        )
        assert len(env.observation_space.spaces) == 0

        env.reset()

        assert len(env.observation_space.spaces) > 0
        assert "agentview_image" in env.observation_space.spaces
        img_space = env.observation_space.spaces["agentview_image"]
        assert img_space.shape == (64, 64, 3)
        env.close()

    def test_gym_context_manager(self):
        from libero_infinity.gym_env import LIBEROScenicEnv

        with LIBEROScenicEnv(
            bddl_path=self.BDDL,
            perturbation="position",
            max_steps=5,
        ) as env:
            obs = env.reset()
            assert isinstance(obs, dict)
