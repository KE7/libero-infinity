from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from libero_infinity.simulator import LIBEROSimulation


class _FakeCameraModel:
    def __init__(self) -> None:
        self.cam_pos = np.array([[0.5, 0.0, 1.35]], dtype=float)
        self.cam_quat = np.array([[0.653, 0.271, 0.271, 0.653]], dtype=float)

    def camera_name2id(self, name: str) -> int:
        if name != "agentview":
            raise KeyError(name)
        return 0


def _fake_camera_sim(params: dict) -> tuple[LIBEROSimulation, _FakeCameraModel]:
    model = _FakeCameraModel()
    sim_handle = SimpleNamespace(model=model, forward=lambda: None)
    env = SimpleNamespace(sim=sim_handle)
    li_sim = LIBEROSimulation.__new__(LIBEROSimulation)
    li_sim.scene = SimpleNamespace(params=params)
    li_sim.libero_env = SimpleNamespace(env=env)
    return li_sim, model


def test_camera_consumer_applies_current_cam_params() -> None:
    li_sim, model = _fake_camera_sim(
        {
            "cam_azimuth": 10.0,
            "cam_elevation": 5.0,
            "cam_distance": 1.1,
        }
    )
    base_pos = model.cam_pos.copy()
    base_quat = model.cam_quat.copy()

    li_sim._apply_camera_perturbation()

    assert not np.allclose(model.cam_pos, base_pos)
    assert not np.allclose(model.cam_quat, base_quat)
    base_radius = np.linalg.norm(base_pos[0] - np.array([0.0, 0.0, 0.82]))
    new_radius = np.linalg.norm(model.cam_pos[0] - np.array([0.0, 0.0, 0.82]))
    np.testing.assert_allclose(new_radius, base_radius * 1.1)


def test_camera_consumer_still_applies_legacy_offset_params() -> None:
    li_sim, model = _fake_camera_sim(
        {
            "camera_x_offset": 0.1,
            "camera_y_offset": -0.2,
            "camera_z_offset": 0.3,
        }
    )

    li_sim._apply_camera_perturbation()

    np.testing.assert_allclose(model.cam_pos[0], [0.6, -0.2, 1.65])


def test_articulation_consumer_skips_state_metadata_suffix() -> None:
    calls: list[float] = []
    state = SimpleNamespace(set_joint=lambda value: calls.append(value))
    env = SimpleNamespace(object_states_dict={"wooden_cabinet_1": state})
    li_sim = LIBEROSimulation.__new__(LIBEROSimulation)
    li_sim.scene = SimpleNamespace(
        params={
            "articulation_wooden_cabinet_1": -0.15,
            "articulation_wooden_cabinet_1_state": "Open",
        }
    )
    li_sim.libero_env = SimpleNamespace(env=env)

    li_sim._apply_articulation_perturbation()

    assert calls == [-0.15]
