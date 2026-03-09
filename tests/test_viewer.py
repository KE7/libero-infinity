"""Unit tests for scripts/viewer.py without opening a real MuJoCo window."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VIEWER_PATH = REPO_ROOT / "scripts" / "viewer.py"


def _load_viewer_module():
    spec = importlib.util.spec_from_file_location("libero_infinity_viewer", VIEWER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _DummySession:
    def __init__(self, *_args, **_kwargs):
        self.closed = False

    def close(self):
        self.closed = True


def _install_fake_mujoco(
    monkeypatch, *, supports_handle_reload: bool, supports_sim_reload: bool = False
):
    mujoco_mod = types.ModuleType("mujoco")
    viewer_mod = types.ModuleType("mujoco.viewer")
    viewer_mod.Handle = (
        type("Handle", (), {"load": lambda self, *_args: None})
        if supports_handle_reload
        else type("Handle", (), {})
    )
    if supports_sim_reload:
        viewer_mod._Simulate = type("Simulate", (), {"load": lambda self, *_args: None})
    mujoco_mod.viewer = viewer_mod
    monkeypatch.setitem(sys.modules, "mujoco", mujoco_mod)
    monkeypatch.setitem(sys.modules, "mujoco.viewer", viewer_mod)


def test_main_uses_restart_loop_when_handle_reload_is_unavailable(monkeypatch, tmp_path):
    viewer = _load_viewer_module()
    calls = []

    _install_fake_mujoco(monkeypatch, supports_handle_reload=False)
    monkeypatch.setattr(viewer, "_require_viewer", lambda: None)
    monkeypatch.setattr(viewer, "_resolve_bddl", lambda *_args: tmp_path / "task.bddl")
    monkeypatch.setattr(viewer, "SceneSession", _DummySession)
    monkeypatch.setattr(viewer, "run_viewer", lambda *_args: calls.append("single"))
    monkeypatch.setattr(
        viewer,
        "run_viewer_restart_on_resample",
        lambda *_args: calls.append("restart"),
    )

    viewer.main(["--bddl", "task", "--suite", "libero_spatial"])

    assert calls == ["restart"]


def test_main_prefers_single_window_reload_when_supported(monkeypatch, tmp_path):
    viewer = _load_viewer_module()
    calls = []

    _install_fake_mujoco(monkeypatch, supports_handle_reload=True)
    monkeypatch.setattr(viewer, "_require_viewer", lambda: None)
    monkeypatch.setattr(viewer, "_resolve_bddl", lambda *_args: tmp_path / "task.bddl")
    monkeypatch.setattr(viewer, "SceneSession", _DummySession)
    monkeypatch.setattr(viewer, "run_viewer", lambda *_args: calls.append("single"))
    monkeypatch.setattr(
        viewer,
        "run_viewer_restart_on_resample",
        lambda *_args: calls.append("restart"),
    )

    viewer.main(["--bddl", "task", "--suite", "libero_spatial"])

    assert calls == ["single"]


def test_main_prefers_single_window_reload_when_simulate_reload_is_available(monkeypatch, tmp_path):
    viewer = _load_viewer_module()
    calls = []

    _install_fake_mujoco(
        monkeypatch,
        supports_handle_reload=False,
        supports_sim_reload=True,
    )
    monkeypatch.setattr(viewer, "_require_viewer", lambda: None)
    monkeypatch.setattr(viewer, "_resolve_bddl", lambda *_args: tmp_path / "task.bddl")
    monkeypatch.setattr(viewer, "SceneSession", _DummySession)
    monkeypatch.setattr(viewer, "run_viewer", lambda *_args: calls.append("single"))
    monkeypatch.setattr(
        viewer,
        "run_viewer_restart_on_resample",
        lambda *_args: calls.append("restart"),
    )

    viewer.main(["--bddl", "task", "--suite", "libero_spatial"])

    assert calls == ["single"]


def test_reload_viewer_handle_uses_hidden_simulate_reload():
    viewer = _load_viewer_module()
    calls = []

    class _Sim:
        def load(self, *args):
            calls.append(args)

    sim = _Sim()

    class _Handle:
        def _sim(self):
            return sim

    viewer._reload_viewer_handle(_Handle(), "model", "data")

    assert calls == [("model", "data", "")]


def test_resolve_scenic_shorthand_uses_repo_scenic_dir():
    viewer = _load_viewer_module()

    path = viewer._resolve_scenic("position_perturbation")

    assert path == (viewer._SCENIC_ROOT / "position_perturbation.scenic").resolve()


def test_main_passes_explicit_scenic_path_to_session(monkeypatch, tmp_path):
    viewer = _load_viewer_module()
    calls = []
    scenic_path = tmp_path / "custom.scenic"
    scenic_path.write_text("model libero_model\n")

    class _RecordingSession:
        def __init__(self, bddl_path, perturbation, scenic_path=None):
            calls.append((bddl_path, perturbation, scenic_path))

        def close(self):
            pass

    _install_fake_mujoco(monkeypatch, supports_handle_reload=True)
    monkeypatch.setattr(viewer, "_require_viewer", lambda: None)
    monkeypatch.setattr(viewer, "_resolve_bddl", lambda *_args: tmp_path / "task.bddl")
    monkeypatch.setattr(viewer, "SceneSession", _RecordingSession)
    monkeypatch.setattr(viewer, "run_viewer", lambda *_args: None)
    monkeypatch.setattr(viewer, "run_viewer_restart_on_resample", lambda *_args: None)

    viewer.main(["--bddl", "task", "--suite", "libero_spatial", "--scenic", str(scenic_path)])

    assert calls == [(tmp_path / "task.bddl", "position", scenic_path.resolve())]


def test_main_warns_when_scenic_and_perturbation_are_both_provided(monkeypatch, tmp_path, capsys):
    viewer = _load_viewer_module()
    scenic_path = tmp_path / "custom.scenic"
    scenic_path.write_text("model libero_model\n")

    _install_fake_mujoco(monkeypatch, supports_handle_reload=True)
    monkeypatch.setattr(viewer, "_require_viewer", lambda: None)
    monkeypatch.setattr(viewer, "_resolve_bddl", lambda *_args: tmp_path / "task.bddl")
    monkeypatch.setattr(viewer, "SceneSession", _DummySession)
    monkeypatch.setattr(viewer, "run_viewer", lambda *_args: None)
    monkeypatch.setattr(viewer, "run_viewer_restart_on_resample", lambda *_args: None)

    viewer.main(
        [
            "--bddl",
            "task",
            "--suite",
            "libero_spatial",
            "--scenic",
            str(scenic_path),
            "--perturbation",
            "object",
        ]
    )

    captured = capsys.readouterr()
    assert "--perturbation is ignored when --scenic is provided" in captured.err
