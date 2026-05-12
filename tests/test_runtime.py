from __future__ import annotations

from types import SimpleNamespace

from libero_infinity import runtime


def test_get_installed_libero_package_root_returns_none_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: None)
    assert runtime.get_installed_libero_package_root() is None


def test_get_installed_libero_assets_link_uses_package_root(monkeypatch, tmp_path) -> None:
    package_root = tmp_path / "site-packages" / "libero"
    package_root.mkdir(parents=True)
    spec = SimpleNamespace(submodule_search_locations=[str(package_root)])
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: spec)

    assets_link = runtime.get_installed_libero_assets_link()

    assert assets_link == package_root / "libero" / "assets"


def test_refresh_libero_assets_link_targets_installed_package(monkeypatch, tmp_path) -> None:
    assets_dir = tmp_path / "cache_assets"
    assets_dir.mkdir()
    link_path = tmp_path / "site-packages" / "libero" / "libero" / "assets"
    link_path.parent.mkdir(parents=True)
    monkeypatch.setattr(runtime, "get_installed_libero_assets_link", lambda: link_path)

    runtime._refresh_libero_assets_link(assets_dir)

    assert link_path.is_symlink()
    assert link_path.resolve() == assets_dir.resolve()
