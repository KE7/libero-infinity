"""Runtime bootstrap for LIBERO-Infinity.

Single source of truth for all runtime path resolution. Call
``ensure_runtime()`` once before any ``import libero.libero.*`` occurs.

Resolution order for assets:
  1. LIBERO_INFINITY_ASSETS_DIR env var (offline / mirrored environments)
  2. Local cache at ~/.cache/libero_infinity/assets/
  3. Download from lerobot/libero-assets@HF_REVISION (first-time setup)

BDDL and init files are repo-owned under
  src/libero_infinity/data/libero_runtime/{bddl_files,init_files}
and are always available without any network access.
"""

from __future__ import annotations

import importlib.resources
import json
import os
import pathlib
import shutil
import tempfile

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

HF_REPO_ID = "lerobot/libero-assets"
HF_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"  # full SHA, pinned after parity PASS

# Canonical parity artifact path (relative to repo root).
# runtime refuses to bootstrap HF assets if this file is absent or not PASS.
_PARITY_ARTIFACT_NAME = "asset_parity_lerobot_libero-assets_0b3ea86b.json"

# ---------------------------------------------------------------------------
# Internal paths
# ---------------------------------------------------------------------------

_CACHE_ROOT = pathlib.Path.home() / ".cache" / "libero_infinity"
_ASSETS_CACHE_DIR = _CACHE_ROOT / "assets"

# vendor/libero symlink target — LIBERO's hardcoded relative asset lookup
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_ASSETS_LINK = _REPO_ROOT / "vendor" / "libero" / "libero" / "libero" / "assets"
_PARITY_ARTIFACT = _REPO_ROOT / "audit" / _PARITY_ARTIFACT_NAME

# Required subdirectories in any valid asset tree
_REQUIRED_ASSET_DIRS = [
    "stable_hope_objects",
    "articulated_objects",
    "stable_scanned_objects",
    "textures",
    "turbosquid_objects",
]


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------


class LiberoAssetsUnavailableError(RuntimeError):
    """Raised when no asset source can be resolved."""


class LiberoAssetValidationError(RuntimeError):
    """Raised when a resolved asset tree is missing required content."""


class LiberoParityArtifactError(RuntimeError):
    """Raised when the parity artifact is missing or does not show PASS."""


# ---------------------------------------------------------------------------
# Public path helpers
# ---------------------------------------------------------------------------


def get_bddl_dir() -> pathlib.Path:
    """Return the repo-owned bddl_files directory (always available)."""
    ref = importlib.resources.files("libero_infinity.data.libero_runtime") / "bddl_files"
    return pathlib.Path(str(ref))


def get_init_dir() -> pathlib.Path:
    """Return the repo-owned init_files directory (always available)."""
    ref = importlib.resources.files("libero_infinity.data.libero_runtime") / "init_files"
    return pathlib.Path(str(ref))


def get_assets_cache_dir() -> pathlib.Path:
    """Return the canonical local asset cache path (may not exist yet)."""
    return _ASSETS_CACHE_DIR


# ---------------------------------------------------------------------------
# Public bootstrap entrypoint
# ---------------------------------------------------------------------------


def ensure_runtime(*, force_assets: bool = False) -> None:
    """Bootstrap the full LIBERO runtime environment.

    Must be called before any ``import libero.libero.*``.

    Steps (in order):
      1. Validate the committed parity artifact (fast, no network).
      2. Resolve and validate the asset tree (downloads on first run).
      3. Write ~/.libero/config.yaml — must happen before any libero import.
      4. Create/update the vendor/libero assets symlink.
      5. Set up robosuite macros_private.py.

    Args:
        force_assets: Re-download assets even if the cache is already valid.
    """
    # Step 1 — enforce the parity gate before any download occurs.
    _check_parity_artifact()

    # Step 2 — resolve (and if necessary download) the asset tree.
    assets_dir = _resolve_assets(force=force_assets)

    # Step 3 — config must be written before any libero.libero.* import.
    _write_libero_config(assets_dir)

    # Step 4 — validate tree and refresh symlink.
    validate_asset_tree(assets_dir)
    _refresh_vendor_symlink(assets_dir)

    # Step 5 — robosuite macros.
    _setup_robosuite_macros()


# ---------------------------------------------------------------------------
# Asset resolution
# ---------------------------------------------------------------------------


def ensure_libero_assets(*, force: bool = False) -> pathlib.Path:
    """Resolve and validate the LIBERO asset directory.

    Prefer calling ``ensure_runtime()`` instead; this helper is exposed for
    scripts that only need the asset path.

    Returns:
        Path to a validated asset directory.

    Raises:
        LiberoAssetsUnavailableError: no source available.
        LiberoAssetValidationError: resolved tree is incomplete.
    """
    return _resolve_assets(force=force)


def validate_asset_tree(assets_dir: pathlib.Path) -> None:
    """Raise LiberoAssetValidationError if *assets_dir* is incomplete."""
    if not assets_dir.is_dir():  # follows symlinks
        raise LiberoAssetValidationError(
            f"Asset path does not exist or is not a directory: {assets_dir}\nRun: make setup-assets"
        )
    missing = [d for d in _REQUIRED_ASSET_DIRS if not (assets_dir / d).is_dir()]
    if missing:
        raise LiberoAssetValidationError(
            f"Asset tree at {assets_dir} is missing required directories: {missing}\n"
            "Run: make setup-assets"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_assets(*, force: bool = False) -> pathlib.Path:
    override = os.environ.get("LIBERO_INFINITY_ASSETS_DIR")
    if override:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        source = pathlib.Path(override)
        if not source.is_dir():
            raise LiberoAssetsUnavailableError(
                f"LIBERO_INFINITY_ASSETS_DIR={override!r} is not a directory."
            )
        validate_asset_tree(source)
        return source

    if not force and _ASSETS_CACHE_DIR.is_dir():
        try:
            validate_asset_tree(_ASSETS_CACHE_DIR)
            return _ASSETS_CACHE_DIR
        except LiberoAssetValidationError:
            pass  # incomplete cache — fall through to re-download

    return _download_hf_assets()


def _download_hf_assets() -> pathlib.Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise LiberoAssetsUnavailableError(
            "huggingface_hub is required to download LIBERO assets.\n"
            "Install it with: uv sync --extra simulation"
        ) from exc

    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_dir = pathlib.Path(tempfile.mkdtemp(dir=_CACHE_ROOT, prefix=".assets_download_"))
    try:
        print(f"Downloading LIBERO assets from {HF_REPO_ID}@{HF_REVISION[:8]} …")
        snapshot_download(
            repo_id=HF_REPO_ID,
            revision=HF_REVISION,
            repo_type="dataset",
            local_dir=str(tmp_dir),
            local_dir_use_symlinks=False,
        )
        validate_asset_tree(tmp_dir)
        if _ASSETS_CACHE_DIR.exists():
            shutil.rmtree(_ASSETS_CACHE_DIR)
        tmp_dir.rename(_ASSETS_CACHE_DIR)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    print(f"Assets ready at {_ASSETS_CACHE_DIR}")
    return _ASSETS_CACHE_DIR


def _check_parity_artifact() -> None:
    """Raise LiberoParityArtifactError if the committed parity artifact is absent or not PASS."""
    if not _PARITY_ARTIFACT.exists():
        raise LiberoParityArtifactError(
            f"Parity artifact not found: {_PARITY_ARTIFACT}\n"
            "Run: python scripts/run_asset_parity.py"
        )
    try:
        report = json.loads(_PARITY_ARTIFACT.read_text())
    except Exception as exc:
        raise LiberoParityArtifactError(f"Parity artifact is not valid JSON: {exc}") from exc

    if report.get("status") != "PASS":
        raise LiberoParityArtifactError(
            f"Parity artifact status is {report.get('status')!r}, expected 'PASS'.\n"
            "Run: python scripts/run_asset_parity.py"
        )
    if report.get("hf_revision") != HF_REVISION:
        raise LiberoParityArtifactError(
            f"Parity artifact revision {report.get('hf_revision')!r} does not match "
            f"runtime HF_REVISION {HF_REVISION!r}. Re-run the parity tool."
        )


def _write_libero_config(assets_dir: pathlib.Path) -> None:
    """Write ~/.libero/config.yaml pointing at repo-owned BDDL/init and validated assets."""
    import yaml  # available because pyyaml is a core dep

    bddl_dir = get_bddl_dir()
    init_dir = get_init_dir()

    config = {
        "benchmark_root": str(bddl_dir.parent),
        "bddl_files": str(bddl_dir),
        "init_states": str(init_dir),
        "datasets": str(_CACHE_ROOT / "datasets"),
        "assets": str(assets_dir),
    }

    config_path_env = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    config_dir = pathlib.Path(config_path_env)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"

    with open(config_file, "w") as f:
        yaml.dump(config, f)


def _refresh_vendor_symlink(assets_dir: pathlib.Path) -> None:
    """Keep vendor/libero/libero/libero/assets → assets_dir in sync.

    LIBERO's arena loaders resolve assets via hardcoded package-relative paths
    (../../assets/…), so a symlink at the vendor package root is required in
    addition to the config.yaml entry.
    """
    target = _VENDOR_ASSETS_LINK
    resolved = assets_dir.resolve()

    if target.is_symlink():
        if target.resolve() == resolved:
            return  # already correct
        target.unlink()
    elif target.exists():
        return  # real directory present (e.g. during tests without full setup)

    try:
        target.symlink_to(resolved)
    except OSError as exc:
        # Non-fatal: the config.yaml path will still work for get_libero_path().
        print(f"[runtime] Warning: could not create asset symlink at {target}: {exc}")


def _setup_robosuite_macros() -> None:
    """Create robosuite's macros_private.py if missing."""
    try:
        import robosuite
    except ImportError:
        return

    rs_dir = pathlib.Path(robosuite.__file__).parent
    macros_private = rs_dir / "macros_private.py"
    if not macros_private.exists():
        macros_src = rs_dir / "macros.py"
        if macros_src.exists():
            shutil.copy2(macros_src, macros_private)
