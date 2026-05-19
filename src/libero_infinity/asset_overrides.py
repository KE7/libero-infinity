"""MJCF override installer for LIBERO assets with broken inertial properties.

Background
----------
A subset of vendored LIBERO turbosquid asset MJCFs (`desk_caddy`,
`wooden_two_layer_shelf`) declare no ``<inertial>`` element and no flat
base geom. When LIBERO attaches a freejoint to the body, MuJoCo computes
mass and inertia automatically from the union of the asset's thin-shell
collision geoms — but those geoms cluster above and off-centre of the
contact-support polygon. The resulting COM tips the body during the
50-step post-reset settle and the free-joint translates ≈0.20 m before
quasi-quiescent contact is reached, exceeding the
``MAX_SETTLE_XY_DRIFT = 0.20`` invariant in
``simulator._validate_settled_positions``.

See ``~/.omar/ea/4/validation_run/rca/g5_settle_drift_caddy.md`` for the
full failure analysis (40 G5 settle-drift fails in Stage-3 Run-2).

Fix layer
---------
This module ships *patched* MJCFs in
``data/asset_overrides/<asset>/<asset>.xml`` with:

  * an explicit ``<inertial>`` element placing the COM at the geometric
    centre with a plausible mass for a thin-walled plastic / wooden
    fixture, and
  * a thin (4 mm) flat collision footprint geom spanning the full
    base so the support polygon is yaw-invariant.

The function :func:`install_mjcf_overrides` is called once on
``libero_infinity`` import. It locates the installed ``libero`` package
asset directory and overwrites the affected XMLs in place (no other
files are touched — the patched XML references the *same* mesh/texture
filenames as the original). The operation is idempotent: a patched file
contains the marker ``libero_infinity_patched`` and is skipped on
subsequent runs.

Why overwrite in place?
~~~~~~~~~~~~~~~~~~~~~~~
LIBERO's ``TurbosquidObjects`` class hard-codes the asset path from
``libero.libero.__file__`` and resolves textures/meshes relative to the
XML location. Re-pointing the loader to an out-of-tree XML would
require either symlinking every mesh/texture or rewriting robosuite's
``xml_path_completion`` lookup — both more invasive than swapping the
XML body in place.
"""

from __future__ import annotations

import pathlib
import pkgutil
from typing import Iterable

PATCH_MARKER = "libero_infinity_patched"

# Override list — (subdir, filename). Override files live in
# ``data/asset_overrides/<subdir>/<filename>`` inside this package.
_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("desk_caddy", "desk_caddy.xml"),
    ("wooden_two_layer_shelf", "wooden_two_layer_shelf.xml"),
)


def _installed_libero_assets_root() -> pathlib.Path | None:
    """Locate the installed LIBERO turbosquid_objects asset directory.

    Returns None if libero is not importable (in which case patching
    has nothing to act on and silently no-ops).
    """
    try:
        import libero.libero  # type: ignore
    except Exception:
        return None
    pkg_dir = pathlib.Path(libero.libero.__file__).parent
    candidate = pkg_dir / "assets" / "turbosquid_objects"
    return candidate if candidate.is_dir() else None


def _read_override(subdir: str, filename: str) -> str | None:
    data = pkgutil.get_data(
        "libero_infinity", f"data/asset_overrides/{subdir}/{filename}"
    )
    if data is None:
        return None
    return data.decode("utf-8")


def install_mjcf_overrides() -> list[pathlib.Path]:
    """Idempotently overwrite vendored MJCFs with libero_infinity-patched copies.

    Returns the list of paths actually rewritten (empty list on a clean re-import).
    """
    root = _installed_libero_assets_root()
    if root is None:
        return []
    rewritten: list[pathlib.Path] = []
    for subdir, filename in _OVERRIDES:
        target = root / subdir / filename
        if not target.is_file():
            continue
        try:
            current = target.read_text(encoding="utf-8")
        except OSError:
            continue
        if PATCH_MARKER in current:
            continue  # already patched
        patched = _read_override(subdir, filename)
        if patched is None:
            continue
        try:
            target.write_text(patched, encoding="utf-8")
            rewritten.append(target)
        except OSError:
            # Read-only install (e.g. system site-packages); fail open.
            continue
    return rewritten


def patched_override_paths() -> Iterable[pathlib.Path]:
    """Yield the in-package paths of the shipped override XMLs.

    Used by regression tests to load the patched MJCF directly.
    """
    pkg_dir = pathlib.Path(__file__).parent
    for subdir, filename in _OVERRIDES:
        yield pkg_dir / "data" / "asset_overrides" / subdir / filename
