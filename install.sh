#!/usr/bin/env bash
# install.sh -- convenience installer for LIBERO-Infinity (no make required)
#
# Equivalent to running: make install
#
# Usage:
#   ./install.sh              # default: Python 3.11, MUJOCO_GL=egl
#   PYTHON=3.12 ./install.sh  # use a different Python version
#
# Requirements: uv  (https://docs.astral.sh/uv/)
#   Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

PYTHON="${PYTHON:-3.11}"
MUJOCO_GL="${MUJOCO_GL:-egl}"

# -- Guard: uv must be available -----------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found."
    echo "Install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "==> Initializing submodules..."
git submodule update --init --recursive

echo "==> Creating virtual environment (Python ${PYTHON})..."
uv venv --python "${PYTHON}"

echo "==> Installing dependencies..."
uv sync --extra dev

echo "==> Installing vendored LIBERO package..."
# Upstream LIBERO is missing libero/__init__.py, so find_packages() finds nothing.
# Create it if absent so the build can discover the package tree.
# Editable install (-e) is required so LIBERO can find its bundled XML/STL assets.
touch vendor/libero/libero/__init__.py
uv pip install --no-deps -e vendor/libero

echo "==> Writing LIBERO config..."
uv run python -c "
import pathlib, yaml, os
config_dir = pathlib.Path(os.environ.get('LIBERO_CONFIG_PATH', os.path.expanduser('~/.libero')))
config_dir.mkdir(parents=True, exist_ok=True)
config_file = config_dir / 'config.yaml'
if not config_file.exists():
    # Point at the repo-owned bddl/init data
    import importlib.resources
    data_root = str(importlib.resources.files('libero_infinity.data.libero_runtime'))
    config = {
        'benchmark_root': data_root,
        'bddl_files': str(pathlib.Path(data_root) / 'bddl_files'),
        'init_states': str(pathlib.Path(data_root) / 'init_files'),
        'datasets': str(pathlib.Path.home() / '.cache' / 'libero_infinity' / 'datasets'),
        'assets': str(pathlib.Path.home() / '.cache' / 'libero_infinity' / 'assets'),
    }
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    print(f'  Config written to {config_file}')
else:
    print(f'  Config already exists at {config_file}')
"

echo ""
echo "Installation complete!"
echo ""
echo "Quick check:"
echo "  uv run python -c 'import libero_infinity; import libero; print(\"OK\")'"
echo ""
echo "To run tests (headless, requires GPU/EGL):"
echo "  MUJOCO_GL=${MUJOCO_GL} uv run python -m pytest tests/ -v"
echo ""
