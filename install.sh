#!/usr/bin/env bash
# install.sh — convenience installer for LIBERO-Infinity (no make required)
#
# Equivalent to running: make install-full
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

# ── Guard: uv must be available ──────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found."
    echo "Install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "==> Creating virtual environment (Python ${PYTHON})..."
uv venv --python "${PYTHON}"

echo "==> Installing dependencies (simulation + dev extras)..."
uv sync --extra simulation --extra dev

echo "==> Bootstrapping HF assets and configuring LIBERO runtime..."
uv run python -c "from libero_infinity.runtime import ensure_runtime; ensure_runtime()"

echo ""
echo "Installation complete!"
echo ""
echo "To verify:"
echo "  MUJOCO_GL=${MUJOCO_GL} uv run python -m pytest tests/test_e2e.py -v"
echo ""
echo "To run the full verification suite:"
echo "  MUJOCO_GL=${MUJOCO_GL} uv run python scripts/verify_build.py"
