.PHONY: install install-full test test-fast verify clean check-uv view

PYTHON ?= 3.11
MUJOCO_GL ?= egl

# Guard: ensure uv is installed before any target that needs it
check-uv:
	@command -v uv >/dev/null 2>&1 || 		{ echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

# Create venv + install all deps (submodule init + Scenic + simulation + dev + vendor/libero)
install: check-uv
	git submodule update --init --recursive
	uv venv --python $(PYTHON)
	uv sync --extra dev
	uv run pip install -e vendor/libero

# Full setup (alias for install)
install-full: install

# Run the full test suite (headless).  Requires GPU/EGL for simulation tests.
test:
	MUJOCO_GL=$(MUJOCO_GL) PYTHONPATH=src .venv/bin/python -m pytest tests/ -v

# Run only Tier 1 (no-GPU) tests: Scenic, perturbation policy, and planner.
# Use this target on machines without a GPU or when iterating quickly.
test-fast:
	MUJOCO_GL=$(MUJOCO_GL) PYTHONPATH=src .venv/bin/python -m pytest \
		tests/test_scenic.py \
		tests/test_perturbation_policy.py \
		tests/test_planner.py \
		-v

# Run full verification script
verify:
	MUJOCO_GL=$(MUJOCO_GL) uv run python scripts/verify_build.py

# Open interactive MuJoCo viewer for a task + perturbation
# Usage: make view BDDL=<task_name_or_path> SUITE=<suite> PERTURBATION=<type>
# Example:
#   make view BDDL=pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate \
#             SUITE=libero_spatial PERTURBATION=position
BDDL        ?=
SUITE       ?=
PERTURBATION ?= position

view: check-uv
	@test -n "$(BDDL)" || { echo "ERROR: BDDL is required.  Usage: make view BDDL=<task_name> [SUITE=<suite>] [PERTURBATION=<type>]"; exit 1; }
	uv run python scripts/viewer.py \
		--bddl "$(BDDL)" \
		$(if $(SUITE),--suite "$(SUITE)") \
		--perturbation "$(PERTURBATION)"

# Delete the venv for a clean rebuild.
# uv.lock is intentionally preserved — it is the reproducibility lockfile and
# must never be deleted by `make clean`.  Run `rm uv.lock` manually if you
# truly want to regenerate it.
clean:
	rm -rf .venv
