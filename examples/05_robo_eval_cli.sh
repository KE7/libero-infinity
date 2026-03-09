#!/usr/bin/env bash
# =============================================================================
# 05_robo_eval_cli.sh — Evaluating a real VLA with robo-eval + LIBERO-Infinity
# =============================================================================
#
# robo-eval (from liten-vla) is the RECOMMENDED way to run full evaluations
# of real VLA policies (Pi0.5, SmolVLA, OpenVLA) against LIBERO-Infinity.
# It manages the entire stack automatically:
#   1. Starts the VLA inference server (Pi0.5 / SmolVLA / OpenVLA)
#   2. Starts a round-robin proxy for load balancing across server replicas
#   3. Launches LIBERO-Infinity sim workers (one per task)
#   4. Runs the evaluation loop: Scenic perturbation → LIBERO env → VLA predict
#   5. Collects and aggregates results with Wilson 95% CIs
#   6. Tears down all servers on completion or Ctrl+C
#
# Prerequisites:
#   1. libero-infinity installed (make install-full in this repo)
#   2. liten-vla installed (uv pip install -e . in the liten-vla repo)
#   3. Pi0.5 model weights available (auto-downloaded on first run)
#
# Rendering backend (set before any command):
#   MUJOCO_GL=egl      — EGL, default on headless Linux servers
#   MUJOCO_GL=osmesa   — software renderer if EGL is unavailable
#   (unset)            — macOS / desktop Linux with a display
# =============================================================================

set -euo pipefail

# Point robo-eval at this libero-infinity checkout so it finds the BDDL files
# and Scenic programs.  The variable is read by robo_eval/config.py.
export LIBERO_INFINITY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# MuJoCo rendering backend — EGL is the standard choice on headless GPU servers.
# Change to osmesa if your server does not have EGL, or unset on macOS.
export MUJOCO_GL=egl


# =============================================================================
# Integration test command — the canonical smoke-test for PI0.5 + LIBERO-Infinity
# =============================================================================
#
# This is the exact command used in the robo-eval × LIBERO-Infinity integration
# test: Pi0.5 on the spatial suite with position perturbation, 3 episodes per
# task, limited to 1 task for a fast smoke-test.
#
# Flag reference:
#   --benchmark libero_infinity   Use the LIBERO-Infinity backend (Scenic perturbation)
#   --vla pi05                    Evaluate Pi0.5 (PhysicalIntelligence 0.5 model)
#   --suites spatial              Run the LIBERO-Spatial suite (10 tasks)
#   --episodes 3                  3 independent perturbed episodes per task
#   --mode direct                 VLA executes directly (no VLM planner)
#   --max-tasks 1                 Limit to 1 task (remove for full 10-task suite)
#   --sim-args '...'              Scenic perturbation kwargs forwarded to LIBERO-Infinity
#                                   perturbation: which axes to randomise (see below)
#   --results-dir ...             Where to save JSON results
#
# Task: pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
# BDDL: src/libero_infinity/data/libero_runtime/bddl_files/libero_spatial/
#         pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl

robo-eval run \
  --benchmark libero_infinity \
  --vla pi05 \
  --suites spatial \
  --episodes 3 \
  --mode direct \
  --max-tasks 1 \
  --sim-args '{"perturbation": "position"}' \
  --results-dir results/pi05_3eps_position_infinity_spatial_smoketest


# =============================================================================
# Variations — uncomment to use
# =============================================================================

# --- Different perturbation axes ---

# Combined perturbation (position + object identity) — standard for benchmarking
# robo-eval run \
#   --benchmark libero_infinity \
#   --vla pi05 \
#   --suites spatial \
#   --episodes 50 \
#   --mode direct \
#   --sim-args '{"perturbation": "combined"}' \
#   --results-dir results/pi05_50eps_combined_infinity_spatial

# Full perturbation (all 6 axes simultaneously) — hardest setting
# robo-eval run \
#   --benchmark libero_infinity \
#   --vla pi05 \
#   --suites spatial \
#   --episodes 50 \
#   --mode direct \
#   --sim-args '{"perturbation": "full", "max_distractors": 5}' \
#   --results-dir results/pi05_50eps_full_infinity_spatial

# --- Different episode counts ---

# Quick 10-episode test (good for rapid iteration)
# robo-eval run \
#   --benchmark libero_infinity \
#   --vla pi05 \
#   --suites spatial \
#   --episodes 10 \
#   --mode direct \
#   --sim-args '{"perturbation": "position"}' \
#   --results-dir results/pi05_10eps_position_infinity_spatial

# Full 50-episode benchmark (recommended for publication-quality results)
# robo-eval run \
#   --benchmark libero_infinity \
#   --vla pi05 \
#   --suites spatial \
#   --episodes 50 \
#   --mode direct \
#   --sim-args '{"perturbation": "combined"}' \
#   --results-dir results/pi05_50eps_combined_infinity_spatial

# --- Different tasks / suites ---

# Goal suite (tasks focused on goal-state evaluation)
# robo-eval run \
#   --benchmark libero_infinity \
#   --vla pi05 \
#   --suites goal \
#   --episodes 50 \
#   --mode direct \
#   --sim-args '{"perturbation": "combined"}' \
#   --results-dir results/pi05_50eps_combined_infinity_goal

# All LIBERO-Infinity suites (spatial + object + goal + 10)
# robo-eval run \
#   --benchmark libero_infinity \
#   --vla pi05 \
#   --episodes 50 \
#   --mode direct \
#   --sim-args '{"perturbation": "combined"}' \
#   --results-dir results/pi05_50eps_combined_infinity_all

# SmolVLA instead of Pi0.5
# robo-eval run \
#   --benchmark libero_infinity \
#   --vla smolvla \
#   --suites spatial \
#   --episodes 50 \
#   --mode direct \
#   --sim-args '{"perturbation": "combined"}' \
#   --results-dir results/smolvla_50eps_combined_infinity_spatial

# --- Multi-GPU / faster evaluation ---

# 4 VLA server replicas on GPUs 0-3 (Pi0.5 needs ~7.4 GB VRAM each)
# robo-eval run \
#   --benchmark libero_infinity \
#   --vla pi05 \
#   --suites spatial \
#   --episodes 50 \
#   --mode direct \
#   --vla-replicas 4 \
#   --gpus 0,1,2,3 \
#   --sim-args '{"perturbation": "combined"}' \
#   --results-dir results/pi05_50eps_combined_infinity_spatial_4gpu


# =============================================================================
# Checking results
# =============================================================================
#
# After the run completes, inspect results with:
#
#   robo-eval status --results-dir results/pi05_3eps_position_infinity_spatial_smoketest
#
# Or read the JSON directly:
#
#   cat results/pi05_3eps_position_infinity_spatial_smoketest/*/summary.json | python3 -m json.tool
#
# Results JSON structure (from libero_infinity.eval.EvalResults):
#   {
#     "success_rate": 0.735,        # mean success rate
#     "ci_95": 0.061,               # Wilson 95% CI half-width
#     "n_success": 147,
#     "n_scenes": 200,
#     "episodes": [                  # per-episode data
#       {
#         "scene_index": 0,
#         "success": true,
#         "steps": 187,
#         "scenic_params": { ... },  # sampled Scenic parameters
#         "object_positions": { ... },
#         "object_classes": { ... },
#         "elapsed_s": 12.4
#       }, ...
#     ]
#   }


# =============================================================================
# Using libero-eval directly (without robo-eval)
# =============================================================================
#
# For quick experiments or when robo-eval is not installed, use the libero-eval
# CLI directly.  Note: this uses the zero-action default policy (robot stays still).
# For a real VLA, use the Python API (examples/01_basic_eval.py, 04_custom_vla.py).
#
# MUJOCO_GL=egl libero-eval \
#   --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_spatial/\
# pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl \
#   --perturbation position \
#   --n-scenes 3 \
#   --verbose
