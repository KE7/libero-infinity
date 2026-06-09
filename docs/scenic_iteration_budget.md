# Task/mode-adaptive Scenic iteration budget (WS-3)

## Problem

Scene generation is a **rejection sampler**: `Scenario.generate(maxIterations=N)`
draws candidate scenes until one satisfies every hard `require` constraint
(footprint clearances, on/in predicates, reachability, …), returning how many
iterations that took. If `N` is exhausted before a valid scene is found, Scenic
raises `RejectionException` — which the gym `reset()` loop surfaces as a
`MAX_SETTLE_RETRIES` failure.

Historically a single **global** `maxIterations=5000` was used everywhere
(`gym_env.py`, `eval.py`). Simple modes (e.g. `position`) need a handful of
iterations, but **harder perturbation modes compound many tight constraints**
and need far more. With a global 5000, hard scenes silently fail to generate,
corrupting the valid-scene distribution (training/validation coverage) **without
a clear diagnostic**.

## Solution

The iteration budget is now **resolved per perturbation mode** from a *measured*
calibration artifact, threaded through the generate paths, with a back-compat
default of 5000 and an under-budget early-warning.

- `LIBEROScenicEnv(..., max_scenic_iterations=None)` and `make_vec_env(...)`
- `evaluate(..., perturbation=None, max_scenic_iterations=None)`
- `evaluate_adversarial(..., perturbation=None, max_scenic_iterations=None)`

Resolution (`libero_infinity.scenic_budget.resolve_iteration_budget`):

1. An explicit `max_scenic_iterations` always wins (full back-compat).
2. Exact mode-name match in the artifact (`combined`, `full`, `position`, …).
3. Composite request (`"position,camera,distractor"`): the max of every
   measured single-axis budget present and every calibrated preset whose
   axis-set is contained in the request.
4. Otherwise the default **5000**.

When a generation consumes **≥ 90 %** of its budget, `warn_if_near_budget`
logs a WARNING — an early signal that the budget is under-provisioned for that
mode/task before it starts silently failing.

## How the budgets were measured (basis)

`scripts/calibrate_scenic_iterations.py` measures the distribution of `n_iters`
for each mode across a **diverse 5-task corpus** (flat placement, articulation +
containment, basket pick, spatial reference, long-horizon kitchen), drawing
16–25 valid scenes per task with a high cap (`maxIterations=300 000`) so the real
tail is observed rather than truncated.

For each mode we pool `n_iters` across tasks and set:

```
budget = ceil_round( max( p99_empirical, mean * ln(1/(1 - 0.999)) ) * 1.30 )
         # floored at 5000, rounded to a legible step
```

Two complementary tail estimators are combined and the **larger** is taken:

- **Empirical `p99`** of the observed counts.
- **Geometric-tail model** `mean * ln(1/(1-0.999))` (factor ≈ 6.9). Because the
  per-iteration accept probability `p` makes `n_iters ~ Geometric(p)`, this
  extrapolates the 99.9 % coverage point past the largest sample actually drawn
  — robust to small-sample noise and to the heavy mixture tail across tasks.

A 1.30× safety margin and round-up to a human-legible step (1k / 5k / 25k by
magnitude) give the final budget. The measured statistics and per-task
breakdown are stored in `src/libero_infinity/data/scenic_iteration_budgets.json`
(see its `_meta` block); re-run the calibration to refresh them.

### Measured result (summary)

| mode | n | mean | p95 | p99 | max | **budget** |
|---|---|---|---|---|---|---|
| position | 125 | 11 | 37 | 68 | 74 | **5000** |
| object | 125 | 1 | 1 | 1 | 1 | **5000** |
| camera | 125 | 1 | 1 | 1 | 1 | **5000** |
| lighting | 125 | 1 | 1 | 1 | 1 | **5000** |
| distractor | 100 | 10 | 35 | 48 | 48 | **5000** |
| background | 100 | 1 | 1 | 1 | 1 | **5000** |
| robot | 100 | 6 | 28 | 34 | 35 | **5000** |
| **combined** | 72 | 4362 | 20464 | 38827 | 41286 | **55000** |
| **full** | 64 | 4115 | 21220 | 28033 | 33048 | **40000** |

Every cheap single axis measured a mean of ~1–11 iterations with a p99 in the
tens, so all **floor at the 5000 back-compat default** — simple-mode behaviour is
unchanged. The heavy presets carry **tens of thousands** of iterations on hard
tasks: `combined` gets **55 000** and `full`'s own measurement gives **40 000**.

> **Note on the `full`/`combined` inversion.** `full`'s axis-set is a strict
> superset of `combined`'s, yet its *measured* budget came out lower (40 000 vs
> 55 000) — pure sampling noise from the heavy tail (n=64 vs 72; extra `full`
> axes like `texture`/`sensor_noise` add no placement constraints). To stay
> *monotone in axis-set inclusion*, `resolve_iteration_budget` folds in every
> contained preset, so `resolve("full")` returns **55 000** (≥ `combined`),
> never the smaller raw value. The authoritative per-mode numbers and per-task
> breakdown live in the JSON artifact.

## Discipline

This is **not** "bump the global to a huge number" — that would slow every
simple-mode run. Budgets are mode-specific and measured: simple modes keep
5000; only the modes that demonstrably need more get more.
