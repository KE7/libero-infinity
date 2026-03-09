# What is Scenic?

[Back to main README](../README.md)

---

Scenic is a **probabilistic scene description language** — you write a
program that describes a *distribution* over environments, and Scenic's
built-in rejection sampler draws concrete instantiations from it.

---

## Why libero-infinity uses Scenic

Evaluation benchmarks in robotics have a reproducibility problem: once a
policy has been trained on a fixed set of initial states, re-evaluating it on
those same states measures memorisation, not generalisation.
libero-infinity replaces LIBERO's fixed initial-state files with Scenic
programs. Each call to `env.reset()` runs the Scenic sampler, which draws a
fresh scene satisfying all physical-plausibility constraints.

Three properties of Scenic make this a good fit:

| Property | What it gives us |
|----------|-----------------|
| **Declarative constraints** | Express "bowl and plate must be ≥ 0.22 m apart" directly; the sampler handles the search |
| **Built-in rejection sampler** | No bespoke collision-checking loop; Scenic loops until all `require` clauses pass |
| **Composability** | Each perturbation axis is a self-contained `.scenic` file; combining axes is one `compose` call |

---

## Three Scenic concepts you need to read `.scenic` files

### 1. `param` / `globalParameters`

`param` declares a named global that can be overridden at instantiation time
from Python:

```scenic
param min_clearance = 0.22   # default value
```

Inside the file you read it back through `globalParameters`:

```scenic
_min_clearance = globalParameters.min_clearance
```

The two-step pattern (declare with `param`, read with `globalParameters`) is
idiomatic Scenic 3. It lets the Python caller inject task-specific values
without editing the `.scenic` source.

### 2. `require` (hard) vs `require[p]` (soft)

**Hard constraint** — the sampler *rejects* any candidate scene where the
condition is false, and resamples:

```scenic
require (distance from bowl to plate) > _min_clearance
```

**Soft constraint** — the sampler *accepts* the scene with probability `p`
even when the condition is false (and rejects with probability `1 - p`):

```scenic
require[0.8] distance from bowl to bowl_train_pt > _ood_margin
```

`require[0.8]` says: "prefer OOD placements 80 % of the time, but occasionally
allow a placement near the training position." This is deliberately weaker than
a hard constraint so the distribution still covers the training region — proper
OOD evaluation, not just out-of-sample evaluation.

### 3. `Range()` / `DiscreteRange()`

`Range(lo, hi)` draws a float uniformly from `[lo, hi]`.
`DiscreteRange(lo, hi)` draws an integer uniformly from `{lo, lo+1, …, hi}`.

```scenic
# Continuous: camera position offset in [-0.10, 0.10] metres
x_offset = Range(-0.10, 0.10)

# Discrete: 1 to 5 distractor objects
n_distractors = DiscreteRange(1, 5)
```

These are the building blocks for all spatial and count distributions in
libero-infinity's perturbation programs.

---

## Annotated example: `position_perturbation.scenic`

Below is a condensed, annotated version of the file that drives the Position
perturbation axis. Line-by-line comments explain the Scenic constructs.

```scenic
model libero_model          # import Layer 2 (table geometry, LIBEROObject class)

# ── Parameters (overridable from Python) ────────────────────────────────────
param task         = "put_the_bowl_on_the_plate"   # task name (informational)
param min_clearance = 0.22   # minimum centre-to-centre distance between objects

# ── Object declarations ──────────────────────────────────────────────────────
# `in SAFE_REGION` is a Scenic *specifier* that constrains the object's
# position to lie inside a BoxRegion representing the reachable workspace.
# Scenic samples (x, y, z) uniformly within the region on every scene draw.
bowl = new LIBEROObject with libero_name "akita_black_bowl_1",
                         with width 0.10, with length 0.10, with height 0.06,
                         in SAFE_REGION

plate = new LIBEROObject with libero_name "plate_1",
                          with width 0.20, with length 0.20, with height 0.02,
                          in PLATE_SAFE_REGION   # wider margin for large objects

# ── Read params into locals (required before using them in `require`) ────────
_min_clearance = globalParameters.min_clearance
_ood_margin    = globalParameters.ood_margin
_btx = globalParameters.bowl_train_x   # canonical training x of the bowl
_bty = globalParameters.bowl_train_y

# ── Hard constraints (rejection sampler loops until these all pass) ──────────
require (distance from bowl to plate) > _min_clearance   # no overlapping objects

# ── Soft constraints (prefer OOD, but don't forbid training region) ──────────
bowl_train_pt = new Point at Vector(_btx, _bty, TABLE_Z)  # anchor at training pos
require[0.8] distance from bowl to bowl_train_pt > _ood_margin
# → 80 % of accepted scenes will have the bowl ≥ 0.15 m from its training spot
```

When you call `scenario.generate()` from Python, Scenic runs this program
repeatedly until it finds a sample where all hard `require` clauses pass.
The resulting `scene.objects` list carries the sampled positions, which
libero-infinity then injects into the MuJoCo XML before `env.reset()` returns.

---

## You do not need to write any Scenic code

libero-infinity completely hides the Scenic layer behind its Python API and
CLI. When you call `generate_scenic_file(cfg, perturbation="combined")` or
pass `--perturbation full` to `libero-eval`, the framework selects, composes,
and parameterises the appropriate `.scenic` programs automatically. The
perturbation programs in `scenic/` are implementation details — you will never
need to edit or even open them during normal use. This primer exists only so
that the documentation and error messages make sense if you ever need to look
inside.

---

## Further reading

- [Scenic language reference](https://scenic-lang.readthedocs.io/en/latest/language_ref.html)
- [Scenic GitHub repository](https://github.com/BerkeleyLearnVerify/Scenic)
- [Scenic paper (PLDI 2022)](https://people.eecs.berkeley.edu/~sseshia/pubs/b2hd-fremont-pldi22.html)
