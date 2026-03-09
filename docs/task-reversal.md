# Task Reversal (Backward Evaluation)

[Back to main README](../README.md)

**Module**: `src/libero_infinity/task_reverser.py`

Reverses a forward BDDL task to create a backward evaluation scenario. Instead of
"put the bowl on the plate", the bowl starts on the plate and must be placed back
on the table. This tests policy robustness from novel initial configurations that
are the goal states of standard tasks.

---

## Reversal rules

| Forward Goal | Reversed Init | Reversed Goal |
|---|---|---|
| `(On A B)` object on object | `(On A B)` | `(On A {A's original table region})` |
| `(On A FIX_REGION)` object on fixture | `(On A FIX_REGION)` | `(On A {A's original table region})` |
| `(In A REGION)` object in container | `(In A REGION)` | `(On A {A's original table region})` |
| `(Open R)` | *(closed by default)* | `(Close R)` |
| `(Close R)` | `(Open R)` | *drop — already open* |
| `(Turnon F)` | *(off by default)* | `(Turnoff F)` |
| `(Turnoff F)` | `(Turnon F)` | *drop — already on* |

---

## Stacking dependencies

When a reversed task has `(On bowl plate)` in init, the bowl must follow the plate
during position perturbation. The scenic generator emits relative positioning:

```scenic
plate = new LIBEROObject at Vector(Range(X_MIN, X_MAX), Range(Y_MIN, Y_MAX), TABLE_Z)
bowl = new LIBEROObject at Vector(plate.position.x, plate.position.y, TABLE_Z)
```

Clearance constraints are automatically skipped between stacked parent-child pairs.
OOD soft constraints are also skipped for stacked objects (they follow their parent).

---

## Language rewriting

Language instructions are constructed from reversed goal predicates rather than
regex-flipping the English string. This is more robust across LIBERO's ~20 language
patterns.

Construction rules:
- `(On A region)` → "place the {A_human} on the table"
- `(Close R)` → "close the {R_human}"
- `(Open R)` → "open the {R_human}"
- `(Turnoff F)` → "turn off the {F_human}"
- Compound goals joined with " and "
- `_human` = instance name with underscores→spaces, trailing `_N` stripped

Example: "put bowl on plate" → reversed predicates `(On bowl table_region)` →
constructed: "pick up the bowl and place it on the table"

Fallback for unrecognized patterns: `"(reversed) {original_language}"`

---

## Usage

### CLI

```bash
# Add --reverse to any evaluation command
libero-eval --bddl path/to/put_the_bowl_on_the_plate.bddl \
  --reverse --perturbation position --n-scenes 100 --verbose

# Reversed + full perturbation (position + object + camera + lighting)
libero-eval --bddl path/to/task.bddl \
  --reverse --perturbation full --n-scenes 200

# Batch-generate reversed BDDL files for inspection
python scripts/generate_reversed_bddls.py \
  --input src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/ \
  --output data/reversed_bddls/libero_goal/
```

### Python API

```python
from libero_infinity.task_reverser import reverse_bddl

original = open("path/to/task.bddl").read()
reversed_bddl = reverse_bddl(original, return_region_margin=0.05)
# reversed_bddl is a complete BDDL string with swapped init/goal
```

---

## Data flow: reversed + perturbed evaluation

This diagram shows the full pipeline when `--reverse --perturbation position` is used:

```
put_the_bowl_on_the_plate.bddl
│
│  1. task_reverser.reverse_bddl()
│     - Goal (On bowl plate) → Init (On bowl plate), Goal (On bowl table_region)
│     - Language: "put the bowl on the plate" → "place the akita black bowl on the table"
│     - Region widened if < 0.10 m span
▼
reversed.bddl  (temp file via patched_bddl_from_string)
│
│  2. TaskConfig.from_bddl()
│     - Parses reversed BDDL: bowl has stacked_on="plate_1"
│     - Plate gets init region position; bowl inherits plate's position
▼
TaskConfig(movable_objects=[plate(init_x, init_y), bowl(stacked_on="plate_1")])
│
│  3. compiler.generate_scenic(perturbation="position")
│     - Plate: at Vector(Range(X_MIN, X_MAX), Range(Y_MIN, Y_MAX), TABLE_Z)
│     - Bowl:  at Vector(plate.position.x, plate.position.y, TABLE_Z)
│     - Clearance constraint between plate and bowl SKIPPED (stacked pair)
│     - OOD soft constraint on bowl SKIPPED (follows parent)
▼
scenic/generated/_gen_place_the_akita_black_bowl_on_the_table_position.scenic
│
│  4. Scenic constraint solver (rejection sampling)
│     - Samples plate position, bowl follows
│     - Validates clearance with other objects
▼
scene.objects = [plate(x=0.12, y=-0.08), bowl(x=0.12, y=-0.08)]
│
│  5. LIBEROSimulation.setup()
│     - env.reset() loads reversed BDDL (bowl starts on plate)
│     - Injects Scenic-sampled positions via set_joint_qpos
│     - 50 settling steps → bowl stabilizes on plate
▼
MuJoCo state: bowl resting on plate at (0.12, -0.08), robot ready
│
│  6. Policy loop: step_with_action() × max_steps
│     - Robot must pick up bowl and place it on the table
│     - check_success() tests (On bowl table_region)
▼
EpisodeResult(success=True/False, steps=N, ...)
```
