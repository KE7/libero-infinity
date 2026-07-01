# g4 §6 — xy / drawer-state-aware support heightfield for the cabinet residual

**Branch:** `fix/g4-cabinet-heightfield` (base `dc5451f` = #35–#39)
**Scope:** the surfaced §2d-option-1 architecture change from
`rca/g4_fixed_point_settle.md` — make the renderer/simulator emit an XY-DEPENDENT
and DRAWER-STATE-AWARE spawn-z for `akita_black_bowl` on `wooden_cabinet`, so
`pose_tolerance` passes WITHOUT any gate change.

---

## 0. Verdict (short)

The scalar `<class>|<surface>` clearance cannot represent the cabinet top: the
realized rest of `(akita_black_bowl, wooden_cabinet)` is genuinely
xy/drawer-state dependent (tri-modal 0.898 / 0.917 / 1.126 across the tasks that
share the key). This branch adds the funded **support HEIGHTFIELD** — an additive
per-`(fixture, relation, drawer_state, class)` measured-rest table
(`data/fixture_heightfields.json`) resolved by `asset_metadata.heightfield_spawn_z`
in BOTH the renderer and the simulator. It is **byte-identical** to today for
every placement it does not cover (proven, §4).

**Closed** (shipped): the **closed-drawer `on_surface` (top_side)** mode — the
bowl falls off the collision-less cabinet top to a table-level rest
(**0.898**, clearance 0.0784 above the arena surface), DETERMINISTIC + STABLE
(36/36 samples, 0.0 mm spread, 99/99 grid cells stable). This closes
`…on_the_cookie_box…` fully and the majority of `…on_the_wooden_cabinet…`.

**Surfaced** (NOT forced): three residuals below (§5) — the open-drawer top_side
metastability, the in-drawer relative-path z, and a two-body contact explosion —
each precisely characterized, none band-aided.

---

## 1. Method — SETTLE-FROM-ABOVE only (never iterate-from-analytic)

`scripts/measure_g4_cabinet_heightfield.py`, `scripts/scan_g4_cabinet_grid.py`,
`scripts/probe_g4_cabinet_open_envelope.py`.

The scalar iterated fixed point (`measure_g4_fixture_fixedpoint.py`) **tunnels**
on the cabinet: re-injecting at the settled pose compounds an initial penetration
of the cabinet's THIN top collision panel, so `f(z)=settle50(z)` converges to a
spurious `z*≈0.898` that is 228 mm BELOW the true solid-top rest (RCA §1). The
only trustworthy rest is **settle-from-above**: inject the bowl once at `z0=1.30`
(well above the top) and settle WITHOUT re-injection. Stability is then confirmed
by a 50-step settle FROM the measured rest at the sampled xy (matches exactly what
`pose_tolerance` sees when the renderer emits that rest).

Key physics finding: the top_side bowl is in **free-fall during the entire 50-step
validation window** (emitted at the analytic 1.229 → settles to 1.1645 mid-fall →
fails by 64 mm). The ONLY z that passes is the object's true STABLE rest, injected
where the 50-step settle barely moves.

## 2. Measured support surface (kitchen arena, arena_z = TABLE_SURFACE_Z = 0.82)

| relation | drawer | class | rest z | clearance | determinism |
|---|---|---|---:|---:|---|
| on_surface (top_side) | **closed** | akita_black_bowl | **0.8984** | **0.0784** | 36/36, spread 0.0 mm, 99/99 grid cells stable → **SHIPPED** |
| inside (top_region drawer) | open | akita_black_bowl | 1.1264 | 0.3064 | 6/6, spread 0.0 mm, stable → surfaced (relative-path, §5.2) |
| on_surface (top_side) | **open** | akita_black_bowl | 0.898 / 1.06–1.126 | — | **METASTABLE**: 10/16 stable, 6/16 roll off the open-drawer knife-edge (>130 mm dz / >200 mm xy) → surfaced (§5.1) |

Dense grid (`scan_g4_cabinet_grid.py`): closed top is UNIFORMLY 0.898 & stable
across the whole realized envelope; the open drawer creates a stable table plateau
(0.898) + a stable drawer plateau (~1.12) separated by a genuinely metastable
transition band the realized envelope straddles.

## 3. Emission (renderer + simulator, in lockstep)

- `asset_metadata.heightfield_spawn_z(arena_z, fixture, relation, state, class)`
  returns `arena_z + measured_clearance`, or **`None`** for any uncovered tuple.
- Renderer (`scenic_renderer._resolved_spawn_z`, `_spawn_z_expr`, the object-axis
  `(class, z)` pairs) uses the heightfield when present, else the unchanged
  `surface_spawn_z`. Relation kind (support edge) and drawer state
  (`plan.articulation_plans[...].state_kind`, from the BDDL `:init` `(Open …)`
  predicate) are both known at codegen. Two specifiers
  (`support_relation_kind`, `cabinet_drawer_state`) are emitted **only when the
  resolver returns a value**, so uncovered cabinet objects gain ZERO specifiers.
- Simulator (`simulator.py`, injection z-resolution) resolves the SAME
  `heightfield_spawn_z` FIRST (before the LIBERO default-z / scalar branches) from
  those specifiers + the actually-sampled `asset_class`, so injected z == emitted
  z == settled pose.

## 4. No-regression proof (byte-identical emission)

`scripts/control_g4_heightfield_byte_identical.py` renders 11 tasks × 3 subsets
across floor / table / kitchen / living_room / stove / cabinet arenas TWICE — with
`fixture_heightfields.json` active vs disabled (= HEAD) — and diffs every emitted
object line: **201 lines compared, 3 differ, ALL LEGAL cabinet on_surface|closed
akita lines, 0 illegal regressions.** Unit test
`tests/test_cabinet_heightfield.py::test_byte_identical_scalar_fallback_broad`
proves the resolver equals the scalar for the full (arena × surface × class ×
relation × state) grid minus the one covered tuple.

CI (scenic-only g0–g3 + Tier-1) is unaffected: the change alters only the emitted
z, not the Scenic xy constraints. Verified: reduced-suite g0–g3 = 46/46 pass on
the affected stove/cabinet tasks; Tier-1 (test_scenic + invariants + policy) all
green; `tests/test_cabinet_heightfield.py` 6/6.

## 5. Residuals — precisely surfaced, NOT forced

### 5.1 Open-drawer top_side bowl — irreducibly metastable
When the drawer is OPEN (the `in_top_drawer` task's second bowl, `On top_side`),
the extended drawer supports part of the realized envelope at ~1.12 and the rest
falls to table 0.898, with a metastable transition band the envelope straddles.
`probe_g4_cabinet_open_envelope.py`: 6/16 samples roll off the drawer edge with
>130 mm dz / >200 mm xy (3 exceed the resampler's own 0.20 m drift threshold);
some are scene-layout (neighbour) dependent. **No emission-time z closes it** —
same class as the RCA residual-B second-energy-minimum tail. Left absent (byte-
identical scalar path); NOT forced.

### 5.2 In-drawer bowl (1.126) — relative-positioning path
The `In top_region` bowl is emitted via RELATIVE positioning
(`at cabinet offset by Vector(rx, ry, 0.0)`), not the absolute spawn-z path, and
the simulator resolves contained objects via the LIBERO default-z. Its rest
(1.126) is deterministic+stable and the heightfield mechanism CAN key it
(`inside|open`), but closing it needs the relative-path z-offset emitted as a
`(rest − cabinet.z)` expression + object-axis handling — a distinct code path,
deferred. Left absent (byte-identical); does not close `in_top_drawer` on its own
anyway (§5.1 still fails there).

### 5.3 Two-body contact explosion (`on_the_wooden_cabinet`, minority)
Live confirmation of RCA §2c. In `…on_the_wooden_cabinet…` the second akita bowl
(on the stove, but planner-placed near origin at 0.898) sits ~0.101 m from the
top_side bowl. The movable↔movable AABB separation constraint (`|dx|>0.1`, from the
akita registry width 0.1) is satisfied at 0.101 m, but the real bowl mesh is wider
than its 0.1 registry AABB, so once the top_side bowl is LOWERED into the same
z-plane the two are injected overlapping and the contact solver launches both to
~1.03 (dz ~120–140 mm). On HEAD this was masked because the top_side bowl floated
at 1.229, z-separated. Net for `…on_the_wooden_cabinet…`: 8/16 → 14/16, with **1
seed regressing** the previously-passing stove bowl. Root cause is a latent
registry-AABB underestimate + fixture-supported task bowls placed near origin —
NOT the heightfield z (which is correct). A clean fix is a companion change
(accurate akita footprint in the separation constraint, or anchoring fixture-bowls
to their fixture), out of the emission-path scope; surfaced for a follow-up.

## 6. Guards honored

- Gate UNCHANGED: `consistency.py` `DEFAULT_POS_TOL 5e-3`, `DEFAULT_ROT_TOL_DEG 1.0`.
- Resampler UNCHANGED: `MAX_SETTLE_ROT_DRIFT 35°`, `MAX_SETTLE_XY_DRIFT 0.20 m`.
- Additive data only: `fixture_geometry.json`, `spawn_clearances*.json`,
  `stack_offsets.json` byte-identical; new `fixture_heightfields.json` covers ONE
  tuple. No masking / forcing / metastable-mode recording.
- Byte-identical emission for all non-heightfield placements (§4).
