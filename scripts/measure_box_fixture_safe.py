"""Crash-safe per-(box_class, fixture) on-fixture spawn-clearance measurement.

The in-process scene-generation measurement (``measure_spawn_clearances.py
--distractor-fixtures-only``) settles full multi-distractor perturbation scenes;
one pathological seed spawns a deeply-interpenetrating config that overflows
MuJoCo's contact arena (``ncon = 5000`` at ``Time = 0.0000``) and SEGFAULTS the
whole run, leaving the data unwritten (RCA ``proxy_footprint_measure.md`` and the
v9 crash log). This driver runs EACH (task, seed) scene in an isolated CHILD
process: a child that overflows dies alone (exit signal), the parent records no
sample for that seed and continues — so the measurement completes and is total.

Measurement semantics are byte-identical to ``measure_distractor_fixtures`` (the
validated path): gate-free admission (real fixture contact + clearance in the
physical band ``[0, _FIXTURE_CLEARANCE_MAX]``), the rest surface is the contacted
fixture geom's world-AABB top nearest the distractor's bottom face, and per-pair
aggregation is the dominant settle MODE (``_dominant_mode``). NO AABB-bottom gate,
NO live-stepping of irregular distractors (the on-fixture path never steps — it
reads ``env.reset()``-settled contacts). The merge is corrective-only: a stored
row is rewritten ONLY when the measured mode diverges > pose_tolerance (5 mm);
within-tolerance rows stay BYTE-IDENTICAL; missing pairs are added.

Usage:
  parent:  python scripts/measure_box_fixture_safe.py [--seeds N]
  child :  python scripts/measure_box_fixture_safe.py --child <task_rel> <seed>
"""
from __future__ import annotations

import json
import pathlib
import random
import subprocess
import sys

# Reuse the VALIDATED helpers + constants from the canonical measurement module.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import measure_spawn_clearances as M  # noqa: E402


def _child(task_rel: str, seed: int) -> int:
    """Measure ONE (task, seed) scene; print per-sample JSON lines to stdout.

    Runs in its own process so a contact-arena overflow segfault is contained.
    """
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.simulator import TABLE_Z
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import resolve_task_path

    bddl = str(resolve_task_path(task_rel))
    try:
        cfg = TaskConfig.from_bddl(bddl)
        random.seed(seed)
        scenario = compile_task_to_scenario(cfg, "distractor")
        scene, _ = scenario.generate(maxIterations=8000)
        env = make_env(scene, bddl_path=bddl)
        env.reset()
    except Exception as exc:  # noqa: BLE001 — recorded as a skip, not masked
        print(json.dumps({"kind": "build_fail", "err": f"{type(exc).__name__}: {exc}"[:120]}))
        return 0

    sim = env._sim.libero_env.env.sim  # noqa: SLF001
    active = getattr(env._sim, "_active_distractor_names", set())  # noqa: SLF001
    for o in scene.objects:
        nm = getattr(o, "libero_name", "")
        if not nm.startswith("distractor_") or nm not in active:
            continue
        surface_class = getattr(o, "support_surface_class", "") or ""
        fixture_inst = getattr(o, "support_parent_name", "") or ""
        cls = getattr(o, "asset_class", "") or ""
        if not cls:
            continue
        bid = None
        for cand in (nm, nm + "_main"):
            try:
                bid = sim.model.body_name2id(cand)
                break
            except Exception:
                continue
        if bid is None:
            continue
        body_z = float(sim.data.body_xpos[bid][2])
        if not surface_class or not fixture_inst:
            # Table-assigned distractor (gate-free table-contact admission).
            clr = body_z - TABLE_Z
            if 0.0 <= clr <= M._FIXTURE_CLEARANCE_MAX and M._settled_on_table_surface(env, nm):
                print(json.dumps({"kind": "table", "cls": cls, "clr": round(clr, 5)}))
            continue
        box = M._body_world_aabb(sim, bid)
        tops = M._distractor_fixture_contact_tops(sim, bid, fixture_inst)
        if box is None or not tops:
            continue
        bottom_z = box[4]
        clearance = body_z - TABLE_Z
        if not (0.0 <= clearance <= M._FIXTURE_CLEARANCE_MAX):
            continue
        nearest_top = min(tops, key=lambda t: abs(t - bottom_z))
        rest_top_above_table = nearest_top - TABLE_Z
        rec = {
            "kind": "fixture",
            "key": f"{cls}|{surface_class}",
            "fclass": surface_class,
            "clr": round(clearance, 5),
            "rest_top": round(rest_top_above_table, 5),
        }
        faabb = M._fixture_world_aabb(sim, fixture_inst)
        if faabb is not None:
            rec["fw"] = round(faabb[1] - faabb[0], 5)
            rec["fl"] = round(faabb[3] - faabb[2], 5)
            rec["fh"] = round(faabb[5] - faabb[4], 5)
        print(json.dumps(rec))
    env.close()
    return 0


def _run_one(args: tuple) -> tuple:
    """Run one (task, seed) child subprocess; return (task, seed, rc, stdout)."""
    task_rel, seed = args
    try:
        cp = subprocess.run(
            [sys.executable, "-u", __file__, "--child", task_rel, str(seed)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        return task_rel, seed, cp.returncode, cp.stdout
    except subprocess.TimeoutExpired:
        return task_rel, seed, -99, ""


def _parent(seeds: int, workers: int = 8) -> int:
    import statistics
    from concurrent.futures import ThreadPoolExecutor

    from libero_infinity.validation.sweep import discover_all_tasks

    avail = set(discover_all_tasks())
    tasks = [t for t in M.MEASURE_TASKS if t in avail]

    samples: dict[str, list[float]] = {}
    table_samples: dict[str, list[float]] = {}
    fix_fp: dict[str, list[tuple]] = {}
    fix_top: dict[str, list[float]] = {}
    n_crash = 0
    n_build_fail = 0

    jobs = [(t, s) for t in tasks for s in range(seeds)]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for task_rel, seed, rc, out in ex.map(_run_one, jobs):
            done += 1
            if rc != 0:
                n_crash += 1
                print(
                    f"# [crash-contained {done}/{len(jobs)}] {task_rel.split('/')[-1][:40]} s{seed} "
                    f"rc={rc} (contact-arena overflow/segfault/timeout — seed skipped)",
                    flush=True,
                )
                continue
            for line in out.splitlines():
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = r.get("kind")
                if k == "build_fail":
                    n_build_fail += 1
                elif k == "table":
                    table_samples.setdefault(r["cls"], []).append(r["clr"])
                elif k == "fixture":
                    samples.setdefault(r["key"], []).append(r["clr"])
                    fix_top.setdefault(r["fclass"], []).append(r["rest_top"])
                    if "fw" in r:
                        fix_fp.setdefault(r["fclass"], []).append((r["fw"], r["fl"], r["fh"]))
            if done % 16 == 0:
                print(f"# [PROGRESS] {done}/{len(jobs)} scenes; "
                      f"{len(samples)} pairs, {n_crash} crashes so far", flush=True)

    dist_rows = {k: round(M._dominant_mode(v), 5) for k, v in sorted(samples.items())}
    table_rows = {c: round(statistics.median(v), 5) for c, v in sorted(table_samples.items())}
    fixture_geometry: dict[str, dict] = {}
    for fclass in sorted(set(fix_fp) | set(fix_top)):
        fps = fix_fp.get(fclass, [])
        tops = fix_top.get(fclass, [])
        entry: dict = {}
        if fps:
            entry["footprint"] = [
                round(statistics.median(p[0] for p in fps), 5),
                round(statistics.median(p[1] for p in fps), 5),
            ]
            entry["height"] = round(statistics.median(p[2] for p in fps), 5)
        if tops:
            entry["top_z"] = round(statistics.median(tops), 5)
        if entry:
            fixture_geometry[fclass] = entry

    print(
        f"\n# SUMMARY: {len(dist_rows)} (class|fixture) rows from {len(tasks)} tasks × {seeds} "
        f"seeds; {n_crash} seeds crash-contained, {n_build_fail} build-fails."
    )
    table_path = pathlib.Path("src/libero_infinity/data/spawn_clearances.json")
    table_clear = json.loads(table_path.read_text()).get("clearances", {})
    print("\n# measured (class|fixture) dominant-mode rows vs stored:")
    vdest = pathlib.Path("src/libero_infinity/data/spawn_clearances_variants.json")
    stored = json.loads(vdest.read_text())["clearances"]
    for k, v in dist_rows.items():
        cls, _, fclass = k.partition("|")
        old = stored.get(k)
        on_table = table_clear.get(cls)
        top = fixture_geometry.get(fclass, {}).get("top_z")
        analytic = f" analytic={top + on_table:.4f}" if (on_table is not None and top is not None) else ""
        delta = "NEW" if old is None else f"{(v - float(old)) * 1000:+.1f}mm"
        print(f"  {k:42} {v:.5f} (n={len(samples[k])}) stored={('—' if old is None else f'{float(old):.5f}')} Δ={delta}{analytic}")

    # ---- corrective merge: only > pose_tolerance, preserve byte-identical ----
    changed = M._merge_fixture_rows(stored, dist_rows)
    vdata = json.loads(vdest.read_text())
    vdata["clearances"] = {k: stored[k] for k in sorted(stored)}
    vdata["_meta"]["n_distractor_fixture_rows"] = len(dist_rows)
    vdest.write_text(json.dumps(vdata, indent=2, sort_keys=False) + "\n")
    print(f"\n# variant rows rewritten/added (>{M._POSE_TOLERANCE * 1000:.0f}mm): {len(changed)}")
    for k, v in sorted(changed.items()):
        print(f"  WROTE {k:42} -> {v:.5f}")

    # table rows: add-missing-only (never overwrite validated object-axis rows)
    M._merge_distractor_table_rows(table_path, table_rows)

    # fixture geometry: add-missing-only (preserve validated deterministic geometry)
    fgdest = pathlib.Path("src/libero_infinity/data/fixture_geometry.json")
    fg = json.loads(fgdest.read_text())
    added_fix = {f: g for f, g in fixture_geometry.items() if f not in fg["fixtures"]}
    fg["fixtures"].update(added_fix)
    fgdest.write_text(json.dumps(fg, indent=2, sort_keys=True) + "\n")
    print(f"# fixture geometry: {len(added_fix)} new added ({sorted(added_fix)}), rest preserved.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        raise SystemExit(_child(sys.argv[2], int(sys.argv[3])))
    _seeds = M._DISTRACTOR_FIXTURE_SEEDS
    for a in sys.argv:
        if a.startswith("--seeds="):
            _seeds = int(a.split("=", 1)[1])
    raise SystemExit(_parent(_seeds))
