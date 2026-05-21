"""G7 three-layer review-sampling driver.

The §3 sampling design (see ``docs/human_review_rubric.md``) layers reviewer
effort:

- **Layer 1 — canonical exhaustive.** 12–20 hand-picked tasks spanning every
  predicate family. For each task, every one of the 512 axis subsets is
  packetized.
- **Layer 2 — all-task edge-case.** All 130 tasks × {none, each singleton
  axis, combined, full} = 1560 packets.
- **Layer 3 — flagged cases.** Tasks/conditions flagged in
  ``~/.omar/ea/4/validation_run/manifests/full_run.jsonl``,
  ``worst_50_rca.md``, and the c.1 addendum.

CLI
---
::

    python -m libero_infinity.validation.review_sampler \\
        --layer {1,2,3,all} --out <root> [--bddl-root <dir>] [--manifest <file>]

Each layer writes packets under ``<root>/layer<N>/<task>/<subset>/<seed>/``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import pathlib
import re
import sys
from collections.abc import Iterable, Iterator
from typing import Optional

from libero_infinity.planner.composition import AXIS_PRESETS, parse_axes
from libero_infinity.validation.review_packet import (
    ALL_AXES,
    generate_packet,
    subset_token,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical (Layer 1) task picks — span every predicate family.
# Names match BDDL filenames under data/libero_runtime/bddl_files/<suite>/.
# Robust to missing files: the sampler skips ones it cannot locate.
# ---------------------------------------------------------------------------

CANONICAL_TASKS: tuple[tuple[str, str, str], ...] = (
    # (suite, bddl_basename_substring, predicate_family_label)
    ("libero_10", "KITCHEN_SCENE3_turn_on_the_stove", "TurnOn + On (stove + moka_pot)"),
    (
        "libero_10",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close",
        "In + Close (long-horizon)",
    ),
    (
        "libero_10",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close",
        "In + Close microwave",
    ),
    ("libero_10", "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove", "multi-object On"),
    (
        "libero_10",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "multi-object In (container)",
    ),
    (
        "libero_10",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "stacking / dual placement",
    ),
    (
        "libero_10",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "In (compartment) / caddy fixture",
    ),
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet", "Open (articulation)"),
    ("libero_goal", "turn_on_the_stove", "TurnOn standalone"),
    ("libero_goal", "put_the_bowl_on_top_of_the_cabinet", "On (movable support)"),
    ("libero_goal", "put_the_wine_bottle_on_the_rack", "On rack (movable support)"),
    ("libero_goal", "put_the_cream_cheese_in_the_bowl", "In container"),
    (
        "libero_object",
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick+place container",
    ),
    (
        "libero_object",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick+place container (variant)",
    ),
    ("libero_spatial", "pick_up_the_black_bowl", "spatial-only pick"),
    ("libero_90", "KITCHEN_SCENE1", "fixture-backed (libero_90)"),
)


# ---------------------------------------------------------------------------
# Subset enumerations
# ---------------------------------------------------------------------------


def all_subsets() -> Iterator[frozenset[str]]:
    """Yield every one of the 2^|ALL_AXES| subsets."""
    axes = list(ALL_AXES)
    for r in range(len(axes) + 1):
        for combo in itertools.combinations(axes, r):
            yield frozenset(combo)


def edge_case_subsets() -> list[frozenset[str]]:
    """Layer-2 subsets: {none, each singleton axis, combined, full}."""
    out: list[frozenset[str]] = [frozenset()]
    out += [frozenset([a]) for a in ALL_AXES]
    out.append(AXIS_PRESETS["combined"])
    out.append(AXIS_PRESETS["full"])
    return out


# ---------------------------------------------------------------------------
# BDDL discovery
# ---------------------------------------------------------------------------


def default_bddl_root() -> pathlib.Path:
    """Return the in-repo bddl_files root."""
    return pathlib.Path(__file__).resolve().parents[1] / "data" / "libero_runtime" / "bddl_files"


def list_all_bddl(bddl_root: pathlib.Path) -> list[pathlib.Path]:
    """All BDDL files under ``bddl_root``, deterministically sorted."""
    return sorted(bddl_root.rglob("*.bddl"))


def resolve_canonical(
    bddl_root: pathlib.Path,
    canonical: Iterable[tuple[str, str, str]] = CANONICAL_TASKS,
) -> list[pathlib.Path]:
    """Resolve canonical-task descriptors against the on-disk BDDL tree."""
    resolved: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for suite, substr, _family in canonical:
        suite_dir = bddl_root / suite
        if not suite_dir.exists():
            logger.warning("canonical suite missing: %s", suite_dir)
            continue
        match: pathlib.Path | None = None
        for bddl in sorted(suite_dir.glob("*.bddl")):
            if substr in bddl.name:
                match = bddl
                break
        if match is None:
            logger.warning("canonical task not found in %s for substring %r", suite, substr)
            continue
        if match in seen:
            continue
        seen.add(match)
        resolved.append(match)
    return resolved


# ---------------------------------------------------------------------------
# Layer-3 flagged extraction
# ---------------------------------------------------------------------------


def flagged_from_manifest(manifest_path: pathlib.Path) -> list[tuple[str, list[str], int]]:
    """Read JSONL manifest; return (task, axis_subset, seed) for any failing row.

    A row counts as failing if any gate column is the string ``"fail"``.
    """
    out: list[tuple[str, list[str], int]] = []
    if not manifest_path.exists():
        return out
    with manifest_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            failed = any(
                str(row.get(g, "")).lower() == "fail" for g in ("g0", "g1", "g2", "g3", "g5", "g6")
            )
            if not failed:
                continue
            task = row.get("task")
            axes = row.get("axis_subset") or []
            seed = int(row.get("seed", 0))
            if isinstance(task, str):
                out.append((task, list(axes), seed))
    return out


def flagged_tasks_from_rca(rca_md_path: pathlib.Path) -> list[str]:
    """Pull task names out of a worst-50 / RCA markdown table.

    Looks for lines like ``| turn_on_the_stove | 63 | 17 |``.
    """
    if not rca_md_path.exists():
        return []
    out: list[str] = []
    seen: set[str] = set()
    row_re = re.compile(r"^\|\s*([a-z][a-z0-9_]+)\s*\|\s*\d")
    for line in rca_md_path.read_text(encoding="utf-8").splitlines():
        m = row_re.match(line)
        if m:
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def _packet_dir(
    root: pathlib.Path,
    layer: int,
    task_bddl: pathlib.Path,
    subset: frozenset[str],
    seed: int,
) -> pathlib.Path:
    safe_task = task_bddl.stem
    return root / f"layer{layer}" / safe_task / subset_token(subset) / f"seed{seed:04d}"


def run_layer1(
    out_root: pathlib.Path,
    bddl_root: pathlib.Path,
    *,
    seed: int = 0,
    render: bool = True,
    max_subsets: Optional[int] = None,
) -> list[pathlib.Path]:
    tasks = resolve_canonical(bddl_root)
    subsets = list(all_subsets())
    if max_subsets is not None:
        subsets = subsets[:max_subsets]
    written: list[pathlib.Path] = []
    for task in tasks:
        for subset in subsets:
            target = _packet_dir(out_root, 1, task, subset, seed)
            if (target / "checklist.yaml").exists():
                continue
            try:
                generate_packet(task, subset, seed, target, render=render)
                written.append(target)
            except Exception:
                logger.exception("layer1 packet failed for %s / %s", task.name, subset)
    return written


def run_layer2(
    out_root: pathlib.Path,
    bddl_root: pathlib.Path,
    *,
    seed: int = 0,
    render: bool = True,
) -> list[pathlib.Path]:
    tasks = list_all_bddl(bddl_root)
    subsets = edge_case_subsets()
    written: list[pathlib.Path] = []
    for task in tasks:
        for subset in subsets:
            target = _packet_dir(out_root, 2, task, subset, seed)
            if (target / "checklist.yaml").exists():
                continue
            try:
                generate_packet(task, subset, seed, target, render=render)
                written.append(target)
            except Exception:
                logger.exception("layer2 packet failed for %s / %s", task.name, subset)
    return written


def run_layer3(
    out_root: pathlib.Path,
    bddl_root: pathlib.Path,
    *,
    manifest: pathlib.Path | None = None,
    rca: pathlib.Path | None = None,
    render: bool = True,
) -> list[pathlib.Path]:
    written: list[pathlib.Path] = []
    triples: list[tuple[str, list[str], int]] = []
    if manifest is not None:
        triples.extend(flagged_from_manifest(manifest))
    if rca is not None:
        for task_name in flagged_tasks_from_rca(rca):
            # When we only know a task name, emit packets for the full preset at seed 0.
            triples.append((task_name + ".bddl", ["full"], 0))

    for task_str, axes_list, seed in triples:
        # task_str may be like "libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl"
        candidate = bddl_root / task_str
        if not candidate.exists():
            # Try a recursive search by basename
            matches = list(bddl_root.rglob(pathlib.Path(task_str).name))
            if not matches:
                logger.warning("layer3 task not found: %s", task_str)
                continue
            candidate = matches[0]
        subset = parse_axes(",".join(axes_list)) if axes_list else frozenset()
        target = _packet_dir(out_root, 3, candidate, subset, seed)
        if (target / "checklist.yaml").exists():
            continue
        try:
            generate_packet(candidate, subset, seed, target, render=render)
            written.append(target)
        except Exception:
            logger.exception("layer3 packet failed for %s / %s", candidate.name, subset)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_DEFAULT_MANIFEST = pathlib.Path.home() / ".omar/ea/4/validation_run/manifests/full_run.jsonl"
_DEFAULT_RCA = pathlib.Path.home() / ".omar/ea/4/validation_run/reports/worst_50_rca.md"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m libero_infinity.validation.review_sampler",
        description="Drive the G7 three-layer review-packet sampling design.",
    )
    p.add_argument("--layer", choices=["1", "2", "3", "all"], required=True)
    p.add_argument("--out", required=True, help="Output root directory.")
    p.add_argument(
        "--bddl-root",
        default=None,
        help="BDDL root (defaults to in-repo data/libero_runtime/bddl_files).",
    )
    p.add_argument(
        "--manifest", default=str(_DEFAULT_MANIFEST), help="Layer-3 input manifest (JSONL)."
    )
    p.add_argument(
        "--rca", default=str(_DEFAULT_RCA), help="Layer-3 RCA markdown to scrape task names from."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-render", action="store_true")
    p.add_argument(
        "--max-subsets",
        type=int,
        default=None,
        help="(Layer-1) cap subsets per task for smoke runs.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    out_root = pathlib.Path(args.out)
    bddl_root = pathlib.Path(args.bddl_root) if args.bddl_root else default_bddl_root()
    render = not args.no_render

    total: list[pathlib.Path] = []
    if args.layer in ("1", "all"):
        total += run_layer1(
            out_root, bddl_root, seed=args.seed, render=render, max_subsets=args.max_subsets
        )
    if args.layer in ("2", "all"):
        total += run_layer2(out_root, bddl_root, seed=args.seed, render=render)
    if args.layer in ("3", "all"):
        total += run_layer3(
            out_root,
            bddl_root,
            manifest=pathlib.Path(args.manifest),
            rca=pathlib.Path(args.rca),
            render=render,
        )
    print(f"wrote {len(total)} packets under {out_root}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
