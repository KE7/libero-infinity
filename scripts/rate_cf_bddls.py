#!/usr/bin/env python3
"""Score every auto-generated CF BDDL variant for quality.

Rates each variant on four dimensions (each 1–5) and computes a weighted
overall score (0–100).  Results are printed as a sorted table and saved to
``cf_quality_ratings.csv``.

Quality dimensions
------------------
language (25 %)
    How natural is the generated instruction?
    Penalises very long class-name phrases, digit-containing tokens, and
    awkward multi-word constructs.

feasibility (35 %)
    Can a Franka Panda gripper actually pick up this object and place it at
    the destination?  Non-graspable fixtures score 1; large-but-graspable
    objects score 3; normal manipulanda score 5.

cf_value (25 %)
    How strong a grounding test is this?  If the CF object is in the same
    visual category as the source (e.g., source=bowl, CF=ramekin), a
    language-blind policy could still guess correctly; different-category
    swaps (source=bowl, CF=milk_bottle) are harder and more diagnostic.

dest_fit (15 %)
    Does placing *this specific CF object* at the destination make physical
    and semantic sense?  A bottle balanced on a small plate scores lower
    than putting a soup can in a basket.

Usage
-----
    python scripts/rate_cf_bddls.py
    python scripts/rate_cf_bddls.py --min-score 60   # show only A/B tier
    python scripts/rate_cf_bddls.py --csv out.csv    # custom output path
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from libero_infinity.bddl_preprocessor import generate_cf_bddls
from libero_infinity.runtime import get_bddl_dir

# ── object categories (for grounding-challenge scoring) ──────────────────────

# Objects in the same visual category look similar; swapping within category
# gives a weaker grounding test.
_CATEGORY: dict[str, str] = {
    # Shallow bowls / plates
    "akita_black_bowl": "bowl", "white_bowl": "bowl",
    "glazed_rim_porcelain_ramekin": "bowl", "plate": "bowl",
    "chefmate_8_frypan": "bowl",
    # Tall cylinders / bottles
    "wine_bottle": "bottle", "ketchup": "bottle", "milk": "bottle",
    "orange_juice": "bottle", "tomato_sauce": "bottle", "bbq_sauce": "bottle",
    # Flat cardboard boxes (food)
    "cream_cheese": "box", "butter": "box", "chocolate_pudding": "box",
    "alphabet_soup": "box", "cookies": "box",
    # Mugs / cups
    "red_coffee_mug": "mug", "white_yellow_mug": "mug",
    "porcelain_mug": "mug", "moka_pot": "mug",
    # Flat/rigid containers / trays
    "basket": "tray", "wooden_tray": "tray", "desk_caddy": "tray",
    # Books
    "black_book": "book",
    # Condiments (squeeze bottles / small jars — similar silhouette to bottles)
    "salad_dressing": "bottle", "new_salad_dressing": "bottle",
}

# ── graspability ──────────────────────────────────────────────────────────────

_FIXTURE = frozenset({
    "wine_rack", "wooden_cabinet", "flat_stove", "microwave",
    "white_cabinet", "floor", "table", "kitchen_table",
    "living_room_table", "study_table", "wooden_two_layer_shelf",
    "wooden_two_layer_shelf",
})

# Objects a gripper can pick but that are harder to place precisely
_LARGE_GRASPABLE = frozenset({"basket", "wooden_tray", "desk_caddy"})


def _feasibility(cf_class: str) -> int:
    if cf_class in _FIXTURE:
        return 1
    if cf_class in _LARGE_GRASPABLE:
        return 3
    return 5


# ── language naturalness ──────────────────────────────────────────────────────

# Pre-scored clean phrases; everything else is scored algorithmically.
_LANG_SCORES: dict[str, int] = {
    # 5 — short, everyday English
    "cream cheese": 5, "wine bottle": 5, "milk": 5, "butter": 5,
    "cookies": 5, "plate": 5, "basket": 5, "ketchup": 5,
    "black book": 5, "moka pot": 5, "white bowl": 5,
    # 4 — slightly technical but clear
    "salad dressing": 4, "tomato sauce": 4, "bbq sauce": 4,
    "orange juice": 4, "wooden tray": 4, "desk caddy": 4,
    "red coffee mug": 4, "chocolate pudding": 4, "alphabet soup": 4,
    "new salad dressing": 4,
    # 3 — multi-word compound, mildly unusual
    "porcelain mug": 3, "white yellow mug": 3, "akita black bowl": 3,
    # 2 — verbose or contains digit
    "glazed rim porcelain ramekin": 2, "chefmate 8 frypan": 2,
}


def _language(cf_class: str) -> int:
    phrase = cf_class.replace("_", " ")
    if phrase in _LANG_SCORES:
        return _LANG_SCORES[phrase]
    # Algorithmic fallback:
    words = phrase.split()
    if any(w.isdigit() or (w[0].isdigit() if w else False) for w in words):
        return 1  # contains digit token
    if len(words) >= 5:
        return 1
    if len(words) == 4:
        return 2
    if len(words) == 3:
        return 3
    return 4  # 1–2 clean words not in the table


# ── destination fitness ───────────────────────────────────────────────────────

def _dest_fit(cf_class: str, predicate: str, dest_phrase: str) -> int:
    """Score whether placing cf_class at dest makes physical/semantic sense."""
    cat = _CATEGORY.get(cf_class, "unknown")
    if predicate == "In":
        # Placing into a basket/container — almost always fine
        if cat in ("bottle", "box", "bowl", "mug"):
            return 5
        if cat in ("tray", "book"):
            return 3
        return 4
    else:  # On
        dest_lower = dest_phrase.lower()
        if "plate" in dest_lower or "bowl" in dest_lower or "tray" in dest_lower:
            if cat in ("box", "mug", "bottle"):
                return 4
            if cat == "bowl":
                return 5  # bowl on plate: canonical LIBERO scenario
            if cat == "tray":
                return 2  # tray balanced on plate: precarious
            if cat == "book":
                return 3
        # Stove / shelf / cabinet top
        if any(k in dest_lower for k in ("stove", "shelf", "cabinet", "table")):
            return 5
        return 4


# ── CF grounding value ────────────────────────────────────────────────────────

def _cf_value(source_class: str, cf_class: str) -> int:
    """How good is this swap as a grounding test?"""
    src_cat = _CATEGORY.get(source_class, "unknown")
    cf_cat = _CATEGORY.get(cf_class, "unknown")
    if cf_class in _FIXTURE:
        return 1
    if src_cat == cf_cat:
        # Same visual category — weaker CF but not useless
        return 3
    if src_cat == "unknown" or cf_cat == "unknown":
        return 3
    return 5  # different categories → strong grounding test


# ── overall score ─────────────────────────────────────────────────────────────

WEIGHTS = {"language": 0.25, "feasibility": 0.35, "cf_value": 0.25, "dest_fit": 0.15}


def _overall(scores: dict[str, int]) -> float:
    raw = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    return round((raw - 1) / 4 * 100, 1)  # scale [1,5] → [0,100]


def _tier(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "D"


# ── main ──────────────────────────────────────────────────────────────────────

LIBERO_BDDL_ROOT = get_bddl_dir()
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def _parse_goal_meta(bddl_content: str) -> tuple[str, str, str]:
    """Return (predicate, source_inst, dest_inst) from the goal block."""
    from libero_infinity.bddl_preprocessor import _extract_block
    goal_block = _extract_block(bddl_content, "goal") or ""
    pred_re = re.compile(r"\((On|In)\s+([^\s()]+)\s+([^\s()]+)\)")
    m = pred_re.search(goal_block)
    if not m:
        return ("", "", "")
    return m.group(1), m.group(2), m.group(3)


def _parse_obj_classes(bddl_content: str) -> dict[str, str]:
    from libero_infinity.bddl_preprocessor import parse_object_classes
    return parse_object_classes(bddl_content)


def _dest_display(dest_inst: str, obj_classes: dict[str, str]) -> str:
    if dest_inst in obj_classes:
        return obj_classes[dest_inst].replace("_", " ")
    # region name like "basket_1_contain_region" → "basket"
    s = re.sub(r"_\d+_\w+_region$", "", dest_inst)
    s = re.sub(r"_\d+$", "", s)
    return s.replace("_", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent / "cf_quality_ratings.csv",
    )
    parser.add_argument("--suite", choices=SUITES)
    args = parser.parse_args()

    rows: list[dict] = []

    suites = [args.suite] if args.suite else SUITES
    for suite_name in suites:
        suite_dir = LIBERO_BDDL_ROOT / suite_name
        if not suite_dir.exists():
            continue
        for bddl_path in sorted(suite_dir.glob("*.bddl")):
            content = bddl_path.read_text()
            predicate, source_inst, dest_inst = _parse_goal_meta(content)
            if not predicate:
                continue
            obj_classes = _parse_obj_classes(content)
            source_class = obj_classes.get(source_inst, source_inst)
            dest_disp = _dest_display(dest_inst, obj_classes)

            variants = generate_cf_bddls(content)
            for suffix, cf_text in variants:
                # Recover cf_class from suffix: "_cf_cream_cheese" → "cream_cheese"
                cf_class = suffix.removeprefix("_cf_")
                lang_score = _language(cf_class)
                feas_score = _feasibility(cf_class)
                cfv_score  = _cf_value(source_class, cf_class)
                df_score   = _dest_fit(cf_class, predicate, dest_disp)
                scores = {
                    "language": lang_score,
                    "feasibility": feas_score,
                    "cf_value": cfv_score,
                    "dest_fit": df_score,
                }
                overall = _overall(scores)
                cf_lang_match = re.search(r"\(:language ([^)]+)\)", cf_text)
                cf_lang = cf_lang_match.group(1) if cf_lang_match else ""
                rows.append({
                    "suite": suite_name,
                    "source_task": bddl_path.stem,
                    "source_class": source_class,
                    "cf_class": cf_class,
                    "predicate": predicate,
                    "dest": dest_disp,
                    "generated_language": cf_lang,
                    "score_language": lang_score,
                    "score_feasibility": feas_score,
                    "score_cf_value": cfv_score,
                    "score_dest_fit": df_score,
                    "overall": overall,
                    "tier": _tier(overall),
                })

    rows.sort(key=lambda r: r["overall"], reverse=True)

    # Filter for display
    display_rows = [r for r in rows if r["overall"] >= args.min_score]

    # Print summary table
    tiers = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in rows:
        tiers[r["tier"]] += 1

    print(f"\n{'─'*90}")
    print(f"  CF Variant Quality Ratings  ({len(rows)} total variants)")
    print(f"{'─'*90}")
    print(f"  Tier A (≥80):  {tiers['A']:3d}   ██ high quality, recommended")
    print(f"  Tier B (≥60):  {tiers['B']:3d}   ▓▓ acceptable")
    print(f"  Tier C (≥40):  {tiers['C']:3d}   ░░ marginal")
    print(f"  Tier D (<40):  {tiers['D']:3d}   ·· poor, discard")
    print(f"{'─'*90}\n")

    # Show worst variants (D tier) for transparency
    d_tier = [r for r in rows if r["tier"] == "D"]
    if d_tier:
        print("D-tier variants (discard):")
        for r in d_tier:
            print(f"  [{r['overall']:5.1f}] {r['suite']}/{r['source_task'][:40]}")
            print(f"         cf={r['cf_class']}  lang='{r['generated_language']}'")
            print(f"         issues: lang={r['score_language']} feas={r['score_feasibility']} "
                  f"cfv={r['score_cf_value']} dest={r['score_dest_fit']}")
        print()

    # Show top 10
    print("Top 10 variants:")
    for r in rows[:10]:
        print(f"  [{r['overall']:5.1f}] {r['suite']}/{r['source_task'][:40]}")
        print(f"         '{r['generated_language']}'")
    print()

    # Save CSV
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved ratings → {args.csv}")


if __name__ == "__main__":
    main()
