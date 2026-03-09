"""BDDL task reversal: transform forward tasks into backward tasks.

Given a BDDL file for "put the bowl on the plate", produces a BDDL file
where the bowl starts on the plate and must be placed back on the table.

Reversal rules:
  (On A B)         -> init: (On A B);       goal: (On A original_region_of_A)
  (On A FIX_REGION)-> init: (On A FIX_REGION); goal: (On A original_region_of_A)
  (In A REGION)    -> init: (In A REGION);  goal: (On A original_region_of_A)
  (Open R)         -> goal: (Close R)  [closed by default, no init change]
  (Close R)        -> init: (Open R);  goal dropped (already open)
  (Turnon F)       -> goal: (Turnoff F)  [off by default, no init change]
  (Turnoff F)      -> init: (Turnon F); goal dropped (already on)
"""

from __future__ import annotations

import re

from libero_infinity.bddl_preprocessor import (
    _extract_block,
    _find_closing_paren,
    _parse_declarations,
    _parse_language,
)

# ---------------------------------------------------------------------------
# Predicate parsing
# ---------------------------------------------------------------------------

_PRED_RE = re.compile(r"\((\w+)\s+([^\s()]+)(?:\s+([^\s()]+))?\)")


def _parse_predicates(block_text: str) -> list[tuple[str, ...]]:
    """Parse predicates like ``(On bowl_1 plate_1)`` from a BDDL block.

    Returns list of tuples: ``(predicate_name, arg1[, arg2])``.
    """
    results: list[tuple[str, ...]] = []
    for m in _PRED_RE.finditer(block_text):
        pred = m.group(1)
        # Skip the outer "And" wrapper
        if pred == "And":
            continue
        args = (m.group(2),) if m.group(3) is None else (m.group(2), m.group(3))
        results.append((pred,) + args)
    return results


# ---------------------------------------------------------------------------
# Language construction
# ---------------------------------------------------------------------------


def _humanize(instance_name: str) -> str:
    """``akita_black_bowl_1`` → ``akita black bowl``."""
    # Strip trailing _N instance suffix
    name = re.sub(r"_\d+$", "", instance_name)
    return name.replace("_", " ")


def _construct_language(
    reversed_goal_preds: list[tuple[str, ...]],
    original_language: str,
) -> str:
    """Build a human-readable description from the reversed goal predicates."""
    parts: list[str] = []
    for pred in reversed_goal_preds:
        ptype = pred[0]
        if ptype == "On" and len(pred) == 3:
            obj_human = _humanize(pred[1])
            parts.append(f"place the {obj_human} on the table")
        elif ptype == "Close" and len(pred) == 2:
            region_human = _humanize(pred[1])
            parts.append(f"close the {region_human}")
        elif ptype == "Open" and len(pred) == 2:
            region_human = _humanize(pred[1])
            parts.append(f"open the {region_human}")
        elif ptype == "Turnoff" and len(pred) == 2:
            fixture_human = _humanize(pred[1])
            parts.append(f"turn off the {fixture_human}")
        elif ptype == "Turnon" and len(pred) == 2:
            fixture_human = _humanize(pred[1])
            parts.append(f"turn on the {fixture_human}")

    if parts:
        return " and ".join(parts)
    return f"(reversed) {original_language}"


# ---------------------------------------------------------------------------
# Region widening
# ---------------------------------------------------------------------------

_MIN_REGION_SPAN = 0.10  # metres


def _widen_region_in_text(
    regions_text: str,
    region_name: str,
    margin: float,
) -> str:
    """Widen a region's ranges if its span is < _MIN_REGION_SPAN.

    Operates on the raw (:regions ...) block text.  Returns modified text.
    """
    # Find the region sub-block by name
    pat = re.compile(
        rf"\({re.escape(region_name)}\s",
        re.DOTALL,
    )
    m = pat.search(regions_text)
    if not m:
        return regions_text

    # Extract the balanced sub-block
    start = m.start()
    try:
        end = _find_closing_paren(regions_text, start) + 1
    except ValueError:
        return regions_text

    inner = regions_text[start:end]

    # Parse current ranges
    ranges_m = re.search(r"\(:ranges\s*\(\s*\(([^)]+)\)", inner)
    if not ranges_m:
        return regions_text

    vals = [float(v) for v in ranges_m.group(1).split()]
    if len(vals) != 4:
        return regions_text

    x_min, y_min, x_max, y_max = vals
    x_span = x_max - x_min
    y_span = y_max - y_min

    if x_span >= _MIN_REGION_SPAN and y_span >= _MIN_REGION_SPAN:
        return regions_text  # already wide enough

    # Widen around centre
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    new_x_min = cx - margin
    new_x_max = cx + margin
    new_y_min = cy - margin
    new_y_max = cy + margin

    old_ranges = ranges_m.group(1)
    new_ranges = f"{new_x_min} {new_y_min} {new_x_max} {new_y_max}"
    new_inner = inner.replace(old_ranges, new_ranges)
    return regions_text[:start] + new_inner + regions_text[end:]


# ---------------------------------------------------------------------------
# Main reversal
# ---------------------------------------------------------------------------


def reverse_bddl(
    bddl_content: str,
    return_region_margin: float = 0.05,
) -> str:
    """Reverse a BDDL task: swap initial and goal conditions.

    Args:
        bddl_content: Full text of the original BDDL file.
        return_region_margin: Half-width of the return region (metres).
            If the object's original init region is smaller than
            2 * margin, it will be widened to this size.

    Returns:
        Modified BDDL string with reversed task semantics.

    Raises:
        ValueError: If the BDDL cannot be reversed (missing blocks, etc).
    """
    # --- Parse blocks ---
    init_body = _extract_block(bddl_content, "init")
    goal_body = _extract_block(bddl_content, "goal")
    objects_body = _extract_block(bddl_content, "objects")

    if not init_body or not goal_body:
        raise ValueError("BDDL must have both (:init ...) and (:goal ...) blocks")

    original_language = _parse_language(bddl_content)

    # --- Parse predicates ---
    init_preds = _parse_predicates(init_body)
    goal_preds = _parse_predicates(goal_body)

    if not goal_preds:
        raise ValueError("No goal predicates found")

    # --- Identify movable objects ---
    movable_instances: set[str] = set()
    if objects_body:
        for inst, _cls in _parse_declarations(objects_body):
            movable_instances.add(inst)

    # --- Build original-region map for movable objects ---
    # e.g. "akita_black_bowl_1" → "main_table_akita_black_bowl_region"
    original_regions: dict[str, str] = {}
    for pred in init_preds:
        if pred[0] == "On" and len(pred) == 3:
            obj_name, target = pred[1], pred[2]
            if obj_name in movable_instances and target not in movable_instances:
                original_regions[obj_name] = target

    # --- Apply reversal rules ---
    new_init_preds: list[tuple[str, ...]] = []
    new_goal_preds: list[tuple[str, ...]] = []
    task_objects: set[str] = set()  # objects whose init changes
    regions_to_widen: list[str] = []  # region names needing widening

    for gpred in goal_preds:
        ptype = gpred[0]

        if ptype == "On" and len(gpred) == 3:
            obj_a, target_b = gpred[1], gpred[2]
            task_objects.add(obj_a)
            # Reversed init: object starts at goal location
            new_init_preds.append(("On", obj_a, target_b))
            # Reversed goal: object returns to its original table region
            orig_region = original_regions.get(obj_a)
            if orig_region:
                new_goal_preds.append(("On", obj_a, orig_region))
                regions_to_widen.append(orig_region)
            else:
                raise ValueError(f"Cannot reverse: no original table region found for {obj_a}")

        elif ptype == "In" and len(gpred) == 3:
            obj_a, container_region = gpred[1], gpred[2]
            task_objects.add(obj_a)
            # Reversed init: object starts inside the container
            new_init_preds.append(("In", obj_a, container_region))
            # Reversed goal: object returns to table
            orig_region = original_regions.get(obj_a)
            if orig_region:
                new_goal_preds.append(("On", obj_a, orig_region))
                regions_to_widen.append(orig_region)
            else:
                raise ValueError(f"Cannot reverse: no original table region found for {obj_a}")

        elif ptype == "Close" and len(gpred) == 2:
            region = gpred[1]
            # Reversed init: fixture starts open
            new_init_preds.append(("Open", region))
            # Reversed goal: dropped (already open is the start state)
            # But we need the drawer open for the robot to take things out,
            # so no Close goal. If the task also has an In predicate,
            # the goal is just placing the object on the table.

        elif ptype == "Open" and len(gpred) == 2:
            region = gpred[1]
            # No init change needed (closed by default)
            # Reversed goal: close it
            new_goal_preds.append(("Close", region))

        elif ptype == "Turnon" and len(gpred) == 2:
            fixture = gpred[1]
            # No init change (off by default)
            new_goal_preds.append(("Turnoff", fixture))

        elif ptype == "Turnoff" and len(gpred) == 2:
            fixture = gpred[1]
            new_init_preds.append(("Turnon", fixture))
            # Goal dropped (already on)

        else:
            raise ValueError(f"Unsupported goal predicate: {gpred}")

    if not new_goal_preds:
        raise ValueError(
            "Reversal produced no goal predicates — "
            "task may be its own reverse (e.g., Turnoff → Turnon)"
        )

    # --- Rebuild init block ---
    # Keep all non-task-object init predicates unchanged
    kept_init: list[tuple[str, ...]] = []
    for pred in init_preds:
        if pred[0] == "On" and len(pred) == 3 and pred[1] in task_objects:
            continue  # replaced by reversed init
        if pred[0] == "Open":
            # Check if the original had this Open and the reversed task
            # changes it — we'll re-add via new_init_preds if needed
            reversed_opens = {p[1] for p in new_init_preds if p[0] == "Open"}
            if pred[1] in reversed_opens:
                continue  # will be re-added
        kept_init.append(pred)

    all_init = kept_init + new_init_preds

    # --- Format predicates ---
    def _fmt_pred(pred: tuple[str, ...]) -> str:
        if len(pred) == 2:
            return f"({pred[0]} {pred[1]})"
        return f"({pred[0]} {pred[1]} {pred[2]})"

    init_lines = "\n".join(f"    {_fmt_pred(p)}" for p in all_init)
    new_init_block = f"(:init\n{init_lines}\n  )"

    goal_inner = " ".join(_fmt_pred(p) for p in new_goal_preds)
    new_goal_block = f"(:goal\n    (And {goal_inner})\n  )"

    # --- Widen return regions ---
    regions_body = _extract_block(bddl_content, "regions")
    new_regions_body = regions_body or ""
    for full_region_name in regions_to_widen:
        # full_region_name is like "main_table_akita_black_bowl_region"
        # The region definition uses just the suffix after the target
        # We need to find the matching region in the regions block
        # Try all known region name patterns
        for candidate in _extract_region_name(full_region_name):
            new_regions_body = _widen_region_in_text(
                new_regions_body,
                candidate,
                return_region_margin,
            )

    # --- Construct language ---
    new_language = _construct_language(new_goal_preds, original_language)

    # --- Reconstruct BDDL ---
    result = bddl_content

    # Replace language
    if original_language:
        result = result.replace(
            f"(:language {original_language})",
            f"(:language {new_language})",
        )

    # Replace init block
    init_start = result.find("(:init")
    if init_start >= 0:
        init_end = _find_closing_paren(result, init_start) + 1
        result = result[:init_start] + new_init_block + result[init_end:]

    # Replace goal block
    goal_start = result.find("(:goal")
    if goal_start >= 0:
        goal_end = _find_closing_paren(result, goal_start) + 1
        result = result[:goal_start] + new_goal_block + result[goal_end:]

    # Replace regions block if widened
    if regions_body and new_regions_body != regions_body:
        result = result.replace(
            f"(:regions{regions_body})",
            f"(:regions{new_regions_body})",
        )

    return result


def _extract_region_name(full_qualified: str) -> list[str]:
    """Given ``main_table_akita_black_bowl_region``, return candidate region names.

    The region definition in (:regions ...) uses the suffix after the target
    prefix.  We try common target prefixes.
    """
    common_targets = [
        "main_table_",
        "kitchen_table_",
        "living_room_table_",
        "study_table_",
        "table_",
    ]
    candidates = []
    for prefix in common_targets:
        if full_qualified.startswith(prefix):
            candidates.append(full_qualified[len(prefix) :])
    # Also try the full name as-is (fixture regions like white_cabinet_1_bottom_region)
    candidates.append(full_qualified)
    return candidates
