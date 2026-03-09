"""BDDL file preprocessor for object class substitution.

When running object perturbation, the canonical BDDL file references objects
by their original class (e.g. "akita_black_bowl_1 - akita_black_bowl").
Scenic samples a replacement asset class; this module rewrites the BDDL
string so LIBERO loads the correct MuJoCo XML asset.

The rewrite is purely textual — a small, targeted regex substitution in the
(:objects ...) block. Everything else (regions, goal predicates, fixtures,
language instruction) is left unchanged.
"""

from __future__ import annotations

import contextlib
import pathlib
import re
import tempfile


def _find_closing_paren(text: str, open_pos: int) -> int:
    """Find the index of the closing paren matching the one at ``open_pos``.

    Args:
        text: The full string to scan.
        open_pos: Index of the opening ``(`` character.

    Returns:
        Index of the matching ``)`` character.

    Raises:
        ValueError: If no matching closing paren is found.
    """
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"No matching closing paren found starting at position {open_pos}")


def _extract_block(content: str, keyword: str) -> str | None:
    """Extract content of a top-level (:<keyword> ...) block using paren matching.

    Returns the text between the keyword and its balanced closing paren,
    or None if the block is not found.
    """
    marker = f"(:{keyword}"
    start = content.find(marker)
    if start == -1:
        return None

    try:
        end = _find_closing_paren(content, start)
        return content[start + len(marker) : end]
    except ValueError:
        return None


def _parse_language(content: str) -> str:
    """Extract the language instruction from a BDDL file's ``(:language ...)`` block."""
    m = re.search(r"\(:language\s+(.+?)\)", content)
    return m.group(1).strip() if m else ""


def _parse_declarations(block_body: str) -> list[tuple[str, str]]:
    """Parse 'instance - class' declaration lines from a BDDL block body.

    Handles both single and multi-instance declarations:
      - ``bowl_1 - akita_black_bowl``
      - ``butter_1 butter_2 - butter``

    Returns list of (instance_name, class_name) tuples.
    """
    result: list[tuple[str, str]] = []
    for line in block_body.splitlines():
        line = line.strip()
        if " - " in line:
            parts = line.split(" - ")
            if len(parts) == 2:
                instances_str = parts[0].strip()
                cls = parts[1].strip()
                for inst in instances_str.split():
                    result.append((inst, cls))
    return result


def substitute_asset(
    bddl_content: str,
    original_class: str,
    replacement_class: str,
) -> str:
    """Replace every occurrence of `original_class` as an object type in BDDL.

    Only substitutes inside the (:objects ...) block. Fixture declarations and
    goal predicates that reference object instance names are unaffected.

    Args:
        bddl_content: Full text of the BDDL file.
        original_class: The canonical BDDL type to replace, e.g. "akita_black_bowl".
        replacement_class: The OOD asset class to substitute in.

    Returns:
        Modified BDDL string.

    Example::

        new_bddl = substitute_asset(bddl_text, "akita_black_bowl", "white_bowl")
    """
    if original_class == replacement_class:
        return bddl_content

    # Isolate the (:objects ...) block so we don't touch (:fixtures ...) etc.
    obj_block_re = re.compile(
        r"(?s)(\(:objects\s+)(.*?)(\))",
        re.MULTILINE,
    )

    def _rewrite_block(m: re.Match) -> str:
        prefix, body, suffix = m.group(1), m.group(2), m.group(3)
        # Replace "instance - original_class" → "instance - replacement_class"
        new_body = re.sub(
            rf"\b{re.escape(original_class)}\b",
            replacement_class,
            body,
        )
        return f"{prefix}{new_body}{suffix}"

    result = obj_block_re.sub(_rewrite_block, bddl_content)

    if result == bddl_content:
        raise ValueError(
            f"Object class '{original_class}' not found in (:objects ...) block. "
            "Check BDDL file and class name spelling."
        )
    return result


def substitute_multi(
    bddl_content: str,
    substitutions: dict[str, str],
) -> str:
    """Apply multiple class substitutions in one pass.

    Args:
        bddl_content: Full BDDL text.
        substitutions: Mapping from original_class → replacement_class.

    Returns:
        Modified BDDL string.
    """
    result = bddl_content
    for orig, repl in substitutions.items():
        if orig != repl:
            try:
                result = substitute_asset(result, orig, repl)
            except ValueError:
                pass  # class not present in this particular BDDL — skip silently
    return _merge_duplicate_object_declarations(result)


def _merge_duplicate_object_declarations(bddl_content: str) -> str:
    """Canonicalise duplicate class lines in (:objects ...) after substitution.

    LIBERO's BDDL parser indexes the objects block by class name and overwrites
    duplicate keys instead of merging them. When object perturbation rewrites
    e.g. ``alphabet_soup -> tomato_sauce`` in a task that already contains a
    ``tomato_sauce`` instance, we must collapse both declaration lines into one.
    """
    obj_marker = "(:objects"
    obj_start = bddl_content.find(obj_marker)
    if obj_start == -1:
        return bddl_content

    obj_end = _find_closing_paren(bddl_content, obj_start)
    _obj_block = bddl_content[obj_start : obj_end + 1]
    obj_body = bddl_content[obj_start + len(obj_marker) : obj_end]

    class_to_instances: dict[str, list[str]] = {}
    class_order: list[str] = []
    for inst, cls in _parse_declarations(obj_body):
        if cls not in class_to_instances:
            class_order.append(cls)
            class_to_instances[cls] = []
        class_to_instances[cls].append(inst)

    indent = "    "
    merged_lines = [f"{indent}{' '.join(class_to_instances[cls])} - {cls}" for cls in class_order]
    merged_block = f"{obj_marker}\n" + "\n".join(merged_lines) + "\n  )"
    return bddl_content[:obj_start] + merged_block + bddl_content[obj_end + 1 :]


@contextlib.contextmanager
def patched_bddl(
    source_path: str | pathlib.Path,
    substitutions: dict[str, str],
):
    """Context manager: write a patched BDDL to a temp file, yield its path.

    Usage::

        with patched_bddl("task.bddl", {"akita_black_bowl": "white_bowl"}) as tmp:
            env = OffScreenRenderEnv(bddl_file_name=tmp, ...)
    """
    source_path = pathlib.Path(source_path)
    original = source_path.read_text()
    patched = substitute_multi(original, substitutions)

    with patched_bddl_from_string(patched, stem=source_path.stem) as tmp:
        yield tmp


@contextlib.contextmanager
def patched_bddl_from_string(content: str, stem: str = "reversed"):
    """Write arbitrary BDDL content to a temp file, yield its path, clean up.

    Usage::

        with patched_bddl_from_string(reversed_text) as tmp:
            env = OffScreenRenderEnv(bddl_file_name=tmp, ...)
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".bddl",
        prefix=f"libero_inf_{stem}_",
        delete=False,
    ) as f:
        f.write(content)
        tmp_path = f.name

    try:
        yield tmp_path
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)


@contextlib.contextmanager
def bddl_for_scene(
    scene,
    bddl_path: str,
    orig_obj_classes: dict[str, str],
):
    """Yield the effective BDDL path for a scene, handling temp file cleanup.

    If the scene has asset substitutions (via ``chosen_asset`` /
    ``perturb_class`` in scene.params or per-object ``asset_class``
    attributes), writes a patched BDDL to a temp file and yields its path.
    Otherwise yields the original *bddl_path*.

    This is the single source of truth for BDDL substitution resolution,
    used by both ``eval.py`` and ``gym_env.py``.
    """
    subs = {}
    for obj in scene.objects:
        asset_cls = getattr(obj, "asset_class", "")
        libero_name = getattr(obj, "libero_name", "")
        if libero_name and asset_cls:
            orig_cls = orig_obj_classes.get(libero_name, "")
            if orig_cls and orig_cls != asset_cls:
                subs[orig_cls] = asset_cls

    if not subs:
        chosen_asset = scene.params.get("chosen_asset")
        perturb_class = scene.params.get("perturb_class")
        if chosen_asset and perturb_class and chosen_asset != perturb_class:
            subs[perturb_class] = chosen_asset

    if subs:
        with patched_bddl(bddl_path, subs) as tmp:
            yield tmp
        return

    yield bddl_path


def add_distractor_objects(
    bddl_content: str,
    distractors: list[tuple[str, str]],
) -> str:
    """Add distractor (non-task) objects to a BDDL file.

    Inserts new object declarations into the (:objects ...) block.
    Does NOT add placement predicates to (:init ...) — distractor positions
    are injected directly into MuJoCo via set_joint_qpos in simulator.py.

    When a distractor shares a class with an existing task object, the
    distractor instance is merged into the existing declaration line.
    This is required because LIBERO's BDDL parser overwrites (rather than
    appends) when the same class key appears twice.

    Args:
        bddl_content: Full BDDL text.
        distractors: List of (instance_name, object_class) pairs,
            e.g. [("distractor_0", "cream_cheese")].

    Returns:
        Modified BDDL string with distractors added.
    """
    if not distractors:
        return bddl_content

    obj_marker = "(:objects"
    obj_start = bddl_content.find(obj_marker)
    if obj_start == -1:
        raise ValueError("No (:objects ...) block found in BDDL")

    obj_end = _find_closing_paren(bddl_content, obj_start)
    _obj_block = bddl_content[obj_start : obj_end + 1]
    obj_body = bddl_content[obj_start + len(obj_marker) : obj_end]

    # Parse existing class → [instances] mapping
    existing: dict[str, list[str]] = {}
    for inst, cls in _parse_declarations(obj_body):
        existing.setdefault(cls, []).append(inst)

    # Separate distractors into merge-able vs new classes
    to_merge: dict[str, list[str]] = {}  # existing class → new instances
    new_classes: dict[str, list[str]] = {}  # brand-new class → instances
    for inst, cls in distractors:
        if cls in existing:
            to_merge.setdefault(cls, []).append(inst)
        else:
            new_classes.setdefault(cls, []).append(inst)

    # Merge into existing declaration lines
    result_block = _obj_block
    for cls, new_insts in to_merge.items():
        # Find the "inst1 inst2 - class" line and prepend new instances
        pattern = re.compile(
            rf"([ \t]*)([^\n]*?)\s+-\s+{re.escape(cls)}\b",
        )

        def _merge(m: re.Match) -> str:
            indent = m.group(1)
            orig_insts = m.group(2).strip()
            all_insts = f"{orig_insts} {' '.join(new_insts)}"
            return f"{indent}{all_insts} - {cls}"

        result_block = pattern.sub(_merge, result_block, count=1)

    # Append new class lines before closing paren
    if new_classes:
        new_lines = "\n".join(
            f"    {' '.join(insts)} - {cls}" for cls, insts in new_classes.items()
        )
        close_idx = result_block.rfind(")")
        result_block = (
            result_block[:close_idx] + "\n" + new_lines + "\n  " + result_block[close_idx:]
        )

    return bddl_content[:obj_start] + result_block + bddl_content[obj_end + 1 :]


def generate_cf_bddls(bddl_content: str) -> list[tuple[str, str]]:
    """Generate counterfactual BDDL variants by swapping the goal object.

    For a task "Put the bowl on the plate", generates CF variants like
    "Put the cream cheese on the plate" — same scene layout, same destination,
    but the language instruction targets a *different* graspable object that
    happens to be present in the scene.

    Only works for tasks with ``On`` or ``In`` goal predicates (≈95% of LIBERO
    tasks).  Tasks with ``Open``/``Close``/``TurnOn``/``TurnOff`` goals are
    returned as an empty list.

    Algorithm
    ---------
    1. Parse ``:goal`` → find ``(On/In source dest)``
    2. Parse ``:objects`` → all graspable instances (non-fixtures)
    3. For each *other* graspable instance (not source, not dest):
       a. Rewrite ``:goal`` with the CF instance
       b. Rewrite ``:language`` with a natural-language phrase
       c. Rewrite ``:obj_of_interest`` to list the CF instance + dest
    4. Return list of ``(filename_suffix, cf_bddl_text)`` pairs

    Args:
        bddl_content: Full text of the original BDDL file.

    Returns:
        List of ``(suffix, cf_bddl)`` pairs where *suffix* is a short string
        suitable for appending to the original filename stem (e.g.
        ``"_cf_cream_cheese"``), and *cf_bddl* is the modified BDDL text.
        Returns an empty list if no CF variants can be generated.
    """
    import re as _re

    # ── 1. Parse goal predicate ──────────────────────────────────────────────
    goal_block = _extract_block(bddl_content, "goal")
    if not goal_block:
        return []

    pred_re = _re.compile(r"\((On|In)\s+([^\s()]+)\s+([^\s()]+)\)")
    goal_match = pred_re.search(goal_block)
    if not goal_match:
        return []  # Open/Close/TurnOn/TurnOff — not swappable

    predicate = goal_match.group(1)  # "On" or "In"
    source_inst = goal_match.group(2)  # e.g. "akita_black_bowl_1"
    dest_inst = goal_match.group(3)  # e.g. "plate_1"

    # ── 2. Parse objects block ───────────────────────────────────────────────
    obj_classes = parse_object_classes(bddl_content)
    if not obj_classes:
        return []

    # Parse fixtures — instances declared in (:fixtures ...) are not graspable
    fixtures_block = _extract_block(bddl_content, "fixtures") or ""
    fixture_instances: set[str] = set()
    for line in fixtures_block.splitlines():
        line = line.strip()
        if " - " in line:
            insts = line.split(" - ")[0].strip().split()
            fixture_instances.update(insts)

    # ── 3. Build CF variants ─────────────────────────────────────────────────

    # Natural-language display names for classes with awkward generated phrases.
    # Used in the language instruction only — BDDL instance names are unchanged.
    _DISPLAY_NAMES: dict[str, str] = {
        "akita_black_bowl": "black bowl",
        "glazed_rim_porcelain_ramekin": "ramekin",
        "white_yellow_mug": "yellow mug",
        "chefmate_8_frypan": "frying pan",
        "porcelain_mug": "mug",
        "new_salad_dressing": "salad dressing",
    }

    # Visual category groupings — swapping within a category is a weak
    # grounding test since the objects look similar.
    _VISUAL_CATEGORY: dict[str, str] = {
        "akita_black_bowl": "bowl",
        "white_bowl": "bowl",
        "glazed_rim_porcelain_ramekin": "bowl",
        "plate": "bowl",
        "chefmate_8_frypan": "cookware",
        "red_coffee_mug": "mug",
        "white_yellow_mug": "mug",
        "porcelain_mug": "mug",
        "moka_pot": "mug",
        "black_book": "book",
        "yellow_book": "book",
        "wine_bottle": "bottle",
        "ketchup": "bottle",
        "milk": "bottle",
        "orange_juice": "bottle",
        "tomato_sauce": "bottle",
        "bbq_sauce": "bottle",
        "salad_dressing": "bottle",
        "new_salad_dressing": "bottle",
        "cream_cheese": "carton",
        "butter": "carton",
        "chocolate_pudding": "carton",
        "alphabet_soup": "carton",
        "cookies": "carton",
        "basket": "container",
        "wooden_tray": "container",
        "desk_caddy": "container",
    }

    # Physical incompatibility: (cf_category, dest_surface) pairs that produce
    # implausible placements. dest_surface is derived from the destination name.
    _INCOMPATIBLE: set[tuple[str, str]] = {
        ("container", "bowl"),  # large tray/caddy balanced on small bowl
        ("bowl", "bowl"),  # plate/bowl stacked on another small bowl
        ("bowl", "stove"),  # bowl on cooking surface — wrong object type
        ("carton", "stove"),  # food carton on stove — semantically odd
        ("book", "stove"),  # book on stove — fire hazard / nonsensical
        ("mug", "stove"),  # mug on stove — odd for robot task
        ("book", "rack"),  # book on wine rack — nonsensical
    }

    # Tall/unstable objects that should be placed "in" a curved/concave
    # surface (bowl, plate) rather than "on" it — avoids physically misleading
    # language like "Put the wine bottle on the bowl".
    _TALL_UNSTABLE: set[str] = {
        "wine_bottle",
        "ketchup",
        "tomato_sauce",
        "bbq_sauce",
        "moka_pot",
    }

    # Cross-category preference groups for CF object selection.
    # Objects sharing a group are visually similar → weaker grounding test.
    # Prefer cross-group swaps; only fall back to same-group if necessary.
    _CF_CATEGORY: dict[str, str] = {
        "alphabet_soup": "food_can",
        "tomato_sauce": "food_can",
        "bbq_sauce": "food_can",
        "ketchup": "food_can",
        "salad_dressing": "food_can",
        "new_salad_dressing": "food_can",
        "cream_cheese": "food_box",
        "butter": "food_box",
        "chocolate_pudding": "food_box",
        "red_coffee_mug": "mug",
        "white_yellow_mug": "mug",
        "black_book": "book",
        "akita_black_bowl": "bowl_plate",
        "white_bowl": "bowl_plate",
        "plate": "bowl_plate",
        "wine_bottle": "bottle",
    }

    def _class_to_phrase(cls: str) -> str:
        name = _DISPLAY_NAMES.get(cls, cls)
        return name.replace("_", " ")

    def _language_for_cf(cf_class: str, prep: str) -> str:
        obj_phrase = _class_to_phrase(cf_class)
        # Override "On" → "In" for tall/unstable objects on curved/concave
        # surfaces (bowl, plate): "Put the wine bottle in the bowl" is more
        # natural and physically accurate than "on the bowl".
        eff_prep = prep
        if prep == "On" and cf_class in _TALL_UNSTABLE and dest_surface == "bowl":
            eff_prep = "In"
        if eff_prep == "In":
            return f"Put the {obj_phrase} in the {dest_phrase_for_lang}"
        return f"Put the {obj_phrase} on the {dest_phrase_for_lang}"

    def _region_to_phrase(region: str) -> str:
        """Convert a BDDL region ID to a ≤2-word human-readable phrase.

        basket_1_contain_region          → 'basket'
        main_table_stove_front_region    → 'stove'
        kitchen_table_porcelain_mug_...  → 'kitchen table'  (object landmark stripped)
        living_room_table_plate_left_... → 'living room table'
        """
        # Numbered container regions: 'basket_1_contain_region' → 'basket'
        m = _re.match(r"(.+?)_\d+_\w+_region$", region)
        if m:
            base = m.group(1)
            return _DISPLAY_NAMES.get(base, base).replace("_", " ")

        # Table surface regions — strip table prefix + position/object suffix.
        # Keep only the table name or the first meaningful fixture keyword.
        _FIXTURE_KEYWORDS = (
            "stove",
            "cabinet",
            "shelf",
            "rack",
            "microwave",
            "drawer",
            "fridge",
        )
        s = region
        for kw in _FIXTURE_KEYWORDS:
            if kw in region:
                return kw  # 'main_table_stove_front_region' → 'stove'

        # Fall back to the table name (1-2 words)
        table_m = _re.match(r"(main|kitchen|living_room|study)_table_", region)
        if table_m:
            table_name = table_m.group(1).replace("_", " ")  # 'living room'
            return f"{table_name} table"

        # Generic fallback
        s = _re.sub(r"_region$", "", region).replace("_", " ")
        return " ".join(s.split()[:2])

    def _region_to_container_inst(region: str) -> str:
        m = _re.match(r"(.+_\d+)_\w+_region$", region)
        return m.group(1) if m else ""

    def _dest_surface_type(inst: str) -> str:
        """Classify the destination as a surface type for incompatibility checks."""
        if inst in obj_classes:
            return _VISUAL_CATEGORY.get(obj_classes[inst], "object")
        name = inst.lower()
        if "stove" in name:
            return "stove"
        if "cabinet" in name:
            return "shelf"
        if "shelf" in name:
            return "shelf"
        if "rack" in name:
            return "rack"
        return "table"

    if dest_inst in obj_classes:
        dest_phrase_for_lang = _class_to_phrase(obj_classes[dest_inst])
        container_inst = dest_inst
    else:
        dest_phrase_for_lang = _region_to_phrase(dest_inst)
        container_inst = _region_to_container_inst(dest_inst)

    dest_class = obj_classes.get(dest_inst, dest_inst)
    dest_surface = _dest_surface_type(dest_inst)

    # Extract the object class embedded in the region name (e.g. "plate" from
    # "main_table_plate_region") so we can skip "put the plate on the plate".
    _dest_region_class = ""
    if dest_inst not in obj_classes and dest_inst.endswith("_region"):
        stripped = _re.sub(r"^(?:main|kitchen|living_room|study)_table_", "", dest_inst)
        stripped = _re.sub(r"_(?:front|back|left|right|top|bottom|side)?_region$", "", stripped)
        stripped = _re.sub(r"_region$", "", stripped)
        stripped = _re.sub(r"_\d+$", "", stripped)
        _dest_region_class = stripped  # e.g. "plate", "akita_black_bowl"

    # Two buckets: prefer cross-category swaps; only fall back to same-category
    # if the scene offers no cross-category alternatives.
    cross_category_results: list[tuple[str, str]] = []
    same_category_results: list[tuple[str, str]] = []

    source_class = obj_classes.get(source_inst, "")
    source_cf_category = _CF_CATEGORY.get(source_class)
    seen_cf_classes: set[str] = set()

    for cf_inst, cf_class in obj_classes.items():
        if cf_inst == source_inst:
            continue  # skip the original source
        if cf_inst in (dest_inst, container_inst):
            continue  # skip the destination object / container
        if cf_inst in fixture_instances:
            continue  # skip fixtures
        if cf_class == source_class:
            continue  # identical class → language would be the same
        if cf_class == dest_class or cf_class == _dest_region_class:
            continue  # "put the plate on the plate" — nonsensical
        cf_category = _VISUAL_CATEGORY.get(cf_class, "object")
        if (cf_category, dest_surface) in _INCOMPATIBLE:
            continue  # physically implausible placement
        if cf_class in seen_cf_classes:
            continue  # already generated a variant for this class
        seen_cf_classes.add(cf_class)

        # (a) Rewrite :goal
        new_goal_pred = f"({predicate} {cf_inst} {dest_inst})"
        new_goal_block = goal_block.replace(goal_match.group(0), new_goal_pred)
        cf_text = bddl_content.replace(
            f"(:goal{goal_block}",
            f"(:goal{new_goal_block}",
        )

        # (b) Rewrite :language
        new_lang = _language_for_cf(cf_class, predicate)
        cf_text = _re.sub(
            r"\(:language\s+[^)]+\)",
            f"(:language {new_lang})",
            cf_text,
        )

        # (c) Rewrite :obj_of_interest — replace source_inst with cf_inst
        def _rewrite_ooi(m: _re.Match) -> str:
            block = m.group(0)
            return block.replace(source_inst, cf_inst)

        cf_text = _re.sub(
            r"\(:obj_of_interest[^)]*\)",
            _rewrite_ooi,
            cf_text,
        )

        suffix = f"_cf_{cf_class}"
        variant = (suffix, cf_text)

        # Bucket by cross-category preference: same _CF_CATEGORY as source →
        # weaker grounding test; different (or uncategorised) → stronger test.
        cf_cf_category = _CF_CATEGORY.get(cf_class)
        if source_cf_category and cf_cf_category == source_cf_category:
            same_category_results.append(variant)
        else:
            cross_category_results.append(variant)

    # Return cross-category variants if any exist; fall back to same-category.
    return cross_category_results if cross_category_results else same_category_results


def parse_object_classes(bddl_content: str) -> dict[str, str]:
    """Extract {instance_name: class_name} from (:objects ...) block.

    Handles both single-instance and multi-instance declarations:
      - ``akita_black_bowl_1 - akita_black_bowl``
      - ``butter_1 butter_2 - butter``
      - ``akita_black_bowl_1 akita_black_bowl_2 akita_black_bowl_3 - akita_black_bowl``

    Returns:
        Dict mapping e.g. "akita_black_bowl_1" → "akita_black_bowl".
    """
    body = _extract_block(bddl_content, "objects")
    if not body:
        return {}
    return {inst: cls for inst, cls in _parse_declarations(body)}
