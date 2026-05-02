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

    Substitutions are applied via a two-phase placeholder pass to avoid
    chained collisions. Without this, substitutions like
    ``{"alphabet_soup": "tomato_sauce", "tomato_sauce": "popcorn"}`` would
    mis-rewrite ``alphabet_soup`` instances all the way to ``popcorn`` because
    the second pass would re-match the output of the first.

    Args:
        bddl_content: Full BDDL text.
        substitutions: Mapping from original_class → replacement_class.

    Returns:
        Modified BDDL string.
    """
    # Drop identity / no-op subs.
    effective = {k: v for k, v in substitutions.items() if k != v}
    if not effective:
        return bddl_content

    # Phase 1: orig → unique placeholder. Placeholders must be impossible to
    # collide with any real BDDL token (no spaces, parens, or hyphens).
    placeholders: dict[str, str] = {}
    result = bddl_content
    for idx, orig in enumerate(effective):
        placeholder = f"__libinf_sub_placeholder_{idx}__"
        placeholders[orig] = placeholder
        try:
            result = substitute_asset(result, orig, placeholder)
        except ValueError:
            # Class not present in this BDDL — skip silently. Drop the
            # placeholder so phase 2 does not try to rewrite something that
            # never landed in the text.
            placeholders.pop(orig)

    # Phase 2: placeholder → final replacement.
    for orig, placeholder in placeholders.items():
        repl = effective[orig]
        try:
            result = substitute_asset(result, placeholder, repl)
        except ValueError:
            # Should be unreachable — phase 1 guarantees the placeholder is
            # present — but stay defensive.
            pass

    return _merge_duplicate_object_declarations(result)


def substitute_per_instance(
    bddl_content: str,
    instance_substitutions: dict[str, str],
) -> str:
    """Apply per-instance class substitutions.

    Unlike :func:`substitute_multi`, this rewrites each instance's class
    independently. Required when multiple instances share an original class
    but receive *different* replacement classes (e.g. ``butter_1 → cream_cheese``
    while ``butter_2 → popcorn``). Splits multi-instance declarations
    (``butter_1 butter_2 - butter``) into per-class lines so each instance
    points at its own asset class. Goal predicates and other blocks are
    untouched (instance names are preserved verbatim).

    Args:
        bddl_content: Full BDDL text.
        instance_substitutions: Mapping from instance_name → replacement_class.

    Returns:
        Modified BDDL string with each instance assigned its requested class.
    """
    if not instance_substitutions:
        return bddl_content

    obj_marker = "(:objects"
    obj_start = bddl_content.find(obj_marker)
    if obj_start == -1:
        return bddl_content

    obj_end = _find_closing_paren(bddl_content, obj_start)
    obj_body = bddl_content[obj_start + len(obj_marker) : obj_end]

    # Group resolved per-instance class assignments, preserving discovery order.
    inst_classes: list[tuple[str, str]] = []
    for inst, cls in _parse_declarations(obj_body):
        new_cls = instance_substitutions.get(inst, cls)
        inst_classes.append((inst, new_cls))

    # Re-group by class (preserving first-seen order) so identical classes
    # share a single declaration line.
    class_to_instances: dict[str, list[str]] = {}
    class_order: list[str] = []
    for inst, cls in inst_classes:
        if cls not in class_to_instances:
            class_order.append(cls)
            class_to_instances[cls] = []
        class_to_instances[cls].append(inst)

    indent = "    "
    merged_lines = [f"{indent}{' '.join(class_to_instances[cls])} - {cls}" for cls in class_order]
    merged_block = f"{obj_marker}\n" + "\n".join(merged_lines) + "\n  )"
    return bddl_content[:obj_start] + merged_block + bddl_content[obj_end + 1 :]


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
    # Collect per-instance assignments first so we can detect same-class
    # multi-instance overwrites and route through the per-instance path.
    inst_subs: dict[str, str] = {}
    for obj in scene.objects:
        asset_cls = getattr(obj, "asset_class", "")
        libero_name = getattr(obj, "libero_name", "")
        if libero_name and asset_cls:
            orig_cls = orig_obj_classes.get(libero_name, "")
            if orig_cls and orig_cls != asset_cls:
                inst_subs[libero_name] = asset_cls

    # Detect whether any original class has multiple instances with *differing*
    # asset assignments (or a mix of "rewrite" + "keep canonical"). When that
    # happens we must split per-instance to avoid the class-level rewrite
    # silently collapsing them into the same replacement class.
    needs_per_instance = False
    by_orig: dict[str, set[str]] = {}
    for libero_name, repl in inst_subs.items():
        orig_cls = orig_obj_classes.get(libero_name, "")
        if not orig_cls:
            continue
        by_orig.setdefault(orig_cls, set()).add(repl)
    for orig_cls, repls in by_orig.items():
        # Count how many *total* instances share this original class.
        siblings = [n for n, c in orig_obj_classes.items() if c == orig_cls]
        rewritten = [n for n in siblings if n in inst_subs]
        if len(siblings) > 1 and (len(repls) > 1 or len(rewritten) != len(siblings)):
            needs_per_instance = True
            break

    if needs_per_instance:
        source_path = pathlib.Path(bddl_path)
        original = source_path.read_text()
        patched = substitute_per_instance(original, inst_subs)
        with patched_bddl_from_string(patched, stem=source_path.stem) as tmp:
            yield tmp
        return

    # Fall back to the simpler class-level rewrite path.
    subs: dict[str, str] = {}
    for libero_name, repl in inst_subs.items():
        orig_cls = orig_obj_classes.get(libero_name, "")
        if orig_cls:
            subs[orig_cls] = repl

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
    pred_matches = list(pred_re.finditer(goal_block))
    if not pred_matches:
        return []  # Open/Close/TurnOn/TurnOff — not swappable
    if len(pred_matches) > 1:
        # Multi-predicate goal (e.g. (And (On a b) (On c d))). Rewriting only
        # the first predicate while leaving the rest untouched produces a CF
        # whose language describes one swap but whose goal still requires the
        # other original predicates — semantically inconsistent. Skip.
        return []
    goal_match = pred_matches[0]

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

    # Container/landmark classes whose names commonly appear inside region
    # identifiers (e.g. ``main_table_basket_region``,
    # ``study_table_desk_caddy_front_left_contain_region``). We probe these
    # before falling back to the table-prefix heuristic so that a region whose
    # surface is a basket / tray / caddy is described as such rather than
    # losing the landmark to "kitchen table" / "study table".
    _REGION_LANDMARKS: tuple[str, ...] = (
        "desk_caddy",
        "wooden_tray",
        "white_storage_box",
        "bowl_drainer",
        "basket",
        "tray",
    )

    def _region_to_phrase(region: str) -> str:
        """Convert a BDDL region ID to a ≤2-word human-readable phrase.

        basket_1_contain_region          → 'basket'
        main_table_stove_front_region    → 'stove'
        kitchen_table_porcelain_mug_...  → 'kitchen table'  (object landmark stripped)
        living_room_table_plate_left_... → 'living room table'
        study_table_desk_caddy_front_... → 'desk caddy'
        main_table_basket_region         → 'basket'
        """
        # Numbered container regions: 'basket_1_contain_region' → 'basket'
        m = _re.match(r"(.+?)_\d+_\w+_region$", region)
        if m:
            base = m.group(1)
            # Strip a leading table prefix if present
            base = _re.sub(r"^(?:main|kitchen|living_room|study)_table_", "", base)
            for landmark in _REGION_LANDMARKS:
                if landmark in base:
                    return _DISPLAY_NAMES.get(landmark, landmark).replace("_", " ")
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
        for kw in _FIXTURE_KEYWORDS:
            if kw in region:
                return kw  # 'main_table_stove_front_region' → 'stove'

        # Container/landmark embedded in region name (no number) — pick that
        # before defaulting to the table label, otherwise we silently lose
        # the actual placement surface.
        for landmark in _REGION_LANDMARKS:
            if landmark in region:
                return _DISPLAY_NAMES.get(landmark, landmark).replace("_", " ")

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

        # (c) Rewrite :obj_of_interest — replace source_inst with cf_inst,
        # bounded by whitespace / paren delimiters to avoid swallowing
        # substring matches (e.g. ``bowl_1`` colliding with ``bowl_10``).
        _bounded = _re.compile(rf"(?<![A-Za-z0-9_]){_re.escape(source_inst)}(?![A-Za-z0-9_])")

        def _rewrite_ooi(m: _re.Match) -> str:
            block = m.group(0)
            return _bounded.sub(cf_inst, block)

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


# ---------------------------------------------------------------------------
# Arena swap — rewrite the workspace fixture to a different table arena
# ---------------------------------------------------------------------------


# Arena-class compatibility table. Each arena's half-extents are sourced from
# LIBERO's ``libero_*_manipulation.py`` problem classes (full_size / 2):
#   table / kitchen_table / study_table → 1.0 × 1.2 m
#   living_room_table / coffee_table     → 0.70 × 1.6 m
# Two arenas are layout-compatible when each authored region rectangle from
# the original BDDL fits inside the target arena's half-extents (so the
# rewrite doesn't push placements off the table edge).
_ARENA_HALF_EXTENTS: dict[str, tuple[float, float]] = {
    "table": (0.5, 0.6),
    "kitchen_table": (0.5, 0.6),
    "study_table": (0.5, 0.6),
    "living_room_table": (0.35, 0.8),
    "coffee_table": (0.35, 0.8),
}


def swap_arena(bddl_content: str, target_arena_class: str) -> str | None:
    """Rewrite a BDDL to use a different table arena while preserving
    region coordinates.

    The (:fixtures …) declaration is rewritten to declare the target
    arena class and every region's ``(:target …)`` is re-pointed at the
    new workspace fixture instance. The region range coordinates are
    kept verbatim — LIBERO's table arenas all use a centred-at-origin
    coordinate convention (table centre at (0, 0, table_z)), so x/y
    region ranges are arena-agnostic *as long as* every original region
    rectangle fits inside the target arena's half-extents.

    Args:
        bddl_content: Full text of the source BDDL.
        target_arena_class: One of the keys in ``_ARENA_HALF_EXTENTS``.

    Returns:
        Rewritten BDDL string, or ``None`` if the swap is rejected
        (incompatible region geometry, or the source declares no
        recognised workspace fixture).
    """
    import re as _re

    if target_arena_class not in _ARENA_HALF_EXTENTS:
        return None

    fixtures_block = _extract_block(bddl_content, "fixtures") or ""
    workspace_inst: str | None = None
    workspace_class: str | None = None
    for line in fixtures_block.splitlines():
        line = line.strip()
        if " - " not in line:
            continue
        insts_str, cls = line.split(" - ", 1)
        cls = cls.strip()
        if cls in _ARENA_HALF_EXTENTS:
            workspace_inst = insts_str.strip().split()[0]
            workspace_class = cls
            break
    if workspace_inst is None or workspace_class is None:
        return None
    if workspace_class == target_arena_class:
        return None  # no-op swap

    # Geometric compatibility check: every region rectangle must fit in
    # the target arena's half-extents (with a 4 cm margin to match
    # ``planner.position._TABLE_*_MARGIN``).
    target_x_half, target_y_half = _ARENA_HALF_EXTENTS[target_arena_class]
    margin = 0.04
    region_re = _re.compile(
        r"\(\s*:ranges\s*\(\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s*\)"
    )
    for m in region_re.finditer(bddl_content):
        x_min, y_min, x_max, y_max = (float(s) for s in m.groups())
        if (
            x_min < -target_x_half + margin
            or x_max > target_x_half - margin
            or y_min < -target_y_half + margin
            or y_max > target_y_half - margin
        ):
            return None  # region would clip off the new table

    # Compute target instance name. Convention: <class>_<n>; we use the
    # source instance number when shaped that way, else default to _1.
    suffix = (
        workspace_inst[len(workspace_class) :]
        if workspace_inst.startswith(workspace_class)
        else "_1"
    )
    target_inst = (
        f"{target_arena_class}{suffix}"
        if suffix.startswith("_") and suffix[1:].isdigit()
        else target_arena_class
    )

    # Rewrite (:fixtures ...) declarations: swap the workspace instance/class.
    def _fix_block(m: _re.Match) -> str:
        body = m.group(2)
        new_body = _re.sub(
            rf"\b{_re.escape(workspace_inst)}\b\s*-\s*\b{_re.escape(workspace_class)}\b",
            f"{target_inst} - {target_arena_class}",
            body,
            count=1,
        )
        return f"{m.group(1)}{new_body}{m.group(3)}"

    out = _re.sub(
        r"(?s)(\(:fixtures\s+)(.*?)(\))",
        _fix_block,
        bddl_content,
        count=1,
    )

    # Rewrite all references to the workspace instance everywhere else
    # (region targets, init predicates that anchor to it, goal predicates).
    bounded = _re.compile(rf"(?<![A-Za-z0-9_]){_re.escape(workspace_inst)}(?![A-Za-z0-9_])")
    out = bounded.sub(target_inst, out)

    return out


# ---------------------------------------------------------------------------
# Task perturbations — generate task-level BDDL variants
# ---------------------------------------------------------------------------


_PREDICATE_INVERSES: dict[str, tuple[str, str]] = {
    # (forward_predicate) → (negated_predicate, language_phrase_for_negated)
    "Open": ("Close", "Close"),
    "Close": ("Open", "Open"),
    "Turnon": ("Turnoff", "Turn off"),
    "Turnoff": ("Turnon", "Turn on"),
}

_VISIBLE_COLOR_VARIANTS: dict[str, list[str]] = {
    # Sub-class variant pools keyed by canonical class. Each variant must be
    # a *visually distinct* version of the same functional category — picking
    # red_coffee_mug instead of white_yellow_mug exercises the policy's
    # color-grounding ability without changing the manipulation skill.
    # These are the variants whose change would meaningfully change the
    # natural-language description ("yellow mug" vs "red mug"), as opposed
    # to variants that change shape only.
    "akita_black_bowl": ["akita_black_bowl", "white_bowl"],
    "white_bowl": ["white_bowl", "akita_black_bowl"],
    "red_coffee_mug": ["red_coffee_mug", "white_yellow_mug", "porcelain_mug"],
    "white_yellow_mug": ["white_yellow_mug", "red_coffee_mug", "porcelain_mug"],
    "porcelain_mug": ["porcelain_mug", "red_coffee_mug", "white_yellow_mug"],
    "black_book": ["black_book", "yellow_book"],
    "yellow_book": ["yellow_book", "black_book"],
    "wine_bottle": ["wine_bottle"],  # no color variant in registry
}


def _bare_class_phrase(cls: str) -> str:
    """Convert a class name to a human phrase used in language strings."""
    overrides = {
        "akita_black_bowl": "black bowl",
        "white_yellow_mug": "yellow mug",
        "red_coffee_mug": "red mug",
        "porcelain_mug": "white mug",
        "white_bowl": "white bowl",
        "yellow_book": "yellow book",
        "black_book": "black book",
        "chefmate_8_frypan": "frying pan",
        "wine_bottle": "wine bottle",
    }
    return overrides.get(cls, cls.replace("_", " "))


def generate_task_perturbed_bddls(
    bddl_content: str,
    *,
    include_destination_swaps: bool = True,
    include_predicate_negations: bool = True,
    include_compositional: bool = True,
    include_color_swaps: bool = True,
) -> list[tuple[str, str]]:
    """Generate task-level BDDL perturbations.

    Variant families produced (each toggleable):

    1. **Destination swaps** (``include_destination_swaps``): for each
       ``(On|In source dest)`` goal predicate, replace ``dest`` with another
       graspable scene object. Produces e.g. "put the bowl on the plate"
       → "put the bowl on the wine bottle". Complements
       :func:`generate_cf_bddls` which swaps the source.
    2. **Predicate negations** (``include_predicate_negations``): for each
       ``(Open|Close|Turnon|Turnoff fixture)`` goal predicate, emit the
       inverse predicate as a separate task. Produces e.g.
       "turn on the stove" → "Turn off the stove".
    3. **Compositional do-also tasks** (``include_compositional``): when a
       goal has a single On/In predicate, emit a 2-predicate variant that
       additionally requires placing a *second* graspable object on a
       second free destination. Produces "and also put the cream cheese
       in the basket"-style multi-step tasks.
    4. **Color/visible-attribute swaps** (``include_color_swaps``): pick a
       visible-color variant of the source object class and swap both the
       BDDL ``(:objects ...)`` declaration and the ``(:language ...)`` so
       the realised task changes from "pick up the red mug" to "pick up
       the yellow mug".

    Args:
        bddl_content: Full text of the original BDDL file.
        include_destination_swaps: Emit family (1).
        include_predicate_negations: Emit family (2).
        include_compositional: Emit family (3).
        include_color_swaps: Emit family (4).

    Returns:
        List of ``(filename_suffix, perturbed_bddl_text)`` pairs. Empty
        list if none of the families could generate a valid variant.
    """
    import re as _re

    out: list[tuple[str, str]] = []

    goal_block = _extract_block(bddl_content, "goal")
    if not goal_block:
        return out

    obj_classes = parse_object_classes(bddl_content)
    fixtures_block = _extract_block(bddl_content, "fixtures") or ""
    fixture_instances: set[str] = set()
    fixture_classes: dict[str, str] = {}
    for line in fixtures_block.splitlines():
        line = line.strip()
        if " - " in line:
            parts = line.split(" - ")
            if len(parts) == 2:
                insts_str, cls = parts[0].strip(), parts[1].strip()
                for inst in insts_str.split():
                    fixture_instances.add(inst)
                    fixture_classes[inst] = cls

    # Combined instance → class lookup so language helpers can resolve both
    # graspable and fixture instances uniformly.
    inst_to_class: dict[str, str] = {**obj_classes, **fixture_classes}

    pred_re = _re.compile(r"\((On|In|Open|Close|Turnon|Turnoff)\s+([^\s()]+)(?:\s+([^\s()]+))?\)")
    pred_matches = list(pred_re.finditer(goal_block))

    # ── 1. Destination swaps ────────────────────────────────────────────
    if include_destination_swaps:
        for m in pred_matches:
            predicate = m.group(1)
            if predicate not in ("On", "In"):
                continue
            source_inst = m.group(2)
            dest_inst = m.group(3)
            if dest_inst is None:
                continue
            for cf_dest, cf_dest_class in obj_classes.items():
                if cf_dest in (source_inst, dest_inst):
                    continue
                if cf_dest in fixture_instances:
                    continue
                source_class = inst_to_class.get(source_inst, "")
                if cf_dest_class == source_class:
                    continue  # same-class destination → physically nonsense
                new_pred = f"({predicate} {source_inst} {cf_dest})"
                cf_text = bddl_content.replace(m.group(0), new_pred)
                source_phrase = _bare_class_phrase(source_class) if source_class else source_inst
                dest_phrase = _bare_class_phrase(cf_dest_class)
                prep = "in" if predicate == "In" else "on"
                lang = f"Put the {source_phrase} {prep} the {dest_phrase}"
                cf_text = _re.sub(
                    r"\(:language\s+[^)]+\)",
                    f"(:language {lang})",
                    cf_text,
                )
                out.append((f"_task_dest_{cf_dest_class}", cf_text))

    # ── 2. Predicate negations ──────────────────────────────────────────
    if include_predicate_negations:
        for m in pred_matches:
            predicate = m.group(1)
            target = m.group(2)
            inverse = _PREDICATE_INVERSES.get(predicate)
            if inverse is None:
                continue
            new_pred_kw, neg_phrase = inverse
            new_pred = f"({new_pred_kw} {target})"
            cf_text = bddl_content.replace(m.group(0), new_pred)
            target_phrase = _bare_class_phrase(inst_to_class.get(target, target))
            lang = f"{neg_phrase} the {target_phrase}"
            cf_text = _re.sub(
                r"\(:language\s+[^)]+\)",
                f"(:language {lang})",
                cf_text,
            )
            out.append((f"_task_neg_{new_pred_kw.lower()}", cf_text))

    # ── 3. Compositional 2-predicate tasks ───────────────────────────────
    if include_compositional and len(pred_matches) == 1:
        m = pred_matches[0]
        predicate = m.group(1)
        if predicate in ("On", "In"):
            source_inst = m.group(2)
            dest_inst = m.group(3)
            # Pick a second graspable as the new source; second new dest = dest.
            for second_inst, second_class in obj_classes.items():
                if second_inst in (source_inst, dest_inst):
                    continue
                if second_inst in fixture_instances:
                    continue
                if second_class == obj_classes.get(source_inst, ""):
                    continue
                # And-compose a second predicate. The original (And ...) is
                # parsed; we emit (And <orig> (predicate second dest)).
                inner_orig = m.group(0)
                second_pred = f"({predicate} {second_inst} {dest_inst})"
                # Locate the (And ...) wrapper; if absent, wrap.
                and_match = _re.search(r"\(And\s+(.*?)\)\s*\)\s*\Z", goal_block, _re.DOTALL)
                if and_match:
                    new_and_body = and_match.group(0)
                    new_and_body = new_and_body.replace(inner_orig, f"{inner_orig} {second_pred}")
                    cf_text = bddl_content.replace(goal_block, "")
                    cf_text = bddl_content.replace(
                        f"(:goal{goal_block}",
                        f"(:goal{goal_block.replace(inner_orig, f'{inner_orig} {second_pred}')}",
                    )
                else:
                    cf_text = bddl_content.replace(
                        f"(:goal{goal_block}",
                        f"(:goal{goal_block} {second_pred}",
                    )
                source_phrase = _bare_class_phrase(inst_to_class.get(source_inst, source_inst))
                second_phrase = _bare_class_phrase(second_class)
                dest_phrase = _bare_class_phrase(inst_to_class.get(dest_inst, dest_inst))
                prep = "in" if predicate == "In" else "on"
                lang = (
                    f"Put the {source_phrase} {prep} the {dest_phrase} "
                    f"and also put the {second_phrase} {prep} the {dest_phrase}"
                )
                cf_text = _re.sub(
                    r"\(:language\s+[^)]+\)",
                    f"(:language {lang})",
                    cf_text,
                )
                out.append((f"_task_compose_{second_class}", cf_text))
                break  # one compositional variant is enough

    # ── 4. Visible-attribute (color) swaps ──────────────────────────────
    if include_color_swaps:
        for inst, cls in obj_classes.items():
            if inst in fixture_instances:
                continue
            variants = _VISIBLE_COLOR_VARIANTS.get(cls, [])
            for variant in variants:
                if variant == cls:
                    continue
                # Only the source object is renamed; goal predicate +
                # objects of interest still reference the same instance,
                # but the (:objects ...) declaration uses the new class.
                cf_text = substitute_multi(bddl_content, {cls: variant})
                # Update the language to reflect the new color/visible
                # attribute (e.g. "red mug" → "yellow mug").
                old_phrase = _bare_class_phrase(cls)
                new_phrase = _bare_class_phrase(variant)
                cf_text = _re.sub(
                    r"\(:language\s+([^)]+)\)",
                    lambda mm: "(:language " + mm.group(1).replace(old_phrase, new_phrase) + ")",
                    cf_text,
                )
                out.append((f"_task_color_{variant}", cf_text))

    return out


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
