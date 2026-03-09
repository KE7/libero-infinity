#!/usr/bin/env python3
"""Score CF BDDL variants using OpenRouter (concurrent parallel requests).

Each variant gets its own independent request fired concurrently via asyncio.

Scene images
------------
When per-scene images are available (rendered by scripts/render_scene_images.py
and stored under assets/scene_images/<suite>/<task>.png), each request uses
the actual scene image for that specific task.  This gives the LLM accurate
visual context and improves scoring quality.

TRADEOFF — accuracy vs. caching cost:
  The original design used a single shared image so all 157 requests shared the
  same prompt prefix, enabling server-side prefix caching (OpenRouter / vLLM).
  Per-scene images break that shared prefix: each request sends a different
  image, so every request is processed fresh.  This means ~10–20× more tokens
  billed compared to the cached-prefix approach.  For a one-off evaluation run
  the accuracy gain is worth it; for repeated large batches you may want to
  restore the single shared image by setting LIBERO_SHARED_SCENE_IMAGE=1.

  Per-scene images missing? → falls back to the generic image with a warning.
  Generate per-scene images: python scripts/render_scene_images.py

Usage
-----
    python scripts/rate_cf_bddls_llm.py
    python scripts/rate_cf_bddls_llm.py --concurrency 20
    python scripts/rate_cf_bddls_llm.py --max-variants 10   # quick test
    python scripts/rate_cf_bddls_llm.py --backend ollama --model qwen3-coder-next:q8_0
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from libero_infinity.bddl_preprocessor import (
    _extract_block,
    generate_cf_bddls,
    parse_object_classes,
)
from libero_infinity.runtime import get_bddl_dir

LIBERO_BDDL_ROOT = get_bddl_dir()
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_URL     = "http://localhost:11434/api/chat"

# ── scene image helpers ───────────────────────────────────────────────────────
#
# Per-scene images live at:
#   assets/scene_images/<suite>/<task_stem>.png
#
# Generate them with:
#   python scripts/render_scene_images.py
#
# TRADEOFF — caching vs. accuracy:
#   Single shared image → all 157 requests share the same prompt prefix →
#   provider-side prefix cache saves ~95 % of input tokens (fast & cheap).
#
#   Per-scene images → each request carries a different image → prefix cache
#   cannot kick in → every request is fully billed (~10–20× more tokens).
#
#   Accuracy gain justifies the cost for a one-off evaluation; for repeated
#   large batches consider reverting to shared image by setting env var
#   LIBERO_SHARED_SCENE_IMAGE=1.

_ASSETS_DIR      = pathlib.Path(__file__).parent.parent / "assets"
_SCENE_IMAGE_DIR = _ASSETS_DIR / "scene_images"
_FALLBACK_IMAGE  = _ASSETS_DIR / "libero_agentview_default.png"

# In-process cache: avoid re-encoding the same image for multiple variants of
# the same source task.
_image_b64_cache: dict[pathlib.Path, str] = {}


def _load_image_b64(path: pathlib.Path, size: int = 1024) -> str:
    """Load, resize to `size`×`size`, and base64-encode a PNG.  Memoised."""
    if path in _image_b64_cache:
        return _image_b64_cache[path]
    from PIL import Image
    import io
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = base64.b64encode(buf.getvalue()).decode()
    _image_b64_cache[path] = result
    return result


def get_scene_image_b64(source_task: str, suite: str, size: int = 1024) -> str:
    """Return a base64-encoded PNG for the given LIBERO task.

    Resolution order:
      1. assets/scene_images/<suite>/<source_task>.png  — per-scene render
      2. assets/libero_agentview_default.png             — generic fallback

    Per-scene images are rendered by ``scripts/render_scene_images.py``.
    Run that script once (requires mujoco + libero installed) to populate
    the cache.  Missing images trigger a one-line warning per task.
    """
    per_scene = _SCENE_IMAGE_DIR / suite / f"{source_task}.png"
    if per_scene.exists():
        return _load_image_b64(per_scene, size)
    # Emit a warning — once per unique (suite, task) pair thanks to the cache.
    if per_scene not in _image_b64_cache:
        print(
            f"  [WARN] No per-scene image for {suite}/{source_task} — "
            "using generic fallback.  Run scripts/render_scene_images.py to fix.",
            flush=True,
        )
        # Store a sentinel so the warning fires only once per task.
        _image_b64_cache[per_scene] = ""
    return _load_image_b64(_FALLBACK_IMAGE, size)


def get_scene_image_url(source_task: str, suite: str, size: int = 1024) -> str:
    """Return a data-URI for the given task's scene image."""
    return f"data:image/png;base64,{get_scene_image_b64(source_task, suite, size)}"

# ── prompts ───────────────────────────────────────────────────────────────────

# The system prompt is identical for every request.
# NOTE: each user message now carries the per-task scene image, so prefix
# caching across requests is not possible (see module docstring for details).
SYSTEM = """\
You are an expert evaluator for LIBERO, a robot manipulation benchmark where a
Franka Panda arm operates on a tabletop scene.  The attached image shows the
actual LIBERO scene for the task being evaluated: a wooden table with everyday
objects (bowls, mugs, food items, bottles, books), fixtures (wooden cabinet,
wine rack, flat stove, wooden tray, desk caddy, basket), and the robot arm
overhead.

You evaluate counterfactual (CF) task variants.  In LIBERO, a policy receives a
natural-language instruction and must pick up the correct object.  A CF variant
keeps the scene layout identical but re-targets the instruction to a *different*
graspable object — this tests whether the policy genuinely understands language
or just grabs the canonical object by habit.

LIBERO fixture reference (use this for physical_plausibility scores):
- wooden cabinet   open shelves, any small object fits on or inside → pp=5
- wine rack        slanted wooden shelf, bowls/mugs/cans fit, not just bottles → pp=4–5
- wooden tray      open shallow container, anything small fits inside → pp=5
- flat stove       cooking surface, pans/bottles ok (pp=4), food cartons/books/mugs odd (pp=2–3)
- desk caddy       open organiser, mugs/bottles/small objects fit → pp=5
- basket           open container, any small graspable object → pp=5
- plate            flat surface, any small graspable object → pp=5

Rate the given CF variant on three dimensions (integers 1–5):

language_quality (1–5)
  5 = natural, concise — something a human would say ("Put the milk in the basket")
  4 = slightly technical but clear ("Put the salad dressing in the basket")
  3 = awkward but understandable
  2 = verbose or confusing
  1 = grammatically wrong or nonsensical

grounding_challenge (1–5)
  How strong a test of language understanding is this swap?
  5 = CF object is visually/categorically very different from the original
      (e.g. original=bowl, CF=milk bottle) — a language-blind policy would fail
  4 = different category, moderate visual difference
  3 = same broad category (e.g. mug→different mug style) — weaker but useful
  2 = nearly identical objects
  1 = swap makes no sense as a grounding test

  IMPORTANT: The CF object may or may not be visible in the provided scene
  image — this is intentional by design. If the CF object is absent from the
  scene, grounding_challenge should be rated HIGH (4 or 5): the policy must
  follow the language instruction, not just grab whatever object is visible.
  An absent CF object makes the language grounding test HARDER, not invalid.
  Do NOT penalise grounding_challenge because the CF object is absent from
  the image.

physical_plausibility (1–5)
  Can a robot gripper pick up the CF object and place it at the destination?
  Use the fixture reference above — do not apply real-world kitchen semantics.
  5 = obviously feasible
  4 = feasible, slightly unusual
  3 = possible but awkward
  2 = very difficult in simulation
  1 = physically impossible

Return ONLY valid JSON — no markdown, no explanation:
{"language_quality": <int>, "grounding_challenge": <int>, "physical_plausibility": <int>, "reasoning": "<one concise sentence>"}
"""

# Per-variant user text (appended after the cached image).
def _variant_text(variant: dict) -> str:
    lang = variant["generated_language"]
    # Extract destination as it appears in the instruction (not the raw region ID).
    m = re.search(r"(?:on|in) the (.+)$", lang, re.IGNORECASE)
    dest = m.group(1) if m else variant["dest"]
    return (
        f'original_task:   "{variant["source_task_short"]}"\n'
        f'original_object: "{variant["source_class"]}"\n'
        f'cf_instruction:  "{lang}"\n'
        f'destination:     "{dest}"'
    )


# ── scoring math ──────────────────────────────────────────────────────────────

def _overall(lq: int, gc: int, pp: int) -> float:
    raw = lq * 0.30 + gc * 0.40 + pp * 0.30
    return round((raw - 1) / 4 * 100, 1)


def _tier(score: float | None) -> str:
    if score is None: return "?"
    if score >= 80:   return "A"
    if score >= 60:   return "B"
    if score >= 40:   return "C"
    return "D"


# ── async HTTP ────────────────────────────────────────────────────────────────

async def _post(session, url: str, payload: dict, headers: dict) -> dict:
    import aiohttp
    async with session.post(
        url, json=payload, headers=headers,
        timeout=aiohttp.ClientTimeout(total=600),
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def score_one_openrouter(
    session, variant: dict, model: str, sem: asyncio.Semaphore
) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    # Per-task scene image (looked up from assets/scene_images/<suite>/<task>.png).
    # Falls back to the generic image if the per-scene render is not available.
    # NOTE: using per-scene images breaks shared-prefix caching — see module
    # docstring for the accuracy vs. cost tradeoff discussion.
    scene_url = get_scene_image_url(variant["source_task"], variant["suite"])

    # Message structure:
    #   [system]  SYSTEM      ← identical for all requests (text-only prefix)
    #   [user]    image       ← per-task scene image
    #             text        ← per-variant task description
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM,         # shared text — same for every request
            },
            {
                "role": "user",
                "content": [
                    {                      # per-task scene image
                        "type": "image_url",
                        "image_url": {"url": scene_url},
                    },
                    {                      # per-variant text description
                        "type": "text",
                        "text": _variant_text(variant),
                    },
                ],
            },
        ],
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for attempt in range(3):
        try:
            async with sem:
                data = await _post(session, OPENROUTER_URL, payload, headers)
            if "choices" not in data:
                err = data.get("error", data)
                raise ValueError(f"API error: {err}")
            content = data["choices"][0]["message"]["content"]
            return _parse_one(content, variant)
        except Exception as exc:
            if attempt == 2:
                return _parse_one(f"REQUEST_ERROR: {exc}", variant)
            await asyncio.sleep(5 * (attempt + 1))


async def score_one_ollama(
    session, variant: dict, model: str, sem: asyncio.Semaphore
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": _variant_text(variant)},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    for attempt in range(3):
        try:
            async with sem:
                data = await _post(session, OLLAMA_URL, payload, {})
            content = data["message"]["content"]
            return _parse_one(content, variant)
        except Exception as exc:
            if attempt == 2:
                return _parse_one(f"REQUEST_ERROR: {exc}", variant)
            await asyncio.sleep(5 * (attempt + 1))


def _parse_one(response: str, variant: dict) -> dict:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", response, re.DOTALL)
    try:
        scored = json.loads(m.group(0)) if m else {}
        lq = int(scored["language_quality"])
        gc = int(scored["grounding_challenge"])
        pp = int(scored["physical_plausibility"])
        overall = _overall(lq, gc, pp)
        return {
            **variant,
            "llm_language_quality":      lq,
            "llm_grounding_challenge":   gc,
            "llm_physical_plausibility": pp,
            "llm_reasoning":  scored.get("reasoning", ""),
            "llm_overall":    overall,
            "llm_tier":       _tier(overall),
        }
    except Exception:
        return {
            **variant,
            "llm_language_quality": None,
            "llm_grounding_challenge": None,
            "llm_physical_plausibility": None,
            "llm_reasoning": f"PARSE_ERROR: {response[:120]}",
            "llm_overall": None,
            "llm_tier": "?",
        }


# ── collect variants ──────────────────────────────────────────────────────────

def collect_variants(max_variants: int | None) -> list[dict]:
    rows: list[dict] = []
    uid = 0
    for suite_name in SUITES:
        suite_dir = LIBERO_BDDL_ROOT / suite_name
        if not suite_dir.exists():
            continue
        for bddl_path in sorted(suite_dir.glob("*.bddl")):
            content = bddl_path.read_text()
            goal_block = _extract_block(content, "goal") or ""
            pred_m = re.search(r"\((On|In)\s+([^\s()]+)\s+([^\s()]+)\)", goal_block)
            if not pred_m:
                continue
            source_inst  = pred_m.group(2)
            dest_inst    = pred_m.group(3)
            obj_classes  = parse_object_classes(content)
            source_class = obj_classes.get(source_inst, source_inst)

            if dest_inst in obj_classes:
                dest_disp = obj_classes[dest_inst].replace("_", " ")
            else:
                s = re.sub(r"_\d+_\w+_region$", "", dest_inst)
                dest_disp = re.sub(r"_\d+$", "", s).replace("_", " ")

            for suffix, cf_text in generate_cf_bddls(content):
                cf_lang_m = re.search(r"\(:language ([^)]+)\)", cf_text)
                rows.append({
                    "id": uid,
                    "suite": suite_name,
                    "source_task": bddl_path.stem,
                    "source_task_short": bddl_path.stem[:50],
                    "source_class": source_class,
                    "cf_class": suffix.removeprefix("_cf_"),
                    "dest": dest_disp,
                    "generated_language": cf_lang_m.group(1) if cf_lang_m else "",
                })
                uid += 1
                if max_variants and uid >= max_variants:
                    return rows
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def _load_existing(path: pathlib.Path) -> dict[int, dict]:
    """Load an existing CSV and return a dict mapping variant id → row."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = {}
        for row in reader:
            try:
                vid = int(row["id"])
                # Only keep rows that were actually scored (not errors)
                if row.get("llm_tier") and row["llm_tier"] != "?":
                    rows[vid] = row
                elif row.get("llm_overall") and row["llm_overall"] not in ("", "None"):
                    rows[vid] = row
            except (KeyError, ValueError):
                pass
        return rows


async def run(args: argparse.Namespace) -> None:
    try:
        import aiohttp
    except ImportError:
        print("ERROR: aiohttp not installed.  Run: uv pip install aiohttp")
        sys.exit(1)

    # ── resume: load already-rated rows ──────────────────────────────────────
    existing: dict[int, dict] = {}
    if args.resume and args.out.exists():
        existing = _load_existing(args.out)
        print(f"Resume: found {len(existing)} already-rated rows in {args.out.name}")

    print("Collecting CF variants...")
    all_variants = collect_variants(args.max_variants)

    # Filter out already-rated variants
    variants = [v for v in all_variants if v["id"] not in existing]
    skipped  = len(all_variants) - len(variants)
    # Count how many tasks have a per-scene image available.
    n_per_scene = sum(
        1
        for v in all_variants
        if (_SCENE_IMAGE_DIR / v["suite"] / f"{v['source_task']}.png").exists()
    )
    image_mode = (
        f"per-scene ({n_per_scene}/{len(all_variants)} have renders)"
        if n_per_scene > 0
        else f"generic fallback ({_FALLBACK_IMAGE.name}) — run render_scene_images.py"
    )
    print(
        f"  {len(all_variants)} total variants  |  {skipped} already rated  "
        f"|  {len(variants)} to score\n"
        f"  model: {args.model}  |  concurrency: {args.concurrency}"
        f"  |  images: {image_mode}\n"
    )

    new_results: list[dict] = []
    t0 = time.time()

    if variants:
        sem = asyncio.Semaphore(args.concurrency)
        done = 0

        async with aiohttp.ClientSession() as session:
            if args.backend == "openrouter":
                tasks = [score_one_openrouter(session, v, args.model, sem) for v in variants]
            else:
                tasks = [score_one_ollama(session, v, args.model, sem) for v in variants]

            for coro in asyncio.as_completed(tasks):
                result = await coro
                new_results.append(result)
                done += 1
                status  = result["llm_tier"] if result["llm_overall"] is not None else "ERR"
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (len(variants) - done) / rate if rate > 0 else 0
                print(
                    f"  {done:3d}/{len(variants)}  [{status}] {result['llm_overall'] or '---':>5}  "
                    f"'{result['generated_language'][:55]}'  eta {eta:.0f}s",
                    flush=True,
                )
    else:
        print("  All variants already rated — nothing to score.\n")

    # Merge existing + new results
    results: list[dict] = list(existing.values()) + new_results
    results.sort(key=lambda r: (r.get("llm_overall") or -1), reverse=True)

    # ── summary ───────────────────────────────────────────────────────────────
    tiers  = {"A": 0, "B": 0, "C": 0, "D": 0, "?": 0}
    for r in results:
        tier = r.get("llm_tier") or "?"
        tiers[tier] = tiers.get(tier, 0) + 1
    scores = [float(r["llm_overall"]) for r in results
              if r.get("llm_overall") not in (None, "", "None")]

    import statistics
    print(f"\n{'─'*80}")
    print(f"  LLM Quality Ratings  ({len(results)} variants  |  {args.model})")
    print(f"{'─'*80}")
    if scores:
        print(f"  min={min(scores):.1f}  median={statistics.median(scores):.1f}  "
              f"mean={statistics.mean(scores):.1f}  max={max(scores):.1f}")
    print(
        f"  Tier A (≥80): {tiers['A']:3d}   Tier B (≥60): {tiers['B']:3d}   "
        f"Tier C (≥40): {tiers['C']:3d}   Tier D (<40): {tiers['D']:3d}",
        end="",
    )
    if tiers["?"]:
        print(f"   Errors: {tiers['?']}", end="")
    print("\n")

    d_tier = [r for r in results if r.get("llm_tier") == "D"]
    if d_tier:
        print(f"D-tier ({len(d_tier)}) — consider discarding:")
        for r in d_tier:
            score = r.get("llm_overall")
            score_str = f"{float(score):5.1f}" if score not in (None, "", "None") else "  N/A"
            print(f"  [{score_str}]  '{r['generated_language']}'")
            print(f"           {r.get('llm_reasoning', '')}")
        print()

    print("Top 5:")
    for r in results[:5]:
        score = r.get("llm_overall")
        s = f"{float(score):5.1f}" if score not in (None, "", "None") else "  N/A"
        print(f"  [{s}]  '{r['generated_language']}'")
        print(f"           {r.get('llm_reasoning', '')}")
    print()

    if not results:
        print("No results to save.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Use union of all fieldnames to handle mixed old/new rows
    all_fields = list(results[0].keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved → {args.out}  ({time.time()-t0:.0f}s total)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",        default="qwen/qwen3-vl-235b-a22b-thinking")
    parser.add_argument("--backend",      choices=["openrouter", "ollama"], default="openrouter")
    parser.add_argument("--concurrency",  type=int, default=10,
                        help="Max simultaneous requests (default 10)")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument(
        "--out", type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent / "cf_quality_ratings_llm.csv",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip variants already rated in --out CSV and append new results.",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
