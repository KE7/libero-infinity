"""LLM-driven language paraphrasing for BDDL files.

Generates *task-equivalent* but textually distinct ``(:language ...)``
strings via a single litellm call — one BDDL in, N paraphrased BDDLs
out, no manual authoring required.

Example
-------

>>> from libero_infinity.language_paraphrase import generate_paraphrased_bddls
>>> import pathlib
>>> bddl = pathlib.Path("path/to/task.bddl").read_text()
>>> variants = generate_paraphrased_bddls(
...     bddl, n_variants=3, model="anthropic/claude-3-5-sonnet-20241022"
... )
>>> for suffix, paraphrased_bddl in variants:
...     print(suffix, "→", paraphrased_bddl[:120])

Why offline (not Scenic)
------------------------
The ``(:language ...)`` block is parsed by LIBERO's BDDL loader at env
construction time, not from Scenic params. Paraphrasing therefore has to
edit the BDDL file fed to ``OffScreenRenderEnv``. We pre-generate
variants offline and let the gym wrapper select one per reset.
"""

from __future__ import annotations

import json
import re

from libero_infinity.bddl_preprocessor import _parse_language

_DEFAULT_MODEL = "anthropic/claude-3-5-sonnet-20241022"

_PARAPHRASE_PROMPT = """\
You will rewrite a robot manipulation task instruction into {n} paraphrased
forms. Each paraphrase must:

  - Refer to exactly the same object(s) and target(s) as the original.
  - Preserve the action semantics (a "put X on Y" instruction must still
    describe placing X on Y, not picking it up or moving it elsewhere).
  - Use natural, varied English — different verbs, sentence structures,
    or word orders are encouraged.
  - Avoid adding objects, qualifiers, or steps the original does not
    mention.
  - Stay under 20 words.

Return a JSON array of {n} strings, no commentary, e.g.

  ["paraphrase 1", "paraphrase 2", "paraphrase 3"]

Original instruction: {instruction}
"""


def generate_paraphrased_bddls(
    bddl_content: str,
    n_variants: int = 3,
    *,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.7,
) -> list[tuple[str, str]]:
    """Return ``[(suffix, paraphrased_bddl)] × n_variants``.

    The function is a no-op (returns ``[]``) if ``litellm`` is not
    installed or the LLM call fails — paraphrasing is optional and a
    failure here should not break downstream BDDL pipelines that may
    work without it.

    Args:
        bddl_content: Full text of the source BDDL.
        n_variants: Number of paraphrases to request.
        model: ``litellm`` model identifier (any provider is fine —
            ``anthropic/…``, ``openai/…``, ``ollama/…``, etc.).
        temperature: Sampling temperature for the LLM call. Higher =
            more variety, lower = more conservative.

    Returns:
        List of ``(suffix, bddl_text)`` pairs. ``suffix`` is the form
        ``"_paraphrase_<i>"`` for stable filename construction.
    """
    if n_variants <= 0:
        return []

    instruction = _parse_language(bddl_content)
    if not instruction:
        return []

    try:
        import litellm
    except ImportError:
        return []

    prompt = _PARAPHRASE_PROMPT.format(n=n_variants, instruction=instruction)
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        raw = response["choices"][0]["message"]["content"]
    except Exception:
        return []

    paraphrases = _extract_json_array(raw)
    if not paraphrases:
        return []

    out: list[tuple[str, str]] = []
    for i, paraphrase in enumerate(paraphrases[:n_variants]):
        if not isinstance(paraphrase, str):
            continue
        cleaned = paraphrase.strip()
        if not cleaned:
            continue
        rewritten = re.sub(
            r"\(:language\s+[^)]+\)",
            f"(:language {cleaned})",
            bddl_content,
            count=1,
        )
        out.append((f"_paraphrase_{i}", rewritten))
    return out


def _extract_json_array(text: str) -> list:
    """Best-effort JSON-array extraction from a free-form LLM response."""
    text = text.strip()
    # Strip surrounding code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match is None:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return parsed if isinstance(parsed, list) else []
