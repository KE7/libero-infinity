"""Aggregator for filled-in G7 review checklists.

Walks a packet-tree (typically produced by :mod:`review_sampler`), reads each
``checklist.yaml``, and produces:

- per-task severity distribution
- inter-rater agreement statistics (Cohen's κ; Krippendorff's α when 3+
  reviewers)
- list of critical-severity items requiring fix-and-rerun
- a markdown summary suitable as evidence for the publication acceptance
  criterion

CLI
---
::

    python -m libero_infinity.validation.review_aggregate \\
        --in <packet_root> --out <report.md>
"""

from __future__ import annotations

import argparse
import collections
import logging
import pathlib
import sys
from typing import Any

import yaml

logger = logging.getLogger(__name__)


CRITICAL_SEVERITY = 4
SIGNIFICANT_SEVERITY = 3

# Severity field names within each checklist item.
SEVERITY_FIELDS: tuple[str, ...] = (
    "severity",
    "reviewer2_severity",
    "adjudicator_severity",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def iter_checklists(root: pathlib.Path) -> list[tuple[pathlib.Path, dict]]:
    """Return ``(path, parsed_yaml)`` for every ``checklist.yaml`` under ``root``."""
    out: list[tuple[pathlib.Path, dict]] = []
    for path in sorted(root.rglob("checklist.yaml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            logger.warning("could not parse %s", path)
            continue
        if isinstance(doc, dict):
            out.append((path, doc))
    return out


def _per_item_severities(item: dict) -> list[int]:
    """Return the list of non-null integer severities recorded for an item."""
    sevs: list[int] = []
    for field in SEVERITY_FIELDS:
        v = item.get(field)
        if isinstance(v, int):
            sevs.append(v)
    return sevs


def _overall_severity(items: list[dict]) -> int | None:
    """Max severity across questions (any reviewer)."""
    seen: list[int] = []
    for item in items:
        seen.extend(_per_item_severities(item))
    return max(seen) if seen else None


# ---------------------------------------------------------------------------
# Cohen's κ (two-rater, ordinal-linear weights) and Krippendorff α (ordinal)
# ---------------------------------------------------------------------------


def cohens_kappa(pairs: list[tuple[int, int]], weights: str = "linear") -> float | None:
    """Cohen's κ on integer scores. Weighted (linear default) for ordinal data.

    Returns ``None`` if fewer than two pairs or no variance.
    """
    if len(pairs) < 2:
        return None
    cats = sorted({c for pair in pairs for c in pair})
    if len(cats) < 2:
        # Perfect agreement on a single category — undefined; return 1.0 by convention.
        return 1.0
    n = len(pairs)
    idx = {c: i for i, c in enumerate(cats)}
    K = len(cats)
    # Confusion matrix
    M = [[0] * K for _ in range(K)]
    for a, b in pairs:
        M[idx[a]][idx[b]] += 1
    row_tot = [sum(row) / n for row in M]
    col_tot = [sum(M[r][c] for r in range(K)) / n for c in range(K)]

    def w(i: int, j: int) -> float:
        if weights == "linear":
            return 1.0 - abs(i - j) / (K - 1)
        if weights == "quadratic":
            return 1.0 - ((i - j) ** 2) / ((K - 1) ** 2)
        return 1.0 if i == j else 0.0

    po = sum(w(i, j) * M[i][j] / n for i in range(K) for j in range(K))
    pe = sum(w(i, j) * row_tot[i] * col_tot[j] for i in range(K) for j in range(K))
    if pe == 1.0:
        return None
    return (po - pe) / (1.0 - pe)


def krippendorff_alpha(units: list[list[int]], levels: int = 5) -> float | None:
    """Krippendorff's α with ordinal distance, for a list-of-lists of ratings.

    Each ``units[i]`` is the list of ratings *units of measurement* received
    on item ``i`` (length ≥ 1).  Missing data is implicit (units of varying
    length).  ``levels`` is the number of ordinal categories (0..levels-1).
    """
    # Compute observed and expected disagreement
    valid_units = [u for u in units if len(u) >= 2]
    if not valid_units:
        return None

    def ordinal_dist(a: int, b: int) -> float:
        return float((a - b) ** 2)

    # Observed disagreement
    num_obs = 0.0
    n_pairs_obs = 0.0
    for u in valid_units:
        m = len(u)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                num_obs += ordinal_dist(u[i], u[j])
                n_pairs_obs += 1
    do = num_obs / n_pairs_obs if n_pairs_obs else 0.0

    # Expected disagreement (chance baseline) — use the marginal frequencies
    all_vals: list[int] = [v for u in valid_units for v in u]
    if len(all_vals) < 2:
        return None
    counts = collections.Counter(all_vals)
    keys = sorted(counts)
    total = len(all_vals)
    de_num = 0.0
    de_den = total * (total - 1)
    for a in keys:
        for b in keys:
            if a == b:
                continue
            de_num += counts[a] * counts[b] * ordinal_dist(a, b)
    de = de_num / de_den if de_den else 0.0
    if de == 0.0:
        return None
    return 1.0 - do / de


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def aggregate(root: pathlib.Path) -> dict[str, Any]:
    """Walk ``root``, return a structured aggregation dict."""
    checklists = iter_checklists(root)

    per_task: dict[str, list[int]] = collections.defaultdict(list)
    per_q_pairs: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    per_q_units: dict[str, list[list[int]]] = collections.defaultdict(list)
    critical: list[dict[str, Any]] = []
    significant: list[dict[str, Any]] = []
    total_packets = 0
    total_with_two_reviewers = 0

    for path, doc in checklists:
        total_packets += 1
        pkt = doc.get("packet", {}) or {}
        task = str(pkt.get("task", "?"))
        items = doc.get("items") or []
        if not isinstance(items, list):
            continue
        overall = _overall_severity(items)
        if overall is not None:
            per_task[task].append(overall)
        has_two = False
        for item in items:
            qid = item.get("id", "?")
            sevs = _per_item_severities(item)
            if len(sevs) >= 2:
                per_q_pairs[qid].append((sevs[0], sevs[1]))
                has_two = True
            if sevs:
                per_q_units[qid].append(sevs)
            max_sev = max(sevs) if sevs else None
            if max_sev == CRITICAL_SEVERITY:
                critical.append(
                    {
                        "packet": str(path.parent),
                        "task": task,
                        "question": qid,
                        "severity": max_sev,
                        "notes": item.get("notes", ""),
                    }
                )
            elif max_sev == SIGNIFICANT_SEVERITY:
                significant.append(
                    {
                        "packet": str(path.parent),
                        "task": task,
                        "question": qid,
                        "severity": max_sev,
                        "notes": item.get("notes", ""),
                    }
                )
        if has_two:
            total_with_two_reviewers += 1

    per_q_kappa: dict[str, float | None] = {
        q: cohens_kappa(pairs) for q, pairs in per_q_pairs.items()
    }
    per_q_alpha: dict[str, float | None] = {
        q: krippendorff_alpha(units) for q, units in per_q_units.items()
    }

    kvals = [v for v in per_q_kappa.values() if v is not None]
    avals = [v for v in per_q_alpha.values() if v is not None]

    return {
        "root": str(root),
        "total_packets": total_packets,
        "total_with_two_reviewers": total_with_two_reviewers,
        "per_task_severities": {t: collections.Counter(sevs) for t, sevs in per_task.items()},
        "per_question_kappa": per_q_kappa,
        "per_question_alpha": per_q_alpha,
        "mean_kappa": sum(kvals) / len(kvals) if kvals else None,
        "min_kappa": min(kvals) if kvals else None,
        "mean_alpha": sum(avals) / len(avals) if avals else None,
        "critical_items": critical,
        "significant_items": significant,
    }


def render_markdown(stats: dict[str, Any]) -> str:
    out: list[str] = ["# G7 Human Review — Aggregated Report", ""]
    out.append(f"**Packet root:** `{stats['root']}`")
    out.append(f"**Total packets graded:** {stats['total_packets']}")
    out.append(f"**Packets with ≥2 reviewers:** {stats['total_with_two_reviewers']}")
    out.append("")

    # Inter-rater agreement
    out.append("## Inter-rater agreement")
    out.append("")
    out.append("| Statistic | Value |")
    out.append("|---|---|")
    out.append(f"| Mean Cohen's κ (across questions) | {_fmt(stats['mean_kappa'])} |")
    out.append(f"| Min Cohen's κ (across questions)  | {_fmt(stats['min_kappa'])} |")
    out.append(f"| Mean Krippendorff α (across questions) | {_fmt(stats['mean_alpha'])} |")
    out.append("")
    out.append("### Per-question κ / α")
    out.append("")
    out.append("| Question | κ | α |")
    out.append("|---|---|---|")
    all_q = sorted(set(stats["per_question_kappa"]) | set(stats["per_question_alpha"]))
    for q in all_q:
        k = stats["per_question_kappa"].get(q)
        a = stats["per_question_alpha"].get(q)
        out.append(f"| `{q}` | {_fmt(k)} | {_fmt(a)} |")
    out.append("")

    # Severity distribution per task
    out.append("## Per-task severity distribution (max-over-questions)")
    out.append("")
    out.append("| Task | n_packets | 0 | 1 | 2 | 3 | 4 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for task, counter in sorted(stats["per_task_severities"].items()):
        n = sum(counter.values())
        cells = " | ".join(str(counter.get(i, 0)) for i in range(5))
        out.append(f"| `{task}` | {n} | {cells} |")
    out.append("")

    # Critical items
    out.append("## Critical (severity 4) items — fix-and-rerun required")
    out.append("")
    if stats["critical_items"]:
        out.append("| Packet | Task | Question | Notes |")
        out.append("|---|---|---|---|")
        for row in stats["critical_items"]:
            notes = (row["notes"] or "").replace("\n", " ").replace("|", "\\|")
            out.append(f"| `{row['packet']}` | `{row['task']}` | `{row['question']}` | {notes} |")
    else:
        out.append("_(no critical findings)_")
    out.append("")

    # Significant items
    out.append("## Significant (severity 3) items")
    out.append("")
    if stats["significant_items"]:
        out.append("| Packet | Task | Question | Notes |")
        out.append("|---|---|---|---|")
        for row in stats["significant_items"]:
            notes = (row["notes"] or "").replace("\n", " ").replace("|", "\\|")
            out.append(f"| `{row['packet']}` | `{row['task']}` | `{row['question']}` | {notes} |")
    else:
        out.append("_(no significant findings)_")
    out.append("")

    return "\n".join(out) + "\n"


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m libero_infinity.validation.review_aggregate",
        description="Aggregate filled G7 review checklists into a markdown report.",
    )
    p.add_argument("--in", dest="in_dir", required=True, help="Packet root directory.")
    p.add_argument("--out", required=True, help="Output markdown report path.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    in_dir = pathlib.Path(args.in_dir)
    out_path = pathlib.Path(args.out)
    if not in_dir.is_dir():
        print(f"error: {in_dir} is not a directory", file=sys.stderr)
        return 2
    stats = aggregate(in_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(stats), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
