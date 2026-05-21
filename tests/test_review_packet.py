"""Unit tests for the G7 review-tooling pipeline.

These cover the BDDL → packet generation path, the sampler's subset/canonical
enumeration, the layer-3 flagged-condition extractor, and the aggregator's
kappa / alpha / report rendering.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from libero_infinity.validation import review_aggregate, review_packet, review_sampler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BDDL_ROOT = REPO_ROOT / "src/libero_infinity/data/libero_runtime/bddl_files"


def _pick_bddl() -> pathlib.Path:
    """Pick a small, well-known BDDL file for tests."""
    candidate = (
        BDDL_ROOT / "libero_10" / "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl"
    )
    if candidate.exists():
        return candidate
    # Fall back to whatever's first under the root.
    for p in sorted(BDDL_ROOT.rglob("*.bddl")):
        return p
    pytest.skip("no BDDL files installed; cannot exercise packet generator")


# ---------------------------------------------------------------------------
# review_packet
# ---------------------------------------------------------------------------


def test_subset_token_round_trip():
    assert review_packet.subset_token("") == "none"
    assert review_packet.subset_token("none") == "none"
    assert review_packet.subset_token("position") == "position"
    assert review_packet.subset_token("position,camera") == "camera+position"
    assert review_packet.subset_token("full") == "full"
    assert review_packet.subset_token("combined") == "combined"


def test_checklist_questions_are_eleven():
    assert len(review_packet.CHECKLIST_QUESTIONS) == 11
    # Each is a (id, question_text) pair.
    for qid, qtext in review_packet.CHECKLIST_QUESTIONS:
        assert isinstance(qid, str) and qid.startswith("q")
        assert isinstance(qtext, str) and qtext.endswith("?")


def test_generate_packet_writes_all_files(tmp_path):
    bddl = _pick_bddl()
    result = review_packet.generate_packet(
        bddl,
        "position",
        seed=0,
        out_dir=tmp_path,
        render=False,
    )
    for rel in review_packet.PACKET_FILES:
        assert (tmp_path / rel).exists(), f"missing {rel}"
    # checklist.yaml is a valid YAML doc with 11 items and the right ids.
    doc = yaml.safe_load((tmp_path / "checklist.yaml").read_text())
    assert isinstance(doc, dict)
    assert len(doc["items"]) == 11
    assert {it["id"] for it in doc["items"]} == {
        qid for qid, _ in review_packet.CHECKLIST_QUESTIONS
    }
    assert doc["packet"]["axis_subset"] == "position"
    # axis_params.json is well-formed json with the seed embedded.
    params = json.loads((tmp_path / "axis_params.json").read_text())
    assert params["_seed"] == 0
    assert params["_axes_requested"] == ["position"]
    # No fatal errors expected on a clean task.
    assert "scenic/generated.scenic" not in result.errors


def test_generate_packet_none_subset(tmp_path):
    bddl = _pick_bddl()
    review_packet.generate_packet(bddl, "none", seed=7, out_dir=tmp_path, render=False)
    diff = (tmp_path / "scenic/baseline_diff.md").read_text()
    assert "axes=none" in diff or "no diff" in diff


# ---------------------------------------------------------------------------
# review_sampler
# ---------------------------------------------------------------------------


def test_all_subsets_count():
    subsets = list(review_sampler.all_subsets())
    # 2^9 = 512 subsets
    assert len(subsets) == 512
    # All distinct
    assert len(set(subsets)) == 512


def test_edge_case_subsets_count():
    edge = review_sampler.edge_case_subsets()
    # none + 9 singletons + combined + full = 12
    assert len(edge) == 12
    assert frozenset() in edge


def test_default_bddl_root_exists():
    root = review_sampler.default_bddl_root()
    # Either the runtime is present, or we skip downstream tests.
    if root.exists():
        bddls = review_sampler.list_all_bddl(root)
        assert len(bddls) >= 1


def test_resolve_canonical_finds_something():
    root = review_sampler.default_bddl_root()
    if not root.exists():
        pytest.skip("no BDDL root")
    resolved = review_sampler.resolve_canonical(root)
    # Most canonical picks should resolve; require at least 8 of the 16.
    assert len(resolved) >= 8
    for p in resolved:
        assert p.suffix == ".bddl"


def test_flagged_from_manifest(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task": "x.bddl",
                        "axis_subset": ["position"],
                        "seed": 0,
                        "g0": "pass",
                        "g1": "pass",
                        "g2": "pass",
                        "g3": "pass",
                        "g5": "pass",
                        "g6": "pass",
                    }
                ),
                json.dumps(
                    {
                        "task": "y.bddl",
                        "axis_subset": ["camera"],
                        "seed": 1,
                        "g0": "pass",
                        "g1": "pass",
                        "g2": "pass",
                        "g3": "fail",
                        "g5": "pass",
                        "g6": "pass",
                    }
                ),
                "",  # blank line tolerated
                "not json",
            ]
        )
    )
    out = review_sampler.flagged_from_manifest(manifest)
    assert out == [("y.bddl", ["camera"], 1)]


def test_flagged_tasks_from_rca(tmp_path):
    md = tmp_path / "rca.md"
    md.write_text(
        "header text\n\n"
        "| Task | A | B |\n"
        "|---|---|---|\n"
        "| turn_on_the_stove | 63 | 17 |\n"
        "| put_the_bowl_on_the_plate | 39 | 9 |\n"
    )
    out = review_sampler.flagged_tasks_from_rca(md)
    assert "turn_on_the_stove" in out and "put_the_bowl_on_the_plate" in out


# ---------------------------------------------------------------------------
# review_aggregate
# ---------------------------------------------------------------------------


def _make_checklist(severities: dict[str, int], reviewer2: dict[str, int] | None = None) -> dict:
    items = []
    for qid, _ in review_packet.CHECKLIST_QUESTIONS:
        item = {"id": qid, "severity": severities.get(qid)}
        if reviewer2 and qid in reviewer2:
            item["reviewer2_severity"] = reviewer2[qid]
        items.append(item)
    return {
        "packet": {"task": "synthetic.bddl", "axis_subset": "none", "seed": 0},
        "items": items,
    }


def test_aggregate_critical_flag(tmp_path):
    pkt = tmp_path / "p1"
    pkt.mkdir()
    doc = _make_checklist({"q1_goal_semantics": 4})
    (pkt / "checklist.yaml").write_text(yaml.safe_dump(doc))
    stats = review_aggregate.aggregate(tmp_path)
    assert stats["total_packets"] == 1
    assert any(c["question"] == "q1_goal_semantics" for c in stats["critical_items"])


def test_aggregate_kappa_two_reviewers(tmp_path):
    # Perfect agreement should give κ = 1 on questions with variance.
    severities = {qid: i % 3 for i, (qid, _) in enumerate(review_packet.CHECKLIST_QUESTIONS)}
    for k in range(5):
        pkt = tmp_path / f"p{k}"
        pkt.mkdir()
        doc = _make_checklist(severities, reviewer2=severities)
        (pkt / "checklist.yaml").write_text(yaml.safe_dump(doc))
    stats = review_aggregate.aggregate(tmp_path)
    assert stats["total_with_two_reviewers"] == 5
    # All non-None per-question kappas should be ~1.0
    for v in stats["per_question_kappa"].values():
        if v is not None:
            assert v == pytest.approx(1.0, abs=1e-9)


def test_kappa_disagreement_below_one():
    # Two raters scoring opposite ends → κ should be negative or small.
    pairs = [(0, 4), (1, 3), (4, 0), (3, 1)]
    k = review_aggregate.cohens_kappa(pairs)
    assert k is not None and k < 0.5


def test_alpha_basic():
    units = [[0, 0], [1, 1], [2, 2], [3, 3]]
    a = review_aggregate.krippendorff_alpha(units)
    assert a is not None and a > 0.9


def test_render_markdown_smoke(tmp_path):
    pkt = tmp_path / "p1"
    pkt.mkdir()
    doc = _make_checklist({"q1_goal_semantics": 3, "q2_entities_present": 0})
    (pkt / "checklist.yaml").write_text(yaml.safe_dump(doc))
    stats = review_aggregate.aggregate(tmp_path)
    md = review_aggregate.render_markdown(stats)
    assert "G7 Human Review" in md
    assert "Inter-rater agreement" in md
    assert "Significant" in md
