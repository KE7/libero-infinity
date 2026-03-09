# LIBERO-Infinity User Review: Fresh Eyes Assessment

**Date:** 2026-03-06
**Methodology:** Three independent reviewers simulated first-time users discovering the repo:
1. **First Impressions** -- A researcher who just landed on the GitHub page (30-second attention span)
2. **Deep Diver** -- A researcher who wants to evaluate their own VLA policy on LIBERO benchmarks
3. **Nitpicker** -- A pedantic code reviewer checking quality, consistency, and correctness

---

## 1. Executive Summary

**Overall Score: 7/10**

LIBERO-Infinity is a well-engineered research framework with professional-quality documentation, a working build system, and a passing test suite -- rare for a robotics research repo. The hero image instantly communicates the value proposition, the architecture docs are excellent, and `make install-full && make test` works out of the box.

However, the repo has a critical "last mile" problem: **a VLA researcher cannot figure out how to evaluate their own policy.** The `--policy` CLI flag raises `NotImplementedError`, the Python API requires undocumented chaining of internal functions, and there is no end-to-end example of plugging in a real model. Combined with placeholder URLs (`<org>`), TODO citations, and duplicate documentation, the repo feels 80% polished but not ready for external users.

| Dimension | Score | Notes |
|-----------|-------|-------|
| Clarity | 7/10 | Good value prop from hero image; jargon in feature descriptions |
| Visual Appeal | 8/10 | Professional layout, good diagrams, well-structured markdown |
| Completeness | 6/10 | Strong internal docs, weak on "how to use this for MY research" |
| Code Quality | 8/10 | Clean source, good tests, proper project structure |
| Ready for Publication | 4/10 | Placeholders, TODOs, broken CLI flag, duplicate docs |

---

## 2. Critical Issues (Things That Would Make a User LEAVE)

### C1. `--policy` CLI flag is broken (NotImplementedError)
**Source:** Deep Diver, First Impressions

The CLI defines `--policy POLICY` with help text "Path to policy checkpoint (optional)" and the eval.py docstring shows `--policy path/to/policy.pt` as an example. But using it raises:
```
NotImplementedError: Policy loading is architecture-specific.
Pass `policy` kwarg directly to evaluate() instead.
```

This is the #1 thing a VLA researcher comes to this repo to do. The flag should either be implemented (with a plugin/registry system) or **removed entirely** with clear documentation that the Python API is required.

### C2. No documentation for evaluating YOUR OWN policy end-to-end
**Source:** Deep Diver, First Impressions

There is no complete, copy-paste-ready example showing:
- How to wrap a HuggingFace VLA as a policy callable
- How to pass task instructions to the policy (the obs dict doesn't include them)
- How to handle action chunks (VLAs like pi0.5 output 50 actions at once)
- How to set image resolution to match your model's training resolution
- How to build state vectors from the obs dict (quaternion-to-axisangle conversion)
- How to iterate over all tasks in a benchmark suite

The Python API example shows `policy=my_policy_fn` but the complete workflow for perturbation + evaluation requires manually chaining `TaskConfig.from_bddl()` -> `generate_scenic_file()` -> `evaluate()`, which is never documented.

### C3. Placeholder URLs (`<org>`) throughout the repo
**Source:** First Impressions, Nitpicker

The clone URL `git clone https://github.com/<org>/libero-infinity.git` appears in:
- README.md Quick Start (lines 59, 68)
- README.md Links table (line 267)
- docs/installation.md (lines 19, 37)
- docs/contributing.md (line 15, as `<your-username>`)

A user landing on the GitHub page literally cannot clone the repo by copy-pasting the Quick Start.

### C4. Quick Start "first evaluation" produces 0% success with no explanation
**Source:** First Impressions

Running the Quick Start evaluation command without `--policy` results in 0% success (the robot sits still for 300 steps using a zero-action default policy). There is no explanation that a policy is needed, no indication this is expected behavior. A new user would conclude the tool is broken.

### C5. Citation has unfilled placeholders
**Source:** Nitpicker, First Impressions

```bibtex
author = {TODO},
eprint = {XXXX.XXXXX},
```

Combined with "arXiv: coming soon" badge, this signals the project is incomplete/unpublished and discourages adoption.

---

## 3. Major Issues (Confusing/Frustrating but Workable)

### M1. Duplicate documentation files
- `docs/evaluation.md` (170 lines) and `docs/evaluation_pipeline.md` (246 lines) cover the same topic with ~90% overlap but have discrepancies (e.g., `--max-distractors` default: 3 vs 5)
- `docs/perturbations.md` (247 lines) and `docs/scenic_perturbations.md` (368 lines) overlap significantly
- `docs/evaluation.md` and `docs/perturbations.md` are orphaned (not linked from README or any other doc)

**Impact:** Users who read all docs will encounter contradictory information and waste time reading the same content twice.

### M2. Inconsistent project display name (7 forms)
The project name appears as: `LIBERO-$\infty$`, `Libero-Infinity`, `LIBERO-Infinity`, `Libero-∞`, `libero-infinity`, `libero_infinity`. While some variation is expected (Python imports vs display name), having 4+ display-name variants across docs is confusing. Pick one canonical form.

### M3. Policy gets no task instruction text
The `evaluate()` function's policy signature is `(obs_dict) -> action`, but VLAs need a natural language instruction (e.g., "put the bowl on the plate"). The obs dict doesn't contain this. Users must capture it from `TaskConfig.language` and close over it in their policy function -- but this is never discussed anywhere.

### M4. Key technical details buried or missing
- **Control frequency** (20 Hz) -- set in simulator.py but never documented
- **Image orientation** -- OpenGL origin (bottom-left) requires flipping, but the exact flip convention for VLA models is ambiguous (`[::-1]` vs `[::-1, ::-1]`)
- **State encoding** -- quaternion-to-axisangle conversion not documented
- **No-op warmup steps** -- standard LIBERO eval runs 10 no-op steps; libero-infinity runs 50 physics settling steps (different thing). Not discussed.
- **GPU/hardware requirements** -- MuJoCo is CPU-only, VLA inference needs GPU. No guidance.

### M5. Internal docs mixed with user-facing docs
`docs/scenic_docs_review.md` (Scenic language gap analysis) and `docs/scenic_test_results.md` + `.json` (internal test results) are development artifacts in the user-facing `docs/` folder. They should be moved to `docs/internal/` or removed.

### M6. Jargon without definitions
Terms used without explanation for the target audience (VLA researchers):
- **BDDL** (Behavior Description Definition Language) -- used 15+ times, never defined
- **Scenic 3** -- in the subtitle but never explained in the README body
- **i.i.d. test scenes** -- stats jargon not all robotics researchers know
- **Wilson 95% confidence intervals** -- unexplained statistical methodology
- **Cross-entropy Bayesian optimization** -- feature description for adversarial search

---

## 4. Minor Issues (Polish Items)

### m1. `docs/architecture.md` file map references stale filenames
References `perturbations.md` and `evaluation.md` instead of their updated counterparts `scenic_perturbations.md` and `evaluation_pipeline.md`.

### m2. Inconsistent asset class counts
README says "34 asset variant pools"; BLOGPOST says "39 classes"; perturbation docs say "10 classes" (for distractors) and "34 classes" (for variants). The numbers refer to different things but are presented confusingly.

### m3. `--max-distractors` default inconsistency
`docs/evaluation.md` says default is 3; `docs/evaluation_pipeline.md` says 5; source code says 5.

### m4. Large hero image
`assets/perturbation_gallery.png` is 1.5 MB. Consider compressing with pngquant or similar.

### m5. Copyright year
LICENSE says "Copyright (c) 2025" but current year is 2026.

### m6. Missing `[project.urls]` in pyproject.toml
No homepage, repository, or documentation URLs defined.

### m7. No benchmark results in README
No table showing actual VLA policy performance (e.g., "Pi0.5 achieves X% on libero_spatial with position perturbation"). Without results, users can't judge if the framework produces meaningful evaluations.

### m8. Blog post in repo root
`BLOGPOST.md` (321 lines) is referenced from README but lives as a markdown file in the repo root. Consider hosting on GitHub Pages or a blog platform.

### m9. Gym deprecation warning
Running eval prints `Gym has been unmaintained since 2022...` due to pinned `gym==0.25.2`. The warning isn't suppressed and looks alarming.

### m10. Missing Python 3.12+ classifiers
pyproject.toml only lists `Programming Language :: Python :: 3.11` but not 3.12 or 3.13.

---

## 5. Nitpicks (Formatting, Typos)

### n1. No actual typos found
All prose text was checked -- no spelling errors or grammar issues detected (credit to the authors).

### n2. Mermaid diagrams require GitHub rendering
`assets/architecture_diagram.md` and `assets/perturbation_types.md` contain mermaid diagrams that only render on GitHub.

### n3. GitHub-specific HTML centering
README uses `<div align="center">` blocks -- standard for GitHub but not portable.

### n4. `scenic_test_results.json` in docs/
JSON data file in `docs/` is an unconventional location. Consider `tests/` or `data/`.

### n5. `uv.lock` gitignored
Deliberate choice. For a publishable tool, consider committing for reproducible installs.

---

## 6. What Works Well (Praise Where Deserved)

### Hero image is outstanding
The perturbation gallery (default vs 5 perturbation types) instantly communicates the value proposition without reading a word. Best-in-class for a research repo README.

### Installation actually works
`make install-full` completes cleanly, `make test` passes 94/94 tests. This is rare for robotics research codebases and deserves recognition.

### Architecture documentation is excellent
`docs/architecture.md` has clear ASCII art diagrams, explains design decisions with rationale, provides a complete file map, and documents MuJoCo naming conventions. Professional quality.

### Perturbation documentation is thorough
Both the table in the README and the detailed docs explain all 6 perturbation axes with parameters, distributions, constraints, and visual examples. The composability system is well-designed.

### Clean project structure
Proper `src/` layout, hatchling build system, entry points, test suite with skip markers, `.gitignore` coverage. The codebase follows modern Python packaging best practices.

### Well-structured test suite
Tests are tiered (Scenic-only -> LIBERO integration -> Gym wrapper), properly use skip markers for optional dependencies, and have shared fixtures.

### Python API design is clean
The `evaluate(scenic_path, bddl_path, policy, n_scenes)` interface is intuitive. The Gym wrapper following standard `gym.Env` conventions is the right choice.

### Task reversal is innovative
Auto-generating reversed tasks from forward tasks is a novel feature that's well-documented with clear rules.

---

## Appendix: Top 5 Complaints (Priority Order)

1. **Cannot evaluate my own policy** -- `--policy` CLI flag broken, no end-to-end Python example, policy callable can't receive task instructions
2. **Placeholder URLs everywhere** -- `<org>` in clone URLs means a GitHub visitor literally cannot follow the Quick Start
3. **Quick Start gives 0% success with no explanation** -- First experience is confusion/failure
4. **Duplicate/contradictory documentation** -- Two docs for evaluation, two for perturbations, with inconsistencies between them
5. **Citation/arXiv placeholders** -- `author = {TODO}` and "coming soon" badge signal the project isn't ready for use

---

*Report generated by 3-persona parallel review (First Impressions + Deep Diver + Nitpicker), 2026-03-06.*
