# VLM Ambiguity Validation — Model Pinning & Reproducibility

The perturbation audit's deterministic visible-change score is **primary**. A small
secondary Vision-Language-Model (VLM) check (`vision_validation.run_curated_ambiguity_check`)
only resolves borderline cases. Because that VLM verdict can feed regression
comparisons, the model it runs against must be **pinned and recorded**, otherwise new
audit runs silently diverge from the results already on record.

## Canonical model

| Setting | Value |
|---|---|
| Canonical default model | `vertex_ai/gemini-2.0-flash` |
| Override env var | `LIBERO_VLM_MODEL` |
| Default Vertex location | `global` (Gemini 3 preview), region otherwise |

`vertex_ai/gemini-2.0-flash` is the **GA** model that produced the recorded baseline
audit results. It is the canonical default because:

- **Reproducibility:** recorded regression baselines were generated with this model;
  the default must match them so regressions reflect *scene* changes, not *model* changes.
- **Stability:** it is a generally-available model, not a `*-preview` build. Preview
  models (e.g. `gemini-3-flash-preview`) can change behaviour or be withdrawn without
  notice, which makes them unsuitable as a reproducibility anchor.

### Drift that this pin corrects

The code default had drifted to `vertex_ai/gemini-3-flash-preview`, a preview model,
without a corresponding update to recorded baselines — so audit runs silently used a
different VLM than the recorded results. The canonical default is now reconciled back to
`vertex_ai/gemini-2.0-flash`, and code + docs are asserted to agree by a test
(`tests/test_vision_validation.py::test_doc_exists_and_pins_canonical_model`).

### Evaluating a newer model

To trial a different VLM (e.g. Gemini 3) without editing code:

```bash
export LIBERO_VLM_MODEL="vertex_ai/gemini-3-flash-preview"
```

If a newer model is adopted as the baseline, regenerate the recorded results with it,
then update `DEFAULT_VERTEX_VISION_MODEL` in `src/libero_infinity/vision_validation.py`
**and** the canonical value in this document in the same change.

## Reproducibility metadata

Every `VisionValidationResult` records the inputs needed to reproduce the verdict:

- `model` — the **actual** resolved model used for the call (explicit arg → env → default).
- `project`, `location` — resolved Vertex routing.
- `timeout_seconds` — the request timeout applied to this run.
- `timed_out` — `True` only when the failure was a request timeout.

## Timeout handling

| Setting | Value |
|---|---|
| Default timeout | `60` s (`DEFAULT_VLM_TIMEOUT_SECONDS`) |
| Override env var | `LIBERO_VLM_TIMEOUT` (seconds) |

**Basis:** multimodal Gemini Flash calls carrying two images have an observed p95 latency
of roughly 30–40 s; the 60 s default is p95 + headroom. The value is a named, env-tunable
constant rather than a bare literal so the basis is explicit and adjustable.

**Timeout vs. model decision:** a request timeout now yields a distinct
`decision == "timeout"` with `timed_out == True`, instead of being collapsed into the
generic `request_error`. This prevents a network/latency timeout from being mistaken for
a real model verdict (`clear` / `ambiguous` / `not_visible`) during analysis.
