"""Tier-1 tests for VLM model pinning + reproducibility metadata (WS-6).

These tests never make live API calls — the litellm module is always mocked.
They lock in:
  * the canonical default model (no silent drift from the documented baseline),
  * env-var override resolution for model + timeout,
  * recording of the actual model/timeout on every result, and
  * a true timeout being reported distinctly from a generic request error.
"""

from __future__ import annotations

import pathlib
import types

import pytest

from libero_infinity import vision_validation as vv

CANONICAL_MODEL = "vertex_ai/gemini-2.0-flash"
REPO_ROOT = pathlib.Path(vv.__file__).resolve().parents[2]
VLM_DOC = REPO_ROOT / "docs" / "vlm_validation.md"


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _StubAnchorSummary:
    mean_displacement_px = 8.0
    perturbed_visible_fraction = 0.9
    perturbed_in_frame_fraction = 0.95


class _StubVisibleChange:
    combined_score = 0.5
    rgb_mean_delta = 0.02
    should_run_vlm_check = True
    anchor_summary = _StubAnchorSummary()


def _fake_response(content: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


class _FakeLiteLLM:
    """Records the kwargs passed to completion(); returns canned content or raises."""

    def __init__(self, *, content: str | None = None, exc: BaseException | None = None):
        self._content = content
        self._exc = exc
        self.calls: list[dict] = []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return _fake_response(self._content)


def _run(litellm_module, **overrides):
    params = dict(
        task_instruction="pick up the bowl",
        visible_change=_StubVisibleChange(),
        canonical_image="https://example.com/canonical.png",
        perturbed_image="https://example.com/perturbed.png",
        project="test-project",
        location="us-central1",
        litellm_module=litellm_module,
    )
    params.update(overrides)
    return vv.run_curated_ambiguity_check(**params)


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #
def test_default_model_is_documented_baseline():
    assert vv.DEFAULT_VERTEX_VISION_MODEL == CANONICAL_MODEL


def test_resolve_model_default(monkeypatch):
    monkeypatch.delenv(vv.VLM_MODEL_ENV_VAR, raising=False)
    assert vv.resolve_vision_model() == CANONICAL_MODEL


def test_resolve_model_env_override(monkeypatch):
    monkeypatch.setenv(vv.VLM_MODEL_ENV_VAR, "vertex_ai/gemini-3-flash-preview")
    assert vv.resolve_vision_model() == "vertex_ai/gemini-3-flash-preview"


def test_resolve_model_explicit_beats_env(monkeypatch):
    monkeypatch.setenv(vv.VLM_MODEL_ENV_VAR, "vertex_ai/from-env")
    assert vv.resolve_vision_model("vertex_ai/explicit") == "vertex_ai/explicit"


def test_resolve_model_ignores_blank_env(monkeypatch):
    monkeypatch.setenv(vv.VLM_MODEL_ENV_VAR, "   ")
    assert vv.resolve_vision_model() == CANONICAL_MODEL


# --------------------------------------------------------------------------- #
# Timeout resolution
# --------------------------------------------------------------------------- #
def test_resolve_timeout_default(monkeypatch):
    monkeypatch.delenv(vv.VLM_TIMEOUT_ENV_VAR, raising=False)
    assert vv.resolve_vision_timeout() == vv.DEFAULT_VLM_TIMEOUT_SECONDS == 60


def test_resolve_timeout_env_override(monkeypatch):
    monkeypatch.setenv(vv.VLM_TIMEOUT_ENV_VAR, "90")
    assert vv.resolve_vision_timeout() == 90


def test_resolve_timeout_explicit_beats_env(monkeypatch):
    monkeypatch.setenv(vv.VLM_TIMEOUT_ENV_VAR, "90")
    assert vv.resolve_vision_timeout(15) == 15


def test_resolve_timeout_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv(vv.VLM_TIMEOUT_ENV_VAR, "not-a-number")
    assert vv.resolve_vision_timeout() == vv.DEFAULT_VLM_TIMEOUT_SECONDS


# --------------------------------------------------------------------------- #
# Reproducibility metadata recorded on the result
# --------------------------------------------------------------------------- #
def test_result_records_resolved_default_model(monkeypatch):
    monkeypatch.delenv(vv.VLM_MODEL_ENV_VAR, raising=False)
    fake = _FakeLiteLLM(content='{"decision": "clear", "confidence": 0.9, "reasoning": "ok"}')
    result = _run(fake)

    assert result.decision == "clear"
    # actual model recorded in metadata
    assert result.model == CANONICAL_MODEL
    assert result.to_dict()["model"] == CANONICAL_MODEL
    # and the same model was actually sent to the API call
    assert fake.calls[0]["model"] == CANONICAL_MODEL
    # timeout recorded for reproducibility
    assert result.timeout_seconds == vv.DEFAULT_VLM_TIMEOUT_SECONDS
    assert result.timed_out is False


def test_result_records_env_override_model(monkeypatch):
    monkeypatch.setenv(vv.VLM_MODEL_ENV_VAR, "vertex_ai/gemini-3-flash-preview")
    fake = _FakeLiteLLM(content='{"decision": "ambiguous", "confidence": 0.4, "reasoning": "x"}')
    result = _run(fake)

    assert result.model == "vertex_ai/gemini-3-flash-preview"
    assert fake.calls[0]["model"] == "vertex_ai/gemini-3-flash-preview"


def test_explicit_model_and_timeout_recorded(monkeypatch):
    monkeypatch.delenv(vv.VLM_MODEL_ENV_VAR, raising=False)
    fake = _FakeLiteLLM(content='{"decision": "clear", "confidence": 1, "reasoning": "ok"}')
    result = _run(fake, model="vertex_ai/custom", timeout=12)

    assert result.model == "vertex_ai/custom"
    assert result.timeout_seconds == 12
    assert fake.calls[0]["timeout"] == 12


# --------------------------------------------------------------------------- #
# Timeout vs request error distinction
# --------------------------------------------------------------------------- #
class _Timeout(Exception):
    """Exception whose class name mimics litellm.Timeout / APITimeoutError."""


def test_timeout_reported_distinctly(monkeypatch):
    monkeypatch.delenv(vv.VLM_MODEL_ENV_VAR, raising=False)
    fake = _FakeLiteLLM(exc=_Timeout("deadline exceeded"))
    result = _run(fake, timeout=42)

    assert result.decision == "timeout"
    assert result.timed_out is True
    assert result.timeout_seconds == 42
    # model still recorded even on failure
    assert result.model == CANONICAL_MODEL


def test_builtin_timeouterror_detected():
    fake = _FakeLiteLLM(exc=TimeoutError("hard timeout"))
    result = _run(fake)
    assert result.decision == "timeout"
    assert result.timed_out is True


def test_generic_error_is_request_error():
    fake = _FakeLiteLLM(exc=ValueError("bad request"))
    result = _run(fake)
    assert result.decision == "request_error"
    assert result.timed_out is False


# --------------------------------------------------------------------------- #
# Code <-> docs must agree (no silent drift)
# --------------------------------------------------------------------------- #
def test_doc_exists_and_pins_canonical_model():
    assert VLM_DOC.exists(), f"Missing canonical-model doc: {VLM_DOC}"
    text = VLM_DOC.read_text(encoding="utf-8")
    assert CANONICAL_MODEL in text, "docs/vlm_validation.md must document the canonical model"
    assert vv.VLM_MODEL_ENV_VAR in text, "doc must mention the override env var"
