"""Unit tests for secondary VLM ambiguity validation helpers."""

from __future__ import annotations

from types import SimpleNamespace

from libero_infinity.perturbation_audit import AnchorPixelSummary, VisibleChangeScore
from libero_infinity.vision_validation import (
    DEFAULT_VERTEX_VISION_MODEL,
    build_ambiguity_messages,
    parse_vision_validation_response,
    run_curated_ambiguity_check,
)


def _sample_visible_change(*, should_run_vlm_check: bool = True) -> VisibleChangeScore:
    return VisibleChangeScore(
        rgb_mean_delta=0.024,
        rgb_score=0.8,
        anchor_summary=AnchorPixelSummary(
            anchor_count=2,
            canonical_visible_fraction=1.0,
            perturbed_visible_fraction=0.5,
            canonical_in_frame_fraction=1.0,
            perturbed_in_frame_fraction=0.5,
            matched_fraction=0.5,
            mean_displacement_px=7.5,
            max_displacement_px=10.0,
        ),
        anchor_displacement_score=0.625,
        anchor_visibility_score=0.5,
        combined_score=0.67,
        material_rgb_change=True,
        material_anchor_motion=True,
        anchor_visibility_ok=False,
        material_visible_change=False,
        should_run_vlm_check=should_run_vlm_check,
    )


def test_parse_vision_validation_response_accepts_fenced_json():
    response = """```json
    {"decision": "not_visible", "confidence": 85, "reasoning": "target is occluded"}
    ```"""

    result = parse_vision_validation_response(
        response,
        model=DEFAULT_VERTEX_VISION_MODEL,
        project="demo-project",
        location="global",
    )

    assert result.decision == "not_visible"
    assert result.confidence == 0.85
    assert result.reasoning == "target is occluded"


def test_build_ambiguity_messages_includes_deterministic_summary():
    messages = build_ambiguity_messages(
        task_instruction="place the bowl on the plate",
        visible_change=_sample_visible_change(),
        canonical_image="https://example.com/canonical.png",
        perturbed_image="https://example.com/perturbed.png",
    )

    user_parts = messages[1]["content"]
    assert any(
        "combined_score=0.670" in part.get("text", "")
        for part in user_parts
        if part["type"] == "text"
    )
    assert user_parts[2]["image_url"]["url"] == "https://example.com/canonical.png"
    assert user_parts[4]["image_url"]["url"] == "https://example.com/perturbed.png"


def test_run_curated_ambiguity_check_uses_mocked_litellm():
    calls: list[dict] = []

    class _FakeLiteLLM:
        @staticmethod
        def completion(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"decision":"ambiguous","confidence":0.42,'
                                '"reasoning":"borderline view"}'
                            )
                        )
                    )
                ]
            )

    result = run_curated_ambiguity_check(
        task_instruction="place the bowl on the plate",
        visible_change=_sample_visible_change(),
        canonical_image="data:image/png;base64,AAAA",
        perturbed_image="data:image/png;base64,BBBB",
        project="demo-project",
        location="global",
        litellm_module=_FakeLiteLLM,
    )

    assert result.decision == "ambiguous"
    assert result.confidence == 0.42
    assert calls[0]["model"] == DEFAULT_VERTEX_VISION_MODEL
    assert calls[0]["vertex_project"] == "demo-project"
    assert calls[0]["vertex_location"] == "global"
    assert calls[0]["response_mime_type"] == "application/json"


def test_run_curated_ambiguity_check_returns_request_error_on_completion_failure():
    class _FailingLiteLLM:
        @staticmethod
        def completion(**_kwargs):
            raise RuntimeError("Vertex quota exhausted")

    result = run_curated_ambiguity_check(
        task_instruction="place the bowl on the plate",
        visible_change=_sample_visible_change(),
        canonical_image="data:image/png;base64,AAAA",
        perturbed_image="data:image/png;base64,BBBB",
        project="demo-project",
        location="global",
        litellm_module=_FailingLiteLLM,
    )

    assert result.decision == "request_error"
    assert "quota exhausted" in result.reasoning
