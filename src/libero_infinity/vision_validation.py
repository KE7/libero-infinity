"""Secondary VLM-based ambiguity checks for perturbation audits."""

from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from libero_infinity.perturbation_audit import VisibleChangeScore

DEFAULT_VERTEX_VISION_MODEL = "vertex_ai/gemini-3-flash-preview"
DEFAULT_VERTEX_LOCATION = "global"

_SYSTEM_PROMPT = """You are validating whether a perturbation remains visually interpretable.
The deterministic audit score is primary. You are only resolving borderline ambiguity.
Respond with JSON only:
{"decision": "clear|ambiguous|not_visible", "confidence": 0.0, "reasoning": "..."}"""


@dataclass(frozen=True)
class VisionValidationResult:
    """Structured result from the secondary ambiguity check."""

    decision: str
    confidence: float | None
    reasoning: str
    raw_response: str
    model: str
    project: str
    location: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_curated_ambiguity_check(
    *,
    task_instruction: str,
    visible_change: VisibleChangeScore,
    canonical_image: np.ndarray | bytes | pathlib.Path | str,
    perturbed_image: np.ndarray | bytes | pathlib.Path | str,
    model: str = DEFAULT_VERTEX_VISION_MODEL,
    project: str | None = None,
    location: str | None = None,
    timeout: int = 60,
    temperature: float = 0.0,
    litellm_module: Any | None = None,
) -> VisionValidationResult:
    """Run a small VLM ambiguity check after deterministic scoring."""
    resolved_project = resolve_vertex_project(project)
    resolved_location = resolve_vertex_location(location, model=model)
    litellm_module = litellm_module or _import_litellm()

    messages = build_ambiguity_messages(
        task_instruction=task_instruction,
        visible_change=visible_change,
        canonical_image=canonical_image,
        perturbed_image=perturbed_image,
    )

    try:
        response = litellm_module.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            vertex_project=resolved_project,
            vertex_location=resolved_location,
            response_mime_type="application/json",
        )
    except Exception as exc:
        return VisionValidationResult(
            decision="request_error",
            confidence=None,
            reasoning=str(exc),
            raw_response=str(exc),
            model=model,
            project=resolved_project,
            location=resolved_location,
        )

    content = _extract_response_text(response)
    parsed = parse_vision_validation_response(
        content,
        model=model,
        project=resolved_project,
        location=resolved_location,
    )
    return parsed


def build_ambiguity_messages(
    *,
    task_instruction: str,
    visible_change: VisibleChangeScore,
    canonical_image: np.ndarray | bytes | pathlib.Path | str,
    perturbed_image: np.ndarray | bytes | pathlib.Path | str,
) -> list[dict[str, Any]]:
    """Build a multimodal prompt anchored by the deterministic score."""
    summary = (
        f"Task instruction: {task_instruction}\n"
        f"Deterministic visible-change summary:\n"
        f"- combined_score={visible_change.combined_score:.3f}\n"
        f"- rgb_mean_delta={visible_change.rgb_mean_delta:.4f}\n"
        f"- anchor_mean_displacement_px="
        f"{_format_optional_float(visible_change.anchor_summary.mean_displacement_px)}\n"
        f"- perturbed_visible_fraction="
        f"{_format_optional_float(visible_change.anchor_summary.perturbed_visible_fraction)}\n"
        f"- perturbed_in_frame_fraction="
        f"{_format_optional_float(visible_change.anchor_summary.perturbed_in_frame_fraction)}\n"
        f"- deterministic_vlm_gate={visible_change.should_run_vlm_check}\n\n"
        "Decide only whether the perturbed scene remains visually clear, is borderline "
        "ambiguous, or makes the relevant object/scene not visible."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": summary},
                {"type": "text", "text": "Canonical reference image:"},
                _image_content_part(canonical_image),
                {"type": "text", "text": "Perturbed audit image:"},
                _image_content_part(perturbed_image),
            ],
        },
    ]


def parse_vision_validation_response(
    response: str,
    *,
    model: str,
    project: str,
    location: str,
) -> VisionValidationResult:
    """Parse JSON-ish VLM output into a stable result shape."""
    normalized = re.sub(r"```(?:json)?", "", response).replace("```", "").strip()
    match = re.search(r"\{.*\}", normalized, re.DOTALL)
    if not match:
        return VisionValidationResult(
            decision="parse_error",
            confidence=None,
            reasoning="No JSON object found in response",
            raw_response=response,
            model=model,
            project=project,
            location=location,
        )

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return VisionValidationResult(
            decision="parse_error",
            confidence=None,
            reasoning=f"Invalid JSON response: {exc}",
            raw_response=response,
            model=model,
            project=project,
            location=location,
        )

    decision = _normalize_decision(
        payload.get("decision") or payload.get("verdict") or payload.get("label")
    )
    if decision is None:
        return VisionValidationResult(
            decision="parse_error",
            confidence=None,
            reasoning="Missing or unsupported decision label",
            raw_response=response,
            model=model,
            project=project,
            location=location,
        )

    return VisionValidationResult(
        decision=decision,
        confidence=_coerce_optional_confidence(payload.get("confidence")),
        reasoning=str(payload.get("reasoning") or payload.get("reason") or ""),
        raw_response=response,
        model=model,
        project=project,
        location=location,
    )


def resolve_vertex_project(project: str | None = None) -> str:
    """Resolve the Vertex project from explicit input, env, ADC, or gcloud."""
    if project:
        return project

    for env_name in ("VERTEXAI_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value

    try:
        import google.auth

        _credentials, detected_project = google.auth.default()
    except Exception:
        detected_project = None

    if detected_project:
        return str(detected_project)

    gcloud_value = _gcloud_config_value("project")
    if gcloud_value:
        return gcloud_value

    raise RuntimeError(
        "Unable to resolve a Vertex AI project. Set VERTEXAI_PROJECT or configure "
        "gcloud application-default credentials."
    )


def resolve_vertex_location(
    location: str | None = None,
    *,
    model: str = DEFAULT_VERTEX_VISION_MODEL,
) -> str:
    """Resolve the Vertex location, defaulting Gemini 3 preview models to global."""
    if location:
        return location

    for env_name in ("VERTEXAI_LOCATION", "VERTEX_LOCATION"):
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value

    if "gemini-3-" in model:
        return DEFAULT_VERTEX_LOCATION
    return _gcloud_config_value("ai/region") or DEFAULT_VERTEX_LOCATION


def _import_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:
        raise RuntimeError(
            "litellm is required for VLM ambiguity checks. Install 'litellm[google]'."
        ) from exc
    return litellm


def _gcloud_config_value(key: str) -> str | None:
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", key],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    value = result.stdout.strip()
    if not value or value == "(unset)":
        return None
    return value


def _image_content_part(
    image: np.ndarray | bytes | pathlib.Path | str,
) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": _coerce_image_url(image),
        },
    }


def _coerce_image_url(image: np.ndarray | bytes | pathlib.Path | str) -> str:
    if isinstance(image, np.ndarray):
        return _data_url_from_bytes(_encode_png_bytes(image), mime_type="image/png")
    if isinstance(image, (bytes, bytearray)):
        return _data_url_from_bytes(bytes(image), mime_type="image/png")
    if isinstance(image, pathlib.Path):
        return _data_url_from_path(image)
    if isinstance(image, str):
        if image.startswith(("https://", "http://", "data:")):
            return image
        path = pathlib.Path(image)
        if path.exists():
            return _data_url_from_path(path)
    raise ValueError(f"Unsupported image input: {type(image).__name__}")


def _data_url_from_path(path: pathlib.Path) -> str:
    mime_type = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    return _data_url_from_bytes(path.read_bytes(), mime_type=mime_type)


def _encode_png_bytes(image: np.ndarray) -> bytes:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to send numpy image arrays to the VLM helper."
        ) from exc

    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim not in (2, 3):
        raise ValueError(f"Expected image array with 2 or 3 dims, got {array.shape!r}")

    with io.BytesIO() as buffer:
        Image.fromarray(array).save(buffer, format="PNG")
        return buffer.getvalue()


def _data_url_from_bytes(payload: bytes, *, mime_type: str) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except Exception as exc:
        raise RuntimeError(f"Unexpected LiteLLM response shape: {exc}") from exc

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_chunks = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_chunks.append(str(part.get("text", "")))
        if text_chunks:
            return "\n".join(text_chunks)
    raise RuntimeError(f"Unsupported response content type: {type(content).__name__}")


def _normalize_decision(value: object) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "clear": "clear",
        "clear_change": "clear",
        "visible_change": "clear",
        "ambiguous": "ambiguous",
        "uncertain": "ambiguous",
        "borderline": "ambiguous",
        "not_visible": "not_visible",
        "hidden": "not_visible",
        "occluded": "not_visible",
    }
    return aliases.get(normalized)


def _coerce_optional_confidence(value: object) -> float | None:
    if value is None:
        return None
    confidence = float(value)
    if confidence > 1.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
