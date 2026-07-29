from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from typing import Any

import httpx

from .models import VisualMatch

LOGGER = logging.getLogger(__name__)
_SCHEMA = {
    "type": "object",
    "properties": {
        "same_card": {"type": "boolean"},
        "confidence": {"type": "number"},
        "candidate_labels": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "conflicts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["same_card", "confidence", "candidate_labels", "evidence", "conflicts"],
}
_MULTI_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reference_label": {"type": "string"},
                    "same_card": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "candidate_labels": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "conflicts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["reference_label", "same_card", "confidence"],
            },
        },
    },
    "required": ["matches"],
}


class GeminiBudgetReached(RuntimeError):
    pass


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Gemini response must be a JSON object")
    return value


class GeminiReferenceMatcher:
    def __init__(self, api_key: str, config: dict, limits: dict, *, transport: httpx.BaseTransport | None = None) -> None:
        self.api_key = api_key
        self.models = [str(value) for value in config.get("models") or [config.get("model") or "gemini-3.6-flash"]]
        self.screening_model = str(config.get("screening_model") or "gemini-3.1-flash-lite")
        self.endpoint = f"https://generativelanguage.googleapis.com/{config.get('api_version', 'v1beta')}/interactions"
        self.api_revision = str(config.get("api_revision") or "")
        self.thinking_level = str(config.get("thinking_level") or "low")
        request = limits["gemini_request"]
        budget = limits["token_budget"]
        self.timeout = float(request["request_timeout_seconds"])
        self.max_retries = int(request["max_retries_per_model"])
        self.retry_base = float(request["retry_base_seconds"])
        self.max_output_tokens = int(request["max_completion_tokens"])
        self.max_tokens = int(budget["max_total_tokens_per_run"])
        self.reserve = int(budget["reserve_per_request"])
        self.max_requests = int(budget["max_requests_per_run"])
        self.requests_sent = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.thinking_tokens = 0
        self.total_tokens = 0
        self.model_usage: dict[str, int] = {}
        self.client = httpx.AsyncClient(timeout=self.timeout, transport=transport)

    async def close(self) -> None:
        await self.client.aclose()

    def _budget_check(self) -> None:
        if self.max_requests > 0 and self.requests_sent >= self.max_requests:
            raise GeminiBudgetReached(f"Gemini request cap of {self.max_requests} reached")
        if self.total_tokens + self.reserve >= self.max_tokens:
            raise GeminiBudgetReached(f"Gemini token budget of {self.max_tokens:,} reached")

    @staticmethod
    def _image_part(data: bytes) -> dict[str, str]:
        return {"type": "image", "data": base64.b64encode(data).decode("ascii"), "mime_type": "image/jpeg"}

    def _extract(self, payload: dict[str, Any]) -> str:
        blocks: list[str] = []
        for step in payload.get("steps") or []:
            if step.get("type") != "model_output":
                continue
            for item in step.get("content") or []:
                if item.get("type") == "text" and str(item.get("text") or "").strip():
                    blocks.append(str(item["text"]))
        if not blocks:
            raise ValueError("Gemini response did not contain output text")
        return "\n".join(blocks)

    def _record_usage(self, payload: dict[str, Any], model: str) -> None:
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("total_input_tokens") or 0)
        output_tokens = int(usage.get("total_output_tokens") or 0)
        total = int(usage.get("total_tokens") or input_tokens + output_tokens)
        thinking = max(0, total - input_tokens - output_tokens)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.thinking_tokens += thinking
        self.total_tokens += total
        self.model_usage[model] = self.model_usage.get(model, 0) + 1

    async def _request(
        self,
        model: str,
        prompt: str,
        reference_jpeg: bytes,
        candidate_jpeg: bytes,
        *,
        schema: dict[str, Any] = _SCHEMA,
    ) -> dict[str, Any]:
        self._budget_check()
        body = {
            "model": model,
            "input": [
                {"type": "text", "text": prompt},
                self._image_part(reference_jpeg),
                self._image_part(candidate_jpeg),
            ],
            "store": False,
            "generation_config": {
                "thinking_level": self.thinking_level,
                "max_output_tokens": self.max_output_tokens,
            },
            "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
        }
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        if self.api_revision:
            headers["Api-Revision"] = self.api_revision
        last_error: Exception | None = None
        for retry in range(self.max_retries + 1):
            self._budget_check()
            self.requests_sent += 1
            response = await self.client.post(self.endpoint, headers=headers, json=body)
            if not response.is_error:
                payload = response.json()
                self._record_usage(payload, model)
                return _json_object(self._extract(payload))
            message = response.text[:1200]
            last_error = RuntimeError(f"Gemini {model} HTTP {response.status_code}: {message}")
            if response.status_code not in {429, 500, 502, 503, 504} or retry >= self.max_retries:
                break
            await asyncio.sleep(min(30.0, self.retry_base * (2 ** retry)))
        raise last_error or RuntimeError("Gemini request failed")

    async def compare(
        self,
        *,
        target_id: str,
        reference_name: str,
        reference_jpeg: bytes,
        candidate_jpeg: bytes,
        stage: str,
    ) -> VisualMatch:
        if stage == "screening":
            model_candidates = [self.screening_model, *self.models]
            prompt = f"""
You are visually screening a Japanese Pokemon-card marketplace listing.
IMAGE 1 is the canonical PriceCharting reference for: {reference_name}.
IMAGE 2 is a labelled contact sheet from one Sendico/Mercari listing.
Find whether any visible card appears to be the same exact printing and artwork as IMAGE 1.
Do not require readable card text or number. Use artwork, layout, borders, colours and illustration composition.
Be recall-oriented: uncertainty should produce a moderate confidence rather than an automatic false.
Return candidate_labels such as O1-1, O1-2, C1-1 for cells that may contain the target.
"confidence" is the probability from 0.0 to 1.0 that the reference card IS
present in IMAGE 2. 0.0 means certainly absent, 1.0 means certainly present.
Never report a high confidence for an absence -- report a low number instead.
"""
        else:
            model_candidates = self.models
            prompt = f"""
Perform an exact visual comparison of a Pokemon card.
IMAGE 1 is the canonical PriceCharting reference for: {reference_name}.
IMAGE 2 is a labelled contact sheet of candidate listing cards/crops.
Decide whether any candidate is the same exact card printing as the reference.
Artwork and card layout are primary. Printed name, number and set are supporting evidence only and may be unreadable.
Different artwork, framing, pose, background composition or card template are conflicts.
Return same_card true only when the visual identity is convincing, but do not reject merely because text is blurred.
"confidence" is the probability from 0.0 to 1.0 that the reference card IS
present in IMAGE 2. 0.0 means certainly absent, 1.0 means certainly present.
Never report a high confidence for an absence -- report a low number instead.
"""
        errors: list[str] = []
        for model in dict.fromkeys(model_candidates):
            try:
                data = await self._request(model, prompt, reference_jpeg, candidate_jpeg)
                confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
                return VisualMatch(
                    target_id=target_id,
                    stage=stage,
                    confidence=confidence,
                    same_card=bool(data.get("same_card")),
                    candidate_labels=[str(value) for value in data.get("candidate_labels") or []],
                    evidence=[str(value) for value in data.get("evidence") or []],
                    conflicts=[str(value) for value in data.get("conflicts") or []],
                    model=model,
                )
            except GeminiBudgetReached:
                raise
            except Exception as exc:
                errors.append(f"{model}: {exc}")
                LOGGER.warning("Gemini visual comparison failed with %s: %s", model, exc)
        raise RuntimeError("All Gemini models failed: " + " | ".join(errors))

    async def screen_multi(
        self,
        *,
        targets: list[tuple[str, str]],
        reference_strip_jpeg: bytes,
        candidate_jpeg: bytes,
    ) -> dict[str, VisualMatch]:
        """Screen one listing contact sheet against every reference in a single call.

        ``targets`` is ``(target_id, reference_name)`` in the same order the
        reference strip's R1..Rn labels were laid out.
        """
        labels = [f"R{index + 1}" for index in range(len(targets))]
        by_label = dict(zip(labels, targets))
        reference_list = "\n".join(
            f"{label}: {name}" for label, (_, name) in by_label.items()
        )
        prompt = f"""
You are visually screening a Japanese Pokemon-card marketplace listing against
multiple canonical reference cards at once.
IMAGE 1 is a labelled reference strip. Each cell is one canonical PriceCharting
reference card:
{reference_list}
IMAGE 2 is a labelled contact sheet from one Sendico/Mercari listing.
For every reference label that plausibly appears among the listing images in
IMAGE 2, report it in "matches". Do not require readable card text or number.
Use artwork, layout, borders, colours and illustration composition.
Be recall-oriented: uncertainty should produce a moderate confidence rather
than omitting the reference. Omit a reference label entirely only when it
clearly does not appear anywhere in IMAGE 2.
Return candidate_labels such as O1-1, O1-2, C1-1 for listing cells that may
match a given reference.
"confidence" is the probability from 0.0 to 1.0 that that specific reference
card IS present in IMAGE 2. 0.0 means certainly absent, 1.0 means certainly
present. Never report a high confidence for an absence -- report a low number
instead.
"""
        errors: list[str] = []
        for model in dict.fromkeys([self.screening_model, *self.models]):
            try:
                data = await self._request(
                    model,
                    prompt,
                    reference_strip_jpeg,
                    candidate_jpeg,
                    schema=_MULTI_SCHEMA,
                )
                results: dict[str, VisualMatch] = {}
                for item in data.get("matches") or []:
                    label = str(item.get("reference_label") or "")
                    target = by_label.get(label)
                    if target is None:
                        continue
                    target_id, _ = target
                    confidence = max(0.0, min(1.0, float(item.get("confidence") or 0.0)))
                    results[target_id] = VisualMatch(
                        target_id=target_id,
                        stage="screening",
                        confidence=confidence,
                        same_card=bool(item.get("same_card")),
                        candidate_labels=[
                            str(value) for value in item.get("candidate_labels") or []
                        ],
                        evidence=[str(value) for value in item.get("evidence") or []],
                        conflicts=[str(value) for value in item.get("conflicts") or []],
                        model=model,
                    )
                return results
            except GeminiBudgetReached:
                raise
            except Exception as exc:
                errors.append(f"{model}: {exc}")
                LOGGER.warning("Gemini multi-target screening failed with %s: %s", model, exc)
        raise RuntimeError("All Gemini models failed: " + " | ".join(errors))
