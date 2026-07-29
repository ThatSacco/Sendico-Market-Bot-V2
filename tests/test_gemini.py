import asyncio
import json

import httpx
import pytest

from pokemon_deal_bot.gemini import (
    GeminiBudgetReached,
    GeminiDegenerateOutputError,
    GeminiReferenceMatcher,
    _json_object,
)


def _limits(max_tokens: int = 10000):
    return {
        "token_budget": {
            "max_total_tokens_per_run": max_tokens,
            "reserve_per_request": 5000,
            "max_requests_per_run": 0,
        },
        "gemini_request": {
            "request_timeout_seconds": 30,
            "max_retries_per_model": 0,
            "retry_base_seconds": 0,
            "max_completion_tokens": 100,
        },
    }


def test_token_budget_stops_before_request():
    config = {"models": ["gemini-test"], "screening_model": "gemini-lite"}
    matcher = GeminiReferenceMatcher("x", config, _limits())
    matcher.total_tokens = 5000
    with pytest.raises(GeminiBudgetReached):
        matcher._budget_check()
    asyncio.run(matcher.close())


def test_interactions_request_uses_two_inline_images_and_structured_output():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        result = {
            "same_card": True,
            "confidence": 0.91,
            "candidate_labels": ["O1-1"],
            "evidence": ["Same artwork"],
            "conflicts": [],
        }
        return httpx.Response(
            200,
            json={
                "usage": {
                    "total_input_tokens": 100,
                    "total_output_tokens": 20,
                    "total_tokens": 125,
                },
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {"type": "text", "text": json.dumps(result)}
                        ],
                    }
                ],
            },
        )

    matcher = GeminiReferenceMatcher(
        "x",
        {
            "models": ["gemini-3.6-flash"],
            "screening_model": "gemini-3.5-flash-lite",
        },
        _limits(20000),
        transport=httpx.MockTransport(handler),
    )
    match = asyncio.run(
        matcher.compare(
            target_id="victini",
            reference_name="Victini #97",
            reference_jpeg=b"reference",
            candidate_jpeg=b"listing",
            stage="screening",
        )
    )
    asyncio.run(matcher.close())

    assert match.same_card is True
    assert match.confidence == 0.91
    assert [part["type"] for part in captured["input"]] == [
        "text",
        "image",
        "image",
    ]
    assert captured["response_format"]["mime_type"] == "application/json"


def test_screen_multi_maps_reference_labels_back_to_target_ids():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        result = {
            "matches": [
                {
                    "reference_label": "R2",
                    "same_card": True,
                    "confidence": 0.8,
                    "candidate_labels": ["O1-1"],
                    "evidence": ["Same artwork"],
                    "conflicts": [],
                }
            ]
        }
        return httpx.Response(
            200,
            json={
                "usage": {
                    "total_input_tokens": 200,
                    "total_output_tokens": 30,
                    "total_tokens": 230,
                },
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {"type": "text", "text": json.dumps(result)}
                        ],
                    }
                ],
            },
        )

    matcher = GeminiReferenceMatcher(
        "x",
        {"models": ["gemini-3.6-flash"], "screening_model": "gemini-3.5-flash-lite"},
        _limits(20000),
        transport=httpx.MockTransport(handler),
    )
    results = asyncio.run(
        matcher.screen_multi(
            targets=[("victini", "Victini #97"), ("ampharos", "Ampharos #123")],
            reference_strip_jpeg=b"strip",
            candidate_jpeg=b"listing",
        )
    )
    asyncio.run(matcher.close())

    assert set(results.keys()) == {"ampharos"}
    assert results["ampharos"].confidence == 0.8
    assert results["ampharos"].same_card is True
    assert captured["response_format"]["schema"]["required"] == ["matches"]


def test_identify_lot_cards_parses_cards_and_visible_count():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        result = {
            "visible_card_count": 3,
            "cards": [
                {
                    "name": "Pikachu",
                    "card_number": "025",
                    "set_name": "Scarlet & Violet",
                    "language": "Japanese",
                    "variant": "AR",
                    "grade": "Ungraded",
                    "confidence": 0.9,
                },
                {"name": "", "confidence": 0.5},
            ],
        }
        return httpx.Response(
            200,
            json={
                "usage": {
                    "total_input_tokens": 50,
                    "total_output_tokens": 10,
                    "total_tokens": 60,
                },
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": json.dumps(result)}],
                    }
                ],
            },
        )

    matcher = GeminiReferenceMatcher(
        "x",
        {"models": ["gemini-3.6-flash"]},
        _limits(20000),
        transport=httpx.MockTransport(handler),
    )
    cards, visible_count = asyncio.run(
        matcher.identify_lot_cards(candidate_jpeg=b"lot")
    )
    asyncio.run(matcher.close())

    assert len(cards) == 1
    assert cards[0].name == "Pikachu"
    assert cards[0].variant == "AR"
    assert visible_count == 3
    assert len(captured["input"]) == 2


def test_json_object_raises_on_degenerate_repetition():
    text = "Charizard VMAX " * 40
    with pytest.raises(GeminiDegenerateOutputError):
        _json_object(text)


def test_json_object_does_not_flag_max_size_legitimate_lot():
    # The lot-identification prompt caps output at 25 cards, so a maximally
    # sized, structurally-repetitive-but-legitimate response (every card
    # sharing the same fields) still can't hit the 30x repeat threshold.
    result = {
        "visible_card_count": 25,
        "cards": [
            {"name": f"Card {i}", "grade": "Ungraded", "confidence": 0.9}
            for i in range(25)
        ],
    }
    assert _json_object(json.dumps(result)) == result


def test_json_object_still_parses_valid_json():
    result = {"same_card": True, "confidence": 0.5}
    assert _json_object(json.dumps(result)) == result


def test_identify_lot_cards_falls_through_on_degenerate_repetition():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        calls.append(model)
        if model == "gemini-3.6-flash":
            # Simulate the observed failure mode: the model gets stuck
            # repeating a short phrase instead of emitting JSON.
            text = "Charizard VMAX " * 200
        else:
            result = {
                "visible_card_count": 1,
                "cards": [
                    {
                        "name": "Charizard",
                        "card_number": "006",
                        "set_name": "Base Set",
                        "language": "Japanese",
                        "variant": "Holo",
                        "grade": "Ungraded",
                        "confidence": 0.8,
                    }
                ],
            }
            text = json.dumps(result)
        return httpx.Response(
            200,
            json={
                "usage": {
                    "total_input_tokens": 50,
                    "total_output_tokens": 500,
                    "total_tokens": 550,
                },
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": text}],
                    }
                ],
            },
        )

    matcher = GeminiReferenceMatcher(
        "x",
        {"models": ["gemini-3.6-flash", "gemini-3.5-flash-lite"]},
        _limits(20000),
        transport=httpx.MockTransport(handler),
    )
    cards, visible_count = asyncio.run(
        matcher.identify_lot_cards(candidate_jpeg=b"lot")
    )
    asyncio.run(matcher.close())

    assert calls == ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
    assert len(cards) == 1
    assert cards[0].name == "Charizard"
    assert visible_count == 1
