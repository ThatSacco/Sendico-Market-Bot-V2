import json
from pathlib import Path

import httpx
import pytest

from pokemon_deal_bot.discord import DiscordNotifier
from pokemon_deal_bot.fx import FxRates
from pokemon_deal_bot.lot_valuation import lot_value
from pokemon_deal_bot.models import LotCard, ReferenceCard, SendicoListing, VisualMatch


def test_send_raises_sanitized_error_without_webhook_url():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Invalid Webhook Token", "code": 50027})

    notifier = DiscordNotifier(
        "https://discord.com/api/webhooks/123/super-secret-token",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError) as excinfo:
        notifier._send({"title": "test"})
    notifier.close()

    message = str(excinfo.value)
    assert "super-secret-token" not in message
    assert "discord.com" not in message
    assert "401" in message
    assert "Invalid Webhook Token" in message


def test_send_succeeds_on_2xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    notifier = DiscordNotifier(
        "https://discord.com/api/webhooks/123/super-secret-token",
        transport=httpx.MockTransport(handler),
    )
    assert notifier._send({"title": "test"}) is True
    notifier.close()


def test_send_without_configured_webhook_is_suppressed():
    notifier = DiscordNotifier(None)
    assert notifier._send({"title": "test"}) is False
    notifier.close()


def test_confirmed_embed_reflects_actual_configured_threshold():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(204)

    notifier = DiscordNotifier(
        "https://discord.com/api/webhooks/123/token",
        transport=httpx.MockTransport(handler),
    )
    listing = SendicoListing(code="m123", url="https://example.com/m123", title="Lot", price_yen=5000)
    reference = ReferenceCard(
        target_id="victini_black_bolt_97",
        source_url="https://pricecharting.com/game/x/victini-97",
        product_id="1",
        name="Victini",
        set_name="Black Bolt",
        card_number="97",
        image_url="",
        image_path=Path("data/reference_images/1.jpg"),
    )
    match = VisualMatch(target_id="victini_black_bolt_97", stage="detailed", confidence=0.9, same_card=True)
    fx_rates = FxRates(jpy_to_aud=0.01, usd_to_aud=1.5, fetched_at="2026-07-30T00:00:00Z", source="live")

    # A non-default threshold (0.8, not the usual 0.95) must show up verbatim.
    valuation = lot_value(
        [LotCard(name="Pikachu", priced_usd=10.0, price_similarity=0.85)],
        visible_card_count=1,
        price_match_threshold=0.8,
    )

    assert notifier.confirmed(
        listing,
        reference,
        match,
        valuation=valuation,
        fx_rates=fx_rates,
        costs={},
        fee_yen=500,
    )
    notifier.close()

    fields = captured["body"]["embeds"][0]["fields"]
    coverage = next(field["value"] for field in fields if field["name"] == "Coverage")
    priced_field_name = next(
        field["name"] for field in fields if field["name"].startswith("Cards priced")
    )
    assert "80%" in coverage
    assert "95%" not in coverage
    assert priced_field_name == "Cards priced at ≥80% match"
