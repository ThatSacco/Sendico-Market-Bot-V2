import json
from pathlib import Path

import httpx
import pytest

from pokemon_deal_bot.discord import DiscordNotifier
from pokemon_deal_bot.fx import FxRates
from pokemon_deal_bot.lot_valuation import lot_value
from pokemon_deal_bot.models import (
    LotCard,
    PendingConfirmation,
    ReferenceCard,
    SendicoListing,
    VisualMatch,
)


def _listing(**overrides) -> SendicoListing:
    defaults = dict(
        code="m123",
        url="https://example.com/m123",
        title="Lot",
        price_yen=5000,
    )
    defaults.update(overrides)
    return SendicoListing(**defaults)


def _reference(**overrides) -> ReferenceCard:
    defaults = dict(
        target_id="victini_black_bolt_97",
        source_url="https://pricecharting.com/game/x/victini-97",
        product_id="1",
        name="Victini",
        set_name="Black Bolt",
        card_number="97",
        image_url="",
        image_path=Path("data/reference_images/1.jpg"),
    )
    defaults.update(overrides)
    return ReferenceCard(**defaults)


def _fx_rates() -> FxRates:
    return FxRates(
        jpy_to_aud=0.01,
        usd_to_aud=1.5,
        fetched_at="2026-07-30T00:00:00Z",
        source="live",
    )


def _notifier(handler) -> tuple[DiscordNotifier, dict]:
    captured: dict = {}

    def wrapped(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return handler(request)

    notifier = DiscordNotifier(
        "https://discord.com/api/webhooks/123/token",
        transport=httpx.MockTransport(wrapped),
    )
    return notifier, captured


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


def test_send_succeeds_and_returns_the_message_id():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"id": "1234567890"})

    notifier = DiscordNotifier(
        "https://discord.com/api/webhooks/123/super-secret-token",
        transport=httpx.MockTransport(handler),
    )
    assert notifier._send({"title": "test"}) == "1234567890"
    notifier.close()

    # ?wait=true is what makes Discord return the message (with its id)
    # instead of an empty 204 -- a later run needs that id to recognise
    # which alert a reaction landed on.
    assert captured["params"]["wait"] == "true"


def test_send_without_configured_webhook_is_suppressed():
    notifier = DiscordNotifier(None)
    assert notifier._send({"title": "test"}) is None
    notifier.close()


def test_confirmed_embed_reflects_actual_configured_threshold():
    notifier, captured = _notifier(lambda request: httpx.Response(200, json={"id": "555"}))
    listing = _listing()
    reference = _reference()
    match = VisualMatch(target_id="victini_black_bolt_97", stage="detailed", confidence=0.9, same_card=True)

    # A non-default threshold (0.8, not the usual 0.95) must show up verbatim.
    valuation = lot_value(
        [LotCard(name="Pikachu", priced_usd=10.0, price_similarity=0.85)],
        visible_card_count=1,
        price_match_threshold=0.8,
    )

    message_id, embed = notifier.confirmed(
        listing,
        reference,
        match,
        valuation=valuation,
        fx_rates=_fx_rates(),
        fee_yen=500,
        seller_criteria={"minimum_positive_ratings": 0},
        confirmed_threshold=0.85,
    )
    notifier.close()

    assert message_id == "555"
    fields = captured["body"]["embeds"][0]["fields"]
    assert fields == embed["fields"]
    coverage = next(field["value"] for field in fields if field["name"] == "Coverage")
    priced_field_name = next(
        field["name"] for field in fields if field["name"].startswith("Cards priced")
    )
    assert "80%" in coverage
    assert "95%" not in coverage
    assert priced_field_name == "Cards priced at ≥80% match"


def test_confirmed_embed_priced_card_links_to_pricecharting():
    notifier, captured = _notifier(lambda request: httpx.Response(200, json={"id": "1"}))
    valuation = lot_value(
        [
            LotCard(
                name="Pikachu",
                priced_usd=10.0,
                price_similarity=0.99,
                price_source_url="https://pricecharting.com/game/x/pikachu-1",
            )
        ],
        visible_card_count=1,
    )

    notifier.confirmed(
        _listing(),
        _reference(),
        VisualMatch(target_id="victini_black_bolt_97", stage="detailed", confidence=0.9, same_card=True),
        valuation=valuation,
        fx_rates=_fx_rates(),
        fee_yen=500,
        seller_criteria={"minimum_positive_ratings": 0},
        confirmed_threshold=0.85,
    )
    notifier.close()

    fields = captured["body"]["embeds"][0]["fields"]
    priced_field = next(field for field in fields if field["name"].startswith("Cards priced"))
    assert "[PriceCharting](https://pricecharting.com/game/x/pikachu-1)" in priced_field["value"]


def test_confirmed_embed_shows_manual_seller_check_by_default():
    notifier, captured = _notifier(lambda request: httpx.Response(200, json={"id": "1"}))
    listing = _listing(seller_positive_ratings=None)

    message_id, embed = notifier.confirmed(
        listing,
        _reference(),
        VisualMatch(target_id="victini_black_bolt_97", stage="detailed", confidence=0.95, same_card=True),
        valuation=lot_value([], visible_card_count=0),
        fx_rates=_fx_rates(),
        fee_yen=500,
        seller_criteria={"minimum_positive_ratings": 0},
        confirmed_threshold=0.85,
    )
    notifier.close()

    assert embed["title"].startswith("MANUAL SELLER CHECK")
    assert embed["color"] == 16763904
    status = next(f["value"] for f in embed["fields"] if f["name"] == "Status")
    assert "must be verified manually" in status
    seller_field = next(f["value"] for f in embed["fields"] if f["name"] == "Seller positives")
    assert "Unverified" in seller_field
    watchlist_field = next(f["value"] for f in embed["fields"] if f["name"] == "Matched watchlist")
    assert watchlist_field == "`victini_black_bolt_97`"
    important = next(f["value"] for f in embed["fields"] if f["name"] == "Important")
    assert "sufficient positive ratings" in important


def test_confirmed_embed_shows_bot_verified_when_seller_meets_minimum():
    notifier, captured = _notifier(lambda request: httpx.Response(200, json={"id": "1"}))
    listing = _listing(seller_positive_ratings=500)

    message_id, embed = notifier.confirmed(
        listing,
        _reference(),
        VisualMatch(target_id="victini_black_bolt_97", stage="detailed", confidence=0.95, same_card=True),
        valuation=lot_value([], visible_card_count=0),
        fx_rates=_fx_rates(),
        fee_yen=500,
        seller_criteria={"minimum_positive_ratings": 300},
        confirmed_threshold=0.85,
    )
    notifier.close()

    assert embed["title"].startswith("WATCHLIST CARD CONFIRMED")
    assert embed["color"] == 5763719
    status = next(f["value"] for f in embed["fields"] if f["name"] == "Status")
    assert "verified" in status.lower()
    seller_field = next(f["value"] for f in embed["fields"] if f["name"] == "Seller positives")
    assert "meets minimum 300" in seller_field
    important = next(f["value"] for f in embed["fields"] if f["name"] == "Important")
    assert "at least 300 positive ratings" in important


def test_confirmed_embed_seller_positives_below_minimum():
    notifier, captured = _notifier(lambda request: httpx.Response(200, json={"id": "1"}))
    listing = _listing(seller_positive_ratings=10)

    _, embed = notifier.confirmed(
        listing,
        _reference(),
        VisualMatch(target_id="victini_black_bolt_97", stage="detailed", confidence=0.95, same_card=True),
        valuation=lot_value([], visible_card_count=0),
        fx_rates=_fx_rates(),
        fee_yen=500,
        seller_criteria={"minimum_positive_ratings": 300},
        confirmed_threshold=0.85,
    )
    notifier.close()

    # Below-minimum sellers should never reach this embed in practice --
    # main.py filters them out before matching even starts -- but the
    # field logic must still be honest if it ever does.
    assert embed["title"].startswith("MANUAL SELLER CHECK")
    seller_field = next(f["value"] for f in embed["fields"] if f["name"] == "Seller positives")
    assert "below minimum 300" in seller_field


def test_confirmed_embed_total_cost_is_listing_plus_fee_only():
    notifier, captured = _notifier(lambda request: httpx.Response(200, json={"id": "1"}))
    listing = _listing(price_yen=2100)

    notifier.confirmed(
        listing,
        _reference(),
        VisualMatch(target_id="victini_black_bolt_97", stage="detailed", confidence=0.95, same_card=True),
        valuation=lot_value([], visible_card_count=0),
        fx_rates=FxRates(jpy_to_aud=0.01, usd_to_aud=1.5, fetched_at="2026-07-30T00:00:00Z", source="live"),
        fee_yen=800,
        seller_criteria={"minimum_positive_ratings": 0},
        confirmed_threshold=0.85,
    )
    notifier.close()

    fields = captured["body"]["embeds"][0]["fields"]
    # (2100 + 800) * 0.01 = A$29.00 -- no shipping, freight or GST folded in.
    total_field = next(f["value"] for f in fields if f["name"] == "Total Sendico cost")
    assert total_field == "A$29.00"
    comparison = next(f["value"] for f in fields if f["name"] == "Lot value comparison")
    assert "¥2,100 listing + ¥800 fee" in comparison
    for excluded in ("shipping", "freight", "GST"):
        assert excluded not in comparison
    important = next(f["value"] for f in fields if f["name"] == "Important")
    assert "Shipping, domestic freight, GST and condition adjustments are excluded" in important


def test_probable_embed_includes_watchlist_badge_and_status():
    notifier, captured = _notifier(lambda request: httpx.Response(200, json={"id": "1"}))
    match = VisualMatch(
        target_id="victini_black_bolt_97",
        stage="screening",
        confidence=0.6,
        same_card=True,
        candidate_labels=["O1-1"],
    )

    message_id = notifier.probable(_listing(), _reference(), match)
    notifier.close()

    assert message_id == "1"
    embed = captured["body"]["embeds"][0]
    assert embed["title"] == "POSSIBLE WATCHLIST CARD FOUND — Victini 97"
    assert embed["description"] == "Victini Black Bolt #97"
    watchlist_field = next(f for f in embed["fields"] if f["name"] == "Matched watchlist")
    assert watchlist_field["value"] == "`victini_black_bolt_97`"
    status_field = next(f for f in embed["fields"] if f["name"] == "Status")
    assert "possible" in status_field["value"].lower()


def test_card_confirmed_replays_stored_embed_when_present():
    notifier, captured = _notifier(lambda request: httpx.Response(200, json={"id": "999"}))
    stored_embed = {
        "title": "MANUAL SELLER CHECK — Victini 97",
        "description": "Victini Black Bolt #97",
        "color": 16763904,
        "fields": [{"name": "Status", "value": "...", "inline": False}],
    }
    confirmation = PendingConfirmation(
        message_id="1",
        listing_code="m1",
        listing_url="https://sendico.test/m1",
        target_id="victini_black_bolt_97",
        card_name="Victini #97 (Black Bolt)",
        alert_type="confirmed",
        sent_at="2026-07-30T00:00:00+00:00",
        embed=stored_embed,
    )

    message_id = notifier.card_confirmed(confirmation)
    notifier.close()

    assert message_id == "999"
    sent = captured["body"]["embeds"][0]
    assert sent["title"] == "CARD CONFIRMED — Victini #97 (Black Bolt)"
    assert sent["description"] == "Victini Black Bolt #97"
    field_names = [f["name"] for f in sent["fields"]]
    assert field_names == ["Status", "Originally alerted as", "Alert sent"]


def test_card_confirmed_falls_back_to_lightweight_embed_without_stored_detail():
    notifier, captured = _notifier(lambda request: httpx.Response(200, json={"id": "999"}))
    confirmation = PendingConfirmation(
        message_id="1",
        listing_code="m1",
        listing_url="https://sendico.test/m1",
        target_id="victini_black_bolt_97",
        card_name="Victini #97 (Black Bolt)",
        alert_type="probable",
        sent_at="2026-07-30T00:00:00+00:00",
    )

    notifier.card_confirmed(confirmation)
    notifier.close()

    sent = captured["body"]["embeds"][0]
    assert sent["title"] == "CARD CONFIRMED"
    assert sent["description"] == "Verified: **Victini #97 (Black Bolt)**"
