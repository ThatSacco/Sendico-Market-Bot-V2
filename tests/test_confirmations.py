import asyncio

import httpx

from pokemon_deal_bot.confirmations import ConfirmationStore, process_pending_confirmations
from pokemon_deal_bot.discord import DiscordNotifier
from pokemon_deal_bot.discord_reactions import DiscordReactionClient
from pokemon_deal_bot.models import PendingConfirmation


def _confirmation(message_id: str = "123", **overrides) -> PendingConfirmation:
    defaults = dict(
        message_id=message_id,
        listing_code="m1",
        listing_url="https://sendico.test/m1",
        target_id="victini_black_bolt_97",
        card_name="Victini #97 (Pokemon Japanese Black Bolt)",
        alert_type="probable",
        sent_at="2026-07-30T00:00:00+00:00",
    )
    defaults.update(overrides)
    return PendingConfirmation(**defaults)


def test_add_pending_and_list(tmp_path):
    store = ConfirmationStore(tmp_path / "confirmations.json")
    store.add_pending(_confirmation())

    assert [c.message_id for c in store.pending()] == ["123"]


def test_resolve_confirmed_moves_out_of_pending(tmp_path):
    store = ConfirmationStore(tmp_path / "confirmations.json")
    store.add_pending(_confirmation())

    resolved = store.resolve("123", confirmed=True)

    assert resolved.listing_code == "m1"
    assert store.pending() == []
    assert "123" in store.data["confirmed"]
    assert "confirmed_at" in store.data["confirmed"]["123"]
    assert "123" not in store.data["rejected"]


def test_resolve_rejected_moves_out_of_pending(tmp_path):
    store = ConfirmationStore(tmp_path / "confirmations.json")
    store.add_pending(_confirmation())

    resolved = store.resolve("123", confirmed=False)

    assert resolved.listing_code == "m1"
    assert store.pending() == []
    assert "123" in store.data["rejected"]
    assert "rejected_at" in store.data["rejected"]["123"]


def test_resolve_unknown_message_returns_none(tmp_path):
    store = ConfirmationStore(tmp_path / "confirmations.json")
    assert store.resolve("does-not-exist", confirmed=True) is None


def test_save_and_reload_round_trips(tmp_path):
    path = tmp_path / "confirmations.json"
    store = ConfirmationStore(path)
    store.add_pending(_confirmation("a"))
    store.add_pending(_confirmation("b"))
    store.resolve("a", confirmed=True)
    store.save()

    reloaded = ConfirmationStore(path)
    assert [c.message_id for c in reloaded.pending()] == ["b"]
    assert "a" in reloaded.data["confirmed"]


def test_missing_file_starts_empty(tmp_path):
    store = ConfirmationStore(tmp_path / "does-not-exist.json")
    assert store.pending() == []
    assert store.data["confirmed"] == {}
    assert store.data["rejected"] == {}


def test_process_pending_confirmations_resolves_and_posts_confirmed(tmp_path):
    store = ConfirmationStore(tmp_path / "confirmations.json")
    store.add_pending(_confirmation("confirm-me"))
    store.add_pending(_confirmation("reject-me"))
    store.add_pending(_confirmation("still-pending"))

    def reaction_handler(request: httpx.Request) -> httpx.Response:
        message_id = request.url.path.rsplit("/", 1)[-1]
        reaction = {
            "confirm-me": "✅",
            "reject-me": "❌",
        }.get(message_id)
        reactions = [{"emoji": {"name": reaction}}] if reaction else []
        return httpx.Response(200, json={"reactions": reactions})

    posted = []

    def notifier_handler(request: httpx.Request) -> httpx.Response:
        posted.append(request)
        return httpx.Response(200, json={"id": "999"})

    reaction_client = DiscordReactionClient(
        "token", "channel", transport=httpx.MockTransport(reaction_handler)
    )
    confirmed_notifier = DiscordNotifier(
        "https://discord.com/api/webhooks/1/token",
        transport=httpx.MockTransport(notifier_handler),
    )

    confirmed_count, rejected_count = asyncio.run(
        process_pending_confirmations(
            store, reaction_client, confirmed_notifier, request_interval_seconds=0
        )
    )
    reaction_client.close()
    confirmed_notifier.close()

    assert (confirmed_count, rejected_count) == (1, 1)
    assert len(posted) == 1
    assert [c.message_id for c in store.pending()] == ["still-pending"]
    assert "confirm-me" in store.data["confirmed"]
    assert "reject-me" in store.data["rejected"]


def test_process_pending_confirmations_survives_notifier_failure(tmp_path):
    store = ConfirmationStore(tmp_path / "confirmations.json")
    store.add_pending(_confirmation("confirm-me"))
    store.add_pending(_confirmation("confirm-me-too"))

    def reaction_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"reactions": [{"emoji": {"name": "✅"}}]})

    def failing_notifier_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    reaction_client = DiscordReactionClient(
        "token", "channel", transport=httpx.MockTransport(reaction_handler)
    )
    confirmed_notifier = DiscordNotifier(
        "https://discord.com/api/webhooks/1/token",
        transport=httpx.MockTransport(failing_notifier_handler),
    )

    # A failed post to the confirmed-cards channel must not stop the rest
    # of the batch from being checked and resolved.
    confirmed_count, rejected_count = asyncio.run(
        process_pending_confirmations(
            store, reaction_client, confirmed_notifier, request_interval_seconds=0
        )
    )
    reaction_client.close()
    confirmed_notifier.close()

    assert (confirmed_count, rejected_count) == (2, 0)
    assert store.pending() == []


def _confirming_reaction_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"reactions": [{"emoji": {"name": "✅"}}]})


def _clients(posted: list):
    def notifier_handler(request: httpx.Request) -> httpx.Response:
        import json

        posted.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "999"})

    return (
        DiscordReactionClient(
            "token", "channel", transport=httpx.MockTransport(_confirming_reaction_handler)
        ),
        DiscordNotifier(
            "https://discord.com/api/webhooks/1/token",
            transport=httpx.MockTransport(notifier_handler),
        ),
    )


def test_enricher_supplies_an_embed_for_a_probable_confirmation(tmp_path):
    store = ConfirmationStore(tmp_path / "confirmations.json")
    store.add_pending(_confirmation("confirm-me"))
    posted: list = []
    reaction_client, confirmed_notifier = _clients(posted)
    enriched_embed = {
        "title": "MANUAL SELLER CHECK — Victini 97",
        "fields": [{"name": "Coverage", "value": "1 cards identified", "inline": False}],
    }
    seen: list = []

    async def enrich(confirmation):
        seen.append(confirmation.message_id)
        return enriched_embed

    confirmed_count, _ = asyncio.run(
        process_pending_confirmations(
            store,
            reaction_client,
            confirmed_notifier,
            request_interval_seconds=0,
            enrich=enrich,
        )
    )
    reaction_client.close()
    confirmed_notifier.close()

    assert confirmed_count == 1
    assert seen == ["confirm-me"]
    sent = posted[0]["embeds"][0]
    assert sent["title"] == "CARD CONFIRMED — Victini #97 (Pokemon Japanese Black Bolt)"
    field_names = [field["name"] for field in sent["fields"]]
    assert field_names == ["Coverage", "Originally alerted as", "Alert sent"]
    # The valuation must survive in the store, not just in the Discord post.
    assert store.data["confirmed"]["confirm-me"]["embed"] == enriched_embed


def test_enricher_is_skipped_when_an_embed_was_already_stored(tmp_path):
    store = ConfirmationStore(tmp_path / "confirmations.json")
    stored = {"title": "already here", "fields": []}
    store.add_pending(
        _confirmation("confirm-me", alert_type="confirmed", embed=stored)
    )
    posted: list = []
    reaction_client, confirmed_notifier = _clients(posted)
    calls: list = []

    async def enrich(confirmation):
        calls.append(confirmation.message_id)
        return {"title": "should not be used", "fields": []}

    asyncio.run(
        process_pending_confirmations(
            store,
            reaction_client,
            confirmed_notifier,
            request_interval_seconds=0,
            enrich=enrich,
        )
    )
    reaction_client.close()
    confirmed_notifier.close()

    # A confirmed-type alert already carries the exact embed it sent;
    # re-valuing it would spend Gemini calls to produce worse data.
    assert calls == []
    assert posted[0]["embeds"][0]["title"].startswith("CARD CONFIRMED")
    assert store.data["confirmed"]["confirm-me"]["embed"] == stored


def test_enricher_failure_falls_back_to_the_lightweight_embed(tmp_path):
    store = ConfirmationStore(tmp_path / "confirmations.json")
    store.add_pending(_confirmation("confirm-me"))
    posted: list = []
    reaction_client, confirmed_notifier = _clients(posted)

    async def enrich(confirmation):
        raise RuntimeError("Gemini exploded")

    confirmed_count, _ = asyncio.run(
        process_pending_confirmations(
            store,
            reaction_client,
            confirmed_notifier,
            request_interval_seconds=0,
            enrich=enrich,
        )
    )
    reaction_client.close()
    confirmed_notifier.close()

    # A valuation failure is a presentation loss, never a lost confirmation.
    assert confirmed_count == 1
    assert store.pending() == []
    sent = posted[0]["embeds"][0]
    assert sent["title"] == "CARD CONFIRMED"
    assert store.data["confirmed"]["confirm-me"]["embed"] is None


def test_listing_snapshot_round_trips_through_the_store(tmp_path):
    path = tmp_path / "confirmations.json"
    store = ConfirmationStore(path)
    snapshot = {
        "title": "【未鑑定】まとめ売り",
        "price_yen": 2100,
        "image_urls": ["https://static.mercdn.net/item/detail/orig/photos/m1_1.jpg"],
        "seller_positive_ratings": None,
        "match_confidence": 0.62,
        "match_evidence": ["Visual similarity to the PriceCharting reference"],
    }
    store.add_pending(_confirmation("a", listing_snapshot=snapshot))
    store.save()

    reloaded = ConfirmationStore(path).pending()[0]
    assert reloaded.listing_snapshot == snapshot
