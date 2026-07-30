import httpx

from pokemon_deal_bot.confirmations import ConfirmationStore, process_pending_confirmations
from pokemon_deal_bot.discord import DiscordNotifier
from pokemon_deal_bot.discord_reactions import DiscordReactionClient
from pokemon_deal_bot.models import PendingConfirmation


def _confirmation(message_id: str = "123") -> PendingConfirmation:
    return PendingConfirmation(
        message_id=message_id,
        listing_code="m1",
        listing_url="https://sendico.test/m1",
        target_id="victini_black_bolt_97",
        card_name="Victini #97 (Pokemon Japanese Black Bolt)",
        alert_type="probable",
        sent_at="2026-07-30T00:00:00+00:00",
    )


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

    confirmed_count, rejected_count = process_pending_confirmations(
        store, reaction_client, confirmed_notifier
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
    confirmed_count, rejected_count = process_pending_confirmations(
        store, reaction_client, confirmed_notifier
    )
    reaction_client.close()
    confirmed_notifier.close()

    assert (confirmed_count, rejected_count) == (2, 0)
    assert store.pending() == []
