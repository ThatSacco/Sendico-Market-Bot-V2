import httpx

from pokemon_deal_bot.discord_reactions import DiscordReactionClient


def _client(handler) -> DiscordReactionClient:
    return DiscordReactionClient(
        "bot-token-abc",
        "999888777",
        transport=httpx.MockTransport(handler),
    )


def test_check_reaction_detects_confirm_emoji():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"reactions": [{"emoji": {"name": "✅"}, "count": 1}]},
        )

    client = _client(handler)
    assert client.check_reaction("123") == "confirmed"
    client.close()


def test_check_reaction_detects_reject_emoji():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"reactions": [{"emoji": {"name": "❌"}, "count": 1}]},
        )

    client = _client(handler)
    assert client.check_reaction("123") == "rejected"
    client.close()


def test_check_reaction_returns_none_when_unreacted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"reactions": []})

    client = _client(handler)
    assert client.check_reaction("123") is None
    client.close()


def test_check_reaction_returns_none_when_reactions_key_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "123"})

    client = _client(handler)
    assert client.check_reaction("123") is None
    client.close()


def test_check_reaction_returns_none_on_http_error_rather_than_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Unknown Message"})

    client = _client(handler)
    assert client.check_reaction("123") is None
    client.close()


def test_requests_use_bot_authorization_and_channel_path():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"reactions": []})

    client = _client(handler)
    client.check_reaction("456")
    client.close()

    assert captured["path"] == "/api/v10/channels/999888777/messages/456"
    assert captured["authorization"] == "Bot bot-token-abc"
