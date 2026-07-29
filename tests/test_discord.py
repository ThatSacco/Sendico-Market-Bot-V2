import httpx
import pytest

from pokemon_deal_bot.discord import DiscordNotifier


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
