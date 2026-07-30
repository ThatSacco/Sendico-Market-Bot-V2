from __future__ import annotations

import logging
import time

import httpx

LOGGER = logging.getLogger(__name__)

CONFIRM_EMOJI = "✅"  # checkmark
REJECT_EMOJI = "❌"  # cross mark


class DiscordReactionClient:
    """Reads reactions on past alert messages using a bot token.

    Separate from DiscordNotifier's webhook: a webhook can only send, so
    telling whether a user reacted to a message needs an actual bot
    credential with read access to the alert channel.
    """

    def __init__(
        self,
        bot_token: str,
        channel_id: str,
        *,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.channel_id = channel_id
        self.client = httpx.Client(
            base_url="https://discord.com/api/v10",
            headers={"Authorization": f"Bot {bot_token}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self.client.close()

    def check_reaction(self, message_id: str, *, max_retries: int = 3) -> str | None:
        """Return "confirmed", "rejected", or None if not yet reacted to.

        A lookup failure (message deleted, channel misconfigured, network
        error) is treated the same as "no reaction yet" rather than raised
        -- a transient problem checking one message must not discard state
        the user hasn't actually acted on, or drop the rest of the batch.

        Discord's per-route limit on this endpoint is tight (observed: 5
        requests/second) -- checking even a couple dozen pending messages
        back-to-back reliably hits a 429, so a single retry attempt isn't
        enough; this backs off using the server's own retry_after each time.
        """

        response: httpx.Response | None = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.get(
                    f"/channels/{self.channel_id}/messages/{message_id}"
                )
            except httpx.HTTPError as exc:
                LOGGER.warning("Could not fetch Discord message %s: %s", message_id, exc)
                return None
            if response.status_code != 429 or attempt >= max_retries:
                break
            retry_after = self._retry_after_seconds(response)
            LOGGER.info(
                "Rate limited checking message %s (attempt %d/%d); retrying in %.2fs",
                message_id,
                attempt + 1,
                max_retries,
                retry_after,
            )
            time.sleep(retry_after)

        if response.is_error:
            LOGGER.warning(
                "Could not fetch Discord message %s: HTTP %s",
                message_id,
                response.status_code,
            )
            return None
        reactions = response.json().get("reactions") or []
        names = {
            str((reaction.get("emoji") or {}).get("name") or "")
            for reaction in reactions
        }
        if CONFIRM_EMOJI in names:
            return "confirmed"
        if REJECT_EMOJI in names:
            return "rejected"
        return None

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float:
        try:
            return max(0.1, float(response.json().get("retry_after", 1.0)))
        except Exception:
            return 1.0
