from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .discord import DiscordNotifier
from .discord_reactions import DiscordReactionClient
from .models import PendingConfirmation

# Given a confirmation the user just approved, produce a rich embed for the
# confirmed-cards channel, or None to fall back to the lightweight one.
Enricher = Callable[[PendingConfirmation], Awaitable[dict[str, Any] | None]]

LOGGER = logging.getLogger(__name__)


class ConfirmationStore:
    """Tracks alert messages awaiting a Discord reaction, and their outcome.

    Kept separate from StateStore's dedup bookkeeping since this is a
    slower-moving, user-curated record meant to be read back later (which
    listings were actually verified, and which were false alarms), not an
    internal cache.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        empty = {"version": 1, "pending": {}, "confirmed": {}, "rejected": {}}
        if not self.path.exists():
            return empty
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return empty
        if not isinstance(value, dict):
            return empty
        for bucket in ("pending", "confirmed", "rejected"):
            value.setdefault(bucket, {})
        value.setdefault("version", 1)
        return value

    def add_pending(self, confirmation: PendingConfirmation) -> None:
        self.data["pending"][confirmation.message_id] = asdict(confirmation)

    def pending(self) -> list[PendingConfirmation]:
        return [
            PendingConfirmation(**record) for record in self.data["pending"].values()
        ]

    def resolve(
        self, message_id: str, *, confirmed: bool
    ) -> PendingConfirmation | None:
        """Move a pending confirmation to the confirmed/rejected bucket.

        Returns the resolved record, or None if ``message_id`` wasn't pending
        (already resolved, or never tracked).
        """

        record = self.data["pending"].pop(message_id, None)
        if record is None:
            return None
        bucket = "confirmed" if confirmed else "rejected"
        timestamp_key = "confirmed_at" if confirmed else "rejected_at"
        self.data[bucket][message_id] = {
            **record,
            timestamp_key: datetime.now(timezone.utc).isoformat(),
        }
        return PendingConfirmation(**record)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


async def process_pending_confirmations(
    store: ConfirmationStore,
    reaction_client: DiscordReactionClient,
    confirmed_notifier: DiscordNotifier,
    *,
    request_interval_seconds: float = 0.3,
    enrich: Enricher | None = None,
) -> tuple[int, int]:
    """Check every pending alert for a reaction, resolving what's changed.

    Returns (confirmed_count, rejected_count). Safe to call every run --
    reactions that haven't happened yet are simply left pending.

    Spaced out deliberately: Discord's per-route limit on the message-fetch
    endpoint is tight (observed: 5 requests/second), and a run can easily
    have a few dozen pending alerts to check. check_reaction() already
    retries on a 429, but pacing requests up front means most checks never
    need to.

    ``enrich`` is called only for confirmations that don't already carry a
    stored embed -- in practice, "probable" alerts, which never computed a
    lot valuation at alert time. Paying that cost here, once a human has
    actually vouched for the listing, is far cheaper than doing it for every
    probable alert up front.
    """

    confirmed_count = 0
    rejected_count = 0
    pending_confirmations = store.pending()
    for index, pending in enumerate(pending_confirmations):
        if index > 0 and request_interval_seconds > 0:
            await asyncio.sleep(request_interval_seconds)
        outcome = reaction_client.check_reaction(pending.message_id)
        if outcome is None:
            continue
        resolved = store.resolve(pending.message_id, confirmed=outcome == "confirmed")
        if resolved is None:
            continue
        if outcome == "confirmed":
            confirmed_count += 1
            if enrich is not None and not resolved.embed:
                try:
                    embed = await enrich(resolved)
                    if embed:
                        resolved.embed = embed
                        store.data["confirmed"][resolved.message_id]["embed"] = embed
                except Exception as exc:
                    # Enrichment is a presentation upgrade, never the
                    # confirmation itself -- fall through to the lightweight
                    # embed rather than losing the post entirely.
                    LOGGER.warning(
                        "Could not value the lot for confirmed listing %s: %s",
                        resolved.listing_code,
                        exc,
                    )
            try:
                confirmed_notifier.card_confirmed(resolved)
            except Exception as exc:
                LOGGER.warning(
                    "Could not post confirmed card %s to Discord: %s",
                    resolved.listing_code,
                    exc,
                )
        else:
            rejected_count += 1
    return confirmed_count, rejected_count
