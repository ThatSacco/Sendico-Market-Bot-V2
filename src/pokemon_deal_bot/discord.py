from __future__ import annotations

import logging
from typing import Any

import httpx

from .models import ReferenceCard, ScanStats, SendicoListing, VisualMatch

LOGGER = logging.getLogger(__name__)


class DiscordNotifier:
    def __init__(
        self,
        webhook_url: str | None,
        username: str = "Pokemon Deal Scout",
    ) -> None:
        self.webhook_url = webhook_url
        self.username = username
        self.client = httpx.Client(timeout=30.0)

    def close(self) -> None:
        self.client.close()

    def _send(self, embed: dict[str, Any]) -> bool:
        if not self.webhook_url:
            LOGGER.info("Discord webhook is not configured; alert suppressed")
            return False
        clean_embed = {key: value for key, value in embed.items() if value is not None}
        response = self.client.post(
            self.webhook_url,
            json={"username": self.username, "embeds": [clean_embed]},
        )
        response.raise_for_status()
        return True

    @staticmethod
    def _evidence(match: VisualMatch) -> str:
        values = match.evidence[:3] or [
            "Visual similarity to the PriceCharting reference"
        ]
        return "\n".join(f"• {value}" for value in values)[:900]

    def probable(
        self,
        listing: SendicoListing,
        reference: ReferenceCard,
        match: VisualMatch,
    ) -> bool:
        return self._send(
            {
                "title": "POSSIBLE WATCHLIST CARD FOUND",
                "description": (
                    f"A listing image may contain **{reference.display_name}**. "
                    "Detailed confirmation is continuing now."
                ),
                "url": listing.url,
                "color": 16763904,
                "thumbnail": (
                    {"url": listing.image_urls[0]} if listing.image_urls else None
                ),
                "fields": [
                    {
                        "name": "Listing",
                        "value": listing.title[:1000],
                        "inline": False,
                    },
                    {
                        "name": "Confidence",
                        "value": f"{match.confidence:.0%}",
                        "inline": True,
                    },
                    {
                        "name": "Candidate cells",
                        "value": ", ".join(match.candidate_labels)
                        or "Not isolated",
                        "inline": True,
                    },
                    {
                        "name": "Evidence",
                        "value": self._evidence(match),
                        "inline": False,
                    },
                ],
            }
        )

    def confirmed(
        self,
        listing: SendicoListing,
        reference: ReferenceCard,
        match: VisualMatch,
        *,
        jpy_to_aud: float,
        usd_to_aud: float,
        fee_yen: int,
    ) -> bool:
        acquisition = (listing.price_yen + fee_yen) * jpy_to_aud
        market = (reference.ungraded_usd or 0.0) * usd_to_aud
        variance = market - acquisition
        return self._send(
            {
                "title": "WATCHLIST CARD VISUALLY CONFIRMED",
                "description": (
                    "The listing appears to contain the same exact card artwork as "
                    f"**{reference.display_name}**."
                ),
                "url": listing.url,
                "color": 5763719,
                "thumbnail": (
                    {"url": listing.image_urls[0]} if listing.image_urls else None
                ),
                "fields": [
                    {
                        "name": "Confidence",
                        "value": f"{match.confidence:.0%}",
                        "inline": True,
                    },
                    {
                        "name": "Listing + fee",
                        "value": f"A${acquisition:,.2f}",
                        "inline": True,
                    },
                    {
                        "name": "Target ungraded value",
                        "value": f"A${market:,.2f}",
                        "inline": True,
                    },
                    {
                        "name": "Target-only variance",
                        "value": f"A${variance:+,.2f}",
                        "inline": True,
                    },
                    {
                        "name": "PriceCharting",
                        "value": reference.source_url,
                        "inline": False,
                    },
                    {
                        "name": "Evidence",
                        "value": self._evidence(match),
                        "inline": False,
                    },
                ],
                "footer": {
                    "text": (
                        "Target-only value is conservative and does not include "
                        "other cards in the lot."
                    )
                },
            }
        )

    def completion(
        self,
        stats: ScanStats,
        *,
        status: str = "Completed normally",
    ) -> bool:
        models = ", ".join(
            f"{name} ({count})" for name, count in stats.models_used.items()
        ) or "None"
        listing_value = (
            f"Found: **{stats.found}**\n"
            f"Candidates: **{stats.candidates}**\n"
            f"Hydrated: **{stats.hydrated}**\n"
            f"Held: **{stats.held}**"
        )
        match_value = (
            f"Screened: **{stats.screened}**\n"
            f"Detailed: **{stats.detailed}**\n"
            f"Probable: **{stats.probable_matches}**\n"
            f"Confirmed: **{stats.confirmed_matches}**\n"
            f"Alerts: **{stats.alerts_sent}**"
        )
        gemini_value = (
            f"Requests: **{stats.requests_sent}**\n"
            f"Tokens: **{stats.total_tokens:,}**\n"
            f"Models: **{models}**\n"
            f"Errors: **{stats.processing_errors}**"
        )
        return self._send(
            {
                "title": "SENDICO SCAN COMPLETED",
                "description": status,
                "color": 5793266,
                "fields": [
                    {"name": "Listings", "value": listing_value, "inline": True},
                    {"name": "Matches", "value": match_value, "inline": True},
                    {"name": "Gemini", "value": gemini_value, "inline": True},
                ],
            }
        )
