from __future__ import annotations

import logging
from typing import Any

import httpx

from .fx import FxRates
from .lot_valuation import LotValuation
from .models import (
    PendingConfirmation,
    ReferenceCard,
    ScanStats,
    SendicoListing,
    VisualMatch,
)

LOGGER = logging.getLogger(__name__)


class DiscordNotifier:
    def __init__(
        self,
        webhook_url: str | None,
        username: str = "Pokemon Deal Scout",
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.webhook_url = webhook_url
        self.username = username
        self.client = httpx.Client(timeout=30.0, transport=transport)

    def close(self) -> None:
        self.client.close()

    def _send(self, embed: dict[str, Any]) -> str | None:
        """Post an embed, returning the sent message's Discord ID (or None if suppressed).

        Uses ?wait=true so the webhook returns the created message instead of
        an empty 204 -- the ID is how a later run recognises which alert a
        reaction landed on.
        """

        if not self.webhook_url:
            LOGGER.info("Discord webhook is not configured; alert suppressed")
            return None
        clean_embed = {key: value for key, value in embed.items() if value is not None}
        response = self.client.post(
            self.webhook_url,
            params={"wait": "true"},
            json={"username": self.username, "embeds": [clean_embed]},
        )
        if response.is_error:
            # Not response.raise_for_status(): its message embeds the request
            # URL, which for a webhook *is* the secret token. That message
            # tends to get logged or persisted (e.g. into state/completion
            # summaries), so build one that never carries the URL.
            raise RuntimeError(
                f"Discord webhook post failed: HTTP {response.status_code} "
                f"{response.reason_phrase}: {response.text[:300]}"
            )
        try:
            return str(response.json()["id"])
        except (ValueError, KeyError):
            return None

    @staticmethod
    def _evidence(match: VisualMatch) -> str:
        values = match.evidence[:3] or [
            "Visual similarity to the PriceCharting reference"
        ]
        return "\n".join(f"• {value}" for value in values)[:900]

    @staticmethod
    def _short_identity(reference: ReferenceCard) -> str:
        """Name + number only, for a scannable embed title."""

        return " ".join(
            part for part in [reference.name, reference.card_number] if part
        ).strip()

    @staticmethod
    def _full_identity(reference: ReferenceCard) -> str:
        """Name + set + number, for the embed description/subtitle."""

        parts = [
            reference.name,
            reference.set_name,
            f"#{reference.card_number}" if reference.card_number else "",
        ]
        return " ".join(part for part in parts if part).strip()

    @staticmethod
    def _watchlist_badge(reference: ReferenceCard) -> dict[str, Any]:
        return {
            "name": "Matched watchlist",
            "value": f"`{reference.target_id}`",
            "inline": False,
        }

    @staticmethod
    def _seller_status(
        listing: SendicoListing, minimum_positive_ratings: int
    ) -> tuple[bool, str]:
        """Report the seller's positive-ratings position.

        Returns ``(bot_verified, field_value)``. ``bot_verified`` is only
        true when a real minimum is configured AND the listing's own ratings
        count is known to meet it -- a disabled threshold (0) or an unknown
        count can never count as bot-verified, only "verify manually".
        """

        ratings = listing.seller_positive_ratings
        if minimum_positive_ratings > 0:
            if ratings is not None and ratings >= minimum_positive_ratings:
                return True, (
                    f"{ratings:,} positive ratings "
                    f"(meets minimum {minimum_positive_ratings:,})"
                )
            if ratings is not None:
                return False, (
                    f"{ratings:,} positive ratings "
                    f"(below minimum {minimum_positive_ratings:,}) "
                    "— verify manually before buying"
                )
            return False, (
                f"Unverified — confirm at least {minimum_positive_ratings:,} "
                "positive ratings before buying"
            )
        if ratings is not None:
            return False, (
                f"{ratings:,} positive ratings — no minimum configured; "
                "confirm seller reputation manually before buying"
            )
        return False, "Unverified — confirm seller reputation manually before buying"

    def probable(
        self,
        listing: SendicoListing,
        reference: ReferenceCard,
        match: VisualMatch,
    ) -> str | None:
        return self._send(
            {
                "title": (
                    f"POSSIBLE WATCHLIST CARD FOUND — "
                    f"{self._short_identity(reference)}"
                ),
                "description": self._full_identity(reference),
                "url": listing.url,
                "color": 16763904,
                "thumbnail": (
                    {"url": listing.image_urls[0]} if listing.image_urls else None
                ),
                "fields": [
                    {
                        "name": "Status",
                        "value": (
                            "Watchlist match possible; detailed confirmation "
                            "is continuing now."
                        ),
                        "inline": False,
                    },
                    self._watchlist_badge(reference),
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

    def _build_confirmed_embed(
        self,
        listing: SendicoListing,
        reference: ReferenceCard,
        match: VisualMatch,
        *,
        valuation: LotValuation,
        fx_rates: FxRates,
        fee_yen: int,
        seller_criteria: dict,
        confirmed_threshold: float,
        status_prefix: str = "Watchlist match confirmed",
    ) -> dict[str, Any]:
        jpy_to_aud = fx_rates.jpy_to_aud
        usd_to_aud = fx_rates.usd_to_aud

        # Total Sendico cost is deliberately just the listing price plus the
        # Sendico fee -- shipping, freight and GST are excluded, not
        # estimated, since they vary too much per seller/lot to be a useful
        # per-listing number (see the Important field below).
        listing_aud = listing.price_yen * jpy_to_aud
        fee_aud = fee_yen * jpy_to_aud
        total_sendico_cost_aud = listing_aud + fee_aud

        lot_value_aud = valuation.total_priced_usd * usd_to_aud
        variance_aud = lot_value_aud - total_sendico_cost_aud
        variance_pct = (
            (variance_aud / total_sendico_cost_aud * 100)
            if total_sendico_cost_aud
            else 0.0
        )
        variance_direction = "above" if variance_aud >= 0 else "below"
        variance_value = (
            f"A${variance_aud:+,.2f} value {variance_direction} Sendico cost "
            f"({variance_pct:+.1f}%)"
        )

        minimum_positive_ratings = int(
            seller_criteria.get("minimum_positive_ratings", 0) or 0
        )
        bot_verified, seller_field_value = self._seller_status(
            listing, minimum_positive_ratings
        )
        headline = "WATCHLIST CARD CONFIRMED" if bot_verified else "MANUAL SELLER CHECK"
        status = (
            (
                f"{status_prefix}; seller rating verified "
                f"({listing.seller_positive_ratings:,} ≥ "
                f"{minimum_positive_ratings:,} positive ratings)"
            )
            if bot_verified
            else f"{status_prefix}; seller rating must be verified manually"
        )

        comparison_lines = [
            f"PriceCharting value: A${lot_value_aud:,.2f}",
            (
                f"Sendico cost: A${total_sendico_cost_aud:,.2f} "
                f"(¥{listing.price_yen:,} listing + ¥{fee_yen:,} fee)"
            ),
            f"Variance: {variance_value}",
        ]

        priced_lines = [
            (
                f"• 1x {card.display_name} [{card.variant} · {card.grade}] "
                f"— A${card.priced_usd * usd_to_aud:,.2f} "
                f"({card.price_similarity:.0%} price match; {card.grade})"
                + (
                    f" · [PriceCharting]({card.price_source_url})"
                    if card.price_source_url
                    else ""
                )
            )
            for card in valuation.priced_cards
        ] or ["No individual cards cleared the price-match threshold."]

        threshold_pct = f"{valuation.price_match_threshold:.0%}"
        coverage = (
            f"{valuation.identified_count} cards identified at the configured "
            f"confidence; {len(valuation.priced_cards)} priced at "
            f"≥{threshold_pct} match confidence; "
            f"{valuation.unpriced_identified_count} identified cards unpriced; "
            f"{valuation.unidentified_visible_count} visible cards unidentified"
        )

        seller_clause = (
            f"at least {minimum_positive_ratings:,} positive ratings"
            if minimum_positive_ratings > 0
            else "sufficient positive ratings"
        )
        important_text = (
            f"Verify the seller has {seller_clause} before purchase. Shipping, "
            "domestic freight, GST and condition adjustments are excluded. "
            "Premium variants are valued only when explicitly confirmed; "
            "otherwise Normal/Holo is assumed. Graded pricing is used only when "
            "a grading company and grade are detected from the slab or "
            "explicitly claimed in the listing title. Verify the slab, "
            "certification number, identity and authenticity."
        )

        return {
            "title": f"{headline} — {self._short_identity(reference)}",
            "description": self._full_identity(reference),
            "url": listing.url,
            "color": 5763719 if bot_verified else 16763904,
            "thumbnail": (
                {"url": listing.image_urls[0]} if listing.image_urls else None
            ),
            "fields": [
                {"name": "Status", "value": status, "inline": False},
                self._watchlist_badge(reference),
                {
                    "name": "Mercari price",
                    "value": f"¥{listing.price_yen:,} / A${listing_aud:,.2f}",
                    "inline": True,
                },
                {
                    "name": "Sendico fee",
                    "value": f"¥{fee_yen:,} / A${fee_aud:,.2f}",
                    "inline": True,
                },
                {
                    "name": "Total Sendico cost",
                    "value": f"A${total_sendico_cost_aud:,.2f}",
                    "inline": True,
                },
                {
                    "name": "PriceCharting lot value",
                    "value": f"A${lot_value_aud:,.2f}",
                    "inline": True,
                },
                {
                    "name": "Price variance",
                    "value": variance_value,
                    "inline": True,
                },
                {
                    "name": "Seller positives",
                    "value": seller_field_value,
                    "inline": True,
                },
                {
                    "name": "Lot value comparison",
                    "value": "\n".join(comparison_lines),
                    "inline": False,
                },
                {
                    "name": f"Cards priced at ≥{threshold_pct} match",
                    "value": "\n".join(priced_lines)[:1000],
                    "inline": False,
                },
                {
                    "name": "Coverage",
                    "value": coverage,
                    "inline": False,
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
                {
                    "name": "Important",
                    "value": important_text,
                    "inline": False,
                },
            ],
            "footer": {
                "text": (
                    f"Confirmed match ≥{confirmed_threshold:.0%} · "
                    f"lot pricing ≥{threshold_pct} · "
                    f"FX: 1 JPY=A${jpy_to_aud:.6f}, 1 USD=A${usd_to_aud:.4f} "
                    f"({fx_rates.source}, {fx_rates.fetched_at})"
                )
            },
        }

    def confirmed(
        self,
        listing: SendicoListing,
        reference: ReferenceCard,
        match: VisualMatch,
        *,
        valuation: LotValuation,
        fx_rates: FxRates,
        fee_yen: int,
        seller_criteria: dict,
        confirmed_threshold: float,
    ) -> tuple[str | None, dict[str, Any]]:
        embed = self._build_confirmed_embed(
            listing,
            reference,
            match,
            valuation=valuation,
            fx_rates=fx_rates,
            fee_yen=fee_yen,
            seller_criteria=seller_criteria,
            confirmed_threshold=confirmed_threshold,
        )
        return self._send(embed), embed

    def completion(
        self,
        stats: ScanStats,
        *,
        status: str = "Completed normally",
    ) -> str | None:
        models = ", ".join(
            f"{name} ({count})" for name, count in stats.models_used.items()
        ) or "None"
        listing_value = (
            f"Found: **{stats.found}**\n"
            f"Candidates: **{stats.candidates}**\n"
            f"Hydrated: **{stats.hydrated}**\n"
            f"Sold: **{stats.skipped_sold}**\n"
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

    def card_confirmed(self, confirmation: PendingConfirmation) -> str | None:
        """Post to the confirmed-cards channel once a user reacts to an alert.

        Call this on an instance whose webhook_url points at that channel,
        not the main alerts one. When the original alert stored its full
        embed (confirmed-type alerts only -- see PendingConfirmation.embed),
        replay those exact numbers here instead of a bare identity stub, so
        this channel carries the same information the user already reacted
        to even if prices/FX have since moved.
        """

        if confirmation.embed:
            embed = {
                **confirmation.embed,
                "title": f"CARD CONFIRMED — {confirmation.card_name}",
                "color": 3066993,
                "fields": [
                    *confirmation.embed.get("fields", []),
                    {
                        "name": "Originally alerted as",
                        "value": confirmation.alert_type,
                        "inline": True,
                    },
                    {
                        "name": "Alert sent",
                        "value": confirmation.sent_at,
                        "inline": True,
                    },
                ],
            }
            return self._send(embed)

        return self._send(
            {
                "title": "CARD CONFIRMED",
                "description": f"Verified: **{confirmation.card_name}**",
                "url": confirmation.listing_url,
                "color": 3066993,
                "fields": [
                    {
                        "name": "Listing",
                        "value": confirmation.listing_url,
                        "inline": False,
                    },
                    {
                        "name": "Originally alerted as",
                        "value": confirmation.alert_type,
                        "inline": True,
                    },
                    {
                        "name": "Alert sent",
                        "value": confirmation.sent_at,
                        "inline": True,
                    },
                ],
            }
        )
