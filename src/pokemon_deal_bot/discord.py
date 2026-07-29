from __future__ import annotations

import logging
from typing import Any

import httpx

from .fx import FxRates
from .lot_valuation import LotValuation
from .models import ReferenceCard, ScanStats, SendicoListing, VisualMatch

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

    def _send(self, embed: dict[str, Any]) -> bool:
        if not self.webhook_url:
            LOGGER.info("Discord webhook is not configured; alert suppressed")
            return False
        clean_embed = {key: value for key, value in embed.items() if value is not None}
        response = self.client.post(
            self.webhook_url,
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
        valuation: LotValuation,
        fx_rates: FxRates,
        costs: dict,
        fee_yen: int,
    ) -> bool:
        jpy_to_aud = fx_rates.jpy_to_aud
        usd_to_aud = fx_rates.usd_to_aud

        listing_aud = listing.price_yen * jpy_to_aud
        fee_aud = fee_yen * jpy_to_aud
        domestic_shipping_aud = (
            float(costs.get("domestic_shipping_yen", 0.0)) * jpy_to_aud
        )
        international_freight_aud = float(costs.get("international_freight_aud", 0.0))
        gst_rate = float(costs.get("au_import_gst_rate", 0.0))
        subtotal_aud = (
            listing_aud + fee_aud + domestic_shipping_aud + international_freight_aud
        )
        gst_aud = subtotal_aud * gst_rate
        total_sendico_cost_aud = subtotal_aud + gst_aud

        lot_value_aud = valuation.total_priced_usd * usd_to_aud
        variance_aud = lot_value_aud - total_sendico_cost_aud
        variance_pct = (
            (variance_aud / total_sendico_cost_aud * 100)
            if total_sendico_cost_aud
            else 0.0
        )

        comparison_lines = [
            f"PriceCharting value: A${lot_value_aud:,.2f}",
            (
                f"Sendico cost: A${total_sendico_cost_aud:,.2f} "
                f"(¥{listing.price_yen:,} listing + ¥{fee_yen:,} fee, "
                "est. shipping/freight/GST)"
            ),
            f"Variance: A${variance_aud:+,.2f} ({variance_pct:+.0f}%)",
        ]

        priced_lines = [
            (
                f"• 1x {card.display_name} [{card.variant} · {card.grade}] "
                f"— A${card.priced_usd * usd_to_aud:,.2f} "
                f"({card.price_similarity:.0%} price match; {card.grade})"
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
                        "name": "Mercari price / Sendico fee / Total Sendico cost",
                        "value": (
                            f"¥{listing.price_yen:,} / ¥{fee_yen:,} / "
                            f"A${total_sendico_cost_aud:,.2f}"
                        ),
                        "inline": True,
                    },
                    {
                        "name": "PriceCharting lot value / Price variance",
                        "value": (
                            f"A${lot_value_aud:,.2f} / "
                            f"A${variance_aud:+,.2f} ({variance_pct:+.0f}%)"
                        ),
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
                ],
                "footer": {
                    "text": (
                        f"FX: 1 JPY=A${jpy_to_aud:.6f}, 1 USD=A${usd_to_aud:.4f} "
                        f"({fx_rates.source}, {fx_rates.fetched_at}). "
                        "Shipping, freight and GST are estimates, not exact "
                        "landed cost."
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
