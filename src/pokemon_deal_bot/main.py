from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import logging
from collections.abc import Iterable
from datetime import datetime, timezone

from PIL import Image

from .config import AppConfig, build_search_plan, load_config, scan_signature
from .confirmations import ConfirmationStore, process_pending_confirmations
from .discord import DiscordNotifier
from .discord_reactions import DiscordReactionClient
from .fx import FxRateClient
from .gemini import GeminiBudgetReached, GeminiReferenceMatcher
from .image_processing import (
    download_listing_images,
    extract_card_crops,
    image_file_bytes,
    make_contact_sheet,
)
from .lot_valuation import lot_value
from .mercari import MercariScanner
from .models import (
    LotCard,
    PendingConfirmation,
    ReferenceCard,
    ScanStats,
    SendicoListing,
    VisualMatch,
)
from .pricecharting_search import PriceChartingSearchClient
from .reference import PriceChartingReferenceClient
from .reporting import write_report
from .sendico import SendicoScanner
from .state import StateStore

LOGGER = logging.getLogger(__name__)
SEARCH_CONCURRENCY = 3
HYDRATE_PREFETCH = 4


def _alert_fingerprint(
    listing: SendicoListing,
    target_id: str,
    stage: str,
) -> str:
    """Build a stable alert key that does not change with model confidence."""

    text = (
        f"{listing.code}|{listing.price_yen}|{target_id}|{stage}|"
        f"{'|'.join(listing.image_urls)}"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _candidate_targets(
    listing: SendicoListing,
    references: dict[str, ReferenceCard],
    *,
    compare_all: bool,
) -> list[str]:
    """Prioritise search-associated targets; gate on them when compare_all is false."""

    associated = list(
        dict.fromkeys(
            target_id
            for target_id in listing.candidate_target_ids
            if target_id in references
        )
    )
    if not compare_all:
        return associated
    remaining = [target_id for target_id in references if target_id not in associated]
    return [*associated, *remaining]


def _round_robin_truncate(
    per_term_listings: list[list[SendicoListing]],
    limit: int,
) -> list[SendicoListing]:
    """Interleave unique listings across search terms before truncating.

    Insertion-order truncation would always sacrifice whichever term was
    searched last; round-robin gives every term a fair share of the cap.
    """

    return [
        listing
        for round_items in itertools.zip_longest(*per_term_listings)
        for listing in round_items
        if listing is not None
    ][:limit]


def _load_reference_thumbnail(path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def _resolve_candidate_image(
    batch: list[Image.Image],
    prefix: str,
    labels: list[str],
) -> Image.Image | None:
    """Map a model-claimed candidate label (e.g. "C1-3") back to its source image.

    Normalises case and the letter-O/digit-zero mix-up vision models
    sometimes make when transcribing a small rendered label (our prefixes are
    always letter-initial: "C1-", "O1-").
    """

    normalized_prefix = prefix.upper()
    for label in labels:
        normalized = label.strip().upper()
        if normalized[:1] == "0" and normalized_prefix[:1] != "0":
            normalized = "O" + normalized[1:]
        if not normalized.startswith(normalized_prefix):
            continue
        suffix = normalized[len(normalized_prefix):]
        if not suffix.isdigit():
            continue
        index = int(suffix) - 1
        if 0 <= index < len(batch):
            return batch[index]
    return None


def _resolve_cropped_candidate(
    batch: list[Image.Image],
    batch_is_crop: list[bool],
    prefix: str,
    labels: list[str],
) -> Image.Image | None:
    """Resolve a claimed candidate label, but only if it's a genuine crop.

    A whole listing photo has already been judged once at the "detailed"
    stage; re-showing the model that identical image as a "zoomed in"
    cross-check verifies nothing; it just re-asks the same question of the
    same picture. Only a real OpenCV-extracted crop is a materially closer
    look than what was already judged.
    """

    candidate = _resolve_candidate_image(batch, prefix, labels)
    if candidate is None:
        return None
    index = next(
        (position for position, image in enumerate(batch) if image is candidate),
        None,
    )
    if index is None or not batch_is_crop[index]:
        return None
    return candidate


def _batches(values: list[Image.Image], size: int, maximum: int) -> Iterable[list[Image.Image]]:
    size = max(1, size)
    count = 0
    for start in range(0, len(values), size):
        if maximum > 0 and count >= maximum:
            break
        yield values[start : start + size]
        count += 1


def _report_match(
    rows: list[dict],
    listing: SendicoListing,
    match: VisualMatch,
    *,
    stage: str,
    batch_number: int,
) -> None:
    rows.append(
        {
            "listing_code": listing.code,
            "listing_url": listing.url,
            "target_id": match.target_id,
            "stage": stage,
            "batch": batch_number,
            "confidence": match.confidence,
            "match_score": match.match_score,
            "same_card": match.same_card,
            "model": match.model,
            "candidate_labels": ",".join(match.candidate_labels),
            "evidence": " | ".join(match.evidence),
            "conflicts": " | ".join(match.conflicts),
        }
    )


def _select_scanner(config: AppConfig):
    """Pick the discovery source named in config.yaml's ``source`` key.

    Defaults to Mercari: Sendico began returning a Cloudflare bot
    challenge (HTTP 403) to the scanner on 2026-08-06, taking every search
    to zero results. Sendico remains selectable so the original path can be
    restored the moment it works again.
    """

    source = str(config.raw.get("source") or "mercari").strip().lower()
    if source == "sendico":
        LOGGER.info("Discovery source: Sendico")
        return SendicoScanner, config.raw["sendico"]
    if source != "mercari":
        raise ValueError(
            f"config.yaml 'source' must be 'mercari' or 'sendico', not {source!r}"
        )
    LOGGER.info("Discovery source: Mercari (buy links still point at Sendico)")
    return MercariScanner, config.raw.get("mercari") or {}


def _listing_snapshot(listing: SendicoListing, match: VisualMatch) -> dict:
    """Capture what a later lot valuation will need, at alert time.

    A "probable" alert may sit unreacted for days; by the time someone
    confirms it, the listing can be sold or delisted. Keeping the image
    URLs and price here means the valuation uses exactly what was alerted
    on, and never has to reach the source page again.
    """

    return {
        "title": listing.title,
        "price_yen": listing.price_yen,
        "image_urls": list(listing.image_urls),
        "seller_positive_ratings": listing.seller_positive_ratings,
        "match_confidence": match.confidence,
        "match_evidence": list(match.evidence),
    }


async def _catalogue_lot(
    images: list[Image.Image],
    crops: list[Image.Image],
    matcher: GeminiReferenceMatcher,
    price_client: PriceChartingSearchClient,
    detailed_limits: dict,
) -> tuple[list[LotCard], int]:
    """Identify every card visible in a lot, then price each one.

    Shared by the scan's confirmed-match path and the deferred
    reaction-time valuation, so the two can never drift apart in what they
    count or how they price it.
    """

    lot_cards: list[LotCard] = []
    visible_card_count = 0
    identification_source = crops or images
    for batch_number, batch in enumerate(
        _batches(
            identification_source,
            int(detailed_limits["images_per_batch"]),
            int(detailed_limits["max_batches_per_listing"]),
        ),
        start=1,
    ):
        sheet = make_contact_sheet(
            batch,
            prefix=f"L{batch_number}-",
            max_dimension=int(detailed_limits["max_image_dimension_px"]),
            quality=int(detailed_limits["jpeg_quality"]),
            columns=int(detailed_limits["contact_sheet_columns"]),
        )
        batch_cards, batch_visible = await matcher.identify_lot_cards(
            candidate_jpeg=sheet.jpeg,
        )
        lot_cards.extend(batch_cards)
        visible_card_count += batch_visible

    for card in lot_cards:
        # find_price is synchronous blocking I/O (a cache miss does a real
        # HTTP request); run it off the event loop so it doesn't stall
        # concurrent work.
        price_match = await asyncio.to_thread(
            price_client.find_price,
            name=card.name,
            card_number=card.card_number,
            set_name=card.set_name,
        )
        if price_match is not None:
            card.priced_usd = price_match.ungraded_usd
            card.price_similarity = price_match.similarity
            card.price_source_url = price_match.source_url

    return lot_cards, visible_card_count


def _load_valuation_context(config: AppConfig):
    """Build everything needed to value a lot: references, FX, Gemini, pricing.

    Shared by run() and check_confirmations() so the standalone
    confirmation command values a lot exactly the way a scan would.
    Returns ``(references, fx_rates, matcher, price_client)``; the caller
    owns closing the matcher and price client.
    """

    references_client = PriceChartingReferenceClient(
        config.root,
        cache_hours=int(config.raw.get("pricing", {}).get("reference_cache_hours", 24)),
    )
    references: dict[str, ReferenceCard] = {}
    try:
        for target in config.targets:
            try:
                references[target.id] = references_client.resolve(target)
                LOGGER.info("Loaded reference: %s", references[target.id].display_name)
            except Exception as exc:
                LOGGER.error(
                    "Could not load PriceCharting reference for %s: %s", target.id, exc
                )
    finally:
        references_client.close()
    if not references:
        raise RuntimeError("No PriceCharting references could be loaded")

    pricing_config = config.raw.get("pricing") or {}
    fx_client = FxRateClient(
        config.root,
        manual_jpy_to_aud=float(pricing_config.get("manual_jpy_to_aud", 0.0102)),
        manual_usd_to_aud=float(pricing_config.get("manual_usd_to_aud", 1.52)),
        cache_hours=int(pricing_config.get("fx_cache_hours", 6)),
    )
    fx_rates = fx_client.fetch()
    fx_client.close()
    LOGGER.info(
        "FX rates (%s): 1 JPY = A$%.6f, 1 USD = A$%.4f",
        fx_rates.source,
        fx_rates.jpy_to_aud,
        fx_rates.usd_to_aud,
    )

    lot_valuation_config = config.raw.get("lot_valuation") or {}
    price_client = PriceChartingSearchClient(
        config.root,
        cache_hours=int(lot_valuation_config.get("price_search_cache_hours", 336)),
    )

    matcher_config = dict(config.raw.get("gemini") or {})
    matcher_config["screening_model"] = str(
        config.raw.get("gemini", {}).get("screening_model") or "gemini-3.5-flash-lite"
    )
    matcher = GeminiReferenceMatcher(
        config.gemini_api_key, matcher_config, config.run_limits
    )
    return references, fx_rates, matcher, price_client


def _build_confirmation_enricher(
    config: AppConfig,
    *,
    references: dict[str, ReferenceCard],
    matcher: GeminiReferenceMatcher,
    price_client: PriceChartingSearchClient,
    fx_rates,
    notifier: DiscordNotifier,
):
    """Value a lot at reaction time, for alerts that never got valued.

    A "probable" alert fires straight off screening, before any lot
    cataloguing has happened -- deliberately, since most probable alerts are
    noise and cataloguing every one would be expensive. Once a human reacts
    to say a listing is real, though, it has earned the cost. This rebuilds
    the listing from the snapshot captured at alert time (never re-fetching
    the source page, which may be gone or unreachable by now) and produces
    the same rich embed a bot-confirmed match would have got.
    """

    detailed_limits = config.run_limits["detailed_analysis"]
    lot_valuation_config = config.raw.get("lot_valuation") or {}
    seller_criteria = config.criteria["seller"]
    matching = config.criteria["reference_image_matching"]
    fee_yen = int(config.raw["sendico_fee"]["yen"])

    async def enrich(confirmation: PendingConfirmation) -> dict | None:
        snapshot = confirmation.listing_snapshot or {}
        image_urls = list(snapshot.get("image_urls") or [])
        reference = references.get(confirmation.target_id)
        if not image_urls or reference is None:
            return None

        images = await download_listing_images(
            image_urls,
            maximum=int(detailed_limits["max_images_downloaded"]),
        )
        if not images:
            LOGGER.warning(
                "No images could be downloaded for confirmed listing %s; "
                "posting without a lot valuation",
                confirmation.listing_code,
            )
            return None

        crops = await asyncio.to_thread(
            extract_card_crops,
            images,
            maximum=int(detailed_limits["max_card_crops_per_listing"]),
        )
        lot_cards, visible_card_count = await _catalogue_lot(
            images, crops, matcher, price_client, detailed_limits
        )
        valuation = lot_value(
            lot_cards,
            visible_card_count=visible_card_count,
            price_match_threshold=float(
                lot_valuation_config.get("price_match_threshold", 0.95)
            ),
        )

        listing = SendicoListing(
            code=confirmation.listing_code,
            url=confirmation.listing_url,
            title=str(snapshot.get("title") or confirmation.listing_code),
            price_yen=int(snapshot.get("price_yen") or 0),
            image_urls=image_urls,
            seller_positive_ratings=snapshot.get("seller_positive_ratings"),
        )
        match = VisualMatch(
            target_id=confirmation.target_id,
            stage="reaction_confirmed",
            confidence=float(snapshot.get("match_confidence") or 0.0),
            same_card=True,
            evidence=list(snapshot.get("match_evidence") or []),
        )
        return notifier._build_confirmed_embed(
            listing,
            reference,
            match,
            valuation=valuation,
            fx_rates=fx_rates,
            fee_yen=fee_yen,
            seller_criteria=seller_criteria,
            confirmed_threshold=float(matching["confirmed_threshold"]),
            status_prefix=(
                "Manually confirmed by reaction (the bot could not "
                "independently verify this one)"
            ),
        )

    return enrich


async def _check_pending_confirmations(
    config: AppConfig,
    confirmation_store: ConfirmationStore,
    confirmed_notifier: DiscordNotifier,
    *,
    enabled: bool,
    enrich=None,
) -> None:
    """Check every pending alert for a reaction and resolve it, if configured.

    Shared between the full scan (run()) and the standalone
    check_confirmations() entry point, so the two never drift apart.
    """

    if not enabled:
        return
    reaction_client = DiscordReactionClient(
        config.discord_bot_token, config.discord_alert_channel_id
    )
    try:
        confirmed_count, rejected_count = await process_pending_confirmations(
            confirmation_store, reaction_client, confirmed_notifier, enrich=enrich
        )
        LOGGER.info(
            "Checked pending alert reactions: %d confirmed, %d rejected, %d still pending",
            confirmed_count,
            rejected_count,
            len(confirmation_store.pending()),
        )
    except Exception as exc:
        LOGGER.warning("Could not check alert confirmations: %s", exc)
    finally:
        reaction_client.close()


async def check_confirmations(config_path: str = "config.yaml") -> int:
    """Check pending alert reactions and resolve them, without running a full scan.

    Much cheaper than a full run: no Sendico scanning at all -- just the bot
    token and the confirmed-cards webhook, so this can be triggered as often
    as wanted after reacting to alerts instead of waiting for the next scan.

    GEMINI_API_KEY stays optional. When it is set, a confirmed "probable"
    alert also gets its lot valued and posted with the full embed; without
    it, the confirmed-cards post falls back to the lightweight format and
    everything else behaves identically.
    """

    config = load_config(config_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    discord_enabled = bool(config.raw.get("discord", {}).get("enabled", True))
    discord_username = str(
        config.raw.get("discord", {}).get("username") or "Pokemon Deal Scout"
    )
    confirmed_notifier = DiscordNotifier(
        config.discord_confirmed_webhook_url if discord_enabled else None,
        discord_username,
    )
    confirmation_store = ConfirmationStore(config.path("data/confirmations.json"))
    confirmation_tracking_enabled = bool(
        discord_enabled and config.discord_bot_token and config.discord_alert_channel_id
    )
    if not confirmation_tracking_enabled:
        LOGGER.warning(
            "Confirmation tracking is not configured (need DISCORD_BOT_TOKEN and "
            "discord.alert_channel_id); nothing to check."
        )

    enrich = None
    matcher = None
    price_client = None
    if confirmation_tracking_enabled and config.gemini_api_key:
        try:
            references, fx_rates, matcher, price_client = _load_valuation_context(config)
            enrich = _build_confirmation_enricher(
                config,
                references=references,
                matcher=matcher,
                price_client=price_client,
                fx_rates=fx_rates,
                notifier=confirmed_notifier,
            )
        except Exception as exc:
            LOGGER.warning(
                "Could not prepare lot valuation for confirmations; "
                "confirmed cards will post without it: %s",
                exc,
            )
            enrich = None
    elif confirmation_tracking_enabled:
        LOGGER.info(
            "GEMINI_API_KEY is not set; confirmed probable alerts will post "
            "without a lot valuation."
        )

    try:
        await _check_pending_confirmations(
            config,
            confirmation_store,
            confirmed_notifier,
            enabled=confirmation_tracking_enabled,
            enrich=enrich,
        )
    finally:
        confirmation_store.save()
        if matcher is not None:
            await matcher.close()
        if price_client is not None:
            price_client.close()
        confirmed_notifier.close()
    return 0


async def run(config_path: str = "config.yaml", dry_run: bool = False) -> int:
    config = load_config(config_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not config.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    references, fx_rates, matcher, price_client = _load_valuation_context(config)
    lot_valuation_config = config.raw.get("lot_valuation") or {}
    discord_enabled = bool(config.raw.get("discord", {}).get("enabled", True))
    discord_username = str(
        config.raw.get("discord", {}).get("username") or "Pokemon Deal Scout"
    )
    notifier = DiscordNotifier(
        config.discord_webhook_url if discord_enabled and not dry_run else None,
        discord_username,
    )
    confirmed_notifier = DiscordNotifier(
        config.discord_confirmed_webhook_url
        if discord_enabled and not dry_run
        else None,
        discord_username,
    )
    state = StateStore(
        config.path("data/seen.json"),
        max_listings=int(config.run_limits["state"]["max_seen_listings"]),
    )
    confirmation_store = ConfirmationStore(config.path("data/confirmations.json"))
    confirmation_tracking_enabled = bool(
        discord_enabled
        and not dry_run
        and config.discord_bot_token
        and config.discord_alert_channel_id
    )
    await _check_pending_confirmations(
        config,
        confirmation_store,
        confirmed_notifier,
        enabled=confirmation_tracking_enabled,
        enrich=_build_confirmation_enricher(
            config,
            references=references,
            matcher=matcher,
            price_client=price_client,
            fx_rates=fx_rates,
            notifier=confirmed_notifier,
        ),
    )

    signature = scan_signature(config)
    stats = ScanStats()
    report_rows: list[dict] = []
    status = "Completed normally"

    matching = config.criteria["reference_image_matching"]
    seller_criteria = config.criteria["seller"]
    search_limits = config.run_limits["search"]
    screening_limits = config.run_limits["screening"]
    detailed_limits = config.run_limits["detailed_analysis"]
    search_limit = int(search_limits["total_listings_per_run"])
    screening_limit = int(screening_limits["max_listings_per_run"])
    detailed_limit = int(detailed_limits["max_listings_per_run"])
    compare_all = bool(matching.get("compare_all_active_targets", True))

    reference_jpegs: dict[str, bytes] = {
        target_id: image_file_bytes(reference.image_path)
        for target_id, reference in references.items()
    }
    reference_thumbnails: dict[str, Image.Image] = {
        target_id: _load_reference_thumbnail(reference.image_path)
        for target_id, reference in references.items()
    }

    listings_by_code: dict[str, SendicoListing] = {}
    plan = build_search_plan(config.targets)
    scanner_factory, scanner_config = _select_scanner(config)
    try:
        async with scanner_factory(scanner_config, config.run_limits) as scanner:
            search_semaphore = asyncio.Semaphore(SEARCH_CONCURRENCY)

            async def _run_search(task):
                async with search_semaphore:
                    try:
                        return task, await scanner.search(task.term)
                    except Exception as exc:
                        LOGGER.warning(
                            "Search failed for term %r: %s", task.term, exc
                        )
                        return task, []

            search_results = await asyncio.gather(
                *(_run_search(task) for task in plan)
            )
            per_term_listings: list[list[SendicoListing]] = []
            for task, results in search_results:
                stats.found += len(results)
                first_seen: list[SendicoListing] = []
                for listing in results:
                    current = listings_by_code.get(listing.code)
                    if current is None:
                        listing.candidate_target_ids.extend(task.target_ids)
                        listings_by_code[listing.code] = listing
                        first_seen.append(listing)
                    else:
                        current.candidate_target_ids.extend(task.target_ids)
                per_term_listings.append(first_seen)

            candidates = _round_robin_truncate(per_term_listings, search_limit)
            stats.candidates = len(candidates)
            if len(listings_by_code) > len(candidates):
                stats.held += len(listings_by_code) - len(candidates)

            listing_fingerprints = [
                state.listing_fingerprint(listing, signature) for listing in candidates
            ]
            eligible_positions = [
                index
                for index, listing in enumerate(candidates)
                if state.should_process_fingerprint(
                    listing.code, listing_fingerprints[index]
                )
            ]
            hydrate_semaphore = asyncio.Semaphore(HYDRATE_PREFETCH)
            hydration_tasks: dict[int, asyncio.Task] = {}

            def _start_hydration(position: int) -> None:
                if position >= len(eligible_positions) or position in hydration_tasks:
                    return
                pending_listing = candidates[eligible_positions[position]]

                async def _bounded() -> SendicoListing:
                    async with hydrate_semaphore:
                        return await scanner.hydrate(pending_listing)

                hydration_tasks[position] = asyncio.create_task(_bounded())

            for position in range(min(HYDRATE_PREFETCH, len(eligible_positions))):
                _start_hydration(position)

            eligible_pos = 0
            for index, listing in enumerate(candidates):
                discovery_fingerprint = listing_fingerprints[index]
                if not state.should_process_fingerprint(
                    listing.code, discovery_fingerprint
                ):
                    stats.skipped_seen += 1
                    continue
                # Capped on listings, not target comparisons. Comparing
                # against stats.screened made every active watchlist card
                # consume the budget again: with 3 cards a limit named
                # "max_listings_per_run: 200" actually stopped the run after
                # 67 listings (observed live, 2026-08-06 -- screened hit 201).
                if (
                    screening_limit > 0
                    and stats.listings_screened >= screening_limit
                ):
                    stats.held += len(candidates) - index
                    break

                position = eligible_pos
                hydration_task = hydration_tasks.pop(position)
                eligible_pos += 1
                _start_hydration(position + HYDRATE_PREFETCH)

                try:
                    listing = await hydration_task
                    stats.hydrated += 1
                    if listing.sold_out:
                        # Free to know, and worth acting on: an already-sold
                        # listing can never be bought, so spending image
                        # downloads and Gemini calls on it is pure waste.
                        stats.skipped_sold += 1
                        state.mark_processed(
                            listing,
                            signature,
                            "already sold",
                            fingerprint=discovery_fingerprint,
                        )
                        continue
                    minimum_ratings = int(
                        seller_criteria["minimum_positive_ratings"]
                    )
                    if (
                        listing.seller_positive_ratings is not None
                        and listing.seller_positive_ratings < minimum_ratings
                    ):
                        stats.skipped_seller += 1
                        state.mark_processed(
                            listing,
                            signature,
                            "seller below threshold",
                            fingerprint=discovery_fingerprint,
                        )
                        continue
                    if (
                        listing.seller_positive_ratings is None
                        and not bool(
                            seller_criteria["analyse_unverified_sellers"]
                        )
                    ):
                        stats.skipped_seller += 1
                        state.mark_processed(
                            listing,
                            signature,
                            "seller could not be verified",
                            fingerprint=discovery_fingerprint,
                        )
                        continue

                    images = await download_listing_images(
                        listing.image_urls,
                        maximum=int(detailed_limits["max_images_downloaded"]),
                    )
                    if not images:
                        state.mark_processed(
                            listing,
                            signature,
                            "no listing images",
                            fingerprint=discovery_fingerprint,
                        )
                        continue

                    outcomes: list[str] = []
                    listing_detailed = False
                    target_ids = _candidate_targets(
                        listing,
                        references,
                        compare_all=compare_all,
                    )
                    min_detail_score = float(
                        matching["minimum_screening_confidence_for_detail"]
                    )
                    probable_threshold = float(matching["probable_alert_threshold"])

                    best_screen_by_target: dict[str, VisualMatch | None] = {
                        target_id: None for target_id in target_ids
                    }
                    probable_alerted: set[str] = set()
                    remaining_targets = list(target_ids)
                    stats.listings_screened += 1

                    for batch_number, batch in enumerate(
                        _batches(
                            images,
                            int(screening_limits["images_per_batch"]),
                            int(screening_limits["max_batches_per_listing"]),
                        ),
                        start=1,
                    ):
                        if not remaining_targets:
                            break
                        overview = make_contact_sheet(
                            batch,
                            prefix=f"O{batch_number}-",
                            max_dimension=int(
                                screening_limits["max_image_dimension_px"]
                            ),
                            quality=int(screening_limits["jpeg_quality"]),
                            columns=2,
                        )
                        strip_targets = [
                            (target_id, references[target_id].display_name)
                            for target_id in remaining_targets
                        ]
                        reference_strip = make_contact_sheet(
                            [
                                reference_thumbnails[target_id]
                                for target_id, _ in strip_targets
                            ],
                            prefix="R",
                            max_dimension=int(
                                screening_limits["max_image_dimension_px"]
                            ),
                            quality=int(screening_limits["jpeg_quality"]),
                            columns=min(4, max(1, len(strip_targets))),
                        )
                        screen_results = await matcher.screen_multi(
                            targets=strip_targets,
                            reference_strip_jpeg=reference_strip.jpeg,
                            candidate_jpeg=overview.jpeg,
                        )
                        for target_id in list(remaining_targets):
                            screen = screen_results.get(target_id)
                            if screen is None:
                                continue
                            _report_match(
                                report_rows,
                                listing,
                                screen,
                                stage="screening",
                                batch_number=batch_number,
                            )
                            current_best = best_screen_by_target[target_id]
                            if (
                                current_best is None
                                or screen.match_score > current_best.match_score
                            ):
                                best_screen_by_target[target_id] = screen

                            if (
                                screen.match_score >= probable_threshold
                                and target_id not in probable_alerted
                            ):
                                stats.probable_matches += 1
                                probable_alerted.add(target_id)
                                fingerprint = _alert_fingerprint(
                                    listing, target_id, "probable"
                                )
                                if (
                                    bool(matching["alert_on_probable_match"])
                                    and not state.alert_sent(
                                        listing.code,
                                        target_id,
                                        "probable",
                                        fingerprint,
                                    )
                                ):
                                    message_id = notifier.probable(
                                        listing, references[target_id], screen
                                    )
                                    if message_id:
                                        stats.alerts_sent += 1
                                        state.record_alert(
                                            listing.code,
                                            target_id,
                                            "probable",
                                            fingerprint,
                                        )
                                        if confirmation_tracking_enabled:
                                            confirmation_store.add_pending(
                                                PendingConfirmation(
                                                    message_id=message_id,
                                                    listing_code=listing.code,
                                                    listing_url=listing.url,
                                                    target_id=target_id,
                                                    card_name=references[
                                                        target_id
                                                    ].display_name,
                                                    alert_type="probable",
                                                    sent_at=datetime.now(
                                                        timezone.utc
                                                    ).isoformat(),
                                                    listing_snapshot=(
                                                        _listing_snapshot(
                                                            listing, screen
                                                        )
                                                    ),
                                                )
                                            )

                            # A sufficiently strong batch resolves this target immediately.
                            if screen.match_score >= min_detail_score:
                                remaining_targets.remove(target_id)

                    for target_id in target_ids:
                        reference = references[target_id]
                        reference_jpeg = reference_jpegs[target_id]
                        stats.screened += 1
                        best_screen = best_screen_by_target[target_id]
                        if (
                            best_screen is None
                            or best_screen.match_score < min_detail_score
                        ):
                            match_score = best_screen.match_score if best_screen else 0.0
                            outcomes.append(
                                f"{target_id}: screen negative {match_score:.2f}"
                            )
                            continue

                        if (
                            not listing_detailed
                            and detailed_limit > 0
                            and stats.detailed >= detailed_limit
                        ):
                            stats.held += 1
                            outcomes.append(
                                f"{target_id}: held for detailed-listing cap"
                            )
                            continue
                        if not listing_detailed:
                            stats.detailed += 1
                            listing_detailed = True

                        # extract_card_crops is CPU-bound OpenCV work (contour
                        # detection, perspective warp); run it off the event
                        # loop so it doesn't stall concurrent hydration prefetch.
                        crops = await asyncio.to_thread(
                            extract_card_crops,
                            images,
                            maximum=int(
                                detailed_limits["max_card_crops_per_listing"]
                            ),
                        )
                        # Include originals as a fallback even when OpenCV found crops.
                        # This prevents a missed crop from becoming a false negative.
                        detail_images = [*images, *crops]
                        # Tracks which entries are genuine OpenCV crops vs whole
                        # listing photos, so the cross-check below can tell
                        # whether it's actually zooming in or just re-showing
                        # the model the same whole photo it already judged.
                        is_crop = [False] * len(images) + [True] * len(crops)
                        best_detail: VisualMatch | None = None
                        confirmed_detail: VisualMatch | None = None
                        for batch_number, tagged_batch in enumerate(
                            _batches(
                                list(zip(detail_images, is_crop)),
                                int(detailed_limits["images_per_batch"]),
                                int(detailed_limits["max_batches_per_listing"]),
                            ),
                            start=1,
                        ):
                            batch = [image for image, _ in tagged_batch]
                            batch_is_crop = [flag for _, flag in tagged_batch]
                            batch_prefix = ("C" if crops else "O") + f"{batch_number}-"
                            detail_sheet = make_contact_sheet(
                                batch,
                                prefix=batch_prefix,
                                max_dimension=int(
                                    detailed_limits["max_image_dimension_px"]
                                ),
                                quality=int(detailed_limits["jpeg_quality"]),
                                columns=int(detailed_limits["contact_sheet_columns"]),
                            )
                            detail = await matcher.compare(
                                target_id=target_id,
                                reference_name=reference.display_name,
                                reference_jpeg=reference_jpeg,
                                candidate_jpeg=detail_sheet.jpeg,
                                stage="detailed",
                            )
                            _report_match(
                                report_rows,
                                listing,
                                detail,
                                stage="detailed",
                                batch_number=batch_number,
                            )
                            if (
                                best_detail is None
                                or detail.match_score > best_detail.match_score
                            ):
                                best_detail = detail
                            if detail.match_score >= float(
                                matching["confirmed_threshold"]
                            ):
                                # A whole-grid judgement can hallucinate a
                                # match; re-check the specific claimed card in
                                # isolation before trusting a confirmation.
                                # Only a genuine crop counts as a materially
                                # closer look (observed live: a whole-lot
                                # photo with no extractable crops "confirmed"
                                # itself twice over by re-examining the
                                # identical image it had already judged).
                                candidate_image = _resolve_cropped_candidate(
                                    batch,
                                    batch_is_crop,
                                    batch_prefix,
                                    detail.candidate_labels,
                                )
                                if candidate_image is not None:
                                    zoom_sheet = make_contact_sheet(
                                        [candidate_image],
                                        prefix="Z1-",
                                        max_dimension=int(
                                            detailed_limits["max_image_dimension_px"]
                                        ),
                                        quality=int(detailed_limits["jpeg_quality"]),
                                        columns=1,
                                    )
                                    verified = await matcher.compare(
                                        target_id=target_id,
                                        reference_name=reference.display_name,
                                        reference_jpeg=reference_jpeg,
                                        candidate_jpeg=zoom_sheet.jpeg,
                                        stage="detailed",
                                    )
                                    _report_match(
                                        report_rows,
                                        listing,
                                        verified,
                                        stage="detailed_cross_check",
                                        batch_number=batch_number,
                                    )
                                    if verified.match_score >= float(
                                        matching["confirmed_threshold"]
                                    ):
                                        confirmed_detail = verified
                                        break

                        if confirmed_detail is not None:
                            stats.confirmed_matches += 1
                            fingerprint = _alert_fingerprint(
                                listing, target_id, "confirmed"
                            )
                            if (
                                bool(matching["alert_on_confirmed_match"])
                                and not state.alert_sent(
                                    listing.code,
                                    target_id,
                                    "confirmed",
                                    fingerprint,
                                )
                            ):
                                fee_yen = int(config.raw["sendico_fee"]["yen"])

                                lot_cards: list[LotCard] = []
                                visible_card_count = 0
                                try:
                                    lot_cards, visible_card_count = (
                                        await _catalogue_lot(
                                            images,
                                            crops,
                                            matcher,
                                            price_client,
                                            detailed_limits,
                                        )
                                    )
                                except GeminiBudgetReached:
                                    raise
                                except Exception as exc:
                                    # Lot cataloguing is enrichment, not the
                                    # confirmation itself -- never let it cost
                                    # the whole alert.
                                    LOGGER.warning(
                                        "Lot cataloguing failed for %s: %s",
                                        listing.code,
                                        exc,
                                    )
                                    lot_cards = []
                                    visible_card_count = 0

                                valuation = lot_value(
                                    lot_cards,
                                    visible_card_count=visible_card_count,
                                    price_match_threshold=float(
                                        lot_valuation_config.get(
                                            "price_match_threshold", 0.95
                                        )
                                    ),
                                )

                                message_id, alert_embed = notifier.confirmed(
                                    listing,
                                    reference,
                                    confirmed_detail,
                                    valuation=valuation,
                                    fx_rates=fx_rates,
                                    fee_yen=fee_yen,
                                    seller_criteria=seller_criteria,
                                    confirmed_threshold=float(
                                        matching["confirmed_threshold"]
                                    ),
                                )
                                if message_id:
                                    stats.alerts_sent += 1
                                    state.record_alert(
                                        listing.code,
                                        target_id,
                                        "confirmed",
                                        fingerprint,
                                    )
                                    if confirmation_tracking_enabled:
                                        confirmation_store.add_pending(
                                            PendingConfirmation(
                                                message_id=message_id,
                                                listing_code=listing.code,
                                                listing_url=listing.url,
                                                target_id=target_id,
                                                card_name=reference.display_name,
                                                alert_type="confirmed",
                                                sent_at=datetime.now(
                                                    timezone.utc
                                                ).isoformat(),
                                                embed=alert_embed,
                                            )
                                        )
                            outcomes.append(
                                f"{target_id}: confirmed "
                                f"{confirmed_detail.match_score:.2f}"
                            )
                        else:
                            match_score = best_detail.match_score if best_detail else 0.0
                            outcomes.append(
                                f"{target_id}: not confirmed {match_score:.2f}"
                            )

                    state.mark_processed(
                        listing,
                        signature,
                        "; ".join(outcomes) or "no target assessed",
                        fingerprint=discovery_fingerprint,
                    )
                except GeminiBudgetReached as exc:
                    LOGGER.warning("Stopping at Gemini budget: %s", exc)
                    stats.held += len(candidates) - index
                    status = f"Stopped early: {exc}"
                    break
                except Exception as exc:
                    stats.processing_errors += 1
                    LOGGER.exception(
                        "Listing processing failed for %s: %s",
                        listing.code,
                        exc,
                    )
                    # Leave processing errors eligible for the next scheduled run.
                    state.save()

            for pending_task in hydration_tasks.values():
                pending_task.cancel()
            if hydration_tasks:
                await asyncio.gather(*hydration_tasks.values(), return_exceptions=True)
    except Exception as exc:
        status = f"Failed: {exc}"
        raise
    finally:
        state.save()
        confirmation_store.save()
        stats.requests_sent = matcher.requests_sent
        stats.input_tokens = matcher.input_tokens
        stats.output_tokens = matcher.output_tokens
        stats.thinking_tokens = matcher.thinking_tokens
        stats.total_tokens = matcher.total_tokens
        stats.models_used = matcher.model_usage
        await matcher.close()
        price_client.close()
        write_report(config.root, stats, report_rows)
        if bool(
            config.raw.get("discord", {}).get(
                "send_completion_summary", True
            )
        ):
            try:
                notifier.completion(stats, status=status)
            except Exception as exc:
                LOGGER.error("Could not send completion summary: %s", exc)
        notifier.close()
        confirmed_notifier.close()
    return 0


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Scan Sendico using PriceCharting reference images"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-confirmations",
        action="store_true",
        help=(
            "Only check pending alert reactions and resolve them; skip the "
            "full scan."
        ),
    )
    args = parser.parse_args()
    if args.check_confirmations:
        raise SystemExit(asyncio.run(check_confirmations(args.config)))
    raise SystemExit(asyncio.run(run(args.config, args.dry_run)))


if __name__ == "__main__":
    cli()
