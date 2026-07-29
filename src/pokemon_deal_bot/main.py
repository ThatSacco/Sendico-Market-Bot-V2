from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
from collections.abc import Iterable

from PIL import Image

from .config import build_search_plan, load_config, scan_signature
from .discord import DiscordNotifier
from .gemini import GeminiBudgetReached, GeminiReferenceMatcher
from .image_processing import (
    download_listing_images,
    extract_card_crops,
    image_file_bytes,
    make_contact_sheet,
)
from .models import ReferenceCard, ScanStats, SendicoListing, VisualMatch
from .reference import PriceChartingReferenceClient
from .reporting import write_report
from .sendico import SendicoScanner
from .state import StateStore

LOGGER = logging.getLogger(__name__)


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


async def run(config_path: str = "config.yaml", dry_run: bool = False) -> int:
    config = load_config(config_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not config.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    references_client = PriceChartingReferenceClient(
        config.root,
        cache_hours=int(
            config.raw.get("pricing", {}).get("reference_cache_hours", 24)
        ),
    )
    references: dict[str, ReferenceCard] = {}
    try:
        for target in config.targets:
            try:
                references[target.id] = references_client.resolve(target)
                LOGGER.info(
                    "Loaded reference: %s", references[target.id].display_name
                )
            except Exception as exc:
                LOGGER.error(
                    "Could not load PriceCharting reference for %s: %s",
                    target.id,
                    exc,
                )
    finally:
        references_client.close()
    if not references:
        raise RuntimeError("No PriceCharting references could be loaded")

    matcher_config = dict(config.raw.get("gemini") or {})
    matcher_config["screening_model"] = str(
        config.raw.get("gemini", {}).get("screening_model")
        or "gemini-3.5-flash-lite"
    )
    matcher = GeminiReferenceMatcher(
        config.gemini_api_key,
        matcher_config,
        config.run_limits,
    )
    discord_enabled = bool(config.raw.get("discord", {}).get("enabled", True))
    notifier = DiscordNotifier(
        config.discord_webhook_url if discord_enabled and not dry_run else None,
        str(
            config.raw.get("discord", {}).get("username")
            or "Pokemon Deal Scout"
        ),
    )
    state = StateStore(
        config.path("data/seen.json"),
        max_listings=int(config.run_limits["state"]["max_seen_listings"]),
    )
    signature = scan_signature(config)
    stats = ScanStats()
    report_rows: list[dict] = []

    matching = config.criteria["reference_image_matching"]
    seller_criteria = config.criteria["seller"]
    search_limits = config.run_limits["search"]
    screening_limits = config.run_limits["screening"]
    detailed_limits = config.run_limits["detailed_analysis"]
    search_limit = int(search_limits["total_listings_per_run"])
    screening_limit = int(screening_limits["max_listings_per_run"])
    detailed_limit = int(detailed_limits["max_listings_per_run"])
    compare_all = bool(matching.get("compare_all_active_targets", True))

    listings_by_code: dict[str, SendicoListing] = {}
    plan = build_search_plan(config.targets)
    try:
        async with SendicoScanner(
            config.raw["sendico"], config.run_limits
        ) as scanner:
            for task in plan:
                results = await scanner.search(task.term)
                stats.found += len(results)
                for listing in results:
                    current = listings_by_code.get(listing.code)
                    if current is None:
                        listing.matched_search_terms.append(task.term)
                        listing.candidate_target_ids.extend(task.target_ids)
                        listings_by_code[listing.code] = listing
                    else:
                        current.matched_search_terms.append(task.term)
                        current.candidate_target_ids.extend(task.target_ids)

            candidates = list(listings_by_code.values())[:search_limit]
            stats.candidates = len(candidates)
            if len(listings_by_code) > len(candidates):
                stats.held += len(listings_by_code) - len(candidates)

            for index, listing in enumerate(candidates):
                discovery_fingerprint = state.listing_fingerprint(listing, signature)
                if not state.should_process_fingerprint(
                    listing.code, discovery_fingerprint
                ):
                    stats.skipped_seen += 1
                    continue
                if screening_limit > 0 and stats.screened >= screening_limit:
                    stats.held += len(candidates) - index
                    break

                try:
                    listing = await scanner.hydrate(listing)
                    stats.hydrated += 1
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
                    for target_id in target_ids:
                        reference = references[target_id]
                        reference_jpeg = image_file_bytes(reference.image_path)
                        best_screen: VisualMatch | None = None
                        probable_alerted = False

                        for batch_number, batch in enumerate(
                            _batches(
                                images,
                                int(screening_limits["images_per_batch"]),
                                int(screening_limits["max_batches_per_listing"]),
                            ),
                            start=1,
                        ):
                            overview = make_contact_sheet(
                                batch,
                                prefix=f"O{batch_number}-",
                                max_dimension=int(
                                    screening_limits["max_image_dimension_px"]
                                ),
                                quality=int(screening_limits["jpeg_quality"]),
                                columns=2,
                            )
                            screen = matcher.compare(
                                target_id=target_id,
                                reference_name=reference.display_name,
                                reference_jpeg=reference_jpeg,
                                candidate_jpeg=overview.jpeg,
                                stage="screening",
                            )
                            _report_match(
                                report_rows,
                                listing,
                                screen,
                                stage="screening",
                                batch_number=batch_number,
                            )
                            if (
                                best_screen is None
                                or screen.match_score > best_screen.match_score
                            ):
                                best_screen = screen

                            if (
                                screen.match_score
                                >= float(matching["probable_alert_threshold"])
                                and not probable_alerted
                            ):
                                stats.probable_matches += 1
                                probable_alerted = True
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
                                    if notifier.probable(
                                        listing, reference, screen
                                    ):
                                        stats.alerts_sent += 1
                                        state.record_alert(
                                            listing.code,
                                            target_id,
                                            "probable",
                                            fingerprint,
                                        )

                            # A sufficiently strong batch can proceed immediately.
                            if screen.match_score >= float(
                                matching["minimum_screening_confidence_for_detail"]
                            ):
                                break

                        stats.screened += 1
                        if (
                            best_screen is None
                            or best_screen.match_score
                            < float(matching["minimum_screening_confidence_for_detail"])
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

                        crops = extract_card_crops(
                            images,
                            maximum=int(
                                detailed_limits["max_card_crops_per_listing"]
                            ),
                        )
                        # Include originals as a fallback even when OpenCV found crops.
                        # This prevents a missed crop from becoming a false negative.
                        detail_images = [*images, *crops]
                        best_detail: VisualMatch | None = None
                        confirmed_detail: VisualMatch | None = None
                        for batch_number, batch in enumerate(
                            _batches(
                                detail_images,
                                int(detailed_limits["images_per_batch"]),
                                int(detailed_limits["max_batches_per_listing"]),
                            ),
                            start=1,
                        ):
                            detail_sheet = make_contact_sheet(
                                batch,
                                prefix=("C" if crops else "O")
                                + f"{batch_number}-",
                                max_dimension=int(
                                    detailed_limits["max_image_dimension_px"]
                                ),
                                quality=int(detailed_limits["jpeg_quality"]),
                                columns=int(detailed_limits["contact_sheet_columns"]),
                            )
                            detail = matcher.compare(
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
                                confirmed_detail = detail
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
                                pricing = config.raw["pricing"]
                                fee_yen = int(config.raw["sendico_fee"]["yen"])
                                if notifier.confirmed(
                                    listing,
                                    reference,
                                    confirmed_detail,
                                    jpy_to_aud=float(
                                        pricing["manual_jpy_to_aud"]
                                    ),
                                    usd_to_aud=float(
                                        pricing["manual_usd_to_aud"]
                                    ),
                                    fee_yen=fee_yen,
                                ):
                                    stats.alerts_sent += 1
                                    state.record_alert(
                                        listing.code,
                                        target_id,
                                        "confirmed",
                                        fingerprint,
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
                    break
                except Exception as exc:
                    stats.processing_errors += 1
                    LOGGER.exception(
                        "Listing processing failed for %s: %s",
                        listing.code,
                        exc,
                    )
                    # Leave processing errors eligible for the next scheduled run.
    finally:
        stats.requests_sent = matcher.requests_sent
        stats.input_tokens = matcher.input_tokens
        stats.output_tokens = matcher.output_tokens
        stats.thinking_tokens = matcher.thinking_tokens
        stats.total_tokens = matcher.total_tokens
        stats.models_used = matcher.model_usage
        matcher.close()
        write_report(config.root, stats, report_rows)
        if bool(
            config.raw.get("discord", {}).get(
                "send_completion_summary", True
            )
        ):
            try:
                notifier.completion(stats)
            except Exception as exc:
                LOGGER.error("Could not send completion summary: %s", exc)
        notifier.close()
    return 0


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Scan Sendico using PriceCharting reference images"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.config, args.dry_run)))


if __name__ == "__main__":
    cli()
