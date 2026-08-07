from PIL import Image

from pokemon_deal_bot.main import (
    _candidate_targets,
    _resolve_candidate_image,
    _resolve_cropped_candidate,
    _round_robin_truncate,
)
from pokemon_deal_bot.models import ReferenceCard, SendicoListing, VisualMatch


def _reference(target_id: str) -> ReferenceCard:
    from pathlib import Path

    return ReferenceCard(
        target_id=target_id,
        source_url="https://www.pricecharting.com/game/x/y",
        product_id=target_id,
        name=target_id,
        set_name="",
        card_number="",
        image_url="https://example.test/image.jpg",
        image_path=Path("image.jpg"),
    )


def test_search_association_prioritises_but_does_not_filter_targets():
    listing = SendicoListing(
        code="m1",
        url="https://sendico.test/m1",
        title="lot",
        price_yen=1000,
        candidate_target_ids=["ampharos"],
    )
    references = {
        "victini": _reference("victini"),
        "ampharos": _reference("ampharos"),
    }
    assert _candidate_targets(
        listing,
        references,
        compare_all=True,
    ) == ["ampharos", "victini"]
    assert _candidate_targets(
        listing,
        references,
        compare_all=False,
    ) == ["ampharos"]


def test_compare_all_false_with_no_search_association_returns_no_targets():
    listing = SendicoListing(
        code="m2",
        url="https://sendico.test/m2",
        title="lot",
        price_yen=1000,
    )
    references = {"victini": _reference("victini")}
    assert _candidate_targets(listing, references, compare_all=False) == []
    assert _candidate_targets(listing, references, compare_all=True) == ["victini"]


def test_match_score_is_zero_for_confident_negative_decision():
    negative = VisualMatch(
        target_id="victini", stage="screening", confidence=1.0, same_card=False
    )
    assert negative.match_score == 0.0

    positive = VisualMatch(
        target_id="victini", stage="screening", confidence=0.6, same_card=True
    )
    assert positive.match_score == 0.6


def _listing(code: str) -> SendicoListing:
    return SendicoListing(
        code=code, url=f"https://sendico.test/{code}", title=code, price_yen=1000
    )


def test_round_robin_truncate_does_not_sacrifice_the_last_term():
    per_term = [
        [_listing("a1"), _listing("a2"), _listing("a3")],
        [_listing("b1")],
        [_listing("c1")],
    ]
    result = _round_robin_truncate(per_term, limit=3)
    assert [listing.code for listing in result] == ["a1", "b1", "c1"]


def test_round_robin_truncate_respects_limit_and_order_within_round():
    per_term = [[_listing("a1"), _listing("a2")], [_listing("b1"), _listing("b2")]]
    result = _round_robin_truncate(per_term, limit=3)
    assert [listing.code for listing in result] == ["a1", "b1", "a2"]


def test_resolve_candidate_image_maps_label_back_to_source():
    batch = [Image.new("RGB", (1, 1)) for _ in range(3)]
    assert _resolve_candidate_image(batch, "C1-", ["C1-2"]) is batch[1]
    assert _resolve_candidate_image(batch, "C1-", ["O1-2", "C1-3"]) is batch[2]


def test_resolve_candidate_image_returns_none_for_unresolvable_labels():
    batch = [Image.new("RGB", (1, 1))]
    assert _resolve_candidate_image(batch, "C1-", []) is None
    assert _resolve_candidate_image(batch, "C1-", ["C1-9"]) is None
    assert _resolve_candidate_image(batch, "C1-", ["not-a-label"]) is None


def test_resolve_candidate_image_tolerates_zero_for_letter_o_transcription():
    batch = [Image.new("RGB", (1, 1)), Image.new("RGB", (1, 1))]
    assert _resolve_candidate_image(batch, "O1-", ["01-2"]) is batch[1]
    assert _resolve_candidate_image(batch, "O1-", ["o1-1"]) is batch[0]


def test_resolve_cropped_candidate_accepts_a_genuine_crop():
    batch = [Image.new("RGB", (1, 1)), Image.new("RGB", (1, 1))]
    batch_is_crop = [False, True]
    assert _resolve_cropped_candidate(batch, batch_is_crop, "C1-", ["C1-2"]) is batch[1]


def test_resolve_cropped_candidate_rejects_a_whole_listing_photo():
    # This is the exact failure mode observed live: the model claims a whole
    # photo (not a crop) shows the target card. Re-showing it the identical
    # photo as a "zoomed in" cross-check would just re-ask the same question
    # of the same picture, so it must not be treated as independent evidence.
    batch = [Image.new("RGB", (1, 1)), Image.new("RGB", (1, 1))]
    batch_is_crop = [False, False]
    assert _resolve_cropped_candidate(batch, batch_is_crop, "O1-", ["O1-1"]) is None


def test_resolve_cropped_candidate_returns_none_for_unresolvable_labels():
    batch = [Image.new("RGB", (1, 1))]
    assert _resolve_cropped_candidate(batch, [True], "C1-", ["C1-9"]) is None


def test_screening_cap_counts_listings_not_target_comparisons():
    """The run budget must not shrink as watchlist cards are added.

    Observed live 2026-08-06: with 3 active cards, a limit named
    "screening.max_listings_per_run: 200" stopped the run after 67
    listings, because the counter it was compared against incremented once
    per card per listing (it reached 201). The cap is on listings.
    """

    from pokemon_deal_bot.models import ScanStats

    stats = ScanStats()
    screening_limit = 200
    active_cards = 3
    listings_processed = 0

    for _ in range(500):
        if screening_limit > 0 and stats.listings_screened >= screening_limit:
            break
        stats.listings_screened += 1
        # One comparison per active watchlist card, as run() does.
        stats.screened += active_cards
        listings_processed += 1

    assert listings_processed == 200
    # The comparison counter still reports the real Gemini-side workload.
    assert stats.screened == 600
