from pokemon_deal_bot.main import _candidate_targets, _round_robin_truncate
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
