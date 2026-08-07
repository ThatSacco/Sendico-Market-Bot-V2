"""The bot must work for any card PriceCharting carries, not just Pokemon.

The watchlist's Pokemon entries are test data; nothing in the pipeline
should assume a game.
"""

from pathlib import Path

import pytest

from pokemon_deal_bot.main import _candidate_targets, _shared_game_label
from pokemon_deal_bot.models import ReferenceCard, SendicoListing
from pokemon_deal_bot.reference import parse_pricecharting_page

# Real page titles, captured live 2026-08-06.
TITLES = {
    "pokemon_plain": (
        "Victini #97 Prices | Pokemon Japanese Black Bolt | Pokemon Cards",
        ("Victini", "97", "Pokemon Japanese Black Bolt", "Pokemon Cards"),
    ),
    "pokemon_fraction": (
        "Charizard #4/102 Prices | Pokemon Base Set | Pokemon Cards",
        ("Charizard", "4/102", "Pokemon Base Set", "Pokemon Cards"),
    ),
    "one_piece_set_prefixed": (
        "Monkey.D.Luffy [Magazine Promo] ST21-014 Prices | "
        "One Piece Japanese Starter Deck 21: Gear5 | One Piece Cards",
        (
            "Monkey.D.Luffy [Magazine Promo]",
            "ST21-014",
            "One Piece Japanese Starter Deck 21: Gear5",
            "One Piece Cards",
        ),
    ),
    "one_piece_promo": (
        "Monkey.D.Luffy [One Piece Day] P-110 Prices | "
        "One Piece Japanese Promo | One Piece Cards",
        (
            "Monkey.D.Luffy [One Piece Day]",
            "P-110",
            "One Piece Japanese Promo",
            "One Piece Cards",
        ),
    ),
    "yugioh": (
        "Blue-Eyes White Dragon LOB-001 Prices | "
        "Yu-Gi-Oh Legend of Blue Eyes | Yu-Gi-Oh Cards",
        (
            "Blue-Eyes White Dragon",
            "LOB-001",
            "Yu-Gi-Oh Legend of Blue Eyes",
            "Yu-Gi-Oh Cards",
        ),
    ),
}


def _page(title: str) -> str:
    return f"""
    <html><head><title>{title}</title>
    <meta property="og:image" content="https://storage.googleapis.com/example/card.jpg">
    </head><body>
    <table><tr><td>Ungraded</td><td>$12.34</td></tr></table>
    <div>PriceCharting ID: 1234567</div>
    </body></html>
    """


@pytest.mark.parametrize("key", list(TITLES))
def test_card_numbers_parse_across_games(key):
    title, (name, number, set_name, game) = TITLES[key]
    parsed = parse_pricecharting_page(
        "https://www.pricecharting.com/game/x/y", _page(title)
    )
    assert parsed["name"] == name
    # Hardcoding one game's numbering (digits only) made every other game's
    # cards fail to resolve at all.
    assert parsed["card_number"] == number
    assert parsed["set_name"] == set_name
    assert parsed["game"] == game


def _reference(target_id: str, game: str) -> ReferenceCard:
    return ReferenceCard(
        target_id=target_id,
        source_url="https://www.pricecharting.com/game/x/y",
        product_id=target_id,
        name=target_id,
        set_name="",
        card_number="",
        image_url="https://example.test/i.jpg",
        image_path=Path("i.jpg"),
        game=game,
    )


def test_game_label_strips_the_cards_suffix():
    assert _reference("a", "One Piece Cards").game_label == "One Piece"
    assert _reference("a", "Pokemon Cards").game_label == "Pokemon"
    # Never leave a prompt reading "a  trading card" with a blank game.
    assert _reference("a", "").game_label == "trading"


def _listing(*target_ids: str) -> SendicoListing:
    return SendicoListing(
        code="m1",
        url="https://sendico.test/m1",
        title="lot",
        price_yen=1000,
        candidate_target_ids=list(target_ids),
    )


def test_compare_all_does_not_cross_games():
    references = {
        "victini": _reference("victini", "Pokemon Cards"),
        "ampharos": _reference("ampharos", "Pokemon Cards"),
        "luffy": _reference("luffy", "One Piece Cards"),
    }
    # Found by a One Piece search: it can never contain a Pokemon card, so
    # comparing against Pokemon references is a guaranteed miss that still
    # costs a Gemini call.
    assert _candidate_targets(
        _listing("luffy"), references, compare_all=True
    ) == ["luffy"]

    # Found by a Pokemon search: still broadens across Pokemon cards, which
    # is the behaviour compare_all exists for.
    assert _candidate_targets(
        _listing("victini"), references, compare_all=True
    ) == ["victini", "ampharos"]


def test_unknown_game_is_still_compared():
    # An older cache entry has no game recorded. A wasted comparison is
    # cheaper than silently never matching that card again.
    references = {
        "victini": _reference("victini", "Pokemon Cards"),
        "legacy": _reference("legacy", ""),
    }
    assert _candidate_targets(
        _listing("victini"), references, compare_all=True
    ) == ["victini", "legacy"]


def test_compare_all_false_is_unaffected_by_game():
    references = {
        "victini": _reference("victini", "Pokemon Cards"),
        "luffy": _reference("luffy", "One Piece Cards"),
    }
    assert _candidate_targets(
        _listing("luffy"), references, compare_all=False
    ) == ["luffy"]


def test_shared_game_label_for_prompts():
    references = {
        "victini": _reference("victini", "Pokemon Cards"),
        "ampharos": _reference("ampharos", "Pokemon Cards"),
        "luffy": _reference("luffy", "One Piece Cards"),
    }
    assert _shared_game_label(references, ["victini", "ampharos"]) == "Pokemon"
    assert _shared_game_label(references, ["luffy"]) == "One Piece"
    # Mixed batches must not tell the model to look for the wrong game.
    assert _shared_game_label(references, ["victini", "luffy"]) == "trading"
    assert _shared_game_label(references, []) == "trading"
