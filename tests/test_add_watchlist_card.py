import httpx
import pytest

from add_watchlist_card import (
    add_card,
    build_card_block,
    parse_search_terms,
    resolve_and_summarize,
    resolve_id,
)
from pokemon_deal_bot.config import load_watchlist

AMPHAROS_HTML = """
<html><head>
<title>Ampharos EX #27 Prices | Pokemon Japanese Bandit Ring | Pokemon Cards</title>
<meta property="og:image" content="https://storage.googleapis.com/example/ampharos.jpg">
</head><body>
<h1>Ampharos EX #27</h1>
<table><tr><td>Ungraded</td><td>$6.97</td></tr></table>
<div>PriceCharting ID: 3462297</div>
</body></html>
"""

HEADER = """# Add a card using only:
# 1. an id;
# 2. its PriceCharting product link; and
# 3. the Sendico/Mercari search phrases used to discover listings.
cards:
  - id: victini_black_bolt_97
    active: true
    pricecharting_url: "https://www.pricecharting.com/game/pokemon-japanese-black-bolt/victini-97"
    searches:
      - term: "ブラックボルト まとめ売り"
        active: true
"""


def _watchlist(tmp_path, content: str = HEADER):
    path = tmp_path / "watchlist.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_search_terms_splits_on_newlines_and_trims():
    assert parse_search_terms("term one\nterm two\n\n  term three  \n") == [
        "term one",
        "term two",
        "term three",
    ]


def test_parse_search_terms_tolerates_commas_and_dedupes():
    assert parse_search_terms("a, b,a\nb") == ["a", "b"]


def test_resolve_id_derives_from_url_when_no_custom_id():
    assert (
        resolve_id("https://www.pricecharting.com/game/pokemon-japanese-x/ampharos-ex-27", "")
        == "ampharos_ex_27"
    )


def test_resolve_id_sanitizes_a_custom_id():
    assert resolve_id("https://www.pricecharting.com/game/x/y", "  My Card! #1  ") == "my_card_1"


def test_resolve_id_rejects_a_custom_id_with_no_usable_characters():
    with pytest.raises(ValueError, match="no usable characters"):
        resolve_id("https://www.pricecharting.com/game/x/y", "###")


def test_build_card_block_escapes_japanese_text_and_quotes():
    block = build_card_block(
        card_id="test_card",
        pricecharting_url="https://www.pricecharting.com/game/x/y",
        terms=['ビクティニ "プロモ"', "second term"],
    )
    assert block == (
        "  - id: test_card\n"
        "    active: true\n"
        '    pricecharting_url: "https://www.pricecharting.com/game/x/y"\n'
        "    searches:\n"
        '      - term: "ビクティニ \\"プロモ\\""\n'
        "        active: true\n"
        '      - term: "second term"\n'
        "        active: true\n"
    )


def test_add_card_appends_and_preserves_existing_content(tmp_path):
    path = _watchlist(tmp_path)
    original = path.read_text(encoding="utf-8")

    card_id = add_card(
        path,
        pricecharting_url="https://www.pricecharting.com/game/pokemon-japanese-bandit-ring/ampharos-ex-27",
        search_terms_raw="XY7 まとめ売り\nバンデットリング まとめ売り",
    )

    assert card_id == "ampharos_ex_27"
    new_content = path.read_text(encoding="utf-8")
    assert new_content.startswith(original)

    targets = load_watchlist(path)
    ids = [target.id for target in targets]
    assert ids == ["victini_black_bolt_97", "ampharos_ex_27"]
    new_target = targets[1]
    assert new_target.active is True
    assert [search.term for search in new_target.searches] == [
        "XY7 まとめ売り",
        "バンデットリング まとめ売り",
    ]


def test_add_card_accepts_a_custom_id(tmp_path):
    path = _watchlist(tmp_path)
    card_id = add_card(
        path,
        pricecharting_url="https://www.pricecharting.com/game/pokemon-japanese-bandit-ring/ampharos-ex-27",
        search_terms_raw="XY7 まとめ売り",
        custom_id="my-ampharos",
    )
    assert card_id == "my_ampharos"
    assert load_watchlist(path)[1].id == "my_ampharos"


def test_add_card_rejects_non_pricecharting_url_without_modifying_file(tmp_path):
    path = _watchlist(tmp_path)
    original = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="PriceCharting"):
        add_card(path, pricecharting_url="https://example.com/card", search_terms_raw="term")

    assert path.read_text(encoding="utf-8") == original


def test_add_card_rejects_empty_search_terms(tmp_path):
    path = _watchlist(tmp_path)
    original = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="search term"):
        add_card(
            path,
            pricecharting_url="https://www.pricecharting.com/game/x/y",
            search_terms_raw="   \n  ",
        )

    assert path.read_text(encoding="utf-8") == original


def test_add_card_rejects_duplicate_id(tmp_path):
    path = _watchlist(tmp_path)

    with pytest.raises(ValueError, match="already exists"):
        add_card(
            path,
            pricecharting_url="https://www.pricecharting.com/game/pokemon-japanese-bandit-ring/ampharos-ex-27",
            search_terms_raw="term",
            custom_id="victini_black_bolt_97",
        )


def test_add_card_rejects_duplicate_url(tmp_path):
    path = _watchlist(tmp_path)

    with pytest.raises(ValueError, match="already on the watchlist"):
        add_card(
            path,
            pricecharting_url="https://www.pricecharting.com/game/pokemon-japanese-black-bolt/victini-97",
            search_terms_raw="term",
            custom_id="a_different_id",
        )


def test_resolve_and_summarize_fetches_and_reports_the_new_card(tmp_path):
    path = _watchlist(tmp_path)
    add_card(
        path,
        pricecharting_url="https://www.pricecharting.com/game/pokemon-japanese-bandit-ring/ampharos-ex-27",
        search_terms_raw="XY7 まとめ売り\nバンデットリング まとめ売り",
        custom_id="ampharos_bandit_ring_27",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.pricecharting.com":
            return httpx.Response(200, text=AMPHAROS_HTML)
        return httpx.Response(200, content=b"fake-jpeg", headers={"content-type": "image/jpeg"})

    summary = resolve_and_summarize(
        "ampharos_bandit_ring_27",
        path,
        tmp_path,
        transport=httpx.MockTransport(handler),
    )

    assert "Ampharos EX" in summary
    assert "$6.97" in summary
    assert "XY7 まとめ売り" in summary
    assert (tmp_path / "data/reference_cache.json").exists()


def test_resolve_and_summarize_raises_for_a_card_not_on_the_watchlist(tmp_path):
    path = _watchlist(tmp_path)
    with pytest.raises(ValueError, match="not on the watchlist"):
        resolve_and_summarize("does_not_exist", path, tmp_path)
