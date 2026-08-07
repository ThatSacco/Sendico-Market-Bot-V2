import json

import pytest

from pokemon_deal_bot.mercari import (
    extract_product_node,
    is_sold_out,
    item_code_from_href,
    listing_from_product,
    parse_seller_ratings,
    product_price_yen,
    search_url,
    sendico_url,
)

# Captured verbatim from a live jp.mercari.com item page (2026-08-06), so
# these tests break if Mercari's real structured-data shape changes.
REAL_LD_JSON = json.dumps(
    {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "Organization",
                "name": "メルカリ",
                "url": "https://jp.mercari.com",
            },
            {
                "@type": "BreadcrumbList",
                "itemListElement": [{"@type": "ListItem", "position": 1}],
            },
            {
                "@type": "Product",
                "productID": "m55515637157",
                "image": [
                    "https://static.mercdn.net/item/detail/orig/photos/m55515637157_1.jpg?1785111356",
                    "https://static.mercdn.net/item/detail/orig/photos/m55515637157_2.jpg?1785111356",
                ],
                "name": "ポケモンカード ブラックボルト ホワイトフレア カードファイルセット",
                "description": "最大156枚収納可能なカードファイル",
                "offers": {
                    "@type": "Offer",
                    "url": "https://jp.mercari.com/item/m55515637157",
                    "availability": "https://schema.org/SoldOut",
                    "price": 7999,
                    "priceCurrency": "JPY",
                },
            },
        ],
    },
    ensure_ascii=False,
)

IN_STOCK_LD_JSON = json.dumps(
    {
        "@graph": [
            {
                "@type": "Product",
                "productID": "m33417629380",
                "image": ["https://static.mercdn.net/item/detail/orig/photos/m33417629380_1.jpg"],
                "name": "ポケモンカードゲーム プロモ41枚まとめ売り",
                "offers": {
                    "availability": "https://schema.org/InStock",
                    "price": 24000,
                    "priceCurrency": "JPY",
                },
            }
        ]
    },
    ensure_ascii=False,
)


def _product(raw: str = REAL_LD_JSON) -> dict:
    node = extract_product_node([raw])
    assert node is not None
    return node


def test_sendico_url_is_built_from_the_mercari_item_id():
    # The whole Mercari-as-source approach rests on this equivalence.
    assert (
        sendico_url("m55515637157")
        == "https://sendico.com/shop/mercari/catalog/m55515637157"
    )


def test_search_url_encodes_japanese_terms():
    url = search_url("ビクティニ プロモ")
    assert url.startswith("https://jp.mercari.com/search?keyword=")
    assert " " not in url


def test_item_code_from_href_handles_relative_and_absolute():
    assert item_code_from_href("/item/m55515637157") == "m55515637157"
    assert item_code_from_href("https://jp.mercari.com/item/m33417629380?x=1") == "m33417629380"
    assert item_code_from_href("/search?keyword=x") is None
    assert item_code_from_href("") is None


def test_extract_product_node_skips_organization_and_breadcrumbs():
    node = extract_product_node([REAL_LD_JSON])
    assert node is not None
    assert node["@type"] == "Product"
    assert node["productID"] == "m55515637157"


def test_extract_product_node_returns_none_without_a_product():
    only_org = json.dumps({"@graph": [{"@type": "Organization", "name": "x"}]})
    assert extract_product_node([only_org]) is None


def test_extract_product_node_survives_malformed_json():
    assert extract_product_node(["{not json", ""]) is None
    # A broken block must not stop a later valid one from being found.
    assert extract_product_node(["{oops", REAL_LD_JSON]) is not None


def test_price_is_read_from_structured_data_in_yen():
    # The rendered page shows AU$ to an Australian viewer; only the
    # structured offer carries the real JPY figure.
    assert product_price_yen(_product()) == 7999


def test_non_jpy_structured_price_is_refused_rather_than_misread():
    node = {"offers": {"price": 83.91, "priceCurrency": "AUD"}}
    assert product_price_yen(node) == 0


def test_missing_or_unparseable_price_returns_zero():
    assert product_price_yen({}) == 0
    assert product_price_yen({"offers": {"price": None, "priceCurrency": "JPY"}}) == 0
    assert product_price_yen({"offers": {"price": "abc", "priceCurrency": "JPY"}}) == 0


def test_sold_out_detection():
    assert is_sold_out(_product()) is True
    assert is_sold_out(_product(IN_STOCK_LD_JSON)) is False
    assert is_sold_out({}) is False


def test_listing_from_product_populates_everything_downstream_needs():
    listing = listing_from_product(_product(), "m55515637157")

    assert listing.code == "m55515637157"
    # Alerts must remain actionable: the link is where the user buys.
    assert listing.url == "https://sendico.com/shop/mercari/catalog/m55515637157"
    assert listing.title.startswith("ポケモンカード")
    assert listing.price_yen == 7999
    assert len(listing.image_urls) == 2
    assert all("/item/detail/orig/photos/" in url for url in listing.image_urls)


def test_listing_from_product_accepts_a_single_image_string():
    node = {"name": "x", "image": "https://static.mercdn.net/item/detail/orig/photos/m1_1.jpg",
            "offers": {"price": 100, "priceCurrency": "JPY"}}
    listing = listing_from_product(node, "m1")
    assert listing.image_urls == [
        "https://static.mercdn.net/item/detail/orig/photos/m1_1.jpg"
    ]


def test_listing_from_product_dedupes_thumb_and_orig_of_the_same_photo():
    node = {
        "name": "x",
        "image": [
            "https://static.mercdn.net/thumb/item/webp/m1_1.jpg",
            "https://static.mercdn.net/item/detail/orig/photos/m1_1.jpg",
        ],
        "offers": {"price": 100, "priceCurrency": "JPY"},
    }
    listing = listing_from_product(node, "m1")
    # One physical photo -- keep the full-resolution copy, not the thumbnail.
    assert listing.image_urls == [
        "https://static.mercdn.net/item/detail/orig/photos/m1_1.jpg"
    ]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ともも\n\n21\n本人確認済", 21),
        ("菊ちゃん\n\n619\n本人確認済", 619),
        ("1234さん\n\n7\n本人確認済", 7),  # digits in the display name
        ("1,024\n", 1024),
        ("名前のみ", None),
        ("", None),
    ],
)
def test_parse_seller_ratings(text, expected):
    assert parse_seller_ratings(text) == expected
