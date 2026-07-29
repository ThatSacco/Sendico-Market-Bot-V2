import json
from pathlib import Path
import httpx

from pokemon_deal_bot.models import WatchSearch, WatchTarget
from pokemon_deal_bot.reference import PriceChartingReferenceClient, parse_pricecharting_page

HTML = """
<html><head>
<title>Victini #97 Prices | Pokemon Japanese Black Bolt | Pokemon Cards</title>
<meta property="og:image" content="https://storage.googleapis.com/example/victini.jpg">
</head><body>
<h1>Victini #97</h1>
<table><tr><td>Ungraded</td><td>$17.84</td></tr><tr><td>PSA 10</td><td>$73.63</td></tr></table>
<div>PriceCharting ID: 9647332</div>
</body></html>
"""


def test_parse_pricecharting_reference():
    parsed = parse_pricecharting_page("https://www.pricecharting.com/game/pokemon-japanese-black-bolt/victini-97", HTML)
    assert parsed["name"] == "Victini"
    assert parsed["card_number"] == "97"
    assert parsed["set_name"] == "Pokemon Japanese Black Bolt"
    assert parsed["image_url"].endswith("victini.jpg")
    assert parsed["ungraded_usd"] == 17.84
    assert parsed["psa10_usd"] == 73.63


def test_reference_client_caches_page_and_image(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.pricecharting.com":
            return httpx.Response(200, text=HTML)
        return httpx.Response(200, content=b"fake-jpeg", headers={"content-type": "image/jpeg"})
    client = PriceChartingReferenceClient(tmp_path, transport=httpx.MockTransport(handler))
    target = WatchTarget("victini", "https://www.pricecharting.com/game/pokemon-japanese-black-bolt/victini-97", [WatchSearch("x")])
    reference = client.resolve(target)
    client.close()
    assert reference.name == "Victini"
    assert reference.image_path.read_bytes() == b"fake-jpeg"
    assert (tmp_path / "data/reference_cache.json").exists()


SIZE_KEYED_HTML = HTML.replace(
    "https://storage.googleapis.com/example/victini.jpg",
    "https://storage.googleapis.com/example/240.jpg",
)


def test_reference_client_prefers_largest_available_image(tmp_path):
    requested_image_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.pricecharting.com":
            return httpx.Response(200, text=SIZE_KEYED_HTML)
        requested_image_urls.append(str(request.url))
        if str(request.url).endswith("/1600.jpg"):
            return httpx.Response(
                200,
                content=b"x" * 3000,
                headers={"content-type": "image/jpeg"},
            )
        return httpx.Response(404)

    client = PriceChartingReferenceClient(tmp_path, transport=httpx.MockTransport(handler))
    target = WatchTarget(
        "victini",
        "https://www.pricecharting.com/game/pokemon-japanese-black-bolt/victini-97",
        [WatchSearch("x")],
    )
    reference = client.resolve(target)
    client.close()
    assert reference.image_path.read_bytes() == b"x" * 3000
    assert reference.image_url.endswith("/1600.jpg")
    assert requested_image_urls[0].endswith("/1600.jpg")

    cache = json.loads((tmp_path / "data/reference_cache.json").read_text())
    assert cache[target.pricecharting_url]["image_url"].endswith("/1600.jpg")
