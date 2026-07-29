import httpx

from pokemon_deal_bot.pricecharting_search import PriceChartingSearchClient, similarity_score

RESULTS_HTML = """
<html><body>
<table id="games_table">
<tbody>
<tr data-product="111">
  <td class="title"><a href="https://www.pricecharting.com/game/pokemon-japanese-promo/illustrator-pikachu">Illustrator Pikachu</a></td>
  <td class="console"><a href="/console/pokemon-japanese-promo">Pokemon Japanese Promo</a></td>
  <td class="price numeric used_price"><span class="js-price">$1,079,036.22</span></td>
</tr>
<tr data-product="222">
  <td class="title"><a href="https://www.pricecharting.com/game/pokemon-japanese-black-bolt/victini-ar-097">Victini AR 097/086 Black Bolt sv11B</a></td>
  <td class="console"><a href="/console/pokemon-japanese-black-bolt">Pokemon Japanese Black Bolt</a></td>
  <td class="price numeric used_price"><span class="js-price">$26.80</span></td>
</tr>
</tbody>
</table>
</body></html>
"""

PRODUCT_HTML = """
<html><head>
<title>Victini #97 Prices | Pokemon Japanese Black Bolt | Pokemon Cards</title>
<meta property="og:image" content="https://storage.googleapis.com/example/victini.jpg">
</head><body>
<h1>Victini #97</h1>
<table><tr><td>Ungraded</td><td>$17.84</td></tr></table>
<div>PriceCharting ID: 9647332</div>
</body></html>
"""


def test_similarity_score_ignores_case_and_punctuation():
    assert similarity_score("Victini #97", "victini 97") > 0.9


def test_find_price_picks_best_match_from_results_table(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=RESULTS_HTML)

    client = PriceChartingSearchClient(tmp_path, transport=httpx.MockTransport(handler))
    match = client.find_price(
        name="Victini AR", card_number="097", set_name="Black Bolt"
    )
    client.close()

    assert match is not None
    assert match.product_id == "222"
    assert match.ungraded_usd == 26.80
    assert match.similarity > 0.5


def test_find_price_follows_redirect_to_single_product_page(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if "search-products" in str(request.url):
            return httpx.Response(
                302,
                headers={
                    "location": (
                        "https://www.pricecharting.com/game/"
                        "pokemon-japanese-black-bolt/victini-97"
                    )
                },
            )
        return httpx.Response(200, text=PRODUCT_HTML)

    client = PriceChartingSearchClient(tmp_path, transport=httpx.MockTransport(handler))
    match = client.find_price(name="Victini", card_number="97", set_name="Black Bolt")
    client.close()

    assert match is not None
    assert match.product_id == "9647332"
    assert match.ungraded_usd == 17.84


def test_find_price_caches_result_on_disk(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text=RESULTS_HTML)

    client = PriceChartingSearchClient(tmp_path, transport=httpx.MockTransport(handler))
    client.find_price(name="Victini AR", card_number="097", set_name="Black Bolt")
    client.find_price(name="Victini AR", card_number="097", set_name="Black Bolt")
    client.close()

    assert len(calls) == 1
    assert (tmp_path / "data/price_search_cache.json").exists()
