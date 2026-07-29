import json

import httpx

from pokemon_deal_bot.fx import FxRateClient


def test_fetch_uses_live_rate_and_caches(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"amount": 1.0, "base": "AUD", "date": "2026-07-29", "rates": {"JPY": 100.0, "USD": 0.5}})

    client = FxRateClient(
        tmp_path,
        manual_jpy_to_aud=0.0102,
        manual_usd_to_aud=1.52,
        transport=httpx.MockTransport(handler),
    )
    rates = client.fetch()
    client.close()

    assert rates.source == "live"
    assert rates.jpy_to_aud == 0.01
    assert rates.usd_to_aud == 2.0
    assert (tmp_path / "data/fx_cache.json").exists()


def test_fetch_uses_fresh_cache_without_network_call(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"rates": {"JPY": 100.0, "USD": 0.5}})

    client = FxRateClient(
        tmp_path,
        manual_jpy_to_aud=0.0102,
        manual_usd_to_aud=1.52,
        transport=httpx.MockTransport(handler),
    )
    first = client.fetch()
    second = client.fetch()
    client.close()

    assert len(calls) == 1
    assert second.jpy_to_aud == first.jpy_to_aud


def test_fetch_falls_back_to_manual_rate_on_failure(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    client = FxRateClient(
        tmp_path,
        manual_jpy_to_aud=0.0102,
        manual_usd_to_aud=1.52,
        transport=httpx.MockTransport(handler),
    )
    rates = client.fetch()
    client.close()

    assert rates.source == "manual_fallback"
    assert rates.jpy_to_aud == 0.0102
    assert rates.usd_to_aud == 1.52


def test_fetch_prefers_stale_cache_over_manual_rate_on_failure(tmp_path):
    cache_path = tmp_path / "data"
    cache_path.mkdir(parents=True)
    (cache_path / "fx_cache.json").write_text(
        json.dumps(
            {
                "jpy_to_aud": 0.0088,
                "usd_to_aud": 1.44,
                "fetched_at": "2020-01-01T00:00:00+00:00",
                "source": "live",
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    client = FxRateClient(
        tmp_path,
        manual_jpy_to_aud=0.0102,
        manual_usd_to_aud=1.52,
        transport=httpx.MockTransport(handler),
    )
    rates = client.fetch()
    client.close()

    assert rates.source == "live"
    assert rates.jpy_to_aud == 0.0088
