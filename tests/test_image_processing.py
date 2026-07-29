import asyncio
from io import BytesIO

import httpx
from PIL import Image

from pokemon_deal_bot.image_processing import download_listing_images, make_contact_sheet


def test_contact_sheet_upscales_small_marketplace_thumbnails():
    source = Image.new("RGB", (240, 240), "white")
    sheet = make_contact_sheet(
        [source],
        prefix="O",
        max_dimension=1400,
        quality=80,
        columns=2,
    )
    output = Image.open(BytesIO(sheet.jpeg))
    # The old pipeline left a 240 px image floating inside a large white canvas.
    # The reference matcher now receives a materially enlarged thumbnail.
    assert output.width >= 600
    assert output.height >= 600


def _jpeg_bytes(color: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (10, 10), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _dominant_channel(image: Image.Image) -> str:
    r, g, b = image.getpixel((0, 0))
    return max([("r", r), ("g", g), ("b", b)], key=lambda item: item[1])[0]


def test_download_listing_images_preserves_url_order_despite_concurrency():
    colors = {"a": "red", "b": "green", "c": "blue"}

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, content=_jpeg_bytes(colors[key]))

    images = asyncio.run(
        download_listing_images(
            [
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/c",
            ],
            maximum=10,
            transport=httpx.MockTransport(handler),
        )
    )
    assert [_dominant_channel(image) for image in images] == ["r", "g", "b"]


def test_download_listing_images_skips_failures_without_losing_others():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("bad"):
            return httpx.Response(500)
        return httpx.Response(200, content=_jpeg_bytes("red"))

    images = asyncio.run(
        download_listing_images(
            [
                "https://example.com/good1",
                "https://example.com/bad",
                "https://example.com/good2",
            ],
            maximum=10,
            transport=httpx.MockTransport(handler),
        )
    )
    assert len(images) == 2


def test_download_listing_images_runs_concurrently_not_sequentially():
    in_flight = 0
    max_in_flight = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return httpx.Response(200, content=_jpeg_bytes("red"))

    urls = [f"https://example.com/{i}" for i in range(5)]
    asyncio.run(
        download_listing_images(
            urls,
            maximum=10,
            concurrency=5,
            transport=httpx.MockTransport(handler),
        )
    )
    # The old implementation awaited each download in a sequential for-loop,
    # so max_in_flight would never exceed 1.
    assert max_in_flight > 1
