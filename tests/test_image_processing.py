from io import BytesIO

from PIL import Image

from pokemon_deal_bot.image_processing import make_contact_sheet


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
