from pokemon_deal_bot.lot_valuation import lot_value
from pokemon_deal_bot.models import LotCard


def _card(name: str, priced_usd: float | None, similarity: float = 1.0) -> LotCard:
    return LotCard(
        name=name,
        identification_confidence=0.9,
        priced_usd=priced_usd,
        price_similarity=similarity,
    )


def test_lot_value_sums_only_confidently_priced_cards():
    cards = [
        _card("Pikachu", 10.0, similarity=0.97),
        _card("Eevee", 5.0, similarity=0.5),
        _card("Mew", None),
    ]
    valuation = lot_value(cards, visible_card_count=5)

    assert valuation.total_priced_usd == 10.0
    assert len(valuation.priced_cards) == 1
    assert valuation.unpriced_identified_count == 2
    assert valuation.unidentified_visible_count == 2
    assert valuation.identified_count == 3


def test_lot_value_respects_custom_threshold():
    cards = [_card("Pikachu", 10.0, similarity=0.8)]
    strict = lot_value(cards, visible_card_count=1, price_match_threshold=0.95)
    lenient = lot_value(cards, visible_card_count=1, price_match_threshold=0.7)

    assert strict.total_priced_usd == 0.0
    assert lenient.total_priced_usd == 10.0


def test_lot_value_never_reports_negative_unidentified_count():
    cards = [_card("Pikachu", 10.0)]
    valuation = lot_value(cards, visible_card_count=0)
    assert valuation.unidentified_visible_count == 0
