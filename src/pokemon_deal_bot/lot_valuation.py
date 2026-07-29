from __future__ import annotations

from dataclasses import dataclass

from .models import LotCard


@dataclass(slots=True)
class LotValuation:
    identified_cards: list[LotCard]
    priced_cards: list[LotCard]
    unpriced_identified_count: int
    unidentified_visible_count: int
    total_priced_usd: float

    @property
    def identified_count(self) -> int:
        return len(self.identified_cards)


def lot_value(
    cards: list[LotCard],
    *,
    visible_card_count: int,
    price_match_threshold: float = 0.95,
) -> LotValuation:
    """Sum priced cards and report coverage honestly.

    A card only counts toward ``total_priced_usd`` when it has a price AND
    that price's name/number/set similarity crosses ``price_match_threshold``
    -- an unconfident price match is worse than admitting the card is
    unpriced.
    """

    priced = [
        card
        for card in cards
        if card.priced_usd is not None
        and card.price_similarity >= price_match_threshold
    ]
    unpriced = len(cards) - len(priced)
    return LotValuation(
        identified_cards=cards,
        priced_cards=priced,
        unpriced_identified_count=unpriced,
        unidentified_visible_count=max(0, visible_card_count - len(cards)),
        total_priced_usd=sum(card.priced_usd or 0.0 for card in priced),
    )
