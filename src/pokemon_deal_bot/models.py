from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WatchSearch:
    term: str
    active: bool = True


@dataclass(slots=True)
class WatchTarget:
    id: str
    pricecharting_url: str
    searches: list[WatchSearch] = field(default_factory=list)
    active: bool = True


@dataclass(slots=True)
class ReferenceCard:
    target_id: str
    source_url: str
    product_id: str
    name: str
    set_name: str
    card_number: str
    image_url: str
    image_path: Path
    ungraded_usd: float | None = None
    psa10_usd: float | None = None
    fetched_at: str = ""

    @property
    def display_name(self) -> str:
        number = f" #{self.card_number}" if self.card_number else ""
        set_name = f" ({self.set_name})" if self.set_name else ""
        return f"{self.name}{number}{set_name}".strip()


@dataclass(slots=True)
class SearchTask:
    term: str
    target_ids: list[str]


@dataclass(slots=True)
class SendicoListing:
    code: str
    url: str
    title: str
    price_yen: int
    image_urls: list[str] = field(default_factory=list)
    description: str = ""
    raw_text: str = ""
    seller_positive_ratings: int | None = None
    candidate_target_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VisualMatch:
    target_id: str
    stage: str
    confidence: float
    same_card: bool
    candidate_labels: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    model: str = ""

    @property
    def match_score(self) -> float:
        """Directional probability the target is present, not verdict confidence."""
        return self.confidence if self.same_card else 0.0


@dataclass(slots=True)
class ScanStats:
    found: int = 0
    candidates: int = 0
    hydrated: int = 0
    screened: int = 0
    detailed: int = 0
    probable_matches: int = 0
    confirmed_matches: int = 0
    held: int = 0
    skipped_seen: int = 0
    skipped_seller: int = 0
    processing_errors: int = 0
    alerts_sent: int = 0
    requests_sent: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    models_used: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
