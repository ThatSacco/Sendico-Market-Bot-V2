from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .reference import parse_pricecharting_page

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PriceMatch:
    product_id: str
    name: str
    set_name: str
    card_number: str
    source_url: str
    ungraded_usd: float | None
    similarity: float


def _normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity_score(a: str, b: str) -> float:
    """Fuzzy 0.0-1.0 similarity between two card descriptions."""

    return difflib.SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _cache_key(name: str, card_number: str, set_name: str) -> str:
    return _normalize(f"{name} {card_number} {set_name}")


class PriceChartingSearchClient:
    """Resolves an identified card (name/number/set) to a PriceCharting price.

    There is no paid PriceCharting API key configured for this project, so
    this scrapes the public search-results page and scores candidates by
    name+number+set similarity. Results are cached aggressively on disk,
    keyed on the normalised name+number+set, since the same commons recur
    constantly across lots.
    """

    def __init__(
        self,
        root: Path,
        *,
        cache_hours: int = 24 * 14,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.root = root
        self.cache_path = root / "data/price_search_cache.json"
        self.cache_hours = max(1, int(cache_hours))
        self.client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "Chrome/130 Safari/537.36"
                ),
                "Accept-Language": "en-AU,en;q=0.9",
            },
            transport=transport,
        )
        self.cache = self._load_cache()

    def close(self) -> None:
        self.client.close()

    def _load_cache(self) -> dict[str, dict]:
        if not self.cache_path.exists():
            return {}
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _fresh(self, record: dict) -> bool:
        try:
            fetched = datetime.fromisoformat(str(record["fetched_at"]))
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - fetched < timedelta(hours=self.cache_hours)
        except Exception:
            return False

    def find_price(
        self,
        *,
        name: str,
        card_number: str = "",
        set_name: str = "",
    ) -> PriceMatch | None:
        key = _cache_key(name, card_number, set_name)
        cached = self.cache.get(key)
        if cached is not None and self._fresh(cached):
            match = cached.get("match")
            return PriceMatch(**match) if match else None

        query = " ".join(part for part in [name, card_number, set_name] if part).strip()
        match: PriceMatch | None = None
        if query:
            try:
                response = self.client.get(
                    "https://www.pricecharting.com/search-products",
                    params={"q": query, "type": "prices"},
                )
                response.raise_for_status()
                match = self._resolve(
                    response.text,
                    str(response.url),
                    name=name,
                    card_number=card_number,
                    set_name=set_name,
                )
            except httpx.HTTPError as exc:
                LOGGER.warning("PriceCharting search failed for %r: %s", query, exc)
                return None

        self.cache[key] = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "match": asdict(match) if match else None,
        }
        self._save_cache()
        return match

    def _resolve(
        self,
        html: str,
        final_url: str,
        *,
        name: str,
        card_number: str,
        set_name: str,
    ) -> PriceMatch | None:
        if "/search-products" not in urlparse(final_url).path:
            # An unambiguous query redirects straight to the product page.
            try:
                parsed = parse_pricecharting_page(final_url, html)
            except ValueError:
                return None
            candidate_text = f"{parsed['name']} {parsed['card_number']} {parsed['set_name']}"
            return PriceMatch(
                product_id=str(parsed["product_id"]),
                name=str(parsed["name"]),
                set_name=str(parsed["set_name"]),
                card_number=str(parsed["card_number"]),
                source_url=final_url,
                ungraded_usd=parsed["ungraded_usd"],
                similarity=similarity_score(
                    candidate_text, f"{name} {card_number} {set_name}"
                ),
            )

        soup = BeautifulSoup(html, "html.parser")
        target_text = f"{name} {card_number} {set_name}"
        best: PriceMatch | None = None
        for row in soup.select("#games_table tbody tr[data-product]"):
            title_link = row.select_one("td.title a")
            if not title_link:
                continue
            title = title_link.get_text(" ", strip=True)
            set_link = row.select_one("td.console a")
            row_set_name = set_link.get_text(" ", strip=True) if set_link else ""
            price_cell = row.select_one("td.price.used_price .js-price")
            price_text = price_cell.get_text(strip=True) if price_cell else ""
            price_match = re.search(r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)", price_text)
            ungraded_usd = (
                float(price_match.group(1).replace(",", "")) if price_match else None
            )
            candidate_text = f"{title} {row_set_name}"
            score = similarity_score(candidate_text, target_text)
            if best is None or score > best.similarity:
                best = PriceMatch(
                    product_id=str(row.get("data-product") or ""),
                    name=title,
                    set_name=row_set_name,
                    card_number=card_number,
                    source_url=str(title_link.get("href") or ""),
                    ungraded_usd=ungraded_usd,
                    similarity=score,
                )
        return best
