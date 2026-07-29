from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .models import ReferenceCard, WatchTarget

LOGGER = logging.getLogger(__name__)
_PRICE_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
_TITLE_RE = re.compile(r"^(.*?)\s+#?([0-9]+(?:/[0-9]+)?)\s+Prices\s*\|\s*(.*?)\s*\|", re.I)
_PRODUCT_ID_RE = re.compile(r"PriceCharting ID:\s*[^0-9]*([0-9]+)", re.I)


def _money(text: str) -> float | None:
    match = _PRICE_RE.search(text or "")
    return float(match.group(1).replace(",", "")) if match else None


def _clean_number(value: str) -> str:
    return str(value or "").strip().lstrip("#")


def parse_pricecharting_page(url: str, html: str) -> dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    title_text = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()
    heading = soup.find("h1")
    heading_text = heading.get_text(" ", strip=True) if heading else ""
    combined_title = title_text or heading_text
    name = ""
    number = ""
    set_name = ""
    match = _TITLE_RE.search(combined_title)
    if match:
        name, number, set_name = [part.strip() for part in match.groups()]
    else:
        h_match = re.search(r"^(.*?)\s+#?([0-9]+(?:/[0-9]+)?)$", heading_text)
        if h_match:
            name, number = h_match.groups()
        detail_match = re.search(r"\((Pokemon[^)]+)\)", soup.get_text(" ", strip=True), re.I)
        if detail_match:
            set_name = detail_match.group(1).strip()

    image_url = ""
    for attrs in [
        {"property": "og:image"},
        {"name": "twitter:image"},
        {"property": "twitter:image"},
    ]:
        node = soup.find("meta", attrs=attrs)
        if node and node.get("content"):
            image_url = urljoin(url, str(node["content"]))
            break
    if not image_url:
        candidates = soup.find_all("img")
        scored: list[tuple[int, str]] = []
        for image in candidates:
            src = image.get("src") or image.get("data-src") or image.get("data-original")
            if not src:
                continue
            alt = str(image.get("alt") or "").casefold()
            source = str(src)
            score = 0
            if "main image" in alt or (name and name.casefold() in alt):
                score += 5
            if "storage.googleapis.com" in source or "images.pricecharting" in source:
                score += 4
            if "product" in str(image.get("id") or "").casefold():
                score += 3
            if score:
                scored.append((score, urljoin(url, source)))
        if scored:
            image_url = max(scored, key=lambda item: item[0])[1]

    page_text = soup.get_text("\n", strip=True)
    product_id = ""
    product_match = _PRODUCT_ID_RE.search(page_text)
    if product_match:
        product_id = product_match.group(1)
    if not product_id:
        product_id = Path(urlparse(url).path).name

    ungraded = None
    psa10 = None
    for row in soup.find_all("tr"):
        text = row.get_text(" ", strip=True)
        if re.search(r"\bUngraded\b", text, re.I) and ungraded is None:
            ungraded = _money(text)
        if re.search(r"\bPSA\s*10\b", text, re.I) and psa10 is None:
            psa10 = _money(text)
    if ungraded is None:
        match = re.search(
            r"\bUngraded\b\s*(?:[|:]\s*)?(\$[0-9,.]+)",
            page_text,
            re.I,
        )
        ungraded = _money(match.group(1)) if match else None
    if psa10 is None:
        match = re.search(
            r"\bPSA\s*10\b\s*(?:[|:]\s*)?(\$[0-9,.]+)",
            page_text,
            re.I,
        )
        psa10 = _money(match.group(1)) if match else None

    if not name or not image_url:
        raise ValueError("PriceCharting page did not expose a product name and main reference image")
    return {
        "product_id": product_id,
        "name": name.strip(),
        "set_name": set_name.strip(),
        "card_number": _clean_number(number),
        "image_url": image_url,
        "ungraded_usd": ungraded,
        "psa10_usd": psa10,
    }


class PriceChartingReferenceClient:
    def __init__(
        self,
        root: Path,
        *,
        cache_hours: int = 24,
        timeout_seconds: float = 45.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.root = root
        self.cache_path = root / "data/reference_cache.json"
        self.image_dir = root / "data/reference_images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
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

    def _load_cache(self) -> dict[str, dict[str, object]]:
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

    def _fresh(self, record: dict[str, object]) -> bool:
        try:
            fetched = datetime.fromisoformat(str(record["fetched_at"]))
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - fetched < timedelta(hours=self.cache_hours)
        except Exception:
            return False

    def _fetch_reference_image(self, url: str) -> tuple[bytes, str, str]:
        """PriceCharting storage paths are size-keyed; prefer the largest available."""
        for size in ("1600.jpg", "1024.jpg", "640.jpg", "400.jpg"):
            candidate = re.sub(r"/\d+\.jpg$", f"/{size}", url)
            if candidate == url:
                break
            try:
                response = self.client.get(candidate)
                if not response.is_error and len(response.content) > 2000:
                    content_type = response.headers.get("content-type", "")
                    return response.content, candidate, content_type
            except httpx.HTTPError:
                continue
        response = self.client.get(url)
        response.raise_for_status()
        return response.content, url, response.headers.get("content-type", "")

    def resolve(self, target: WatchTarget, *, force: bool = False) -> ReferenceCard:
        record = self.cache.get(target.pricecharting_url) or {}
        image_path = self.root / str(record.get("image_path") or "")
        if not force and record and self._fresh(record) and image_path.is_file():
            return self._to_reference(target.id, target.pricecharting_url, record)

        response = self.client.get(target.pricecharting_url)
        response.raise_for_status()
        parsed = parse_pricecharting_page(target.pricecharting_url, response.text)
        image_content, resolved_image_url, content_type = self._fetch_reference_image(
            str(parsed["image_url"])
        )
        suffix = ".png" if "png" in content_type else ".jpg"
        filename = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(parsed["product_id"])) + suffix
        relative_path = Path("data/reference_images") / filename
        absolute_path = self.root / relative_path
        absolute_path.write_bytes(image_content)
        record = {
            **parsed,
            "image_url": resolved_image_url,
            "image_path": relative_path.as_posix(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        self.cache[target.pricecharting_url] = record
        self._save_cache()
        LOGGER.info("Cached PriceCharting reference for %s at %s", target.id, relative_path)
        return self._to_reference(target.id, target.pricecharting_url, record)

    def _to_reference(self, target_id: str, source_url: str, record: dict[str, object]) -> ReferenceCard:
        return ReferenceCard(
            target_id=target_id,
            source_url=source_url,
            product_id=str(record.get("product_id") or target_id),
            name=str(record.get("name") or target_id),
            set_name=str(record.get("set_name") or ""),
            card_number=str(record.get("card_number") or ""),
            image_url=str(record.get("image_url") or ""),
            image_path=self.root / str(record.get("image_path") or ""),
            ungraded_usd=float(record["ungraded_usd"]) if record.get("ungraded_usd") is not None else None,
            psa10_usd=float(record["psa10_usd"]) if record.get("psa10_usd") is not None else None,
            fetched_at=str(record.get("fetched_at") or ""),
        )
