from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from urllib.parse import quote_plus, unquote, urljoin, urlparse

from playwright.async_api import (
    Browser,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from .models import SendicoListing

LOGGER = logging.getLogger(__name__)
PRICE_PATTERNS = [
    re.compile(r"(?:¥|JPY|円)\s*([0-9][0-9,]*)", re.I),
    re.compile(r"([0-9][0-9,]*)\s*(?:JPY|円)", re.I),
]
SELLER_PATTERNS = [
    re.compile(
        r"(?:positive(?:\s+ratings?)?|thumbs?\s*up|good\s+ratings?)"
        r"\D{0,30}([0-9][0-9,]*)",
        re.I,
    ),
    re.compile(
        r"([0-9][0-9,]*)\s*"
        r"(?:positive(?:\s+ratings?)?|thumbs?\s*up|good\s+ratings?)",
        re.I,
    ),
    re.compile(r"(?:良い|高評価)\D{0,20}([0-9][0-9,]*)"),
]
_MERCARI_ID = re.compile(r"m\d{8,}", re.I)
_IMAGE_EXT = re.compile(r"\.(?:jpe?g|png|webp)(?:$|\?)", re.I)
_PHOTO_INDEX_RE = re.compile(r"_(\d+)\.(?:jpe?g|png|webp)(?:\?|$)", re.I)


def parse_yen(text: str) -> int | None:
    for pattern in PRICE_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def parse_seller_positive_ratings(text: str) -> int | None:
    values = [
        int(match.group(1).replace(",", ""))
        for pattern in SELLER_PATTERNS
        for match in pattern.finditer(text or "")
    ]
    return max(values) if values else None


def listing_code(url: str) -> str:
    return urlparse(url).path.rstrip("/").split("/")[-1]


def is_listing_image_url(url: str, code: str) -> bool:
    candidate = unquote(str(url or ""))
    if not candidate or not code or not _IMAGE_EXT.search(candidate):
        return False
    found = {item.lower() for item in _MERCARI_ID.findall(candidate)}
    return code.lower() in candidate.lower() and (
        not found or found == {code.lower()}
    )


def _photo_key(url: str) -> str:
    """Identify which physical photo a URL is, independent of resolution."""

    match = _PHOTO_INDEX_RE.search(unquote(str(url or "")))
    return match.group(1) if match else url


def _photo_quality_rank(url: str) -> int:
    """Lower is better. Prefer full-resolution "orig" photos over thumbnails."""

    return 1 if "/thumb/" in str(url or "") else 0


def dedupe_listing_photos(urls: list[str]) -> list[str]:
    """Collapse thumb/orig duplicates of the same photo, keeping the best copy.

    Mercari listing pages link both a low-res thumbnail and a full-res "orig"
    image for every photo; without this, both count against
    max_images_downloaded for what is really one distinct photo, silently
    truncating away later photos in listings with many pictures.
    """

    best: dict[str, str] = {}
    order: list[str] = []
    for url in urls:
        key = _photo_key(url)
        current = best.get(key)
        if current is None:
            order.append(key)
            best[key] = url
        elif _photo_quality_rank(url) < _photo_quality_rank(current):
            best[key] = url
    return [best[key] for key in order]


def listing_from_search_item(item: dict[str, Any]) -> SendicoListing | None:
    """Convert one Sendico result-card payload into a listing.

    Search result cards do not consistently expose their price in the anchor or
    surrounding text. A missing search-page price must therefore not exclude an
    otherwise valid Mercari listing. Hydration will read the detail page and
    populate the price later.
    """

    url = str(item.get("href") or "").strip()
    if not url or "/categories/" in url:
        return None

    code = listing_code(url)
    if not _MERCARI_ID.fullmatch(code):
        return None

    text = str(item.get("text") or "").strip()
    title = str(item.get("title") or "").strip()
    if not title:
        title = next((line.strip() for line in text.splitlines() if line.strip()), code)

    return SendicoListing(
        code=code,
        url=url,
        title=title,
        price_yen=parse_yen(text) or 0,
        image_urls=[str(item["image"])] if item.get("image") else [],
        raw_text=text,
    )


class SendicoScanner:
    def __init__(self, config: dict, limits: dict) -> None:
        self.config = config
        self.limits = limits
        self.playwright = None
        self.browser: Browser | None = None

    async def __aenter__(self) -> "SendicoScanner":
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def _new_page(self, *, block_media: bool = False) -> Page:
        if not self.browser:
            raise RuntimeError("Scanner must be used as an async context manager")
        page = await self.browser.new_page(
            viewport={"width": 1440, "height": 1200},
            locale="en-AU",
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/130 Safari/537.36"
            ),
        )
        page.set_default_timeout(int(self.limits["search"]["page_timeout_ms"]))
        if block_media:
            async def _block_media(route):
                if route.request.resource_type in {"image", "font", "media"}:
                    await route.abort()
                else:
                    await route.continue_()

            await page.route(re.compile(r".*"), _block_media)
        return page

    async def _goto(self, page: Page, url: str) -> None:
        timeout = int(self.limits["search"]["page_timeout_ms"])
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        except PlaywrightTimeoutError:
            if not page.url or page.url == "about:blank":
                raise
            LOGGER.warning("Navigation timed out after commit; continuing: %s", url)

    async def search(self, term: str) -> list[SendicoListing]:
        page = await self._new_page(block_media=True)
        try:
            LOGGER.info("Starting Sendico search: %s", term)
            await self._goto(page, self.config["category_url"])
            await self._dismiss_cookies(page)
            search_method = await self._submit_search(page, term)
            try:
                await page.wait_for_selector('a[href*="/shop/mercari/catalog/"]')
            except PlaywrightTimeoutError:
                LOGGER.warning(
                    "No listing links appeared after search: term=%r", term
                )
            await self._scroll(page)

            raw = await page.locator(
                'a[href*="/shop/mercari/catalog/"]'
            ).evaluate_all(
                """
                (anchors) => anchors.map((a) => {
                  let node = a;
                  for (let i = 0; i < 7 && node; i++, node = node.parentElement) {
                    const txt = (node.innerText || '').trim();
                    const img = node.querySelector && node.querySelector('img');
                    if (txt.length > 3 && img) return {
                      href: a.href, text: txt,
                      title: (a.innerText || a.getAttribute('title') || '').trim(),
                      image: img.currentSrc || img.src || ''
                    };
                  }
                  return {
                    href: a.href,
                    text: (a.innerText || '').trim(),
                    title: '',
                    image: ''
                  };
                })
                """
            )

            results: list[SendicoListing] = []
            seen: set[str] = set()
            limit = int(self.limits["search"]["results_per_term"])
            rejected_invalid = 0
            rejected_duplicate = 0
            retained_without_price = 0

            for item in raw:
                listing = listing_from_search_item(item)
                if listing is None:
                    rejected_invalid += 1
                    continue
                if listing.url in seen:
                    rejected_duplicate += 1
                    continue

                seen.add(listing.url)
                if listing.price_yen <= 0:
                    retained_without_price += 1
                results.append(listing)
                if len(results) >= limit:
                    break

            LOGGER.info(
                "Sendico search completed: term=%r method=%s final_url=%s "
                "raw_anchors=%d retained=%d missing_result_price=%d "
                "duplicates=%d invalid=%d",
                term,
                search_method,
                page.url,
                len(raw),
                len(results),
                retained_without_price,
                rejected_duplicate,
                rejected_invalid,
            )
            if not results:
                LOGGER.warning(
                    "Sendico search returned no usable listings: term=%r "
                    "final_url=%s page_title=%r raw_anchors=%d",
                    term,
                    page.url,
                    await page.title(),
                    len(raw),
                )
            return results
        finally:
            try:
                await page.close()
            except Exception:
                pass

    async def _scroll(self, page: Page) -> None:
        maximum = int(self.limits["search"]["max_scroll_rounds"])
        stable_required = int(self.limits["search"]["stable_rounds_before_stop"])
        pause = int(self.limits["search"]["scroll_pause_ms"])
        raw_limit = int(self.limits["search"]["raw_links_per_term"])
        previous = -1
        stable = 0
        for round_number in range(maximum + 1):
            count = await page.locator(
                'a[href*="/shop/mercari/catalog/"]'
            ).count()
            LOGGER.info("Search load round %d: %d listing links", round_number, count)
            if count >= raw_limit:
                return
            stable = stable + 1 if count <= previous else 0
            if stable >= stable_required:
                return
            previous = count
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(pause)

    async def hydrate(self, listing: SendicoListing) -> SendicoListing:
        page = await self._new_page()
        try:
            await self._goto(page, listing.url)
            await self._dismiss_cookies(page)
            await page.wait_for_timeout(1300)
            # Exclude the "Related items" / "Other Items From This Seller"
            # carousels: they carry unrelated prices that a whole-page scan
            # can mistake for this listing's own price.
            body = await page.evaluate(
                """
                () => {
                  const clone = document.body.cloneNode(true);
                  const headings = Array.from(clone.querySelectorAll('h3')).filter((h) => {
                    const text = (h.textContent || '').trim();
                    return text === 'Related items'
                      || text === 'Other Items From This Seller';
                  });
                  for (const heading of headings) {
                    const section = heading.closest('.mt-7\\\\.5') || heading.parentElement;
                    if (section && section.parentElement) {
                      section.remove();
                    }
                  }
                  return clone.innerText;
                }
                """
            )
            heading = page.locator("h1").first
            if await heading.count():
                listing.title = (await heading.inner_text()).strip() or listing.title
            urls = await page.locator("img").evaluate_all(
                """
                (imgs) => imgs.flatMap((img) => [
                  img.currentSrc,
                  img.src,
                  img.getAttribute('data-src'),
                  img.getAttribute('data-original')
                ]).filter(Boolean)
                """
            )
            selected: list[str] = []
            for url in urls:
                absolute = urljoin(listing.url, str(url))
                if (
                    is_listing_image_url(absolute, listing.code)
                    and absolute not in selected
                ):
                    selected.append(absolute)
            if not selected:
                html = (await page.content()).replace("\\/", "/")
                for url in re.findall(
                    r"https?://[^\"'\s<>]+?\.(?:jpe?g|png|webp)"
                    r"(?:\?[^\"'\s<>]*)?",
                    html,
                    re.I,
                ):
                    if (
                        is_listing_image_url(url, listing.code)
                        and url not in selected
                    ):
                        selected.append(url)
            listing.image_urls = dedupe_listing_photos(
                list(dict.fromkeys([*listing.image_urls, *selected]))
            )
            listing.description = body
            listing.raw_text = body
            seller_badge = page.locator(
                'span[data-slot="base"]:has(span[class*="thumbs-up"]) '
                'span[data-slot="label"]'
            ).first
            if await seller_badge.count():
                rating_text = (await seller_badge.inner_text()).strip()
                listing.seller_positive_ratings = parse_seller_positive_ratings(
                    f"positive ratings {rating_text}"
                )
            else:
                listing.seller_positive_ratings = parse_seller_positive_ratings(body)
            if listing.price_yen <= 0:
                listing.price_yen = parse_yen(body) or 0
                if listing.price_yen > 0:
                    LOGGER.info(
                        "Hydration recovered missing price for %s: ¥%s",
                        listing.code,
                        f"{listing.price_yen:,}",
                    )
                else:
                    LOGGER.warning(
                        "Could not determine listing price after hydration: %s",
                        listing.code,
                    )
            return listing
        finally:
            try:
                await page.close()
            except Exception:
                pass

    async def _submit_search(self, page: Page, term: str) -> str:
        selectors = [
            'input[type="search"]',
            'input[placeholder*="Search" i]',
            'input[aria-label*="Search" i]',
            'input[name="search"]',
        ]
        for selector in selectors:
            locator = page.locator(selector).first
            if await locator.count() and await locator.is_visible():
                await locator.fill(term)
                await locator.press("Enter")
                LOGGER.info(
                    "Submitted Sendico search through visible input: %s", selector
                )
                return f"input:{selector}"

        separator = "&" if "?" in page.url else "?"
        fallback_url = f"{page.url}{separator}search={quote_plus(term)}"
        LOGGER.warning(
            "No visible Sendico search input found; using URL fallback: %s",
            fallback_url,
        )
        await self._goto(page, fallback_url)
        return "url-fallback"

    @staticmethod
    async def _dismiss_cookies(page: Page) -> None:
        for label in ["Accept", "Accept all", "I agree", "Got it"]:
            button = page.get_by_role(
                "button", name=re.compile(label, re.I)
            ).first
            try:
                if await button.count() and await button.is_visible():
                    await button.click()
                    await asyncio.sleep(0.1)
                    return
            except Exception:
                continue
