"""Discover listings from Mercari directly, priced and imaged from its own
schema.org data, while still linking the user to Sendico to buy.

Why this exists: Sendico resells Mercari listings, and as of 2026-08-06 it
sits behind a Cloudflare bot challenge that returns HTTP 403 to the scanner
(see the scan report where ``found`` went to 0 across every term). Mercari
is the original source for the same inventory, and publishes a schema.org
``Product`` block on each item page -- machine-readable by design -- giving
a canonical JPY price, the full-resolution photo set, and, uniquely,
whether the item is already sold.

Only paths Mercari's robots.txt permits are used: ``/search`` and
``/item/``. Its ``/v1/`` and ``/v2/`` API paths are explicitly disallowed
there and are deliberately not touched.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import quote, urljoin

from playwright.async_api import (
    Browser,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from .models import SendicoListing
from .sendico import dedupe_listing_photos

LOGGER = logging.getLogger(__name__)

MERCARI_ORIGIN = "https://jp.mercari.com"
SEARCH_PATH = "/search?keyword="
ITEM_PATH = "/item/"
SENDICO_ITEM_URL = "https://sendico.com/shop/mercari/catalog/{code}"
NEXT_PAGE_SELECTOR = '[data-testid="pagination-next-button"] a'
ITEM_ANCHOR_SELECTOR = 'a[href*="/item/m"]'
SELLER_SELECTOR = '[data-testid="seller-link"]'

_ITEM_ID = re.compile(r"m\d{8,}", re.I)
_SOLD_OUT = re.compile(r"SoldOut", re.I)
_RATING_COUNT = re.compile(r"(\d[\d,]*)")


def sendico_url(code: str) -> str:
    """Mercari's item id is exactly the code Sendico catalogues it under."""

    return SENDICO_ITEM_URL.format(code=code)


def mercari_item_url(code: str) -> str:
    return f"{MERCARI_ORIGIN}{ITEM_PATH}{code}"


def search_url(term: str) -> str:
    return f"{MERCARI_ORIGIN}{SEARCH_PATH}{quote(term)}"


def item_code_from_href(href: str) -> str | None:
    match = _ITEM_ID.search(str(href or ""))
    return match.group(0) if match else None


def extract_product_node(ld_json_blocks: list[str]) -> dict[str, Any] | None:
    """Pull the schema.org Product node out of a page's ld+json blocks.

    Mercari wraps its structured data in an ``@graph`` alongside
    Organization and BreadcrumbList nodes, so the Product has to be picked
    out rather than assumed to be the whole document.
    """

    for block in ld_json_blocks:
        try:
            data = json.loads(block)
        except (TypeError, ValueError):
            continue
        candidates = data.get("@graph") if isinstance(data, dict) else None
        if candidates is None:
            candidates = [data] if isinstance(data, dict) else list(data or [])
        for node in candidates:
            if isinstance(node, dict) and node.get("@type") == "Product":
                return node
    return None


def is_sold_out(product: dict[str, Any]) -> bool:
    offers = product.get("offers") or {}
    if isinstance(offers, list):
        offers = offers[0] if offers else {}
    return bool(_SOLD_OUT.search(str(offers.get("availability") or "")))


def product_price_yen(product: dict[str, Any]) -> int:
    """Read the canonical JPY price, ignoring any display-currency figure.

    Mercari localises the *rendered* price to the viewer's region (an
    Australian IP sees AU$), so the visible text is not a reliable source.
    The Product node's own offer always carries JPY.
    """

    offers = product.get("offers") or {}
    if isinstance(offers, list):
        offers = offers[0] if offers else {}
    currency = str(offers.get("priceCurrency") or "").upper()
    if currency and currency != "JPY":
        LOGGER.warning("Ignoring non-JPY structured price (%s)", currency)
        return 0
    try:
        return int(float(offers.get("price")))
    except (TypeError, ValueError):
        return 0


def parse_seller_ratings(text: str) -> int | None:
    """Read the rating count out of the seller badge's text.

    The badge reads like "ともも\\n21\\n本人確認済" -- display name, then the
    rating count, then an identity-verified label. Only the number is
    wanted, and the name may itself contain digits, so this reads the
    standalone numeric line rather than the first digits it finds.
    """

    for line in reversed(str(text or "").splitlines()):
        stripped = line.strip().replace(",", "")
        if stripped.isdigit():
            return int(stripped)
    return None


def listing_from_product(product: dict[str, Any], code: str) -> SendicoListing:
    """Build a listing from a Product node.

    ``url`` deliberately points at Sendico: that is where the user actually
    buys, and every downstream consumer (Discord alerts, the confirmation
    store) expects an actionable purchase link.
    """

    images = product.get("image") or []
    if isinstance(images, str):
        images = [images]
    return SendicoListing(
        code=code,
        url=sendico_url(code),
        title=str(product.get("name") or code),
        price_yen=product_price_yen(product),
        image_urls=dedupe_listing_photos([str(url) for url in images]),
        description=str(product.get("description") or ""),
        raw_text=str(product.get("description") or ""),
    )


class MercariScanner:
    """Drop-in replacement for SendicoScanner's search/hydrate interface."""

    def __init__(self, config: dict, limits: dict) -> None:
        self.config = config or {}
        self.limits = limits
        self.playwright = None
        self.browser: Browser | None = None

    async def __aenter__(self) -> "MercariScanner":
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
            viewport={"width": 1440, "height": 1600},
            locale="ja-JP",
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

    async def _settle(self, page: Page) -> int:
        """Wait for a search page's grid to finish rendering.

        Deliberately never treats an empty grid as "finished": Mercari
        renders results client-side, and a page that simply has not painted
        yet looks identical to a genuinely empty one. Reading zero as
        "done" is what silently truncated an earlier version of this crawl
        to two pages when a third had 63 more items on it.
        """

        pause = int(self.limits["search"]["scroll_pause_ms"])
        stable_required = int(self.limits["search"]["stable_rounds_before_stop"])
        maximum = int(self.limits["search"]["max_scroll_rounds"])
        previous = -1
        stable = 0
        for _ in range(max(1, maximum)):
            count = await page.locator(ITEM_ANCHOR_SELECTOR).count()
            if count and count == previous:
                stable += 1
                if stable >= stable_required:
                    break
            else:
                stable = 0
            previous = count
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(pause)
        return await page.locator(ITEM_ANCHOR_SELECTOR).count()

    async def search(self, term: str) -> list[SendicoListing]:
        """Collect item codes across every result page for a search term.

        Only codes and the Sendico link are established here; price,
        photos and sold status come from each item's own structured data
        during hydrate(), which is the authoritative source.
        """

        page = await self._new_page(block_media=True)
        try:
            LOGGER.info("Starting Mercari search: %s", term)
            await self._goto(page, search_url(term))
            await self._dismiss_cookies(page)

            codes: list[str] = []
            seen: set[str] = set()
            raw_limit = int(self.limits["search"]["raw_links_per_term"])
            max_pages = int(self.config.get("max_search_pages", 0) or 0)
            page_number = 0

            while True:
                page_number += 1
                await self._settle(page)
                hrefs = await page.locator(ITEM_ANCHOR_SELECTOR).evaluate_all(
                    "(els) => els.map((e) => e.getAttribute('href'))"
                )
                new_this_page = 0
                for href in hrefs:
                    code = item_code_from_href(href)
                    if not code or code in seen:
                        continue
                    seen.add(code)
                    codes.append(code)
                    new_this_page += 1
                LOGGER.info(
                    "Mercari search page %d: %d links, %d new (running total %d)",
                    page_number,
                    len(hrefs),
                    new_this_page,
                    len(codes),
                )

                if raw_limit > 0 and len(codes) >= raw_limit:
                    LOGGER.info("Reached raw_links_per_term (%d)", raw_limit)
                    break
                if max_pages and page_number >= max_pages:
                    LOGGER.info("Reached max_search_pages (%d)", max_pages)
                    break

                next_link = page.locator(NEXT_PAGE_SELECTOR).first
                if not await next_link.count():
                    LOGGER.info("No further result pages for %r", term)
                    break
                href = await next_link.get_attribute("href")
                if not href:
                    break
                await self._goto(page, urljoin(MERCARI_ORIGIN, href))

            results = [
                SendicoListing(code=code, url=sendico_url(code), title=code, price_yen=0)
                for code in codes
            ]
            if raw_limit > 0:
                results = results[:raw_limit]
            LOGGER.info(
                "Mercari search completed: term=%r pages=%d listings=%d",
                term,
                page_number,
                len(results),
            )
            if not results:
                LOGGER.warning(
                    "Mercari search returned no listings: term=%r final_url=%s "
                    "page_title=%r",
                    term,
                    page.url,
                    await page.title(),
                )
            return results
        finally:
            try:
                await page.close()
            except Exception:
                pass

    async def hydrate(self, listing: SendicoListing) -> SendicoListing:
        """Fill in price, photos, title and seller rating from the item page."""

        page = await self._new_page()
        try:
            await self._goto(page, mercari_item_url(listing.code))
            await self._dismiss_cookies(page)
            try:
                # state="attached" is essential, not incidental: the default
                # is "visible", and a <script> tag has no layout box, so it
                # can never become visible and the wait always times out.
                await page.wait_for_selector(
                    'script[type="application/ld+json"]',
                    state="attached",
                    timeout=int(self.limits["search"]["page_timeout_ms"]),
                )
            except PlaywrightTimeoutError:
                LOGGER.warning(
                    "No structured data appeared for %s; leaving unhydrated",
                    listing.code,
                )
                return listing

            blocks = await page.locator(
                'script[type="application/ld+json"]'
            ).all_inner_texts()
            product = extract_product_node(blocks)
            if product is None:
                LOGGER.warning(
                    "No schema.org Product node for %s; leaving unhydrated",
                    listing.code,
                )
                return listing

            hydrated = listing_from_product(product, listing.code)
            listing.title = hydrated.title
            listing.price_yen = hydrated.price_yen
            listing.image_urls = hydrated.image_urls
            listing.description = hydrated.description
            listing.raw_text = hydrated.raw_text
            listing.sold_out = is_sold_out(product)

            # The seller badge hydrates client-side, later than the ld+json
            # in the document head, so it needs its own wait. Optional data:
            # a listing is still perfectly usable without a rating count, so
            # a timeout here must not fail the hydration.
            try:
                await page.wait_for_selector(
                    SELLER_SELECTOR,
                    state="attached",
                    timeout=int(self.config.get("seller_timeout_ms", 15000)),
                )
            except PlaywrightTimeoutError:
                LOGGER.info("No seller badge rendered for %s", listing.code)
            seller = page.locator(SELLER_SELECTOR).first
            if await seller.count():
                listing.seller_positive_ratings = parse_seller_ratings(
                    await seller.inner_text()
                )

            if listing.price_yen <= 0:
                LOGGER.warning(
                    "Could not determine a JPY price for %s", listing.code
                )
            return listing
        finally:
            try:
                await page.close()
            except Exception:
                pass

    @staticmethod
    async def _dismiss_cookies(page: Page) -> None:
        for label in ["同意", "Accept", "Accept all", "OK", "閉じる"]:
            button = page.get_by_role("button", name=re.compile(label, re.I)).first
            try:
                if await button.count() and await button.is_visible():
                    await button.click()
                    await page.wait_for_timeout(200)
                    return
            except Exception:
                continue
