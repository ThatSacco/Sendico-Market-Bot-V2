from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from .models import SearchTask, WatchSearch, WatchTarget


@dataclass(slots=True)
class AppConfig:
    root: Path
    raw: dict[str, Any]
    run_limits: dict[str, Any]
    criteria: dict[str, Any]
    targets: list[WatchTarget]

    @property
    def gemini_api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY")

    @property
    def discord_webhook_url(self) -> str | None:
        return os.getenv("DISCORD_WEBHOOK_URL")

    @property
    def discord_confirmed_webhook_url(self) -> str | None:
        return os.getenv("DISCORD_CONFIRMED_WEBHOOK_URL")

    @property
    def discord_bot_token(self) -> str | None:
        return os.getenv("DISCORD_BOT_TOKEN")

    @property
    def discord_alert_channel_id(self) -> str | None:
        return str(self.raw.get("discord", {}).get("alert_channel_id") or "") or None

    def path(self, relative: str) -> Path:
        return self.root / relative


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a YAML mapping")
    return value


def _require_number(value: Any, label: str, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if number < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return number


def validate_run_limits(data: dict[str, Any]) -> None:
    if int(data.get("version", 0)) != 1:
        raise ValueError("data/run_limits.yaml version must be 1")
    search = data.get("search") or {}
    screening = data.get("screening") or {}
    detailed = data.get("detailed_analysis") or {}
    token = data.get("token_budget") or {}
    state = data.get("state") or {}
    # 0 means unlimited for results_per_term/raw_links_per_term, matching
    # token_budget.max_total_tokens_per_run's convention.
    results = int(_require_number(search.get("results_per_term"), "search.results_per_term", minimum=0))
    total = int(_require_number(search.get("total_listings_per_run"), "search.total_listings_per_run", minimum=1))
    raw_links = int(_require_number(search.get("raw_links_per_term"), "search.raw_links_per_term", minimum=0))
    if results and total < results:
        raise ValueError("search.total_listings_per_run must be >= search.results_per_term")
    if results and raw_links and raw_links < results:
        raise ValueError("search.raw_links_per_term must be >= search.results_per_term")
    screen_total = int(_require_number(screening.get("max_listings_per_run"), "screening.max_listings_per_run"))
    _require_number(screening.get("images_per_batch"), "screening.images_per_batch", minimum=1)
    _require_number(screening.get("max_batches_per_listing"), "screening.max_batches_per_listing", minimum=1)
    _require_number(screening.get("max_image_dimension_px"), "screening.max_image_dimension_px", minimum=300)
    _require_number(screening.get("jpeg_quality"), "screening.jpeg_quality", minimum=1, maximum=100)
    detail_total = int(_require_number(detailed.get("max_listings_per_run"), "detailed_analysis.max_listings_per_run"))
    _require_number(detailed.get("max_images_downloaded"), "detailed_analysis.max_images_downloaded", minimum=1)
    _require_number(detailed.get("max_card_crops_per_listing"), "detailed_analysis.max_card_crops_per_listing", minimum=1)
    _require_number(detailed.get("images_per_batch"), "detailed_analysis.images_per_batch", minimum=1)
    _require_number(detailed.get("max_batches_per_listing"), "detailed_analysis.max_batches_per_listing", minimum=1)
    _require_number(detailed.get("max_image_dimension_px"), "detailed_analysis.max_image_dimension_px", minimum=300)
    _require_number(detailed.get("jpeg_quality"), "detailed_analysis.jpeg_quality", minimum=1, maximum=100)
    _require_number(detailed.get("contact_sheet_columns"), "detailed_analysis.contact_sheet_columns", minimum=1)
    if screen_total and screen_total > total:
        raise ValueError("screening.max_listings_per_run cannot exceed search.total_listings_per_run")
    if detail_total and detail_total > total:
        raise ValueError("detailed_analysis.max_listings_per_run cannot exceed search.total_listings_per_run")
    # 0 means unlimited, matching token_budget.max_requests_per_run's convention.
    ceiling = int(_require_number(token.get("max_total_tokens_per_run"), "token_budget.max_total_tokens_per_run", minimum=0))
    reserve = int(_require_number(token.get("reserve_per_request"), "token_budget.reserve_per_request", minimum=0))
    if ceiling and reserve >= ceiling:
        raise ValueError("token_budget.reserve_per_request must be below max_total_tokens_per_run")
    _require_number(state.get("max_seen_listings"), "state.max_seen_listings", minimum=100)


def validate_criteria(data: dict[str, Any]) -> None:
    if int(data.get("version", 0)) != 1:
        raise ValueError("data/search_criteria.yaml version must be 1")
    matching = data.get("reference_image_matching") or {}
    for key in [
        "minimum_screening_confidence_for_detail",
        "probable_alert_threshold",
        "confirmed_threshold",
    ]:
        _require_number(matching.get(key), f"reference_image_matching.{key}", minimum=0, maximum=1)
    if float(matching["minimum_screening_confidence_for_detail"]) > float(matching["probable_alert_threshold"]):
        raise ValueError(
            "minimum_screening_confidence_for_detail should not exceed "
            "probable_alert_threshold"
        )
    if float(matching["probable_alert_threshold"]) > float(matching["confirmed_threshold"]):
        raise ValueError("probable_alert_threshold cannot exceed confirmed_threshold")
    seller = data.get("seller") or {}
    _require_number(seller.get("minimum_positive_ratings"), "seller.minimum_positive_ratings", minimum=0)
    for key in [
        "alert_on_probable_match",
        "alert_on_confirmed_match",
        "compare_all_active_targets",
    ]:
        if not isinstance(matching.get(key), bool):
            raise ValueError(f"reference_image_matching.{key} must be true or false")
    if not isinstance(seller.get("analyse_unverified_sellers"), bool):
        raise ValueError("seller.analyse_unverified_sellers must be true or false")


def _validate_pricecharting_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() not in {"pricecharting.com", "www.pricecharting.com"}:
        raise ValueError(f"Watchlist reference must be a PriceCharting HTTPS product URL: {url}")
    if "/game/" not in parsed.path:
        raise ValueError(f"PriceCharting URL must point to a product page: {url}")


def _derive_id_from_url(url: str) -> str:
    """Turn a PriceCharting product slug (.../victini-97) into victini_97."""

    slug = Path(urlparse(url).path).name
    return re.sub(r"[^a-zA-Z0-9]+", "_", slug).strip("_").lower()


def load_watchlist(path: Path) -> list[WatchTarget]:
    data = _load_yaml(path, "data/watchlist.yaml")
    targets: list[WatchTarget] = []
    for item in data.get("cards") or []:
        if not isinstance(item, dict) or not item.get("active", True):
            continue
        url = str(item.get("pricecharting_url") or "").strip()
        _validate_pricecharting_url(url)
        target_id = str(item.get("id") or "").strip() or _derive_id_from_url(url)
        if not target_id:
            raise ValueError(f"Could not derive a watchlist id from {url!r}")
        searches: list[WatchSearch] = []
        for raw_search in item.get("searches") or []:
            if isinstance(raw_search, str):
                term = raw_search.strip()
                if term:
                    searches.append(WatchSearch(term=term, active=True))
                continue
            if not isinstance(raw_search, dict) or not raw_search.get("active", True):
                continue
            term = str(raw_search.get("term") or "").strip()
            if not term:
                continue
            searches.append(WatchSearch(term=term, active=True))
        if not searches:
            raise ValueError(f"Active watchlist card {target_id!r} needs at least one active search")
        targets.append(
            WatchTarget(
                id=target_id,
                pricecharting_url=url,
                searches=searches,
                active=True,
            )
        )
    if not targets:
        raise ValueError("data/watchlist.yaml has no active cards")
    ids = [target.id for target in targets]
    if len(ids) != len(set(ids)):
        raise ValueError("Active watchlist ids must be unique")
    return targets


def build_search_plan(targets: list[WatchTarget]) -> list[SearchTask]:
    grouped: dict[str, list[str]] = {}
    for target in targets:
        for search in target.searches:
            grouped.setdefault(search.term, []).append(target.id)
    return [
        SearchTask(term=term, target_ids=target_ids)
        for term, target_ids in grouped.items()
    ]


def scan_signature(config: AppConfig) -> str:
    payload = {
        "targets": [asdict(target) for target in sorted(config.targets, key=lambda item: item.id)],
        "criteria": config.criteria,
        "pipeline": "pricecharting-reference-image-v8",
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path).resolve()
    raw = _load_yaml(config_path, "config.yaml")
    root = config_path.parent
    run_limits_path = root / str(raw.get("run_limits_file") or "data/run_limits.yaml")
    criteria_path = root / str(raw.get("search_criteria_file") or "data/search_criteria.yaml")
    watchlist_path = root / str(raw.get("watchlist_file") or "data/watchlist.yaml")
    limits = _load_yaml(run_limits_path, "data/run_limits.yaml")
    criteria = _load_yaml(criteria_path, "data/search_criteria.yaml")
    validate_run_limits(limits)
    validate_criteria(criteria)
    targets = load_watchlist(watchlist_path)
    return AppConfig(root=root, raw=raw, run_limits=limits, criteria=criteria, targets=targets)
