from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import SendicoListing


class StateStore:
    def __init__(self, path: Path, *, max_listings: int = 5000) -> None:
        self.path = path
        self.max_listings = max(100, int(max_listings))
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 2, "listings": {}}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and "listings" in value:
                return value
            if isinstance(value, dict):
                return {"version": 2, "listings": value}
        except Exception:
            pass
        return {"version": 2, "listings": {}}

    @staticmethod
    def listing_fingerprint(listing: SendicoListing, signature: str) -> str:
        """Fingerprint the lightweight search-card state seen before hydration."""

        payload = {
            "signature": signature,
            "code": listing.code,
            "title": listing.title,
            "price_yen": listing.price_yen,
            "thumbnail": listing.image_urls[0] if listing.image_urls else "",
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def should_process(self, listing: SendicoListing, signature: str) -> bool:
        return self.should_process_fingerprint(
            listing.code,
            self.listing_fingerprint(listing, signature),
        )

    def should_process_fingerprint(
        self,
        listing_code: str,
        fingerprint: str,
    ) -> bool:
        current = self.data["listings"].get(listing_code) or {}
        return current.get("fingerprint") != fingerprint

    def alert_sent(
        self,
        listing_code: str,
        target_id: str,
        stage: str,
        fingerprint: str,
    ) -> bool:
        record = self.data["listings"].get(listing_code) or {}
        alerts = record.get("alerts") or {}
        return alerts.get(f"{target_id}:{stage}") == fingerprint

    def record_alert(
        self,
        listing_code: str,
        target_id: str,
        stage: str,
        fingerprint: str,
    ) -> None:
        """Buffer the alert in memory; call save() to flush to disk."""

        record = self.data["listings"].setdefault(listing_code, {})
        record.setdefault("alerts", {})[f"{target_id}:{stage}"] = fingerprint
        record["updated_at"] = datetime.now(timezone.utc).isoformat()

    def mark_processed(
        self,
        listing: SendicoListing,
        signature: str,
        outcome: str,
        *,
        fingerprint: str | None = None,
    ) -> None:
        """Buffer the outcome in memory; call save() to flush to disk."""

        previous = self.data["listings"].get(listing.code) or {}
        self.data["listings"][listing.code] = {
            "fingerprint": fingerprint
            or self.listing_fingerprint(listing, signature),
            "last_outcome": outcome,
            "alerts": previous.get("alerts") or {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _prune(self) -> None:
        listings = self.data.setdefault("listings", {})
        excess = len(listings) - self.max_listings
        if excess <= 0:
            return
        oldest = sorted(
            listings,
            key=lambda code: str((listings.get(code) or {}).get("updated_at") or ""),
        )
        for code in oldest[:excess]:
            listings.pop(code, None)

    def save(self) -> None:
        self._prune()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                self.data,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
