from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

LOGGER = logging.getLogger(__name__)

_FIELDS = ("jpy_to_aud", "usd_to_aud", "fetched_at", "source")


@dataclass(slots=True)
class FxRates:
    jpy_to_aud: float
    usd_to_aud: float
    fetched_at: str
    source: str  # "live" or "manual_fallback"


class FxRateClient:
    """Fetches JPY/USD -> AUD rates from Frankfurter.app, caching on disk.

    Falls back to a stale cached rate, and finally to the configured manual
    rates, if the live fetch fails -- a scan must never crash for lack of an
    FX rate.
    """

    def __init__(
        self,
        root: Path,
        *,
        manual_jpy_to_aud: float,
        manual_usd_to_aud: float,
        cache_hours: int = 6,
        timeout_seconds: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.root = root
        self.cache_path = root / "data/fx_cache.json"
        self.manual_jpy_to_aud = manual_jpy_to_aud
        self.manual_usd_to_aud = manual_usd_to_aud
        self.cache_hours = max(1, int(cache_hours))
        self.client = httpx.Client(
            timeout=timeout_seconds, follow_redirects=True, transport=transport
        )

    def close(self) -> None:
        self.client.close()

    def _load_cache(self) -> dict | None:
        if not self.cache_path.exists():
            return None
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and all(key in value for key in _FIELDS):
                return value
        except Exception:
            pass
        return None

    def _save_cache(self, record: dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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

    def fetch(self) -> FxRates:
        cached = self._load_cache()
        if cached and cached.get("source") == "live" and self._fresh(cached):
            return FxRates(**{key: cached[key] for key in _FIELDS})

        try:
            response = self.client.get(
                "https://api.frankfurter.app/latest",
                params={"base": "AUD", "symbols": "JPY,USD"},
            )
            response.raise_for_status()
            payload = response.json()
            rates = payload["rates"]
            record = {
                "jpy_to_aud": 1.0 / float(rates["JPY"]),
                "usd_to_aud": 1.0 / float(rates["USD"]),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "source": "live",
            }
            self._save_cache(record)
            return FxRates(**record)
        except Exception as exc:
            LOGGER.warning("Live FX rate fetch failed, falling back: %s", exc)
            if cached:
                return FxRates(**{key: cached[key] for key in _FIELDS})
            return FxRates(
                jpy_to_aud=self.manual_jpy_to_aud,
                usd_to_aud=self.manual_usd_to_aud,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                source="manual_fallback",
            )
