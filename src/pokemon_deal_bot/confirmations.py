from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import PendingConfirmation


class ConfirmationStore:
    """Tracks alert messages awaiting a Discord reaction, and their outcome.

    Kept separate from StateStore's dedup bookkeeping since this is a
    slower-moving, user-curated record meant to be read back later (which
    listings were actually verified, and which were false alarms), not an
    internal cache.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        empty = {"version": 1, "pending": {}, "confirmed": {}, "rejected": {}}
        if not self.path.exists():
            return empty
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return empty
        if not isinstance(value, dict):
            return empty
        for bucket in ("pending", "confirmed", "rejected"):
            value.setdefault(bucket, {})
        value.setdefault("version", 1)
        return value

    def add_pending(self, confirmation: PendingConfirmation) -> None:
        self.data["pending"][confirmation.message_id] = asdict(confirmation)

    def pending(self) -> list[PendingConfirmation]:
        return [
            PendingConfirmation(**record) for record in self.data["pending"].values()
        ]

    def resolve(
        self, message_id: str, *, confirmed: bool
    ) -> PendingConfirmation | None:
        """Move a pending confirmation to the confirmed/rejected bucket.

        Returns the resolved record, or None if ``message_id`` wasn't pending
        (already resolved, or never tracked).
        """

        record = self.data["pending"].pop(message_id, None)
        if record is None:
            return None
        bucket = "confirmed" if confirmed else "rejected"
        timestamp_key = "confirmed_at" if confirmed else "rejected_at"
        self.data[bucket][message_id] = {
            **record,
            timestamp_key: datetime.now(timezone.utc).isoformat(),
        }
        return PendingConfirmation(**record)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
