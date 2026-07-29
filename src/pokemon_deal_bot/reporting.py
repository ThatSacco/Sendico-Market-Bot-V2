from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import ScanStats


REPORT_FIELDS = [
    "listing_code",
    "listing_url",
    "target_id",
    "stage",
    "batch",
    "confidence",
    "match_score",
    "same_card",
    "model",
    "candidate_labels",
    "evidence",
    "conflicts",
]


def write_report(
    root: Path,
    stats: ScanStats,
    rows: list[dict[str, Any]],
) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    payload = {"stats": stats.as_dict(), "matches": rows}
    (reports / "latest.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (reports / "latest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in REPORT_FIELDS})
