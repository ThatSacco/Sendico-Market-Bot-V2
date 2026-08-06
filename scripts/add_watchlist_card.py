"""Append a new card to data/watchlist.yaml without hand-editing YAML.

Built for the "Add Watchlist Card" GitHub Actions workflow (workflow_dispatch
inputs render as a form in the GitHub mobile app), but works the same way
run locally:

    python scripts/add_watchlist_card.py \\
        --url "https://www.pricecharting.com/game/pokemon-japanese-x/card-1" \\
        --terms "term one
term two"

Appends a text block matching the file's existing style instead of
round-tripping the whole file through a YAML dumper, so the header comment
and every other card's formatting are left untouched. The result is
re-parsed through the real load_watchlist() before anything is trusted, so
a bad append can never land silently.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokemon_deal_bot.config import (  # noqa: E402
    derive_id_from_url,
    load_watchlist,
    validate_pricecharting_url,
)

DEFAULT_WATCHLIST_PATH = REPO_ROOT / "data/watchlist.yaml"


def _quote(value: str) -> str:
    """A YAML double-quoted scalar uses JSON-compatible escaping."""

    return json.dumps(value, ensure_ascii=False)


def parse_search_terms(raw: str) -> list[str]:
    """One term per line (the form's natural shape); tolerate stray commas
    too, since mobile keyboards sometimes autocorrect line breaks."""

    parts = [
        piece.strip()
        for line in raw.splitlines()
        for piece in line.split(",")
    ]
    terms = [part for part in parts if part]
    # De-dupe while preserving order -- pasting the same term twice is an
    # easy mistake on a phone.
    return list(dict.fromkeys(terms))


def resolve_id(pricecharting_url: str, custom_id: str) -> str:
    custom_id = (custom_id or "").strip()
    if not custom_id:
        return derive_id_from_url(pricecharting_url)
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", custom_id).strip("_").lower()
    if not cleaned:
        raise ValueError(f"id {custom_id!r} has no usable characters after cleanup")
    return cleaned


def build_card_block(*, card_id: str, pricecharting_url: str, terms: list[str]) -> str:
    lines = [
        f"  - id: {card_id}",
        "    active: true",
        f"    pricecharting_url: {_quote(pricecharting_url)}",
        "    searches:",
    ]
    for term in terms:
        lines.append(f"      - term: {_quote(term)}")
        lines.append("        active: true")
    return "\n".join(lines) + "\n"


def append_block(watchlist_path: Path, block: str) -> None:
    content = watchlist_path.read_text(encoding="utf-8")
    if not content.endswith("\n"):
        content += "\n"
    watchlist_path.write_text(content + "\n" + block, encoding="utf-8")


def add_card(
    watchlist_path: Path,
    *,
    pricecharting_url: str,
    search_terms_raw: str,
    custom_id: str = "",
) -> str:
    """Validate, append, and re-validate. Returns the new card's id.

    Raises ValueError on any problem *before* touching the file -- ID
    collisions and duplicate URLs are checked against the watchlist as it
    exists today, and the file on disk is only written once every check
    has passed.
    """

    pricecharting_url = pricecharting_url.strip()
    validate_pricecharting_url(pricecharting_url)

    terms = parse_search_terms(search_terms_raw)
    if not terms:
        raise ValueError("At least one search term is required")

    card_id = resolve_id(pricecharting_url, custom_id)

    existing = load_watchlist(watchlist_path)
    if any(target.id == card_id for target in existing):
        raise ValueError(f"Watchlist id {card_id!r} already exists")
    if any(target.pricecharting_url == pricecharting_url for target in existing):
        raise ValueError("That PriceCharting URL is already on the watchlist")

    block = build_card_block(card_id=card_id, pricecharting_url=pricecharting_url, terms=terms)
    append_block(watchlist_path, block)

    # If this raises, the append produced something the app itself
    # wouldn't accept -- better to fail loudly here than commit it.
    load_watchlist(watchlist_path)
    return card_id


def resolve_and_summarize(
    card_id: str, watchlist_path: Path, root: Path, *, transport=None
) -> str:
    """Actually fetch the new card's PriceCharting reference.

    Catches a bad URL (typo, removed listing, a page PriceCharting's own
    parser can't handle) immediately instead of letting it surface silently
    on the next scan run. Also downloads and caches the reference image, so
    the next scan doesn't have to. Returns a short Markdown summary meant
    for GITHUB_STEP_SUMMARY. ``transport`` exists for tests -- production
    calls always use a real network transport.
    """

    from pokemon_deal_bot.reference import PriceChartingReferenceClient

    target = next(
        (t for t in load_watchlist(watchlist_path) if t.id == card_id), None
    )
    if target is None:
        raise ValueError(f"{card_id!r} is not on the watchlist after appending it")

    client = PriceChartingReferenceClient(root, transport=transport)
    try:
        reference = client.resolve(target)
    finally:
        client.close()

    price = (
        f"${reference.ungraded_usd:,.2f}"
        if reference.ungraded_usd is not None
        else "not found"
    )
    terms = ", ".join(search.term for search in target.searches)
    return (
        f"### Added `{target.id}`\n\n"
        f"- **Card**: {reference.display_name}\n"
        f"- **PriceCharting ungraded price**: {price}\n"
        f"- **Search terms**: {terms}\n"
    )


def main() -> int:
    # Search terms and card names are Japanese text; a Windows console
    # (cp1252, not UTF-8) would otherwise raise UnicodeEncodeError on print.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="PriceCharting product page URL")
    parser.add_argument(
        "--terms",
        required=True,
        help="Search phrases, one per line (or comma-separated)",
    )
    parser.add_argument(
        "--id",
        default="",
        help="Optional custom watchlist id; auto-derived from the URL if omitted",
    )
    parser.add_argument("--watchlist", default=str(DEFAULT_WATCHLIST_PATH))
    parser.add_argument("--root", default=str(REPO_ROOT), help="Repo root, for data/ paths")
    parser.add_argument(
        "--skip-resolve",
        action="store_true",
        help="Skip the live PriceCharting fetch (offline/local dry runs)",
    )
    args = parser.parse_args()

    try:
        card_id = add_card(
            Path(args.watchlist),
            pricecharting_url=args.url,
            search_terms_raw=args.terms,
            custom_id=args.id,
        )
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    print(f"Added watchlist card: {card_id}")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"card_id={card_id}\n")

    if args.skip_resolve:
        return 0

    try:
        summary = resolve_and_summarize(card_id, Path(args.watchlist), Path(args.root))
    except Exception as exc:
        print(
            f"::error::Added {card_id!r} but could not resolve its PriceCharting "
            f"reference: {exc}",
            file=sys.stderr,
        )
        return 1

    print(summary)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
