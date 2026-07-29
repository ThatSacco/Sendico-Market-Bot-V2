# Search Criteria Guide

The bot now has three user-editable control files:

| File | Purpose |
|---|---|
| `data/watchlist.yaml` | Cards, PriceCharting links and Sendico search terms |
| `data/run_limits.yaml` | Search volume, Gemini volume and token budget |
| `data/search_criteria.yaml` | Filtering strictness and deal qualification |

## To allow more discovered listings through

Edit `data/search_criteria.yaml`:

```yaml
discovery:
  prefilter_watchlist_relevance: true
  allow_query_only_candidates: true

lot:
  require_strong_lot_evidence: true

screening:
  minimum_target_probability: 0.30

detailed_analysis:
  minimum_card_confidence: 0.90
  minimum_target_confidence: 0.80
```

A looser diagnostic profile is:

```yaml
discovery:
  prefilter_watchlist_relevance: false
  allow_query_only_candidates: true

lot:
  require_strong_lot_evidence: false

screening:
  minimum_target_probability: 0.20

detailed_analysis:
  minimum_card_confidence: 0.85
  minimum_target_confidence: 0.70
```

Use the loose profile temporarily. It can materially increase token use and
unrelated listings.

## A zero-listing run

`Listings found: 0` occurs before filtering. Adjust active terms in
`data/watchlist.yaml`; changing criteria cannot create marketplace results.

For Ampharos EX, useful focused searches include:

```yaml
searches:
  - term: "XY7 まとめ売り"
    mode: focused_lot
    active: true
  - term: "バンデットリング まとめ売り"
    mode: focused_lot
    active: true
  - term: "デンリュウEX まとめ売り"
    mode: focused_lot
    active: true
```

Avoid making every term so exact that Sendico returns no results. The exact card
number can remain in an inactive `exact` search while focused lot terms discover
unnamed cards in photos.

## Seen-state behaviour

The search-criteria content is included in the scan signature. Editing
`data/search_criteria.yaml` therefore permits listings to be reconsidered under
the new criteria without manually emptying `data/seen.json`.
