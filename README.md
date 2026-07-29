# Sendico Pokemon Reference Matcher

This version uses each PriceCharting product page as the single source of truth
for target identity, canonical artwork, card/set metadata and market price.

## User-editable files

- `data/watchlist.yaml`: PriceCharting links and Sendico search phrases.
- `data/run_limits.yaml`: listing, Gemini and token limits.
- `data/search_criteria.yaml`: visual-match and Discord alert thresholds.

A watchlist card no longer needs manually duplicated card names, set codes,
numbers, rarity or language fields.

## Matching flow

1. Fetch and cache the PriceCharting product page and its main card image.
2. Search Sendico with the watchlist phrases.
3. Compare the reference image against listing overview images with Gemini Flash-Lite.
4. Send a probable-match Discord alert immediately when the probable threshold is reached.
5. Crop card-shaped regions and run a detailed reference-image comparison.
6. Send a confirmed alert immediately when the confirmed threshold is reached.
7. Send the normal completion summary when the run ends.

The scheduled workflow runs hourly. GitHub Actions is near-real-time rather than
instant; alerts are sent while each run is processing, not held until completion.

## Required GitHub secrets

- `GEMINI_API_KEY`
- `DISCORD_WEBHOOK_URL`
