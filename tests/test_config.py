from pathlib import Path
import yaml
import pytest

from pokemon_deal_bot.config import (
    build_search_plan,
    load_config,
    load_watchlist,
    validate_run_limits,
)

ROOT = Path(__file__).resolve().parents[1]


def _base_run_limits(token_budget: dict) -> dict:
    return {
        "version": 1,
        "search": {
            "results_per_term": 80,
            "total_listings_per_run": 200,
            "raw_links_per_term": 120,
        },
        "screening": {
            "max_listings_per_run": 200,
            "images_per_batch": 4,
            "max_batches_per_listing": 3,
            "max_image_dimension_px": 1400,
            "jpeg_quality": 80,
        },
        "detailed_analysis": {
            "max_listings_per_run": 100,
            "max_images_downloaded": 20,
            "max_card_crops_per_listing": 40,
            "images_per_batch": 12,
            "max_batches_per_listing": 4,
            "max_image_dimension_px": 1600,
            "jpeg_quality": 86,
            "contact_sheet_columns": 4,
        },
        "token_budget": token_budget,
        "state": {"max_seen_listings": 5000},
    }


def test_repository_configuration_loads():
    config = load_config(ROOT / "config.yaml")
    assert config.targets
    assert config.targets[0].pricecharting_url.startswith("https://www.pricecharting.com/game/")
    # 0 means unlimited -- removed deliberately so a full diagnostic run
    # isn't cut short before covering every candidate.
    assert config.run_limits["token_budget"]["max_total_tokens_per_run"] == 0


def test_validate_run_limits_allows_zero_token_budget_as_unlimited():
    validate_run_limits(
        _base_run_limits(
            {
                "max_total_tokens_per_run": 0,
                "reserve_per_request": 5000,
                "max_requests_per_run": 0,
            }
        )
    )


def test_validate_run_limits_still_rejects_reserve_at_or_above_a_real_ceiling():
    with pytest.raises(ValueError, match="reserve_per_request must be below"):
        validate_run_limits(
            _base_run_limits(
                {
                    "max_total_tokens_per_run": 5000,
                    "reserve_per_request": 5000,
                    "max_requests_per_run": 0,
                }
            )
        )


def test_watchlist_requires_only_reference_and_searches(tmp_path):
    path = tmp_path / "watchlist.yaml"
    path.write_text("""
cards:
  - id: test
    pricecharting_url: https://www.pricecharting.com/game/pokemon-japanese-test/card-1
    searches:
      - term: テスト まとめ売り
        mode: focused_lot
""", encoding="utf-8")
    targets = load_watchlist(path)
    assert targets[0].id == "test"
    assert not hasattr(targets[0], "card_number")


def test_minimal_watchlist_derives_id_and_accepts_bare_search_strings(tmp_path):
    path = tmp_path / "watchlist.yaml"
    path.write_text("""
cards:
  - pricecharting_url: "https://www.pricecharting.com/game/pokemon-japanese-black-bolt/victini-97"
    searches:
      - "ブラックボルト まとめ売り"
      - "sv11B まとめ売り"
""", encoding="utf-8")
    targets = load_watchlist(path)
    assert targets[0].id == "victini_97"
    assert [search.term for search in targets[0].searches] == [
        "ブラックボルト まとめ売り",
        "sv11B まとめ売り",
    ]


def test_search_plan_groups_shared_terms():
    config = load_config(ROOT / "config.yaml")
    plan = build_search_plan(config.targets)
    assert all(task.term for task in plan)
    assert all(task.target_ids for task in plan)


def test_non_pricecharting_reference_is_rejected(tmp_path):
    path = tmp_path / "watchlist.yaml"
    path.write_text("""
cards:
  - id: bad
    pricecharting_url: https://example.com/card
    searches:
      - term: bad
""", encoding="utf-8")
    with pytest.raises(ValueError, match="PriceCharting"):
        load_watchlist(path)
