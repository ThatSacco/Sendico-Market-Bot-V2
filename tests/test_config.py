from pathlib import Path
import yaml
import pytest

from pokemon_deal_bot.config import build_search_plan, load_config, load_watchlist

ROOT = Path(__file__).resolve().parents[1]


def test_repository_configuration_loads():
    config = load_config(ROOT / "config.yaml")
    assert config.targets
    assert config.targets[0].pricecharting_url.startswith("https://www.pricecharting.com/game/")
    assert config.run_limits["token_budget"]["max_total_tokens_per_run"] == 150000


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
