from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_runtime_files_are_not_part_of_v8_package():
    legacy = [
        "src/pokemon_deal_bot/groq_model_pool.py",
        "src/pokemon_deal_bot/updated_main.py",
        "src/pokemon_deal_bot/tier2_vision.py",
        "apply_v5_update.py",
        "verify_v5_update.py",
    ]
    assert all(not (ROOT / path).exists() for path in legacy)


def test_workflow_runs_hourly_and_persists_reference_cache():
    workflow = (ROOT / ".github/workflows/scan.yml").read_text(encoding="utf-8")
    assert 'cron: "7 * * * *"' in workflow
    assert "data/reference_cache.json" in workflow
    assert "actions/cache@v4" in workflow
    assert "data/reference_images" in workflow
    assert "data/price_cache.json" not in workflow
    assert "python -m pokemon_deal_bot.main" in workflow
