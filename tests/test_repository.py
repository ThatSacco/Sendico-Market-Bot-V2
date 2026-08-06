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


def test_workflow_is_manual_only_and_persists_reference_cache():
    workflow = (ROOT / ".github/workflows/scan.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch" in workflow
    assert "schedule" not in workflow
    assert "cron" not in workflow
    assert "data/reference_cache.json" in workflow
    assert "data/confirmations.json" in workflow
    assert "actions/cache@v4" in workflow
    assert "data/reference_images" in workflow
    assert "data/price_cache.json" not in workflow
    assert "python -m pokemon_deal_bot.main" in workflow


def test_confirmations_workflow_is_manual_only_and_lightweight():
    workflow = (
        ROOT / ".github/workflows/check-confirmations.yml"
    ).read_text(encoding="utf-8")
    assert "workflow_dispatch" in workflow
    assert "schedule" not in workflow
    assert "cron" not in workflow
    assert "--check-confirmations" in workflow
    assert "data/confirmations.json" in workflow
    # No scan happens here, so no need to spend time installing a browser.
    # GEMINI_API_KEY *is* passed (a confirmed probable alert gets its lot
    # valued at reaction time), but that is a handful of requests for the
    # listings a human actually vouched for -- not a scan.
    assert "playwright" not in workflow
    assert "sendico" not in workflow.lower().replace("sendico-pokemon-scan", "")
