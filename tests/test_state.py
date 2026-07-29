from pokemon_deal_bot.models import SendicoListing
from pokemon_deal_bot.state import StateStore


def test_seen_state_changes_with_scan_signature(tmp_path):
    store = StateStore(tmp_path / "seen.json")
    listing = SendicoListing(
        "m12345678",
        "https://sendico.test/m12345678",
        "Lot",
        1000,
        ["https://x/m12345678_1.jpg"],
    )
    assert store.should_process(listing, "a")
    store.mark_processed(listing, "a", "done")
    assert not store.should_process(listing, "a")
    assert store.should_process(listing, "b")


def test_discovery_fingerprint_remains_stable_after_hydration(tmp_path):
    store = StateStore(tmp_path / "seen.json")
    listing = SendicoListing(
        "m12345678",
        "https://sendico.test/m12345678",
        "Search title",
        1000,
        ["https://x/m12345678_1.jpg"],
    )
    fingerprint = store.listing_fingerprint(listing, "signature")
    listing.title = "Hydrated title"
    listing.image_urls.append("https://x/m12345678_2.jpg")
    store.mark_processed(
        listing,
        "signature",
        "done",
        fingerprint=fingerprint,
    )
    assert not store.should_process_fingerprint(listing.code, fingerprint)


def test_alert_stages_are_deduplicated_separately(tmp_path):
    store = StateStore(tmp_path / "seen.json")
    store.record_alert("m1", "victini", "probable", "p")
    assert store.alert_sent("m1", "victini", "probable", "p")
    assert not store.alert_sent("m1", "victini", "confirmed", "p")


def test_mark_processed_and_record_alert_buffer_until_save(tmp_path):
    path = tmp_path / "seen.json"
    store = StateStore(path)
    listing = SendicoListing(
        "m12345678",
        "https://sendico.test/m12345678",
        "Lot",
        1000,
        ["https://x/m12345678_1.jpg"],
    )
    store.mark_processed(listing, "a", "done")
    store.record_alert("m12345678", "victini", "probable", "p")
    assert not path.exists()
    store.save()
    assert path.exists()
    persisted = StateStore(path)
    assert not persisted.should_process(listing, "a")
    assert persisted.alert_sent("m12345678", "victini", "probable", "p")


def test_seen_state_is_pruned_to_prevent_repository_bloat(tmp_path):
    store = StateStore(tmp_path / "seen.json", max_listings=100)
    for index in range(110):
        listing = SendicoListing(
            f"m{index:08d}",
            f"https://sendico.test/m{index:08d}",
            f"Lot {index}",
            1000,
            [f"https://x/m{index:08d}_1.jpg"],
        )
        store.mark_processed(listing, "signature", "done")
    store.save()
    assert len(store.data["listings"]) == 100
