# V8.2 GitHub-only migration

The repository is currently a partial hybrid:
- some v8 modules replaced old modules;
- old tests and legacy Python files are still present; and
- several required v8 modules/tests were not uploaded.

Do not run the normal Tests workflow until all files below are uploaded.

## Upload folder by folder in GitHub

GitHub's ordinary upload does not remove old files and may not preserve every
nested folder when a ZIP is selected. Open each destination folder in GitHub
and upload the files from the matching local folder.

### Repository root
Upload:
- `.gitignore`
- `README.md`
- `config.yaml`
- `pyproject.toml`
- `requirements.txt`

### `data/`
Upload:
- `reference_cache.json`
- `run_limits.yaml`
- `search_criteria.yaml`
- `seen.json`
- `watchlist.yaml`

Create `data/reference_images/.gitkeep`.

### `src/pokemon_deal_bot/`
Upload all 11 files in the package's matching folder.

### `tests/`
Upload all 8 test files in the package's matching folder.

### `.github/workflows/`
Upload:
- `scan.yml`
- `tests.yml`
- `v8-migrate.yml`

Commit the upload with:

`Upload complete PriceCharting matcher v8.2`

## Run the one-time migration

1. Open **Actions**.
2. Select **Finalize v8.2 Migration**.
3. Select **Run workflow**.
4. Run it on `main`.

The workflow checks that all v8.2 files exist before deleting anything. It then:
- removes legacy source modules and tests;
- removes obsolete migration documentation and caches;
- runs compilation and pytest;
- commits the cleanup; and
- removes itself.

After it succeeds, run the normal Tests workflow and then a manual scanner run.
