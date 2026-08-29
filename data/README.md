# Data directory

Generated data belongs here and is intentionally git-ignored.

Suggested layout:

- `boundaries/` — Nigeria ADM1 and selected-state GeoJSON
- `catalog/` — compact STAC manifests by source/state/month
- `patches/` — optional derived imagery chips or patch-level Parquet
- `features/` — derived state-month feature Parquet
- `nbs/` — NBS resource index, raw downloads, normalized targets

Do not commit large raw satellite imagery.
