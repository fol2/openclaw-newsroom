# Cleanup Run 2026-02-11

This folder captures a one-off maintenance run to improve the semantic quality of the newsroom event pool after the DB integrity fixes and clustering retrieval tightening.

## Safety

- A point-in-time backup of the live DB was created before applying any changes:
  - `~/.openclaw/data/newsroom/news_pool.sqlite3.bak-20260211-010036`
- Posted-event identity fields were checked before/after (thread/run IDs are hashed in the snapshots to avoid leaking raw IDs).

## What Was Run (Live DB)

Target DB:

- `~/.openclaw/data/newsroom/news_pool.sqlite3`

Steps:

1. Before metrics:
   - `before.json`, `before_summary.md`
   - `posted_snapshot_before.json`
2. Integrity fixer (apply):
   - `uv run python scripts/fix_pool_integrity.py --db-path ~/.openclaw/data/newsroom/news_pool.sqlite3 --apply`
3. One-off recluster (bounded):
   - `PYTHONPATH=. uv run python scripts/recluster_events.py --passes 2 --max-merges 50 --delay 1.0 --min-score 0.25 --db-path ~/.openclaw/data/newsroom/news_pool.sqlite3`
   - Result: `recluster_result.json`
4. One-off merge pass (cross-category + within-category), executed in a bounded mode for reliability:
   - `GEMINI_HTTP_READ_TIMEOUT_SECONDS=240 PYTHONPATH=. uv run python scripts/merge_events_oneoff.py --db-path ~/.openclaw/data/newsroom/news_pool.sqlite3 --batch-size 1`
   - Result: `merge_events_result.json`
5. After metrics:
   - `after.json`, `after_summary.md`
   - `posted_snapshot_after.json` (compared against baseline)

Notes:

- The live DB is continuously updated by the normal ingestion pipeline. Absolute event counts can increase during the run; the metric files still provide an auditable before/after snapshot at the recorded timestamps.

## Live Before/After Highlights

From `before_summary.md` (UTC `2026-02-11T01:00:48Z`):

- Root events total: `1218`
- Avg `link_count` (root events): `2.023`
- Max `link_count` (root events): `98`

From `after_summary.md` (UTC `2026-02-11T01:54:15Z`):

- Root events total: `1249` (note: live ingestion continued during the run)
- Avg `link_count` (root events): `1.946`
- Max `link_count` (root events): `97`
- Posted event identity check: `OK`

## Controlled Snapshot Run (Sandbox Copy)

Because the live DB is mutable, a controlled run was also executed against a frozen copy of the pre-run backup. This isolates the effect of the maintenance steps from ongoing ingestion.

Sandbox DB:

- `/tmp/news_pool_cleanup_sandbox_20260211.sqlite3` (created from the backup above; not committed)

Files:

- `sandbox_before.json`, `sandbox_before_summary.md`
- `sandbox_after.json`, `sandbox_after_summary.md`
- `sandbox_recluster_result.json`
- `sandbox_merge_events_result.json`

Snapshot result:

- Root events total decreased from `1218` to `1215`.
- Posted event identity check: `OK` (baseline and current posted events both `151`).

## Files In This Folder

- Metrics (machine-readable): `before.json`, `after.json`, `sandbox_before.json`, `sandbox_after.json`
- Metrics (human-readable): `before_summary.md`, `after_summary.md`, `sandbox_before_summary.md`, `sandbox_after_summary.md`
- Posted integrity snapshots (hashed): `posted_snapshot_before.json`, `posted_snapshot_after.json`, `sandbox_posted_snapshot_before.json`, `sandbox_posted_snapshot_after.json`
- Maintenance run results: `recluster_result.json`, `merge_events_result.json`, `sandbox_recluster_result.json`, `sandbox_merge_events_result.json`

