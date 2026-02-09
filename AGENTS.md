# Newsroom Agent and Cron System

## 1. Overview

The newsroom uses a two-tier agent architecture: LLM planners (run via cron) select stories, then a deterministic runner executes the story-writing pipeline. This separation ensures LLM creativity for story selection while maintaining reliability for execution.

The key insight behind the architecture is that OpenClaw cron turns are LLM turns -- they are great for planning but unreliable for long-running background processes. Background processes started with `exec ... &` do not reliably persist after the cron turn ends. The solution splits work into:

- **Planners** (LLM, cron-triggered): evaluate news candidates, pick stories, write job JSON files.
- **Runner** (deterministic Python, spawned on-demand): picks up job files, orchestrates story writing via sub-agents, validates results, publishes to Discord.

The entire pipeline from news discovery to publication flows through these stages:

```
Cron trigger
  -> Inputs script (Brave/GDELT/RSS fetch + LLM clustering)
  -> Planner LLM (story selection + job file creation)
  -> Runner (source extraction + sub-agent spawn + polling + validation + Discord publish)
```

---

## 2. Cron Agent: Daily Planner

**Schedule:** `0 7,19 * * *` (07:00 and 19:00 UK time, Europe/London timezone)

**Timeout:** 900 seconds (15 minutes)

**What it does:**

1. Runs the inputs script (`scripts/newsroom_daily_inputs.py`) to fetch news links from Brave Search API (global + UK queries), GDELT, and RSS feeds, then clusters them into events using Gemini Flash LLM calls, and outputs ranked candidates.
2. Runs a recent-context check against Discord to identify what has already been posted in the last 24 hours.
3. Selects exactly 5 distinct story candidates from the inputs, ensuring:
   - Each has a primary URL and 1-2 supporting URLs from different domains.
   - Each has a concrete verifiable anchor (decision, number, filing, ruling, etc.).
   - Stories are within the last 36-48 hours.
   - Titles are in Traditional Chinese (Cantonese phrasing).
4. Writes a run folder under `workspace/jobs/discord-multi-YYYY-MM-DD-HH-mm/` containing:
   - `run.json` (run_job_v1 schema)
   - `story_01.json` through `story_05.json` (story_job_v1 schema, status = PLANNED)
5. Launches the deterministic runner in the background via `newsroom_write_run_job.py --launch-runner`.

**Hard constraints on the planner LLM:**

- Do NOT use `web_search`, `web_fetch`, `browser`, or `message` tools.
- Maximum 4 `exec` calls total: (1) recent-context, (2) inputs/brave primary, (3) inputs/brave fallback if needed, (4) write-all-files.
- Do NOT run pip installs or debug commands.
- Final output MUST be only a JSON object with `status`, `run_dir`, and `stories[]`.

**Environment requirements:**

- `OPENCLAW_HOME` environment variable (or defaults to `~/.openclaw`).
- Brave API key configured in `data/newsroom/brave_key_state.json`.
- Gemini auth profiles in `agents/main/agent/auth-profiles.json` (used by GeminiClient for clustering).

**Prompt file:** `newsroom/prompts/planner_daily_v1.md`

---

## 3. Cron Agent: Hourly Planner

**Schedule:** `30 * * * *` (every hour at XX:30, Europe/London timezone)

**Timeout:** 600 seconds (10 minutes)

**What it does:**

This planner focuses exclusively on breaking and developing news. It runs at half past every hour and creates 0, 1, or 2 story jobs.

1. Runs `scripts/newsroom_hourly_inputs.py` with a single `exec` call. This fetches from Brave (single query), RSS, clusters pending links, and selects up to 3 candidates prioritizing:
   - Developments first (stories with a `parent_event_id`).
   - Freshness (most recently created events first).
   - Link count (more coverage = stronger signal).
2. Picks up to 2 candidates from the output, preferring:
   - `age_minutes <= 120` (breaking) or `<= 360` (developing).
   - Non-financial news first (US Stocks/Crypto/Precious Metals are fallback only).
3. If no candidate qualifies, the planner returns `{"status":"no_post","reason":"no_qualifying_breaking_or_developing_story"}` and creates no files.
4. If stories are selected, writes job files and launches the runner, all in a single `exec` call via `newsroom_write_run_job.py --launch-runner`.

**Hard constraints:**

- Maximum 2 `exec` calls total: (1) inputs, (2) write jobs + launch runner.
- Same tool prohibitions as the daily planner (no web_search, web_fetch, browser, message).
- `no_post` is a valid and expected outcome -- no stories is better than low-quality stories.

**Breaking/developing definitions:**

- **Breaking:** key development happened or confirmed within the last 2 hours relative to the run time.
- **Developing:** ongoing story with a material new update within the last 6 hours.

**Prompt file:** `newsroom/prompts/planner_hourly_v1.md`

---

## 4. Deterministic Runner

The runner (`newsroom/runner.py`, entry point `scripts/newsroom_runner.py`) is a pure Python process that picks up job files written by the planner and executes the story-writing pipeline without any LLM planning decisions.

### Launch mechanism

The runner is spawned in the background by `newsroom_write_run_job.py --launch-runner`. It runs as a detached process (`start_new_session=True`) via `uv run python3 scripts/newsroom_runner.py --path <run_dir> --summary-only`. stdout/stderr go to `runner_background.out.log` and `runner_background.err.log` in the run directory.

No OS-level crontab entry is needed -- the runner is spawned on-demand by the planner's write-run script.

### Job discovery

The runner discovers jobs by scanning a run directory for `story_*.json` files with `schema_version: "story_job_v1"`. It skips jobs that are already in terminal states (SUCCESS, FAILURE, RESCUED, SKIPPED).

When given `--jobs-root` instead of `--path`, it scans all subdirectories under the jobs root for pending work via `discover_jobs_under()`.

### Run group execution

Jobs within a run directory are processed as a group via `run_group()`:

1. Loads `run.json` for group-level defaults (concurrency, stagger, timeout, dm_targets, stop_run_on_failure).
2. Sorts jobs deterministically by path.
3. Starts jobs up to the configured concurrency limit, with optional stagger delay between spawns.
4. Polls active jobs via `_poll_and_advance()` until all complete.
5. Writes a `run_summary.json` artifact for audit.
6. Sends DM summaries to configured targets if any.

### Per-story pipeline

For each story job, the runner executes:

1. **Schema validation** -- validates against `newsroom/schemas/story_job_v1.schema.json`.
2. **Dedupe check** -- checks both marker-file-based deduplication (by primary URL and event key) and semantic title similarity against recent Discord posts.
3. **Source pack extraction** -- deterministically fetches and extracts text from primary and supporting URLs (no LLM tokens). Backfills additional supporting URLs from the local 48-hour news pool if needed.
4. **Source sufficiency gate** -- requires at least 2 usable sources with 400+ characters of on-topic text. If insufficient, the job is marked SKIPPED_MISSING_SOURCES and no Discord thread is created.
5. **Asset preparation** -- builds market data snapshots, chart PNGs (for finance stories), OG image downloads, and generated infographic/card images.
6. **Discord container creation** -- creates the title post and thread (for `agent_posts` mode) or defers until after the worker succeeds (for `script_posts` mode).
7. **Worker sub-agent spawn** -- spawns a sub-agent via `sessions_spawn` with the rendered prompt template and source pack.
8. **Polling loop** -- polls the sub-agent session via `sessions_history` at the configured interval (default 5 seconds) until:
   - A valid result JSON is found in the assistant's messages, OR
   - The timeout expires, OR
   - Stale-poll detection triggers (24 consecutive polls with messages but no parseable result).
9. **Result validation** -- runs the result through a Python validator module (specified by `validator_id`).
10. **Result repair** -- if validation fails on minor structural issues, attempts deterministic repair via `result_repair.py`.
11. **Rescue** -- if validation still fails and `recover.enabled` is true, spawns a rescue sub-agent with the error context and the worker's partial output. Max 1 rescue attempt by default.
12. **Publishing** -- in `script_posts` mode, the runner publishes the validated draft body, read-more URLs, and image attachments to the Discord thread. Message content is split into Discord-safe chunks (max 1900 chars).
13. **Event recording** -- marks the event as posted in the events database (`news_pool.sqlite3`) so future planners do not re-select it.
14. **Dedupe marker** -- writes a marker file under `logs/newsroom/dedupe/` to prevent re-posting within 24 hours.

### Publisher modes

- **`agent_posts`**: The worker sub-agent posts directly to Discord during its execution. The runner creates the Discord thread before spawning the worker.
- **`script_posts`**: The worker returns a JSON draft containing the report body, read-more URLs, and optional image prompts. The runner publishes deterministically after validation succeeds. This avoids empty threads when workers fail.

### Retry and error handling

- Gateway tool calls use exponential backoff retry (default 4 attempts, `base_sleep * 2^attempt`).
- Sub-agent spawn retries up to 3 times with backoff.
- On failure, the runner posts a failure notice to the Discord thread and marks the job with `final_status: "FAILURE"`.
- Malformed job files are "jailed" (renamed with `.jailed` suffix) to prevent infinite retry loops.

---

## 5. Worker Sub-Agents

The runner spawns worker sub-agents for story writing via the OpenClaw gateway's `sessions_spawn` tool. Workers are full LLM sessions that receive a rendered prompt template and pre-extracted source material.

### Spawning

Workers are spawned with:

- **task**: the fully rendered prompt template (with source pack, assets, and input JSON injected).
- **label**: `nr:<story_id>:<phase>:<hash>` (deterministic, short).
- **agentId**: typically `"main"` (configurable per job via `spawn.agent_id`).
- **runTimeoutSeconds**: the job's timeout + 60 seconds of slack.
- **cleanup**: `"keep"` -- sessions are retained so the runner can reliably read the final result JSON.

### Monitoring

The runner monitors workers by polling `sessions_history` for the child session key. It extracts the result JSON from the last assistant message containing a JSON object with a `"status"` field. If the gateway API returns empty messages, the runner falls back to reading the session JSONL file directly from disk.

### Timeout handling

If a worker exceeds its deadline, the runner transitions to rescue mode (if enabled) or marks the job as FAILURE. The runner polls before checking the deadline to avoid a race condition where the worker completes between the last poll and the deadline check.

### Rescue sub-agents

When a worker fails (timeout, validation failure, empty output), the runner can spawn a rescue sub-agent with:

- The rescue prompt template (specified by `recover.rescue_prompt_id`, typically `news_rescue_script_v1`).
- The original worker's error type and message injected as template variables.
- A separate timeout (default 600 seconds).
- Maximum 1 rescue attempt by default (configurable via `recover.max_rescue_attempts`).

If the rescue sub-agent produces a valid result, the job's final status is set to `RESCUED`.

---

## 6. Inputs Pipeline

Both the daily and hourly planners rely on inputs scripts that run a multi-stage pipeline before the LLM planner makes any decisions.

### Pipeline stages

1. **Pool update** (`scripts/news_pool_update.py`): Fetches links from the Brave News API with configurable queries, result count, freshness filter, and offset rotation. Upserts links into the SQLite pool database (`data/newsroom/news_pool.sqlite3`). The daily pipeline runs two calls (global + UK-specific queries); the hourly pipeline runs one. Rate-limiting state is tracked per `state-key` to enforce minimum intervals between API calls.

2. **GDELT update** (`scripts/gdelt_pool_update.py`, daily only): Fetches additional links from the GDELT API for broader global coverage.

3. **RSS update** (`scripts/rss_pool_update.py`): Fetches links from configured RSS feeds for sources not well-covered by Brave.

4. **LLM clustering** (`newsroom/event_manager.py:cluster_all_pending()`): Groups unclustered links into events using one-link-per-prompt Gemini Flash calls. Each link is presented alongside existing fresh events; the LLM either assigns it to an existing event or creates a new one. The clustering prompt asks for: event category (from a fixed list of 12 categories), jurisdiction, English summary, and whether this link represents a development of an existing event.

5. **Post-clustering merge** (`newsroom/event_manager.py:merge_events()`): A second LLM pass that merges events covering the same underlying story but not caught by the initial clustering.

6. **Candidate selection** (`newsroom/news_pool_db.py:get_daily_candidates()` or `get_hourly_candidates()`):
   - **Daily**: selects up to 15 events ranked by link_count DESC, with category balance (max 2 finance, max 3 per category), HK slot guarantee.
   - **Hourly**: selects up to 3 events prioritizing developments (parent_event_id IS NOT NULL), freshness, and link count. Second picks must differ in category from the first.

7. **Candidate enrichment** (`_enrich_candidates()`): adds backward-compatible fields for the job writer.

### Candidate format

Each candidate in the output JSON contains:

| Field | Type | Description |
|-------|------|-------------|
| `i` | int | 1-based index (assigned by the inputs script) |
| `id` | int | Event ID in the SQLite events table |
| `title` | string | Event title (from primary link or LLM summary) |
| `description` | string | English summary from LLM clustering |
| `suggested_category` | string | One of the 12 fixed categories |
| `primary_url` | string | Best representative URL for the event |
| `supporting_urls` | list[string] | Up to 4 additional URLs from linked articles |
| `age_minutes` | int or null | Minutes since best_published_ts |
| `event_id` | int | Same as `id`, used by runner to mark events as posted |
| `link_count` | int | Number of links clustered into this event |
| `domains` | list[string] | Unique domains contributing to this event |
| `parent_event_id` | int or null | Non-null for development events |
| `event_key` | string | `"event:<id>"` for deduplication |
| `semantic_event_key` | string | Same as event_key |
| `suggest_flags` | list[string] | Auto-inferred flags (breaking, developing) |

### Output files

- Daily: `data/newsroom/daily_inputs_last.json`
- Hourly: `data/newsroom/hourly_inputs_last.json`

---

## 7. Job File Format

### story_job_v1

Each story job is a JSON file conforming to the `story_job_v1` schema (`newsroom/schemas/story_job_v1.schema.json`).

**Required top-level fields:**

| Field | Description |
|-------|-------------|
| `schema_version` | Must be `"story_job_v1"` |
| `run` | `{ run_id, trigger, run_time_uk }` |
| `story` | Story metadata (see below) |
| `destination` | `{ platform, title_channel_id, thread_name_template }` |
| `spawn` | `{ prompt_id, agent_id, publisher_mode, input_mapping }` |
| `monitor` | `{ poll_seconds, timeout_seconds, result_json_required }` |
| `post` | `{ on_success_dm_targets, on_failure_dm_targets }` |
| `recover` | `{ enabled, rescue_prompt_id, max_rescue_attempts, rescue_timeout_seconds }` |
| `validation` | `{ validator_id, stop_run_on_failure }` |
| `state` | Mutable runtime state (see below) |
| `result` | Final result container (see below) |

**story object:**

- `story_id`: e.g. `"story_01"`
- `content_type`: e.g. `"news_deep_dive"`
- `category`: one of the 12 fixed categories
- `title`: Traditional Chinese (Cantonese phrasing)
- `primary_url`: main source URL
- `supporting_urls`: list of corroborating URLs
- `concrete_anchor`: one short verifiable fact
- `flags`: list of strings (e.g. `["breaking"]`, `["developing"]`)
- `dedupe_key`: URL-based or event-based deduplication key
- `event_id`: integer event ID from clustering (used to mark events as posted)

**Status lifecycle:**

```
PLANNED -> DISPATCHED -> SUCCESS
                      -> FAILURE -> (rescue) -> RESCUED
                                             -> FAILURE
                      -> SKIPPED (duplicate / policy / missing sources)
```

- `PLANNED`: job file created by planner, awaiting runner pickup.
- `DISPATCHED`: runner has spawned a worker sub-agent.
- `SUCCESS`: worker produced a valid result that passed validation.
- `FAILURE`: worker failed (timeout, validation failure, empty output) and rescue also failed or was disabled.
- `RESCUED`: initial worker failed but rescue sub-agent produced a valid result.
- `SKIPPED`: job skipped due to deduplication, policy, or insufficient source material.

### run_job_v1

Each run directory contains a `run.json` file with group-level configuration:

```json
{
  "schema_version": "run_job_v1",
  "run": { "run_id": "<folder_name>", "trigger": "cron_daily", "run_time_uk": "..." },
  "destination": { "platform": "discord" },
  "runner": {
    "concurrency": 3,
    "stagger_seconds": 60,
    "default_timeout_seconds": 900,
    "dm_targets": [],
    "stop_run_on_failure": false
  }
}
```

Valid triggers: `cron_daily`, `cron_hourly`, `manual`.

---

## 8. Prompt Templates

### Template rendering

Prompt templates are Markdown files stored under `newsroom/prompts/`. They use `{{VARIABLE_NAME}}` placeholders that are replaced at render time by `_render_template()`.

Standard variables injected by the runner:

| Variable | Source |
|----------|--------|
| `{{INPUT_JSON}}` | Worker input object serialized as JSON (includes story metadata, source pack, and assets) |
| `{{SKILL_PATH}}` | Path to the news-reporter skill file |
| `{{WORKER_ERROR_TYPE}}` | Error type from failed worker (rescue prompts only) |
| `{{WORKER_ERROR_MESSAGE}}` | Error message from failed worker (rescue prompts only) |
| `{{OPENCLAW_HOME}}` | OpenClaw home directory path (from env or default) |
| `{{RUN_TIME_UK}}` | UK-time string for the current run (planner prompts) |

Unreplaced placeholders cause a `PromptRegistryError` at render time to prevent silent prompt bugs.

### Prompt registry

The file `newsroom/prompt_registry.json` maps prompt IDs to template files and validators:

```json
{
  "prompts": {
    "news_reporter_script_v1": {
      "template_path": "newsroom/prompts/news_reporter_script_v1.md",
      "validator_id": "news_reporter_script_v1"
    }
  },
  "validators": {
    "news_reporter_script_v1": {
      "type": "python",
      "path": "newsroom/validators/news_reporter_script_v1.py"
    }
  },
  "content_types": {
    "news_deep_dive": {
      "default_prompt_id": "news_reporter_v2_2_inline"
    }
  }
}
```

### Category-based prompt selection

The file `newsroom/prompt_policy.py` provides `prompt_id_for_category()`, which maps story categories to specialized prompt templates:

| Category | Prompt ID |
|----------|-----------|
| Global News | `news_reporter_script_default_v1` |
| Politics | `news_reporter_script_politics_v1` |
| UK Parliament / Politics | `news_reporter_script_uk_parliament_v1` |
| UK News | `news_reporter_script_uk_news_v1` |
| Hong Kong News | `news_reporter_script_hk_news_v1` |
| Hong Kong Entertainment | `news_reporter_script_hk_entertainment_v1` |
| Entertainment | `news_reporter_script_entertainment_v1` |
| Sports | `news_reporter_script_sports_v1` |
| AI | `news_reporter_script_ai_v1` |
| US Stocks / Crypto / Precious Metals | `news_reporter_script_finance_v1` |

Unknown categories fall back to `news_reporter_script_default_v1`.

The `newsroom_write_run_job.py` script calls `prompt_id_for_category()` when constructing each story job's `spawn.prompt_id` field.

---

## 9. Adding a New Cron Job

To wire up a new planner schedule, use `openclaw cron add`. For example, to add the daily planner:

```bash
openclaw cron add \
  --name "Newsroom Daily Planner (07:00,19:00 UK)" \
  --agent newsroom \
  --session isolated \
  --cron "0 7,19 * * *" \
  --tz "Europe/London" \
  --disabled \
  --timeout-seconds 900 \
  --message "$(cat /path/to/newsroom/prompts/planner_daily_v1.md)"
```

Key flags:

- `--agent newsroom`: runs in the newsroom agent context.
- `--session isolated`: each cron execution gets its own session (no state leakage between runs).
- `--tz "Europe/London"`: cron expression interpreted in UK time.
- `--disabled`: creates the job in disabled state so you can review before enabling.
- `--timeout-seconds`: hard limit on the LLM planner turn.

The `--message` flag receives the full prompt template. Use `$(cat ...)` to inject the prompt file content. The template uses `{{OPENCLAW_HOME}}` which is resolved by the cron system at runtime.

To add the hourly planner:

```bash
openclaw cron add \
  --name "Newsroom Hourly Planner (xx:30 UK)" \
  --agent newsroom \
  --session isolated \
  --cron "30 * * * *" \
  --tz "Europe/London" \
  --disabled \
  --timeout-seconds 600 \
  --message "$(cat /path/to/newsroom/prompts/planner_hourly_v1.md)"
```

After adding, enable the job via `openclaw cron enable --id <job_id>`.

No OS-level crontab entry is needed for the runner. The runner is spawned on-demand by the `newsroom_write_run_job.py --launch-runner` script, which the planner calls in its final `exec` step.

---

## 10. Troubleshooting

### 429 rate limits (Brave API)

The inputs scripts enforce minimum intervals between Brave API calls via `--min-interval-seconds` (default 900 seconds = 15 minutes). State is tracked per `--state-key` in `data/newsroom/brave_key_state.json`. If you see rate limit errors:

- Check the key state file to see when the last successful call was made.
- Increase `--min-interval-seconds` if hitting limits frequently.
- The `--key-label` flag selects between different Brave API keys if multiple are configured.

### 429 rate limits (Gemini / LLM)

The `GeminiClient` handles 429 errors with automatic profile rotation and cooldowns:

- **RATE_LIMIT_EXCEEDED (per-minute):** retries after a short delay (default 10 seconds).
- **QUOTA_EXHAUSTED (per-day):** rotates to the next auth profile; applies a 1-hour cooldown on the exhausted profile.
- If all profiles are exhausted, clustering calls will fail. The inputs scripts catch these errors and continue with whatever clustering was completed.

### `no_post` outcomes (hourly planner)

A `no_post` result from the hourly planner is normal and expected. It means no breaking or developing story met the quality thresholds. Common reasons:

- No candidates with `age_minutes <= 360`.
- All candidates were already posted in the last 24 hours (deduplication).
- Candidates lacked concrete anchors.

This is not an error condition. The system is designed to publish nothing rather than publish low-quality content.

### Runner not launching

If stories are written but the runner never executes:

1. Check that `--launch-runner` was passed to `newsroom_write_run_job.py`.
2. Check `runner_background.out.log` and `runner_background.err.log` in the run directory.
3. Verify `uv` is available in PATH (the runner is launched via `uv run python3`).
4. Check that the gateway is running and accessible (the runner needs gateway connectivity for `sessions_spawn`, `sessions_history`, and `message` tool calls).
5. Manually trigger the runner: `uv run python3 scripts/newsroom_runner.py --path <run_dir>`.

### Missing environment variables

Required:

- `OPENCLAW_HOME` (or defaults to `~/.openclaw`) -- used to resolve all relative paths.
- Gemini auth profiles at `agents/main/agent/auth-profiles.json` -- needed for LLM clustering in the inputs pipeline.
- Brave API keys in `data/newsroom/brave_key_state.json` -- needed for news discovery.
- Gateway configuration -- needed by the runner for sub-agent spawning and Discord messaging.

### Job stuck in DISPATCHED

If a job remains in DISPATCHED status indefinitely:

1. The runner may have crashed. Check `runner_background.err.log`.
2. The worker sub-agent may have timed out silently. Check the story log in `logs/newsroom/<run_id>/`.
3. Stale locks: the runner uses file-based locks with a configurable TTL (default 3600 seconds). Locks older than the TTL are treated as stale and can be reclaimed by a new runner invocation.
4. Re-run: `uv run python3 scripts/newsroom_runner.py --path <run_dir>` -- the runner will detect the existing DISPATCHED state and resume monitoring.

### Empty Discord threads

The `script_posts` publisher mode (used by default in the current prompt policy) delays thread creation until after the worker successfully produces a validated draft. This prevents empty threads from appearing when workers fail. If you still see empty threads, check whether the job is using `agent_posts` mode instead, which creates the thread before the worker runs.

### Validation failures

Check the job file's `result.errors[]` array for specific validation error messages. Common issues:

- Missing required fields in the worker's result JSON (e.g. `draft.body`, `draft.read_more_urls`).
- Report body too short (CJK character count below threshold).
- The runner attempts deterministic repair via `result_repair.py` before escalating to rescue.

### Debugging a specific story

Each story produces a JSONL log at `logs/newsroom/<run_id>/<story_id>.jsonl`. Events include: `job_seen`, `spawn_ok`, `monitor_poll`, `result_found`, `validation_ok`, `discord_publish_progress`, and error events. The final job JSON (with all state mutations) is persisted in the story file under the run directory.
