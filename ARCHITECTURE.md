# OpenClaw Newsroom -- Technical Architecture

This document describes the internals of the openclaw-newsroom system: an
automated news pipeline that ingests links from multiple sources, clusters them
into events using LLM classification, generates Cantonese-language news reports,
and publishes them to Discord.

---

## 1. System Overview

```
                         +-----------+
                         | Brave API |
                         | GDELT/RSS |
                         +-----+-----+
                               |
                        (1) INGESTION
                               |
                               v
                    +----------+----------+
                    |   News Pool (SQLite) |
                    |  links, events,      |
                    |  article_cache,      |
                    |  fetch_state         |
                    +----------+----------+
                               |
                        (2) CLUSTERING
                         (Gemini Flash)
                               |
                               v
                    +----------+----------+
                    |   Events Table       |
                    |  category, status,   |
                    |  jurisdiction,       |
                    |  parent/child tree   |
                    +----------+----------+
                               |
                    +----------+----------+
                    | Candidate Selection  |
                    | (daily/hourly quota, |
                    |  category balance,   |
                    |  HK guarantee)       |
                    +----------+----------+
                               |
                        (3) WRITING
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
    +-------------------+            +-------------------+
    |  Source Pack       |            |  Story Runner      |
    |  (fetch + extract  |            |  (prompt render,   |
    |   + on-topic score)|            |   LLM/agent call,  |
    +-------------------+            |   validate, repair,|
                                     |   rescue)          |
                                     +-------------------+
                               |
                        (4) PUBLISHING
                               |
                               v
                    +----------+----------+
                    |  Discord Gateway     |
                    |  (title post,        |
                    |   thread creation,   |
                    |   body + images,     |
                    |   read-more links)   |
                    +----------+----------+
                               |
                               v
                    +----------+----------+
                    |  Discord Channel     |
                    +---------------------+
```

The pipeline runs on two cadences driven by cron:

- **Hourly**: ingests fresh links, clusters them, selects up to 3 development-
  priority candidates, writes and publishes.
- **Daily**: broader candidate selection (up to 15 events), category-balanced
  with finance caps and HK jurisdiction guarantees.

---

## 2. Data Flow

The lifecycle of a news link from ingestion to publication:

```
Brave API response
    |
    v
normalize_url()  -->  PoolLink dataclass
    |
    v
NewsPoolDB.upsert_links()
    |  (UNIQUE on norm_url; seen_count incremented on update)
    v
links table  [id, url, norm_url, domain, title, description,
              age, page_age, first_seen_ts, last_seen_ts,
              seen_count, event_id=NULL, published_at_ts]
    |
    v  (event_id IS NULL => unassigned)
cluster_all_pending()
    |  for each unassigned link, oldest first:
    |    build_clustering_prompt(link, fresh_events)
    |    gemini.generate_json(prompt)
    |    parse_clustering_response() -> action
    |
    |  action = "assign"      => link.event_id = existing event
    |  action = "new_event"   => create new event row, link.event_id = new id
    |  action = "development" => create child event, link.event_id = child id
    |
    v
events table  [id, parent_event_id, category, jurisdiction,
               summary_en, status='new'|'active'|'posted',
               link_count, expires_at_ts, ...]
    |
    v  (merge_events: post-clustering LLM dedup pass)
    |
    v
get_daily_candidates() / get_hourly_candidates()
    |  -> enriched candidate dicts with supporting_urls, domains, age_minutes
    |
    v
Planner script emits story_job_v1 JSON files
    |
    v
NewsroomRunner.run_job()
    |
    +---> build_source_pack()   [fetch HTML, extract text, cache, score on-topic]
    +---> render prompt template with {{SOURCES_PACK}}, {{TITLE}}, etc.
    +---> execute worker (inline LLM or spawn sub-agent)
    +---> extract result JSON from response
    +---> validate(result_json, job) via Python validator module
    +---> repair_result_json() if validation fails
    +---> re-validate after repair
    +---> if still failing: rescue pass with rescue prompt
    +---> publish to Discord (title post + thread + body chunks + images)
    +---> mark event as posted (status='posted')
    +---> write dedupe marker
```

---

## 3. News Pool

### SQLite Schema (v5)

The pool database lives at `data/newsroom/news_pool.sqlite3` and uses WAL mode
for concurrent read/write access.

**Core tables:**

| Table           | Purpose                                          |
|-----------------|--------------------------------------------------|
| `meta`          | Key-value store (schema_version, etc.)            |
| `links`         | Ingested news links with dedup on `norm_url`      |
| `events`        | LLM-classified event clusters (v5 event-centric)  |
| `fetch_state`   | Per-query pagination/timing state for collectors   |
| `article_cache` | Extracted article text cache (keyed on `norm_url`) |
| `pool_runs`     | Audit log of ingestion runs and request counts     |

**`links` columns:**
`id`, `url`, `norm_url` (UNIQUE), `domain`, `title`, `description`, `age`,
`page_age`, `first_seen_ts`, `last_seen_ts`, `seen_count`, `last_query`,
`last_offset`, `last_fetched_at_ts`, `event_id` (FK to events), `published_at_ts`.

**`events` columns:**
`id`, `parent_event_id` (self-FK for development chains), `category`,
`jurisdiction`, `summary_en`, `development`, `title`, `primary_url`,
`link_count`, `best_published_ts`, `status` (CHECK: 'new'|'active'|'posted'),
`created_at_ts`, `updated_at_ts`, `expires_at_ts`, `posted_at_ts`,
`thread_id`, `run_id`, `model`.

### TTL and Expiry

- Default TTL: **48 hours** (`_DEFAULT_TTL_SECONDS = 48 * 3600`).
- Events have `expires_at_ts`; each link assignment bumps it forward by 48h.
- `get_fresh_events()` returns only events where `expires_at_ts > now`.
- Link pruning: `prune_links(cutoff_ts)` deletes links older than cutoff.
- Article cache pruning: `prune_article_cache(cutoff_ts)` deletes stale rows.

### Source Adapters

**Brave News API** (`brave_news.py`):

- Endpoint: `https://api.search.brave.com/res/v1/news/search`
- Multi-key rotation: keys loaded from `BRAVE_SEARCH_API_KEYS` env var (or single-key `BRAVE_SEARCH_API_KEY`).
- Key state persisted in `data/newsroom/brave_key_state.json` (tracks usage,
  rate limit events per key_id = sha256 prefix).
- `select_brave_api_key()` skips keys that are temporarily cooled down or exhausted.
- `record_brave_rate_limit()` persists the last-seen quota headers and can mark a key as exhausted until reset.
- `record_brave_cooldown()` records a short-term cooldown after transient errors (e.g. 429/503) so callers can rotate keys.
- Each key has a label (e.g., "free", "paid") parsed from `label:key` format.
- URL normalization strips fragments and tracking params (`utm_*`, `fbclid`, etc.).

**GDELT and RSS**: supported as additional source adapters through the same
`upsert_links()` interface on `NewsPoolDB`.

### Pool Runs Audit

Every ingestion run logs a `pool_runs` row with:
`run_ts`, `state_key`, `window_hours`, `should_fetch`, `query`, `offset_start`,
`pages`, `count`, `freshness`, `requests_made`, `results`, `inserted`,
`updated`, `pruned`, `pruned_articles`, `notes`.

`sum_requests_made(since_ts)` provides rate-limit budget tracking across keys.

---

## 4. Event Clustering Pipeline

### Architecture

Event clustering uses a **one-link-per-prompt LLM classification** approach
via Gemini Flash. Each unassigned link is individually classified against
the current set of fresh events.

### Flow (`event_manager.py`)

```
cluster_all_pending(db, gemini, max_links=100)
    |
    for each unassigned link (oldest first):
    |
    +---> db.get_fresh_events()         # refresh after each link
    +---> build_clustering_prompt()     # one link + up to 50 root events
    +---> gemini.generate_json()        # Gemini Flash call
    +---> parse_clustering_response()   # validate action dict
    +---> cluster_link()                # execute action:
    |       assign:      db.assign_link_to_event()
    |       new_event:   db.create_event() + assign
    |       development: db.create_event(parent_event_id) + assign
    |
    +---> 3s delay between calls (rate limiting)
    +---> early exit after 3 consecutive failures
```

### Clustering Prompt Structure

The prompt presents:
1. The link's title and description (first 300 chars).
2. Up to 50 fresh root events with their children (developments) shown as
   indented `+-- [id=N] "development label"` entries.
3. Three possible actions: ASSIGN, DEVELOPMENT, NEW EVENT.
4. Category list (12 categories) and jurisdiction codes.
5. Cross-language matching instruction.

### Post-Clustering Merge Pass

After initial classification, `merge_events()` performs a second LLM-based
deduplication pass:

1. Group root events (status in 'new'|'active') by category.
2. Sort alphabetically within each category (similar summaries cluster).
3. Use a sliding window with overlap (batch_size=50, overlap=10) to handle
   boundary cases.
4. For each batch, ask the LLM to identify duplicate groups.
5. Winner selection: most links, tiebreak by oldest `created_at_ts`.
6. `db.merge_events_into()`: atomically reassign links, re-parent children,
   recalculate counters, and delete loser events.

### Token-Based Pre-Filtering (story_index.py)

The legacy `story_index.py` module provides deterministic token-based clustering
used for dedup key generation and category suggestion. It is not the primary
clustering mechanism (LLM clustering replaced it) but remains active for:

- **Tokenization**: Latin word extraction + CJK bigram segmentation.
- **Stopword filtering**: 130+ English stopwords + CJK boilerplate removal.
- **Document frequency filtering**: tokens appearing in >5% of docs are dropped.
- **Anchor terms**: numbers (>=3 digits), tickers (2-6 uppercase), capitalized
  entity words, CJK chunks (2-6 chars). Generic tickers (UK, US, AI) excluded.
- **Jaccard similarity**: `jaccard(a, b) = |a & b| / |a | b|` for clustering.
- **Cluster event keys**: `sha1(sorted(core_terms))[:20]` for stable dedup keys.
- **Category suggestion**: keyword/pattern matching against 12 categories.

---

## 5. LLM Client Architecture

### GeminiClient (`gemini_client.py`)

A lightweight REST client for Gemini that operates independently of the full
agent framework. Two API paths are supported:

```
                    +-------------------+
                    |   GeminiClient    |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
     OAuth SSE Path                 API Key Path
     (cloudcode-pa)                 (generativelanguage)
              |                             |
     +--------+--------+          +--------+--------+
     | Profile 1 (Flash)|         | gemini-2.0-flash |
     | Profile 1 (Pro)  |         +--------+--------+
     | Profile 2 (Flash)|                  |
     | Profile 2 (Pro)  |           (fallback only)
     | ...               |
     +------------------+
```

### OAuth Profile Rotation

- Profiles loaded from `agents/main/agent/auth-profiles.json`.
- Provider filter: only `google-gemini-cli` profiles.
- Profile order: `GEMINI_PROFILE_ORDER` env var or `order` section in JSON.
- At init, profiles are **re-sorted by remaining flash quota** (descending)
  via the `retrieveUserQuota` endpoint.
- Round-robin across profiles for each `generate()` call.
- Profiles with expired tokens are auto-refreshed using OAuth2 refresh tokens.
- Expiry buffer: profiles expiring within 60 seconds are skipped.

### Flash/Pro Fallback Chain

For each profile attempt:
1. Try `gemini-3-flash-preview` (fast, lower cost).
2. If Flash fails, try `gemini-3-pro-preview` (higher quality).
3. If both fail, move to the next profile.

### Rate Limit Handling (429 Classification)

The client mirrors `gemini-cli`'s `classifyGoogleError` logic:

| Signal | Classification | Cooldown |
|--------|---------------|----------|
| `QuotaFailure` + `PerDay`/`Daily` quotaId | **Terminal** (daily exhaustion) | Parsed from response or 1h default |
| `ErrorInfo` reason=`QUOTA_EXHAUSTED` | **Terminal** | Parsed or 1h |
| `ErrorInfo` reason=`RATE_LIMIT_EXCEEDED` | **Retryable** (per-minute) | Parsed delay or 10s |
| `RetryInfo.retryDelay` present | **Retryable** | Parsed duration |
| `PerMinute` in quotaId/metadata | **Retryable** | 60s |
| "reset after XhYmZs" in body | **Terminal** | Parsed seconds |
| `Retry-After` header | **Retryable** | Header value |
| Unknown 429 | **Retryable** | 10s default |

Terminal 429s set a per-profile cooldown (`_exhausted_until`); the profile is
skipped until the cooldown expires. Retryable 429s use shorter cooldowns.

### API Key Fallback

If all OAuth profiles are exhausted or fail:
1. Fall back to `GEMINI_API_KEY` (from env or `.env` file).
2. Uses `generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent`.
3. Non-streaming, simpler request format.

**API-key-only mode**: set `GEMINI_API_KEY_ONLY_UNTIL=<ISO timestamp>` in `.env`
to skip OAuth entirely until the given time (useful when all OAuth quotas are
exhausted).

### JSON Extraction

`generate_json()` wraps `generate()` and extracts the first JSON object from
the response text. Handles markdown code fences and brace-matching extraction.

---

## 6. Story Runner Pipeline

### Entry Point

`NewsroomRunner` in `runner.py` is the core orchestrator. It processes
`story_job_v1` JSON files through a deterministic pipeline.

### Job Discovery

Jobs are JSON files with schema `story_job_v1` containing:
- `run`: run_id, trigger (cron_hourly, cron_daily), run_time_uk.
- `story`: story_id, content_type, category, title, primary_url,
  supporting_urls, concrete_anchor, flags, dedupe_key, event_id.
- `destination`: platform (discord), title_channel_id.
- `spawn`: prompt_id, agent_id, publisher_mode.
- `state`: status (PLANNED -> RUNNING -> DONE/SKIPPED), discord state, worker
  state, rescue state.
- `result`: final_status, worker_result_json, rescue_result_json, errors.
- `validation`: validator_id.

Legacy/minimal planner output is auto-coerced into `story_job_v1` via
`_coerce_story_job_format()`.

### Pipeline Phases

```
1. LOCK         acquire file lock (prevents concurrent runners on same job)
2. DEDUPE       check dedupe marker + semantic Discord title match
3. SOURCE PACK  fetch/extract/cache article text from URLs
4. PROMPT       resolve prompt_id -> template, render with variables
5. WORKER       execute LLM call (inline generate_json or spawn sub-agent)
6. PARSE        extract JSON result from worker response
7. VALIDATE     run Python validator module
8. REPAIR       deterministic fixes for common validation failures
9. RE-VALIDATE  check if repairs resolved all errors
10. RESCUE      if still failing, run rescue prompt (simpler, more constrained)
11. PUBLISH     post to Discord (title, thread, body chunks, images, read-more)
12. RECORD      mark event as posted, write dedupe marker, unlock
```

### Worker Execution Modes

- **Inline LLM** (`publisher_mode=script_posts`): The runner calls
  `gemini.generate_json()` directly with the rendered prompt. The runner
  handles all Discord publishing deterministically.
- **Spawn sub-agent** (`publisher_mode=agent_posts`): The runner spawns an
  OpenClaw agent session via the gateway, passes the prompt, polls for
  completion, and extracts the result JSON from session messages.

### JSON Schema Validation

Jobs are validated against JSON schemas:
- `newsroom/schemas/story_job_v1.schema.json`
- `newsroom/schemas/run_job_v1.schema.json`

Using the `jsonschema` library. Invalid jobs raise `JobSchemaError`.

### Retry and Rescue

If the worker's output fails validation (even after repair):
1. The runner switches to `phase=rescue`.
2. A rescue prompt template is rendered with the original output + error list.
3. A fresh LLM call generates a corrected result.
4. The rescue result is validated with the same validator.
5. If rescue also fails, the job is marked FAILED with error details.

Gateway tool invocations (`message.send`, `message.read`) use exponential
backoff retry: 4 attempts with `sleep = base * 2^attempt` (base=1s).

---

## 7. Prompt System

### Architecture

```
prompt_policy.py          prompt_registry.json          prompts/*.md
     |                          |                            |
     |  category ->             |  prompt_id ->              |  {{VAR}}
     |  prompt_id               |  template_path             |  placeholders
     |                          |  validator_id              |
     v                          v                            v
+----------+           +--------------+            +----------------+
| "Sports" | --------> | "news_       | ---------> | news_reporter_ |
|          |  mapping  |  reporter_   |  file path | script_sports_ |
|          |           |  script_     |            | v1.md          |
|          |           |  sports_v1"  |            +----------------+
+----------+           +--------------+
                              |
                              | validator_id
                              v
                       +--------------+
                       | "news_       |
                       |  reporter_   | ---------> validators/
                       |  script_v1"  |  file path  news_reporter_
                       +--------------+             script_v1.py
```

### prompt_policy.py

Deterministic mapping from story category to prompt_id:

| Category | Prompt ID |
|----------|-----------|
| US Stocks, Crypto, Precious Metals | `news_reporter_script_finance_v1` |
| Global News | `news_reporter_script_default_v1` |
| Politics | `news_reporter_script_politics_v1` |
| UK Parliament / Politics | `news_reporter_script_uk_parliament_v1` |
| UK News | `news_reporter_script_uk_news_v1` |
| Hong Kong News | `news_reporter_script_hk_news_v1` |
| Hong Kong Entertainment | `news_reporter_script_hk_entertainment_v1` |
| Entertainment | `news_reporter_script_entertainment_v1` |
| Sports | `news_reporter_script_sports_v1` |
| AI | `news_reporter_script_ai_v1` |
| (unknown) | `news_reporter_script_default_v1` |

### prompt_registry.json

Schema version: `prompt_registry_v1`. Contains three sections:

- **`prompts`**: Maps prompt_id to `{template_path, validator_id}`.
  18 prompt definitions covering news reporters (11 category-specific variants),
  rescue prompts (2), inline reporter (2), blog posts, social posts.
- **`validators`**: Maps validator_id to `{type: "python", path}`.
  6 validator modules.
- **`content_types`**: Maps content_type to default prompt_id.
  `news_deep_dive` -> `news_reporter_v2_2_inline`, `blog_post_long`,
  `twitter_post`, `facebook_post`.

### Template Rendering

Templates use `{{VARIABLE}}` placeholders. `_render_template()` replaces all
occurrences and raises `PromptRegistryError` if any unreplaced placeholders
remain (prevents silent prompt bugs).

Common variables: `{{OPENCLAW_HOME}}`, `{{TITLE}}`, `{{SOURCES_PACK}}`,
`{{PRIMARY_URL}}`, `{{SUPPORTING_URLS}}`, `{{CATEGORY}}`, etc.

### Validators

Each validator is a Python module exporting `validate(result_json, job)`.
Validators return a `ValidationResult(ok, errors)` where `errors` is a list of
error code strings.

Validators are dynamically loaded via `importlib.util.spec_from_file_location()`
and cached in `_validator_cache` by validator_id.

**Validator modules:**

| Validator ID | File | Purpose |
|-------------|------|---------|
| `news_reporter_script_v1` | `validators/news_reporter_script_v1.py` | Category-specific reporter output |
| `news_reporter_v2_1_strict` | `validators/news_reporter_v2_1_strict.py` | Strict inline reporter validation |
| `news_rescue_v1` | `validators/news_rescue_v1.py` | Rescue prompt output |
| `news_rescue_script_v1` | `validators/news_rescue_script_v1.py` | Script rescue output |
| `blog_post_long_v1` | `validators/blog_post_long_v1.py` | Blog post validation |
| `social_post_v1` | `validators/social_post_v1.py` | Twitter/Facebook post validation |

Common validation checks (from `news_reporter_script_v1.py`):
- `status` field must be SUCCESS or an allowed failure type.
- `title` must contain sufficient Traditional Chinese characters.
- `draft.body` must be 2700-3300 CJK characters.
- `draft.read_more_urls` must contain 3-5 URLs, including the primary URL and
  at least one URL from a different domain.
- `sources_used` must have at least 2 entries.

---

## 8. Discord Publishing

### Gateway Client (`gateway_client.py`)

Publishing uses the **tool invocation model**: the runner calls Discord
operations through an OpenClaw gateway HTTP API rather than the Discord API
directly.

```
GatewayClient
    |
    +---> POST /tools/invoke
    |       {tool: "message", action: "send", args: {...}, sessionKey: "..."}
    |
    +---> POST /tools/invoke
    |       {tool: "message", action: "read", args: {...}}
    |
    +---> Authorization: Bearer <gateway_token>
```

### Gateway Configuration

- Loaded from `openclaw.json` (`gateway.port`, `gateway.auth.token`).
- Token from `OPENCLAW_GATEWAY_TOKEN` env var or config file.
- HTTP URL from `OPENCLAW_GATEWAY_HTTP_URL` or derived from port.

### Publication Flow (script_posts mode)

```
1. POST title message to title_channel_id
       tool=message, action=send
       message = cleaned title (max 180 chars)
       -> extract title_message_id

2. CREATE thread on the title message
       tool=message, action=send
       target = channel:<title_channel_id>
       threadTarget = message:<title_message_id>
       -> extract thread_id

3. POST body chunks to the thread (1900 char limit each)
       _split_discord_messages() splits on paragraph boundaries
       _normalize_report_body() converts numbered headings to bracketed format
       Each chunk: tool=message, action=send, target=channel:<thread_id>

4. ATTACH images (at most 1 per message)
       Finance categories: chart image on first message, hero on second
       Other categories: hero image (card > infographic > OG image) on first
       Images passed via filePath arg in message.send

5. POST read-more URLs as final message
       "延伸閱讀\n" + newline-separated URLs

6. Track published_message_ids for idempotent resume after crash
```

### Image Attachments

The runner generates/fetches images in priority order:
1. **Card image**: LLM-generated vector-style news summary card (if worker
   provided `card_prompt`). Aspect ratio 2:3 or 3:2, padded to exact ratio.
2. **Infographic**: LLM-generated infographic (if no card). Same aspect
   handling.
3. **OG image**: Downloaded from source article's OpenGraph `og:image` tag.
4. **Chart image**: Rendered line chart PNG (finance categories only).

Image prompts are sanitized: logos replaced with "icons", trademark mentions
removed, brand+color patterns neutralized, aspect ratios normalized.

### Dedup Against Discord Titles

Before publishing, the runner calls `message.read` on the title channel to
fetch recent messages (last 24h, up to 160 messages). It then runs
`best_semantic_duplicate()` against these titles to prevent posting the same
story twice.

---

## 9. Deduplication

The system implements **multi-level deduplication** to prevent duplicate
publications:

### Level 1: Pool-Level URL Dedup

- `links.norm_url` has a UNIQUE constraint.
- `normalize_url()` strips fragments, tracking params (`utm_*`, `fbclid`, etc.).
- Duplicate URLs increment `seen_count` rather than creating new rows.

### Level 2: Cluster-Level Event Dedup

- LLM clustering assigns links to existing events rather than creating
  duplicates (action="assign").
- Post-clustering merge pass (`merge_events()`) identifies and merges
  duplicate events within each category using LLM comparison.
- Token-based dedup keys: `sha1(sorted(core_terms))[:20]` provide stable
  event identifiers across different wordings.

### Level 3: Publication-Level Title Dedup

Three sub-mechanisms:

**a) Dedupe markers** (file-based):
- SHA1 hash of dedupe_key -> `logs/dedupe/<hash>.json`.
- Contains `dedupe_key`, `primary_url`, `thread_id`, `title`.
- TTL: 24 hours (`_DEDUPE_TTL_SECONDS`).
- Primary URL index: in-memory map from normalized URL to marker path.

**b) Semantic title matching** (`dedupe.py`):
- `title_features()` extracts key_tokens and anchor_tokens.
- `semantic_match(a, b)` computes weighted score:
  `score = 0.65 * key_jaccard + 0.35 * anchor_jaccard + bonuses`.
- Bonuses: shared numeric anchors (+0.15/+0.10), 3+ anchor overlaps (+0.10).
- Duplicate thresholds (OR logic):
  - 1+ numeric overlap + 1+ entity overlap + key_jaccard >= 0.07
  - 2+ entity overlaps + key_jaccard >= 0.12
  - key_jaccard >= 0.45
  - score >= 0.52
  - key_jaccard >= 0.34 + 5-8 key overlaps + 1 strong anchor overlap

**c) Cross-lingual matching** (`dedupe.py`):
- `cross_lingual_entity_duplicate()` catches English headlines that duplicate
  existing Cantonese posts.
- Exploits the fact that Cantonese news titles preserve English proper nouns
  (e.g., "Leicester 違反財務規則即時扣 6 分").
- Extracts capitalized words (>=6 chars) from English title, checks if they
  appear in any CJK recent title.
- Requires the proper noun to also appear in `cluster_terms` (if provided)
  to avoid false positives.

---

## 10. Configuration

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENCLAW_HOME` | Root directory (default: `~/.openclaw`) |
| `OPENCLAW_GATEWAY_TOKEN` | Gateway authentication token |
| `OPENCLAW_GATEWAY_HTTP_URL` | Gateway HTTP endpoint override |
| `BRAVE_SEARCH_API_KEYS` | Comma/newline-separated Brave API keys (format: `label:key`) |
| `BRAVE_SEARCH_API_KEY` | Single Brave API key (legacy) |
| `GEMINI_API_KEY` | Fallback Gemini API key |
| `GEMINI_AUTH_PROFILES` | Path to auth-profiles.json override |
| `GEMINI_PROFILE_ORDER` | Comma-separated profile rotation order |
| `GEMINI_OAUTH_CLIENT_ID` | OAuth client ID override |
| `GEMINI_OAUTH_CLIENT_SECRET` | OAuth client secret override |
| `GEMINI_API_KEY_ONLY_UNTIL` | ISO timestamp to skip OAuth until |

### .env File

The system reads `$OPENCLAW_HOME/.env` for keys not set as environment
variables. Simple `KEY=VALUE` parsing (supports quotes, comments with `#`).
Used primarily for `GEMINI_API_KEY` and `GEMINI_API_KEY_ONLY_UNTIL`.

### openclaw.json

Main configuration file at `$OPENCLAW_HOME/openclaw.json`:
- `gateway.port`: Local gateway port number.
- `gateway.auth.token`: Gateway authentication token.

### File Layout

```
$OPENCLAW_HOME/
  .env                              # API keys
  openclaw.json                     # Gateway config
  data/newsroom/
    news_pool.sqlite3               # Main pool database
    brave_key_state.json            # Brave API key rotation state
    brave_news_cache/               # Cached Brave API responses
    hourly_inputs_last.json         # Last hourly run inputs
    daily_inputs_last.json          # Last daily run inputs
  newsroom/
    prompt_registry.json            # Prompt/validator registry
    prompts/                        # Template files (*.md)
    validators/                     # Python validator modules
    schemas/                        # JSON schema files
    examples/                       # Example job files
  agents/main/agent/
    auth-profiles.json              # Gemini OAuth profiles
```

---

## 11. Error Handling

### Result Repair (`result_repair.py`)

Deterministic, no-LLM repair for common validation failures. The repair
function receives the `result_json`, `job`, and list of error codes from
the validator.

**Repair strategies:**

| Error Code Pattern | Repair | Code |
|-------------------|--------|------|
| `*title_traditional_chinese` | Copy Cantonese title from `job.story.title` if it has >=4 CJK chars | `title_from_job` |
| `*read_more_urls_N_to_M` | Fill from job's primary_url + supporting_urls + source_pack URLs; ensure primary URL present; ensure at least one other domain | `read_more_urls` |
| `*read_more_includes_primary_url` | Insert primary_url at position 0 | (part of `read_more_urls`) |
| `*read_more_has_other_domain` | Scan candidates for a URL from a different domain | (part of `read_more_urls`) |
| `*sources_used_min_2` | Fill from on-topic source_pack URLs | `sources_used` |
| `*body_cjk_2700_to_3300` (underflow <=60 chars) | Append a short generic Cantonese closing paragraph | `body_cjk_pad` |
| `*body_cjk_2700_to_3300` (underflow <=800 chars) | Append up to 4 pre-written generic Cantonese filler paragraphs until minimum met | `body_cjk_expand` |

All repairs operate on a deep copy of the result. Repairs are logged as repair
codes (e.g., `["title_from_job", "read_more_urls"]`).

### Rescue Prompts

When repair is insufficient, the runner enters rescue mode:
1. The rescue prompt template includes the original worker output and error list.
2. A fresh LLM call generates a corrected version.
3. Rescue prompts are simpler and more constrained than reporter prompts.
4. Rescue output is validated with a dedicated rescue validator
   (`news_rescue_v1` or `news_rescue_script_v1`) that may have relaxed
   thresholds.

### Failure Types

Validators recognize these failure types (from `news_reporter_script_v1.py`):
`discord_429`, `discord_403`, `discord_error`, `http_429`, `http_503`,
`timeout`, `paywall`, `missing_data`, `validation_failed`, `unknown`.

When the worker returns a non-SUCCESS status with a recognized failure type,
the runner skips publishing and records the failure without entering rescue mode.

### Consecutive Failure Circuit Breaker

Both clustering and merge operations track consecutive LLM failures. After
`max_consecutive_failures` (default: 3), remaining items are skipped with a
warning. This prevents wasting quota when Gemini is systematically failing.

### Source Pack Quality Gates

Before writing, the runner checks `_usable_sources_count()`:
- Prefers `source_pack.stats.on_topic_sources_count` (anchored relevance).
- Falls back to `usable_sources_count` (length-based).
- Falls back to counting sources with `selected_chars >= 400`.
- If insufficient on-topic sources exist, the job is skipped with
  `SKIPPED_MISSING_SOURCES`.

### Idempotent Discord Publishing

Published message IDs are persisted in `state.discord.published_message_ids`.
On crash recovery, the runner resumes from the last successfully published
message index, avoiding duplicate Discord posts.
