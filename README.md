# openclaw-newsroom

Automated AI newsroom -- LLM clustering, multi-source news pool, Discord publishing.

## Architecture Overview

openclaw-newsroom is a pipeline that transforms raw news links into polished, clustered stories published to Discord. The pipeline has four stages:

1. **News pool ingestion** -- Multiple source adapters (Brave News API, GDELT DOC 2.0, curated RSS/Atom feeds) continuously fetch links and upsert them into a local SQLite news pool database with deduplication and TTL expiry.

2. **Clustering and deduplication** -- An LLM-powered event manager (Gemini Flash) clusters incoming links into coherent events using one-link-per-prompt calls. Token-based and cross-lingual entity matching provides fast pre-filtering before LLM classification. A post-clustering merge pass collapses near-duplicate events.

3. **LLM story writing** -- Selected events are handed to Gemini (Flash primary, Pro fallback) with category-aware prompt routing. The runner produces structured JSON output (headline, body, sources, images) validated against JSON schemas, with automatic result repair for malformed LLM responses.

4. **Discord publishing** -- Finished stories are published to Discord channels via the OpenClaw Gateway, including embedded images, charts, and source attribution. The system supports both hourly breaking-news runs and daily digest runs with category balancing.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) -- Detailed technical architecture, data flow diagrams, and component deep-dives
- [AGENTS.md](AGENTS.md) -- Cron agent system, planner/runner architecture, job file formats
- [PROMPTS.md](PROMPTS.md) -- Prompt template system, category routing, validators
- [CONTRIBUTING.md](CONTRIBUTING.md) -- How to add news sources, prompts, and validators

## Prerequisites

- **Python 3.12+**
- **OpenClaw Gateway** -- The newsroom publishes to Discord through the OpenClaw Gateway HTTP API. The gateway is a separate service (not included in this repository) that provides authenticated Discord access via a token-based tool invocation interface. The newsroom works in dry-run mode without a running gateway.
- **[uv](https://docs.astral.sh/uv/)** -- Package manager used for dependency management and virtual environment setup.

## Installation

```bash
git clone https://github.com/fol2/openclaw-newsroom.git openclaw-newsroom
cd openclaw-newsroom
uv sync
```

This creates a virtual environment and installs all dependencies from `pyproject.toml`.
To include optional chart dependencies (Pillow): `uv sync --extra charts`.
To include dev dependencies (pytest, pytest-cov): `uv sync --dev`.

## Configuration

Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for LLM calls (clustering, story writing). |
| `BRAVE_SEARCH_API_KEYS` | Recommended | Multiple Brave Search API keys (comma/newline separated). Supports `label:key` entries for rotation. |
| `BRAVE_SEARCH_API_KEY` | Yes (if `BRAVE_SEARCH_API_KEYS` unset) | Single Brave Search API key (legacy). |
| `OPENCLAW_GATEWAY_TOKEN` | Yes | Bearer token for the OpenClaw Gateway HTTP API (Discord publishing). |
| `OPENCLAW_HOME` | No | Override the OpenClaw home directory. Defaults to `~/.openclaw`. |
| `OPENCLAW_GATEWAY_HTTP_URL` | No | Gateway HTTP endpoint. Defaults to `http://127.0.0.1:3000`. |
| `GEMINI_PROFILE_ORDER` | No | Comma-separated list of google-gemini-cli OAuth profile names for round-robin rotation. |
| `GEMINI_AUTH_PROFILES` | No | Path to `auth-profiles.json` for OAuth-based Gemini access (alternative to API key). |
| `GEMINI_API_KEY_ONLY_UNTIL` | No | ISO 8601 timestamp after which the system switches from API key to OAuth profiles. |
| `NANO_BANANA_SCRIPT` | No | Path to an image generation script invoked via `uv run`. |

### RSS Feeds (Optional)

To override the built-in RSS/Atom feed list, copy the example config to `newsroom/rss_feeds.yaml`:

```bash
cp newsroom/examples/rss_feeds.example.yaml newsroom/rss_feeds.yaml
```

## Quick Start

Run the newsroom in dry-run mode (no Discord publishing, no API keys needed for the runner itself):

```bash
# 1. Populate the news pool with Brave News results
uv run python scripts/news_pool_update.py --db data/newsroom/news_pool.sqlite3

# 2. Run the hourly planner (cluster + select events)
uv run python scripts/newsroom_hourly_inputs.py --db data/newsroom/news_pool.sqlite3

# 3. Run the story writer + publisher in dry-run mode
uv run python scripts/newsroom_runner.py --dry-run
```

## Scripts Reference

| Script | Description |
|---|---|
| `newsroom_runner.py` | Main entry point -- discovers pending story jobs, runs the LLM write pipeline, publishes to Discord. Supports `--dry-run`. |
| `newsroom_hourly_inputs.py` | Hourly planner -- fetches pool links, LLM-clusters into events, selects 1-3 events for the hourly run. |
| `newsroom_daily_inputs.py` | Daily planner -- same clustering pipeline, selects 10-15 events with category balance and HK guarantee. |
| `newsroom_write_run_job.py` | Creates a story job JSON file from event data (used by hourly/daily planners). |
| `news_pool_update.py` | Fetches news from Brave News API and upserts links into the pool database. |
| `gdelt_pool_update.py` | Fetches articles from GDELT DOC 2.0 API (free, no auth) and upserts into the pool. |
| `rss_pool_update.py` | Fetches articles from curated RSS/Atom feeds and upserts into the pool. |
| `news_pool_status.py` | Diagnostic tool -- shows pool statistics, cluster counts, and link freshness. |

All scripts live in the `scripts/` directory and are invoked as standalone Python scripts.

## Testing

```bash
uv run pytest newsroom/tests/ -v
```

## Key Modules

| Module | Description |
|---|---|
| `runner.py` | Core newsroom runner -- job discovery, LLM story generation, Discord posting, dedup against recent titles. |
| `event_manager.py` | Event-centric LLM clustering -- groups pool links into events using Gemini Flash one-link-per-prompt calls, with a post-clustering merge pass. |
| `gemini_client.py` | Lightweight Gemini REST client -- OAuth profile round-robin, Flash/Pro fallback, rate-limit handling. |
| `news_pool_db.py` | SQLite news pool database -- link storage, TTL expiry, source tracking, pool run logging. |
| `brave_news.py` | Brave News API adapter -- search, URL normalization, multi-key rotation, rate-limit recording. |
| `gdelt_news.py` | GDELT DOC 2.0 API adapter -- article fetching and normalization. |
| `rss_news.py` | RSS/Atom feed parser -- feed config loading, article extraction via lxml. |
| `gateway_client.py` | OpenClaw Gateway HTTP client -- tool invocation for Discord message/embed posting. |
| `story_index.py` | Token-based clustering and ranking -- Jaccard similarity, CJK-aware tokenization, anchor term extraction. |
| `dedupe.py` | Cross-lingual deduplication -- English proper-noun matching across Cantonese/English titles. |
| `source_pack.py` | Source aggregation -- collects and deduplicates article metadata, OG image extraction for embeds. |
| `prompt_policy.py` | Category-to-prompt routing -- maps story categories (finance, general, etc.) to prompt template IDs. |
| `result_repair.py` | LLM output repair -- fixes malformed JSON, truncated responses, and schema violations from Gemini output. |
| `charts.py` | Pure-Python PNG chart renderer -- line charts for market data embeds (no matplotlib dependency). |
| `image_fetch.py` | OG image extractor and downloader -- HTML meta parsing, content-type validation, caching. |
| `market_data.py` | Market data fetcher -- stock/crypto ticker extraction and price lookup for finance stories. |
| `job_store.py` | Job file I/O -- atomic JSON writes, file locking, path jailing, timestamp utilities. |
| `validators/` | JSON schema validation for LLM output and job files. |

## GatewayClient Dependency

The newsroom publishes stories to Discord via `GatewayClient`, which talks to the OpenClaw Gateway over HTTP. The gateway is a separate process that manages Discord bot connections and exposes a tool invocation API.

If the gateway is not running or `OPENCLAW_GATEWAY_TOKEN` is not set, the newsroom still functions in dry-run mode: stories are generated and written to job files but not posted to Discord. This makes local development and testing straightforward without needing the full OpenClaw infrastructure.

## License

[MIT](LICENSE)
