# OpenClaw Newsroom -- Architecture Review

**Date**: 2026-02-09
**Scope**: Full repo structural review of `openclaw-newsroom/`
**Codebase snapshot**: 18 modules (11,271 LOC), 9 scripts (2,230 LOC), 19 test files (3,776 LOC), 6 validators (978 LOC), 22 prompt templates, 2 JSON schemas

---

## 1. Current State Assessment

### What is good

1. **Clear pipeline architecture.** The four-stage flow (Ingestion -> Clustering -> Writing -> Publishing) is well-defined and documented in ARCHITECTURE.md. Each stage has a distinct responsibility.

2. **Clean dependency direction.** The module dependency graph is a DAG with no circular imports. `runner.py` is the hub; lower-level modules (brave_news, gdelt_news, rss_news, story_index, etc.) are leaf nodes that do not import each other (with two minor exceptions: source_pack imports from brave_news/image_fetch/news_pool_db/story_index, and dedupe imports from story_index).

3. **SQLite-backed state.** Using a single news_pool.sqlite3 with WAL mode for concurrent reader/writer access is pragmatic and well-suited to the workload.

4. **Job file schema.** The story_job_v1 / run_job_v1 JSON schema system with examples and validation is a solid contract between planner and runner.

5. **Prompt registry.** The prompt_registry.json + validator pairing is a reasonable decoupled design. Adding a new prompt template + validator does not require touching runner.py.

6. **Test suite.** 19 test files covering validators, result repair, clustering, DB operations, schema validation, and integration. The gemini_client tests (1,132 LOC) are thorough.

7. **Deterministic runner.** Separating the LLM planner from the deterministic Python runner was a smart decision, well-articulated in CRON.md.

### What needs attention

1. **runner.py is 4,220 lines.** This is the single largest risk. It contains 7 classes and ~60 methods encompassing prompt rendering, Discord publishing, image generation, sub-agent spawning, deduplication, validation orchestration, source pack preparation, market data, and more. Any change to one concern risks breaking another.

2. **Utility consolidation (mostly done).** Shared validator helpers and CJK counting live in `newsroom/_util.py` (`count_cjk`, `ALLOWED_FAILURE_TYPES`, `ValidationResult`, type helpers). Remaining per-module helpers should stay minimal to avoid reintroducing copy-paste drift.

3. **Scripts use sys.path hacking.** All 9 scripts contain `OPENCLAW_HOME = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(OPENCLAW_HOME))`. This is fragile and prevents proper package installation. Two modules (gdelt_news.py, rss_news.py) previously used absolute `from newsroom.brave_news import` instead of relative imports (now fixed).

4. **Packaging exists.** `pyproject.toml` is present and the project is installable (Hatchling). Dependencies are declared in `pyproject.toml`, and optional chart support is via the `charts` extra (`Pillow`). For contributors, `uv sync --dev` is the canonical dev install path (lockfile-driven).

5. **Configuration is scattered.** Environment variables are read in at least 6 different modules: gemini_client.py (GEMINI_API_KEY, GEMINI_AUTH_PROFILES, GEMINI_PROFILE_ORDER, etc.), gateway_client.py (OPENCLAW_GATEWAY_TOKEN, OPENCLAW_GATEWAY_URL), brave_news.py (BRAVE_SEARCH_API_KEY), runner.py (NANO_BANANA_SCRIPT, NANO_BANANA_API_KEY, OPENCLAW_HOME), and scripts (OPENCLAW_HOME). There is no central config object.

6. **Documentation volume.** 89 KB of markdown across 8 files. ARCHITECTURE.md (31 KB) and AGENTS.md (26 KB) are massive. There is overlap between README.md, ARCHITECTURE.md, and newsroom/README.md. CLAUDE.md is the most useful for development but is only 1.2 KB.

7. **Flat module layout.** All 18 modules sit directly in `newsroom/`. While this works at the current size, it conflates source adapters (brave_news, gdelt_news, rss_news), LLM clients (gemini_client), publishing infrastructure (gateway_client, runner), text processing (dedupe, story_index, result_repair, source_pack), media (charts, image_fetch), persistence (news_pool_db, job_store), and config (prompt_policy, prompt_registry.json).

8. **Test naming.** `test_newsroom.py` is a generic catch-all. `test_runner_script_posts_attachments.py` and `test_runner_recent_titles.py` test runner internals but are named ambiguously. Several tests import private symbols (`_URL_RE`, `_normalize_report_body`, `_count_cjk`).

---

## 2. Dependency Analysis

### Module dependency graph (internal imports only)

```
runner.py (4220 LOC) -- THE HUB
  +-- gateway_client.py
  +-- job_store.py
  +-- brave_news.py
  +-- dedupe.py -------> story_index.py
  +-- news_pool_db.py
  +-- result_repair.py
  +-- charts.py
  +-- image_fetch.py
  +-- market_data.py
  +-- source_pack.py ---> brave_news.py
  |                  +--> image_fetch.py
  |                  +--> news_pool_db.py
  |                  +--> story_index.py
  +-- story_index.py

event_manager.py (standalone -- used by scripts only)
gemini_client.py (standalone -- used by scripts and event_manager indirectly)
prompt_policy.py (standalone -- used by scripts only)

gdelt_news.py -----> brave_news.py (relative import: "from .brave_news")
rss_news.py -------> brave_news.py (relative import: "from .brave_news")
```

### Script dependency graph

```
newsroom_runner.py -------> gateway_client, runner
newsroom_hourly_inputs.py -> event_manager, gemini_client, news_pool_db
newsroom_daily_inputs.py --> event_manager, gemini_client, news_pool_db
newsroom_write_run_job.py -> brave_news, job_store, prompt_policy
news_pool_update.py -------> brave_news, news_pool_db
news_pool_status.py -------> news_pool_db, story_index
brave_news_pool.py --------> (DELETED -- was standalone, duplicated logic from brave_news.py)
rss_pool_update.py --------> brave_news, news_pool_db, rss_news
gdelt_pool_update.py ------> brave_news, gdelt_news, news_pool_db
```

### Observations

- **No circular dependencies.** The graph is a clean DAG.
- **runner.py is a mega-hub** importing 11 of 17 other modules. This is the main structural problem.
- **brave_news.py is the most-imported leaf** (used by runner, source_pack, gdelt_news, rss_news, and 4 scripts). It exports `normalize_url` which is really a general URL utility, not Brave-specific.
- **event_manager.py and gemini_client.py are only used by scripts**, never by runner.py. This is a clean separation (clustering pipeline vs. publishing pipeline).
- **brave_news_pool.py script** has been deleted (was a legacy standalone script duplicating brave_news.py logic; superseded by news_pool_update.py).

---

## 3. Proposed Structure

### Option A: Minimal Changes (low risk, high ROI)

Keep the flat layout. Focus on extracting duplicated code and splitting runner.py.

```
newsroom/
    __init__.py
    # --- Core utilities (new) ---
    _util.py              # _count_cjk(), normalize_url(), ValidationResult, ALLOWED_FAILURE_TYPES
    config.py             # Central config: OPENCLAW_HOME, env var loading, data paths

    # --- Source adapters (unchanged) ---
    brave_news.py         # normalize_url() moves to _util.py, re-exported here for compat
    gdelt_news.py
    rss_news.py

    # --- Text / NLP ---
    story_index.py
    dedupe.py
    source_pack.py
    result_repair.py

    # --- LLM clients ---
    gemini_client.py
    event_manager.py

    # --- Persistence ---
    news_pool_db.py
    job_store.py

    # --- Publishing (split from runner.py) ---
    runner.py             # Slimmed: NewsroomRunner, run_group, poll_and_advance (~1500 LOC)
    discord_publish.py    # _prepare_discord, _publish_script_posts_draft, title translation
    image_pipeline.py     # _ensure_og_image_paths, _ensure_generated_image_paths, _ensure_card_paths, _ensure_infographic_paths
    prompt_render.py      # PromptRegistry, _render_worker_task, template loading

    # --- Media ---
    charts.py
    image_fetch.py
    market_data.py

    # --- Config / policy ---
    prompt_policy.py
    prompt_registry.json
    prompts/
    schemas/
    examples/
    validators/
    tests/
```

**Key changes:**
- Extract `_util.py` with shared code (eliminates 5x `_count_cjk`, 6x `ALLOWED_FAILURE_TYPES`, etc.)
- Extract `config.py` to centralize env var loading and path construction
- Split runner.py into 3-4 files, keeping `NewsroomRunner` as the coordinator
- ~~Fix gdelt_news.py and rss_news.py to use relative imports~~ (DONE)

### Option B: Ideal Long-Term (higher risk, cleaner architecture)

```
newsroom/
    __init__.py
    config.py                # Central config, env loading, OPENCLAW_HOME
    _util.py                 # Shared low-level utilities

    sources/
        __init__.py
        brave.py             # Brave Search API client + URL normalization
        gdelt.py             # GDELT DOC 2.0 client
        rss.py               # RSS/Atom parser + feed config

    llm/
        __init__.py
        gemini.py            # Gemini REST client (OAuth, rotation, quota)
        clustering.py        # Event manager (LLM clustering)
        prompts.py           # PromptRegistry, template rendering, prompt_policy

    pipeline/
        __init__.py
        pool.py              # NewsPoolDB (SQLite news pool)
        indexing.py           # story_index (tokenization, clustering, ranking)
        dedupe.py             # Deduplication logic
        source_pack.py        # Source fetching + extraction + on-topic scoring
        result_repair.py      # LLM output repair

    publishing/
        __init__.py
        runner.py             # NewsroomRunner (orchestration only)
        discord.py            # Discord thread creation, title posting, body publishing
        images.py             # OG image, generated image, card, infographic pipelines
        gateway.py            # OpenClaw gateway HTTP client

    media/
        __init__.py
        charts.py             # Pure-PNG line chart renderer
        image_fetch.py        # OG image extraction + download
        market_data.py        # Stooq/finance data

    jobs/
        __init__.py
        store.py              # FileLock, atomic_write_json, load_json_file
        schemas.py            # Schema validation wrappers

    prompts/                  # Prompt templates (unchanged)
    schemas/                  # JSON schemas (unchanged)
    examples/                 # Example job files (unchanged)
    validators/               # Output validators (unchanged, but import from _util)
    tests/                    # Test suite (unchanged location)

scripts/                      # CLI entry points (eventually -> console_scripts)
```

**Why this is better long-term:**
- Runner.py's 60 methods naturally decompose into 4 concerns: orchestration, Discord publishing, image pipelines, and prompt rendering
- Source adapters are self-contained behind a clean interface
- LLM client code is isolated from pipeline logic
- Adding a new source, publisher, or LLM provider requires touching only one subdirectory

**Why not do this immediately:**
- Production cron jobs reference scripts by file path
- Existing tests import by current module paths
- Risk of import path breakage in a system that runs 24/7

---

## 4. Package Setup

A `pyproject.toml` should be added. This is the highest-ROI single change.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "openclaw-newsroom"
version = "0.1.0"
description = "Automated AI newsroom: LLM clustering, multi-source news pool, Discord publishing"
requires-python = ">=3.12"
dependencies = [
    "requests>=2.28",
    "jsonschema>=4.0",
    "lxml>=4.9",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
charts = ["Pillow>=10.0"]
dev = [
    "pytest>=7.0",
    "pytest-cov",
]

[project.scripts]
newsroom-runner = "scripts.newsroom_runner:main"
newsroom-hourly = "scripts.newsroom_hourly_inputs:main"
newsroom-daily = "scripts.newsroom_daily_inputs:main"
newsroom-write-job = "scripts.newsroom_write_run_job:main"
newsroom-pool-update = "scripts.news_pool_update:main"
newsroom-pool-status = "scripts.news_pool_status:main"
newsroom-rss-update = "scripts.rss_pool_update:main"
newsroom-gdelt-update = "scripts.gdelt_pool_update:main"

[tool.pytest.ini_options]
testpaths = ["newsroom/tests"]
```

**Notes:**
- Console scripts require the scripts to accept `argv` as a parameter (they already do -- `main(sys.argv[1:])`)
- The `sys.path.insert(0, ...)` hack in every script can be removed once the package is installed via `pip install -e .`
- For backward compatibility, keep the `scripts/` directory and `if __name__ == "__main__"` blocks

---

## 5. Configuration Consolidation

### Current state

Environment variables are read in 6+ modules at import time via module-level code. There is no single source of truth for "what env vars does this system need?"

### Recommendation: `newsroom/config.py`

```python
"""Central configuration for the newsroom package.

All environment variable reads are consolidated here.
Modules should import from this module instead of reading os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class NewsroomConfig:
    openclaw_home: Path
    brave_api_keys: list[str]          # from BRAVE_SEARCH_API_KEY (comma-sep)
    gemini_api_key: str | None         # from GEMINI_API_KEY
    gemini_auth_profiles: Path | None  # from GEMINI_AUTH_PROFILES
    gemini_profile_order: list[str]    # from GEMINI_PROFILE_ORDER
    gateway_token: str | None          # from OPENCLAW_GATEWAY_TOKEN
    gateway_http_url: str | None       # from OPENCLAW_GATEWAY_HTTP_URL
    nano_banana_script: Path | None    # from NANO_BANANA_SCRIPT
    nano_banana_api_key: str | None    # from NANO_BANANA_API_KEY
    data_dir: Path                     # openclaw_home / "data" / "newsroom"
    db_path: Path                      # data_dir / "news_pool.sqlite3"

    @classmethod
    def from_env(cls, *, openclaw_home: Path | None = None) -> "NewsroomConfig":
        home = openclaw_home or Path(
            os.environ.get("OPENCLAW_HOME", str(Path.home() / ".openclaw"))
        ).expanduser()
        data_dir = home / "data" / "newsroom"
        # ... populate from env vars ...
```

**Migration path:** New code imports from `config.py`. Old code continues working. Gradually replace `os.environ.get()` calls module by module.

---

## 6. Prompt System

### Current state

The prompt system is spread across 4 locations:
- `prompts/` -- 22 markdown template files
- `prompt_registry.json` -- maps prompt_id -> template + validator
- `prompt_policy.py` -- maps category -> prompt_id (32 LOC)
- `validators/` -- 6 Python validator modules

### Assessment: Keep as-is, with minor improvements

The current design is actually solid. The registry JSON is a clean declarative mapping. The validator modules are self-contained. The prompt templates are versioned in their filenames.

**Minor improvements:**
1. Move `prompt_policy.py` logic into `prompt_registry.json` as a `category_defaults` section. This eliminates a file and makes category-to-prompt mapping declarative.
2. Extract `ALLOWED_FAILURE_TYPES` and `ValidationResult` from validators into `_util.py` or `validators/__init__.py`.
3. Extract `_count_cjk()` into `_util.py` and import it in all 5 locations that currently duplicate it.
4. The `PromptRegistry` class (currently in runner.py, lines 265-310) should move to its own module or into a hypothetical `prompt_render.py`.

---

## 7. Test Strategy

### Current coverage

| Module | Test file | LOC | Coverage quality |
|--------|-----------|-----|-----------------|
| news_pool_db.py (1152) | test_news_pool_db.py | 586 | Good (schema, CRUD, events) |
| gemini_client.py (748) | test_gemini_client.py | 1132 | Excellent (retry, rotation, SSE) |
| event_manager.py (547) | test_event_manager.py | 475 | Good (clustering, merge) |
| runner.py (4220) | test_newsroom.py + 3 others | 649 | Weak (schema + a few methods) |
| story_index.py (748) | test_story_index.py | 133 | Fair |
| result_repair.py (295) | test_result_repair.py | 92 | Fair |
| rss_news.py (311) | test_rss_news.py | 178 | Good |
| gdelt_news.py (243) | test_gdelt_news.py | 126 | Good |
| market_data.py (503) | test_market_data_symbols.py + test_stooq_parse.py | 58 | Weak |
| dedupe.py (245) | test_dedupe.py | 23 | Very weak |
| brave_news.py (540) | (none) | 0 | None |
| source_pack.py (805) | (none) | 0 | None |
| charts.py (304) | (none) | 0 | None |
| image_fetch.py (225) | test_image_fetch.py | 23 | Minimal |
| gateway_client.py (171) | (none) | 0 | None |
| job_store.py (180) | test_job_store.py | 39 | Minimal |
| prompt_policy.py (32) | test_prompt_policy.py | 26 | Good (100%) |

### Key gaps

1. **runner.py (4220 LOC, ~38% of all code)** has only ~650 LOC of tests covering schema validation, prompt registry, recent-titles, and attachment logic. The core orchestration (`_start_job`, `_poll_and_advance`, `run_group`), Discord publishing, image pipelines, source pack preparation, and deduplication logic are untested.

2. **brave_news.py** (540 LOC) has zero tests. URL normalization, API key rotation, and rate limit recording are all untested.

3. **source_pack.py** (805 LOC) has zero tests. This is the most complex data-gathering module after runner.py.

4. **gateway_client.py** has zero tests.

### Recommended improvements

1. **Split runner.py tests by concern.** Instead of `test_newsroom.py` (generic name), create `test_runner_orchestration.py`, `test_runner_discord.py`, `test_runner_dedup.py`.

2. **Add brave_news.py tests.** URL normalization is a pure function -- easy to test. API key rotation has clear edge cases.

3. **Add source_pack.py tests.** The on-topic scoring and text extraction logic are pure functions.

4. **Move conftest.py content.** Currently it only has `collect_ignore_glob = ["_archive/*"]`. This should be in `pyproject.toml` under `[tool.pytest.ini_options]`.

5. **Stop importing private symbols in tests.** `_URL_RE`, `_normalize_report_body`, `_count_cjk` -- these should be tested via their public callers, or promoted to public API.

---

## 8. Migration Path

Given this is running in production via cron, migrations must be zero-downtime.

### Phase 1: Foundation (no import path changes)

1. Add `pyproject.toml` with metadata and dependencies
2. Create `newsroom/_util.py` with extracted shared code
3. Update validators to import `ALLOWED_FAILURE_TYPES`, `ValidationResult`, `_count_cjk` from `_util.py`
4. Fix absolute imports in `gdelt_news.py` and `rss_news.py` to use relative imports
5. Run full test suite, deploy

**Risk**: Zero. All existing import paths still work.

### Phase 2: Runner decomposition (no import path changes)

1. Extract `PromptRegistry` + template rendering into `newsroom/prompt_render.py`
2. Extract Discord publishing methods into `newsroom/discord_publish.py`
3. Extract image pipeline methods into `newsroom/image_pipeline.py`
4. Keep `NewsroomRunner` in `runner.py` as the coordinator that delegates to new modules
5. Keep all existing public API (`NewsroomRunner`, `PromptRegistry`, etc.) re-exported from `runner.py`
6. Run full test suite, deploy

**Risk**: Low. `runner.py` still exports the same names. Scripts are unaffected.

### Phase 3: Package installation

1. Add `pip install -e .` to CONTRIBUTING.md
2. Remove `sys.path.insert()` from all scripts (replace with proper imports)
3. ~~Delete `brave_news_pool.py`~~ (DONE -- superseded by `news_pool_update.py`)
4. Update cron job commands to use installed console_scripts
5. Run full test suite, deploy

**Risk**: Medium. Cron commands change. Requires coordinated cron update.

### Phase 4: Subpackages (long-term, optional)

1. Move source adapters to `newsroom/sources/`
2. Move LLM clients to `newsroom/llm/`
3. Add `__init__.py` re-exports for backward compatibility
4. Update tests
5. Run full test suite, deploy

**Risk**: Medium. Many import paths change. Use `__init__.py` re-exports to maintain backward compatibility for one release cycle.

---

## 9. Prioritized Action Items

Ordered by impact/effort ratio (highest first).

| # | Action | Impact | Effort | Notes |
|---|--------|--------|--------|-------|
| 1 | **Add pyproject.toml** | High | Low | Enables `pip install -e .`, proper dependency declaration, console_scripts. 30 min. |
| 2 | **Extract `_util.py`** | High | Low | Eliminates 5x `_count_cjk`, 6x `ALLOWED_FAILURE_TYPES`, 2x `ValidationResult`. 1 hour. |
| 3 | ~~**Fix relative imports** in gdelt_news.py, rss_news.py~~ | Medium | Trivial | DONE. Changed `from newsroom.X` to `from .X`. |
| 4 | ~~**Delete `scripts/brave_news_pool.py`**~~ | Low | Trivial | DONE. Deleted; superseded by `news_pool_update.py`. |
| 5 | **Split runner.py** into 3-4 modules | Very High | Medium | Biggest structural improvement. Extract discord_publish, image_pipeline, prompt_render. 4-6 hours. |
| 6 | **Add `config.py`** | Medium | Medium | Centralize env vars. Reduces coupling and makes configuration testable. 2-3 hours. |
| 7 | **Add tests for brave_news.py** | Medium | Low | Pure-function URL normalization is trivially testable. 1 hour. |
| 8 | **Add tests for source_pack.py** | Medium | Medium | Complex module with zero coverage. 3-4 hours. |
| 9 | **Remove sys.path hacking from scripts** | Medium | Low | After pyproject.toml is in place. 30 min. |
| 10 | **Consolidate documentation** | Low | Medium | Merge newsroom/README.md into README.md. Trim ARCHITECTURE.md (31 KB is too long). 2 hours. |
| 11 | **Move PromptRegistry out of runner.py** | Medium | Low | Natural first step of runner decomposition. 1 hour. |
| 12 | **Add prompt_policy mapping to prompt_registry.json** | Low | Low | Eliminates prompt_policy.py (32 LOC). 30 min. |
| 13 | **Move to subpackage layout** (Option B) | High | High | Long-term goal. Only after items 1-9 are done. 1-2 days. |

### Recommended immediate sprint (items 1-4)

These 4 items can be done in a single commit with zero risk to production:
- Add `pyproject.toml`
- Create `newsroom/_util.py`
- ~~Fix 2 absolute imports~~ (DONE)
- ~~Delete legacy script~~ (DONE)

Total effort: ~2 hours. This unblocks all subsequent improvements.
