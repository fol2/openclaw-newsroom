# Prompt System Reference

## Overview

The newsroom uses a registry-based prompt system. Each story category maps to a prompt template and validator pair. The runner resolves prompts at execution time, renders templates with job-specific variables, sends the rendered prompt to the LLM, and validates the structured JSON response against the paired validator.

There are two publisher modes:
- **`script_posts`** -- The LLM produces a draft JSON; the runner handles Discord publishing. Category-specific prompts are routed via `prompt_policy.py`.
- **`agent_posts`** -- The LLM posts directly to Discord via tool calls and returns a result JSON. Uses content-type-based prompt routing.

## Prompt Registry (`prompt_registry.json`)

The registry file at `newsroom/prompt_registry.json` uses schema version `prompt_registry_v1` and contains three top-level sections: `prompts`, `validators`, and `content_types`.

### Registered Prompts

| Prompt ID | Template File | Validator ID |
|---|---|---|
| `news_reporter_script_v1` | `newsroom/prompts/news_reporter_script_v1.md` | `news_reporter_script_v1` |
| `news_reporter_script_default_v1` | `newsroom/prompts/news_reporter_script_default_v1.md` | `news_reporter_script_v1` |
| `news_reporter_script_politics_v1` | `newsroom/prompts/news_reporter_script_politics_v1.md` | `news_reporter_script_v1` |
| `news_reporter_script_uk_parliament_v1` | `newsroom/prompts/news_reporter_script_uk_parliament_v1.md` | `news_reporter_script_v1` |
| `news_reporter_script_uk_news_v1` | `newsroom/prompts/news_reporter_script_uk_news_v1.md` | `news_reporter_script_v1` |
| `news_reporter_script_hk_news_v1` | `newsroom/prompts/news_reporter_script_hk_news_v1.md` | `news_reporter_script_v1` |
| `news_reporter_script_finance_v1` | `newsroom/prompts/news_reporter_script_finance_v1.md` | `news_reporter_script_v1` |
| `news_reporter_script_sports_v1` | `newsroom/prompts/news_reporter_script_sports_v1.md` | `news_reporter_script_v1` |
| `news_reporter_script_entertainment_v1` | `newsroom/prompts/news_reporter_script_entertainment_v1.md` | `news_reporter_script_v1` |
| `news_reporter_script_hk_entertainment_v1` | `newsroom/prompts/news_reporter_script_hk_entertainment_v1.md` | `news_reporter_script_v1` |
| `news_reporter_script_ai_v1` | `newsroom/prompts/news_reporter_script_ai_v1.md` | `news_reporter_script_v1` |
| `news_rescue_script_v1` | `newsroom/prompts/news_rescue_script_v1.md` | `news_rescue_script_v1` |
| `news_reporter_v2_2_inline` | `newsroom/prompts/news_reporter_v2_2_inline.md` | `news_reporter_v2_1_strict` |
| `news_reporter_v2_1` | `newsroom/prompts/news_reporter_v2_1.md` | `news_reporter_v2_1_strict` |
| `news_rescue_v1` | `newsroom/prompts/news_rescue_v1.md` | `news_rescue_v1` |
| `blog_post_long_v1` | `newsroom/prompts/blog_post_long_v1.md` | `blog_post_long_v1` |
| `twitter_post_v1` | `newsroom/prompts/twitter_post_v1.md` | `social_post_v1` |
| `facebook_post_v1` | `newsroom/prompts/facebook_post_v1.md` | `social_post_v1` |

### Content Type Defaults

| Content Type | Default Prompt ID |
|---|---|
| `news_deep_dive` | `news_reporter_v2_2_inline` |
| `blog_post_long` | `blog_post_long_v1` |
| `twitter_post` | `twitter_post_v1` |
| `facebook_post` | `facebook_post_v1` |

## Category Routing (`prompt_policy.py`)

The function `prompt_id_for_category()` in `newsroom/prompt_policy.py` maps story categories to prompt IDs for `script_posts` mode. Unknown categories fall back to the default prompt.

| Category | Prompt ID |
|---|---|
| `US Stocks` | `news_reporter_script_finance_v1` |
| `Crypto` | `news_reporter_script_finance_v1` |
| `Precious Metals` | `news_reporter_script_finance_v1` |
| `Global News` | `news_reporter_script_default_v1` |
| `Politics` | `news_reporter_script_politics_v1` |
| `UK Parliament / Politics` | `news_reporter_script_uk_parliament_v1` |
| `UK News` | `news_reporter_script_uk_news_v1` |
| `Hong Kong News` | `news_reporter_script_hk_news_v1` |
| `Hong Kong Entertainment` | `news_reporter_script_hk_entertainment_v1` |
| `Entertainment` | `news_reporter_script_entertainment_v1` |
| `Sports` | `news_reporter_script_sports_v1` |
| `AI` | `news_reporter_script_ai_v1` |
| _(any other / unknown)_ | `news_reporter_script_default_v1` |

## Template Variables

Templates use Mustache-style `{{VARIABLE}}` placeholders. The `_render_template()` function in `runner.py` performs substitution and raises `PromptRegistryError` if any placeholders remain unresolved.

### Standard Variables

| Variable | Source | Description |
|---|---|---|
| `{{OPENCLAW_HOME}}` | Auto-injected by `_render_template()` | Path to the OpenClaw home directory (defaults to `~/.openclaw`). Used for referencing scripts and data paths within prompts. |
| `{{RUN_TIME_UK}}` | `run.run_time_uk` field in the job | Current run time in UK timezone (e.g. `2026-02-09 14:05 GMT`). Used by planners for recency decisions. |
| `{{INPUT_JSON}}` | Built by `build_worker_input()` from `spawn.input_mapping` | Serialized JSON containing job data for worker prompts. Constructed via JSONPath expressions (`$.story.title`, `$.state.source_pack`, etc.). |
| `{{WORKER_ERROR_TYPE}}` | From failed worker result | Error type from the primary worker run. Injected into rescue prompts so the rescue agent knows what went wrong. |
| `{{WORKER_ERROR_MESSAGE}}` | From failed worker result | Error message from the primary worker run. Injected into rescue prompts. |

### Job Schema Fields (story_job_v1)

Key fields available through `spawn.input_mapping` JSONPath expressions:

- `$.story.story_id` -- Unique story identifier
- `$.story.content_type` -- Content type (e.g. `news_deep_dive`, `blog_post_long`)
- `$.story.category` -- Story category (e.g. `Global News`, `US Stocks`, `Hong Kong News`)
- `$.story.title` -- Story headline
- `$.story.primary_url` -- Primary source URL
- `$.story.supporting_urls` -- Array of additional source URLs
- `$.story.concrete_anchor` -- Factual anchor point for the story
- `$.story.flags` -- Array of story flags
- `$.story.instructions` -- Optional special instructions
- `$.state.discord.thread_id` -- Discord thread ID (may be null at draft time)
- `$.state.source_pack` -- Pre-fetched source content pack with extracted article text

## Planner Prompts

Planner prompts are used by cron-triggered planning agents that discover news and create story job files. They are not registered in `prompt_registry.json` (they are used directly by the planner scripts).

| Template | Description |
|---|---|
| `planner_hourly_v1.md` | Hourly planner (script-first). Creates 0-2 story jobs for breaking/developing news. Runs clustering and deduplication via a single exec call, then writes jobs and launches the runner. Max 2 exec calls. |
| `planner_hourly_v1_trigger_runner.md` | Hourly planner with runner trigger. Similar to the above but creates 0-1 story jobs and explicitly triggers the runner after writing the job file. Max 3 exec calls. |
| `planner_daily_v1.md` | Daily planner. Creates exactly 5 story jobs with category balance. Uses Brave News pool for discovery. Max 4 exec calls. |
| `planner_daily_script_v1.md` | Daily planner (script-first). Creates exactly 10 story jobs with category balance and HK guarantee. Uses a single inputs script for pool update + clustering + dedupe, then writes all files and launches the runner. Max 2 exec calls. |

All planner prompts enforce strict tool restrictions: no `web_search`, no `web_fetch`, no `browser`, no `message` tool. They operate exclusively through `exec` calls to Python scripts.

## Worker Prompts

Worker prompts are used by the runner to generate story content. They are resolved through the prompt registry.

### Script Posts Workers (draft mode -- runner publishes)

These prompts instruct the LLM to produce a structured JSON draft without posting to Discord. The runner handles all Discord publishing. All share the `news_reporter_script_v1` validator.

| Prompt ID | Category Focus | Description |
|---|---|---|
| `news_reporter_script_v1` | Generic | Base script-posts worker. Generates a Cantonese Traditional Chinese deep-dive news report draft. Body target: 600-3000 CJK characters. |
| `news_reporter_script_default_v1` | Global News / fallback | Default category prompt for general news coverage. |
| `news_reporter_script_politics_v1` | Politics | Tailored for political news with emphasis on policy implications. |
| `news_reporter_script_uk_parliament_v1` | UK Parliament / Politics | Specialized for UK parliamentary and political developments. |
| `news_reporter_script_uk_news_v1` | UK News | Tailored for general UK news stories. |
| `news_reporter_script_hk_news_v1` | Hong Kong News | Specialized for Hong Kong news with local context. |
| `news_reporter_script_hk_entertainment_v1` | Hong Kong Entertainment | Hong Kong entertainment and celebrity news. |
| `news_reporter_script_finance_v1` | US Stocks / Crypto / Precious Metals | Finance-focused prompt with market data emphasis. |
| `news_reporter_script_sports_v1` | Sports | Sports news coverage. |
| `news_reporter_script_entertainment_v1` | Entertainment | General entertainment news. |
| `news_reporter_script_ai_v1` | AI | Artificial intelligence and technology news. |

### Agent Posts Workers (direct Discord posting)

These prompts instruct the LLM to post directly to Discord via tool calls and return a result JSON.

| Prompt ID | Content Type | Description |
|---|---|---|
| `news_reporter_v2_2_inline` | `news_deep_dive` | Deep-dive reporter with inline source pack. Posts directly to Discord thread. Validated by `news_reporter_v2_1_strict`. Report target: 2700-3300 characters. |
| `news_reporter_v2_1` | `news_deep_dive` | Earlier version of the deep-dive reporter. Also validated by `news_reporter_v2_1_strict`. |
| `blog_post_long_v1` | `blog_post_long` | Long-form blog post writer. Posts 1200-2200 CJK characters to Discord thread. |
| `twitter_post_v1` | `twitter_post` | Twitter/X style social post. Max 380 characters, punchy tone, at most 1 hashtag. |
| `facebook_post_v1` | `facebook_post` | Facebook style social post. 450-900 CJK characters, conversational tone. |

### Rescue Prompts

Rescue prompts are invoked when a primary worker fails or times out. They receive the worker's error context and attempt a shorter recovery output.

| Prompt ID | Mode | Description |
|---|---|---|
| `news_rescue_script_v1` | Script posts | Rescue reporter for draft mode. Produces a shorter Cantonese report draft (400-2000 CJK characters). Does not post to Discord. Validated by `news_rescue_script_v1`. |
| `news_rescue_v1` | Agent posts | Rescue reporter that posts directly to Discord. Produces a shorter report (800-1200 characters). Validated by `news_rescue_v1`. |

## Validators

Each validator receives the LLM's structured JSON output and the original job file, and returns a `ValidationResult(ok, errors)`. Errors are coded strings (e.g. `missing_key:title`, `success_requires:body_cjk_600_to_3000`) that identify the specific validation failure.

### `news_reporter_script_v1`

Used by all `news_reporter_script_*` prompts. Checks:
- **Required keys**: status, story_id, category, title, primary_url, thread_id, content_posted, content_message_ids, images_attached_count, read_more_urls_count, report_char_count, draft, concrete_anchor_provided, concrete_anchor_used, sources_used, error_type, error_message.
- **Identity cross-checks**: story_id, primary_url, and thread_id must match the job file.
- **Draft structure**: `draft.body` must be a non-empty string; `draft.read_more_urls` must be a list.
- **On SUCCESS**: title must contain at least 4 CJK characters; body must be 600-3000 CJK characters; body must not contain URLs or "read more" section markers; read_more_urls must contain 3-5 entries including the primary URL; read_more_urls must include at least one different domain; sources_used must have at least 2 entries; error fields must be null.
- **On FAILURE**: error_type must be from the allowed set; error_message must be non-empty.

### `news_reporter_v2_1_strict`

Used by `news_reporter_v2_2_inline` and `news_reporter_v2_1` (agent-posts mode). Checks:
- **Required keys**: status, story_id, category, title, primary_url, thread_id, content_posted, content_message_ids, images_attached_count, read_more_urls_count, report_char_count, concrete_anchor_provided, concrete_anchor_used, sources_used, error_type, error_message.
- **Identity cross-checks**: story_id, primary_url, thread_id.
- **On SUCCESS**: content_posted must be true; content_message_ids must be non-empty; images 0-3; read_more_urls_count 3-5; report_char_count 2700-3300; concrete_anchor_used must be true; primary_url must appear in sources_used.

### `news_rescue_script_v1`

Used by the `news_rescue_script_v1` prompt (script-posts rescue mode). Checks:
- **Required keys**: same as `news_reporter_script_v1` plus `mode` (must be `RESCUE`), minus `concrete_anchor_provided` and `concrete_anchor_used`.
- **Draft structure**: body and read_more_urls validated.
- **On SUCCESS**: body 400-2000 CJK characters; no URLs in body; read_more_urls 2-4 entries; sources_used at least 2; primary URL inclusion and domain diversity enforced.

### `news_rescue_v1`

Used by the `news_rescue_v1` prompt (agent-posts rescue mode). Checks:
- **Required keys**: similar to `news_reporter_v2_1_strict` plus `mode` (must be `RESCUE`).
- **On SUCCESS**: content_posted must be true; images 0-3; read_more_urls_count 2-4; report_char_count 800-1200.

### `blog_post_long_v1`

Used by the `blog_post_long_v1` prompt. Checks:
- **Required keys**: includes `mode` (must be `WORKER`) and `content_type` (must be `blog_post_long`).
- **On SUCCESS**: content_posted true; images 0-2; read_more_urls_count 2-5; report_char_count 1200-2200.

### `social_post_v1`

Used by `twitter_post_v1` and `facebook_post_v1`. Checks:
- **Required keys**: includes `mode` (must be `WORKER`) and `content_type` (must be `twitter_post` or `facebook_post`).
- **On SUCCESS**: content_posted true; images 0-1.
- Lighter validation than news reporters (no read_more or report_char_count range checks in required keys).

## Result Repair (`result_repair.py`)

The `repair_result_json()` function in `newsroom/result_repair.py` performs best-effort deterministic fixes for common LLM output issues before re-validation. It is called by the runner when initial validation fails, potentially avoiding the cost of a rescue run.

Repair strategies (applied in order):

1. **Title from job** (`title_from_job`): If the validator reports a `title_traditional_chinese` error and the job file already contains a Cantonese title (at least 4 CJK characters), the job title is copied into the result.

2. **Read-more URLs** (`read_more_urls`): If read_more_urls fail range or inclusion checks, the repair function collects candidate URLs from the existing result, the job's primary URL, supporting URLs, and the source pack. It rebuilds the list to satisfy the minimum count, ensures the primary URL is included, and attempts to include at least one URL from a different domain. URLs are deduplicated and capped at the maximum allowed count.

3. **Sources used** (`sources_used`): If `sources_used_min_2` fails, the repair function supplements the existing sources list with on-topic URLs from the source pack until at least 2 sources are present.

4. **Body CJK padding** (`body_cjk_pad` / `body_cjk_expand`): If the body falls below the minimum CJK character count:
   - For small shortfalls (under 60 characters), a short generic Cantonese closing paragraph is appended.
   - For moderate shortfalls (under 800 characters), longer generic Cantonese filler paragraphs are appended until the minimum is reached.
   - The filler text is intentionally generic and does not introduce new facts. A guardrail ensures the padded body stays within the validator's accepted range.

All repairs operate on a deep copy of the result JSON. The function returns the repaired result and a list of repair codes applied, which the runner logs for observability.
