# Contributing to openclaw-newsroom

## Getting Started

1. Fork and clone the repository:

```bash
git clone <your-fork-url> openclaw-newsroom
cd openclaw-newsroom
```

2. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

3. Copy `.env.example` to `.env` and fill in the required values (see README.md for details). For local development and testing, most features work without API keys when using `--dry-run`.

## Running Tests

All 192 tests must pass before submitting changes:

```bash
PYTHONPATH=. pytest newsroom/tests/ -v
```

Tests are located in `newsroom/tests/` and cover validators, result repair, prompt registry resolution, job schemas, deduplication, tokenization, and more.

## Adding a News Source

The newsroom ingests news through source adapters that fetch articles and upsert them into the SQLite news pool. To add a new source:

1. Create a new adapter module in `newsroom/` following the pattern of existing adapters:
   - `brave_news.py` -- Brave News API (search, URL normalization, multi-key rotation)
   - `gdelt_news.py` -- GDELT DOC 2.0 API (free, no auth)
   - `rss_news.py` -- RSS/Atom feed parser (feed config, lxml extraction)

2. Your adapter should implement a fetch function that returns normalized article data (title, URL, published timestamp, source name) compatible with `news_pool_db.py` upsert operations.

3. Create a corresponding pool update script in `scripts/` (e.g. `my_source_pool_update.py`) that invokes your adapter and upserts results into the pool database. Follow the CLI argument pattern of `news_pool_update.py` or `gdelt_pool_update.py`.

4. Test the adapter with the existing pool database to verify deduplication and schema compatibility.

## Adding a Prompt Template

Prompt templates drive the LLM story writing pipeline. Each template is paired with a validator. To add a new prompt:

1. Create a `.md` file in `newsroom/prompts/`. Use Mustache-style `{{VARIABLE}}` placeholders. The variable `{{OPENCLAW_HOME}}` is auto-injected by `_render_template()` in `runner.py`. Other variables (e.g. `{{INPUT_JSON}}`, `{{RUN_TIME_UK}}`) come from the job file or runner context.

2. Add an entry in `newsroom/prompt_registry.json` under the `prompts` section:

```json
"my_new_prompt_v1": {
  "template_path": "newsroom/prompts/my_new_prompt_v1.md",
  "validator_id": "my_validator_id"
}
```

3. Create or reuse a validator (see "Adding a Validator" below). Register it in the `validators` section of `prompt_registry.json`:

```json
"my_validator_id": {
  "type": "python",
  "path": "newsroom/validators/my_validator_id.py"
}
```

4. If the prompt is category-specific for the `script_posts` publisher mode, add a mapping in `newsroom/prompt_policy.py` inside the `prompt_id_for_category()` function.

5. If the prompt is for a new content type, add it to the `content_types` section of `prompt_registry.json`.

## Adding a Validator

Validators check the structured JSON output from LLM worker responses. Each validator module must export a single function:

```python
def validate(result_json: dict[str, Any], job: dict[str, Any]) -> ValidationResult
```

Where `ValidationResult` is a dataclass with `ok: bool` and `errors: list[str]`.

To create a new validator:

1. Create a Python file in `newsroom/validators/` following the pattern of existing validators (e.g. `news_reporter_script_v1.py`).

2. Define the required keys that must be present in the result JSON.

3. Implement type checks for each field (use helper functions like `_is_non_empty_str`, `_as_int`).

4. Add status-specific checks: `SUCCESS` status should validate content quality constraints (character counts, URL ranges, CJK character requirements); `FAILURE` status should validate that `error_type` is from the allowed set and `error_message` is present.

5. Return `ValidationResult(ok=len(errors) == 0, errors=errors)`.

6. The runner uses validation outcomes to decide whether to accept the result, attempt auto-repair via `result_repair.py`, or trigger a rescue run.

## Code Guidelines

- Follow existing code patterns and module structure.
- Keep modules focused on a single responsibility.
- The project intentionally avoids heavy dependencies like pandas and numpy. Do not introduce them.
- All text processing should handle CJK characters correctly (the codebase includes `_count_cjk()` helpers for this).
- Use `from __future__ import annotations` in new modules.
- Type hints are used throughout; maintain them in new code.
- Test new functionality with unit tests in `newsroom/tests/`.

## Pull Requests

- Keep PRs small and focused on a single change.
- Describe what changed and why in the PR description.
- Include test coverage for new functionality.
- Ensure all 192 existing tests still pass.
- If adding a new prompt or validator, include sample input/output in the PR description to aid review.
