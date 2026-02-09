# Newsroom Engine (Planner + Deterministic Runner)

This implements the architecture described in `ARCHITECTURE.md` and `AGENTS.md`:

- **Planner (LLM)**: selects stories and writes `story_job_v1` JSON files under `workspace/jobs/`, then stops.
- **Runner (deterministic script)**: reads job JSON, creates Discord containers (title + thread), spawns worker sub-agents, monitors them, validates strict RESULT JSON, runs rescue on failure/timeout, and sends a DM summary.

## Folder Layout

- `jobs/`  
  Run folders created by the planner:
  - Daily: `workspace/jobs/discord-multi-YYYY-MM-DD-HH-mm/`
    - `run.json` (optional, shared runner config)
    - `story_01.json` ... `story_05.json`
  - Hourly: `workspace/jobs/discord-<channel_id>-YYYY-MM-DD-HH-mm/`
    - `story.json`

- `newsroom/schemas/`  
  JSON schemas for `story_job_v1` and `run_job_v1`.

- `newsroom/prompts/`  
  Prompt templates for workers/rescue and planner templates.

- `newsroom/prompt_registry.json`  
  Maps `prompt_id` → template file + `validator_id`.

- `logs/newsroom/<run_id>/`  
  JSONL logs per run and per story.

## Running The Runner Manually

Run against discovered jobs:

```bash
uv run python3 scripts/newsroom_runner.py
```

Dry run (validate + render prompts only, no posting/spawning, no job mutation):

```bash
uv run python3 scripts/newsroom_runner.py --dry-run
```

Run a specific folder or file:

```bash
uv run python3 scripts/newsroom_runner.py --path workspace/jobs/discord-multi-2026-02-02-07-00
uv run python3 scripts/newsroom_runner.py --path workspace/jobs/discord-1467628391082496041-2026-02-02-10-30/story.json
```

## Planner Prompts

- Daily: `newsroom/prompts/planner_daily_v1.md`
- Hourly: `newsroom/prompts/planner_hourly_v1.md`

These prompts are designed to:
- read Discord history for de-dupe,
- select stories,
- write `story_job_v1` files under `jobs/`,
- and stop.
