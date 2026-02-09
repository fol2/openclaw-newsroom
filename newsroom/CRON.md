# Cron Wiring (Recommended)

## Why A Separate Runner Exists
OpenClaw cron turns are **LLM turns**. They are great for planning, but they are not a reliable place to run long-running background processes:
- `exec ... &`-style background processes do **not** reliably persist after the cron turn ends.
- Even if they did, you still want a deterministic process that can run/retry without spending LLM tokens.

So the recommended wiring is:
- OpenClaw cron runs the **planner** (LLM) to write job JSON files.
- OS `cron` runs the **runner** (deterministic Python) every minute to pick up new jobs, execute them, then exit.

This file provides CLI commands to add planner jobs without touching your existing (legacy) coordinator jobs.

## Daily Planner (07:00 + 19:00 UK)

```bash
openclaw cron add \\
  --name "Newsroom Daily Planner (07:00,19:00 UK)" \\
  --agent newsroom \\
  --session isolated \\
  --cron "0 7,19 * * *" \\
  --tz "Europe/London" \\
  --disabled \\
  --timeout-seconds 900 \\
  --message "$(cat {{OPENCLAW_HOME}}/newsroom/prompts/planner_daily_v1.md)"
```

## Hourly Planner (xx:30 UK)

```bash
openclaw cron add \\
  --name "Newsroom Hourly Planner (xx:30 UK)" \\
  --agent newsroom \\
  --session isolated \\
  --cron "30 * * * *" \\
  --tz "Europe/London" \\
  --disabled \\
  --timeout-seconds 600 \\
  --message "$(cat {{OPENCLAW_HOME}}/newsroom/prompts/planner_hourly_v1.md)"
```

## Runner

The runner (`newsroom/runner.py`) is now launched directly by the write-run script:

```bash
python3 scripts/newsroom_write_run_job.py --launch-runner
```

The full deterministic pipeline is:
1. Cron → `scripts/newsroom_daily_inputs.py` (gather inputs)
2. Cron → `scripts/newsroom_write_run_job.py --launch-runner` (write job JSON + spawn runner)
3. Runner picks up the job, executes it, and exits

No OS-level `crontab` entry is needed — the runner is spawned on-demand by the write-run script.
