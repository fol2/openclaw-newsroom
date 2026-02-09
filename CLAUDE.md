# openclaw-newsroom

Automated AI newsroom: LLM clustering, multi-source news pool, Discord publishing.

## Key Documentation

- [README.md](README.md) -- Project overview, installation, configuration
- [ARCHITECTURE.md](ARCHITECTURE.md) -- Technical architecture and data flow
- [AGENTS.md](AGENTS.md) -- Cron agent system, planner/runner architecture
- [PROMPTS.md](PROMPTS.md) -- Prompt template system and validator reference

## Project Structure

- `newsroom/` -- Core Python package (18 modules)
- `newsroom/prompts/` -- LLM prompt templates (Mustache-style `{{VAR}}`)
- `newsroom/schemas/` -- JSON schemas for job files
- `newsroom/validators/` -- Output validators for LLM responses
- `newsroom/tests/` -- Test suite (192 tests)
- `scripts/` -- CLI entry points (9 scripts)

## Development

- Python 3.12+, deps in `requirements.txt`
- Tests: `PYTHONPATH=. pytest newsroom/tests/ -v`
- All prompt template paths use `{{OPENCLAW_HOME}}` (resolved at runtime)
- `_render_template()` in runner.py auto-injects `OPENCLAW_HOME`

## Code Style

- No auto-formatting enforced; follow existing patterns
- Validators must inherit structure from `newsroom/validators/`
- New prompts need entries in `prompt_registry.json` + a matching validator
