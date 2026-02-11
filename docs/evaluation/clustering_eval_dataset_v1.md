# Clustering Evaluation Dataset v1

This document defines the labelled evaluation dataset introduced for issue #25.

## Purpose

The dataset provides a replayable regression baseline for clustering decisions.
It is designed to answer:

- Did parser/policy changes alter event assignment outcomes?
- Are EN/ZH/mixed links all represented in evaluation?
- What error slices (for example low-score assignment) are changing over time?

## Dataset Artefacts

- `newsroom/evals/clustering_eval_dataset_v1.jsonl`
- `newsroom/evals/clustering_eval_dataset_v1.meta.json`

Each JSONL line is one labelled sample with this schema version:

- `schema_version`: `"clustering_eval_sample_v1"`
- `sample_id`: Stable row identifier in this generated file.
- `source`: Provenance from `clustering_decisions`, including stored `observed_enforced_action` and `observed_action_drift`.
- `link`: Link metadata from `links` at sample build time.
- `prediction`: Target derived by replaying `raw_llm_response` through current `parse_clustering_response(...)`.
- `labels`: Human-review-ready labels for evaluation.
- `replay`: Frozen payload needed to re-run `parse_clustering_response(...)` offline.

## Label Fields

`labels` always contains:

- `event_id_or_new_event`: Integer event ID or string `"new_event"`.
- `language`: One of `"en"`, `"zh"`, `"mixed"`.
- `correctness`: One of `"correct"`, `"incorrect"`.
- `flags`: Zero or more of:
  - `roundup`
  - `multi_topic`
  - `opinion`
  - `live_updates`
  - `development`
  - `policy_override`
  - `low_retrieval_score`

## Labelling Criteria (v1)

### Event target

1. Start from one `clustering_decisions` row (link + candidates + raw response).
2. Replay `raw_llm_response` against frozen `link` + `candidate_events` using `parse_clustering_response(...)`.
3. If replay action is `assign`, target is the assigned `event_id`.
4. If replay action is `development` or `new_event`, target is `new_event`.
5. Conservative correction rule: when replay action is `assign` and top retrieval score is below `0.22`, relabel target as `new_event`.

### Language

1. If title/description contains both CJK and Latin scripts, label `mixed`.
2. Otherwise use persisted `lang_hint` when valid.
3. Else infer with `detect_link_lang_hint(...)`.

### Correctness

- `correct` if `prediction.event_id_or_new_event == labels.event_id_or_new_event`.
- `incorrect` otherwise.

### Flags

- Start from allowed `link_flags` in replay enforced action (`roundup`, `multi_topic`, `opinion`, `live_updates`).
- Add `development` when replay enforced action is `development`.
- Add `policy_override` when enforcement reasons exist.
- Add `low_retrieval_score` when top candidate score is below `0.22`.

## Reproducible Build Workflow

Build a fresh dataset from recent decision logs:

```bash
uv run python scripts/build_clustering_eval_dataset.py \
  --db ~/.openclaw/data/newsroom/news_pool.sqlite3 \
  --sample-size 240 \
  --since-hours 1080 \
  --seed 25 \
  --lang-min en=150,zh=40,mixed=15 \
  --low-score-threshold 0.22 \
  --overwrite
```

This writes:

- `newsroom/evals/clustering_eval_dataset_v1.jsonl`
- `newsroom/evals/clustering_eval_dataset_v1.meta.json`

## Replay Workflow

Replay dataset rows through parser logic without external API calls:

```bash
uv run python scripts/replay_clustering_eval_dataset.py \
  --dataset newsroom/evals/clustering_eval_dataset_v1.jsonl \
  --summary-out newsroom/evals/clustering_eval_dataset_v1.replay.json \
  --fail-on-parse-error \
  --fail-on-correctness-mismatch
```

This checks that labels remain self-consistent with parser outputs and provides
machine-readable replay statistics.

## Notes

- The v1 dataset is generated from recent clustering runs and is deterministic
  under the same DB snapshot + CLI arguments.
- Manual reviewer edits are allowed for future revisions, but must preserve the
  schema and update metadata to describe the revision.
