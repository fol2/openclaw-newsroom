from __future__ import annotations

import json
from pathlib import Path

from newsroom.eval_dataset import (
    label_flags,
    label_language,
    label_target_from_action,
    replay_record,
)
from scripts.replay_clustering_eval_dataset import main as replay_main


def _base_record(*, labelled_target: int | str, correctness: str) -> dict:
    return {
        "schema_version": "clustering_eval_sample_v1",
        "sample_id": "sample_0001",
        "labels": {
            "event_id_or_new_event": labelled_target,
            "language": "en",
            "correctness": correctness,
            "flags": [],
        },
        "replay": {
            "link": {
                "id": 1,
                "url": "https://example.com/a",
                "norm_url": "https://example.com/a",
                "title": "Example policy update",
                "description": "Government says changes begin next week",
                "lang_hint": "en",
            },
            "candidate_events": [
                {
                    "id": 101,
                    "status": "new",
                    "summary_en": "Government policy update",
                }
            ],
            "raw_llm_response": {
                "action": "assign",
                "event_id": 101,
                "confidence": 0.92,
                "summary_en": "Government policy update",
                "match_basis": ["entity_overlap"],
                "link_flags": [],
            },
        },
    }


def test_label_language_marks_bilingual_as_mixed() -> None:
    assert label_language(title="港鐵 MTR service update", description="", existing_hint=None) == "mixed"


def test_label_target_conservative_low_score_relabels_assign() -> None:
    enforced = {"action": "assign", "event_id": 1001}
    assert label_target_from_action(enforced, top_candidate_score=0.10, low_score_threshold=0.22) == "new_event"
    assert label_target_from_action(enforced, top_candidate_score=0.35, low_score_threshold=0.22) == 1001


def test_label_flags_adds_policy_and_low_score_markers() -> None:
    enforced = {
        "action": "development",
        "link_flags": ["roundup", "unknown_flag"],
        "enforcement": {"reasons": ["missing_confidence"]},
    }
    flags = label_flags(enforced_action=enforced, top_candidate_score=0.15, low_score_threshold=0.22)
    assert flags == ["roundup", "policy_override", "low_retrieval_score", "development"]


def test_replay_record_matches_expected_target_and_correctness() -> None:
    record = _base_record(labelled_target=101, correctness="correct")
    result = replay_record(record)
    assert result["parse_ok"] is True
    assert result["target_match"] is True
    assert result["correctness_match"] is True


def test_replay_record_supports_intentional_incorrect_labels() -> None:
    record = _base_record(labelled_target="new_event", correctness="incorrect")
    result = replay_record(record)
    assert result["parse_ok"] is True
    assert result["target_match"] is False
    assert result["actual_correctness"] == "incorrect"
    assert result["correctness_match"] is True


def test_replay_cli_fails_on_correctness_mismatch(tmp_path: Path) -> None:
    row = _base_record(labelled_target=101, correctness="incorrect")
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    rc = replay_main(
        [
            "--dataset",
            str(dataset_path),
            "--fail-on-correctness-mismatch",
        ]
    )
    assert rc == 1


def test_replay_cli_writes_summary(tmp_path: Path) -> None:
    row = _base_record(labelled_target=101, correctness="correct")
    dataset_path = tmp_path / "dataset.jsonl"
    summary_path = tmp_path / "summary.json"
    dataset_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    rc = replay_main(
        [
            "--dataset",
            str(dataset_path),
            "--summary-out",
            str(summary_path),
            "--fail-on-parse-error",
            "--fail-on-correctness-mismatch",
        ]
    )

    assert rc == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["summary"]["rows"] == 1
    assert payload["summary"]["parse_ok_rows"] == 1
    assert payload["summary"]["correctness_match_rows"] == 1
