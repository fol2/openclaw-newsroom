from __future__ import annotations

import re
from typing import Any, Literal

from newsroom.event_manager import parse_clustering_response
from newsroom.lang_hint import detect_link_lang_hint

DatasetLang = Literal["en", "zh", "mixed"]
DatasetCorrectness = Literal["correct", "incorrect"]
DatasetTarget = int | Literal["new_event"]

_ALLOWED_LINK_FLAGS: frozenset[str] = frozenset({"roundup", "multi_topic", "opinion", "live_updates"})
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _to_dict(obj: Any) -> dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _normalise_flags(raw_flags: Any) -> list[str]:
    if not isinstance(raw_flags, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_flags:
        token = str(item).strip().lower()
        if not token or token in seen or token not in _ALLOWED_LINK_FLAGS:
            continue
        seen.add(token)
        out.append(token)
    return out


def label_language(*, title: str | None, description: str | None, existing_hint: Any | None = None) -> DatasetLang:
    """Label language with explicit mixed detection for bilingual headlines."""
    text = " ".join(part.strip() for part in (title, description) if isinstance(part, str) and part.strip())
    has_cjk = bool(_CJK_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_cjk and has_latin:
        return "mixed"
    return detect_link_lang_hint(title=title, description=description, existing_hint=existing_hint)


def prediction_target_from_action(enforced_action: dict[str, Any]) -> DatasetTarget:
    """Convert an enforced action object into event_id or new_event."""
    action_obj = _to_dict(enforced_action)
    action = str(action_obj.get("action") or "").strip().lower()
    if action == "assign":
        event_id = action_obj.get("event_id")
        if isinstance(event_id, int) and event_id > 0:
            return event_id
    return "new_event"


def label_target_from_action(
    enforced_action: dict[str, Any],
    *,
    top_candidate_score: float | None,
    low_score_threshold: float,
) -> DatasetTarget:
    """Derive the dataset label target.

    Assignments with very weak retrieval score are labelled as new_event in v1
    to seed a conservative review set for false positive assignment detection.
    """
    predicted = prediction_target_from_action(enforced_action)
    if isinstance(predicted, int):
        if top_candidate_score is not None and top_candidate_score < float(low_score_threshold):
            return "new_event"
    return predicted


def label_correctness(*, predicted_target: DatasetTarget, labelled_target: DatasetTarget) -> DatasetCorrectness:
    return "correct" if predicted_target == labelled_target else "incorrect"


def label_flags(
    *,
    enforced_action: dict[str, Any],
    top_candidate_score: float | None,
    low_score_threshold: float,
) -> list[str]:
    """Derive evaluation flags for later slicing/analysis."""
    action_obj = _to_dict(enforced_action)
    out = _normalise_flags(action_obj.get("link_flags"))

    enforcement = action_obj.get("enforcement")
    if isinstance(enforcement, dict) and isinstance(enforcement.get("reasons"), list) and enforcement["reasons"]:
        out.append("policy_override")

    if top_candidate_score is not None and top_candidate_score < float(low_score_threshold):
        out.append("low_retrieval_score")

    if str(action_obj.get("action") or "").strip().lower() == "development":
        out.append("development")

    deduped: list[str] = []
    seen: set[str] = set()
    for token in out:
        low = str(token).strip().lower()
        if not low or low in seen:
            continue
        seen.add(low)
        deduped.append(low)
    return deduped


def top_candidate_score(candidates: list[dict[str, Any]]) -> float | None:
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0] if isinstance(candidates[0], dict) else {}
    return _as_float(first.get("score"))


def _coerce_target(value: Any) -> DatasetTarget | None:
    if value == "new_event":
        return "new_event"
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        v = int(value.strip())
        if v > 0:
            return v
    return None


def parse_replay_enforced_action(
    *,
    link: dict[str, Any],
    candidate_events: list[dict[str, Any]],
    raw_llm_response: Any,
) -> dict[str, Any] | None:
    """Parse replay payload and return the enforced action object."""
    if not isinstance(raw_llm_response, dict):
        return None

    candidates = [c for c in candidate_events if isinstance(c, dict)]
    parsed = parse_clustering_response(raw_llm_response, _to_dict(link), candidates)
    if not isinstance(parsed, dict):
        return None

    enforced = _to_dict(parsed.get("enforced"))
    action = str(enforced.get("action") or "").strip().lower()
    if action not in {"assign", "development", "new_event"}:
        return None
    return enforced


def replay_record(
    record: dict[str, Any],
    *,
    low_score_threshold: float = 0.22,
) -> dict[str, Any]:
    """Replay one dataset row through parse_clustering_response offline."""
    replay = _to_dict(record.get("replay"))
    link = _to_dict(replay.get("link"))
    candidates_any = replay.get("candidate_events")
    candidates = [c for c in candidates_any if isinstance(c, dict)] if isinstance(candidates_any, list) else []
    response = replay.get("raw_llm_response")
    if not isinstance(response, dict):
        return {
            "parse_ok": False,
            "error": "raw_llm_response is not a JSON object",
            "target_match": False,
            "correctness_match": False,
        }

    enforced = parse_replay_enforced_action(link=link, candidate_events=candidates, raw_llm_response=response)
    if not isinstance(enforced, dict):
        return {
            "parse_ok": False,
            "error": "parse_clustering_response returned None",
            "target_match": False,
            "correctness_match": False,
        }
    predicted_target = prediction_target_from_action(enforced)

    labels = _to_dict(record.get("labels"))
    expected_target = _coerce_target(labels.get("event_id_or_new_event"))
    expected_correctness = str(labels.get("correctness") or "").strip().lower()

    target_match = expected_target is not None and predicted_target == expected_target
    actual_correctness = "correct" if target_match else "incorrect"
    correctness_match = expected_correctness in {"correct", "incorrect"} and expected_correctness == actual_correctness

    return {
        "parse_ok": True,
        "error": None,
        "predicted_target": predicted_target,
        "expected_target": expected_target,
        "actual_correctness": actual_correctness,
        "expected_correctness": expected_correctness,
        "target_match": target_match,
        "correctness_match": correctness_match,
        "replay_top_candidate_score": top_candidate_score(candidates),
        "replay_low_score_threshold": float(low_score_threshold),
    }
