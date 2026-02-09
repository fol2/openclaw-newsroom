"""Event-centric clustering manager for the newsroom.

Clusters links into events using one-link-per-prompt LLM calls (Gemini Flash).
Replaces the old token-based clustering (story_index.py) and per-URL
semantic fingerprinting (semantic_cluster.py).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Category list used in clustering prompts.
CATEGORY_LIST = [
    "UK Parliament / Politics",
    "UK News",
    "Politics",
    "AI",
    "Sports",
    "Hong Kong Entertainment",
    "Entertainment",
    "US Stocks",
    "Crypto",
    "Precious Metals",
    "Global News",
    "Hong Kong News",
]


def build_clustering_prompt(
    link: dict[str, Any],
    fresh_events: list[dict[str, Any]],
) -> str:
    """Build a one-link-per-prompt clustering prompt for Gemini."""
    title = str(link.get("title") or "").strip() or "(no title)"
    desc = str(link.get("description") or "").strip()

    link_section = f'Title: "{title}"'
    if desc:
        link_section += f'\nDescription: "{desc[:300]}"'

    # Build event list with parent-child structure.
    events_section_lines: list[str] = []
    if fresh_events:
        # Group children under parents.
        parent_map: dict[int, list[dict[str, Any]]] = {}
        root_events: list[dict[str, Any]] = []
        for ev in fresh_events:
            pid = ev.get("parent_event_id")
            if pid is not None:
                parent_map.setdefault(pid, []).append(ev)
            else:
                root_events.append(ev)

        for i, ev in enumerate(root_events[:50], start=1):
            eid = ev["id"]
            cat = ev.get("category") or "?"
            jur = ev.get("jurisdiction") or "?"
            summary = ev.get("summary_en") or ev.get("title") or "?"
            status = ev.get("status") or "?"
            events_section_lines.append(
                f'E{i}. [id={eid}] "{summary}" | {cat} | {jur} | status={status}'
            )
            # Show children (developments).
            children = parent_map.get(eid, [])
            for child in children[:5]:
                cid = child["id"]
                dev = child.get("development") or child.get("summary_en") or "?"
                events_section_lines.append(
                    f'  └─ [id={cid}] "{dev}" (development)'
                )

    events_section = "\n".join(events_section_lines) if events_section_lines else "(no fresh events)"

    categories_str = ", ".join(f'"{c}"' for c in CATEGORY_LIST)

    prompt = f"""You are a news event classifier.

This is a news link:
{link_section}

Here are the current FRESH EVENTS (last 48h):
{events_section}

Choose ONE action:

A) ASSIGN to existing event (same incident, different source, no new material)
   → {{"action":"assign","event_id":<id>}}

B) NEW DEVELOPMENT of existing event (significant escalation, result, reversal)
   → {{"action":"development","parent_event_id":<id>,
      "summary_en":"<one-sentence English summary>",
      "development":"<short label>",
      "category":"<category>","jurisdiction":"<jurisdiction>"}}

C) NEW EVENT (completely new story)
   → {{"action":"new_event",
      "summary_en":"<one-sentence English summary>",
      "category":"<category>","jurisdiction":"<jurisdiction>"}}

Categories: {categories_str}
Jurisdictions: "US", "UK", "HK", "CN", "EU", "JP", "KR", "GLOBAL", or other 2-letter code.

Rules:
- Same event = same incident/announcement, different source or angle → A
- Significant new development (result, escalation, arrest, resignation) → B
- Completely unrelated story → C
- Match across languages (Chinese headline about same event = same event)
- Only use ASSIGN if a matching event exists in the list above
- Do NOT assign to events with status=posted unless it's truly the same event

Return STRICT JSON only, no explanation."""

    return prompt


def parse_clustering_response(
    response: dict[str, Any],
    link: dict[str, Any],
    fresh_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate and normalize LLM clustering response.

    Returns a validated action dict or None if invalid.
    """
    action = str(response.get("action") or "").strip().lower()

    if action == "assign":
        event_id = response.get("event_id")
        if not isinstance(event_id, int):
            logger.warning("assign: event_id not int: %s", response)
            return None
        # Verify event_id exists in fresh_events.
        valid_ids = {ev["id"] for ev in fresh_events}
        if event_id not in valid_ids:
            logger.warning("assign: event_id %d not in fresh events", event_id)
            return None
        return {"action": "assign", "event_id": event_id}

    if action == "development":
        parent_id = response.get("parent_event_id")
        if not isinstance(parent_id, int):
            logger.warning("development: parent_event_id not int: %s", response)
            return None
        valid_ids = {ev["id"] for ev in fresh_events}
        if parent_id not in valid_ids:
            logger.warning("development: parent_event_id %d not in fresh events", parent_id)
            return None
        summary = str(response.get("summary_en") or "").strip()
        if not summary:
            logger.warning("development: empty summary_en")
            return None
        return {
            "action": "development",
            "parent_event_id": parent_id,
            "summary_en": summary,
            "development": str(response.get("development") or "").strip() or None,
            "category": str(response.get("category") or "").strip() or None,
            "jurisdiction": str(response.get("jurisdiction") or "").strip() or None,
        }

    if action in ("new_event", "new"):
        summary = str(response.get("summary_en") or "").strip()
        if not summary:
            logger.warning("new_event: empty summary_en")
            return None
        return {
            "action": "new_event",
            "summary_en": summary,
            "category": str(response.get("category") or "").strip() or None,
            "jurisdiction": str(response.get("jurisdiction") or "").strip() or None,
        }

    logger.warning("Unknown clustering action: %s", action)
    return None


def cluster_link(
    *,
    link: dict[str, Any],
    fresh_events: list[dict[str, Any]],
    gemini: Any,
    db: Any,
) -> dict[str, Any] | None:
    """Classify a single link against fresh events using Gemini.

    Returns the action result dict or None on failure.
    """
    prompt = build_clustering_prompt(link, fresh_events)

    try:
        response = gemini.generate_json(prompt)
    except Exception as e:
        logger.warning("Gemini call failed for link %s: %s", link.get("norm_url", "?"), e)
        return None

    if not response:
        logger.warning("Empty Gemini response for link %s", link.get("norm_url", "?"))
        return None

    validated = parse_clustering_response(response, link, fresh_events)
    if not validated:
        return None

    link_id = link.get("id")
    if not isinstance(link_id, int):
        logger.warning("link missing id")
        return None

    link_title = str(link.get("title") or "").strip() or None
    link_url = str(link.get("url") or "").strip() or None

    action = validated["action"]

    if action == "assign":
        event_id = validated["event_id"]
        db.assign_link_to_event(link_id=link_id, event_id=event_id)
        logger.info("Assigned link %d to event %d", link_id, event_id)
        return validated

    if action == "development":
        parent_id = validated["parent_event_id"]
        event_id = db.create_event(
            summary_en=validated["summary_en"],
            category=validated.get("category"),
            jurisdiction=validated.get("jurisdiction"),
            title=link_title,
            primary_url=link_url,
            parent_event_id=parent_id,
            development=validated.get("development"),
            model="gemini-flash",
        )
        db.assign_link_to_event(link_id=link_id, event_id=event_id)
        validated["event_id"] = event_id
        logger.info("Created development event %d (parent=%d) for link %d", event_id, parent_id, link_id)
        return validated

    if action == "new_event":
        event_id = db.create_event(
            summary_en=validated["summary_en"],
            category=validated.get("category"),
            jurisdiction=validated.get("jurisdiction"),
            title=link_title,
            primary_url=link_url,
            model="gemini-flash",
        )
        db.assign_link_to_event(link_id=link_id, event_id=event_id)
        validated["event_id"] = event_id
        logger.info("Created new event %d for link %d", event_id, link_id)
        return validated

    return None


def build_merge_prompt(events: list[dict[str, Any]], category: str) -> str:
    """Build a prompt asking the LLM to group duplicate events within a category."""
    lines: list[str] = []
    for ev in events:
        eid = ev["id"]
        summary = ev.get("summary_en") or ev.get("title") or "?"
        lc = ev.get("link_count", 0)
        jur = ev.get("jurisdiction") or "?"
        lines.append(f'{eid}. "{summary}" | {lc} links | {jur}')

    events_block = "\n".join(lines)

    return f"""You are a news deduplication engine.

Below are {len(events)} events from the "{category}" category.
Group events that cover the SAME underlying story (same incident, same announcement, same match).

Events:
{events_block}

Rules:
- SAME story = same specific incident/announcement reported by different sources → group them
- Do NOT merge: related-but-distinct stories, parent vs child developments, or different matches/incidents
- Only group events you are confident are duplicates

Return STRICT JSON with plain integer IDs (the number before each event's period):
{{"groups": [[42, 55], [78, 82]], "no_merge": [90, 91]}}

IMPORTANT: Use plain integers only. NOT strings, NOT "E42", just 42.
- "groups": list of lists, each inner list has 2+ event IDs that should be merged
- "no_merge": list of event IDs that should NOT be merged with anything
- Every event ID must appear exactly once across groups + no_merge"""


def parse_merge_response(
    response: dict[str, Any] | None,
    valid_ids: set[int],
) -> list[list[int]] | None:
    """Validate LLM merge response. Returns list of merge groups or None on failure."""
    if not response or not isinstance(response, dict):
        return None

    groups_raw = response.get("groups")
    if not isinstance(groups_raw, list):
        return None

    seen: set[int] = set()
    valid_groups: list[list[int]] = []

    for group in groups_raw:
        if not isinstance(group, list):
            continue
        # Filter to valid integer IDs not yet seen.
        clean: list[int] = []
        for item in group:
            if isinstance(item, int) and item in valid_ids and item not in seen:
                clean.append(item)
                seen.add(item)
        # Only keep groups with 2+ members.
        if len(clean) >= 2:
            valid_groups.append(clean)

    return valid_groups if valid_groups else None


def merge_events(
    *,
    db: Any,
    gemini: Any,
    delay_seconds: float = 3.0,
    batch_size: int = 50,
    include_expired: bool = False,
    max_consecutive_failures: int = 3,
) -> dict[str, int]:
    """Post-clustering merge pass: compare event summaries within each category,
    merge duplicates using LLM.

    Args:
        include_expired: If True, merge across ALL events (not just fresh).
                         Useful for one-off dedup of the entire table.

    Returns stats dict.
    """
    stats = {
        "categories_processed": 0,
        "llm_calls": 0,
        "groups_merged": 0,
        "events_removed": 0,
        "links_moved": 0,
        "errors": 0,
    }
    consecutive_failures = 0

    fresh_events = db.get_fresh_events(now_ts=0 if include_expired else None)
    # Filter to root events only (no children), status in ('new', 'active').
    root_events = [
        ev for ev in fresh_events
        if ev.get("parent_event_id") is None
        and ev.get("status") in ("new", "active")
    ]

    if not root_events:
        return stats

    # Group by category.
    by_category: dict[str, list[dict[str, Any]]] = {}
    for ev in root_events:
        cat = ev.get("category") or "Other"
        by_category.setdefault(cat, []).append(ev)

    for cat, cat_events in by_category.items():
        if consecutive_failures >= max_consecutive_failures:
            logger.warning(
                "Merge early exit: %d consecutive LLM failures, skipping remaining categories",
                consecutive_failures,
            )
            break

        if len(cat_events) < 2:
            continue

        stats["categories_processed"] += 1

        # Sort alphabetically so similar summaries land in the same chunk.
        cat_events.sort(key=lambda e: (e.get("summary_en") or "").lower())

        # Sliding window: overlap adjacent batches so boundary pairs get compared.
        overlap = min(10, batch_size // 3)
        stride = max(batch_size - overlap, 1)
        merged_ids: set[int] = set()  # Track already-merged losers across windows.

        chunk_start = 0
        while chunk_start < len(cat_events):
            chunk = cat_events[chunk_start : chunk_start + batch_size]
            # Skip events already merged as losers in earlier windows.
            chunk = [ev for ev in chunk if ev["id"] not in merged_ids]
            if len(chunk) < 2:
                chunk_start += stride
                continue

            prompt = build_merge_prompt(chunk, cat)

            try:
                raw_text = gemini.generate(prompt)
            except Exception as e:
                logger.warning("Merge LLM call failed for category %s: %s", cat, e)
                stats["errors"] += 1
                consecutive_failures += 1
                chunk_start += stride
                continue

            stats["llm_calls"] += 1

            if not raw_text:
                logger.warning("Empty merge response for category %s", cat)
                stats["errors"] += 1
                consecutive_failures += 1
                chunk_start += stride
                continue

            consecutive_failures = 0

            # Sanitize: LLM sometimes returns E123 instead of 123.
            sanitized = re.sub(r'\bE(\d+)\b', r'\1', raw_text)
            try:
                response = json.loads(
                    sanitized[sanitized.find("{"):sanitized.rfind("}") + 1]
                )
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "Failed to parse merge JSON for category %s: %s...",
                    cat, raw_text[:200],
                )
                stats["errors"] += 1
                chunk_start += stride
                continue

            valid_ids = {ev["id"] for ev in chunk}
            groups = parse_merge_response(response, valid_ids)
            if not groups:
                chunk_start += stride
                continue

            # Build lookup for winner selection.
            ev_lookup = {ev["id"]: ev for ev in chunk}

            for group in groups:
                # Pick winner: most links, tiebreak oldest (smallest created_at_ts).
                group_events = [ev_lookup[eid] for eid in group if eid in ev_lookup]
                if len(group_events) < 2:
                    continue

                winner = max(
                    group_events,
                    key=lambda e: (e.get("link_count", 0), -(e.get("created_at_ts", 0))),
                )
                loser_ids = [e["id"] for e in group_events if e["id"] != winner["id"]]

                try:
                    links_moved = db.merge_events_into(
                        winner_id=winner["id"], loser_ids=loser_ids
                    )
                    stats["groups_merged"] += 1
                    stats["events_removed"] += len(loser_ids)
                    stats["links_moved"] += links_moved
                    merged_ids.update(loser_ids)
                    logger.info(
                        "Merged %d events into %d (winner), %d links moved",
                        len(loser_ids), winner["id"], links_moved,
                    )
                except Exception as e:
                    logger.warning("Merge DB error for group %s: %s", group, e)
                    stats["errors"] += 1

            if delay_seconds > 0:
                time.sleep(delay_seconds)

            chunk_start += stride

    logger.info(
        "Merge pass complete: %d categories, %d LLM calls, %d groups merged, "
        "%d events removed, %d links moved, %d errors",
        stats["categories_processed"], stats["llm_calls"], stats["groups_merged"],
        stats["events_removed"], stats["links_moved"], stats["errors"],
    )
    return stats


def cluster_all_pending(
    *,
    db: Any,
    gemini: Any,
    max_links: int = 100,
    delay_seconds: float = 3.0,
    max_consecutive_failures: int = 3,
) -> dict[str, int]:
    """Process all unassigned links sequentially, oldest first.

    Returns stats: {"processed": N, "assigned": N, "new_events": N, "developments": N, "errors": N, "skipped": N}
    """
    links = db.get_unassigned_links()
    if not links:
        return {"processed": 0, "assigned": 0, "new_events": 0, "developments": 0, "errors": 0, "skipped": 0}

    links = links[:max_links]

    stats = {"processed": 0, "assigned": 0, "new_events": 0, "developments": 0, "errors": 0, "skipped": 0}
    consecutive_failures = 0

    for link in links:
        # Early exit: if Gemini is consistently failing, stop wasting quota.
        if consecutive_failures >= max_consecutive_failures:
            stats["skipped"] = len(links) - stats["processed"]
            logger.warning(
                "Early exit: %d consecutive Gemini failures, skipping remaining %d links",
                consecutive_failures, stats["skipped"],
            )
            break

        # Refresh fresh events each iteration (new events from prior links are now visible).
        fresh_events = db.get_fresh_events()

        result = cluster_link(link=link, fresh_events=fresh_events, gemini=gemini, db=db)
        stats["processed"] += 1

        if result is None:
            stats["errors"] += 1
            consecutive_failures += 1
        else:
            consecutive_failures = 0
            if result["action"] == "assign":
                stats["assigned"] += 1
            elif result["action"] == "new_event":
                stats["new_events"] += 1
            elif result["action"] == "development":
                stats["developments"] += 1

        # Rate limiting between Gemini calls.
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    logger.info(
        "Clustering complete: %d processed, %d assigned, %d new, %d dev, %d errors, %d skipped",
        stats["processed"], stats["assigned"], stats["new_events"],
        stats["developments"], stats["errors"], stats["skipped"],
    )
    return stats
