#!/usr/bin/env python3
"""Consume durable Graphiti revision events independently from source polling."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from newsroom.authority.auth import AuthenticationProof
from newsroom.authority.canonical import (
    canonical_json_bytes,
    digest_bytes,
)
from newsroom.authority.graphiti_increment4_system import (
    GovernedGraphitiIncrement4AuthoritySystem,
)
from newsroom.control_plane.cycle import (
    GRAPHITI_UNIT_RESERVATION_GBP_MICROUNITS,
    consume_next_graphiti_event,
    load_graphiti_units,
    qualify_fresh_graphiti_event,
)
from newsroom.control_plane.corpus import CorpusIngestUnit
from newsroom.control_plane.graphiti import EvaluationGraphitiRunner
from newsroom.control_plane.graphiti_admission_integration import (
    compose_existing_graphiti_admission_consumer,
)
from newsroom.control_plane.graphiti_admission import (
    GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION,
    graphiti_admission_telemetry,
    graphiti_decided_cohort_generation_identity,
    graphiti_projection_reconciliation_from_json,
)
from newsroom.control_plane.graphiti_events import GraphitiProcessResult
from newsroom.control_plane.graphiti_steady_state import (
    GraphitiCampaignRuntime,
    PRE_FRONTIER_BACKLOG_HOLD_REASON,
    _exact_admission_reconciliation,
    _mint_graphiti_campaign_runtime,
    graphiti_graph_destination_identity,
    graphiti_graph_destination_readback,
    graphiti_operational_partition_snapshot,
    graphiti_store_snapshot_digests,
    validate_graphiti_campaign_packet,
)
from newsroom.control_plane.read_only_snapshot import read_only_snapshot
from newsroom.control_plane.paths import (
    CANONICAL_PROVING_STORE,
    CANONICAL_UNPUBLISHED_STORE,
    ensure_control_plane_state_root,
)
from newsroom.control_plane.store import connect


class GraphitiCampaignStop(RuntimeError):
    """A bounded campaign invariant stopped all later effects."""

    def __init__(
        self,
        message: str,
        *,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.evidence = None if evidence is None else dict(evidence)


def qualify_campaign_adapter_runtime() -> None:
    """Load the pinned Graphiti extra before any campaign event claim."""

    from newsroom.graphiti_adapter.real import _load_graphiti
    from newsroom.graphiti_adapter.types import (
        GraphitiAdapterContractError,
        graphiti_setup_failure_detail,
    )

    try:
        _load_graphiti()
    except GraphitiAdapterContractError as exc:
        evidence: dict[str, object] = {
            "stage": "PRE_DISPATCH_ADAPTER_RUNTIME",
            "failure_type": type(exc).__name__,
            "provider_dispatched": False,
        }
        detail = graphiti_setup_failure_detail(exc)
        if detail is not None:
            evidence["setup_failure_detail"] = detail
        raise GraphitiCampaignStop(
            "campaign Graphiti adapter runtime is unavailable",
            evidence=evidence,
        ) from exc


GovernedGraphitiWorkerRuntime = GraphitiCampaignRuntime


def compose_governed_graphiti_worker_runtime(
    *,
    authority_system: GovernedGraphitiIncrement4AuthoritySystem,
    authority_store_descriptor_digest: str,
    proof: AuthenticationProof,
    bind_unit_authority: Callable[[CorpusIngestUnit], CorpusIngestUnit],
    expected_authority_store_path: str | None = None,
    max_attempts: int = 1,
) -> GovernedGraphitiWorkerRuntime:
    """Wire one exact 4A-4E authority system into the bounded worker."""

    if not isinstance(
        authority_system, GovernedGraphitiIncrement4AuthoritySystem
    ):
        raise TypeError("combined Increment 4 authority system is required")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise TypeError("admission attempt cap must be an integer")
    if max_attempts <= 0:
        raise ValueError("admission attempt cap must be positive")
    authority_store_path = authority_system.authority_store_path
    graph_destination_id = authority_system.graph_destination_id
    if expected_authority_store_path is not None and str(
        Path(expected_authority_store_path).expanduser().resolve()
    ) != authority_store_path:
        raise ValueError("combined authority-store path differs from operator input")
    adapter = authority_system.graphiti
    extraction = authority_system.extraction
    objects = authority_system.objects
    entities = authority_system.entities
    relations = authority_system.relations
    increment4 = authority_system.increment4
    graphiti = EvaluationGraphitiRunner(
        fallback_permitted=False,
        proposal_adapter=adapter,
        extraction_records=extraction,
        proof=proof,
    )

    def admission_factory(connection: sqlite3.Connection):
        return compose_existing_graphiti_admission_consumer(
            connection,
            adapter=adapter,
            extraction=extraction,
            objects=objects,
            entities=entities,
            relations=relations,
            increment4=increment4,
            proof=proof,
            max_attempts=max_attempts,
        )

    def graph_state_fence(
        campaign: Mapping[str, object],
    ) -> Mapping[str, object]:
        expected = _mapping(
            campaign.get("graph_destination_readback"),
            field="campaign graph destination readback",
        )
        reconciliation = increment4.reconcile_active(proof=proof)
        actual = graphiti_graph_destination_readback(
            destination_id=graph_destination_id,
            reconciliation=reconciliation,
        )
        if graphiti_graph_destination_identity(
            actual
        ) != graphiti_graph_destination_identity(expected):
            raise GraphitiCampaignStop("campaign graph identity drifted")
        return actual

    return _mint_graphiti_campaign_runtime(
        graphiti=graphiti,
        admission_factory=admission_factory,
        bind_unit_authority=bind_unit_authority,
        graph_state_fence=graph_state_fence,
        authority_store_source_path=authority_store_path,
        authority_store_descriptor_digest=authority_store_descriptor_digest,
        graph_destination_id=graph_destination_id,
    )


def _run(
    *,
    consume: Callable[[], GraphitiProcessResult | None],
) -> int:
    """Run exactly one selected event and fail closed on any other result."""

    try:
        result = consume()
    except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
        _emit_stop("EXACT_EVENT_EXECUTION_REFUSED", completed=0)
        print(
            json.dumps(
                {
                    "event": "GRAPHITI_WORKER_DIAGNOSTIC",
                    "failure_type": type(exc).__name__,
                    "failure": str(exc),
                    "public_dispatch": False,
                    "auto_publish": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 2
    _emit_result(result)
    if result is None:
        _emit_stop("EXACT_EVENT_NOT_CLAIMED", completed=0)
        return 2
    if result.state != "TERMINAL":
        _emit_stop("NON_TERMINAL_EVENT_RESULT", completed=0, result=result)
        return 2
    _emit_stop("EXACT_EVENT_TERMINAL", completed=1, result=result)
    return 0


def _emit_result(result: GraphitiProcessResult | None) -> None:
    print(
        json.dumps(
            {
                "event": "GRAPHITI_EVENT_IDLE"
                if result is None
                else "GRAPHITI_EVENT_RESULT",
                "result": None if result is None else asdict(result),
                "public_dispatch": False,
                "auto_publish": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _emit_stop(
    reason: str,
    *,
    completed: int,
    result: GraphitiProcessResult | None = None,
) -> None:
    print(
        json.dumps(
            {
                "event": "GRAPHITI_WORKER_STOPPED",
                "reason": reason,
                "completed_events": completed,
                "result": None if result is None else asdict(result),
                "public_dispatch": False,
                "auto_publish": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _current_git_identity() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise GraphitiCampaignStop("campaign requires a clean checkout")
    head = git("rev-parse", "HEAD")
    if head != git("rev-parse", "origin/main"):
        raise GraphitiCampaignStop("campaign requires exact origin/main")
    return head, git("rev-parse", "HEAD^{tree}")


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GraphitiCampaignStop(f"{field} is not a mapping")
    return value


def _integer(
    value: object,
    *,
    field: str,
    positive: bool = False,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (1 if positive else 0)
    ):
        raise GraphitiCampaignStop(f"{field} is invalid")
    return value


def _campaign_receipt_evidence(
    unpublished_store: str,
    *,
    ingest_ids: tuple[str, ...],
    provider: Mapping[str, object],
) -> dict[str, object]:
    """Recheck terminal output and exact provider identities after each effect."""

    if not ingest_ids:
        raise GraphitiCampaignStop("campaign event resolved no ingest identities")
    connection = sqlite3.connect(
        f"{Path(unpublished_store).absolute().as_uri()}?mode=ro",
        uri=True,
    )
    try:
        retained: list[tuple[object, ...]] = []
        for ingest_id in ingest_ids:
            terminal = connection.execute(
                "SELECT ingest.outcome,ingest.proposal_count,"
                "ingest.receipt_digest,receipt.receipt_json "
                "FROM unpublished_graphiti_ingest AS ingest "
                "JOIN unpublished_graphiti_receipts AS receipt USING(ingest_id) "
                "WHERE ingest.ingest_id=?",
                (ingest_id,),
            ).fetchone()
            attempts = connection.execute(
                "SELECT attempt_number,outcome,receipt_digest,receipt_json "
                "FROM unpublished_graphiti_attempt_receipts WHERE ingest_id=? "
                "ORDER BY attempt_number",
                (ingest_id,),
            ).fetchall()
            spends = connection.execute(
                "SELECT attempt_number,reserved_gbp_microunits,"
                "actual_usd_microunits,actual_gbp_microunits,usage_basis,status,"
                "provider_usage_json,dispatch_owner,dispatch_lease_expires_at "
                "FROM unpublished_graphiti_spend WHERE ingest_id=? "
                "ORDER BY attempt_number",
                (ingest_id,),
            ).fetchall()
            retained.append((ingest_id, terminal, attempts, spends))
    finally:
        connection.close()

    provider_id = str(provider.get("provider_id") or "")
    transport_id = str(provider.get("transport_id") or "")
    model_id = str(provider.get("model_id") or "")
    embedding_provider_id = str(provider.get("embedding_provider_id") or "")
    embedding_model_id = str(provider.get("embedding_model_id") or "")
    if not all(
        (
            provider_id,
            transport_id,
            model_id,
            embedding_provider_id,
            embedding_model_id,
        )
    ):
        raise GraphitiCampaignStop("campaign provider identities are incomplete")
    proposal_count = 0
    chat_count = 0
    embedding_count = 0
    actual_gbp_microunits = 0
    invocation_ids: set[str] = set()
    embedding_invocation_ids: set[str] = set()
    for ingest_id, terminal, attempts, spends in retained:
        if terminal is None or len(attempts) != 1 or len(spends) != 1:
            raise GraphitiCampaignStop(
                "campaign attempt or spend denominator differs"
            )
        outcome, raw_count, receipt_digest, receipt_json = terminal
        try:
            receipt = json.loads(str(receipt_json))
        except json.JSONDecodeError as exc:
            raise GraphitiCampaignStop(
                "campaign terminal receipt is malformed"
            ) from exc
        if not isinstance(receipt, dict):
            raise GraphitiCampaignStop("campaign terminal receipt is malformed")
        unsigned = dict(receipt)
        supplied_digest = unsigned.pop("receipt_digest", None)
        if (
            outcome != "COMPLETE"
            or receipt.get("ingest_id") != str(ingest_id)
            or supplied_digest != str(receipt_digest)
            or digest_bytes(canonical_json_bytes(unsigned)) != str(receipt_digest)
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 0
            or receipt.get("proposal_count") != raw_count
            or receipt.get("failure_code") != "NONE"
            or receipt.get("attempt_number") != 1
            or receipt.get("provider_attempt_number") != 1
        ):
            raise GraphitiCampaignStop(
                "campaign terminal receipt integrity differs"
            )
        attempt_number, attempt_outcome, attempt_digest, attempt_json = attempts[0]
        if (
            attempt_number != 1
            or attempt_outcome != "COMPLETE"
            or attempt_digest != receipt_digest
            or str(attempt_json) != str(receipt_json)
        ):
            raise GraphitiCampaignStop("campaign provider attempt receipt differs")
        proposal_count += raw_count
        invocations = receipt.get("chat_invocations")
        if not isinstance(invocations, list) or len(invocations) != 1:
            raise GraphitiCampaignStop("campaign provider receipt is incomplete")
        for invocation in invocations:
            usage = invocation.get("usage") if isinstance(invocation, Mapping) else None
            transport = (
                invocation.get("transport_qualification")
                if isinstance(invocation, Mapping)
                else None
            )
            invocation_id = (
                invocation.get("model_invocation_id")
                if isinstance(invocation, Mapping)
                else None
            )
            if (
                not isinstance(invocation, Mapping)
                or invocation.get("provider") != provider_id
                or invocation.get("model") != model_id
                or invocation.get("outcome") != "COMPLETE"
                or not isinstance(usage, Mapping)
                or usage.get("usage_basis") != "PROVIDER_REPORTED"
                or not isinstance(transport, Mapping)
                or transport.get("transport") != transport_id
                or transport.get("max_retries") != 0
                or not isinstance(invocation_id, str)
                or not invocation_id
                or invocation_id in invocation_ids
            ):
                raise GraphitiCampaignStop(
                    "campaign provider identity, transport, usage or retry drifted"
                )
            invocation_ids.add(invocation_id)
            chat_count += 1
        embedding = receipt.get("embedding_usage")
        if not isinstance(embedding, Mapping):
            raise GraphitiCampaignStop("campaign embedding receipt is incomplete")
        requests = embedding.get("requests")
        request_count = embedding.get("request_count")
        if (
            not isinstance(requests, list)
            or isinstance(request_count, bool)
            or not isinstance(request_count, int)
            or request_count != len(requests)
            or embedding.get("usage_basis")
            != ("PROVIDER_REPORTED" if requests else "NO_EMBEDDING_CALL")
        ):
            raise GraphitiCampaignStop("campaign embedding denominator differs")
        embedding_cost = 0
        for request in requests:
            invocation_id = (
                request.get("model_invocation_id")
                if isinstance(request, Mapping)
                else None
            )
            request_cost = (
                request.get("cost_usd_microunits")
                if isinstance(request, Mapping)
                else None
            )
            if (
                not isinstance(request, Mapping)
                or request.get("provider") != embedding_provider_id
                or request.get("model") != embedding_model_id
                or request.get("outcome") != "COMPLETE"
                or request.get("cost_reported") is not True
                or isinstance(request_cost, bool)
                or not isinstance(request_cost, int)
                or request_cost < 0
                or not isinstance(invocation_id, str)
                or not invocation_id
                or invocation_id in embedding_invocation_ids
            ):
                raise GraphitiCampaignStop("campaign embedding identity drifted")
            embedding_invocation_ids.add(invocation_id)
            embedding_cost += request_cost
            embedding_count += 1
        if embedding.get("cost_usd_microunits") != embedding_cost:
            raise GraphitiCampaignStop("campaign embedding cost differs")

        (
            spend_attempt,
            reserved_gbp,
            actual_usd,
            actual_gbp,
            usage_basis,
            spend_status,
            provider_usage_json,
            dispatch_owner,
            dispatch_lease_expires_at,
        ) = spends[0]
        try:
            provider_usage = json.loads(str(provider_usage_json))
        except (TypeError, json.JSONDecodeError) as exc:
            raise GraphitiCampaignStop("campaign spend usage is malformed") from exc
        accounting = receipt.get("accounting")
        if (
            spend_attempt != 1
            or spend_status != "RECONCILED"
            or usage_basis != embedding.get("usage_basis")
            or provider_usage != dict(embedding)
            or isinstance(reserved_gbp, bool)
            or not isinstance(reserved_gbp, int)
            or reserved_gbp < 0
            or isinstance(actual_usd, bool)
            or not isinstance(actual_usd, int)
            or actual_usd != embedding_cost
            or isinstance(actual_gbp, bool)
            or not isinstance(actual_gbp, int)
            or actual_gbp < 0
            or actual_gbp > reserved_gbp
            or dispatch_owner is not None
            or dispatch_lease_expires_at is not None
            or not isinstance(accounting, Mapping)
            or accounting.get("spend_id") != f"{ingest_id}:1"
            or accounting.get("status") != "RECONCILED"
            or accounting.get("usage_basis") != usage_basis
            or accounting.get("actual_usd_microunits") != actual_usd
            or accounting.get("actual_gbp_microunits") != actual_gbp
            or accounting.get("unused_reservation_released") is not True
        ):
            raise GraphitiCampaignStop("campaign spend accounting drifted")
        actual_gbp_microunits += actual_gbp
    return {
        "proposal_count": proposal_count,
        "chat_invocation_count": chat_count,
        "embedding_request_count": embedding_count,
        "fallback_count": max(0, chat_count - len(ingest_ids)),
        "retry_count": 0,
        "actual_gbp_microunits": actual_gbp_microunits,
    }


def _assert_fresh_campaign_ingests(
    unpublished_store: str,
    *,
    ingest_ids: tuple[str, ...],
) -> None:
    """Stop before F4 dispatch when an exact ingest already has effects."""

    placeholders = ",".join("?" for _ in ingest_ids)
    connection = sqlite3.connect(
        f"{Path(unpublished_store).absolute().as_uri()}?mode=ro",
        uri=True,
    )
    try:
        for table in (
            "unpublished_graphiti_ingest",
            "unpublished_graphiti_receipts",
            "unpublished_graphiti_attempt_receipts",
            "unpublished_graphiti_spend",
            "unpublished_graphiti_failures",
            "unpublished_graphiti_admission_queue",
        ):
            count = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} "
                    f"WHERE ingest_id IN ({placeholders})",
                    ingest_ids,
                ).fetchone()[0]
            )
            if count:
                raise GraphitiCampaignStop(
                    "campaign exact ingest identities already have retained effects"
                )
    finally:
        connection.close()


def _campaign_admission_rows(
    connection: sqlite3.Connection,
    *,
    ingest_ids: tuple[str, ...],
) -> tuple[tuple[str, str, str | None], ...]:
    placeholders = ",".join("?" for _ in ingest_ids)
    return tuple(
        (str(row[0]), str(row[1]), None if row[2] is None else str(row[2]))
        for row in connection.execute(
            "SELECT queue.ingest_id,queue.proposal_kind,decision.action "
            "FROM unpublished_graphiti_admission_queue AS queue "
            "LEFT JOIN unpublished_graphiti_admission_decisions AS decision "
            "USING(proposal_key) "
            f"WHERE queue.ingest_id IN ({placeholders}) "
            "ORDER BY queue.ingest_id,queue.queue_seq",
            ingest_ids,
        )
    )


def _assert_admission_caps(
    rows: tuple[tuple[str, str, str | None], ...],
    *,
    ingest_groups: tuple[tuple[str, ...], ...],
    caps: Mapping[str, object],
    decisions_required: bool,
) -> dict[str, int]:
    per_event = _mapping(caps.get("per_event"), field="per-event caps")
    total = _mapping(caps.get("total"), field="total caps")
    counts = {
        "proposals": len(rows),
        "entity_admits": sum(
            action == "ADMIT" and kind != "RELATION"
            for _ingest, kind, action in rows
        ),
        "relation_admits": sum(
            action == "ADMIT" and kind == "RELATION"
            for _ingest, kind, action in rows
        ),
    }
    counts["effects"] = counts["entity_admits"] + counts["relation_admits"]
    for name, count in counts.items():
        if count > _integer(total.get(name), field=f"total {name} cap"):
            raise GraphitiCampaignStop(f"total {name} cap reached")
    for ingest_group in ingest_groups:
        event_rows = tuple(row for row in rows if row[0] in ingest_group)
        event_counts = {
            "proposals": len(event_rows),
            "entity_admits": sum(
                action == "ADMIT" and kind != "RELATION"
                for _ingest, kind, action in event_rows
            ),
            "relation_admits": sum(
                action == "ADMIT" and kind == "RELATION"
                for _ingest, kind, action in event_rows
            ),
        }
        if not decisions_required:
            # Before 4B/4C, every proposal is a conservative upper bound on an
            # admit and projected effect. This stops before canonical writes.
            event_counts["entity_admits"] = sum(
                kind != "RELATION" for _ingest, kind, _action in event_rows
            )
            event_counts["relation_admits"] = sum(
                kind == "RELATION" for _ingest, kind, _action in event_rows
            )
        event_counts["effects"] = (
            event_counts["entity_admits"] + event_counts["relation_admits"]
        )
        for name, count in event_counts.items():
            if count > _integer(
                per_event.get(name), field=f"per-event {name} cap"
            ):
                raise GraphitiCampaignStop(f"per-event {name} cap reached")
    if not decisions_required:
        conservative_entity = sum(kind != "RELATION" for _, kind, _ in rows)
        conservative_relation = sum(kind == "RELATION" for _, kind, _ in rows)
        conservative = {
            "proposals": len(rows),
            "entity_admits": conservative_entity,
            "relation_admits": conservative_relation,
            "effects": conservative_entity + conservative_relation,
        }
        for name, count in conservative.items():
            if count > _integer(total.get(name), field=f"total {name} cap"):
                raise GraphitiCampaignStop(f"total {name} cap reached")
        return conservative
    if any(action is None for _ingest, _kind, action in rows):
        raise GraphitiCampaignStop("campaign admission cohort is not fully decided")
    return counts


def _campaign_reconciliation_ids(
    connection: sqlite3.Connection,
) -> frozenset[str]:
    return frozenset(
        str(row[0])
        for row in connection.execute(
            "SELECT generation_id FROM "
            "unpublished_graphiti_projection_reconciliations"
        )
    )


def _utc_instant(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise GraphitiCampaignStop(f"{field} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GraphitiCampaignStop(f"{field} is invalid") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _campaign_operational_partition_snapshot(
    *,
    proving_store: str,
    unpublished_store: str,
    authority_store: str,
    observed_at: datetime,
    unpublished_connection: sqlite3.Connection | None = None,
    pre_frontier_hold_watermark: int | None = None,
) -> dict[str, object]:
    """Read the shared exact operational partition without selecting work."""

    owned: list[sqlite3.Connection] = []
    connections: list[sqlite3.Connection] = []
    try:
        for path_value in (proving_store, authority_store):
            path = Path(path_value).expanduser().resolve()
            connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            owned.append(connection)
            connections.append(connection)
        if unpublished_connection is None:
            path = Path(unpublished_store).expanduser().resolve()
            unpublished_connection = sqlite3.connect(
                f"{path.as_uri()}?mode=ro", uri=True
            )
            unpublished_connection.execute("PRAGMA query_only=ON")
            unpublished_connection.execute("BEGIN")
            owned.append(unpublished_connection)
        return graphiti_operational_partition_snapshot(
            connections[0],
            unpublished_connection,
            authority=connections[1],
            observed_at=observed_at,
            pre_frontier_hold_watermark=pre_frontier_hold_watermark,
        )
    finally:
        for connection in reversed(owned):
            connection.close()


def _campaign_reconciliation_evidence(
    connection: sqlite3.Connection,
    *,
    cohort_digest: str,
    generation_id: str,
    ingest_ids: tuple[str, ...],
) -> dict[str, object]:
    rows = connection.execute(
        "SELECT receipt_digest,projector_family_id,generation_id,"
        "authority_watermark,receipt_json FROM "
        "unpublished_graphiti_projection_reconciliations "
        "WHERE generation_id=?",
        (generation_id,),
    ).fetchall()
    if len(rows) != 1:
        raise GraphitiCampaignStop(
            "campaign projection reconciliation differs"
        )
    receipt_digest, family_id, retained_generation, watermark, receipt_json = rows[0]
    try:
        receipt, binding = graphiti_projection_reconciliation_from_json(
            str(receipt_json)
        )
    except (RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GraphitiCampaignStop(
            "campaign projection reconciliation is malformed"
        ) from exc
    if (
        binding
        != {
            "schema_version": (
                GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION
            ),
            "cohort_digest": cohort_digest,
            "ingest_ids": list(ingest_ids),
        }
        or receipt.receipt_digest != str(receipt_digest)
        or receipt.projector_family_id != str(family_id)
        or receipt.generation_id != str(retained_generation)
        or receipt.generation_id != generation_id
        or receipt.authority_watermark != int(watermark)
        or receipt.provider_model_calls != 0
    ):
        raise GraphitiCampaignStop(
            "campaign projection reconciliation integrity differs"
        )
    return {
        "receipt_digest": str(receipt_digest),
        "generation_id": generation_id,
        "authority_watermark": int(watermark),
        "effect_ids": list(receipt.expected_effect_ids),
        "provider_model_calls": 0,
    }


def _campaign_completion_evidence(
    connection: sqlite3.Connection,
    *,
    events: list[object],
    operational_before: Mapping[str, object],
    operational_after: Mapping[str, object],
    reconciliation_ids_before: frozenset[str],
    expected_generation_identity: tuple[str, str] | None,
    expected_ingest_ids: tuple[str, ...],
    proposal_count: int,
    elapsed_seconds: float,
    wall_time_cap: int,
    max_oldest_eligible_seconds: int,
) -> dict[str, object]:
    """Prove exact W0 completion from the shared operational partition."""

    expected: dict[tuple[int, str], Mapping[str, object]] = {}
    for raw_event in events:
        event = _mapping(raw_event, field="campaign event")
        identity = (
            _integer(
                event.get("ledger_seq"),
                field="campaign ledger sequence",
                positive=True,
            ),
            str(event.get("event_id") or ""),
        )
        if not identity[1] or identity in expected:
            raise GraphitiCampaignStop("campaign selected event identities differ")
        expected[identity] = event

    start_observation = _utc_instant(
        operational_before.get("observed_at"),
        field="campaign initial observation time",
    )
    end_observation = _utc_instant(
        operational_after.get("observed_at"),
        field="campaign completion observation time",
    )
    observation_seconds = (end_observation - start_observation).total_seconds()
    if not math.isfinite(observation_seconds) or observation_seconds <= 0:
        raise GraphitiCampaignStop("campaign observation window differs")

    placeholders = ",".join("?" for _ in expected)
    rows = connection.execute(
        "SELECT event_id,ledger_seq,state,attempt_count,terminal_at "
        "FROM unpublished_graphiti_revision_events "
        f"WHERE event_id IN ({placeholders}) ORDER BY ledger_seq",
        tuple(event_id for _ledger_seq, event_id in expected),
    ).fetchall()
    if (
        len(rows) != len(expected)
        or any(
            (int(ledger_seq), str(event_id)) not in expected
            or str(state) != "TERMINAL"
            or int(attempt_count) != 1
            or terminal_at is None
            for event_id, ledger_seq, state, attempt_count, terminal_at in rows
        )
    ):
        raise GraphitiCampaignStop("campaign selected-cohort watermark differs")
    if any(
        not start_observation
        < _utc_instant(terminal_at, field="campaign terminal time")
        <= end_observation
        for _event_id, _ledger_seq, _state, _attempt_count, terminal_at in rows
    ):
        raise GraphitiCampaignStop("campaign selected-cohort terminal time differs")

    def actionables(
        snapshot: Mapping[str, object], *, field: str, observed_at: datetime
    ) -> list[dict[str, object]]:
        raw_items = snapshot.get("actionable")
        if not isinstance(raw_items, list):
            raise GraphitiCampaignStop(f"{field} actionable partition differs")
        retained: list[dict[str, object]] = []
        identities: set[tuple[int, str]] = set()
        for raw_item in raw_items:
            item = dict(_mapping(raw_item, field=f"{field} actionable item"))
            identity = (
                _integer(
                    item.get("ledger_seq"),
                    field=f"{field} actionable ledger sequence",
                    positive=True,
                ),
                str(item.get("event_id") or ""),
            )
            if (
                not identity[1]
                or identity in identities
                or item.get("kind")
                not in {"FRESH_EVENT", "PROJECT_EVENT_GAP", "UNCLASSIFIED_GAP"}
            ):
                raise GraphitiCampaignStop(f"{field} actionable partition differs")
            if (
                _utc_instant(item.get("landed_at"), field=f"{field} landed time")
                > observed_at
            ):
                raise GraphitiCampaignStop(f"{field} landing time differs")
            identities.add(identity)
            retained.append(item)
        return retained

    before = actionables(
        operational_before,
        field="campaign initial",
        observed_at=start_observation,
    )
    after = actionables(
        operational_after,
        field="campaign completion",
        observed_at=end_observation,
    )
    before_by_identity = {
        (int(item["ledger_seq"]), str(item["event_id"])): item for item in before
    }
    after_by_identity = {
        (int(item["ledger_seq"]), str(item["event_id"])): item for item in after
    }
    if set(before_by_identity) != set(expected):
        raise GraphitiCampaignStop(
            "campaign initial selected-event partition differs"
        )
    for identity, event in expected.items():
        candidate = before_by_identity.get(identity)
        if (
            candidate is None
            or candidate.get("kind") != "FRESH_EVENT"
            or candidate.get("manifest_digest") != event.get("manifest_digest")
            or candidate.get("ingest_ids") != event.get("ingest_ids")
        ):
            raise GraphitiCampaignStop(
                "campaign initial selected-event partition differs"
            )

    watermark = max(ledger_seq for ledger_seq, _event_id in expected)
    backlog_at_watermark = [
        item for item in after if int(item["ledger_seq"]) <= watermark
    ]
    before_identities = set(before_by_identity)
    after_identities = set(after_by_identity)
    new_identities = after_identities - before_identities
    post_watermark = [
        item for item in after if int(item["ledger_seq"]) > watermark
    ]
    arrivals = [
        item
        for item in post_watermark
        if (int(item["ledger_seq"]), str(item["event_id"])) in new_identities
    ]
    oldest_post_watermark_lag = max(
        (
            max(
                math.ceil(
                    (
                        end_observation
                        - _utc_instant(
                            item["landed_at"],
                            field="campaign actionable landed time",
                        )
                    ).total_seconds()
                ),
                0,
            )
            for item in post_watermark
        ),
        default=0,
    )
    gaps = [item for item in after if item["kind"] != "FRESH_EVENT"]
    operational_stop_evidence = {
        "watermark": watermark,
        "partition_before_digest": operational_before.get("snapshot_digest"),
        "partition_after_digest": operational_after.get("snapshot_digest"),
        "arrival_count": len(arrivals),
        "arrivals": arrivals,
        "actionable_gaps": gaps,
        "actionable_backlog_at_watermark": backlog_at_watermark,
        "oldest_post_watermark_eligible_seconds": oldest_post_watermark_lag,
    }
    if gaps:
        raise GraphitiCampaignStop(
            "campaign has a projectable or unclassified Graphiti event gap",
            evidence=operational_stop_evidence,
        )
    if backlog_at_watermark:
        raise GraphitiCampaignStop(
            "campaign actionable backlog at watermark is non-zero",
            evidence=operational_stop_evidence,
        )
    if oldest_post_watermark_lag > max_oldest_eligible_seconds:
        raise GraphitiCampaignStop(
            "campaign oldest eligible lag objective differs",
            evidence=operational_stop_evidence,
        )

    campaign_service_ids = sorted(event_id for _ledger_seq, event_id in expected)
    if len(campaign_service_ids) < len(arrivals):
        raise GraphitiCampaignStop(
            "campaign service velocity is below arrival velocity",
            evidence=operational_stop_evidence,
        )

    reconciliation_ids_after = _campaign_reconciliation_ids(connection)
    new_reconciliations = reconciliation_ids_after - reconciliation_ids_before
    if proposal_count:
        if expected_generation_identity is None:
            raise GraphitiCampaignStop("campaign generation identity is missing")
        cohort_digest, generation_id = expected_generation_identity
        if new_reconciliations != {generation_id}:
            raise GraphitiCampaignStop("campaign projection reconciliation differs")
        reconciliation = _campaign_reconciliation_evidence(
            connection,
            cohort_digest=cohort_digest,
            generation_id=generation_id,
            ingest_ids=expected_ingest_ids,
        )
    else:
        if expected_generation_identity is not None or new_reconciliations:
            raise GraphitiCampaignStop("campaign projection reconciliation differs")
        cohort_digest = None
        generation_id = None
        reconciliation = None

    try:
        exact_admission = _exact_admission_reconciliation(connection)
    except (RuntimeError, TypeError, ValueError, KeyError, sqlite3.Error) as exc:
        raise GraphitiCampaignStop(
            "campaign terminal admission reconciliation differs"
        ) from exc
    if exact_admission.get("total") is not True or exact_admission.get(
        "disjoint"
    ) is not True:
        raise GraphitiCampaignStop(
            "campaign terminal admission reconciliation differs"
        )
    if proposal_count:
        exact_cohorts = exact_admission.get("cohorts")
        if not isinstance(exact_cohorts, list):
            raise GraphitiCampaignStop(
                "campaign terminal admission reconciliation differs"
            )
        matching_cohorts = [
            item
            for item in exact_cohorts
            if isinstance(item, Mapping)
            and item.get("ingest_ids") == list(expected_ingest_ids)
        ]
        if len(matching_cohorts) != 1 or (
            matching_cohorts[0].get("cohort_digest") != cohort_digest
            or matching_cohorts[0].get("generation_id") != generation_id
            or exact_admission.get("latest_generation_id") != generation_id
        ):
            raise GraphitiCampaignStop(
                "campaign terminal admission cohort differs"
            )

    admission = graphiti_admission_telemetry(
        connection,
        now=end_observation,
    ).canonical_value()
    if admission["admission_backlog"] != 0:
        raise GraphitiCampaignStop("campaign admission backlog is non-zero")
    if admission["dead_letter_count"] or admission["integrity_hold_receipt_count"]:
        raise GraphitiCampaignStop("campaign admission terminal evidence differs")
    if admission["projection_gap_count"] != 0 or (
        admission["proposal_denominator"]
        and admission["projection_reconciled"] is not True
    ):
        raise GraphitiCampaignStop("campaign projection reconciliation differs")
    if (
        not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
        or elapsed_seconds >= wall_time_cap
    ):
        raise GraphitiCampaignStop("campaign bounded lag objective differs")

    raw_holds = operational_after.get("holds")
    if not isinstance(raw_holds, list) or not all(
        isinstance(item, Mapping) for item in raw_holds
    ):
        raise GraphitiCampaignStop("campaign completion hold partition differs")
    holds = [dict(item) for item in raw_holds]
    observed_operational_seq = max(
        [
            watermark,
            *(int(item["ledger_seq"]) for item in after),
            *(int(item["ledger_seq"]) for item in holds),
        ]
    )
    return {
        "watermark": {
            "target": "selected cohort terminal",
            "terminal_ledger_seq": watermark,
            "observed_operational_ledger_seq": observed_operational_seq,
            "partition_before_digest": operational_before.get("snapshot_digest"),
            "partition_after_digest": operational_after.get("snapshot_digest"),
            "passed": True,
        },
        "backlog": {
            "target": "zero_at_selected_watermark",
            "remaining_actionable_at_watermark": [],
            "post_watermark_actionable": post_watermark,
            "holds": holds,
            "global_admission_backlog": admission["admission_backlog"],
            "passed": True,
        },
        "velocity": {
            "target": "service_at_least_arrival",
            "window_seconds": observation_seconds,
            "arrivals": arrivals,
            "campaign_service_event_ids": campaign_service_ids,
            "arrival_count": len(arrivals),
            "service_count": len(campaign_service_ids),
            "passed": True,
        },
        "lag": {
            "target": {
                "max_oldest_eligible_seconds": max_oldest_eligible_seconds,
            },
            "oldest_post_watermark_eligible_seconds": (
                oldest_post_watermark_lag
            ),
            "elapsed_seconds": elapsed_seconds,
            "wall_time_cap_seconds": wall_time_cap,
            "passed": True,
        },
        "reconciliation": {
            "target": "exact",
            "new_generation_ids": sorted(new_reconciliations),
            "cohort_digest": cohort_digest,
            "generation_id": generation_id,
            "receipt": reconciliation,
            "exact_admission": exact_admission,
            "global_admission": admission,
            "passed": True,
        },
    }


def _campaign_stop_report(
    *,
    packet: Mapping[str, object] | None,
    unpublished_store: str,
    failure: BaseException,
) -> dict[str, object]:
    """Read back durable partial effects without inventing campaign progress."""

    packet_digest = packet.get("packet_digest") if packet is not None else None
    campaign = packet.get("bounded_campaign") if packet is not None else None
    cohort = campaign.get("cohort") if isinstance(campaign, Mapping) else None
    raw_events = cohort.get("events") if isinstance(cohort, Mapping) else None
    selected: list[dict[str, object]] = []
    if isinstance(raw_events, list):
        for raw_event in raw_events:
            if not isinstance(raw_event, Mapping):
                continue
            event_id = raw_event.get("event_id")
            ledger_seq = raw_event.get("ledger_seq")
            ingest_ids = raw_event.get("ingest_ids")
            if (
                isinstance(event_id, str)
                and event_id
                and isinstance(ledger_seq, int)
                and not isinstance(ledger_seq, bool)
                and isinstance(ingest_ids, list)
                and all(isinstance(item, str) and item for item in ingest_ids)
            ):
                selected.append(
                    {
                        "event_id": event_id,
                        "ledger_seq": ledger_seq,
                        "ingest_ids": list(ingest_ids),
                    }
                )

    events: list[dict[str, object]] = []
    spend_rows: list[sqlite3.Row] = []
    admission_queue_count: int | None = None
    admission_decision_count: int | None = None
    projection_receipt_count: int | None = None
    generation_evidence: dict[str, object] | None = None
    generation_attribution_failure: str | None = None
    observation_failure: str | None = None
    try:
        with read_only_snapshot(unpublished_store) as snapshot:
            connection = snapshot.connection
            connection.row_factory = sqlite3.Row
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            required_tables = {
                "unpublished_graphiti_revision_events",
                "unpublished_graphiti_spend",
            }
            if not required_tables.issubset(tables):
                raise GraphitiCampaignStop("campaign evidence tables are absent")
            for item in selected:
                row = connection.execute(
                    "SELECT event_id,ledger_seq,state,attempt_count,"
                    "provider_dispatched,terminal_at,proposal_count,"
                    "last_failure_code "
                    "FROM unpublished_graphiti_revision_events "
                    "WHERE event_id=? AND ledger_seq=?",
                    (item["event_id"], item["ledger_seq"]),
                ).fetchone()
                if row is None:
                    events.append(
                        {
                            "event_id": item["event_id"],
                            "ledger_seq": item["ledger_seq"],
                            "observed": False,
                            "state": None,
                            "attempt_count": None,
                            "provider_dispatched": None,
                            "terminal_at": None,
                            "proposal_count": None,
                            "last_failure_code": None,
                        }
                    )
                    continue
                events.append(
                    {
                        "event_id": str(row["event_id"]),
                        "ledger_seq": int(row["ledger_seq"]),
                        "observed": True,
                        "state": str(row["state"]),
                        "attempt_count": int(row["attempt_count"]),
                        "provider_dispatched": bool(row["provider_dispatched"]),
                        "terminal_at": row["terminal_at"],
                        "proposal_count": row["proposal_count"],
                        "last_failure_code": row["last_failure_code"],
                    }
                )

            ingest_ids = tuple(
                dict.fromkeys(
                    str(ingest_id)
                    for item in selected
                    for ingest_id in item["ingest_ids"]
                )
            )
            exact_ingest_ids = tuple(sorted(set(ingest_ids)))
            for ingest_id in ingest_ids:
                spend_rows.extend(
                    connection.execute(
                        "SELECT ingest_id,status,actual_gbp_microunits "
                        "FROM unpublished_graphiti_spend WHERE ingest_id=?",
                        (ingest_id,),
                    ).fetchall()
                )
            if "unpublished_graphiti_admission_queue" in tables:
                admission_queue_count = sum(
                    int(
                        connection.execute(
                            "SELECT COUNT(*) "
                            "FROM unpublished_graphiti_admission_queue "
                            "WHERE ingest_id=?",
                            (ingest_id,),
                        ).fetchone()[0]
                    )
                    for ingest_id in ingest_ids
                )
                if "unpublished_graphiti_admission_decisions" in tables:
                    admission_decision_count = sum(
                        int(
                            connection.execute(
                                "SELECT COUNT(*) "
                                "FROM unpublished_graphiti_admission_decisions d "
                                "JOIN unpublished_graphiti_admission_queue q "
                                "ON q.proposal_key=d.proposal_key "
                                "WHERE q.ingest_id=?",
                                (ingest_id,),
                            ).fetchone()[0]
                        )
                        for ingest_id in ingest_ids
                    )
                if "unpublished_graphiti_projection_receipts" in tables:
                    projection_receipt_count = sum(
                        int(
                            connection.execute(
                                "SELECT COUNT(*) "
                                "FROM unpublished_graphiti_projection_receipts p "
                                "JOIN unpublished_graphiti_admission_queue q "
                                "ON q.proposal_key=p.proposal_key "
                                "WHERE q.ingest_id=?",
                                (ingest_id,),
                            ).fetchone()[0]
                        )
                        for ingest_id in ingest_ids
                    )
                if (
                    admission_queue_count
                    and admission_decision_count == admission_queue_count
                    and {
                        "unpublished_graphiti_ingest",
                        "unpublished_graphiti_receipts",
                        "unpublished_graphiti_admission_decisions",
                        "unpublished_graphiti_admission_receipt_failures",
                        "unpublished_graphiti_projection_reconciliations",
                    }.issubset(tables)
                ):
                    try:
                        cohort_digest, generation_id = (
                            graphiti_decided_cohort_generation_identity(
                                connection,
                                ingest_ids=exact_ingest_ids,
                            )
                        )
                        reconciliation_rows = connection.execute(
                            "SELECT receipt_digest,projector_family_id,"
                            "generation_id,authority_watermark,receipt_json,"
                            "reconciled_at FROM "
                            "unpublished_graphiti_projection_reconciliations "
                            "WHERE generation_id=? "
                            "ORDER BY reconciled_at,receipt_digest",
                            (generation_id,),
                        ).fetchall()
                        reconciliations: list[dict[str, object]] = []
                        for row in reconciliation_rows:
                            retained, binding = (
                                graphiti_projection_reconciliation_from_json(
                                    str(row["receipt_json"])
                                )
                            )
                            if (
                                binding
                                != {
                                    "schema_version": (
                                        GRAPHITI_ADMISSION_RECONCILIATION_SCHEMA_VERSION
                                    ),
                                    "cohort_digest": cohort_digest,
                                    "ingest_ids": list(exact_ingest_ids),
                                }
                                or retained.receipt_digest
                                != str(row["receipt_digest"])
                                or retained.projector_family_id
                                != str(row["projector_family_id"])
                                or retained.generation_id
                                != str(row["generation_id"])
                                or retained.authority_watermark
                                != int(row["authority_watermark"])
                            ):
                                raise GraphitiCampaignStop(
                                    "campaign reconciliation integrity differs"
                                )
                            reconciliations.append(
                                {
                                    "receipt_digest": str(row["receipt_digest"]),
                                    "projector_family_id": str(
                                        row["projector_family_id"]
                                    ),
                                    "generation_id": str(row["generation_id"]),
                                    "authority_watermark": int(
                                        row["authority_watermark"]
                                    ),
                                    "expected_effect_ids": list(
                                        retained.expected_effect_ids
                                    ),
                                    "actual_effect_ids": list(
                                        retained.actual_effect_ids
                                    ),
                                    "reconciled_at": str(row["reconciled_at"]),
                                }
                            )
                        generation_evidence = {
                            "cohort_digest": cohort_digest,
                            "generation_id": generation_id,
                            "reconciliation_count": len(reconciliations),
                            "reconciliations": reconciliations,
                            "reconciliation_attribution_complete": (
                                len(reconciliations) == 1
                            ),
                        }
                    except (
                        KeyError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        sqlite3.Error,
                    ) as exc:
                        generation_attribution_failure = (
                            f"{type(exc).__name__}: {exc}"
                        )
    except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
        observation_failure = f"{type(exc).__name__}: {exc}"

    observed = [item for item in events if item["observed"] is True]
    terminal_count = sum(item["state"] == "TERMINAL" for item in observed)
    attempted_count = sum(
        isinstance(item["attempt_count"], int) and item["attempt_count"] > 0
        for item in observed
    )
    provider_dispatched_count = sum(
        item["provider_dispatched"] is True for item in observed
    )
    reconciliation_count = (
        int(generation_evidence["reconciliation_count"])
        if generation_evidence is not None
        else 0
    )
    if projection_receipt_count or reconciliation_count:
        last_durable_stage = "PROJECTION_RECORDED"
    elif admission_decision_count:
        last_durable_stage = "ADMISSION_RECORDED"
    elif terminal_count:
        last_durable_stage = "EXTRACTION_RECORDED"
    elif provider_dispatched_count:
        last_durable_stage = "PROVIDER_DISPATCH_RECORDED"
    elif attempted_count:
        last_durable_stage = "ATTEMPT_RECORDED"
    elif selected and len(observed) == len(selected):
        last_durable_stage = "PRE_EFFECT"
    else:
        last_durable_stage = "UNKNOWN"

    status_counts: dict[str, int] = {}
    actual_values: list[int] = []
    spend_complete = observation_failure is None
    for row in spend_rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        actual = row["actual_gbp_microunits"]
        if status != "RECONCILED" or actual is None:
            spend_complete = False
        if actual is not None:
            actual_values.append(int(actual))
    event_observation_complete = (
        bool(selected)
        and len(observed) == len(selected)
        and observation_failure is None
    )
    attempted_event_ids = {
        str(item["event_id"])
        for item in observed
        if isinstance(item["attempt_count"], int) and item["attempt_count"] > 0
    }
    expected_spend_ingest_ids = {
        str(ingest_id)
        for item in selected
        if item["event_id"] in attempted_event_ids
        for ingest_id in item["ingest_ids"]
    }
    observed_spend_ingest_ids = {str(row["ingest_id"]) for row in spend_rows}
    if not expected_spend_ingest_ids.issubset(observed_spend_ingest_ids):
        spend_complete = False
    spend_complete = spend_complete and event_observation_complete
    if not spend_rows and attempted_count:
        spend_complete = False
    actual_spend = (
        sum(actual_values)
        if spend_complete and (actual_values or not spend_rows)
        else None
    )

    return {
        "schema_version": "newsroom.graphiti-campaign-stop-report.v1",
        "state": "CAMPAIGN_STOPPED",
        "packet_digest": packet_digest,
        "failure_type": type(failure).__name__,
        "failure": str(failure),
        "failure_evidence": getattr(failure, "evidence", None),
        "stage": last_durable_stage,
        "observation_complete": event_observation_complete,
        "observation_failure": observation_failure,
        "selected_event_count": len(selected) if selected else None,
        "observed_event_count": (
            len(observed) if observation_failure is None else None
        ),
        "completed_event_count": (
            terminal_count if event_observation_complete else None
        ),
        "terminal_event_count": (
            terminal_count if event_observation_complete else None
        ),
        "attempted_event_count": (
            attempted_count if event_observation_complete else None
        ),
        "provider_dispatched_event_count": (
            provider_dispatched_count if event_observation_complete else None
        ),
        "events": events,
        "spend": {
            "row_count": (
                len(spend_rows) if observation_failure is None else None
            ),
            "status_counts": status_counts if observation_failure is None else None,
            "reconciled_actual_gbp_microunits": (
                sum(actual_values) if actual_values else None
            ),
            "actual_gbp_complete": spend_complete,
            "actual_gbp_microunits": actual_spend,
        },
        "admission": {
            "queue_count": admission_queue_count,
            "decision_count": admission_decision_count,
            "projection_receipt_count": projection_receipt_count,
        },
        "generation": generation_evidence,
        "generation_attribution_failure": generation_attribution_failure,
        "public_dispatch": False,
        "auto_publish": False,
    }


def run_bounded_campaign(
    *,
    packet: Mapping[str, object],
    proving_store: str,
    unpublished_store: str,
    runtime: GovernedGraphitiWorkerRuntime,
    head_sha: str,
    tree_sha: str,
    owner_f4_fence: Callable[[Mapping[str, object]], None],
    clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Execute one sealed exact cohort; no queue, scheduler or second authority."""

    campaign = validate_graphiti_campaign_packet(packet)
    if packet.get("code_identity") != {"head_sha": head_sha, "tree_sha": tree_sha}:
        raise GraphitiCampaignStop("campaign code identity drifted")

    store_snapshots = _mapping(
        packet.get("store_snapshots"), field="campaign store snapshots"
    )
    authority_snapshot = _mapping(
        store_snapshots.get("authority"), field="campaign authority snapshot"
    )
    authority_path = str(authority_snapshot.get("source_path") or "")
    authority_digest = str(authority_snapshot.get("descriptor_digest") or "")
    if (
        runtime.authority_store_source_path != authority_path
        or runtime.authority_store_descriptor_digest != authority_digest
    ):
        raise GraphitiCampaignStop("campaign runtime authority binding drifted")
    graph = _mapping(campaign.get("graph"), field="campaign graph")
    if graph.get("destination_id") != runtime.graph_destination_id:
        raise GraphitiCampaignStop("campaign runtime graph destination drifted")
    expected_snapshot_digests = _mapping(
        campaign.get("source_snapshot_digests"),
        field="campaign source snapshot digests",
    )
    if not authority_path or graphiti_store_snapshot_digests(
        proving_store=proving_store,
        unpublished_store=unpublished_store,
        authority_store=authority_path,
    ) != dict(expected_snapshot_digests):
        raise GraphitiCampaignStop("campaign source snapshot drifted")

    cohort = _mapping(campaign.get("cohort"), field="campaign cohort")
    events = cohort.get("events")
    if not isinstance(events, list) or not events:
        raise GraphitiCampaignStop("campaign cohort events are missing")
    objectives = _mapping(
        campaign.get("success_objectives"),
        field="campaign success objectives",
    )
    lag_objective = _mapping(
        objectives.get("lag"),
        field="campaign lag objective",
    )
    if (
        set(objectives)
        != {"watermark", "backlog", "velocity", "lag", "reconciliation"}
        or objectives.get("watermark") != "selected cohort terminal"
        or objectives.get("backlog") != 0
        or isinstance(objectives.get("backlog"), bool)
        or objectives.get("velocity") != "service_at_least_arrival"
        or objectives.get("reconciliation") != "exact"
        or set(lag_objective) != {"max_oldest_eligible_seconds"}
    ):
        raise GraphitiCampaignStop("campaign success objectives differ")
    max_oldest_eligible_seconds = _integer(
        lag_objective.get("max_oldest_eligible_seconds"),
        field="maximum oldest eligible lag",
        positive=True,
    )
    caps = _mapping(campaign.get("caps"), field="campaign caps")
    total_caps = _mapping(caps.get("total"), field="total campaign caps")
    event_cap = _integer(
        total_caps.get("events"), field="total event cap", positive=True
    )
    if len(events) > event_cap:
        raise GraphitiCampaignStop("total event cap reached")
    provider = _mapping(campaign.get("provider"), field="campaign provider")
    ramp = _mapping(campaign.get("ramp"), field="campaign ramp")
    raw_phases = ramp.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise GraphitiCampaignStop("campaign ramp phases are missing")
    phase_limits: dict[int, dict[str, object]] = {}
    phase_starts: dict[int, dict[str, object]] = {}
    prior_limit = 0
    for raw_phase in raw_phases:
        phase = _mapping(raw_phase, field="campaign ramp phase")
        limit = _integer(
            phase.get("event_limit"), field="ramp event limit", positive=True
        )
        phase_id = str(phase.get("phase_id") or "")
        entry = phase.get("entry_conditions")
        advance = phase.get("advance_conditions")
        if (
            not phase_id
            or limit in phase_limits
            or limit <= prior_limit
            or not isinstance(entry, list)
            or not {
                "EXACT_SNAPSHOT_AND_IDENTITY_RECONFIRMED",
                "OWNER_F4_GO_RETAINED",
            }.issubset(entry)
            or not isinstance(advance, list)
            or not {
                "ALL_EXACT_RECEIPTS_RECONCILED",
                "CAPS_AND_ACCOUNTING_RECONCILED",
                "NO_STOP_CONDITION_OBSERVED",
            }.issubset(advance)
        ):
            raise GraphitiCampaignStop("campaign ramp identity is invalid")
        phase_value = {
            "phase_id": phase_id,
            "event_limit": limit,
            "entry_conditions": list(entry),
            "advance_conditions": list(advance),
        }
        phase_starts[prior_limit] = phase_value
        phase_limits[limit] = phase_value
        prior_limit = limit
    if max(phase_limits) != len(events):
        raise GraphitiCampaignStop("campaign ramp does not end at exact cohort")

    qualify_campaign_adapter_runtime()

    preflights: list[dict[str, object]] = []
    prepared_unit_groups: list[tuple[CorpusIngestUnit, ...]] = []
    all_ingest_ids: list[str] = []
    reserved_total = 0
    preflight_at = clock().astimezone(UTC)
    resolved_corpus = load_graphiti_units(
        proving_store=proving_store,
        evaluated_at=preflight_at,
    )
    units_by_ingest_id = {unit.ingest_id: unit for unit in resolved_corpus}
    if len(units_by_ingest_id) != len(resolved_corpus):
        raise GraphitiCampaignStop("campaign resolved duplicate ingest identities")
    for item in events:
        event = _mapping(item, field="campaign event")
        event_id = str(event.get("event_id") or "")
        ledger_seq = _integer(
            event.get("ledger_seq"), field="campaign ledger sequence", positive=True
        )
        preflight = qualify_fresh_graphiti_event(
            proving_store=proving_store,
            unpublished_store=unpublished_store,
            event_id=event_id,
            ledger_seq=ledger_seq,
            clock=lambda: preflight_at,
            resolved_corpus_units=resolved_corpus,
        )
        resolved = preflight.get("resolved_units")
        if not isinstance(resolved, list) or not resolved:
            raise GraphitiCampaignStop("campaign event resolved no units")
        resolved_ingest_ids = tuple(
            str(value.get("ingest_id") or "")
            for value in resolved
            if isinstance(value, Mapping)
        )
        ingest_ids = tuple(sorted(resolved_ingest_ids))
        if (
            not ingest_ids
            or len(resolved_ingest_ids) != len(resolved)
            or len(set(resolved_ingest_ids)) != len(resolved_ingest_ids)
            or any(not value for value in ingest_ids)
            or list(ingest_ids) != sorted(event.get("ingest_ids") or [])
            or preflight.get("event_manifest_digest") != event.get("manifest_digest")
        ):
            raise GraphitiCampaignStop("campaign exact event preflight drifted")
        if set(all_ingest_ids).intersection(ingest_ids):
            raise GraphitiCampaignStop("campaign ingest identities overlap")
        all_ingest_ids.extend(ingest_ids)
        reserved_total += len(resolved) * GRAPHITI_UNIT_RESERVATION_GBP_MICROUNITS
        preflights.append(preflight)
        try:
            prepared_unit_groups.append(
                tuple(
                    units_by_ingest_id[ingest_id]
                    for ingest_id in resolved_ingest_ids
                )
            )
        except KeyError as exc:
            raise GraphitiCampaignStop(
                "campaign resolved unit identity drifted"
            ) from exc
    exact_ingest_ids = tuple(sorted(all_ingest_ids))
    _assert_fresh_campaign_ingests(
        unpublished_store,
        ingest_ids=exact_ingest_ids,
    )
    spend_cap = _integer(
        total_caps.get("spend_gbp_microunits"), field="total spend cap"
    )
    if reserved_total > spend_cap:
        raise GraphitiCampaignStop("conservative spend cap reached")

    rate = _mapping(caps.get("rate"), field="campaign rate cap")
    events_per_minute = _integer(
        rate.get("events_per_minute"), field="event rate cap", positive=True
    )
    minimum_interval = 60.0 / events_per_minute
    wall_cap = _integer(
        total_caps.get("wall_time_seconds"), field="wall time cap", positive=True
    )
    per_event_caps = _mapping(caps.get("per_event"), field="per-event caps")
    max_proposals = _integer(
        per_event_caps.get("proposals"), field="per-event proposal cap"
    )
    started = monotonic()
    last_dispatch: float | None = None
    completed: list[dict[str, object]] = []
    phase_checkpoints: list[dict[str, object]] = []
    proposal_total = 0
    actual_spend_total = 0
    fallback_total = 0
    retry_total = 0
    operational_before: dict[str, object] | None = None
    for event, preflight, prepared_units in zip(
        events, preflights, prepared_unit_groups, strict=True
    ):
        completed_before = len(completed)
        phase = phase_starts.get(completed_before)
        if phase is not None:
            owner_f4_fence(packet)
            runtime.graph_state_fence(campaign)
        if operational_before is None:
            operational_before = _campaign_operational_partition_snapshot(
                proving_store=proving_store,
                unpublished_store=unpublished_store,
                authority_store=authority_path,
                observed_at=clock(),
            )
            actionables = operational_before.get("actionable")
            if not isinstance(actionables, list) or not all(
                isinstance(item, Mapping) for item in actionables
            ):
                raise GraphitiCampaignStop(
                    "campaign initial actionable partition differs"
                )
            gaps = [
                dict(item)
                for item in actionables
                if item.get("kind") != "FRESH_EVENT"
            ]
            fresh_actionables = [
                item for item in actionables if item.get("kind") == "FRESH_EVENT"
            ]
            packet_candidates: list[dict[str, object]] = []
            operational_candidates: list[dict[str, object]] = []
            for source, target, field in (
                (events, packet_candidates, "campaign event"),
                (
                    fresh_actionables,
                    operational_candidates,
                    "operational candidate",
                ),
            ):
                for raw_candidate in source:
                    candidate = _mapping(raw_candidate, field=field)
                    ingest_ids = candidate.get("ingest_ids")
                    value = {
                        "ledger_seq": _integer(
                            candidate.get("ledger_seq"),
                            field=f"{field} ledger sequence",
                            positive=True,
                        ),
                        "event_id": str(candidate.get("event_id") or ""),
                        "manifest_digest": str(
                            candidate.get("manifest_digest") or ""
                        ),
                        "ingest_ids": ingest_ids,
                    }
                    if (
                        not value["event_id"]
                        or not value["manifest_digest"]
                        or not isinstance(ingest_ids, list)
                        or not all(
                            isinstance(ingest_id, str) and ingest_id
                            for ingest_id in ingest_ids
                        )
                    ):
                        raise GraphitiCampaignStop(
                            "campaign initial candidate identity differs"
                        )
                    target.append(value)
            packet_candidates.sort(
                key=lambda item: (int(item["ledger_seq"]), str(item["event_id"]))
            )
            operational_candidates.sort(
                key=lambda item: (int(item["ledger_seq"]), str(item["event_id"]))
            )
            if gaps or operational_candidates != packet_candidates:
                raise GraphitiCampaignStop(
                    "campaign snapshot or candidate identity drifted",
                    evidence={
                        "stage": "PRE_DISPATCH_OPERATIONAL_PARTITION",
                        "partition_digest": operational_before.get(
                            "snapshot_digest"
                        ),
                        "packet_candidates": packet_candidates,
                        "operational_candidates": operational_candidates,
                        "actionable_gaps": gaps,
                    },
                )
        now = monotonic()
        if last_dispatch is not None and now - last_dispatch < minimum_interval:
            delay = minimum_interval - (now - last_dispatch)
            if now - started + delay >= wall_cap:
                raise GraphitiCampaignStop("wall time cap reached before rate delay")
            sleep(delay)
        event_map = _mapping(event, field="campaign event")
        last_dispatch = monotonic()
        elapsed = last_dispatch - started
        if elapsed >= wall_cap:
            raise GraphitiCampaignStop("wall time cap reached")
        remaining = wall_cap - elapsed
        result = consume_next_graphiti_event(
            proving_store=proving_store,
            unpublished_store=unpublished_store,
            graphiti=runtime.graphiti,
            owner_id=f"hermes-graphiti-campaign:{uuid.uuid4()}",
            clock=clock,
            event_id=str(event_map["event_id"]),
            require_fresh=True,
            recover_model_usage=False,
            max_dispatch_seconds=remaining,
            prepared_event_preflight=preflight,
            prepared_event_units=prepared_units,
            max_reserved_gbp_microunits=(
                len(preflight["resolved_units"])
                * GRAPHITI_UNIT_RESERVATION_GBP_MICROUNITS
            ),
            graphiti_admission_factory=runtime.admission_factory,
            unit_authority_resolver=runtime.bind_unit_authority,
            max_graphiti_admissions=max(1, max_proposals),
            require_graphiti_admission=True,
            defer_graphiti_admission=True,
        )
        completed_at = monotonic()
        if completed_at - started >= wall_cap:
            raise GraphitiCampaignStop("wall time cap reached after provider dispatch")
        if (
            result is None
            or result.event_id != event_map["event_id"]
            or result.ledger_seq != event_map["ledger_seq"]
        ):
            raise GraphitiCampaignStop("campaign extraction event was non-terminal")
        if result.state == "CONFIGURATION_HELD":
            raise GraphitiCampaignStop(
                "campaign extraction event was a pre-dispatch configuration refusal"
            )
        if result.state != "TERMINAL":
            raise GraphitiCampaignStop("campaign extraction event was non-terminal")
        if result.attempt_count != 1:
            raise GraphitiCampaignStop("campaign event retry count drifted")
        ingest_ids = tuple(
            sorted(
                str(value["ingest_id"])
                for value in preflight["resolved_units"]
            )
        )
        receipt = _campaign_receipt_evidence(
            unpublished_store,
            ingest_ids=ingest_ids,
            provider=provider,
        )
        if receipt["proposal_count"] > max_proposals:
            raise GraphitiCampaignStop("per-event proposal cap reached")
        for name, receipt_field in (
            ("fallbacks", "fallback_count"),
            ("retries", "retry_count"),
        ):
            observed = int(receipt[receipt_field])
            if observed > _integer(
                per_event_caps.get(name), field=f"per-event {name} cap"
            ):
                raise GraphitiCampaignStop(f"per-event {name} cap reached")
        proposal_total += int(receipt["proposal_count"])
        if proposal_total > _integer(
            total_caps.get("proposals"), field="total proposal cap"
        ):
            raise GraphitiCampaignStop("total proposal cap reached")
        actual_spend_total += int(receipt["actual_gbp_microunits"])
        fallback_total += int(receipt["fallback_count"])
        retry_total += int(receipt["retry_count"])
        if actual_spend_total > spend_cap:
            raise GraphitiCampaignStop("actual campaign spend cap reached")
        if fallback_total > _integer(
            total_caps.get("fallbacks"), field="total fallback cap"
        ):
            raise GraphitiCampaignStop("total fallback cap reached")
        if retry_total > _integer(
            total_caps.get("retries"), field="total retry cap"
        ):
            raise GraphitiCampaignStop("total retry cap reached")
        completed.append(
            {
                "event_id": result.event_id,
                "ledger_seq": result.ledger_seq,
                "state": "EXTRACTION_TERMINAL_CAMPAIGN_PENDING",
                **receipt,
            }
        )
        completed_count = len(completed)
        if completed_count in phase_limits:
            phase_value = phase_limits[completed_count]
            checkpoint = {
                "phase_id": phase_value["phase_id"],
                "event_count": completed_count,
                "all_exact_receipts_reconciled": (
                    len(completed) == completed_count
                    and all(
                        item["state"] == "EXTRACTION_TERMINAL_CAMPAIGN_PENDING"
                        and item["fallback_count"] == 0
                        and item["retry_count"] == 0
                        for item in completed
                    )
                ),
                "caps_and_accounting_reconciled": (
                    proposal_total
                    <= _integer(
                        total_caps.get("proposals"),
                        field="total proposal cap",
                    )
                    and actual_spend_total <= spend_cap
                    and fallback_total
                    <= _integer(
                        total_caps.get("fallbacks"),
                        field="total fallback cap",
                    )
                    and retry_total
                    <= _integer(
                        total_caps.get("retries"), field="total retry cap"
                    )
                ),
                "no_stop_condition_observed": True,
            }
            if not all(
                checkpoint[key] is True
                for key in (
                    "all_exact_receipts_reconciled",
                    "caps_and_accounting_reconciled",
                    "no_stop_condition_observed",
                )
            ):
                raise GraphitiCampaignStop("campaign ramp advance gate failed")
            phase_checkpoints.append(checkpoint)

    ingest_groups = tuple(
        tuple(
            sorted(
                str(value["ingest_id"])
                for value in preflight["resolved_units"]
            )
        )
        for preflight in preflights
    )
    owner_f4_fence(packet)
    runtime.graph_state_fence(campaign)
    connection = connect(unpublished_store)
    try:
        if monotonic() - started >= wall_cap:
            raise GraphitiCampaignStop("wall time cap reached before admission")
        reconciliation_ids_before = _campaign_reconciliation_ids(connection)
        admission = runtime.admission_factory(connection)
        admission.enqueue_complete_receipts(ingest_ids=exact_ingest_ids)
        rows = _campaign_admission_rows(
            connection,
            ingest_ids=exact_ingest_ids,
        )
        if len(rows) != proposal_total:
            raise GraphitiCampaignStop(
                "campaign proposal denominator differs from admission queue"
            )
        _assert_admission_caps(
            rows,
            ingest_groups=ingest_groups,
            caps=caps,
            decisions_required=False,
        )
        decision_report = admission.drain(
            worker_id=f"hermes-graphiti-campaign:{uuid.uuid4()}",
            limit=max(1, proposal_total),
            ingest_ids=exact_ingest_ids,
            stop_on_failure=True,
        )
        if decision_report.failed or decision_report.dead_lettered:
            raise GraphitiCampaignStop("campaign admission stopped on first failure")
        decided_rows = _campaign_admission_rows(
            connection,
            ingest_ids=exact_ingest_ids,
        )
        actual = _assert_admission_caps(
            decided_rows,
            ingest_groups=ingest_groups,
            caps=caps,
            decisions_required=True,
        )
        expected_generation_identity = (
            graphiti_decided_cohort_generation_identity(
                connection,
                ingest_ids=exact_ingest_ids,
            )
            if proposal_total
            else None
        )
        if monotonic() - started >= wall_cap:
            raise GraphitiCampaignStop("wall time cap reached before projection")
        generation_report = admission.finalise_decided_cohort(
            ingest_ids=exact_ingest_ids
        )
        if generation_report.failed or generation_report.dead_lettered:
            raise GraphitiCampaignStop("campaign generation was incomplete")
        if generation_report.projected != actual["effects"]:
            raise GraphitiCampaignStop("campaign projection denominator differs")
        if connection.in_transaction:
            raise GraphitiCampaignStop(
                "campaign completion began with an open store transaction"
            )
        connection.execute("BEGIN IMMEDIATE")
        completed_elapsed = monotonic() - started
        if completed_elapsed >= wall_cap:
            raise GraphitiCampaignStop("wall time cap reached after projection")
        if operational_before is None:
            raise GraphitiCampaignStop("campaign initial partition is missing")
        completed_at = clock()
        # Retain the selected policy through completion, not a broader legacy
        # backlog that reappears merely because the frontier is now terminal.
        hold_policy = {}
        if any(
            hold.get("reason") == PRE_FRONTIER_BACKLOG_HOLD_REASON
            for hold in operational_before.get("holds", ())
        ):
            hold_policy["pre_frontier_hold_watermark"] = max(
                int(event["ledger_seq"]) for event in events
            )
        operational_after = _campaign_operational_partition_snapshot(
            proving_store=proving_store,
            unpublished_store=unpublished_store,
            authority_store=authority_path,
            observed_at=completed_at,
            unpublished_connection=connection,
            **hold_policy,
        )
        objective_evidence = _campaign_completion_evidence(
            connection,
            events=events,
            operational_before=operational_before,
            operational_after=operational_after,
            reconciliation_ids_before=reconciliation_ids_before,
            expected_generation_identity=expected_generation_identity,
            expected_ingest_ids=exact_ingest_ids,
            proposal_count=proposal_total,
            elapsed_seconds=completed_elapsed,
            wall_time_cap=wall_cap,
            max_oldest_eligible_seconds=max_oldest_eligible_seconds,
        )
        connection.commit()
    finally:
        connection.close()
    return {
        "state": "CAMPAIGN_COMPLETE",
        "packet_digest": packet["packet_digest"],
        "event_count": len(completed),
        "events": completed,
        "ramp_checkpoints": phase_checkpoints,
        "proposal_count": proposal_total,
        "actual_gbp_microunits": actual_spend_total,
        "fallback_count": fallback_total,
        "retry_count": retry_total,
        **actual,
        "generation_promoted": proposal_total > 0,
        "success_objectives": objective_evidence,
        "public_dispatch": False,
        "auto_publish": False,
    }


def main(
    argv: list[str] | None = None,
    *,
    runtime: GovernedGraphitiWorkerRuntime | None = None,
    owner_f4_fence: Callable[[Mapping[str, object]], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Unpublished EVALUATION Graphiti event worker."
    )
    parser.add_argument("--proving", default=str(CANONICAL_PROVING_STORE))
    parser.add_argument("--unpublished", default=str(CANONICAL_UNPUBLISHED_STORE))
    parser.add_argument(
        "--once",
        action="store_true",
        help="retained compatibility flag; exact-event mode is always one event",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=180.0,
        help="hard upper bound for the selected event's provider dispatch",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--event-id",
        help="claim one exact fresh event rather than the generic queue",
    )
    mode.add_argument(
        "--campaign-packet",
        type=Path,
        help="execute one sealed READY campaign through the injected F4 fence",
    )
    parser.add_argument(
        "--ledger-seq",
        type=int,
        help="ledger sequence bound to --event-id provider-free preflight",
    )
    parser.add_argument(
        "--max-reserved-gbp-microunits",
        type=int,
        help="finite conservative embedding reservation cap for the exact event",
    )
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.max_runtime_seconds)
        or args.max_runtime_seconds <= 0
    ):
        parser.error("--max-runtime-seconds must be finite and positive")
    if args.campaign_packet is not None:
        if args.ledger_seq is not None or args.max_reserved_gbp_microunits is not None:
            parser.error("exact-event bounds cannot accompany --campaign-packet")
        if runtime is None or owner_f4_fence is None:
            _emit_stop("CAMPAIGN_AUTHORITY_UNCONFIGURED", completed=0)
            return 2
        packet: Mapping[str, object] | None = None
        try:
            loaded_packet = json.loads(
                args.campaign_packet.read_text(encoding="utf-8")
            )
            if not isinstance(loaded_packet, Mapping):
                raise GraphitiCampaignStop("campaign packet is not a JSON object")
            packet = loaded_packet
            head_sha, tree_sha = _current_git_identity()
            report = run_bounded_campaign(
                packet=packet,
                proving_store=args.proving,
                unpublished_store=args.unpublished,
                runtime=runtime,
                head_sha=head_sha,
                tree_sha=tree_sha,
                owner_f4_fence=owner_f4_fence,
            )
        except (
            GraphitiCampaignStop,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            sqlite3.Error,
            subprocess.SubprocessError,
        ) as exc:
            print(
                json.dumps(
                    {
                        "event": "GRAPHITI_CAMPAIGN_STOPPED",
                        "result": _campaign_stop_report(
                            packet=packet,
                            unpublished_store=args.unpublished,
                            failure=exc,
                        ),
                        "public_dispatch": False,
                        "auto_publish": False,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 2
        print(
            json.dumps(
                {
                    "event": "GRAPHITI_CAMPAIGN_RESULT",
                    "result": report,
                    "public_dispatch": False,
                    "auto_publish": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    if args.ledger_seq is None or args.ledger_seq <= 0:
        parser.error("--ledger-seq must be positive")
    if (
        args.max_reserved_gbp_microunits is None
        or args.max_reserved_gbp_microunits <= 0
    ):
        parser.error("a positive --max-reserved-gbp-microunits is required")
    if runtime is None:
        _emit_stop("AUTHORITY_COMPOSITION_UNCONFIGURED", completed=0)
        return 2

    ensure_control_plane_state_root()
    try:
        preflight = qualify_fresh_graphiti_event(
            proving_store=args.proving,
            unpublished_store=args.unpublished,
            event_id=args.event_id,
            ledger_seq=args.ledger_seq,
        )
        resolved_units = preflight.get("resolved_units")
        if not isinstance(resolved_units, list) or not resolved_units:
            raise ValueError("exact event preflight resolved no units")
    except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
        _emit_stop(
            "PREFLIGHT_REFUSED",
            completed=0,
        )
        print(
            json.dumps(
                {
                    "event": "GRAPHITI_WORKER_DIAGNOSTIC",
                    "failure_type": type(exc).__name__,
                    "failure": str(exc),
                    "public_dispatch": False,
                    "auto_publish": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 2
    reserved_bound = (
        len(resolved_units) * GRAPHITI_UNIT_RESERVATION_GBP_MICROUNITS
    )
    if reserved_bound > args.max_reserved_gbp_microunits:
        _emit_stop("RESERVED_SPEND_BOUND_EXCEEDED", completed=0)
        return 2
    print(
        json.dumps(
            {
                "event": "GRAPHITI_EVENT_PREFLIGHT",
                "preflight": preflight,
                "reserved_gbp_microunits_bound": reserved_bound,
                "max_reserved_gbp_microunits": (
                    args.max_reserved_gbp_microunits
                ),
                "public_dispatch": False,
                "auto_publish": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    owner_id = f"hermes-graphiti:{uuid.uuid4()}"
    return _run(
        consume=lambda: consume_next_graphiti_event(
            proving_store=args.proving,
            unpublished_store=args.unpublished,
            graphiti=runtime.graphiti,
            owner_id=owner_id,
            event_id=args.event_id,
            require_fresh=True,
            recover_model_usage=False,
            max_dispatch_seconds=args.max_runtime_seconds,
            prepared_event_preflight=preflight,
            max_reserved_gbp_microunits=args.max_reserved_gbp_microunits,
            graphiti_admission_factory=runtime.admission_factory,
            unit_authority_resolver=runtime.bind_unit_authority,
            require_graphiti_admission=True,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
