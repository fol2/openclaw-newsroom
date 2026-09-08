"""Controller-owned, append-only model invocation usage accounting.

The service owns the durable relationship between one qualified unit of work and
every provider leaf dispatched for it.  Aggregate token views are telemetry, not
normal production quotas; missing usage is always represented explicitly.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TypeGuard

from newsroom.authority.canonical import canonical_json_bytes, digest_canonical
from newsroom.control_plane.cycle_governor import CONT_WRITER_ROUTE
from newsroom.control_plane.graphiti_requests import (
    GraphitiInternalRequestIdentity,
    load_checked_graphiti_call_shape_policy,
)
from newsroom.control_plane.issue_790_contract import (
    issue_790_approved_plan_contracts,
)
from newsroom.control_plane.issue_790_step16_activation import (
    effective_issue_790_plan_contract,
)
from newsroom.control_plane.sqlite_profile import apply_control_plane_sqlite_profile
from newsroom.control_plane.veto import assert_private_store

MODEL_USAGE_SCHEMA_VERSION = "newsroom.model-usage.v3"
MODEL_USAGE_INTERFACE_SCHEMA_VERSION = "newsroom.model-usage.v4"
MODEL_USAGE_MIGRATION_ID = "model-usage-v4-conservative-disposition"
CONSERVATIVE_DISPOSITION_SCHEMA_VERSION = (
    "newsroom.model-usage.conservative-disposition.v2"
)
CONSERVATIVE_DISPOSITION_AUTHORITY_SCHEMA_VERSION = (
    "newsroom.model-usage.conservative-disposition-authority.v2"
)
_MODEL_USAGE_MIGRATIONS = (
    ("model-usage-v1", "newsroom.model-usage.v1"),
    ("model-usage-v2", "newsroom.model-usage.v2"),
    ("model-usage-v3", "newsroom.model-usage.v3"),
    (MODEL_USAGE_MIGRATION_ID, MODEL_USAGE_INTERFACE_SCHEMA_VERSION),
)
_HERMETIC_CONT_CONFIG_IDENTITIES = frozenset(
    {
        "cont-writer-grok-hermetic-command-v2",
        "cont-writer-grok-hermetic-command-v3",
        "cont-writer-cursor-hermetic-command-v2",
        "native-evidence-assessor-grok-hermetic-command-v1",
    }
)
DAILY_USAGE_ALERT_TOKENS = 500_000


class ModelUsageIntegrityError(ValueError):
    """Retained model-usage evidence is malformed or contradictory."""


class ModelUsageAdmissionError(RuntimeError):
    """A leaf failed deterministic pre-dispatch admission."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "MODEL_USAGE_ADMISSION_HELD",
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class WorkloadClass(StrEnum):
    NATIVE_EVIDENCE_ASSESSOR = "NATIVE_EVIDENCE_ASSESSOR"
    NATIVE_RETRIEVAL_EMBEDDING = "NATIVE_RETRIEVAL_EMBEDDING"
    CONT_WRITER_PRIMARY = "CONT_WRITER_PRIMARY"
    CONT_WRITER_FALLBACK = "CONT_WRITER_FALLBACK"
    CONT_ROUTE_HEALTH_PROBE = "CONT_ROUTE_HEALTH_PROBE"
    GRAPHITI_CHAT_PRIMARY = "GRAPHITI_CHAT_PRIMARY"
    GRAPHITI_CHAT_FALLBACK = "GRAPHITI_CHAT_FALLBACK"
    GRAPHITI_EMBEDDING = "GRAPHITI_EMBEDDING"


class UsageStatus(StrEnum):
    REPORTED = "REPORTED"
    ESTIMATED = "ESTIMATED"
    UNREPORTED = "UNREPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"


_GRAPHITI_COMPLETED_USEFUL_OUTCOMES = frozenset(
    {"GRAPHITI_SUCCESS", "GRAPHITI_SUCCESS_ZERO_PROPOSALS"}
)
_UNRESOLVED_USAGE_STATUSES = frozenset(
    {"UNREPORTED", "AMBIGUOUS", "INVALID"}
)
_GRAPHITI_CIRCUIT_ROUTES = (
    WorkloadClass.GRAPHITI_CHAT_PRIMARY.value,
    WorkloadClass.GRAPHITI_CHAT_FALLBACK.value,
    WorkloadClass.GRAPHITI_EMBEDDING.value,
)


_PROVENANCE = frozenset(
    {"PROVIDER_REPORTED", "CLI_DERIVED", "BOUNDED_ESTIMATE", "UNAVAILABLE"}
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_usage_migrations(
    migration_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_invocation_policies(
    canonical_digest TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL,
    version TEXT NOT NULL,
    workload_class TEXT NOT NULL,
    provider TEXT NOT NULL,
    route TEXT NOT NULL,
    model TEXT NOT NULL,
    qualified INTEGER NOT NULL CHECK(qualified IN (0,1)),
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_work_envelopes(
    envelope_id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL,
    workload_class TEXT NOT NULL,
    admitted_at TEXT NOT NULL,
    canonical_digest TEXT NOT NULL UNIQUE,
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_zero_call_admissions(
    decision_id TEXT PRIMARY KEY,
    decision TEXT NOT NULL CHECK(decision IN ('HOLD','REJECT')),
    cycle_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_invocation_context_manifests(
    context_manifest_digest TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    route TEXT NOT NULL,
    evidence_package_digest TEXT NOT NULL,
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_invocation_context_observations(
    observation_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL UNIQUE,
    context_manifest_digest TEXT NOT NULL,
    provider_context_tokens INTEGER,
    record_json TEXT NOT NULL,
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(context_manifest_digest)
        REFERENCES model_invocation_context_manifests(context_manifest_digest)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_invocation_allocations(
    invocation_id TEXT PRIMARY KEY,
    envelope_id TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    leaf_ordinal INTEGER NOT NULL CHECK(leaf_ordinal > 0),
    workload_class TEXT NOT NULL,
    policy_digest TEXT NOT NULL,
    provider TEXT NOT NULL,
    route TEXT NOT NULL,
    model TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    parent_invocation_id TEXT,
    allocated_at TEXT NOT NULL,
    canonical_digest TEXT NOT NULL UNIQUE,
    record_json TEXT NOT NULL,
    UNIQUE(envelope_id, leaf_ordinal),
    UNIQUE(envelope_id, request_digest),
    FOREIGN KEY(envelope_id) REFERENCES model_work_envelopes(envelope_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(policy_digest) REFERENCES model_invocation_policies(canonical_digest)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(parent_invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_transport_observations(
    observation_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    state TEXT NOT NULL,
    evidence_digest TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_invocation_provider_attempt_links(
    link_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL UNIQUE,
    provider_attempt_id TEXT NOT NULL,
    linked_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_invocation_terminals(
    terminal_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL UNIQUE,
    usage_status TEXT NOT NULL,
    outcome TEXT NOT NULL,
    failure_class TEXT,
    completed_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_provider_telemetry(
    telemetry_record_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL,
    provider_telemetry_digest TEXT NOT NULL,
    record_json TEXT NOT NULL,
    UNIQUE(invocation_id, provider_telemetry_digest),
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_work_outcomes(
    outcome_digest TEXT PRIMARY KEY,
    envelope_id TEXT NOT NULL UNIQUE,
    outcome TEXT NOT NULL,
    terminal_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY(envelope_id) REFERENCES model_work_envelopes(envelope_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_usage_cycle_outcomes(
    cycle_digest TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL UNIQUE,
    outcome_class TEXT NOT NULL,
    terminal_at TEXT NOT NULL,
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_usage_reconciliations(
    reconciliation_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    UNIQUE(invocation_id, reconciliation_digest),
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_usage_conservative_dispositions(
    disposition_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL UNIQUE,
    terminal_digest TEXT NOT NULL UNIQUE,
    allocation_digest TEXT NOT NULL,
    policy_digest TEXT NOT NULL,
    approved_plan_digest TEXT NOT NULL UNIQUE,
    authority_digest TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    approval_reference TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    usage_status TEXT NOT NULL CHECK(usage_status='ESTIMATED'),
    record_json TEXT NOT NULL,
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(terminal_digest) REFERENCES model_invocation_terminals(terminal_digest)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(allocation_digest)
        REFERENCES model_invocation_allocations(canonical_digest)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(policy_digest) REFERENCES model_invocation_policies(canonical_digest)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS model_usage_route_circuit_events(
    event_digest TEXT PRIMARY KEY,
    route TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('OPEN','CLOSED')),
    reason TEXT NOT NULL,
    invocation_id TEXT,
    recorded_at TEXT NOT NULL,
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS graphiti_internal_requests(
    canonical_digest TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL UNIQUE,
    envelope_id TEXT NOT NULL,
    graphiti_attempt_id TEXT NOT NULL,
    internal_ordinal INTEGER NOT NULL CHECK(internal_ordinal > 0),
    semantic_state_digest TEXT NOT NULL,
    provider_attempt_id TEXT NOT NULL,
    call_shape_policy_digest TEXT NOT NULL,
    record_json TEXT NOT NULL,
    UNIQUE(graphiti_attempt_id, semantic_state_digest),
    UNIQUE(graphiti_attempt_id, internal_ordinal),
    UNIQUE(provider_attempt_id),
    FOREIGN KEY(invocation_id) REFERENCES model_invocation_allocations(invocation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY(envelope_id) REFERENCES model_work_envelopes(envelope_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS graphiti_internal_request_refusals(
    refusal_digest TEXT PRIMARY KEY,
    envelope_id TEXT NOT NULL,
    attempted_ordinal INTEGER NOT NULL CHECK(attempted_ordinal > 0),
    route TEXT NOT NULL,
    semantic_state_digest TEXT NOT NULL,
    call_shape_policy_digest TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    refused_at TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY(envelope_id) REFERENCES model_work_envelopes(envelope_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS model_usage_allocated_at
ON model_invocation_allocations(allocated_at, invocation_id);
CREATE INDEX IF NOT EXISTS model_usage_completed_at
ON model_invocation_terminals(completed_at, invocation_id);
CREATE INDEX IF NOT EXISTS model_usage_route_state
ON model_usage_route_circuit_events(route, recorded_at);
"""


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ModelUsageIntegrityError("timestamp must be timezone-aware")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ModelUsageIntegrityError("retained timestamp lacks timezone")
    return parsed.astimezone(UTC)


def _token(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value.encode("utf-8")) > 512
    ):
        raise ModelUsageIntegrityError(f"{field} must be bounded canonical text")
    return value


def _canonical_circuit_route(route: str) -> str:
    return (
        CONT_WRITER_ROUTE
        if route.startswith("CONT_") and route != "CONT_HEALTH_PROBE"
        else route
    )


def _approved_plan_contracts_for_store(
    connection: sqlite3.Connection,
) -> tuple:
    contracts = list(issue_790_approved_plan_contracts())
    if connection.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name='issue_790_step16_activations'"
    ).fetchone():
        for row in connection.execute(
            "SELECT plan_digest FROM issue_790_step16_activations"
        ):
            try:
                contracts.append(
                    effective_issue_790_plan_contract(
                        str(row[0]),
                        connection=connection,
                    )
                )
            except KeyError:
                continue
    return tuple(sorted(contracts, key=lambda item: item.plan_digest))


def _usage_blocking_routes(connection: sqlite3.Connection) -> set[str]:
    approved_contracts = _approved_plan_contracts_for_store(connection)
    approved_bindings = " OR ".join(
        "(d.approved_plan_digest=? AND d.invocation_id=?)"
        for _ in approved_contracts
    )
    parameters = tuple(
        value
        for contract in approved_contracts
        for value in (contract.plan_digest, contract.invocation_id)
    )
    canary_non_success_leaf = ""
    if connection.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name='issue_790_bounded_canary_consumptions'"
    ).fetchone() and connection.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name='issue_790_bounded_canary_outcomes'"
    ).fetchone():
        # Failed #790 canary leaves must not permanently block the next
        # AUTHORISED_OPERATOR_RESET on the same route (wall-time / sequence).
        canary_non_success_leaf = (
            "EXISTS (SELECT 1 "
            "FROM model_invocation_allocations a_canary "
            "JOIN issue_790_bounded_canary_consumptions c "
            "ON c.event_id=json_extract(a_canary.record_json,'$.cycle_id') "
            "JOIN issue_790_bounded_canary_outcomes o "
            "ON o.consumption_digest=c.consumption_digest "
            "WHERE a_canary.invocation_id=t.invocation_id "
            "AND json_extract(o.record_json,'$.result_class') "
            "!= 'TRUTHFUL_PROVIDER_SUCCESS') "
        )
    canary_runtime_leaf_exclusion = (
        f"AND NOT {canary_non_success_leaf}" if canary_non_success_leaf else ""
    )
    policy_breach_clause = "json_extract(t.record_json,'$.policy_breach') IS NOT NULL"
    if canary_non_success_leaf:
        policy_breach_clause = (
            f"({policy_breach_clause} AND NOT ({canary_non_success_leaf}))"
        )
    rows = connection.execute(
        "SELECT a.route FROM model_invocation_terminals t "
        "JOIN model_invocation_allocations a "
        "ON a.invocation_id=t.invocation_id "
        "WHERE (t.usage_status IN "
        "('UNREPORTED','AMBIGUOUS','INVALID') "
        "AND NOT EXISTS (SELECT 1 FROM model_usage_reconciliations r "
        "WHERE r.invocation_id=t.invocation_id) "
        "AND NOT EXISTS (SELECT 1 "
        "FROM model_usage_conservative_dispositions d "
        "WHERE d.invocation_id=t.invocation_id "
        f"AND ({approved_bindings})) "
        f"{canary_runtime_leaf_exclusion}"
        ") "
        f"OR {policy_breach_clause} "
        "OR EXISTS (SELECT 1 FROM model_usage_reconciliations r "
        "WHERE r.invocation_id=t.invocation_id "
        "AND json_extract(r.record_json,'$.policy_breach') IS NOT NULL)",
        parameters,
    ).fetchall()
    return {_canonical_circuit_route(str(row[0])) for row in rows}


def _non_negative(value: int | None, *, field: str) -> int | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ModelUsageIntegrityError(f"{field} must be a non-negative integer")
    return value


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class UsageComponents:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_read_tokens: int | None = None
    cached_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    context_tokens: int | None = None
    total_tokens: int | None = None
    provenance: str = "UNAVAILABLE"

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cached_read_tokens",
            "cached_write_tokens",
            "reasoning_tokens",
            "context_tokens",
            "total_tokens",
        ):
            _non_negative(getattr(self, name), field=name)
        if self.provenance not in _PROVENANCE:
            raise ModelUsageIntegrityError("component provenance is invalid")

    def as_record(self) -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_read_tokens": self.cached_read_tokens,
            "cached_write_tokens": self.cached_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "context_tokens": self.context_tokens,
            "total_tokens": self.total_tokens,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class InvocationEfficiencyPolicy:
    policy_id: str
    version: str
    workload_class: WorkloadClass
    provider: str
    route: str
    model: str
    reasoning: str
    one_turn: bool
    exact_input: bool
    skills_enabled: bool
    tools_enabled: bool
    mcp_enabled: bool
    prior_message_count: int
    command_semantic_version: str
    command_flags: tuple[str, ...]
    context_manifest_schema_version: str
    disabled_capabilities: tuple[str, ...]
    implementation_revision: str
    calibration_only: bool
    allowed_candidate_ids: tuple[str, ...]
    max_prompt_bytes: int
    max_context_tokens: int
    max_output_tokens: int
    max_total_tokens: int
    prompt_contract_version: str
    output_schema_digest: str
    allowed_context_identities: tuple[str, ...]
    allowed_config_identities: tuple[str, ...]
    hard_estimate_ceiling_tokens: int | None
    evidence_digest: str
    qualified: bool
    canonical_digest: str

    @classmethod
    def create(cls, **values: object) -> InvocationEfficiencyPolicy:
        values.pop("canonical_digest", None)
        values.setdefault("command_semantic_version", "UNSPECIFIED")
        values.setdefault("command_flags", ())
        values.setdefault("context_manifest_schema_version", "UNSPECIFIED")
        values.setdefault("disabled_capabilities", ())
        values.setdefault("implementation_revision", "UNSPECIFIED")
        values.setdefault("calibration_only", False)
        values.setdefault("allowed_candidate_ids", ())
        workload = values.get("workload_class")
        if not isinstance(workload, WorkloadClass):
            raise ModelUsageIntegrityError("policy workload class must be typed")
        record = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            **{
                name: (
                    value.value
                    if isinstance(value, StrEnum)
                    else list(value)
                    if isinstance(value, tuple)
                    else value
                )
                for name, value in values.items()
            },
        }
        policy = cls(**values, canonical_digest=digest_canonical(record))  # type: ignore[arg-type]
        policy._validate()
        return policy

    def _validate(self) -> None:
        for name in (
            "policy_id",
            "version",
            "provider",
            "route",
            "model",
            "reasoning",
            "prompt_contract_version",
            "output_schema_digest",
            "evidence_digest",
            "command_semantic_version",
            "implementation_revision",
            "context_manifest_schema_version",
        ):
            _token(str(getattr(self, name)), field=name)
        for name in (
            "max_prompt_bytes",
            "max_context_tokens",
            "max_output_tokens",
            "max_total_tokens",
        ):
            value = getattr(self, name)
            if not _is_int(value) or value <= 0:
                raise ModelUsageIntegrityError(f"{name} must be positive")
        if not isinstance(self.one_turn, bool) or not isinstance(
            self.exact_input, bool
        ):
            raise ModelUsageIntegrityError("policy turn/input controls must be boolean")
        for name in ("skills_enabled", "tools_enabled", "mcp_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ModelUsageIntegrityError(f"policy {name} must be boolean")
        if not isinstance(self.calibration_only, bool):
            raise ModelUsageIntegrityError("policy calibration_only must be boolean")
        if self.calibration_only and not self.allowed_candidate_ids:
            raise ModelUsageIntegrityError(
                "calibration policy must bind at least one candidate"
            )
        if not self.calibration_only and self.allowed_candidate_ids:
            raise ModelUsageIntegrityError(
                "non-calibration policy cannot bind calibration candidates"
            )
        for candidate_id in self.allowed_candidate_ids:
            _token(candidate_id, field="allowed_candidate_id")
        if len(set(self.allowed_candidate_ids)) != len(self.allowed_candidate_ids):
            raise ModelUsageIntegrityError("allowed calibration candidates repeat")
        if not isinstance(self.command_flags, tuple) or not all(
            isinstance(value, str) for value in self.command_flags
        ):
            raise ModelUsageIntegrityError("policy command flags are invalid")
        if len(set(self.disabled_capabilities)) != len(self.disabled_capabilities):
            raise ModelUsageIntegrityError("disabled capabilities repeat")
        for capability in self.disabled_capabilities:
            _token(capability, field="disabled_capability")
        if _is_hermetic_cont_policy(self) and (
            self.command_semantic_version == "UNSPECIFIED"
            or not self.command_flags
            or self.context_manifest_schema_version == "UNSPECIFIED"
            or not self.disabled_capabilities
            or not re.fullmatch(r"[0-9a-f]{40}", self.implementation_revision)
        ):
            raise ModelUsageIntegrityError(
                "hermetic CONT policy lacks an exact command/manifest binding"
            )
        if not _is_int(self.prior_message_count) or self.prior_message_count < 0:
            raise ModelUsageIntegrityError(
                "policy prior message count must be non-negative"
            )
        if self.qualified and (not self.one_turn or not self.exact_input):
            raise ModelUsageIntegrityError(
                "qualified invocation policy must be one-turn exact-input"
            )
        if self.max_total_tokens < self.max_output_tokens:
            raise ModelUsageIntegrityError("policy total is below output maximum")
        if not self.allowed_context_identities or len(
            set(self.allowed_context_identities)
        ) != len(self.allowed_context_identities):
            raise ModelUsageIntegrityError("allowed context identities are invalid")
        for identity in self.allowed_context_identities:
            _token(identity, field="allowed_context_identity")
        if not self.allowed_config_identities or len(
            set(self.allowed_config_identities)
        ) != len(self.allowed_config_identities):
            raise ModelUsageIntegrityError("allowed config identities are invalid")
        for identity in self.allowed_config_identities:
            _token(identity, field="allowed_config_identity")
        if self.hard_estimate_ceiling_tokens is not None and (
            not _is_int(self.hard_estimate_ceiling_tokens)
            or self.hard_estimate_ceiling_tokens < self.max_total_tokens
        ):
            raise ModelUsageIntegrityError(
                "hard estimate ceiling must cover the policy total"
            )
        if not isinstance(self.one_turn, bool) or not isinstance(self.qualified, bool):
            raise ModelUsageIntegrityError("policy booleans must be typed")

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "canonical_digest": self.canonical_digest,
            "policy_id": self.policy_id,
            "version": self.version,
            "workload_class": self.workload_class.value,
            "provider": self.provider,
            "route": self.route,
            "model": self.model,
            "reasoning": self.reasoning,
            "one_turn": self.one_turn,
            "exact_input": self.exact_input,
            "skills_enabled": self.skills_enabled,
            "tools_enabled": self.tools_enabled,
            "mcp_enabled": self.mcp_enabled,
            "prior_message_count": self.prior_message_count,
            "command_semantic_version": self.command_semantic_version,
            "command_flags": list(self.command_flags),
            "context_manifest_schema_version": self.context_manifest_schema_version,
            "disabled_capabilities": list(self.disabled_capabilities),
            "implementation_revision": self.implementation_revision,
            "calibration_only": self.calibration_only,
            "allowed_candidate_ids": list(self.allowed_candidate_ids),
            "max_prompt_bytes": self.max_prompt_bytes,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
            "prompt_contract_version": self.prompt_contract_version,
            "output_schema_digest": self.output_schema_digest,
            "allowed_context_identities": list(self.allowed_context_identities),
            "allowed_config_identities": list(self.allowed_config_identities),
            "hard_estimate_ceiling_tokens": self.hard_estimate_ceiling_tokens,
            "evidence_digest": self.evidence_digest,
            "qualified": self.qualified,
        }


@dataclass(frozen=True, slots=True)
class WorkEnvelope:
    envelope_id: str
    cycle_id: str
    workload_class: WorkloadClass
    admitted_at: datetime
    admission_decision_id: str | None
    candidate_id: str | None
    hypothesis_digest: str | None
    evidence_package_digest: str | None
    ingest_id: str | None
    graphiti_attempt_id: str | None
    canonical_digest: str

    @classmethod
    def create(cls, **values: object) -> WorkEnvelope:
        values.pop("envelope_id", None)
        values.pop("canonical_digest", None)
        workload = values.get("workload_class")
        admitted_at = values.get("admitted_at")
        if not isinstance(workload, WorkloadClass):
            raise ModelUsageIntegrityError("envelope workload class must be typed")
        if not isinstance(admitted_at, datetime):
            raise ModelUsageIntegrityError("envelope admitted_at must be datetime")
        identity = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "cycle_id": values["cycle_id"],
            "workload_class": workload.value,
            "admission_decision_id": values.get("admission_decision_id"),
            "candidate_id": values.get("candidate_id"),
            "hypothesis_digest": values.get("hypothesis_digest"),
            "evidence_package_digest": values.get("evidence_package_digest"),
            "ingest_id": values.get("ingest_id"),
            "graphiti_attempt_id": values.get("graphiti_attempt_id"),
        }
        envelope_id = digest_canonical(identity)
        envelope = cls(
            **values,  # type: ignore[arg-type]
            envelope_id=envelope_id,
            canonical_digest=digest_canonical(
                {
                    **identity,
                    "envelope_id": envelope_id,
                    "admitted_at": _utc_text(admitted_at),
                }
            ),
        )
        envelope._validate()
        return envelope

    def _validate(self) -> None:
        _token(self.cycle_id, field="cycle_id")
        native_assessor = (
            self.workload_class is WorkloadClass.NATIVE_EVIDENCE_ASSESSOR
        )
        native_embedding = (
            self.workload_class is WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING
        )
        cont = self.workload_class in {
            WorkloadClass.CONT_WRITER_PRIMARY,
            WorkloadClass.CONT_WRITER_FALLBACK,
            WorkloadClass.CONT_ROUTE_HEALTH_PROBE,
        }
        graphiti = self.workload_class in {
            WorkloadClass.GRAPHITI_CHAT_PRIMARY,
            WorkloadClass.GRAPHITI_CHAT_FALLBACK,
            WorkloadClass.GRAPHITI_EMBEDDING,
        }
        if (
            native_assessor
            and not all(
                (
                    self.candidate_id,
                    self.hypothesis_digest,
                    self.evidence_package_digest,
                )
            )
        ):
            raise ModelUsageIntegrityError(
                "native assessor envelope lacks acquired evidence identities"
            )
        if (
            cont
            and self.workload_class is not WorkloadClass.CONT_ROUTE_HEALTH_PROBE
            and not all(
                (
                    self.admission_decision_id,
                    self.candidate_id,
                    self.hypothesis_digest,
                    self.evidence_package_digest,
                )
            )
        ):
            raise ModelUsageIntegrityError("CONT envelope lacks editorial identities")
        if graphiti and not self.ingest_id:
            raise ModelUsageIntegrityError("Graphiti envelope lacks ingest identity")
        if native_embedding and not self.ingest_id:
            raise ModelUsageIntegrityError(
                "native embedding envelope lacks passage identity"
            )
        _utc_text(self.admitted_at)

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "canonical_digest": self.canonical_digest,
            "envelope_id": self.envelope_id,
            "cycle_id": self.cycle_id,
            "workload_class": self.workload_class.value,
            "admitted_at": _utc_text(self.admitted_at),
            "admission_decision_id": self.admission_decision_id,
            "candidate_id": self.candidate_id,
            "hypothesis_digest": self.hypothesis_digest,
            "evidence_package_digest": self.evidence_package_digest,
            "ingest_id": self.ingest_id,
            "graphiti_attempt_id": self.graphiti_attempt_id,
        }


@dataclass(frozen=True, slots=True)
class InvocationAllocation:
    invocation_id: str
    envelope_id: str
    cycle_id: str
    leaf_ordinal: int
    workload_class: WorkloadClass
    invocation_policy_digest: str
    provider: str
    route: str
    model: str
    reasoning: str
    prompt_contract_version: str
    prompt_bytes: int
    prompt_digest: str
    request_digest: str
    output_schema_digest: str
    max_output_tokens: int
    context_manifest_digest: str
    context_identity: str
    config_identity: str
    one_turn: bool
    exact_input: bool
    skills_enabled: bool
    tools_enabled: bool
    mcp_enabled: bool
    prior_message_count: int
    allocated_at: datetime
    recovery_deadline_at: datetime
    parent_invocation_id: str | None
    canonical_digest: str

    @classmethod
    def create(cls, **values: object) -> InvocationAllocation:
        values.pop("invocation_id", None)
        values.pop("canonical_digest", None)
        workload = values.get("workload_class")
        allocated_at = values.get("allocated_at")
        if not isinstance(workload, WorkloadClass):
            raise ModelUsageIntegrityError("allocation workload class must be typed")
        if not isinstance(allocated_at, datetime):
            raise ModelUsageIntegrityError("allocation timestamp must be datetime")
        recovery_deadline_at = values.get("recovery_deadline_at")
        if not isinstance(recovery_deadline_at, datetime):
            raise ModelUsageIntegrityError(
                "allocation recovery deadline must be datetime"
            )
        identity = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "envelope_id": values["envelope_id"],
            "cycle_id": values["cycle_id"],
            "leaf_ordinal": values["leaf_ordinal"],
            "workload_class": workload.value,
            "request_digest": values["request_digest"],
            "route": values["route"],
            "parent_invocation_id": values.get("parent_invocation_id"),
        }
        invocation_id = digest_canonical(identity)
        allocation = cls(
            **values,  # type: ignore[arg-type]
            invocation_id=invocation_id,
            canonical_digest=digest_canonical(
                {
                    **identity,
                    "invocation_id": invocation_id,
                    "invocation_policy_digest": values["invocation_policy_digest"],
                    "provider": values["provider"],
                    "model": values["model"],
                    "reasoning": values["reasoning"],
                    "prompt_contract_version": values["prompt_contract_version"],
                    "prompt_bytes": values["prompt_bytes"],
                    "prompt_digest": values["prompt_digest"],
                    "output_schema_digest": values["output_schema_digest"],
                    "max_output_tokens": values["max_output_tokens"],
                    "context_manifest_digest": values["context_manifest_digest"],
                    "context_identity": values["context_identity"],
                    "config_identity": values["config_identity"],
                    "one_turn": values["one_turn"],
                    "exact_input": values["exact_input"],
                    "skills_enabled": values["skills_enabled"],
                    "tools_enabled": values["tools_enabled"],
                    "mcp_enabled": values["mcp_enabled"],
                    "prior_message_count": values["prior_message_count"],
                    "allocated_at": _utc_text(allocated_at),
                    "recovery_deadline_at": _utc_text(recovery_deadline_at),
                }
            ),
        )
        allocation._validate()
        return allocation

    def _validate(self) -> None:
        for name in (
            "envelope_id",
            "cycle_id",
            "invocation_policy_digest",
            "provider",
            "route",
            "model",
            "reasoning",
            "prompt_contract_version",
            "prompt_digest",
            "request_digest",
            "output_schema_digest",
            "context_manifest_digest",
            "context_identity",
            "config_identity",
        ):
            _token(str(getattr(self, name)), field=name)
        if not _is_int(self.leaf_ordinal) or self.leaf_ordinal <= 0:
            raise ModelUsageIntegrityError("leaf ordinal must be positive")
        if not _is_int(self.prompt_bytes) or self.prompt_bytes < 0:
            raise ModelUsageIntegrityError("prompt bytes must be non-negative")
        if not _is_int(self.max_output_tokens) or self.max_output_tokens <= 0:
            raise ModelUsageIntegrityError("max output tokens must be positive")
        for name in (
            "one_turn",
            "exact_input",
            "skills_enabled",
            "tools_enabled",
            "mcp_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ModelUsageIntegrityError(
                    f"allocation {name} must be boolean"
                )
        if not _is_int(self.prior_message_count) or self.prior_message_count < 0:
            raise ModelUsageIntegrityError(
                "allocation prior message count must be non-negative"
            )
        if self.recovery_deadline_at <= self.allocated_at:
            raise ModelUsageIntegrityError(
                "allocation recovery deadline must follow allocation"
            )
        _utc_text(self.allocated_at)
        _utc_text(self.recovery_deadline_at)

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "canonical_digest": self.canonical_digest,
            "invocation_id": self.invocation_id,
            "envelope_id": self.envelope_id,
            "cycle_id": self.cycle_id,
            "leaf_ordinal": self.leaf_ordinal,
            "workload_class": self.workload_class.value,
            "invocation_policy_digest": self.invocation_policy_digest,
            "provider": self.provider,
            "route": self.route,
            "model": self.model,
            "reasoning": self.reasoning,
            "prompt_contract_version": self.prompt_contract_version,
            "prompt_bytes": self.prompt_bytes,
            "prompt_digest": self.prompt_digest,
            "request_digest": self.request_digest,
            "output_schema_digest": self.output_schema_digest,
            "max_output_tokens": self.max_output_tokens,
            "context_manifest_digest": self.context_manifest_digest,
            "context_identity": self.context_identity,
            "config_identity": self.config_identity,
            "one_turn": self.one_turn,
            "exact_input": self.exact_input,
            "skills_enabled": self.skills_enabled,
            "tools_enabled": self.tools_enabled,
            "mcp_enabled": self.mcp_enabled,
            "prior_message_count": self.prior_message_count,
            "allocated_at": _utc_text(self.allocated_at),
            "recovery_deadline_at": _utc_text(self.recovery_deadline_at),
            "parent_invocation_id": self.parent_invocation_id,
        }


@dataclass(frozen=True, slots=True)
class InvocationTerminal:
    terminal_digest: str
    invocation_id: str
    outcome: str
    failure_class: str | None
    usage_status: UsageStatus
    components: UsageComponents
    dispatch_at: datetime | None
    completed_at: datetime
    observed_at: datetime
    provider_telemetry_digest: str | None
    raw_telemetry_pointer: str | None
    estimate_policy_digest: str | None
    estimate_calculation: str | None
    pre_dispatch_zero_proved: bool
    od_011_reference: str | None
    subscription_cli_chat_not_cash_debited: bool
    policy_breach: str | None

    @classmethod
    def create(cls, **values: object) -> InvocationTerminal:
        values.pop("terminal_digest", None)
        values.setdefault("provider_telemetry_digest", None)
        values.setdefault("raw_telemetry_pointer", None)
        values.setdefault("estimate_policy_digest", None)
        values.setdefault("estimate_calculation", None)
        values.setdefault("pre_dispatch_zero_proved", False)
        values.setdefault("od_011_reference", None)
        values.setdefault("policy_breach", None)
        status = values.get("usage_status")
        components = values.get("components")
        if not isinstance(status, UsageStatus):
            raise ModelUsageIntegrityError("usage status must be typed")
        if not isinstance(components, UsageComponents):
            raise ModelUsageIntegrityError("usage components must be typed")
        record = _terminal_record(values, digest="")
        terminal = cls(
            **values,  # type: ignore[arg-type]
            terminal_digest=digest_canonical(record),
        )
        terminal._validate_shape()
        return terminal

    def _validate_shape(self) -> None:
        _token(self.invocation_id, field="invocation_id")
        _token(self.outcome, field="outcome")
        if self.failure_class is not None:
            _token(self.failure_class, field="failure_class")
        if self.dispatch_at is not None:
            _utc_text(self.dispatch_at)
        _utc_text(self.completed_at)
        _utc_text(self.observed_at)
        if self.dispatch_at is not None and self.completed_at < self.dispatch_at:
            raise ModelUsageIntegrityError("completion precedes dispatch")
        if self.observed_at < self.completed_at:
            raise ModelUsageIntegrityError("observation precedes completion")
        if not isinstance(self.pre_dispatch_zero_proved, bool):
            raise ModelUsageIntegrityError("pre-dispatch proof flag must be boolean")

    def as_record(self) -> dict[str, object]:
        return _terminal_record(self.__dict_values(), digest=self.terminal_digest)

    def __dict_values(self) -> dict[str, object]:
        return {
            "invocation_id": self.invocation_id,
            "outcome": self.outcome,
            "failure_class": self.failure_class,
            "usage_status": self.usage_status,
            "components": self.components,
            "dispatch_at": self.dispatch_at,
            "completed_at": self.completed_at,
            "observed_at": self.observed_at,
            "provider_telemetry_digest": self.provider_telemetry_digest,
            "raw_telemetry_pointer": self.raw_telemetry_pointer,
            "estimate_policy_digest": self.estimate_policy_digest,
            "estimate_calculation": self.estimate_calculation,
            "pre_dispatch_zero_proved": self.pre_dispatch_zero_proved,
            "od_011_reference": self.od_011_reference,
            "subscription_cli_chat_not_cash_debited": (
                self.subscription_cli_chat_not_cash_debited
            ),
            "policy_breach": self.policy_breach,
        }


def _terminal_record(values: Mapping[str, object], *, digest: str) -> dict[str, object]:
    status = values["usage_status"]
    components = values["components"]
    dispatch_at = values.get("dispatch_at")
    return {
        "schema_version": MODEL_USAGE_SCHEMA_VERSION,
        "terminal_digest": digest,
        "invocation_id": values["invocation_id"],
        "outcome": values["outcome"],
        "failure_class": values.get("failure_class"),
        "usage_status": status.value if isinstance(status, UsageStatus) else status,
        "components": (
            components.as_record()
            if isinstance(components, UsageComponents)
            else components
        ),
        "dispatch_at": (
            _utc_text(dispatch_at) if isinstance(dispatch_at, datetime) else None
        ),
        "completed_at": _utc_text(values["completed_at"]),  # type: ignore[arg-type]
        "observed_at": _utc_text(values["observed_at"]),  # type: ignore[arg-type]
        "provider_telemetry_digest": values.get("provider_telemetry_digest"),
        "raw_telemetry_pointer": values.get("raw_telemetry_pointer"),
        "estimate_policy_digest": values.get("estimate_policy_digest"),
        "estimate_calculation": values.get("estimate_calculation"),
        "pre_dispatch_zero_proved": values.get("pre_dispatch_zero_proved", False),
        "od_011_reference": values.get("od_011_reference"),
        "subscription_cli_chat_not_cash_debited": values[
            "subscription_cli_chat_not_cash_debited"
        ],
        "policy_breach": values.get("policy_breach"),
    }


def _invalid_reported_components(terminal: InvocationTerminal) -> str | None:
    if terminal.usage_status is not UsageStatus.REPORTED:
        return None
    components = terminal.components
    if components.total_tokens is None:
        return "REPORTED_TOTAL_MISSING"
    if components.provenance not in {"PROVIDER_REPORTED", "CLI_DERIVED"}:
        return "REPORTED_PROVENANCE_INVALID"
    input_tokens = components.input_tokens
    output_tokens = components.output_tokens
    known = tuple(
        value
        for value in (
            input_tokens,
            output_tokens,
            components.cached_read_tokens,
            components.cached_write_tokens,
            components.reasoning_tokens,
        )
        if value is not None
    )
    if not known:
        return None
    possible_totals = {sum(known)}
    if input_tokens is not None and output_tokens is not None:
        possible_totals.add(input_tokens + output_tokens)
        cache_read = components.cached_read_tokens or 0
        cache_write = components.cached_write_tokens or 0
        # Grok 1.0.10 headless: total = uncached input + output + cache
        # buckets; reasoning is nested in output, not additive.
        possible_totals.add(
            input_tokens + output_tokens + cache_read + cache_write
        )
    if components.total_tokens not in possible_totals:
        return "REPORTED_COMPONENT_TOTAL_INVALID"
    return None


def _retain_provider_telemetry(
    connection: sqlite3.Connection,
    *,
    invocation_id: str,
    provider_telemetry: Mapping[str, object],
) -> str:
    telemetry_value = dict(provider_telemetry)
    provider_telemetry_digest = digest_canonical(telemetry_value)
    telemetry_record = {
        "schema_version": MODEL_USAGE_SCHEMA_VERSION,
        "invocation_id": invocation_id,
        "provider_telemetry_digest": provider_telemetry_digest,
        "provider_telemetry": telemetry_value,
    }
    telemetry_record_digest = digest_canonical(telemetry_record)
    connection.execute(
        "INSERT OR IGNORE INTO model_provider_telemetry("
        "telemetry_record_digest,invocation_id,provider_telemetry_digest,record_json) "
        "VALUES(?,?,?,?)",
        (
            telemetry_record_digest,
            invocation_id,
            provider_telemetry_digest,
            _json(telemetry_record),
        ),
    )
    retained = connection.execute(
        "SELECT record_json FROM model_provider_telemetry "
        "WHERE telemetry_record_digest=?",
        (telemetry_record_digest,),
    ).fetchone()
    if retained is None or _object(retained[0]) != telemetry_record:
        raise ModelUsageIntegrityError("conflicting provider telemetry replay")
    return provider_telemetry_digest


class ModelUsageService:
    """Single SQLite authority for model usage, outcomes and exports."""

    def __init__(self, path: str) -> None:
        assert_private_store(path)
        self.path = path
        connection = self._connection()
        try:
            connection.executescript(_SCHEMA)
            applied_at = _utc_text(datetime.now(tz=UTC))
            connection.executemany(
                "INSERT OR IGNORE INTO model_usage_migrations("
                "migration_id,schema_version,applied_at) VALUES(?,?,?)",
                (
                    (migration_id, schema_version, applied_at)
                    for migration_id, schema_version in _MODEL_USAGE_MIGRATIONS
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        apply_control_plane_sqlite_profile(connection)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def register_policy(self, policy: InvocationEfficiencyPolicy) -> None:
        policy._validate()
        record = policy.as_record()
        self._insert_exact(
            table="model_invocation_policies",
            identity_column="canonical_digest",
            identity=policy.canonical_digest,
            record=record,
            sql="INSERT INTO model_invocation_policies("
            "canonical_digest,policy_id,version,workload_class,provider,route,model,"
            "qualified,record_json) VALUES(?,?,?,?,?,?,?,?,?)",
            values=(
                policy.canonical_digest,
                policy.policy_id,
                policy.version,
                policy.workload_class.value,
                policy.provider,
                policy.route,
                policy.model,
                int(policy.qualified),
                _json(record),
            ),
        )

    def qualified_policy(
        self,
        *,
        workload_class: WorkloadClass,
        provider: str,
        route: str,
        model: str,
        reasoning: str,
        candidate_id: str | None = None,
        implementation_revision: str | None = None,
        config_identity: str | None = None,
        output_schema_digest: str | None = None,
    ) -> InvocationEfficiencyPolicy:
        """Resolve one exact qualified route policy without fallback guessing."""

        connection = self._connection()
        try:
            rows = connection.execute(
                "SELECT record_json FROM model_invocation_policies "
                "WHERE workload_class=? AND provider=? AND route=? AND model=? "
                "AND qualified=1 ORDER BY rowid DESC",
                (workload_class.value, provider, route, model),
            ).fetchall()
        finally:
            connection.close()
        policy_records = [_object(row[0]) for row in rows]
        policies = [
            _policy_from_record(record)
            for record in policy_records
            if str(record.get("reasoning")) == reasoning
        ]
        if config_identity is not None:
            policies = [
                policy
                for policy in policies
                if config_identity in policy.allowed_config_identities
            ]
        if output_schema_digest is not None:
            policies = [
                policy
                for policy in policies
                if policy.output_schema_digest == output_schema_digest
            ]
        policies = [
            policy
            for policy in policies
            if not _is_hermetic_cont_policy(policy)
            or (
                policy.command_semantic_version != "UNSPECIFIED"
                and policy.implementation_revision == implementation_revision
            )
        ]
        policies = [
            policy
            for policy in policies
            if not policy.calibration_only
            or (
                candidate_id in policy.allowed_candidate_ids
                and implementation_revision == policy.implementation_revision
            )
        ]
        general_policies = [policy for policy in policies if not policy.calibration_only]
        if general_policies:
            policies = (
                general_policies[:1]
                if all(_is_hermetic_cont_policy(policy) for policy in general_policies)
                else general_policies
            )
        elif policies and all(_is_hermetic_cont_policy(policy) for policy in policies):
            # A later exact-head restaging supersedes an older bootstrap for the
            # same bounded candidate without mutating retained policy history.
            policies = policies[:1]
        if len(policies) != 1:
            raise ModelUsageAdmissionError(
                "exact qualified invocation policy is absent or ambiguous",
                reason_code="INVOCATION_POLICY_UNAVAILABLE",
            )
        return policies[0]

    def open_envelope(self, envelope: WorkEnvelope) -> None:
        envelope._validate()
        record = envelope.as_record()
        self._insert_exact(
            table="model_work_envelopes",
            identity_column="envelope_id",
            identity=envelope.envelope_id,
            record=record,
            sql="INSERT INTO model_work_envelopes("
            "envelope_id,cycle_id,workload_class,admitted_at,canonical_digest,record_json) "
            "VALUES(?,?,?,?,?,?)",
            values=(
                envelope.envelope_id,
                envelope.cycle_id,
                envelope.workload_class.value,
                _utc_text(envelope.admitted_at),
                envelope.canonical_digest,
                _json(record),
            ),
        )

    def resume_or_open_graphiti_envelope(
        self, envelope: WorkEnvelope
    ) -> WorkEnvelope:
        """Open a Graphiti envelope or return its exact durable restart value."""

        envelope._validate()
        if envelope.graphiti_attempt_id is None:
            raise ModelUsageIntegrityError(
                "Graphiti envelope resume lacks an attempt identity"
            )
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT record_json FROM model_work_envelopes WHERE envelope_id=?",
                (envelope.envelope_id,),
            ).fetchone()
            if row is None:
                self._insert_exact(
                    table="model_work_envelopes",
                    identity_column="envelope_id",
                    identity=envelope.envelope_id,
                    record=envelope.as_record(),
                    sql="INSERT INTO model_work_envelopes("
                    "envelope_id,cycle_id,workload_class,admitted_at,"
                    "canonical_digest,record_json) VALUES(?,?,?,?,?,?)",
                    values=(
                        envelope.envelope_id,
                        envelope.cycle_id,
                        envelope.workload_class.value,
                        _utc_text(envelope.admitted_at),
                        envelope.canonical_digest,
                        _json(envelope.as_record()),
                    ),
                    connection=connection,
                )
                connection.commit()
                return envelope
            record = _object(row[0])
            resumed = WorkEnvelope.create(
                cycle_id=str(record["cycle_id"]),
                workload_class=WorkloadClass(str(record["workload_class"])),
                admitted_at=_instant(str(record["admitted_at"])),
                admission_decision_id=record.get("admission_decision_id"),
                candidate_id=record.get("candidate_id"),
                hypothesis_digest=record.get("hypothesis_digest"),
                evidence_package_digest=record.get("evidence_package_digest"),
                ingest_id=record.get("ingest_id"),
                graphiti_attempt_id=record.get("graphiti_attempt_id"),
            )
            if (
                resumed.envelope_id != envelope.envelope_id
                or resumed.canonical_digest != record.get("canonical_digest")
            ):
                raise ModelUsageIntegrityError(
                    "retained Graphiti envelope identity is invalid"
                )
            connection.commit()
            return resumed
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def retain_context_manifest(self, record: Mapping[str, object]) -> None:
        """Retain the non-secret context proof before any provider dispatch."""

        retained = dict(record)
        digest = str(retained.pop("context_manifest_digest", ""))
        if not digest or digest_canonical(retained) != digest:
            raise ModelUsageAdmissionError("context manifest digest is invalid")
        required_text = (
            "provider",
            "route",
            "model",
            "reasoning",
            "command_semantic_version",
            "implementation_revision",
            "schema_version",
            "prompt_digest",
            "schema_digest",
            "system_digest",
            "evidence_package_digest",
        )
        if any(not isinstance(retained.get(field), str) for field in required_text):
            raise ModelUsageAdmissionError("context manifest identity is incomplete")
        evidence_package_bytes = retained.get("evidence_package_bytes")
        if not _is_int(evidence_package_bytes) or evidence_package_bytes <= 0:
            raise ModelUsageAdmissionError(
                "context manifest Evidence Package size is invalid"
            )
        zero_counts = (
            "prior_message_count",
            "skill_count",
            "tool_count",
            "mcp_server_count",
            "mcp_tool_count",
        )
        if any(retained.get(field) != 0 for field in zero_counts) or any(
            retained.get(field) is not False
            for field in ("skills_enabled", "tools_enabled", "mcp_enabled")
        ):
            raise ModelUsageAdmissionError(
                "context manifest contains an ambient capability"
            )
        forbidden = {
            "prompt",
            "system_prompt",
            "schema",
            "passages",
            "source_expression",
            "secret",
        }
        if forbidden.intersection(retained):
            raise ModelUsageAdmissionError("context manifest contains secret input")
        canonical_record = {"context_manifest_digest": digest, **retained}
        self._insert_exact(
            table="model_invocation_context_manifests",
            identity_column="context_manifest_digest",
            identity=digest,
            record=canonical_record,
            sql="INSERT INTO model_invocation_context_manifests("
            "context_manifest_digest,provider,route,evidence_package_digest,record_json) "
            "VALUES(?,?,?,?,?)",
            values=(
                digest,
                retained["provider"],
                retained["route"],
                retained["evidence_package_digest"],
                _json(canonical_record),
            ),
        )

    def retain_zero_call_admission(
        self, *, decision_id: str, decision: str, cycle_id: str, recorded_at: datetime
    ) -> None:
        if decision not in {"HOLD", "REJECT"}:
            raise ModelUsageIntegrityError("zero-call admission must be HOLD or REJECT")
        connection = self._connection()
        try:
            connection.execute(
                "INSERT OR IGNORE INTO model_zero_call_admissions("
                "decision_id,decision,cycle_id,recorded_at) VALUES(?,?,?,?)",
                (decision_id, decision, cycle_id, _utc_text(recorded_at)),
            )
            row = connection.execute(
                "SELECT decision,cycle_id,recorded_at FROM model_zero_call_admissions "
                "WHERE decision_id=?",
                (decision_id,),
            ).fetchone()
            if row is None or tuple(row) != (
                decision,
                cycle_id,
                _utc_text(recorded_at),
            ):
                raise ModelUsageIntegrityError("conflicting zero-call admission replay")
            connection.commit()
        finally:
            connection.close()

    def allocate(
        self,
        allocation: InvocationAllocation,
        *,
        owner_emergency_stop: bool,
    ) -> None:
        allocation._validate()
        if not isinstance(owner_emergency_stop, bool):
            raise ModelUsageAdmissionError(
                "owner emergency stop authority must be an explicit boolean"
            )
        if owner_emergency_stop:
            raise ModelUsageAdmissionError("owner emergency stop is active")
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            envelope = connection.execute(
                "SELECT cycle_id,record_json FROM model_work_envelopes WHERE envelope_id=?",
                (allocation.envelope_id,),
            ).fetchone()
            if envelope is None or str(envelope[0]) != allocation.cycle_id:
                raise ModelUsageAdmissionError("work envelope is absent or mismatched")
            policy_row = connection.execute(
                "SELECT record_json FROM model_invocation_policies WHERE canonical_digest=?",
                (allocation.invocation_policy_digest,),
            ).fetchone()
            if policy_row is None:
                raise ModelUsageAdmissionError("invocation policy is not registered")
            policy = _policy_from_record(_object(policy_row[0]))
            if not policy.qualified:
                raise ModelUsageAdmissionError("invocation policy is not qualified")
            self._validate_preflight(connection, allocation, policy)
            self._insert_allocation(connection, allocation)
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def allocate_graphiti_request(
        self,
        allocation: InvocationAllocation,
        *,
        identity: GraphitiInternalRequestIdentity,
        max_distinct_internal_requests: int,
    ) -> None:
        """Atomically retain one #728 leaf and its Graphiti request identity."""

        allocation._validate()
        if (
            isinstance(max_distinct_internal_requests, bool)
            or not isinstance(max_distinct_internal_requests, int)
            or max_distinct_internal_requests <= 0
        ):
            raise ModelUsageAdmissionError("Graphiti call-shape bound is invalid")
        identity.validate()
        record = identity.as_record()
        self._validate_graphiti_identity(allocation, identity)
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            envelope = connection.execute(
                "SELECT cycle_id FROM model_work_envelopes WHERE envelope_id=?",
                (allocation.envelope_id,),
            ).fetchone()
            if envelope is None or str(envelope[0]) != allocation.cycle_id:
                raise ModelUsageAdmissionError("work envelope is absent or mismatched")
            policy_row = connection.execute(
                "SELECT record_json FROM model_invocation_policies WHERE canonical_digest=?",
                (allocation.invocation_policy_digest,),
            ).fetchone()
            if policy_row is None:
                raise ModelUsageAdmissionError("invocation policy is not registered")
            policy = _policy_from_record(_object(policy_row[0]))
            if not policy.qualified:
                raise ModelUsageAdmissionError("invocation policy is not qualified")
            manifest_row = connection.execute(
                "SELECT record_json FROM model_invocation_context_manifests "
                "WHERE context_manifest_digest=?",
                (identity.context_manifest_digest,),
            ).fetchone()
            if manifest_row is None:
                raise ModelUsageAdmissionError("Graphiti context manifest is absent")
            manifest = _object(manifest_row[0])
            if (
                manifest.get("effective_revision_digest")
                != identity.effective_revision_digest
                or manifest.get("ingest_obligation_id")
                != identity.ingest_obligation_id
                or manifest.get("graphiti_attempt_id")
                != identity.graphiti_attempt_id
                or manifest.get("provider_attempt_id")
                != identity.provider_attempt_id
                or manifest.get("semantic_state_digest")
                != identity.semantic_state_digest
            ):
                raise ModelUsageAdmissionError(
                    "Graphiti context manifest differs from its request identity"
                )
            expected_environment_keys = (
                []
                if allocation.workload_class is WorkloadClass.GRAPHITI_EMBEDDING
                else [
                    "HOME",
                    "LANG",
                    "LC_ALL",
                    "PATH",
                    "TMPDIR",
                    "XDG_CACHE_HOME",
                    "XDG_CONFIG_HOME",
                    "XDG_DATA_HOME",
                    "XDG_STATE_HOME",
                ]
            )
            if (
                manifest.get("working_directory_inventory") != []
                or manifest.get("working_directory_inventory_digest")
                != digest_canonical([])
                or manifest.get("environment_keys") != expected_environment_keys
            ):
                raise ModelUsageAdmissionError(
                    "Graphiti hermetic workspace proof differs from policy"
                )

            semantic_state_digest = str(record["semantic_state_digest"])
            reused_attempt_identity = connection.execute(
                "SELECT 1 FROM graphiti_internal_requests "
                "WHERE graphiti_attempt_id=? AND "
                "(internal_ordinal=? OR provider_attempt_id=?)",
                (
                    identity.graphiti_attempt_id,
                    identity.internal_ordinal,
                    identity.provider_attempt_id,
                ),
            ).fetchone()
            duplicate = connection.execute(
                "SELECT 1 FROM graphiti_internal_requests "
                "WHERE graphiti_attempt_id=? AND semantic_state_digest=?",
                (identity.graphiti_attempt_id, semantic_state_digest),
            ).fetchone()
            retained_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM graphiti_internal_requests "
                    "WHERE graphiti_attempt_id=?",
                    (identity.graphiti_attempt_id,),
                ).fetchone()[0]
            )
            reason_code = (
                "DUPLICATE_INTERNAL_REQUEST"
                if duplicate is not None
                else "GRAPHITI_ATTEMPT_IDENTITY_REUSE"
                if reused_attempt_identity is not None
                else "CALL_SHAPE_DRIFT"
                if retained_count >= max_distinct_internal_requests
                else None
            )
            if reason_code is not None:
                self._retain_graphiti_refusal(
                    connection,
                    allocation=allocation,
                    identity=record,
                    reason_code=reason_code,
                )
                if reason_code == "CALL_SHAPE_DRIFT":
                    self._append_route_state(
                        connection,
                        route=allocation.route,
                        state="OPEN",
                        reason=reason_code,
                        invocation_id=None,
                        recorded_at=allocation.allocated_at,
                    )
                connection.commit()
                raise ModelUsageAdmissionError(
                    "Graphiti internal request was refused before provider I/O",
                    reason_code=reason_code,
                )

            self._validate_preflight(connection, allocation, policy)
            self._insert_allocation(connection, allocation)
            connection.execute(
                "INSERT INTO graphiti_internal_requests("
                "canonical_digest,invocation_id,envelope_id,graphiti_attempt_id,"
                "internal_ordinal,"
                "semantic_state_digest,provider_attempt_id,call_shape_policy_digest,"
                "record_json) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    record["canonical_digest"],
                    allocation.invocation_id,
                    allocation.envelope_id,
                    identity.graphiti_attempt_id,
                    allocation.leaf_ordinal,
                    semantic_state_digest,
                    record["provider_attempt_id"],
                    record["call_shape_policy_digest"],
                    _json(record),
                ),
            )
            self.link_provider_attempt(
                invocation_id=allocation.invocation_id,
                provider_attempt_id=str(record["provider_attempt_id"]),
                linked_at=allocation.allocated_at,
                connection=connection,
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _insert_allocation(
        self, connection: sqlite3.Connection, allocation: InvocationAllocation
    ) -> None:
        record = allocation.as_record()
        try:
            connection.execute(
                "INSERT INTO model_invocation_allocations("
                "invocation_id,envelope_id,cycle_id,leaf_ordinal,workload_class,"
                "policy_digest,provider,route,model,request_digest,parent_invocation_id,"
                "allocated_at,canonical_digest,record_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    allocation.invocation_id,
                    allocation.envelope_id,
                    allocation.cycle_id,
                    allocation.leaf_ordinal,
                    allocation.workload_class.value,
                    allocation.invocation_policy_digest,
                    allocation.provider,
                    allocation.route,
                    allocation.model,
                    allocation.request_digest,
                    allocation.parent_invocation_id,
                    _utc_text(allocation.allocated_at),
                    allocation.canonical_digest,
                    _json(record),
                ),
            )
        except sqlite3.IntegrityError as exc:
            if "request_digest" in str(exc):
                raise ModelUsageAdmissionError(
                    "duplicate request digest in work envelope"
                ) from exc
            raise

    def _validate_graphiti_identity(
        self,
        allocation: InvocationAllocation,
        identity: GraphitiInternalRequestIdentity,
    ) -> None:
        expected = {
            "invocation_id": allocation.invocation_id,
            "envelope_id": allocation.envelope_id,
            "internal_ordinal": allocation.leaf_ordinal,
            "provider": allocation.provider,
            "model": allocation.model,
            "reasoning": allocation.reasoning,
            "prompt_bytes": allocation.prompt_bytes,
            "prompt_digest": allocation.prompt_digest,
            "response_schema_digest": allocation.output_schema_digest,
            "requested_max_tokens": allocation.max_output_tokens,
            "context_manifest_digest": allocation.context_manifest_digest,
            "parent_invocation_id": allocation.parent_invocation_id,
            "invocation_policy_digest": allocation.invocation_policy_digest,
        }
        if any(getattr(identity, field) != value for field, value in expected.items()):
            raise ModelUsageAdmissionError(
                "Graphiti request identity differs from its #728 allocation"
            )

    def _retain_graphiti_refusal(
        self,
        connection: sqlite3.Connection,
        *,
        allocation: InvocationAllocation,
        identity: Mapping[str, object],
        reason_code: str,
    ) -> None:
        record = {
            "schema_version": "newsroom.graphiti-internal-request-refusal.v1",
            "envelope_id": allocation.envelope_id,
            "attempted_ordinal": allocation.leaf_ordinal,
            "route": allocation.route,
            "semantic_state_digest": identity["semantic_state_digest"],
            "call_shape_policy_digest": identity["call_shape_policy_digest"],
            "reason_code": reason_code,
            "refused_at": _utc_text(allocation.allocated_at),
        }
        refusal_digest = digest_canonical(record)
        connection.execute(
            "INSERT OR IGNORE INTO graphiti_internal_request_refusals("
            "refusal_digest,envelope_id,attempted_ordinal,route,"
            "semantic_state_digest,call_shape_policy_digest,reason_code,refused_at,"
            "record_json) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                refusal_digest,
                allocation.envelope_id,
                allocation.leaf_ordinal,
                allocation.route,
                identity["semantic_state_digest"],
                identity["call_shape_policy_digest"],
                reason_code,
                record["refused_at"],
                _json({**record, "refusal_digest": refusal_digest}),
            ),
        )

    def graphiti_request_records(
        self, *, envelope_id: str
    ) -> dict[str, list[dict[str, object]]]:
        connection = self._connection()
        try:
            requests = [
                _object(row[0])
                for row in connection.execute(
                    "SELECT record_json FROM graphiti_internal_requests "
                    "WHERE envelope_id=? ORDER BY internal_ordinal,canonical_digest",
                    (envelope_id,),
                )
            ]
            refusals = [
                _object(row[0])
                for row in connection.execute(
                    "SELECT record_json FROM graphiti_internal_request_refusals "
                    "WHERE envelope_id=? ORDER BY refused_at,refusal_digest",
                    (envelope_id,),
                )
            ]
        finally:
            connection.close()
        return {"requests": requests, "refusals": refusals}

    def next_graphiti_internal_ordinal(self, *, graphiti_attempt_id: str) -> int:
        """Return the next durable leaf ordinal for one Graphiti attempt."""

        _token(graphiti_attempt_id, field="Graphiti attempt id")
        connection = self._connection()
        try:
            row = connection.execute(
                "SELECT COALESCE(MAX(internal_ordinal),0) "
                "FROM graphiti_internal_requests WHERE graphiti_attempt_id=?",
                (graphiti_attempt_id,),
            ).fetchone()
        finally:
            connection.close()
        return int(row[0]) + 1

    def _validate_preflight(
        self,
        connection: sqlite3.Connection,
        allocation: InvocationAllocation,
        policy: InvocationEfficiencyPolicy,
    ) -> None:
        manifest: dict[str, object] = {}
        if policy.command_semantic_version != "UNSPECIFIED":
            manifest_row = connection.execute(
                "SELECT record_json FROM model_invocation_context_manifests "
                "WHERE context_manifest_digest=?",
                (allocation.context_manifest_digest,),
            ).fetchone()
            if manifest_row is None:
                raise ModelUsageAdmissionError("context manifest is absent")
            manifest = _object(manifest_row[0])
        envelope_row = connection.execute(
            "SELECT record_json FROM model_work_envelopes WHERE envelope_id=?",
            (allocation.envelope_id,),
        ).fetchone()
        if envelope_row is None:
            raise ModelUsageAdmissionError("work envelope is absent")
        envelope = _object(envelope_row[0])
        if (
            allocation.workload_class != policy.workload_class
            or allocation.provider != policy.provider
            or allocation.route != policy.route
            or allocation.model != policy.model
            or allocation.reasoning != policy.reasoning
            or allocation.prompt_contract_version != policy.prompt_contract_version
            or allocation.output_schema_digest != policy.output_schema_digest
            or allocation.max_output_tokens > policy.max_output_tokens
            or allocation.one_turn != policy.one_turn
            or allocation.exact_input != policy.exact_input
            or allocation.skills_enabled != policy.skills_enabled
            or allocation.tools_enabled != policy.tools_enabled
            or allocation.mcp_enabled != policy.mcp_enabled
            or allocation.prior_message_count != policy.prior_message_count
        ):
            raise ModelUsageAdmissionError("allocation differs from invocation policy")
        if policy.command_semantic_version != "UNSPECIFIED" and (
            manifest.get("schema_version") != policy.context_manifest_schema_version
            or manifest.get("command_semantic_version")
            != policy.command_semantic_version
            or _record_string_tuple(manifest, "command_flags")
            != policy.command_flags
            or _record_string_tuple(manifest, "disabled_capabilities")
            != policy.disabled_capabilities
            or manifest.get("implementation_revision")
            != policy.implementation_revision
            or manifest.get("implementation_worktree_clean") is not True
        ):
            raise ModelUsageAdmissionError(
                "context manifest command contract differs from invocation policy"
            )
        if (
            policy.command_semantic_version != "UNSPECIFIED"
            and not allocation.workload_class.value.startswith("GRAPHITI_")
            and (
                manifest.get("evidence_package_digest")
                != envelope.get("evidence_package_digest")
            )
        ):
            raise ModelUsageAdmissionError(
                "context manifest Evidence Package differs from work envelope"
            )
        expected_request_digest = digest_canonical(
            {
                "provider": manifest.get("provider"),
                "route": manifest.get("route"),
                "model": manifest.get("model"),
                "reasoning": manifest.get("reasoning"),
                "command_semantic_version": manifest.get(
                    "command_semantic_version"
                ),
                "command_flags": manifest.get("command_flags"),
                "implementation_revision": manifest.get(
                    "implementation_revision"
                ),
                "system_digest": manifest.get("system_digest"),
                "prompt_digest": manifest.get("prompt_digest"),
                "output_schema_digest": manifest.get("output_schema_digest"),
            }
        )
        if policy.command_semantic_version != "UNSPECIFIED" and (
            manifest.get("provider") != allocation.provider
            or manifest.get("route") != allocation.route
            or manifest.get("model") != allocation.model
            or manifest.get("reasoning") != allocation.reasoning
            or manifest.get("prompt_contract_version")
            != allocation.prompt_contract_version
            or manifest.get("prompt_bytes") != allocation.prompt_bytes
            or manifest.get("prompt_digest") != allocation.prompt_digest
            or manifest.get("output_schema_digest")
            != allocation.output_schema_digest
            or manifest.get("context_identity") != allocation.context_identity
            or manifest.get("config_identity") != allocation.config_identity
            or manifest.get("one_turn") != allocation.one_turn
            or manifest.get("exact_input") != allocation.exact_input
            or manifest.get("skills_enabled") != allocation.skills_enabled
            or manifest.get("tools_enabled") != allocation.tools_enabled
            or manifest.get("mcp_enabled") != allocation.mcp_enabled
            or manifest.get("prior_message_count")
            != allocation.prior_message_count
            or manifest.get("request_digest") != expected_request_digest
            or manifest.get("request_digest") != allocation.request_digest
        ):
            raise ModelUsageAdmissionError(
                "context manifest invocation identity differs from allocation"
            )
        if policy.calibration_only and envelope.get("candidate_id") not in (
            policy.allowed_candidate_ids
        ):
            raise ModelUsageAdmissionError(
                "candidate is outside the bounded calibration policy"
            )
        if allocation.prompt_bytes > policy.max_prompt_bytes:
            raise ModelUsageAdmissionError(
                "prompt bytes exceed qualified policy",
                reason_code="EXACT_INPUT_EXCEEDS_QUALIFIED_BOUND",
            )
        if allocation.context_identity not in policy.allowed_context_identities:
            raise ModelUsageAdmissionError(
                "context identity is outside qualified policy"
            )
        if allocation.config_identity not in policy.allowed_config_identities:
            raise ModelUsageAdmissionError(
                "config identity is outside qualified policy"
            )
        if _canonical_circuit_route(allocation.route) in _usage_blocking_routes(
            connection
        ):
            raise ModelUsageAdmissionError(
                "affected route has unresolved usage or a policy breach"
            )
        if self._route_state(connection, allocation.route)["state"] == "OPEN":
            raise ModelUsageAdmissionError("affected route circuit is open")
        duplicate = connection.execute(
            "SELECT 1 FROM model_invocation_allocations "
            "WHERE envelope_id=? AND request_digest=?",
            (allocation.envelope_id, allocation.request_digest),
        ).fetchone()
        if duplicate is not None:
            raise ModelUsageAdmissionError("duplicate request digest in work envelope")
        if allocation.parent_invocation_id is not None:
            parent = connection.execute(
                "SELECT envelope_id FROM model_invocation_allocations WHERE invocation_id=?",
                (allocation.parent_invocation_id,),
            ).fetchone()
            if parent is None or str(parent[0]) != allocation.envelope_id:
                raise ModelUsageAdmissionError(
                    "parent invocation is outside work envelope"
                )

    def observe_transport(
        self,
        *,
        invocation_id: str,
        observed_at: datetime,
        state: str,
        evidence_digest: str,
    ) -> None:
        record = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "invocation_id": invocation_id,
            "observed_at": _utc_text(observed_at),
            "state": _token(state, field="transport state"),
            "evidence_digest": _token(
                evidence_digest, field="transport evidence digest"
            ),
        }
        observation_digest = digest_canonical(record)
        connection = self._connection()
        try:
            connection.execute(
                "INSERT OR IGNORE INTO model_transport_observations("
                "observation_digest,invocation_id,observed_at,state,evidence_digest,record_json) "
                "VALUES(?,?,?,?,?,?)",
                (
                    observation_digest,
                    invocation_id,
                    record["observed_at"],
                    state,
                    evidence_digest,
                    _json({**record, "observation_digest": observation_digest}),
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def has_committed_provider_dispatch(self, *, cycle_id: str) -> bool:
        """Return event dispatch truth from a committed provider-leaf marker."""

        cycle_id = _token(cycle_id, field="cycle id")
        connection = self._connection()
        try:
            row = connection.execute(
                "SELECT EXISTS("
                "SELECT 1 FROM model_invocation_allocations AS allocation "
                "JOIN model_transport_observations AS observation "
                "ON observation.invocation_id=allocation.invocation_id "
                "WHERE allocation.cycle_id=? "
                "AND allocation.workload_class IN (?,?,?) "
                "AND observation.state='DISPATCH_STARTED')",
                (
                    cycle_id,
                    WorkloadClass.GRAPHITI_CHAT_PRIMARY.value,
                    WorkloadClass.GRAPHITI_CHAT_FALLBACK.value,
                    WorkloadClass.GRAPHITI_EMBEDDING.value,
                ),
            ).fetchone()
            return bool(row and row[0])
        finally:
            connection.close()

    def link_provider_attempt(
        self,
        *,
        invocation_id: str,
        provider_attempt_id: str,
        linked_at: datetime,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        record = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "invocation_id": _token(invocation_id, field="invocation id"),
            "provider_attempt_id": _token(
                provider_attempt_id, field="provider attempt id"
            ),
            "linked_at": _utc_text(linked_at),
        }
        digest = digest_canonical(record)
        self._insert_exact(
            table="model_invocation_provider_attempt_links",
            identity_column="invocation_id",
            identity=invocation_id,
            record={**record, "link_digest": digest},
            sql="INSERT INTO model_invocation_provider_attempt_links("
            "link_digest,invocation_id,provider_attempt_id,linked_at,record_json) "
            "VALUES(?,?,?,?,?)",
            values=(
                digest,
                invocation_id,
                provider_attempt_id,
                record["linked_at"],
                _json({**record, "link_digest": digest}),
            ),
            connection=connection,
        )

    def complete(
        self,
        terminal: InvocationTerminal,
        *,
        provider_telemetry: Mapping[str, object] | None = None,
    ) -> InvocationTerminal:
        terminal._validate_shape()
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT a.route,a.workload_class,p.record_json,"
                "json_extract(a.record_json,'$.context_manifest_digest'),"
                "json_extract(a.record_json,'$.max_output_tokens') "
                "FROM model_invocation_allocations a "
                "JOIN model_invocation_policies p ON p.canonical_digest=a.policy_digest "
                "WHERE a.invocation_id=?",
                (terminal.invocation_id,),
            ).fetchone()
            if row is None:
                raise ModelUsageIntegrityError("invocation allocation is absent")
            route, workload = str(row[0]), WorkloadClass(str(row[1]))
            context_manifest_digest = str(row[3])
            requested_max_output_tokens = int(row[4])
            policy = _policy_from_record(_object(row[2]))
            retained = terminal
            if provider_telemetry is not None:
                telemetry_digest = _retain_provider_telemetry(
                    connection,
                    invocation_id=terminal.invocation_id,
                    provider_telemetry=provider_telemetry,
                )
                if retained.provider_telemetry_digest != telemetry_digest:
                    retained = replace(
                        retained,
                        usage_status=UsageStatus.INVALID,
                        failure_class="TELEMETRY_DIGEST_MISMATCH",
                        terminal_digest="",
                    )
            invalid_report = _invalid_reported_components(retained)
            if invalid_report is not None:
                retained = replace(
                    retained,
                    usage_status=UsageStatus.INVALID,
                    failure_class=invalid_report,
                    terminal_digest="",
                )
            policy_breach = self._validate_terminal(
                retained,
                workload,
                policy,
                requested_max_output_tokens=requested_max_output_tokens,
            )
            if policy_breach is not None:
                retained = replace(
                    retained,
                    policy_breach=policy_breach,
                    terminal_digest="",
                )
            record_without_digest = retained.as_record()
            record_without_digest["terminal_digest"] = ""
            retained = replace(
                retained,
                terminal_digest=digest_canonical(record_without_digest),
            )
            record = retained.as_record()
            try:
                connection.execute(
                    "INSERT INTO model_invocation_terminals("
                    "terminal_digest,invocation_id,usage_status,outcome,failure_class,"
                    "completed_at,record_json) VALUES(?,?,?,?,?,?,?)",
                    (
                        retained.terminal_digest,
                        retained.invocation_id,
                        retained.usage_status.value,
                        retained.outcome,
                        retained.failure_class,
                        _utc_text(retained.completed_at),
                        _json(record),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                prior = connection.execute(
                    "SELECT record_json FROM model_invocation_terminals WHERE invocation_id=?",
                    (retained.invocation_id,),
                ).fetchone()
                if prior is None or _object(prior[0]) != record:
                    raise ModelUsageIntegrityError(
                        "conflicting invocation terminal replay"
                    ) from exc
            context_manifest = connection.execute(
                "SELECT 1 FROM model_invocation_context_manifests "
                "WHERE context_manifest_digest=?",
                (context_manifest_digest,),
            ).fetchone()
            if context_manifest is not None:
                context_observation = {
                    "schema_version": MODEL_USAGE_SCHEMA_VERSION,
                    "invocation_id": retained.invocation_id,
                    "context_manifest_digest": context_manifest_digest,
                    "usage_status": retained.usage_status.value,
                    "provider_context_tokens": retained.components.context_tokens,
                    "observed_at": _utc_text(retained.observed_at),
                }
                observation_digest = digest_canonical(context_observation)
                observation_record = {
                    **context_observation,
                    "observation_digest": observation_digest,
                }
                connection.execute(
                    "INSERT OR IGNORE INTO model_invocation_context_observations("
                    "observation_digest,invocation_id,context_manifest_digest,"
                    "provider_context_tokens,record_json) VALUES(?,?,?,?,?)",
                    (
                        observation_digest,
                        retained.invocation_id,
                        context_manifest_digest,
                        retained.components.context_tokens,
                        _json(observation_record),
                    ),
                )
            if retained.usage_status in {
                UsageStatus.UNREPORTED,
                UsageStatus.AMBIGUOUS,
                UsageStatus.INVALID,
            }:
                self._append_route_state(
                    connection,
                    route=route,
                    state="OPEN",
                    reason=(retained.failure_class or retained.usage_status.value),
                    invocation_id=retained.invocation_id,
                    recorded_at=retained.observed_at,
                )
            elif retained.policy_breach:
                self._append_route_state(
                    connection,
                    route=route,
                    state="OPEN",
                    reason=retained.policy_breach,
                    invocation_id=retained.invocation_id,
                    recorded_at=retained.observed_at,
                )
            connection.commit()
            return retained
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _validate_terminal(
        self,
        terminal: InvocationTerminal,
        workload: WorkloadClass,
        policy: InvocationEfficiencyPolicy,
        *,
        requested_max_output_tokens: int,
    ) -> str | None:
        components = terminal.components
        total = components.total_tokens
        if terminal.pre_dispatch_zero_proved:
            if terminal.dispatch_at is not None or total != 0:
                raise ModelUsageIntegrityError(
                    "pre-dispatch zero requires no dispatch and exact zero"
                )
        elif terminal.dispatch_at is None:
            raise ModelUsageIntegrityError(
                "possible provider usage lacks a dispatch observation"
            )
        if terminal.usage_status is UsageStatus.REPORTED:
            if _invalid_reported_components(terminal) is not None:
                raise ModelUsageIntegrityError(
                    "invalid reported usage was not classified"
                )
        elif terminal.usage_status is UsageStatus.ESTIMATED:
            hard_ceiling = policy.hard_estimate_ceiling_tokens
            if (
                hard_ceiling is None
                or total != hard_ceiling
                or components.provenance != "BOUNDED_ESTIMATE"
                or terminal.estimate_policy_digest != policy.canonical_digest
                or not terminal.estimate_calculation
            ):
                raise ModelUsageIntegrityError(
                    "bounded estimate evidence is incomplete"
                )
        elif (
            terminal.usage_status in {UsageStatus.UNREPORTED, UsageStatus.AMBIGUOUS}
            and total is not None
        ):
            raise ModelUsageIntegrityError("unresolved usage must not invent a total")
        if workload in {
            WorkloadClass.GRAPHITI_EMBEDDING,
            WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING,
        }:
            if not terminal.od_011_reference:
                raise ModelUsageIntegrityError("embedding usage lacks OD-011 linkage")
        elif not terminal.subscription_cli_chat_not_cash_debited:
            raise ModelUsageIntegrityError(
                "subscription CLI chat cash-debit confirmation is absent"
            )
        if terminal.usage_status is not UsageStatus.REPORTED:
            return terminal.policy_breach
        if total is not None and total > policy.max_total_tokens:
            return "MAX_TOTAL_TOKENS_EXCEEDED"
        context = components.context_tokens
        if context is not None and context > policy.max_context_tokens:
            return "MAX_CONTEXT_TOKENS_EXCEEDED"
        output = components.output_tokens
        if output is not None and output > requested_max_output_tokens:
            return "REQUESTED_MAX_OUTPUT_TOKENS_EXCEEDED"
        if output is not None and output > policy.max_output_tokens:
            return "MAX_OUTPUT_TOKENS_EXCEEDED"
        return terminal.policy_breach

    def record_work_outcome(
        self,
        *,
        envelope_id: str,
        outcome: str,
        outcome_record_id: str,
        payload_digest: str | None,
        terminal_at: datetime,
        cycle_outcome: str | None = None,
        route_circuit_state: str | None = None,
        route_circuit_reason: str | None = None,
        retained_proposal_count: int | None = None,
        accepted_provider_attempt_id: str | None = None,
        stable_reason_codes: tuple[str, ...] = (),
        connection: sqlite3.Connection | None = None,
    ) -> None:
        record = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "envelope_id": envelope_id,
            "outcome": _token(outcome, field="work outcome"),
            "outcome_record_id": _token(outcome_record_id, field="outcome record id"),
            "payload_digest": payload_digest,
            "terminal_at": _utc_text(terminal_at),
            "cycle_outcome": cycle_outcome,
            "route_circuit_state": route_circuit_state,
            "route_circuit_reason": route_circuit_reason,
            "retained_proposal_count": retained_proposal_count,
            "accepted_provider_attempt_id": accepted_provider_attempt_id,
            "stable_reason_codes": list(stable_reason_codes),
        }
        if retained_proposal_count is not None:
            _non_negative(retained_proposal_count, field="retained proposal count")
        digest = digest_canonical(record)
        owns_connection = connection is None
        current = self._connection() if connection is None else connection
        try:
            if owns_connection:
                current.execute("BEGIN IMMEDIATE")
            envelope_row = current.execute(
                "SELECT record_json FROM model_work_envelopes WHERE envelope_id=?",
                (envelope_id,),
            ).fetchone()
            graphiti_attempt_id = (
                None
                if envelope_row is None
                else _object(envelope_row[0]).get("graphiti_attempt_id")
            )
            unresolved_graphiti_leaf = current.execute(
                "SELECT 1 FROM graphiti_internal_requests g "
                "LEFT JOIN model_invocation_terminals t "
                "ON t.invocation_id=g.invocation_id "
                "WHERE "
                + (
                    "g.graphiti_attempt_id=? "
                    if isinstance(graphiti_attempt_id, str)
                    else "g.envelope_id=? "
                )
                + "AND t.invocation_id IS NULL LIMIT 1",
                (
                    graphiti_attempt_id
                    if isinstance(graphiti_attempt_id, str)
                    else envelope_id,
                ),
            ).fetchone()
            if unresolved_graphiti_leaf is not None:
                raise ModelUsageIntegrityError(
                    "Graphiti work outcome lacks a terminal receipt for every leaf"
                )
            self._insert_exact(
                table="model_work_outcomes",
                identity_column="envelope_id",
                identity=envelope_id,
                record={**record, "outcome_digest": digest},
                sql="INSERT INTO model_work_outcomes("
                "outcome_digest,envelope_id,outcome,terminal_at,record_json) "
                "VALUES(?,?,?,?,?)",
                values=(
                    digest,
                    envelope_id,
                    outcome,
                    record["terminal_at"],
                    _json({**record, "outcome_digest": digest}),
                ),
                connection=current,
            )
            if owns_connection:
                current.commit()
        except Exception:
            if owns_connection and current.in_transaction:
                current.rollback()
            raise
        finally:
            if owns_connection:
                current.close()

    def record_cycle_outcome(
        self,
        *,
        cycle_id: str,
        outcome_class: str,
        terminal_at: datetime,
        writer_unproductive_streak_before: int,
        writer_unproductive_streak_after: int,
        writer_circuit_state: str,
        writer_circuit_open_reason: str,
    ) -> None:
        record = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "cycle_id": _token(cycle_id, field="cycle id"),
            "outcome_class": _token(outcome_class, field="cycle outcome class"),
            "terminal_at": _utc_text(terminal_at),
            "writer_unproductive_streak_before": _non_negative(
                writer_unproductive_streak_before,
                field="writer unproductive streak before",
            ),
            "writer_unproductive_streak_after": _non_negative(
                writer_unproductive_streak_after,
                field="writer unproductive streak after",
            ),
            "writer_circuit_state": _token(
                writer_circuit_state, field="writer circuit state"
            ),
            "writer_circuit_open_reason": writer_circuit_open_reason,
        }
        digest = digest_canonical(record)
        self._insert_exact(
            table="model_usage_cycle_outcomes",
            identity_column="cycle_id",
            identity=cycle_id,
            record={**record, "cycle_digest": digest},
            sql="INSERT INTO model_usage_cycle_outcomes("
            "cycle_digest,cycle_id,outcome_class,terminal_at,record_json) "
            "VALUES(?,?,?,?,?)",
            values=(
                digest,
                cycle_id,
                outcome_class,
                record["terminal_at"],
                _json({**record, "cycle_digest": digest}),
            ),
        )

    def recover_unresolved(self, *, observed_at: datetime) -> int:
        connection = self._connection()
        recovered = 0
        try:
            rows = connection.execute(
                "SELECT a.invocation_id,a.allocated_at,a.workload_class,"
                "(SELECT MIN(o.observed_at) FROM model_transport_observations o "
                "WHERE o.invocation_id=a.invocation_id "
                "AND o.state='DISPATCH_STARTED'),"
                "json_extract(a.record_json,'$.recovery_deadline_at'),"
                "EXISTS(SELECT 1 FROM graphiti_internal_requests g "
                "WHERE g.invocation_id=a.invocation_id) "
                "FROM model_invocation_allocations a "
                "LEFT JOIN model_invocation_terminals t ON t.invocation_id=a.invocation_id "
                "WHERE t.invocation_id IS NULL "
                "AND json_extract(a.record_json,'$.recovery_deadline_at') IS NOT NULL "
                "AND json_extract(a.record_json,'$.recovery_deadline_at')<=? "
                "ORDER BY a.invocation_id",
                (_utc_text(observed_at),),
            ).fetchall()
        finally:
            connection.close()
        for row in rows:
            workload = WorkloadClass(str(row[2]))
            embedding = workload in {
                WorkloadClass.GRAPHITI_EMBEDDING,
                WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING,
            }
            graphiti_pre_dispatch = bool(row[5]) and row[3] is None
            dispatch_at = (
                None
                if graphiti_pre_dispatch
                else _instant(str(row[3] or row[1]))
            )
            recovery_terminal_at = _instant(str(row[4]))
            terminal = InvocationTerminal.create(
                invocation_id=str(row[0]),
                outcome=(
                    "RECOVERED_PRE_DISPATCH"
                    if graphiti_pre_dispatch
                    else "RECOVERED_UNRESOLVED"
                ),
                failure_class=(
                    "PROCESS_LOST_BEFORE_DISPATCH"
                    if graphiti_pre_dispatch
                    else "PROCESS_LOST_AFTER_ALLOCATION"
                ),
                usage_status=(
                    UsageStatus.REPORTED
                    if graphiti_pre_dispatch
                    else UsageStatus.AMBIGUOUS
                ),
                components=(
                    UsageComponents(total_tokens=0, provenance="CLI_DERIVED")
                    if graphiti_pre_dispatch
                    else UsageComponents(provenance="UNAVAILABLE")
                ),
                dispatch_at=dispatch_at,
                completed_at=recovery_terminal_at,
                observed_at=recovery_terminal_at,
                od_011_reference=(
                    "OD-011:EVALUATION_GRAPHITI_EMBEDDING" if embedding else None
                ),
                subscription_cli_chat_not_cash_debited=not embedding,
                pre_dispatch_zero_proved=graphiti_pre_dispatch,
            )
            try:
                self.complete(terminal)
            except ModelUsageIntegrityError as exc:
                if not self._terminal_exists(str(row[0])):
                    raise
                if "conflicting invocation terminal replay" not in str(exc):
                    raise
                continue
            recovered += 1
        connection = self._connection()
        try:
            envelope_rows = connection.execute(
                "SELECT e.envelope_id,MAX(json_extract("
                "a_deadline.record_json,'$.recovery_deadline_at')) "
                "FROM model_work_envelopes e "
                "JOIN model_invocation_allocations a_deadline "
                "ON a_deadline.envelope_id=e.envelope_id "
                "WHERE NOT EXISTS (SELECT 1 FROM model_work_outcomes w "
                "WHERE w.envelope_id=e.envelope_id) "
                "AND EXISTS (SELECT 1 FROM model_invocation_allocations a "
                "WHERE a.envelope_id=e.envelope_id) "
                "AND NOT EXISTS (SELECT 1 FROM model_invocation_allocations a "
                "LEFT JOIN model_invocation_terminals t "
                "ON t.invocation_id=a.invocation_id "
                "WHERE a.envelope_id=e.envelope_id AND t.invocation_id IS NULL) "
                "AND NOT EXISTS (SELECT 1 FROM model_invocation_allocations a "
                "WHERE a.envelope_id=e.envelope_id AND ("
                "json_extract(a.record_json,'$.recovery_deadline_at') IS NULL OR "
                "json_extract(a.record_json,'$.recovery_deadline_at')>?)) "
                "GROUP BY e.envelope_id ORDER BY e.envelope_id",
                (_utc_text(observed_at),),
            ).fetchall()
        finally:
            connection.close()
        for envelope_id, raw_terminal_at in envelope_rows:
            recovery_terminal_at = _instant(str(raw_terminal_at))
            try:
                self.record_work_outcome(
                    envelope_id=str(envelope_id),
                    outcome="AMBIGUOUS_PROCESS_LOST_BEFORE_WORK_OUTCOME",
                    outcome_record_id=digest_canonical(
                        {
                            "envelope_id": str(envelope_id),
                            "recovered_at": _utc_text(recovery_terminal_at),
                        }
                    ),
                    payload_digest=None,
                    terminal_at=recovery_terminal_at,
                    stable_reason_codes=("PROCESS_LOST_BEFORE_WORK_OUTCOME",),
                )
            except ModelUsageIntegrityError as exc:
                if not self._work_outcome_exists(str(envelope_id)):
                    raise
                if "conflicting model_work_outcomes replay" not in str(exc):
                    raise
                continue
            recovered += 1
        return recovered

    def _terminal_exists(self, invocation_id: str) -> bool:
        connection = self._connection()
        try:
            return (
                connection.execute(
                    "SELECT 1 FROM model_invocation_terminals WHERE invocation_id=?",
                    (invocation_id,),
                ).fetchone()
                is not None
            )
        finally:
            connection.close()

    def _work_outcome_exists(self, envelope_id: str) -> bool:
        connection = self._connection()
        try:
            return (
                connection.execute(
                    "SELECT 1 FROM model_work_outcomes WHERE envelope_id=?",
                    (envelope_id,),
                ).fetchone()
                is not None
            )
        finally:
            connection.close()

    def disposition_unreported_subscription_usage(
        self,
        *,
        invocation_id: str,
        expected_terminal_digest: str,
        expected_allocation_digest: str,
        approved_by: str,
        approval_reference: str,
        approved_at: datetime,
        approved_plan_digest: str,
        authority_digest: str,
        observed_at: datetime,
    ) -> dict[str, object]:
        """Retain one authorised upper-bound estimate without rewriting history.

        This deliberately narrow path exists for a dispatched Cursor subscription
        CLI Graphiti primary leaf whose exact provider telemetry is permanently
        absent.  It preserves the unresolved terminal and provider dispatch while
        using the retained qualified policy maximum as the conservative total.
        """

        invocation_id = _token(invocation_id, field="invocation id")
        expected_terminal_digest = _token(
            expected_terminal_digest,
            field="expected terminal digest",
        )
        expected_allocation_digest = _token(
            expected_allocation_digest,
            field="expected allocation digest",
        )
        approved_by = _token(approved_by, field="approved by")
        approval_reference = _token(
            approval_reference,
            field="approval reference",
        )
        authority_digest = _token(authority_digest, field="authority digest")
        approved_plan_digest = _token(
            approved_plan_digest,
            field="approved plan digest",
        )
        try:
            preview = sqlite3.connect(self.path)
            try:
                approved_contract = effective_issue_790_plan_contract(
                    approved_plan_digest,
                    connection=preview,
                )
            finally:
                preview.close()
        except KeyError as exc:
            raise ModelUsageIntegrityError(
                "conservative disposition approved plan differs"
            ) from exc
        if invocation_id != approved_contract.invocation_id:
            raise ModelUsageIntegrityError(
                "conservative disposition approved invocation differs"
            )
        if expected_terminal_digest != approved_contract.terminal_digest:
            raise ModelUsageIntegrityError(
                "conservative disposition approved terminal differs"
            )
        if expected_allocation_digest != approved_contract.allocation_digest:
            raise ModelUsageIntegrityError(
                "conservative disposition approved allocation differs"
            )
        approved_at_text = _utc_text(approved_at)
        observed_at_text = _utc_text(observed_at)
        if (
            approved_by != approved_contract.approved_by
            or approval_reference != approved_contract.approval_reference
            or approved_at_text != approved_contract.approved_at
        ):
            raise ModelUsageIntegrityError(
                "conservative disposition approval authority differs"
            )
        if observed_at < approved_at:
            raise ModelUsageIntegrityError(
                "conservative disposition observation precedes approval"
            )

        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            prior_row = connection.execute(
                "SELECT authority_digest,record_json "
                "FROM model_usage_conservative_dispositions "
                "WHERE invocation_id=?",
                (invocation_id,),
            ).fetchone()
            if prior_row is not None:
                prior = _object(prior_row[1])
                if str(prior_row[0]) != authority_digest:
                    raise ModelUsageIntegrityError(
                        "conservative disposition authority differs"
                    )
                if (
                    prior.get("terminal_digest") != expected_terminal_digest
                    or prior.get("allocation_digest")
                    != expected_allocation_digest
                    or prior.get("approved_by") != approved_by
                    or prior.get("approval_reference") != approval_reference
                    or prior.get("approved_at") != approved_at_text
                    or prior.get("approved_plan_digest")
                    != approved_plan_digest
                ):
                    raise ModelUsageIntegrityError(
                        "conflicting conservative disposition replay"
                    )
                connection.rollback()
                return prior

            retained_row = connection.execute(
                "SELECT t.terminal_digest,t.record_json,a.canonical_digest,"
                "a.record_json,p.canonical_digest,p.record_json "
                "FROM model_invocation_terminals t "
                "JOIN model_invocation_allocations a "
                "ON a.invocation_id=t.invocation_id "
                "JOIN model_invocation_policies p "
                "ON p.canonical_digest=a.policy_digest "
                "WHERE t.invocation_id=?",
                (invocation_id,),
            ).fetchone()
            if retained_row is None:
                raise ModelUsageIntegrityError(
                    "conservative disposition terminal is absent"
                )
            terminal_digest = str(retained_row[0])
            terminal = _object(retained_row[1])
            allocation_digest = str(retained_row[2])
            allocation = _object(retained_row[3])
            policy_digest = str(retained_row[4])
            policy = _policy_from_record(_object(retained_row[5]))

            if terminal_digest != expected_terminal_digest:
                raise ModelUsageIntegrityError(
                    "conservative disposition terminal identity differs"
                )
            if allocation_digest != expected_allocation_digest:
                raise ModelUsageIntegrityError(
                    "conservative disposition allocation identity differs"
                )

            authority = {
                "schema_version": (
                    CONSERVATIVE_DISPOSITION_AUTHORITY_SCHEMA_VERSION
                ),
                "approved_plan_digest": approved_plan_digest,
                "approved_by": approved_by,
                "approval_reference": approval_reference,
                "approved_at": approved_at_text,
                "invocation_id": invocation_id,
                "terminal_digest": terminal_digest,
                "allocation_digest": allocation_digest,
                "scope": approved_contract.scope,
            }
            if digest_canonical(authority) != authority_digest:
                raise ModelUsageIntegrityError(
                    "conservative disposition authority differs"
                )
            terminal_observed_at = _instant(str(terminal["observed_at"]))
            if approved_at < terminal_observed_at:
                raise ModelUsageIntegrityError(
                    "conservative disposition approval precedes terminal"
                )
            if (
                allocation.get("workload_class")
                != WorkloadClass.GRAPHITI_CHAT_PRIMARY.value
                or allocation.get("provider") != "cursor-agent-cli"
                or allocation.get("route")
                != WorkloadClass.GRAPHITI_CHAT_PRIMARY.value
            ):
                raise ModelUsageIntegrityError(
                    "conservative disposition target is outside the approved route"
                )
            if (
                terminal.get("usage_status") != UsageStatus.UNREPORTED.value
                or terminal.get("outcome") != approved_contract.terminal_outcome
                or terminal.get("failure_class")
                != "MISSING_PROVIDER_TELEMETRY"
                or terminal.get("subscription_cli_chat_not_cash_debited")
                is not True
                or terminal.get("policy_breach") is not None
            ):
                raise ModelUsageIntegrityError(
                    "conservative disposition terminal is ineligible"
                )
            if (
                terminal.get("provider_telemetry_digest") is not None
                or terminal.get("raw_telemetry_pointer") is not None
                or connection.execute(
                    "SELECT 1 FROM model_provider_telemetry "
                    "WHERE invocation_id=? LIMIT 1",
                    (invocation_id,),
                ).fetchone()
                is not None
                or connection.execute(
                    "SELECT 1 FROM model_usage_reconciliations "
                    "WHERE invocation_id=? LIMIT 1",
                    (invocation_id,),
                ).fetchone()
                is not None
            ):
                raise ModelUsageIntegrityError(
                    "conservative disposition exact telemetry already exists"
                )
            if connection.execute(
                "SELECT 1 FROM model_transport_observations "
                "WHERE invocation_id=? AND state='DISPATCH_STARTED' LIMIT 1",
                (invocation_id,),
            ).fetchone() is None:
                raise ModelUsageIntegrityError(
                    "conservative disposition committed transport dispatch is absent"
                )
            if not policy.qualified or policy.canonical_digest != policy_digest:
                raise ModelUsageIntegrityError(
                    "conservative disposition policy is not qualified"
                )

            components = UsageComponents(
                total_tokens=policy.max_total_tokens,
                provenance="BOUNDED_ESTIMATE",
            )
            record_without_digest: dict[str, object] = {
                "schema_version": CONSERVATIVE_DISPOSITION_SCHEMA_VERSION,
                "invocation_id": invocation_id,
                "terminal_digest": terminal_digest,
                "allocation_digest": allocation_digest,
                "policy_digest": policy_digest,
                "approved_plan_digest": approved_plan_digest,
                "usage_status": UsageStatus.ESTIMATED.value,
                "components": components.as_record(),
                "estimate_policy_digest": policy_digest,
                "estimate_calculation": (
                    "QUALIFIED_POLICY_MAX_TOTAL_TOKENS_CONSERVATIVE_UPPER_BOUND"
                ),
                "exact_usage_remains_unknown": True,
                "provider_dispatch_preserved": True,
                "unknown_spend_released": False,
                "authority_digest": authority_digest,
                "approved_by": approved_by,
                "approval_reference": approval_reference,
                "approved_at": approved_at_text,
                "observed_at": observed_at_text,
            }
            disposition_digest = digest_canonical(record_without_digest)
            record = {
                **record_without_digest,
                "disposition_digest": disposition_digest,
            }
            connection.execute(
                "INSERT INTO model_usage_conservative_dispositions("
                "disposition_digest,invocation_id,terminal_digest,"
                "allocation_digest,policy_digest,approved_plan_digest,"
                "authority_digest,approved_by,approval_reference,approved_at,"
                "observed_at,usage_status,record_json"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    disposition_digest,
                    invocation_id,
                    terminal_digest,
                    allocation_digest,
                    policy_digest,
                    approved_plan_digest,
                    authority_digest,
                    approved_by,
                    approval_reference,
                    approved_at_text,
                    observed_at_text,
                    UsageStatus.ESTIMATED.value,
                    _json(record),
                ),
            )
            connection.commit()
            return record
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def reconcile(
        self,
        *,
        invocation_id: str,
        components: UsageComponents,
        provider_telemetry: Mapping[str, object],
        observed_at: datetime,
        raw_telemetry_pointer: str,
    ) -> None:
        connection = self._connection()
        try:
            terminal_row = connection.execute(
                "SELECT t.record_json,a.route,p.record_json "
                "FROM model_invocation_terminals t "
                "JOIN model_invocation_allocations a "
                "ON a.invocation_id=t.invocation_id "
                "JOIN model_invocation_policies p "
                "ON p.canonical_digest=a.policy_digest "
                "WHERE t.invocation_id=?",
                (invocation_id,),
            ).fetchone()
            if terminal_row is None:
                raise ModelUsageIntegrityError("terminal usage state is absent")
            terminal = _object(terminal_row[0])
            route = str(terminal_row[1])
            policy = _policy_from_record(_object(terminal_row[2]))
            if terminal.get("usage_status") not in {
                "UNREPORTED",
                "AMBIGUOUS",
                "INVALID",
            }:
                raise ModelUsageIntegrityError("terminal usage is already exact")
            if (
                components.total_tokens is None
                or components.provenance != "PROVIDER_REPORTED"
            ):
                raise ModelUsageIntegrityError(
                    "reconciliation usage is not provider-reported"
                )
            known = [
                value
                for value in (
                    components.input_tokens,
                    components.output_tokens,
                    components.cached_read_tokens,
                    components.cached_write_tokens,
                    components.reasoning_tokens,
                )
                if value is not None
            ]
            direct = sum(
                int(value)
                for value in (components.input_tokens, components.output_tokens)
                if value is not None
            )
            expanded = sum(int(value) for value in known)
            if known and components.total_tokens not in {direct, expanded}:
                raise ModelUsageIntegrityError(
                    "reconciled component total is impossible"
                )
            provider_telemetry_digest = _retain_provider_telemetry(
                connection,
                invocation_id=invocation_id,
                provider_telemetry=provider_telemetry,
            )
            record = {
                "schema_version": MODEL_USAGE_SCHEMA_VERSION,
                "invocation_id": invocation_id,
                "usage_status": UsageStatus.REPORTED.value,
                "components": components.as_record(),
                "provider_telemetry_digest": provider_telemetry_digest,
                "raw_telemetry_pointer": raw_telemetry_pointer,
                "observed_at": _utc_text(observed_at),
                "policy_breach": (
                    "MAX_TOTAL_TOKENS_EXCEEDED"
                    if components.total_tokens > policy.max_total_tokens
                    else "MAX_CONTEXT_TOKENS_EXCEEDED"
                    if components.context_tokens is not None
                    and components.context_tokens > policy.max_context_tokens
                    else "MAX_OUTPUT_TOKENS_EXCEEDED"
                    if components.output_tokens is not None
                    and components.output_tokens > policy.max_output_tokens
                    else None
                ),
            }
            digest = digest_canonical(record)
            connection.execute(
                "INSERT OR IGNORE INTO model_usage_reconciliations("
                "reconciliation_digest,invocation_id,observed_at,record_json) "
                "VALUES(?,?,?,?)",
                (
                    digest,
                    invocation_id,
                    record["observed_at"],
                    _json({**record, "reconciliation_digest": digest}),
                ),
            )
            canonical_route = _canonical_circuit_route(route)
            blocking_cause_on_canonical_route = (
                canonical_route in _usage_blocking_routes(connection)
            )
            prior_route_state = self._route_state(connection, route)
            if record["policy_breach"] is not None:
                self._append_route_state(
                    connection,
                    route=route,
                    state="OPEN",
                    reason=str(record["policy_breach"]),
                    invocation_id=invocation_id,
                    recorded_at=observed_at,
                )
            elif (
                not blocking_cause_on_canonical_route
                and prior_route_state["state"] == "OPEN"
            ):
                self._append_route_state(
                    connection,
                    route=route,
                    state="CLOSED",
                    reason="VALID_PROVIDER_TELEMETRY_RECONCILED",
                    invocation_id=invocation_id,
                    recorded_at=observed_at,
                )
            connection.commit()
        finally:
            connection.close()

    def route_state(self, route: str) -> dict[str, object]:
        connection = self._connection()
        try:
            return self._route_state(connection, route)
        finally:
            connection.close()

    def open_route_circuit(
        self,
        *,
        route: str,
        reason: str,
        invocation_id: str | None,
        recorded_at: datetime,
    ) -> None:
        """Open one affected Graphiti route after a typed systemic outcome."""

        if not route.startswith("GRAPHITI_"):
            raise ModelUsageAdmissionError(
                "Graphiti route circuit operation targeted another workload"
            )
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._append_route_state(
                connection,
                route=route,
                state="OPEN",
                reason=_token(reason, field="route circuit reason"),
                invocation_id=invocation_id,
                recorded_at=recorded_at,
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def release_route_circuit(
        self,
        *,
        route: str,
        release_kind: str,
        bound_failure_reason: str,
        evidence_digest: str,
        recorded_at: datetime,
    ) -> None:
        """Release a Graphiti route using the checked #729 evidence vocabulary."""

        if release_kind not in {
            "DETERMINISTIC_HEALTH_PROBE",
            "AUTHORISED_OPERATOR_RESET",
        }:
            raise ModelUsageAdmissionError("Graphiti circuit release kind is invalid")
        if not route.startswith("GRAPHITI_"):
            raise ModelUsageAdmissionError(
                "Graphiti route circuit operation targeted another workload"
            )
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            state = self._route_state(connection, route)
            if state["state"] != "OPEN":
                raise ModelUsageAdmissionError("Graphiti route circuit is not open")
            if state["reason"] != bound_failure_reason:
                raise ModelUsageAdmissionError(
                    "Graphiti circuit release is not bound to the current failure"
                )
            if _canonical_circuit_route(route) in _usage_blocking_routes(connection):
                raise ModelUsageAdmissionError(
                    "Graphiti circuit has unresolved usage or a policy breach"
                )
            self._append_route_state(
                connection,
                route=route,
                state="CLOSED",
                reason=f"{release_kind}:{_token(evidence_digest, field='evidence digest')}",
                invocation_id=None,
                recorded_at=recorded_at,
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _route_state(
        self, connection: sqlite3.Connection, route: str
    ) -> dict[str, object]:
        canonical_route = _canonical_circuit_route(route)
        usage_blocking = canonical_route in _usage_blocking_routes(connection)
        has_canonical = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='unpublished_route_circuits'"
        ).fetchone()
        if canonical_route == CONT_WRITER_ROUTE and has_canonical is not None:
            canonical = connection.execute(
                "SELECT state,open_reason,opened_at FROM unpublished_route_circuits "
                "WHERE route=?",
                (canonical_route,),
            ).fetchone()
            if canonical is not None and (
                str(canonical[0]) == "OPEN" or not usage_blocking
            ):
                return {
                    "route": canonical_route,
                    "state": str(canonical[0]),
                    "reason": str(canonical[1]),
                    "invocation_id": None,
                    "recorded_at": canonical[2],
                    "authority": "UNPUBLISHED_ROUTE_CIRCUIT",
                }
        row = connection.execute(
            "SELECT state,reason,invocation_id,recorded_at "
            "FROM model_usage_route_circuit_events WHERE route=? "
            "ORDER BY recorded_at DESC,rowid DESC LIMIT 1",
            (canonical_route,),
        ).fetchone()
        if row is None:
            if usage_blocking:
                return {
                    "route": canonical_route,
                    "state": "OPEN",
                    "reason": "UNRESOLVED_USAGE_OR_POLICY_BREACH",
                    "invocation_id": None,
                    "authority": "MODEL_USAGE_RECEIPT",
                }
            return {
                "route": route,
                "state": "CLOSED",
                "reason": "",
                "invocation_id": None,
            }
        state = "OPEN" if usage_blocking else str(row[0])
        return {
            "route": route,
            "state": state,
            "reason": (
                "UNRESOLVED_USAGE_OR_POLICY_BREACH"
                if usage_blocking and str(row[0]) != "OPEN"
                else str(row[1])
            ),
            "invocation_id": row[2],
            "recorded_at": str(row[3]),
        }

    def _append_route_state(
        self,
        connection: sqlite3.Connection,
        *,
        route: str,
        state: str,
        reason: str,
        invocation_id: str | None,
        recorded_at: datetime,
    ) -> None:
        canonical_route = _canonical_circuit_route(route)
        has_canonical = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='unpublished_route_circuits'"
        ).fetchone()
        if (
            state == "OPEN"
            and canonical_route == CONT_WRITER_ROUTE
            and has_canonical is not None
        ):
            connection.execute(
                "INSERT INTO unpublished_route_circuits("
                "route,state,open_reason,opened_at,release_evidence_json,"
                "release_evidence_digest,last_probe_at) VALUES(?,?,?,?,NULL,NULL,NULL) "
                "ON CONFLICT(route) DO UPDATE SET state='OPEN',open_reason=excluded.open_reason,"
                "opened_at=COALESCE(unpublished_route_circuits.opened_at,excluded.opened_at),"
                "release_evidence_json=NULL,release_evidence_digest=NULL",
                (
                    canonical_route,
                    "OPEN",
                    reason,
                    _utc_text(recorded_at),
                ),
            )
        record = {
            "schema_version": MODEL_USAGE_SCHEMA_VERSION,
            "route": canonical_route,
            "state": state,
            "reason": reason,
            "invocation_id": invocation_id,
            "recorded_at": _utc_text(recorded_at),
        }
        digest = digest_canonical(record)
        connection.execute(
            "INSERT OR IGNORE INTO model_usage_route_circuit_events("
            "event_digest,route,state,reason,invocation_id,recorded_at,record_json) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                digest,
                canonical_route,
                state,
                reason,
                invocation_id,
                record["recorded_at"],
                _json({**record, "event_digest": digest}),
            ),
        )

    def query(self, *, start: datetime, end: datetime) -> dict[str, object]:
        if end <= start:
            raise ValueError("usage query end must follow start")
        connection = self._connection()
        try:
            envelope_rows = connection.execute(
                "SELECT record_json FROM model_work_envelopes "
                "WHERE admitted_at>=? AND admitted_at<? ORDER BY admitted_at,envelope_id",
                (_utc_text(start), _utc_text(end)),
            ).fetchall()
            leaf_rows = connection.execute(
                "SELECT a.record_json,t.record_json,o.record_json,e.record_json "
                "FROM model_invocation_allocations a "
                "JOIN model_work_envelopes e ON e.envelope_id=a.envelope_id "
                "LEFT JOIN model_invocation_terminals t ON t.invocation_id=a.invocation_id "
                "LEFT JOIN model_work_outcomes o ON o.envelope_id=a.envelope_id "
                "WHERE (a.allocated_at>=? AND a.allocated_at<?) "
                "OR (t.completed_at>=? AND t.completed_at<?) "
                "OR EXISTS (SELECT 1 FROM model_usage_reconciliations r "
                "WHERE r.invocation_id=a.invocation_id "
                "AND r.observed_at>=? AND r.observed_at<?) "
                "OR EXISTS (SELECT 1 "
                "FROM model_usage_conservative_dispositions d "
                "WHERE d.invocation_id=a.invocation_id "
                "AND d.observed_at>=? AND d.observed_at<?) "
                "OR EXISTS (SELECT 1 FROM model_transport_observations x "
                "WHERE x.invocation_id=a.invocation_id "
                "AND x.state='DISPATCH_STARTED' "
                "AND x.observed_at>=? AND x.observed_at<?) "
                "OR (o.terminal_at>=? AND o.terminal_at<?) "
                "ORDER BY a.allocated_at,a.cycle_id,a.envelope_id,a.leaf_ordinal",
                (
                    _utc_text(start),
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                ),
            ).fetchall()
            outcome_rows = connection.execute(
                "SELECT o.record_json,e.record_json "
                "FROM model_work_outcomes o "
                "JOIN model_work_envelopes e ON e.envelope_id=o.envelope_id "
                "WHERE o.terminal_at<? AND ("
                "(e.admitted_at>=? AND e.admitted_at<?) OR "
                "(o.terminal_at>=? AND o.terminal_at<?)) "
                "ORDER BY o.terminal_at,o.envelope_id",
                (
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                    _utc_text(start),
                    _utc_text(end),
                ),
            ).fetchall()
            reconciliations = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT invocation_id,record_json FROM model_usage_reconciliations "
                    "WHERE rowid IN (SELECT MAX(rowid) FROM model_usage_reconciliations "
                    "WHERE observed_at<? GROUP BY invocation_id)",
                    (_utc_text(end),),
                )
            }
            conservative_dispositions = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT invocation_id,record_json "
                    "FROM model_usage_conservative_dispositions "
                    "WHERE observed_at<?",
                    (_utc_text(end),),
                )
            }
            provider_attempts = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT invocation_id,record_json "
                    "FROM model_invocation_provider_attempt_links"
                )
            }
            graphiti_internal_requests = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT invocation_id,record_json "
                    "FROM graphiti_internal_requests"
                )
            }
            context_manifests = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT context_manifest_digest,record_json "
                    "FROM model_invocation_context_manifests"
                )
            }
            context_observations = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT invocation_id,record_json "
                    "FROM model_invocation_context_observations"
                )
            }
            dispatch_observations = {
                str(row[0]): str(row[1])
                for row in connection.execute(
                    "SELECT invocation_id,MIN(observed_at) "
                    "FROM model_transport_observations "
                    "WHERE state='DISPATCH_STARTED' AND observed_at<? "
                    "GROUP BY invocation_id",
                    (_utc_text(end),),
                )
            }
            projected_cycle_outcomes = {
                str(row[0]): _object(row[1])
                for row in connection.execute(
                    "SELECT cycle_id,record_json FROM model_usage_cycle_outcomes "
                    "WHERE terminal_at>=? AND terminal_at<? ORDER BY terminal_at,cycle_id",
                    (_utc_text(start), _utc_text(end)),
                )
            }
            has_governor = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='unpublished_governed_cycles'"
            ).fetchone()
            canonical_cycle_outcomes: dict[str, dict[str, object]] = {}
            if has_governor is not None:
                for row in connection.execute(
                    "SELECT cycle_id,outcome_class,terminal_at,"
                    "writer_unproductive_streak_before,"
                    "writer_unproductive_streak_after,writer_circuit_state,"
                    "writer_circuit_open_reason,cooldown_seconds,"
                    "cooldown_policy_version,next_cycle_eligible_at "
                    "FROM unpublished_governed_cycles "
                    "WHERE lease_state IN ('TERMINAL','RECOVERED') "
                    "AND terminal_at>=? AND terminal_at<? "
                    "ORDER BY terminal_at,cycle_id",
                    (_utc_text(start), _utc_text(end)),
                ):
                    canonical_cycle_outcomes[str(row[0])] = {
                        "schema_version": MODEL_USAGE_SCHEMA_VERSION,
                        "cycle_id": str(row[0]),
                        "outcome_class": str(row[1]),
                        "terminal_at": str(row[2]),
                        "writer_unproductive_streak_before": int(row[3]),
                        "writer_unproductive_streak_after": int(row[4]),
                        "writer_circuit_state": str(row[5]),
                        "writer_circuit_open_reason": str(row[6]),
                        "cooldown_seconds": (
                            None if row[7] is None else int(row[7])
                        ),
                        "cooldown_policy_version": (
                            None if row[8] is None else str(row[8])
                        ),
                        "next_cycle_eligible_at": (
                            None if row[9] is None else str(row[9])
                        ),
                        "authority": "UNPUBLISHED_GOVERNED_CYCLE_TERMINAL",
                    }
            cycle_outcomes = {
                **projected_cycle_outcomes,
                **canonical_cycle_outcomes,
            }
            zero_call_admissions = [
                {
                    "decision_id": str(row[0]),
                    "decision": str(row[1]),
                    "cycle_id": str(row[2]),
                    "recorded_at": str(row[3]),
                }
                for row in connection.execute(
                    "SELECT decision_id,decision,cycle_id,recorded_at "
                    "FROM model_zero_call_admissions "
                    "WHERE recorded_at>=? AND recorded_at<? "
                    "ORDER BY recorded_at,decision_id",
                    (_utc_text(start), _utc_text(end)),
                )
            ]
        finally:
            connection.close()
        outcomes: dict[str, dict[str, object]] = {}
        envelopes = []
        envelope_ids: set[str] = set()
        for row in envelope_rows:
            record = _object(row[0])
            envelopes.append(record)
            envelope_ids.add(str(record["envelope_id"]))
        for outcome_json, envelope_json in outcome_rows:
            outcome = _object(outcome_json)
            envelope = _object(envelope_json)
            envelope_id = str(envelope["envelope_id"])
            outcomes[envelope_id] = outcome
            if envelope_id not in envelope_ids:
                envelopes.append(envelope)
                envelope_ids.add(envelope_id)
        leaves: list[dict[str, object]] = []
        for allocation_json, terminal_json, outcome_json, envelope_json in leaf_rows:
            allocation = _object(allocation_json)
            envelope = _object(envelope_json)
            terminal = None if terminal_json is None else _object(terminal_json)
            if terminal is not None and _instant(str(terminal["completed_at"])) >= end:
                terminal = None
            outcome = None if outcome_json is None else _object(outcome_json)
            if outcome is not None and _instant(str(outcome["terminal_at"])) >= end:
                outcome = None
            effective = dict(terminal or {})
            invocation_id = str(allocation["invocation_id"])
            disposition = conservative_dispositions.get(invocation_id)
            reconciliation = reconciliations.get(invocation_id)
            provider_attempt = provider_attempts.get(invocation_id)
            if disposition is not None:
                effective.update(
                    {
                        "usage_status": disposition["usage_status"],
                        "components": disposition["components"],
                        "estimate_policy_digest": disposition[
                            "estimate_policy_digest"
                        ],
                        "estimate_calculation": disposition[
                            "estimate_calculation"
                        ],
                        "completed_at": disposition["observed_at"],
                        "observed_at": disposition["observed_at"],
                    }
                )
            if reconciliation is not None:
                effective.update(
                    {
                        "usage_status": reconciliation["usage_status"],
                        "components": reconciliation["components"],
                        "provider_telemetry_digest": reconciliation[
                            "provider_telemetry_digest"
                        ],
                        "raw_telemetry_pointer": reconciliation[
                            "raw_telemetry_pointer"
                        ],
                        "reconciled_at": reconciliation["observed_at"],
                        "completed_at": reconciliation["observed_at"],
                        "policy_breach": reconciliation.get("policy_breach"),
                    }
                )
            components = effective.get("components")
            if not isinstance(components, dict):
                components = UsageComponents().as_record()
            row = {
                **allocation,
                "schema_version": MODEL_USAGE_INTERFACE_SCHEMA_VERSION,
                "allocation_schema_version": allocation.get("schema_version"),
                "admission_decision_id": envelope.get("admission_decision_id"),
                "candidate_id": envelope.get("candidate_id"),
                "hypothesis_digest": envelope.get("hypothesis_digest"),
                "evidence_package_digest": envelope.get("evidence_package_digest"),
                "ingest_id": envelope.get("ingest_id"),
                "graphiti_attempt_id": envelope.get("graphiti_attempt_id"),
                "provider_attempt_id": (
                    None
                    if provider_attempt is None
                    else provider_attempt.get("provider_attempt_id")
                ),
                "graphiti_internal_request": graphiti_internal_requests.get(
                    str(allocation["invocation_id"])
                ),
                "context_manifest": context_manifests.get(
                    str(allocation["context_manifest_digest"])
                ),
                "context_manifest_observation": context_observations.get(
                    str(allocation["invocation_id"])
                ),
                "usage_status": effective.get("usage_status"),
                "terminal_digest": (
                    None if terminal is None else terminal.get("terminal_digest")
                ),
                "terminal_usage_status": (
                    None if terminal is None else terminal.get("usage_status")
                ),
                "terminal_components": (
                    None if terminal is None else terminal.get("components")
                ),
                "terminal_completed_at": (
                    None if terminal is None else terminal.get("completed_at")
                ),
                "reconciliation_usage_status": (
                    None
                    if reconciliation is None
                    else reconciliation.get("usage_status")
                ),
                "reconciliation_components": (
                    None
                    if reconciliation is None
                    else reconciliation.get("components")
                ),
                "reconciled_at": effective.get("reconciled_at"),
                "conservative_disposition_digest": (
                    None
                    if disposition is None
                    else disposition.get("disposition_digest")
                ),
                "disposition_usage_status": (
                    None if disposition is None else disposition.get("usage_status")
                ),
                "disposition_components": (
                    None if disposition is None else disposition.get("components")
                ),
                "disposed_at": (
                    None if disposition is None else disposition.get("observed_at")
                ),
                "disposition_authority_digest": (
                    None
                    if disposition is None
                    else disposition.get("authority_digest")
                ),
                "disposition_approval_reference": (
                    None
                    if disposition is None
                    else disposition.get("approval_reference")
                ),
                "disposition_approved_plan_digest": (
                    None
                    if disposition is None
                    else disposition.get("approved_plan_digest")
                ),
                "exact_usage_remains_unknown": (
                    None
                    if disposition is None
                    else reconciliation is None
                ),
                "provider_dispatch_preserved": (
                    None
                    if disposition is None
                    else disposition.get("provider_dispatch_preserved")
                ),
                "unknown_spend_released": (
                    None
                    if disposition is None
                    else disposition.get("unknown_spend_released")
                ),
                "invocation_outcome": effective.get("outcome"),
                "failure_class": effective.get("failure_class"),
                **components,
                "dispatch_at": (
                    None
                    if effective.get("pre_dispatch_zero_proved")
                    else dispatch_observations.get(
                        str(allocation["invocation_id"]),
                        effective.get("dispatch_at"),
                    )
                ),
                "transport_dispatch_observed": str(allocation["invocation_id"])
                in dispatch_observations,
                "actual_provider_dispatch": bool(
                    not effective.get("pre_dispatch_zero_proved")
                    and (
                        str(allocation["invocation_id"]) in dispatch_observations
                        or (
                            terminal is not None
                            and terminal.get("dispatch_at") is not None
                            and terminal.get("outcome") != "RECOVERED_UNRESOLVED"
                        )
                    )
                ),
                "completed_at": effective.get("completed_at"),
                "observed_at": effective.get("observed_at"),
                "provider_telemetry_digest": effective.get("provider_telemetry_digest"),
                "raw_telemetry_pointer": effective.get("raw_telemetry_pointer"),
                "estimate_policy_digest": effective.get("estimate_policy_digest"),
                "estimate_calculation": effective.get("estimate_calculation"),
                "od_011_reference": effective.get("od_011_reference"),
                "subscription_cli_chat_not_cash_debited": effective.get(
                    "subscription_cli_chat_not_cash_debited"
                ),
                "pre_dispatch_zero_proved": effective.get(
                    "pre_dispatch_zero_proved", False
                ),
                "policy_breach": effective.get("policy_breach"),
                "uncertainty": (
                    effective.get("usage_status")
                    if effective.get("usage_status")
                    in {"ESTIMATED", "UNREPORTED", "AMBIGUOUS", "INVALID"}
                    else ""
                ),
                "work_outcome": None if outcome is None else outcome.get("outcome"),
                "work_outcome_record_id": (
                    None if outcome is None else outcome.get("outcome_record_id")
                ),
                "work_outcome_terminal_at": (
                    None if outcome is None else outcome.get("terminal_at")
                ),
                "payload_digest": None
                if outcome is None
                else outcome.get("payload_digest"),
                "cycle_outcome": None
                if outcome is None
                else outcome.get("cycle_outcome"),
                "route_circuit_state": None
                if outcome is None
                else outcome.get("route_circuit_state"),
                "route_circuit_reason": None
                if outcome is None
                else outcome.get("route_circuit_reason"),
                "retained_proposal_count": None
                if outcome is None
                else outcome.get("retained_proposal_count"),
                "accepted_provider_attempt_id": None
                if outcome is None
                else outcome.get("accepted_provider_attempt_id"),
                "stable_reason_codes": []
                if outcome is None
                else outcome.get("stable_reason_codes", []),
            }
            leaves.append(row)
            if str(envelope["envelope_id"]) not in envelope_ids:
                envelopes.append(envelope)
                envelope_ids.add(str(envelope["envelope_id"]))
            if outcome is not None:
                outcomes[str(outcome["envelope_id"])] = outcome
        for envelope in envelopes:
            work_outcome = outcomes.get(str(envelope["envelope_id"]))
            if work_outcome is not None:
                envelope.update(work_outcome)
                envelope["work_outcome_terminal_at"] = work_outcome["terminal_at"]
            envelope.update(cycle_outcomes.get(str(envelope["cycle_id"]), {}))
        return {
            "envelopes": envelopes,
            "leaves": leaves,
            "cycle_outcomes": [
                cycle_outcomes[key]
                for key in sorted(
                    cycle_outcomes,
                    key=lambda item: (str(cycle_outcomes[item]["terminal_at"]), item),
                )
            ],
            "zero_call_admissions": zero_call_admissions,
        }

    def report(
        self, *, start: datetime, end: datetime, bucket_seconds: int = 300
    ) -> dict[str, object]:
        if bucket_seconds <= 0:
            raise ValueError("usage bucket must be positive")
        data = self.query(start=start, end=end)
        leaves = data["leaves"]
        envelopes = data["envelopes"]
        assert isinstance(leaves, list)
        assert isinstance(envelopes, list)
        allocated_leaves = [
            row
            for row in leaves
            if start <= _instant(str(row["allocated_at"])) < end
        ]
        terminal_leaves = [
            row
            for row in leaves
            if row["usage_status"] is not None
            and isinstance(row.get("completed_at"), str)
            and start <= _instant(str(row["completed_at"])) < end
        ]
        allocation_terminals = [
            row for row in allocated_leaves if row["usage_status"] is not None
        ]
        accounted_leaves = [
            row
            for row in terminal_leaves
            if row["usage_status"] in {"REPORTED", "ESTIMATED"}
        ]
        totals = [
            int(row["total_tokens"])
            for row in accounted_leaves
            if _is_int(row.get("total_tokens"))
        ]
        observed_total = sum(totals)
        reported_total = sum(
            int(row["total_tokens"])
            for row in accounted_leaves
            if row["usage_status"] == "REPORTED" and _is_int(row.get("total_tokens"))
        )
        estimated_total = sum(
            int(row["total_tokens"])
            for row in terminal_leaves
            if row["usage_status"] == "ESTIMATED" and _is_int(row.get("total_tokens"))
        )
        unresolved = sum(
            row["usage_status"] in {"UNREPORTED", "AMBIGUOUS", "INVALID"}
            for row in terminal_leaves
        )
        accepted_envelopes = {
            str(row["envelope_id"])
            for row in leaves
            if row["work_outcome"] == "ACCEPTED" and row["payload_digest"]
        }
        graphiti_envelopes = {
            str(row["envelope_id"])
            for row in leaves
            if row["work_outcome"] in _GRAPHITI_COMPLETED_USEFUL_OUTCOMES
        }

        def productive_leaf(row: Mapping[str, object]) -> bool:
            workload = str(row["workload_class"])
            if workload.startswith("CONT_WRITER_"):
                accepted_attempt = row.get("accepted_provider_attempt_id")
                return bool(
                    accepted_attempt
                    and row.get("provider_attempt_id") == accepted_attempt
                    and row.get("invocation_outcome") == "ACCEPTED_OUTPUT"
                )
            if workload.startswith("GRAPHITI_"):
                return (
                    str(row["envelope_id"]) in graphiti_envelopes
                    and row.get("invocation_outcome") == "COMPLETE"
                )
            return str(row["envelope_id"]) in accepted_envelopes

        productive = sum(
            int(row["total_tokens"])
            for row in accounted_leaves
            if productive_leaf(row) and _is_int(row.get("total_tokens"))
        )
        no_result = observed_total - productive
        no_result_reasons: Counter[str] = Counter()
        for row in accounted_leaves:
            if productive_leaf(row):
                continue
            if _is_int(row.get("total_tokens")):
                stable_reasons = row.get("stable_reason_codes")
                stable_reason = (
                    str(stable_reasons[0])
                    if isinstance(stable_reasons, list) and stable_reasons
                    else None
                )
                no_result_reasons[
                    str(
                        row["failure_class"]
                        or stable_reason
                        or row["invocation_outcome"]
                        or "UNKNOWN"
                    )
                ] += int(row["total_tokens"])
        workload_totals: Counter[str] = Counter()
        provider_totals: Counter[str] = Counter()
        model_totals: Counter[str] = Counter()
        status_totals: Counter[str] = Counter()
        outcome_totals: Counter[str] = Counter()
        context_tokens = 0
        provider_input_tokens = 0
        for row in accounted_leaves:
            total = row.get("total_tokens")
            if _is_int(total):
                workload_totals[str(row["workload_class"])] += total
                provider_totals[str(row["provider"])] += total
                model_totals[str(row["model"])] += total
                status_totals[str(row["usage_status"])] += total
                outcome_totals[
                    str(
                        row["work_outcome"] or row["invocation_outcome"] or "UNRESOLVED"
                    )
                ] += total
            context = row.get("context_tokens")
            if _is_int(context):
                context_tokens += context
            input_tokens = row.get("input_tokens")
            if _is_int(input_tokens) and str(row["workload_class"]).startswith("CONT_"):
                provider_input_tokens += input_tokens
        graphiti_tokens = sum(
            int(row["total_tokens"])
            for row in accounted_leaves
            if str(row["envelope_id"]) in graphiti_envelopes
            and _is_int(row.get("total_tokens"))
        )
        proposals = sum(
            int(row["retained_proposal_count"])
            for row in data["envelopes"]  # type: ignore[union-attr]
            if str(row.get("envelope_id")) in graphiti_envelopes
            and _is_int(row.get("retained_proposal_count"))
        )
        fixed_buckets = self._fixed_buckets(
            leaves=accounted_leaves,
            start=start,
            end=end,
            bucket_seconds=bucket_seconds,
        )
        daily: Counter[str] = Counter()
        for row in accounted_leaves:
            total = row.get("total_tokens")
            completed = row.get("completed_at")
            if _is_int(total) and isinstance(completed, str):
                daily[_instant(completed).date().isoformat()] += total
        zero_calls = self._zero_call_counts(start=start, end=end)
        rolling_data = self.query(
            start=start - timedelta(seconds=300), end=end
        )["leaves"]
        assert isinstance(rolling_data, list)
        rolling_accounted = [
            row
            for row in rolling_data
            if row["usage_status"] in {"REPORTED", "ESTIMATED"}
        ]
        rolling = self._rolling_dispatch_usage(
            rolling_accounted, start=start, end=end
        )
        cycle_rows = data["cycle_outcomes"]
        assert isinstance(cycle_rows, list)
        cycle_counts = Counter(str(row["outcome_class"]) for row in cycle_rows)
        latest_cycle = cycle_rows[-1] if cycle_rows else None
        writer_leaves = [
            row
            for row in terminal_leaves
            if row["workload_class"] in {"CONT_WRITER_PRIMARY", "CONT_WRITER_FALLBACK"}
        ]
        fallback_leaves = [
            row
            for row in writer_leaves
            if row["workload_class"] == "CONT_WRITER_FALLBACK"
        ]
        recovered_fallback_envelopes = {
            str(row["envelope_id"])
            for row in fallback_leaves
            if str(row["envelope_id"]) in accepted_envelopes
        }
        fallback_tokens = sum(
            int(row["total_tokens"])
            for row in fallback_leaves
            if row["usage_status"] in {"REPORTED", "ESTIMATED"}
            and _is_int(row.get("total_tokens"))
        )
        fallback_no_result_tokens = sum(
            int(row["total_tokens"])
            for row in fallback_leaves
            if str(row["envelope_id"]) not in accepted_envelopes
            and row["usage_status"] in {"REPORTED", "ESTIMATED"}
            and _is_int(row.get("total_tokens"))
        )
        accepted_by_cycle: Counter[str] = Counter(
            str(row["cycle_id"])
            for row in data["envelopes"]  # type: ignore[union-attr]
            if row.get("outcome") == "ACCEPTED" and row.get("payload_digest")
        )
        for row in leaves:
            if str(row["workload_class"]).startswith("CONT_WRITER_"):
                accepted_by_cycle.setdefault(str(row["cycle_id"]), 0)
        dispatches = [
            row
            for row in leaves
            if isinstance(row.get("dispatch_at"), str)
            and row.get("actual_provider_dispatch") is True
            and start <= _instant(str(row["dispatch_at"])) < end
        ]
        outstanding = len(allocated_leaves) - len(allocation_terminals)
        envelope_outcome_counts = Counter(
            str(row["outcome"]) for row in envelopes if row.get("outcome")
        )
        return {
            "schema_version": MODEL_USAGE_INTERFACE_SCHEMA_VERSION,
            "start": _utc_text(start),
            "end": _utc_text(end),
            "bucket_seconds": bucket_seconds,
            "envelope_count": len(data["envelopes"]),  # type: ignore[arg-type]
            "envelopes": envelopes,
            "envelope_outcome_counts": dict(sorted(envelope_outcome_counts.items())),
            "allocation_count": len(allocated_leaves),
            "actual_provider_dispatch_count": len(dispatches),
            "terminal_count": len(allocation_terminals),
            "outstanding_count": outstanding,
            "allocation_reconciliation": {
                "allocation_count": len(allocated_leaves),
                "terminal_count": len(allocation_terminals),
                "outstanding_count": outstanding,
                "reconciles": len(allocated_leaves)
                == len(allocation_terminals) + outstanding,
            },
            "leaf_dispatch_count": len(dispatches),
            "terminal_leaf_count": len(terminal_leaves),
            "leaf_dispatch_count_reconciles": len(allocated_leaves)
            == len(allocation_terminals) + outstanding,
            "reported_tokens": reported_total,
            "estimated_tokens": estimated_total,
            "observed_total_tokens": observed_total,
            "envelope_allocated_tokens": observed_total,
            "context_tokens": context_tokens,
            "unresolved_invocation_count": unresolved,
            "accepted_payload_count": len(accepted_envelopes),
            "productive_tokens": productive,
            "no_result_tokens": no_result,
            "tokens_per_accepted_payload": (
                {"numerator": productive, "denominator": len(accepted_envelopes)}
                if accepted_envelopes
                else None
            ),
            "writer_leaf_calls_per_accepted_payload": (
                {"numerator": len(writer_leaves), "denominator": len(accepted_envelopes)}
                if accepted_envelopes
                else None
            ),
            "accepted_unpublished_payloads_by_cycle": dict(
                sorted(accepted_by_cycle.items())
            ),
            "no_result_reasons": dict(sorted(no_result_reasons.items())),
            "workload_totals": dict(sorted(workload_totals.items())),
            "provider_totals": dict(sorted(provider_totals.items())),
            "model_totals": dict(sorted(model_totals.items())),
            "usage_status_totals": dict(sorted(status_totals.items())),
            "outcome_totals": dict(sorted(outcome_totals.items())),
            "provider_context_to_input_ratio": {
                "numerator": context_tokens,
                "denominator": provider_input_tokens,
            },
            "context_to_newsroom_input_ratio": None,
            "context_to_newsroom_input_ratio_reason": (
                "AWAITING_730_EXACT_NEWSROOM_INPUT_TOKEN_MEASURE"
            ),
            "fallback_leaf_count": len(fallback_leaves),
            "fallback_tokens": fallback_tokens,
            "fallback_no_result_tokens": fallback_no_result_tokens,
            "fallback_recovery_rate": {
                "numerator": len(recovered_fallback_envelopes),
                "denominator": len(fallback_leaves),
            },
            "fallback_no_result_rate": {
                "numerator": sum(
                    str(row["envelope_id"]) not in accepted_envelopes
                    for row in fallback_leaves
                ),
                "denominator": len(fallback_leaves),
            },
            "graphiti_valid_ingest_count": len(graphiti_envelopes),
            "graphiti_tokens_per_valid_ingest": (
                {"numerator": graphiti_tokens, "denominator": len(graphiti_envelopes)}
                if graphiti_envelopes
                else None
            ),
            "graphiti_tokens_per_retained_proposal": (
                {"numerator": graphiti_tokens, "denominator": proposals}
                if proposals
                else None
            ),
            "zero_call_admission_counts": zero_calls,
            "cycle_outcome_counts": dict(sorted(cycle_counts.items())),
            "writer_unproductive_streak": (
                None
                if latest_cycle is None
                else latest_cycle["writer_unproductive_streak_after"]
            ),
            "fixed_buckets": fixed_buckets,
            "rolling_300_at_dispatch": rolling,
            "utc_day_totals": dict(sorted(daily.items())),
            "daily_500k_alert": any(
                value > DAILY_USAGE_ALERT_TOKENS for value in daily.values()
            ),
            "normal_daily_hard_cut": None,
            "missing_usage_is_zero": False,
            "graphiti_result_telemetry": self._graphiti_result_telemetry(
                leaves=leaves,
                allocated_leaves=allocated_leaves,
                terminal_leaves=terminal_leaves,
                accounted_leaves=accounted_leaves,
                envelopes=envelopes,
                start=start,
                end=end,
            ),
        }

    def _graphiti_result_telemetry(
        self,
        *,
        leaves: list[dict[str, object]],
        allocated_leaves: list[dict[str, object]],
        terminal_leaves: list[dict[str, object]],
        accounted_leaves: list[dict[str, object]],
        envelopes: list[dict[str, object]],
        start: datetime,
        end: datetime,
    ) -> dict[str, object]:
        def graphiti_leaf(row: Mapping[str, object]) -> bool:
            return str(row.get("workload_class") or "").startswith("GRAPHITI_")

        def token_sum(
            rows: list[dict[str, object]],
            *,
            status: str | None = None,
            workload: str | None = None,
        ) -> int:
            total = 0
            for row in rows:
                if status is not None and row.get("usage_status") != status:
                    continue
                if workload is not None and row.get("workload_class") != workload:
                    continue
                value = row.get("total_tokens")
                if _is_int(value):
                    total += value
            return total

        def ratio(numerator: int, denominator: int) -> dict[str, int] | None:
            if denominator == 0:
                return None
            return {"numerator": numerator, "denominator": denominator}

        with_proposals = {
            str(row["envelope_id"])
            for row in leaves
            if row.get("work_outcome") == "GRAPHITI_SUCCESS"
        }
        zero_proposals = {
            str(row["envelope_id"])
            for row in leaves
            if row.get("work_outcome") == "GRAPHITI_SUCCESS_ZERO_PROPOSALS"
        }
        completed = with_proposals | zero_proposals
        request_leaves = [
            row
            for row in leaves
            if isinstance(row.get("graphiti_internal_request"), dict)
        ]
        distinct_internal_requests = len(request_leaves)
        connection = self._connection()
        try:
            refusal_rows = connection.execute(
                "SELECT reason_code FROM graphiti_internal_request_refusals "
                "WHERE refused_at>=? AND refused_at<?",
                (_utc_text(start), _utc_text(end)),
            ).fetchall()
        finally:
            connection.close()
        graphiti_terminals = [row for row in terminal_leaves if graphiti_leaf(row)]
        graphiti_accounted = [row for row in accounted_leaves if graphiti_leaf(row)]
        failed_envelopes = {
            str(row["envelope_id"])
            for row in leaves
            if graphiti_leaf(row)
            and row.get("work_outcome")
            and str(row["envelope_id"]) not in completed
        }
        failed_tokens = sum(
            int(row["total_tokens"])
            for row in graphiti_accounted
            if str(row["envelope_id"]) in failed_envelopes
            and _is_int(row.get("total_tokens"))
        )
        success_tokens = sum(
            int(row["total_tokens"])
            for row in graphiti_accounted
            if str(row["envelope_id"]) in completed
            and _is_int(row.get("total_tokens"))
        )
        proposals = sum(
            int(row["retained_proposal_count"])
            for row in envelopes
            if str(row.get("envelope_id")) in completed
            and _is_int(row.get("retained_proposal_count"))
        )
        context_values = [
            int(row["context_tokens"])
            for row in request_leaves
            if _is_int(row.get("context_tokens"))
        ]
        context_overhead = (
            None
            if len(context_values) != distinct_internal_requests
            else ratio(sum(context_values), distinct_internal_requests)
        )
        call_shape = load_checked_graphiti_call_shape_policy()
        return {
            "completed_ingests_with_proposals": len(with_proposals),
            "completed_ingests_zero_proposals": len(zero_proposals),
            "completed_useful_ingest_count": len(completed),
            "distinct_internal_requests": distinct_internal_requests,
            "distinct_internal_requests_per_completed_ingest": ratio(
                distinct_internal_requests, len(completed)
            ),
            "call_shape_max_distinct_internal_requests": (
                call_shape.max_distinct_internal_requests
            ),
            "call_shape_headroom": call_shape.headroom,
            "duplicate_request_refusals": sum(
                reason == "DUPLICATE_INTERNAL_REQUEST" for reason, in refusal_rows
            ),
            "call_shape_drift_refusals": sum(
                reason == "CALL_SHAPE_DRIFT" for reason, in refusal_rows
            ),
            "primary_leaf_count": sum(
                row["workload_class"] == WorkloadClass.GRAPHITI_CHAT_PRIMARY.value
                for row in allocated_leaves
            ),
            "fallback_leaf_count": sum(
                row["workload_class"] == WorkloadClass.GRAPHITI_CHAT_FALLBACK.value
                for row in allocated_leaves
            ),
            "embedding_leaf_count": sum(
                row["workload_class"] == WorkloadClass.GRAPHITI_EMBEDDING.value
                for row in allocated_leaves
            ),
            "fallback_recovery_count": len(
                {
                    str(row["envelope_id"])
                    for row in allocated_leaves
                    if row["workload_class"]
                    == WorkloadClass.GRAPHITI_CHAT_FALLBACK.value
                    and str(row["envelope_id"]) in completed
                }
            ),
            "reported_tokens": token_sum(graphiti_terminals, status="REPORTED"),
            "estimated_tokens": token_sum(graphiti_terminals, status="ESTIMATED"),
            "unresolved_invocation_count": sum(
                row.get("usage_status") in _UNRESOLVED_USAGE_STATUSES
                for row in graphiti_terminals
            ),
            "failed_or_rolled_back_attempt_tokens": failed_tokens,
            "tokens_per_proposal": ratio(success_tokens, proposals),
            "context_overhead_per_internal_request": context_overhead,
            "route_circuit_states": {
                route: str(self.route_state(route)["state"])
                for route in _GRAPHITI_CIRCUIT_ROUTES
            },
            "embedding_tokens": token_sum(
                graphiti_accounted,
                workload=WorkloadClass.GRAPHITI_EMBEDDING.value,
            ),
            "embedding_od_011_references": sorted(
                {
                    str(row["od_011_reference"])
                    for row in graphiti_terminals
                    if row["workload_class"]
                    == WorkloadClass.GRAPHITI_EMBEDDING.value
                    and isinstance(row.get("od_011_reference"), str)
                    and row["od_011_reference"]
                }
            ),
            "cli_chat_cash_debited": any(
                row.get("subscription_cli_chat_not_cash_debited") is False
                for row in graphiti_terminals
                if row["workload_class"]
                in {
                    WorkloadClass.GRAPHITI_CHAT_PRIMARY.value,
                    WorkloadClass.GRAPHITI_CHAT_FALLBACK.value,
                }
            ),
            "normal_daily_hard_cut": None,
            "missing_usage_is_zero": False,
        }

    def _fixed_buckets(
        self,
        *,
        leaves: list[dict[str, object]],
        start: datetime,
        end: datetime,
        bucket_seconds: int,
    ) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        start_utc = start.astimezone(UTC)
        end_utc = end.astimezone(UTC)
        epoch_seconds = int(start_utc.timestamp())
        cursor = datetime.fromtimestamp(
            epoch_seconds - (epoch_seconds % bucket_seconds), tz=UTC
        )
        while cursor < end:
            boundary = cursor + timedelta(seconds=bucket_seconds)
            proof_start = max(cursor, start_utc)
            proof_end = min(boundary, end_utc)
            total = 0
            for row in leaves:
                completed = row.get("completed_at")
                tokens = row.get("total_tokens")
                if (
                    isinstance(completed, str)
                    and _is_int(tokens)
                    and proof_start <= _instant(completed) < proof_end
                ):
                    total += tokens
            result.append(
                {
                    "window_start": _utc_text(cursor),
                    "window_end": _utc_text(boundary),
                    "observed_total_tokens": total,
                }
            )
            cursor = boundary
        return result

    def _rolling_dispatch_usage(
        self,
        leaves: list[dict[str, object]],
        *,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for row in leaves:
            dispatched = row.get("dispatch_at")
            if (
                not isinstance(dispatched, str)
                or row.get("actual_provider_dispatch") is not True
            ):
                continue
            at = _instant(dispatched)
            if not start <= at < end:
                continue
            total = 0
            for other in leaves:
                completed = other.get("completed_at")
                tokens = other.get("total_tokens")
                if (
                    isinstance(completed, str)
                    and _is_int(tokens)
                    and at - timedelta(seconds=300) < _instant(completed) <= at
                ):
                    total += tokens
            result.append(
                {
                    "invocation_id": row["invocation_id"],
                    "observed_total_tokens": total,
                }
            )
        return result

    def _zero_call_counts(self, *, start: datetime, end: datetime) -> dict[str, int]:
        connection = self._connection()
        try:
            rows = connection.execute(
                "SELECT decision,COUNT(*) FROM model_zero_call_admissions "
                "WHERE recorded_at>=? AND recorded_at<? GROUP BY decision",
                (_utc_text(start), _utc_text(end)),
            ).fetchall()
        finally:
            connection.close()
        counts = {"HOLD": 0, "REJECT": 0}
        for decision, count in rows:
            counts[str(decision)] = int(count)
        return counts

    def export_csv(self, *, start: datetime, end: datetime) -> str:
        leaves = self.query(start=start, end=end)["leaves"]
        assert isinstance(leaves, list)
        fields = (
            "schema_version",
            "envelope_id",
            "invocation_id",
            "allocation_schema_version",
            "cycle_id",
            "leaf_ordinal",
            "workload_class",
            "admission_decision_id",
            "candidate_id",
            "hypothesis_digest",
            "evidence_package_digest",
            "ingest_id",
            "graphiti_attempt_id",
            "provider_attempt_id",
            "work_outcome_record_id",
            "provider",
            "route",
            "model",
            "reasoning",
            "parent_invocation_id",
            "invocation_policy_digest",
            "prompt_contract_version",
            "prompt_bytes",
            "prompt_digest",
            "request_digest",
            "output_schema_digest",
            "max_output_tokens",
            "context_manifest_digest",
            "allocated_at",
            "dispatch_at",
            "completed_at",
            "work_outcome",
            "invocation_outcome",
            "failure_class",
            "usage_status",
            "terminal_usage_status",
            "reconciliation_usage_status",
            "reconciled_at",
            "conservative_disposition_digest",
            "disposition_usage_status",
            "disposed_at",
            "disposition_authority_digest",
            "disposition_approval_reference",
            "disposition_approved_plan_digest",
            "exact_usage_remains_unknown",
            "provider_dispatch_preserved",
            "unknown_spend_released",
            "input_tokens",
            "output_tokens",
            "cached_read_tokens",
            "cached_write_tokens",
            "reasoning_tokens",
            "context_tokens",
            "total_tokens",
            "provenance",
            "uncertainty",
            "estimate_policy_digest",
            "estimate_calculation",
            "payload_digest",
            "provider_telemetry_digest",
            "raw_telemetry_pointer",
            "od_011_reference",
            "subscription_cli_chat_not_cash_debited",
        )
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in leaves:
            writer.writerow({field: row.get(field) for field in fields})
        return output.getvalue()

    def export_envelope_csv(self, *, start: datetime, end: datetime) -> str:
        envelopes = self.query(start=start, end=end)["envelopes"]
        assert isinstance(envelopes, list)
        fields = (
            "schema_version",
            "envelope_id",
            "cycle_id",
            "workload_class",
            "admitted_at",
            "admission_decision_id",
            "candidate_id",
            "hypothesis_digest",
            "evidence_package_digest",
            "ingest_id",
            "graphiti_attempt_id",
            "canonical_digest",
            "outcome",
            "outcome_record_id",
            "outcome_digest",
            "payload_digest",
            "work_outcome_terminal_at",
            "terminal_at",
            "cycle_outcome",
            "route_circuit_state",
            "route_circuit_reason",
            "retained_proposal_count",
            "accepted_provider_attempt_id",
            "stable_reason_codes",
        )
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in envelopes:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "stable_reason_codes": json.dumps(
                        row.get("stable_reason_codes", []),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
        return output.getvalue()

    def export_bucket_csv(
        self, *, start: datetime, end: datetime, bucket_seconds: int = 300
    ) -> str:
        """Export the canonical fixed-bucket shape used by the 300s incident CSV."""

        data = self.query(start=start, end=end)
        leaves = data["leaves"]
        cycles = data["cycle_outcomes"]
        admissions = data["zero_call_admissions"]
        assert isinstance(leaves, list)
        assert isinstance(cycles, list)
        assert isinstance(admissions, list)
        fields = (
            "bucket_start_utc",
            "bucket_end_utc",
            "cycle_results",
            "minted_reported",
            "graphiti_successes_reported",
            "grok_writer_sessions",
            "grok_completed_sessions",
            "grok_model_calls",
            "grok_input_tokens",
            "grok_output_tokens",
            "grok_total_tokens",
            "grok_cached_read_tokens",
            "grok_reasoning_tokens",
            "cursor_fallback_sessions",
            "stored_outputs",
            "stored_grok_outputs",
            "stored_cursor_outputs",
            "stored_other_outputs",
            "reported_tokens",
            "estimated_tokens",
            "unresolved_invocations",
            "productive_tokens",
            "no_result_tokens",
            "unreported_invocations",
            "ambiguous_invocations",
            "invalid_invocations",
            "admission_only_hold",
            "admission_only_reject",
            "idle_qualified_zero_cycles",
            "productive_cycles",
            "unproductive_provider_cycles",
            "systemic_provider_failure_cycles",
            "cont_reported_tokens",
            "graphiti_reported_tokens",
            "cycle_ids",
            "cycle_outcome_classes",
            "cooldown_seconds_values",
            "next_cycle_eligible_at_values",
        )
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        start_utc = start.astimezone(UTC)
        end_utc = end.astimezone(UTC)
        epoch_seconds = int(start_utc.timestamp())
        cursor = datetime.fromtimestamp(
            epoch_seconds - (epoch_seconds % bucket_seconds), tz=UTC
        )
        while cursor < end_utc:
            boundary = cursor + timedelta(seconds=bucket_seconds)

            def in_bucket(
                value: object,
                bucket_start: datetime = cursor,
                bucket_end: datetime = boundary,
            ) -> bool:
                return (
                    isinstance(value, str)
                    and bucket_start <= _instant(value) < bucket_end
                )

            terminal = [row for row in leaves if in_bucket(row.get("completed_at"))]
            grok = [row for row in terminal if row.get("provider") == "grok-build-cli"]
            cursor_rows = [
                row for row in terminal if row.get("provider") == "cursor-agent-cli"
            ]
            dispatched_grok = [
                row
                for row in leaves
                if row.get("provider") == "grok-build-cli"
                and row.get("actual_provider_dispatch") is True
                and in_bucket(row.get("dispatch_at"))
            ]
            outcomes: dict[str, dict[str, object]] = {}
            for row in leaves:
                if in_bucket(row.get("work_outcome_terminal_at")):
                    outcomes[str(row["envelope_id"])] = row
            accepted = [
                row
                for row in outcomes.values()
                if row.get("work_outcome") == "ACCEPTED"
                and row.get("payload_digest")
            ]
            graphiti_successes = [
                row
                for row in outcomes.values()
                if row.get("work_outcome") in _GRAPHITI_COMPLETED_USEFUL_OUTCOMES
            ]
            accounted = [
                row
                for row in terminal
                if row.get("usage_status") in {"REPORTED", "ESTIMATED"}
            ]
            accepted_providers = Counter(
                str(row.get("provider") or "other")
                for accepted_row in accepted
                for row in leaves
                if row.get("envelope_id") == accepted_row.get("envelope_id")
                and row.get("provider_attempt_id")
                == accepted_row.get("accepted_provider_attempt_id")
            )
            bucket_cycles = [
                row for row in cycles if in_bucket(row.get("terminal_at"))
            ]
            reported = [
                row for row in terminal if row.get("usage_status") == "REPORTED"
            ]

            def token_sum(rows: list[dict[str, object]], field: str) -> int:
                return sum(
                    int(value)
                    for row in rows
                    if _is_int(value := row.get(field))
                    and row.get("usage_status") in {"REPORTED", "ESTIMATED"}
                )

            def json_column(values: list[object]) -> str:
                return json.dumps(
                    values, ensure_ascii=False, separators=(",", ":")
                )

            def productive_as_of_bucket(
                row: dict[str, object], *, bucket_boundary: datetime = boundary
            ) -> bool:
                outcome_at = row.get("work_outcome_terminal_at")
                if (
                    not isinstance(outcome_at, str)
                    or _instant(outcome_at) >= bucket_boundary
                ):
                    return False
                return bool(
                    bool(row.get("accepted_provider_attempt_id"))
                    and row.get("provider_attempt_id")
                    == row.get("accepted_provider_attempt_id")
                    or row.get("work_outcome")
                    in _GRAPHITI_COMPLETED_USEFUL_OUTCOMES
                )

            writer.writerow(
                {
                    "bucket_start_utc": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "bucket_end_utc": boundary.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "cycle_results": sum(
                        in_bucket(row.get("terminal_at")) for row in cycles
                    ),
                    "minted_reported": len(accepted),
                    "graphiti_successes_reported": len(graphiti_successes),
                    "grok_writer_sessions": sum(
                        str(row.get("workload_class", "")).startswith(
                            "CONT_WRITER_"
                        )
                        for row in grok
                    ),
                    "grok_completed_sessions": len(grok),
                    "grok_model_calls": len(dispatched_grok),
                    "grok_input_tokens": token_sum(grok, "input_tokens"),
                    "grok_output_tokens": token_sum(grok, "output_tokens"),
                    "grok_total_tokens": token_sum(grok, "total_tokens"),
                    "grok_cached_read_tokens": token_sum(
                        grok, "cached_read_tokens"
                    ),
                    "grok_reasoning_tokens": token_sum(grok, "reasoning_tokens"),
                    "cursor_fallback_sessions": len(cursor_rows),
                    "stored_outputs": len(accepted),
                    "stored_grok_outputs": accepted_providers["grok-build-cli"],
                    "stored_cursor_outputs": accepted_providers["cursor-agent-cli"],
                    "stored_other_outputs": len(accepted)
                    - accepted_providers["grok-build-cli"]
                    - accepted_providers["cursor-agent-cli"],
                    "reported_tokens": token_sum(
                        [
                            row
                            for row in terminal
                            if row.get("usage_status") == "REPORTED"
                        ],
                        "total_tokens",
                    ),
                    "estimated_tokens": token_sum(
                        [
                            row
                            for row in terminal
                            if row.get("usage_status") == "ESTIMATED"
                        ],
                        "total_tokens",
                    ),
                    "unresolved_invocations": sum(
                        row.get("usage_status")
                        in {"UNREPORTED", "AMBIGUOUS", "INVALID"}
                        for row in terminal
                    ),
                    "productive_tokens": sum(
                        int(row["total_tokens"])
                        for row in accounted
                        if _is_int(row.get("total_tokens"))
                        and productive_as_of_bucket(row)
                    ),
                    "no_result_tokens": sum(
                        int(row["total_tokens"])
                        for row in accounted
                        if _is_int(row.get("total_tokens"))
                        and not productive_as_of_bucket(row)
                    ),
                    "unreported_invocations": sum(
                        row.get("usage_status") == "UNREPORTED" for row in terminal
                    ),
                    "ambiguous_invocations": sum(
                        row.get("usage_status") == "AMBIGUOUS" for row in terminal
                    ),
                    "invalid_invocations": sum(
                        row.get("usage_status") == "INVALID" for row in terminal
                    ),
                    "admission_only_hold": sum(
                        row.get("decision") == "HOLD"
                        and in_bucket(row.get("recorded_at"))
                        for row in admissions
                    ),
                    "admission_only_reject": sum(
                        row.get("decision") == "REJECT"
                        and in_bucket(row.get("recorded_at"))
                        for row in admissions
                    ),
                    "idle_qualified_zero_cycles": sum(
                        row.get("outcome_class") == "IDLE_QUALIFIED_ZERO"
                        for row in bucket_cycles
                    ),
                    "productive_cycles": sum(
                        row.get("outcome_class") == "PRODUCTIVE"
                        for row in bucket_cycles
                    ),
                    "unproductive_provider_cycles": sum(
                        row.get("outcome_class") == "UNPRODUCTIVE_PROVIDER"
                        for row in bucket_cycles
                    ),
                    "systemic_provider_failure_cycles": sum(
                        row.get("outcome_class") == "SYSTEMIC_PROVIDER_FAILURE"
                        for row in bucket_cycles
                    ),
                    "cont_reported_tokens": token_sum(
                        [
                            row
                            for row in reported
                            if str(row.get("workload_class") or "").startswith(
                                "CONT_"
                            )
                        ],
                        "total_tokens",
                    ),
                    "graphiti_reported_tokens": token_sum(
                        [
                            row
                            for row in reported
                            if str(row.get("workload_class") or "").startswith(
                                "GRAPHITI_"
                            )
                        ],
                        "total_tokens",
                    ),
                    "cycle_ids": json_column(
                        [row.get("cycle_id") for row in bucket_cycles]
                    ),
                    "cycle_outcome_classes": json_column(
                        [row.get("outcome_class") for row in bucket_cycles]
                    ),
                    "cooldown_seconds_values": json_column(
                        [row.get("cooldown_seconds") for row in bucket_cycles]
                    ),
                    "next_cycle_eligible_at_values": json_column(
                        [
                            row.get("next_cycle_eligible_at")
                            for row in bucket_cycles
                        ]
                    ),
                }
            )
            cursor = boundary
        return output.getvalue()

    def _insert_exact(
        self,
        *,
        table: str,
        identity_column: str,
        identity: str,
        record: Mapping[str, object],
        sql: str,
        values: tuple[object, ...],
        connection: sqlite3.Connection | None = None,
    ) -> None:
        owns_connection = connection is None
        current = self._connection() if connection is None else connection
        try:
            try:
                current.execute(sql, values)
            except sqlite3.IntegrityError as exc:
                row = current.execute(
                    f"SELECT record_json FROM {table} WHERE {identity_column}=?",
                    (identity,),
                ).fetchone()
                if row is None or _object(row[0]) != dict(record):
                    raise ModelUsageIntegrityError(
                        f"conflicting {table} replay"
                    ) from exc
            if owns_connection:
                current.commit()
        finally:
            if owns_connection:
                current.close()


def _json(value: Mapping[str, object]) -> str:
    return canonical_json_bytes(dict(value)).decode("utf-8")


def _object(value: object) -> dict[str, object]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ModelUsageIntegrityError(
            "retained model usage JSON is malformed"
        ) from exc
    if not isinstance(parsed, dict):
        raise ModelUsageIntegrityError("retained model usage JSON is not an object")
    return parsed


def _policy_from_record(record: Mapping[str, object]) -> InvocationEfficiencyPolicy:
    return InvocationEfficiencyPolicy(
        policy_id=str(record["policy_id"]),
        version=str(record["version"]),
        workload_class=WorkloadClass(str(record["workload_class"])),
        provider=str(record["provider"]),
        route=str(record["route"]),
        model=str(record["model"]),
        reasoning=str(record["reasoning"]),
        one_turn=bool(record["one_turn"]),
        exact_input=bool(record["exact_input"]),
        skills_enabled=bool(record["skills_enabled"]),
        tools_enabled=bool(record["tools_enabled"]),
        mcp_enabled=bool(record["mcp_enabled"]),
        prior_message_count=_record_int(record, "prior_message_count"),
        command_semantic_version=str(
            record.get("command_semantic_version", "UNSPECIFIED")
        ),
        command_flags=_record_string_tuple(record, "command_flags", default=()),
        context_manifest_schema_version=str(
            record.get("context_manifest_schema_version", "UNSPECIFIED")
        ),
        disabled_capabilities=_record_string_tuple(
            record, "disabled_capabilities", default=()
        ),
        implementation_revision=str(
            record.get("implementation_revision", "UNSPECIFIED")
        ),
        calibration_only=bool(record.get("calibration_only", False)),
        allowed_candidate_ids=_record_string_tuple(
            record, "allowed_candidate_ids", default=()
        ),
        max_prompt_bytes=_record_int(record, "max_prompt_bytes"),
        max_context_tokens=_record_int(record, "max_context_tokens"),
        max_output_tokens=_record_int(record, "max_output_tokens"),
        max_total_tokens=_record_int(record, "max_total_tokens"),
        prompt_contract_version=str(record["prompt_contract_version"]),
        output_schema_digest=str(record["output_schema_digest"]),
        allowed_context_identities=tuple(
            str(value)
            for value in record["allowed_context_identities"]  # type: ignore[union-attr]
        ),
        allowed_config_identities=tuple(
            str(value)
            for value in record["allowed_config_identities"]  # type: ignore[union-attr]
        ),
        hard_estimate_ceiling_tokens=(
            None
            if record.get("hard_estimate_ceiling_tokens") is None
            else _record_int(record, "hard_estimate_ceiling_tokens")
        ),
        evidence_digest=str(record["evidence_digest"]),
        qualified=bool(record["qualified"]),
        canonical_digest=str(record["canonical_digest"]),
    )


def _record_int(record: Mapping[str, object], field: str) -> int:
    value = record[field]
    if not _is_int(value):
        raise ModelUsageIntegrityError(f"retained policy {field} is invalid")
    return value


def _is_hermetic_cont_policy(policy: InvocationEfficiencyPolicy) -> bool:
    return bool(
        _HERMETIC_CONT_CONFIG_IDENTITIES.intersection(
            policy.allowed_config_identities
        )
    )


def _record_string_tuple(
    record: Mapping[str, object],
    field: str,
    *,
    default: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    value = record.get(field, default)
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ModelUsageIntegrityError(f"retained policy {field} is invalid")
    return tuple(value)


__all__ = [
    "MODEL_USAGE_INTERFACE_SCHEMA_VERSION",
    "MODEL_USAGE_SCHEMA_VERSION",
    "InvocationAllocation",
    "InvocationEfficiencyPolicy",
    "InvocationTerminal",
    "ModelUsageAdmissionError",
    "ModelUsageIntegrityError",
    "ModelUsageService",
    "UsageComponents",
    "UsageStatus",
    "WorkEnvelope",
    "WorkloadClass",
]
