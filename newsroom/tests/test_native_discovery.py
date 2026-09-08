from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from newsroom.authority import UtcTimestamp
from newsroom.checks import TriggerKind, ObservableTransitionKind
from newsroom.control_plane.native_discovery import NativeDiscovery
from newsroom.control_plane.graphiti_operational_readiness import _source_requests
from newsroom.discovery import GateOutcome
from newsroom.tests.check_3c_authority_helpers import proof
from newsroom.tests.discovery_3d_authority_helpers import open_discovery_system
from newsroom.tests.test_graphiti_operational_readiness import _unit, _rights, _next_revision

NOW = UtcTimestamp.parse("2026-09-02T12:02:00.000000Z")
LATER = UtcTimestamp.parse("2026-09-02T12:03:00.000000Z")


def _seed(system, unit, prior=None):
    requests = _source_requests(unit, _rights(), prior_revision_id=prior)
    methods = (
        system.sources.register_definition, system.sources.record_definition_version,
        system.sources.register_item, system.sources.record_revision,
        system.sources.record_representation,
    )
    for method, request in zip(methods, requests, strict=True):
        method(request, proof=proof())


def _controller(system, proving):
    return NativeDiscovery(
        sources=system.sources, checks=system.checks, discovery=system.discovery,
        proving=proving,
    )


def test_retained_revision_enters_native_discovery_and_replays_after_reopen(tmp_path, monkeypatch):
    database = tmp_path / "authority.sqlite3"
    unit = _unit()
    monkeypatch.setattr(
        "newsroom.control_plane.cycle._dispatch_rights_decision",
        lambda *args, **kwargs: _rights(),
    )
    with sqlite3.connect(":memory:") as proving:
        with open_discovery_system(database, clock=lambda: NOW) as system:
            _seed(system, unit)
            controller = _controller(system, proving)
            delivered = controller.deliver(unit, now=NOW, proof=proof())
            assert system.checks.request(delivered.outcome.request.request_id, proof=proof()).request.trigger.kind is TriggerKind.DELIVERED_INPUT
            assert delivered.transition.request.kind is ObservableTransitionKind.FIRST_OBSERVED
            assert delivered.outcome.request.completed_at == NOW
            assert system.sources.revision(delivered.transition.request.current_revision_id, proof=proof()).request.observed_at != NOW
            status = controller.admit_lead(delivered, now=NOW, proof=proof())
            assert status.current_gate.request.outcome is GateOutcome.PROMOTED_TO_LEAD
            assert status.lead is not None
            lead_id = status.lead.request.lead_id
        with sqlite3.connect(database) as connection:
            counts = tuple(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("ledger_events", "discovery_occurrences", "check_outcomes", "news_leads"))
        with open_discovery_system(database, clock=lambda: LATER) as system:
            controller = _controller(system, proving)
            resumed = controller.deliver(unit, now=LATER, proof=proof())
            assert resumed.outcome.request.completed_at == NOW
            assert controller.admit_lead(resumed, now=LATER, proof=proof()).lead.request.lead_id == lead_id
        with sqlite3.connect(database) as connection:
            assert tuple(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("ledger_events", "discovery_occurrences", "check_outcomes", "news_leads")) == counts


def test_missing_current_rights_hold_and_recovery_is_automatic(tmp_path, monkeypatch):
    rights = [None]
    monkeypatch.setattr("newsroom.control_plane.cycle._dispatch_rights_decision", lambda *args, **kwargs: rights[0])
    with sqlite3.connect(":memory:") as proving, open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: NOW) as system:
        unit = _unit()
        _seed(system, unit)
        controller = _controller(system, proving)
        delivered = controller.deliver(unit, now=NOW, proof=proof())
        held = controller.admit_lead(delivered, now=NOW, proof=proof())
        assert held.current_gate.request.outcome is GateOutcome.OPERATIONAL_HOLD
        assert held.lead is None
        rights[0] = _rights()
        ready = controller.admit_lead(delivered, now=LATER, proof=proof())
        assert ready.current_gate.request.decision_ordinal == 2
        assert ready.lead is not None
        rights[0] = None
        held_again = controller.admit_lead(delivered, now=LATER, proof=proof())
        assert held_again.current_gate.request.outcome is GateOutcome.OPERATIONAL_HOLD
        assert held_again.lead is None


def test_tampered_delivered_fields_fail_before_check_writes(tmp_path):
    with sqlite3.connect(":memory:") as proving, open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: NOW) as system:
        unit = _unit()
        _seed(system, unit)
        with pytest.raises(ValueError, match="exact retained"):
            _controller(system, proving).deliver(replace(unit, body="tampered"), now=NOW, proof=proof())
    with sqlite3.connect(tmp_path / "authority.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM check_requests").fetchone()[0] == 0


def test_changed_revision_retains_predecessor_and_other_items_continue(tmp_path, monkeypatch):
    monkeypatch.setattr("newsroom.control_plane.cycle._dispatch_rights_decision", lambda *args, **kwargs: _rights())
    later = UtcTimestamp.parse("2026-09-02T12:40:00.000000Z")
    with sqlite3.connect(":memory:") as proving, open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: later) as system:
        unit = _unit()
        _seed(system, unit)
        controller = _controller(system, proving)
        first = controller.deliver(unit, now=NOW, proof=proof())
        second = _next_revision(unit)
        _seed(system, second, first.transition.request.current_revision_id)
        changed = controller.deliver(second, now=later, proof=proof())
        assert changed.transition.request.kind is ObservableTransitionKind.REVISED
        assert changed.transition.request.prior_revision_id == first.transition.request.current_revision_id
        other = _unit(item_key="unrelated")
        _seed(system, other)
        fresh = controller.deliver(other, now=later, proof=proof())
        assert controller.admit_lead(fresh, now=later, proof=proof()).lead is not None
        assert controller.admit_lead(changed, now=later, proof=proof()).lead is not None


def test_parser_reobservation_is_suppressed_and_old_delivery_stays_replayable(tmp_path, monkeypatch):
    from newsroom.sources import DiscoveryRepresentationId
    monkeypatch.setattr("newsroom.control_plane.cycle._dispatch_rights_decision", lambda *args, **kwargs: _rights())
    with sqlite3.connect(":memory:") as proving, open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: NOW) as system:
        unit = _unit()
        _seed(system, unit)
        controller = _controller(system, proving)
        first = controller.deliver(unit, now=NOW, proof=proof())
        controller.admit_lead(first, now=NOW, proof=proof())
        request = _source_requests(unit, _rights())[-1]
        reparsed = replace(
            request, representation_id=DiscoveryRepresentationId.new(),
            parser_version="retained-parser-v2", idempotency_key="reparsed-representation",
        )
        system.sources.record_representation(reparsed, proof=proof())
        new_unit = replace(unit, authority=replace(unit.authority, representation_id=str(reparsed.representation_id)))
        second = controller.deliver(new_unit, now=LATER, proof=proof())
        assert second.transition.request.kind is ObservableTransitionKind.REOBSERVED
        assert controller.admit_lead(second, now=LATER, proof=proof()).current_gate.request.outcome is GateOutcome.SUPPRESSED_NON_CHANGE
        assert controller.deliver(unit, now=LATER, proof=proof()).outcome.request == first.outcome.request
    with sqlite3.connect(tmp_path / "authority.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM news_leads").fetchone()[0] == 1


def test_delivered_history_does_not_become_fresh_from_native_check_time(tmp_path, monkeypatch):
    monkeypatch.setattr("newsroom.control_plane.cycle._dispatch_rights_decision", lambda *args, **kwargs: _rights())
    old = UtcTimestamp.parse("2026-09-12T12:00:00.000000Z")
    with sqlite3.connect(":memory:") as proving, open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: old) as system:
        unit = _unit()
        _seed(system, unit)
        controller = _controller(system, proving)
        delivered = controller.deliver(unit, now=old, proof=proof())
        status = controller.admit_lead(delivered, now=old, proof=proof())
        assert status.lead is None
        assert status.current_gate.request.basis.time_validity.value == "STALE"


def test_native_rights_lookup_does_not_mint_or_consume_beta_fixture_packets(tmp_path, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("legacy fixture rights route was invoked")
    monkeypatch.setattr("newsroom.control_plane.cycle._dispatch_rights_decision", forbid)
    with sqlite3.connect(":memory:") as proving, open_discovery_system(tmp_path / "authority.sqlite3", clock=lambda: NOW) as system:
        unit = _unit()
        _seed(system, unit)
        calls = []
        def current_rights(source_id, locator, now):
            calls.append((source_id, locator, now))
            return {"packet_digest": "sha256:" + "a" * 64}
        controller = NativeDiscovery(sources=system.sources, checks=system.checks,
                                     discovery=system.discovery, proving=proving,
                                     rights_for=current_rights)
        status = controller.admit_lead(controller.deliver(unit, now=NOW, proof=proof()), now=NOW, proof=proof())
        assert status.lead is not None
        assert calls == [(unit.source_id, unit.source_definition_url, NOW)]
        assert proving.execute("SELECT name FROM sqlite_master").fetchall() == []
