from __future__ import annotations

import json
import sqlite3
from contextlib import nullcontext
from dataclasses import replace

import pytest

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.control_plane.native_collision import (
    NativeCollisionAuthority,
    NativeCollisionIdentity,
)
from newsroom.control_plane.native_cycle import advance_native_cycle
from newsroom.increment6.collision import (
    CurrentCollisionEligibilityBlocked,
    CurrentCollisionEligibilityRequest,
)
from newsroom.tests.discovery_3d_authority_helpers import (
    exact_admission_request,
    proof,
    seed_check_lineage,
)
from newsroom.tests.test_native_triage import (
    _actor_digest,
    _no_match_retrieval,
    _shared_system,
)


def _native_collision(tmp_path, retrieval) -> NativeCollisionAuthority:
    receipt = json.loads(retrieval.receipt_bytes)
    authority = receipt["authority_evidence"]
    return NativeCollisionAuthority(
        authority_path=tmp_path / "native.sqlite3",
        journal_path=tmp_path / "native-collision.sqlite3",
        identity=NativeCollisionIdentity(
            authority["authority_scope_id"],
            receipt["actor_id"],
            receipt["authenticated_principal_digest"],
            "sha256:" + "c" * 64,
            "00000000-0000-4000-8000-000000006201",
            authority["adapter_contract_digest"],
            authority["adapter_config_digest"],
        ),
    )


def test_native_collision_reads_current_candidate_and_replays_after_restart(
    tmp_path, monkeypatch
) -> None:
    retrieval_authority, retrieval = _no_match_retrieval(tmp_path)
    collision = _native_collision(tmp_path, retrieval)
    with _shared_system(
        tmp_path, monkeypatch, retrieval_authority, collision=collision.enforcer
    ) as system:
        seed_check_lineage(system)
        admitted = system.discovery.admit_signal_to_lead(
            exact_admission_request(), proof=proof()
        )
        status = system.discovery.current_status(
            admitted.lead.request.signal_id, proof=proof()
        )
        outcome = advance_native_cycle(
            system,
            (status,),
            retrieval=type("Retrieval", (), {
                "retrieve": lambda self, lead, *, proof: retrieval
            })(),
            collision_requests=collision,
            actor_identity_digest=_actor_digest(),
            proof=proof(),
            owner_stop_check=lambda: None,
            owner_stop_fence=nullcontext,
        )[0]
        assert outcome.state == "CANDIDATE_ADMITTED"
        assert outcome.triage is not None
        candidate = outcome.triage.candidate
        assert candidate is not None

    reopened_collision = _native_collision(tmp_path, retrieval)
    with _shared_system(
        tmp_path, monkeypatch, retrieval_authority,
        collision=reopened_collision.enforcer,
    ) as reopened:
        replay = advance_native_cycle(
            reopened,
            (status,),
            retrieval=type("Retrieval", (), {
                "retrieve": lambda self, lead, *, proof: retrieval
            })(),
            collision_requests=reopened_collision,
            actor_identity_digest=_actor_digest(),
            proof=proof(),
            owner_stop_check=lambda: None,
            owner_stop_fence=nullcontext,
        )[0]
        assert replay.state == "CANDIDATE_ADMITTED"
        assert replay.triage is not None
        assert replay.triage.candidate == candidate

    with sqlite3.connect(tmp_path / "native-collision.sqlite3") as journal:
        states = {
            row[0] for row in journal.execute(
                "SELECT collision_state FROM native_collision_receipts"
            )
        }
    assert states == {"UNOCCUPIED", "OCCUPIED"}


def test_native_collision_rechecks_current_watermark(tmp_path, monkeypatch) -> None:
    retrieval_authority, retrieval = _no_match_retrieval(tmp_path)
    collision = _native_collision(tmp_path, retrieval)
    with _shared_system(
        tmp_path, monkeypatch, retrieval_authority, collision=collision.enforcer
    ) as system:
        seed_check_lineage(system)
        admitted = system.discovery.admit_signal_to_lead(
            exact_admission_request(), proof=proof()
        )
        status = system.discovery.current_status(
            admitted.lead.request.signal_id, proof=proof()
        )
        held = advance_native_cycle(
            system,
            (status,),
            retrieval=type("Retrieval", (), {
                "retrieve": lambda self, lead, *, proof: retrieval
            })(),
            collision_requests=type("Hold", (), {
                "request": lambda self, triage, retrieval, *, proof: None
            })(),
            actor_identity_digest=_actor_digest(),
            proof=proof(),
            owner_stop_check=lambda: None,
            owner_stop_fence=nullcontext,
        )[0]
        request = collision.request(held.triage, retrieval, proof=proof())
        old = replace(
            request.binding,
            authority_watermark=request.binding.authority_watermark - 1,
        )
        old_digest = digest_bytes(canonical_json_bytes({
            "schema_version": "newsroom.increment6.native-collision-request.v1",
            "actor_id": "triage_worker",
            "authenticated_principal_digest": json.loads(
                retrieval.receipt_bytes
            )["authenticated_principal_digest"],
            "binding": old.canonical_value(),
        }))
        stale = CurrentCollisionEligibilityRequest(old, old_digest)
        with pytest.raises(CurrentCollisionEligibilityBlocked) as exc_info:
            collision.enforcer.enforce(request=stale, effect=lambda _: None)
        assert exc_info.value.decision.outcome.value == "BINDING_MISMATCH"
