from pathlib import Path

import pytest

from newsroom.authority import ObjectAdmissionRequest
from newsroom.authority.persistence import AuthorityWriterBusy
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.increment5.retrieval_context import RetrievalContextJournal
from newsroom.increment6.collision import (
    CurrentCollisionEffectEnforcer, TrustedCurrentCollisionAuthorityBoundary,
)
from newsroom.increment6.work_items import RetrievalContextAuthority
from newsroom.tests.authority_helpers import FIXED_NOW
from newsroom.tests.discovery_3d_authority_helpers import seed_check_lineage
from newsroom.tests.increment5b2_helpers import config
from newsroom.tests.projection_b2_helpers import MemoryNeo4jAdapter


def _args(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "newsroom.authority._graphiti_increment4_system._open_structural_graph_adapter",
        lambda _: MemoryNeo4jAdapter(),
    )
    journal = RetrievalContextJournal(tmp_path / "retrieval.sqlite3")
    return dict(
        authority_path=tmp_path / "authority.sqlite3",
        object_root=tmp_path / "objects", workspace_root=tmp_path,
        intake_path=tmp_path / "intake.sqlite3", target_path=tmp_path / "serving.sqlite3",
        target_id="hermes-private-serving", credential="token-1",
        principal_id="principal.alpha", authority_domain="newsroom.authority",
        neo4j_config=config(), retrieval_authority=RetrievalContextAuthority(journal.path, {}),
        collision_enforcer=CurrentCollisionEffectEnforcer(
            current_authority_provider=lambda _: None,
            trusted_boundary=TrustedCurrentCollisionAuthorityBoundary(
                "fixture-scope", "fixture-profile", "sha256:" + "a" * 64,
                "sha256:" + "b" * 64, "fixture-port",
            ),
        ), clock=lambda: FIXED_NOW,
    )


def test_native_runtime_real_policy_composition_and_reopen(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        seed_check_lineage(runtime.authority)
        source = runtime.authority.objects.admit(
            ObjectAdmissionRequest("evidence.source", "source-1"), b"source bytes",
            proof=runtime.proof,
        ).admission
        assert source.allowed_use == "publication_evidence"
        assert runtime.ingress.receipt_count == 0
        assert runtime.publication is not None
        with pytest.raises(AuthorityWriterBusy):
            open_native_runtime(**args)
    with open_native_runtime(**args) as reopened:
        same = reopened.authority.objects.admit(
            ObjectAdmissionRequest("evidence.source", "source-1"), b"source bytes",
            proof=reopened.proof,
        ).admission
        assert same == source
        assert reopened.ingress.receipt_count == 0
    # No target row, provider invocation or fake retrieval success was created.


def test_native_runtime_rejects_overlapping_store_identity_before_open(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    args["target_path"] = args["authority_path"]
    with pytest.raises(ValueError, match="must be distinct"):
        open_native_runtime(**args)
    assert not args["authority_path"].exists()
