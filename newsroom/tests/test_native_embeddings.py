import io
import json
import sqlite3
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime

import pytest

from newsroom.authority.canonical import digest_canonical
from newsroom.control_plane import native_embeddings as embedding
from newsroom.control_plane.model_usage import InvocationEfficiencyPolicy, ModelUsageService, WorkloadClass
from newsroom.control_plane.native_runtime import open_native_runtime
from newsroom.increment5.native_retrieval import NativeRetrievalHold
from newsroom.tests.test_native_runtime import _args

NOW = datetime(2026, 9, 8, 14, tzinfo=UTC)


def _policy():
    return InvocationEfficiencyPolicy.create(
        policy_id="test-native-embedding", version="v1", workload_class=WorkloadClass.NATIVE_RETRIEVAL_EMBEDDING,
        provider="openrouter", route=embedding.ROUTE, model=embedding.OPENROUTER_EMBEDDING_SLUG,
        reasoning="none", one_turn=True, exact_input=True, skills_enabled=False,
        tools_enabled=False, mcp_enabled=False, prior_message_count=0,
        command_semantic_version=embedding.VERSION, command_flags=("POST=/embeddings",),
        context_manifest_schema_version=embedding.VERSION, disabled_capabilities=("tools",),
        implementation_revision=embedding.implementation_digest(), max_prompt_bytes=20_000,
        max_context_tokens=8_000, max_output_tokens=1, max_total_tokens=8_000,
        prompt_contract_version=embedding.VERSION, output_schema_digest=embedding.SCHEMA_DIGEST,
        allowed_context_identities=(embedding.VERSION,), allowed_config_identities=(embedding.VERSION,),
        hard_estimate_ceiling_tokens=None, evidence_digest=digest_canonical({"test-only": "bounded route"}), qualified=True,
    )


def _response():
    return {"id": "provider-request-1", "object": "list", "model": embedding.OPENROUTER_EMBEDDING_SLUG,
            "data": [{"index": 0, "embedding": [0.25] * 1024}],
            "usage": {"prompt_tokens": 4, "total_tokens": 4, "cost": 0.00001}}


@pytest.mark.parametrize("case", ["complete", "bad_vector", "missing_usage", "transport_failed", "signed_stop"])
def test_one_accounted_native_embedding_with_real_sqlite_and_governed_objects(tmp_path, monkeypatch, case):
    args = _args(tmp_path, monkeypatch)
    usage_path = str(tmp_path / "usage.sqlite3")
    service = ModelUsageService(usage_path)
    policy = _policy()
    service.register_policy(policy)
    calls = []
    value = _response()
    if case == "bad_vector": value["data"][0]["embedding"] = [0.25]
    if case == "missing_usage": value.pop("usage")
    class Response(io.BytesIO):
        status = 200
        def geturl(self): return embedding.URL
    class Opener:
        def open(self, request, timeout):
            calls.append((request.full_url, request.data, timeout))
            if case == "transport_failed": raise OSError("transport unavailable")
            return Response(json.dumps(value).encode())
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    @contextmanager
    def fence():
        if case == "signed_stop": raise RuntimeError("signed owner stop")
        yield
    with open_native_runtime(**args) as runtime:
        engine = embedding.NativePassageEmbedder(
            api_key="test-key-never-live", objects=runtime.authority.objects, usage=service,
            policy=policy, dispatch_fence=fence, implementation_worktree_clean=True, clock=lambda: NOW,
        )
        params = dict(text="Exact source passage.", passage_id="actual-passage-id", cycle_id="native-cycle-1", proof=runtime.proof)
        if case == "complete":
            reference = engine.retain(**params)
            assert reference.vector_admission_id != reference.receipt_admission_id
        else:
            with pytest.raises(NativeRetrievalHold, match="RESULT_HOLD"):
                engine.retain(**params)
    with sqlite3.connect(usage_path) as database:
        assert database.execute("SELECT COUNT(*) FROM model_invocation_allocations").fetchone()[0] == 1
        raw = database.execute("SELECT record_json FROM model_invocation_terminals").fetchone()[0]
        terminal = json.loads(raw)
        if case == "signed_stop":
            assert terminal["pre_dispatch_zero_proved"] is True
            assert terminal["components"]["total_tokens"] == 0
            assert not calls
        else:
            assert len(calls) == 1
            request = json.loads(calls[0][1])
            assert request == {"input": "Exact source passage.", "model": embedding.OPENROUTER_EMBEDDING_SLUG,
                               "dimensions": 1024, "encoding_format": "float"}
            assert "test-key" not in raw
            assert terminal["components"]["total_tokens"] == (4 if case in {"complete", "bad_vector"} else None)
            assert terminal["dispatch_at"] is not None
        assert terminal["od_011_reference"] == "OD-011:NATIVE_RETRIEVAL_EMBEDDING"
        assert terminal["policy_breach"] is None
        assert terminal["usage_status"] == ("UNREPORTED" if case in {"missing_usage", "transport_failed"} else "REPORTED")


def test_native_embedding_requires_exact_qualified_implementation_before_effects(tmp_path, monkeypatch):
    from dataclasses import replace
    args = _args(tmp_path, monkeypatch)
    with open_native_runtime(**args) as runtime:
        with pytest.raises(NativeRetrievalHold, match="POLICY_HOLD"):
            embedding.NativePassageEmbedder(api_key="test-key", objects=runtime.authority.objects,
                usage=ModelUsageService(str(tmp_path / "usage.sqlite3")),
                policy=replace(_policy(), implementation_revision="different-code"),
                dispatch_fence=nullcontext, implementation_worktree_clean=True)
