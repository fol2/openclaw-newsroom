from dataclasses import replace

import pytest

from newsroom.authority.policy import PayloadSchemaValidationError
from newsroom.control_plane.native_policies import native_policy_components


def test_private_policy_bindings_are_derived_and_have_no_source_or_public_grant(tmp_path):
    target = tmp_path / "serving.sqlite3"
    args = dict(principal_id="newsroom.hermes", authority_domain="newsroom.authority", target_path=target, target_id="hermes-private-serving")
    policies = native_policy_components(**args)
    same = native_policy_components(**args)
    assert policies.publication == same.publication
    assert not target.exists()
    assert len(policies.registry.definitions()) == 6
    assert len(policies.admission_registry.definitions()) == 13
    assert not any("fixture" in definition.definition_version for definition in policies.registry.definitions())
    assert policies.required_scopes.isdisjoint({"authority.objects.manage", "authority.sources.manage", "authority.graphiti.execute"})
    hyd = {contract.purpose: contract for contract in policies.hydration_policies.contracts()}
    assert hyd["evidence.source"].allowed_uses == frozenset({"publication_evidence"})
    assert hyd["evidence.source"].allowed_principal_ids == frozenset({"newsroom.hermes"})
    assert "proposal.extraction" not in hyd["evidence.source"].allowed_uses
    moved = native_policy_components(**{**args, "target_path": tmp_path / "other.sqlite3"})
    assert moved.publication.target_context_digest != policies.publication.target_context_digest
    assert moved.publication.target_policy_digest == policies.publication.target_policy_digest
    renamed = native_policy_components(**{**args, "target_id": "other-private-serving"})
    assert renamed.publication.target_policy_digest != policies.publication.target_policy_digest
    contract = policies.payload_schemas.contracts()[0]
    with pytest.raises(PayloadSchemaValidationError):
        contract.canonicalize({"status": "PASS"})
