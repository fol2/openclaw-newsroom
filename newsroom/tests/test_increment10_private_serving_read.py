from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from newsroom.increment10.private_serving import (
    PrivateServingError,
    open_private_serving_read_port,
)
from newsroom.tests.authority_helpers import proof
from newsroom.tests.test_increment10_private_serving import (
    _close,
    _context,
    _delivery,
)

TARGET_ID = "newsroom-app-serving-launch"
TARGET_CONTEXT = "sha256:" + "a" * 64


def test_query_only_reader_exposes_only_exact_retained_ack(tmp_path: Path) -> None:
    context = _context(tmp_path)
    candidate_port, story_receipt, publication_receipt = (
        context[1],
        context[-2],
        context[-1],
    )
    delivery = _delivery(tmp_path, context)
    target = tmp_path / "private-serving.sqlite3"
    attempt_receipt, _ = delivery.begin(
        publication_receipt,
        story_receipt=story_receipt,
        candidate_port=candidate_port,
        proof=proof(),
    )
    rows = delivery.apply(
        attempt_receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=candidate_port,
        applied_at="2026-07-16T11:00:00Z",
        proof=proof(),
    )

    pending = open_private_serving_read_port(
        target,
        target_id=TARGET_ID,
        target_context_digest=TARGET_CONTEXT,
        proof=None,
    )
    assert pending.acknowledged_rows() is None
    assert not any(
        hasattr(pending, name)
        for name in ("apply", "begin", "observe", "record", "query")
    )
    assert pending._connection.execute("PRAGMA query_only").fetchone()[0] == 1
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        pending._connection.execute("DELETE FROM private_serving_payloads")
    pending.close()

    evidence = delivery.observe(
        attempt_receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=candidate_port,
        observed_at="2026-07-16T11:30:00Z",
        proof=proof(),
    )
    evidence_receipt = delivery.record(
        evidence,
        attempt_receipt,
        expected_version=0,
        proof=proof(),
    )
    read_proof = delivery.acknowledged_read_proof(
        evidence_receipt,
        attempt_receipt,
        publication_receipt=publication_receipt,
        story_receipt=story_receipt,
        candidate_port=candidate_port,
        proof=proof(),
    )
    assert read_proof is not None

    reader = open_private_serving_read_port(
        target,
        target_id=TARGET_ID,
        target_context_digest=TARGET_CONTEXT,
        proof=read_proof,
    )
    acknowledged = reader.acknowledged_rows()
    assert acknowledged is not None and acknowledged.rows == rows
    assert reader._connection.total_changes == 0
    reader.close()

    reopened = open_private_serving_read_port(
        target,
        target_id=TARGET_ID,
        target_context_digest=TARGET_CONTEXT,
        proof=read_proof,
    )
    assert reopened.acknowledged_rows() == acknowledged
    reopened.close()

    delivery._connection.execute(
        "UPDATE private_serving_payloads SET payload_bytes=? WHERE operation_key=?",
        (b"{}", rows[0].operation_key),
    )
    corrupt = open_private_serving_read_port(
        target,
        target_id=TARGET_ID,
        target_context_digest=TARGET_CONTEXT,
        proof=read_proof,
    )
    with pytest.raises(PrivateServingError, match="payload differs"):
        corrupt.acknowledged_rows()
    corrupt.close()
    _close(context, delivery)
