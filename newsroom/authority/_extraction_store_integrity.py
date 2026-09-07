from __future__ import annotations

import sqlite3

from newsroom.authority.persistence import AuthoritySchemaError


_EXTRACTION_TABLES = frozenset(
    {
        "extractor_contracts",
        "extraction_runs",
        "extraction_run_passages",
        "extraction_run_versions",
        "extraction_run_heads",
        "extraction_outputs",
        "extraction_proposal_sets",
        "extraction_proposals",
        "extraction_proposal_evidence",
    }
)


class _ExtractionIntegrityMixin:
    """Re-derive every retained Increment 4A authority record on startup."""

    def _validate_schema_and_integrity(self) -> None:
        super()._validate_schema_and_integrity()
        if not self._should_validate_row_integrity():
            return
        self._validate_extraction_integrity(self._connection)

    def _validate_extraction_integrity(self, conn: sqlite3.Connection) -> None:
        missing = _EXTRACTION_TABLES - self._table_names()
        if missing:
            raise AuthoritySchemaError(
                f"extraction authority schema tables are missing: {sorted(missing)!r}"
            )
        self._validate_all_extraction_rows(conn)
        self._validate_extraction_run_heads(conn)
        self._validate_extraction_event_coverage(conn)
        self._validate_extraction_relational_coverage(conn)

    def _validate_all_extraction_rows(self, conn: sqlite3.Connection) -> None:
        for row in conn.execute(
            "SELECT * FROM extractor_contracts ORDER BY contract_id"
        ).fetchall():
            self._contract_from_row(conn, row, replayed=False)
        for row in conn.execute(
            "SELECT * FROM extraction_run_versions "
            "ORDER BY run_id,version_number"
        ).fetchall():
            # This reconstructs and verifies the stable run, passages, request,
            # retained output, proposal set, every proposal and evidence range.
            # Current rights are deliberately not re-evaluated here: revocation
            # must block later use without making retained history unreadable.
            self._run_version_from_row(conn, row, replayed=False)

    @staticmethod
    def _validate_extraction_run_heads(conn: sqlite3.Connection) -> None:
        missing = conn.execute(
            "SELECT r.run_id FROM extraction_runs r "
            "LEFT JOIN extraction_run_heads h ON h.run_id=r.run_id "
            "WHERE h.run_id IS NULL LIMIT 1"
        ).fetchone()
        if missing is not None:
            raise AuthoritySchemaError(
                "extraction run lacks a retained immutable version head"
            )

        for head in conn.execute(
            "SELECT * FROM extraction_run_heads ORDER BY run_id"
        ).fetchall():
            rows = conn.execute(
                "SELECT run_version_id,version_number,previous_run_version_id,"
                "outcome,recorded_at FROM extraction_run_versions "
                "WHERE run_id=? ORDER BY version_number",
                (str(head["run_id"]),),
            ).fetchall()
            if not rows:
                raise AuthoritySchemaError(
                    "extraction run head has no retained run versions"
                )
            previous: str | None = None
            terminal = False
            for expected, row in enumerate(rows, start=1):
                predecessor = (
                    None
                    if row["previous_run_version_id"] is None
                    else str(row["previous_run_version_id"])
                )
                if (
                    int(row["version_number"]) != expected
                    or predecessor != previous
                    or terminal
                ):
                    raise AuthoritySchemaError(
                        "extraction run version chain is not contiguous"
                    )
                previous = str(row["run_version_id"])
                terminal = str(row["outcome"]) in {
                    "SUCCESS",
                    "BLOCKING_FAILURE",
                    "INVALID_OUTPUT",
                }
            last = rows[-1]
            if (
                int(head["current_version_number"]) != len(rows)
                or str(head["current_run_version_id"])
                != str(last["run_version_id"])
                or int(head["terminal"]) != int(terminal)
                or str(head["updated_at"]) != str(last["recorded_at"])
            ):
                raise AuthoritySchemaError(
                    "extraction run head differs from the immutable version chain"
                )

    @staticmethod
    def _validate_extraction_event_coverage(conn: sqlite3.Connection) -> None:
        specs = (
            ("extraction.contract.registered", "extractor_contracts"),
            ("extraction.run.executed", "extraction_run_versions"),
        )
        for event_type, table in specs:
            missing = conn.execute(
                f"SELECT e.event_id FROM ledger_events e "
                f"LEFT JOIN {table} r ON r.authority_event_id=e.event_id "
                "WHERE e.event_type=? AND r.authority_event_id IS NULL LIMIT 1",
                (event_type,),
            ).fetchone()
            if missing is not None:
                raise AuthoritySchemaError(
                    f"{event_type} has no exact extraction authority record"
                )

    @staticmethod
    def _validate_extraction_relational_coverage(
        conn: sqlite3.Connection,
    ) -> None:
        checks = (
            (
                "extraction run has no exact creating version/event",
                "SELECT r.run_id FROM extraction_runs r "
                "LEFT JOIN extraction_run_versions v "
                "ON v.run_id=r.run_id AND v.version_number=1 "
                "AND v.authority_event_id=r.created_by_event_id "
                "WHERE v.run_version_id IS NULL LIMIT 1",
            ),
            (
                "extraction output has no exact run usage lineage",
                "SELECT o.output_id FROM extraction_outputs o "
                "LEFT JOIN extraction_run_versions v "
                "ON v.run_version_id=o.run_version_id AND v.run_id=o.run_id "
                "AND v.output_bytes=o.byte_length "
                "WHERE v.run_version_id IS NULL LIMIT 1",
            ),
            (
                "proposal set has no exact valid output/run lineage",
                "SELECT s.proposal_set_id FROM extraction_proposal_sets s "
                "LEFT JOIN extraction_outputs o ON o.output_id=s.output_id "
                "AND o.run_id=s.run_id AND o.run_version_id=s.run_version_id "
                "AND o.validation_state='VALID' "
                "LEFT JOIN extraction_run_versions v "
                "ON v.run_version_id=s.run_version_id AND v.run_id=s.run_id "
                "AND v.proposal_count=s.proposal_count "
                "WHERE o.output_id IS NULL OR v.run_version_id IS NULL LIMIT 1",
            ),
            (
                "proposal set count differs from retained proposal rows",
                "SELECT s.proposal_set_id FROM extraction_proposal_sets s "
                "LEFT JOIN extraction_proposals p "
                "ON p.proposal_set_id=s.proposal_set_id "
                "GROUP BY s.proposal_set_id,s.proposal_count "
                "HAVING COUNT(p.proposal_id)!=s.proposal_count LIMIT 1",
            ),
            (
                "proposal has no retained evidence range",
                "SELECT p.proposal_id FROM extraction_proposals p "
                "LEFT JOIN extraction_proposal_evidence e "
                "ON e.proposal_id=p.proposal_id "
                "GROUP BY p.proposal_id HAVING COUNT(e.evidence_ordinal)=0 LIMIT 1",
            ),
            (
                "run usage evidence count differs from retained evidence rows",
                "SELECT v.run_version_id FROM extraction_run_versions v "
                "LEFT JOIN extraction_proposal_evidence e "
                "ON e.run_id=v.run_id "
                "LEFT JOIN extraction_proposals p "
                "ON p.proposal_id=e.proposal_id "
                "AND p.run_version_id=v.run_version_id "
                "GROUP BY v.run_version_id,v.evidence_range_count "
                "HAVING COUNT(p.proposal_id)!=v.evidence_range_count LIMIT 1",
            ),
        )
        for message, query in checks:
            if conn.execute(query).fetchone() is not None:
                raise AuthoritySchemaError(message)


__all__ = ["_ExtractionIntegrityMixin"]
