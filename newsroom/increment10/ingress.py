"""Durable, non-public Evidence Intake receipt for admitted Candidate Versions."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar, Self

from newsroom.authority.canonical import canonical_json_bytes, digest_bytes
from newsroom.increment6.candidates import (
    CandidateContractError,
    StoryCandidateReadPort,
    StoryCandidateVersion,
)

NON_PUBLIC_EVIDENCE_INTAKE_BOUNDARY = "evidence-intake:non-public:v1"
_ACKNOWLEDGEMENT_SCHEMA = "newsroom.increment10.evidence-intake-acknowledgement.v1"
_APPLICATION_ID = 0x4E524549
_SCHEMA_VERSION = 1
_DIGEST_PREFIX = "sha256:"


class EvidenceIntakeError(ValueError):
    """Raised when Evidence Intake cannot prove an exact durable receipt."""


def _identity(prefix: str, value: object) -> str:
    return f"{prefix}:sha256:{sha256(canonical_json_bytes(value)).hexdigest()}"


def _text(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EvidenceIntakeError(f"invalid {field}")
    return value


def _digest(value: object, field: str) -> str:
    text = _text(value, field)
    if len(text) != 71 or not text.startswith(_DIGEST_PREFIX):
        raise EvidenceIntakeError(f"invalid {field}")
    try:
        int(text[7:], 16)
    except ValueError as exc:
        raise EvidenceIntakeError(f"invalid {field}") from exc
    return text


@dataclass(frozen=True, slots=True)
class IntakeAcknowledgement:
    """Receipt for pinned admitted discovery input, with no effect authority."""

    schema_identity: ClassVar[str] = _ACKNOWLEDGEMENT_SCHEMA
    acknowledgement_id: str
    request_id: str
    handoff_id: str
    receipt_id: str
    candidate_version_id: str
    candidate_version_digest: str
    governing_manifest_digest: str
    boundary_id: str
    received_epoch_seconds: int

    def __post_init__(self) -> None:
        for name in (
            "acknowledgement_id",
            "request_id",
            "handoff_id",
            "receipt_id",
            "candidate_version_id",
            "boundary_id",
        ):
            _text(getattr(self, name), name)
        _digest(self.candidate_version_digest, "candidate_version_digest")
        _digest(self.governing_manifest_digest, "governing_manifest_digest")
        if self.boundary_id != NON_PUBLIC_EVIDENCE_INTAKE_BOUNDARY:
            raise EvidenceIntakeError(
                "acknowledgement is outside the non-public boundary"
            )
        if (
            type(self.received_epoch_seconds) is not int
            or self.received_epoch_seconds < 0
        ):
            raise EvidenceIntakeError("invalid received_epoch_seconds")
        expected = _identity(
            "intake-acknowledgement",
            {
                "request_id": self.request_id,
                "handoff_id": self.handoff_id,
                "receipt_id": self.receipt_id,
                "received_epoch_seconds": self.received_epoch_seconds,
            },
        )
        if self.acknowledgement_id != expected:
            raise EvidenceIntakeError("acknowledgement identity differs")

    @property
    def receipt_only(self) -> bool:
        return True

    @property
    def evidence_authority(self) -> bool:
        return False

    @property
    def publication_authority(self) -> bool:
        return False

    @property
    def runtime_authority(self) -> bool:
        return False

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {"schema_identity": self.schema_identity, **asdict(self)}
        )

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> Self:
        fields = set(cls.__dataclass_fields__)
        try:
            pairs: list[tuple[str, Any]] = json.loads(
                raw, object_pairs_hook=lambda value: value
            )
            if not isinstance(pairs, list) or any(
                not isinstance(item, tuple) for item in pairs
            ):
                raise EvidenceIntakeError("acknowledgement must be an object")
            value = dict(pairs)
            if len(value) != len(pairs) or set(value) != fields | {"schema_identity"}:
                raise EvidenceIntakeError("acknowledgement fields differ")
            if value.pop("schema_identity") != cls.schema_identity:
                raise EvidenceIntakeError("acknowledgement schema differs")
            result = cls(**value)
            if result.canonical_bytes != raw:
                raise EvidenceIntakeError("acknowledgement is non-canonical")
            return result
        except EvidenceIntakeError:
            raise
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvidenceIntakeError("acknowledgement is malformed") from exc


_SCHEMA = """
CREATE TABLE evidence_intake_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    boundary_id TEXT NOT NULL
) STRICT;
CREATE TABLE evidence_intake_handoffs (
    handoff_id TEXT PRIMARY KEY,
    candidate_version_id TEXT NOT NULL UNIQUE,
    candidate_version_bytes BLOB NOT NULL,
    candidate_version_digest TEXT NOT NULL,
    governing_manifest_digest TEXT NOT NULL,
    boundary_id TEXT NOT NULL,
    receipt_id TEXT NOT NULL UNIQUE,
    canonical_bytes BLOB NOT NULL,
    canonical_digest TEXT NOT NULL
) STRICT;
CREATE TABLE evidence_intake_acknowledgements (
    acknowledgement_id TEXT PRIMARY KEY,
    handoff_id TEXT NOT NULL UNIQUE REFERENCES evidence_intake_handoffs(handoff_id),
    canonical_bytes BLOB NOT NULL,
    canonical_digest TEXT NOT NULL
) STRICT;
CREATE TABLE evidence_intake_attempts (
    request_id TEXT PRIMARY KEY,
    handoff_id TEXT NOT NULL REFERENCES evidence_intake_handoffs(handoff_id),
    acknowledgement_id TEXT NOT NULL
        REFERENCES evidence_intake_acknowledgements(acknowledgement_id),
    observed_epoch_seconds INTEGER NOT NULL CHECK (observed_epoch_seconds >= 0)
) STRICT;
"""


class EvidenceIntakeIngress:
    """SQLite receiver bound to one fixed non-public Evidence Intake boundary."""

    __slots__ = ("__connection", "__boundary_id", "__closed")

    def __init__(self, connection: sqlite3.Connection, boundary_id: str) -> None:
        self.__connection = connection
        self.__boundary_id = boundary_id
        self.__closed = False

    @property
    def receipt_count(self) -> int:
        self._require_open()
        return int(
            self.__connection.execute(
                "SELECT count(*) FROM evidence_intake_handoffs"
            ).fetchone()[0]
        )

    @property
    def attempt_count(self) -> int:
        self._require_open()
        return int(
            self.__connection.execute(
                "SELECT count(*) FROM evidence_intake_attempts"
            ).fetchone()[0]
        )

    def receive(
        self,
        candidate_port: StoryCandidateReadPort,
        *,
        candidate_version_id: str,
        expected_governing_manifest_digest: str,
        boundary_id: str,
        request_id: str,
        received_epoch_seconds: int,
    ) -> IntakeAcknowledgement:
        """Pin admitted input and return its receipt.

        Current evidence, publication and runtime eligibility remains separate.
        """

        self._require_open()
        if type(candidate_port) is not StoryCandidateReadPort:
            raise EvidenceIntakeError("authenticated Candidate read port required")
        _text(candidate_version_id, "candidate_version_id")
        expected_manifest = _digest(
            expected_governing_manifest_digest,
            "expected_governing_manifest_digest",
        )
        if boundary_id != self.__boundary_id:
            raise EvidenceIntakeError(
                "request differs from the retained non-public boundary"
            )
        _text(request_id, "request_id")
        if type(received_epoch_seconds) is not int or received_epoch_seconds < 0:
            raise EvidenceIntakeError("invalid received_epoch_seconds")

        try:
            version = candidate_port.require_retained_version_in_transaction(
                candidate_version_id
            )
        except CandidateContractError as exc:
            raise EvidenceIntakeError(
                "Candidate authority verification failed"
            ) from exc
        manifest_digest = version.governing_manifest.canonical_digest
        if manifest_digest != expected_manifest:
            raise EvidenceIntakeError("expected governing manifest differs")
        return self._retain(version, request_id, received_epoch_seconds)

    def receipt(self, receipt_id: str) -> IntakeAcknowledgement:
        self._require_open()
        _text(receipt_id, "receipt_id")
        row = self.__connection.execute(
            "SELECT a.canonical_bytes FROM evidence_intake_acknowledgements a "
            "JOIN evidence_intake_handoffs h ON h.handoff_id=a.handoff_id "
            "WHERE h.receipt_id=?",
            (receipt_id,),
        ).fetchone()
        if row is None:
            raise EvidenceIntakeError("receipt is absent")
        return IntakeAcknowledgement.from_canonical_bytes(bytes(row[0]))

    def close(self) -> None:
        if not self.__closed:
            self.__connection.close()
            self.__closed = True

    def _require_open(self) -> None:
        if self.__closed:
            raise EvidenceIntakeError("Evidence Intake ingress is closed")

    def _retain(
        self,
        version: StoryCandidateVersion,
        request_id: str,
        received_epoch_seconds: int,
    ) -> IntakeAcknowledgement:
        manifest_digest = version.governing_manifest.canonical_digest
        handoff_id = _identity(
            "intake-handoff",
            {
                "candidate_version_id": version.version_id,
                "boundary_id": self.__boundary_id,
            },
        )
        receipt_id = _identity("intake-receipt", {"handoff_id": handoff_id})
        handoff_value = {
            "handoff_id": handoff_id,
            "candidate_version_id": version.version_id,
            "candidate_version_digest": version.canonical_digest,
            "governing_manifest_digest": manifest_digest,
            "boundary_id": self.__boundary_id,
            "receipt_id": receipt_id,
        }
        handoff_bytes = canonical_json_bytes(handoff_value)
        try:
            self.__connection.execute("BEGIN IMMEDIATE")
            prior_request = self.__connection.execute(
                "SELECT handoff_id,acknowledgement_id FROM evidence_intake_attempts "
                "WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if prior_request is not None:
                if str(prior_request[0]) != handoff_id:
                    raise EvidenceIntakeError("request identity conflicts")
                acknowledgement = self._acknowledgement(str(prior_request[1]))
                self.__connection.rollback()
                return acknowledgement

            retained = self.__connection.execute(
                "SELECT canonical_bytes FROM evidence_intake_handoffs "
                "WHERE candidate_version_id=?",
                (version.version_id,),
            ).fetchone()
            if retained is None:
                self.__connection.execute(
                    "INSERT INTO evidence_intake_handoffs VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        handoff_id,
                        version.version_id,
                        version.canonical_bytes,
                        version.canonical_digest,
                        manifest_digest,
                        self.__boundary_id,
                        receipt_id,
                        handoff_bytes,
                        digest_bytes(handoff_bytes),
                    ),
                )
                acknowledgement = IntakeAcknowledgement(
                    acknowledgement_id=_identity(
                        "intake-acknowledgement",
                        {
                            "request_id": request_id,
                            "handoff_id": handoff_id,
                            "receipt_id": receipt_id,
                            "received_epoch_seconds": received_epoch_seconds,
                        },
                    ),
                    request_id=request_id,
                    handoff_id=handoff_id,
                    receipt_id=receipt_id,
                    candidate_version_id=version.version_id,
                    candidate_version_digest=version.canonical_digest,
                    governing_manifest_digest=manifest_digest,
                    boundary_id=self.__boundary_id,
                    received_epoch_seconds=received_epoch_seconds,
                )
                acknowledgement_bytes = acknowledgement.canonical_bytes
                self.__connection.execute(
                    "INSERT INTO evidence_intake_acknowledgements VALUES(?,?,?,?)",
                    (
                        acknowledgement.acknowledgement_id,
                        handoff_id,
                        acknowledgement_bytes,
                        digest_bytes(acknowledgement_bytes),
                    ),
                )
            else:
                if bytes(retained[0]) != handoff_bytes:
                    raise EvidenceIntakeError("semantic intake duplicate conflicts")
                acknowledgement = self._acknowledgement_for_handoff(handoff_id)
                if received_epoch_seconds < acknowledgement.received_epoch_seconds:
                    raise EvidenceIntakeError("replay precedes the retained receipt")

            self.__connection.execute(
                "INSERT INTO evidence_intake_attempts VALUES(?,?,?,?)",
                (
                    request_id,
                    handoff_id,
                    acknowledgement.acknowledgement_id,
                    received_epoch_seconds,
                ),
            )
            self.__connection.commit()
            return acknowledgement
        except (sqlite3.Error, EvidenceIntakeError) as exc:
            if self.__connection.in_transaction:
                self.__connection.rollback()
            if isinstance(exc, EvidenceIntakeError):
                raise
            raise EvidenceIntakeError("Evidence Intake transaction failed") from exc

    def _acknowledgement(self, acknowledgement_id: str) -> IntakeAcknowledgement:
        row = self.__connection.execute(
            "SELECT canonical_bytes FROM evidence_intake_acknowledgements "
            "WHERE acknowledgement_id=?",
            (acknowledgement_id,),
        ).fetchone()
        if row is None:
            raise EvidenceIntakeError("retained acknowledgement is absent")
        return IntakeAcknowledgement.from_canonical_bytes(bytes(row[0]))

    def _acknowledgement_for_handoff(
        self, handoff_id: str
    ) -> IntakeAcknowledgement:
        row = self.__connection.execute(
            "SELECT canonical_bytes FROM evidence_intake_acknowledgements "
            "WHERE handoff_id=?",
            (handoff_id,),
        ).fetchone()
        if row is None:
            raise EvidenceIntakeError("retained acknowledgement is absent")
        return IntakeAcknowledgement.from_canonical_bytes(bytes(row[0]))


def _install(connection: sqlite3.Connection, boundary_id: str) -> None:
    application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    objects = connection.execute(
        "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ).fetchall()
    if (application_id, user_version) == (0, 0) and not objects:
        try:
            connection.executescript("BEGIN IMMEDIATE;\n" + _SCHEMA)
            connection.execute(
                "INSERT INTO evidence_intake_metadata VALUES(1,?)", (boundary_id,)
            )
            connection.execute(f"PRAGMA application_id={_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            connection.commit()
        except sqlite3.Error as exc:
            if connection.in_transaction:
                connection.rollback()
            raise EvidenceIntakeError(
                "Evidence Intake schema installation failed"
            ) from exc


def _verify(connection: sqlite3.Connection, boundary_id: str) -> None:
    try:
        if (
            connection.execute("PRAGMA application_id").fetchone()[0]
            != _APPLICATION_ID
            or connection.execute("PRAGMA user_version").fetchone()[0]
            != _SCHEMA_VERSION
            or connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1
            or connection.execute("PRAGMA quick_check").fetchone()[0] != "ok"
            or connection.execute("PRAGMA foreign_key_check").fetchone() is not None
        ):
            raise EvidenceIntakeError("retained intake database differs")
        with sqlite3.connect(":memory:", isolation_level=None) as expected:
            expected.executescript(_SCHEMA)
            expected_schema = expected.execute(
                "SELECT type,name,sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
            ).fetchall()
        retained_schema = connection.execute(
            "SELECT type,name,sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        ).fetchall()
        if retained_schema != expected_schema:
            raise EvidenceIntakeError("retained intake schema differs")
        metadata = connection.execute(
            "SELECT singleton,boundary_id FROM evidence_intake_metadata"
        ).fetchall()
        if metadata != [(1, boundary_id)]:
            raise EvidenceIntakeError("retained intake boundary differs")

        handoffs: dict[str, tuple[str, str, str, str, str]] = {}
        for row in connection.execute(
            "SELECT handoff_id,candidate_version_id,candidate_version_bytes,"
            "candidate_version_digest,governing_manifest_digest,boundary_id,"
            "receipt_id,canonical_bytes,canonical_digest FROM evidence_intake_handoffs"
        ):
            version = StoryCandidateVersion.from_canonical_bytes(bytes(row[2]))
            expected_handoff = _identity(
                "intake-handoff",
                {
                    "candidate_version_id": version.version_id,
                    "boundary_id": boundary_id,
                },
            )
            expected_receipt = _identity(
                "intake-receipt", {"handoff_id": expected_handoff}
            )
            value = {
                "handoff_id": expected_handoff,
                "candidate_version_id": version.version_id,
                "candidate_version_digest": version.canonical_digest,
                "governing_manifest_digest": (
                    version.governing_manifest.canonical_digest
                ),
                "boundary_id": boundary_id,
                "receipt_id": expected_receipt,
            }
            raw = canonical_json_bytes(value)
            if tuple(row[:2]) != (expected_handoff, version.version_id) or tuple(
                row[3:]
            ) != (
                version.canonical_digest,
                version.governing_manifest.canonical_digest,
                boundary_id,
                expected_receipt,
                raw,
                digest_bytes(raw),
            ):
                raise EvidenceIntakeError("retained intake handoff differs")
            handoffs[expected_handoff] = (
                version.version_id,
                version.canonical_digest,
                version.governing_manifest.canonical_digest,
                boundary_id,
                expected_receipt,
            )

        acknowledgements: dict[str, IntakeAcknowledgement] = {}
        for row in connection.execute(
            "SELECT acknowledgement_id,handoff_id,canonical_bytes,canonical_digest "
            "FROM evidence_intake_acknowledgements"
        ):
            acknowledgement = IntakeAcknowledgement.from_canonical_bytes(bytes(row[2]))
            if (
                row[0] != acknowledgement.acknowledgement_id
                or row[1] != acknowledgement.handoff_id
                or row[3] != digest_bytes(acknowledgement.canonical_bytes)
                or acknowledgement.handoff_id not in handoffs
                or (
                    acknowledgement.candidate_version_id,
                    acknowledgement.candidate_version_digest,
                    acknowledgement.governing_manifest_digest,
                    acknowledgement.boundary_id,
                    acknowledgement.receipt_id,
                )
                != handoffs[acknowledgement.handoff_id]
            ):
                raise EvidenceIntakeError("retained intake acknowledgement differs")
            acknowledgements[acknowledgement.acknowledgement_id] = acknowledgement

        primary_attempts: set[tuple[str, str, int]] = set()
        for row in connection.execute(
            "SELECT request_id,handoff_id,acknowledgement_id,observed_epoch_seconds "
            "FROM evidence_intake_attempts"
        ):
            acknowledgement = acknowledgements.get(str(row[2]))
            if (
                acknowledgement is None
                or row[1] != acknowledgement.handoff_id
                or row[3] < acknowledgement.received_epoch_seconds
            ):
                raise EvidenceIntakeError("retained intake attempt differs")
            primary_attempts.add((str(row[0]), str(row[2]), int(row[3])))
        if any(
            (
                item.request_id,
                item.acknowledgement_id,
                item.received_epoch_seconds,
            )
            not in primary_attempts
            for item in acknowledgements.values()
        ):
            raise EvidenceIntakeError("retained intake primary attempt differs")
    except EvidenceIntakeError:
        raise
    except (sqlite3.Error, CandidateContractError, TypeError, ValueError) as exc:
        raise EvidenceIntakeError("retained intake database is corrupt") from exc


def open_evidence_intake_ingress(
    database: str | Path,
    *,
    boundary_id: str = NON_PUBLIC_EVIDENCE_INTAKE_BOUNDARY,
) -> EvidenceIntakeIngress:
    """Open a local receiver; opening grants no runtime or publication authority."""

    if boundary_id != NON_PUBLIC_EVIDENCE_INTAKE_BOUNDARY:
        raise EvidenceIntakeError("Evidence Intake requires the non-public boundary")
    path = Path(database)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path, isolation_level=None, timeout=30)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        _install(connection, boundary_id)
        _verify(connection, boundary_id)
        return EvidenceIntakeIngress(connection, boundary_id)
    except EvidenceIntakeError:
        if connection is not None:
            connection.close()
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        if connection is not None:
            connection.close()
        raise EvidenceIntakeError("Evidence Intake open failed") from exc


__all__ = [
    "NON_PUBLIC_EVIDENCE_INTAKE_BOUNDARY",
    "EvidenceIntakeError",
    "EvidenceIntakeIngress",
    "IntakeAcknowledgement",
    "open_evidence_intake_ingress",
]
