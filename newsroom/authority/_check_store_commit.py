from __future__ import annotations

from newsroom.sources import (
    DiscoveryOccurrence,
    DiscoveryOccurrenceRequest,
    SourceStateError,
)

from ._check_store_commit_core import _CheckStoreCommitCoreMixin
from ._check_store_support import _outcome_observed_item_error
from ._check_store_commit_decisions import _CheckStoreCommitDecisionMixin
from ._check_store_commit_findings import _CheckStoreCommitFindingMixin


class _CheckStoreCommitMixin(
    _CheckStoreCommitFindingMixin,
    _CheckStoreCommitDecisionMixin,
    _CheckStoreCommitCoreMixin,
):
    """Single-transaction Increment 3C authority commits."""

    def commit_discovery_occurrence(
        self,
        grant,
        *,
        request: DiscoveryOccurrenceRequest,
    ) -> DiscoveryOccurrence:
        if not isinstance(request, DiscoveryOccurrenceRequest):
            raise TypeError("occurrence commit requires a typed request")
        with self._lock:
            outcome = self._connection.execute(
                "SELECT * FROM check_outcomes WHERE outcome_id=?",
                (str(request.check_outcome_id),),
            ).fetchone()
            if outcome is None:
                raise SourceStateError(
                    "discovery occurrence requires an exact retained Check Outcome"
                )
            revision = self._revision_row(
                self._connection,
                str(request.revision_id),
            )
            if request.representation_id is None:
                raise SourceStateError(
                    "discovery occurrence requires an exact observed Representation"
                )
            representation = self._representation_row(
                self._connection,
                str(request.representation_id),
            )
            outcome_record = self._check_outcome_from_row(
                self._connection,
                outcome,
                replayed=False,
            )
            observed_error = _outcome_observed_item_error(
                self,
                self._connection,
                outcome_record.request,
                revision_item_id=str(revision["item_id"]),
                representation_digest=str(
                    representation["representation_digest"]
                ),
            )
            if observed_error is not None:
                raise SourceStateError(observed_error)
            if (
                str(outcome["definition_id"])
                != str(revision["definition_id"])
                or str(outcome["definition_version_id"])
                != str(request.definition_version_id)
                or str(outcome["completed_at"])
                != request.observed_at.to_text()
                or outcome["receipt_digest"] != request.receipt_digest
                or str(outcome["kind"])
                not in {
                    "SUCCESS_UNCHANGED",
                    "SUCCESS_CHANGED",
                    "SUCCESS_PARTIAL",
                    "SUCCESS_TRUNCATED",
                }
            ):
                raise SourceStateError(
                    "discovery occurrence differs from exact Check Outcome lineage"
                )
        return super().commit_discovery_occurrence(
            grant,
            request=request,
        )


__all__ = ["_CheckStoreCommitMixin"]
