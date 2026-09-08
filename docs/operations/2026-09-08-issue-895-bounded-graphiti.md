# Issue #895 bounded Graphiti operation

[Issue #895](https://github.com/fol2/newsroom/issues/895) defines the outcome;
[its checkpoint](https://github.com/fol2/newsroom/issues/895#issuecomment-5547891812)
records the current code, owned process and immutable evidence. This runbook is
not an activation grant or a claim of PRE151 completion. Publication, continuous
service, 11R and #151 remain separate.

## Start once

Use the prepared clean exact-main worktree, its existing virtual environment and
observed exact-head Focus evidence. Do not wait for Full Repository Health, run
a duplicate Focus workflow, repeat dependency bootstrap or add a review gate.

```sh
ROOT=/Users/jamesto/Coding/newsroom-evidence/issue-895
WT="$ROOT/worktree-f4-bc5ca15a81bf43f41594ee86e066c429ea62169c"
cd "$WT" || exit 1
```

Only when a **new extraction** is needed and every preceding effect is classified,
use the existing `scripts.issue_790_conservative_disposition supply-event` with
canonical proving/unpublished stores, the exact current frontier and one fresh
UTC observation time. Its receipt must identify exactly one untouched queued
event, attempt count zero and no provider dispatch. Never supply another event
to diagnose a deterministic admission defect.

```sh
"$WT/.venv/bin/python" -m scripts.issue_790_conservative_disposition supply-event \
  --proving-store /Users/jamesto/Coding/newsroom/data/newsroom/proving_store.sqlite3 \
  --store /Users/jamesto/Coding/newsroom/data/newsroom/unpublished_store.sqlite3 \
  --expected-frontier-ledger-seq "$CURRENT_FRONTIER" \
  --observed-at "$OBSERVED_AT" > "$NEW_SUPPLY_RECEIPT"
```

The supported external `seal-and-dispatch-once.py` entry receives
`--event-id`, `--focus-manifest-digest` and a fresh `--output-dir`. It loads only
the existing purpose-provisioned Cursor credential, seals once, then hands the
still-owned authority runtime to the existing executor's `--dispatch` path.
Do not print the credential, call a separate `--preflight`, close/reopen the
runtime between seal and dispatch, or repeat a consumed invocation. Dispatch
already performs complete invocation binding before effects. Packet caps,
one event/minute, zero source/provider retries and fallbacks, current owner
fences and signed stops remain effective.

```sh
PYTHONPATH="$WT" "$WT/.venv/bin/python" "$ROOT/seal-and-dispatch-once.py" \
  --event-id "$EXACT_SUPPLIED_EVENT_ID" \
  --focus-manifest-digest "$EXACT_FOCUS_MANIFEST_DIGEST" \
  --output-dir "$NEW_SEAL_OUTPUT_DIR" > "$NEW_RUNTIME_LOG" 2>&1
STATUS=$?
printf 'PROCESS_EXIT=%s\n' "$STATUS"
```

## Inspect and stop

Read the retained runtime stdout/stderr, `process-exit.json`, packet,
`.issue-895-f4-invocation-<packet digest>.json` and
`f4-campaign-<packet digest>/` evidence. These reads do not start another worker.
A missing result is not success. Preserve any stopped campaign as stopped even
if later admission work completes.

The existing executor observes authenticated owner issue-authority actions
`stop`, `revoke` and `restrict` at its guarded boundaries, for example
`{"authority_action":"stop","reason":"operator stop"}`. It does not promise
to interrupt an already in-flight provider call. Preserve the invocation marker
and classify effects before any successor; never delete a marker or reset an
attempt to stop or recover work.

## Continue retained admission, without extraction or resealing

Use the [existing admission consumer operations](2026-08-24-graphiti-admission-consumer.md).
A corrected, exact failure epoch is eligible through authenticated
`resume_graphiti_admission`; the same consumer then replays already committed
mention/proposal command identities, decides the retained cohort and finalises
one full generation. No new source event, provider call or seal is involved.

For the retained event 13769, the existing external entry accepts:

```sh
PYTHONPATH="$WT" "$WT/.venv/bin/python" "$ROOT/admission-continuation-13769.py" \
  --ci-evidence "$EXACT_MAIN_CI_EVIDENCE" \
  --failure-evidence "$EXACT_RETAINED_QUEUE_FAILURE_EVIDENCE" \
  --output-dir "$NEW_CONTINUATION_OUTPUT_DIR"
```

Set those evidence paths from the checkpoint, not from an old command example.
Run one owned invocation and retain its actual exit code. It opens the authority
once, binds exactly four original proposals and a 600-second wall cap, and uses
the existing authenticated command, consumer, full-generation reconciliation
and admitted-only context hydrator. It never invokes the event dispatcher.

Completion requires actual exit zero and `ADMISSION_CONTINUATION_COMPLETE`,
exactly four retained decisions, one reconciled active generation, no admission
backlog/dead letters/projection gaps, current admitted-only context and unchanged
original event/provider-attempt/model/spend rows. Retain before/after partitions,
actual duration and denominators. The source counts in this separate window are
explicitly admission-only; do not reconstruct a missing original W0 baseline or
relabel the original `CAMPAIGN_STOPPED`.

A fresh failure is evidence to reproduce and correct off-line, not an automatic
resume. Each resume retains the exact prior error and failed-attempt count in
the existing immutable command journal. Stale fingerprints, active claims,
admission decisions, projections, tombstones and integrity holds are rejected.
No blanket dead-letter reset, new backup, repeated full-store plan or operator
approval for an ordinary code correction is required.
