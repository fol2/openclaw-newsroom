# Attribute Newsroom RAM/CPU — Mini packet (#898)

**Superseded.** This note recorded the first Mini packet and an invalid
`NO_GO`. The authoritative corrected packet is
[`2026-09-02-issue-898-ram-cpu.md`](2026-09-02-issue-898-ram-cpu.md).

- Role: Dated provider-free research evidence
- Status: Superseded; do not treat the `NO_GO` below as a Rust decision
- Owner: fol2
- Canonical language: English
- Date: 2026-09-02
- Ticket: [#898](https://github.com/fol2/newsroom/issues/898)
- Run parent: `038a0c06db21d8424854e6992ca6ef98339e523f`
- Canonical measurements: [`2026-09-02-issue-898-ram-cpu-measurements.json`](2026-09-02-issue-898-ram-cpu-measurements.json)

This note is non-normative. It authorises no Rust production code, provider
call, queue claim, writable canonical-store access, Neo4j mutation,
publication or activation.

## Decision

**`NO_GO`**

H2/H3 full-corpus reconstruction is the largest measured local-memory
contributor. Removable peak RSS is about **1,908 MiB** on the Mini (D1
2,000 MiB peak minus 92 MiB import baseline), which clears the 20% / 64 MiB
floor. The pre-registered direction when H2 wins is **exact row selection plus
bounded or streaming resolution**, not a Rust translation of the scan.

A Rust atom that reimplemented `load_graphiti_units` / `_resolve_graphiti_event_units`
as they stand would preserve the full-corpus allocation. Python remains
authoritative. Next correction is a Python query that loads only the target
event’s rows.

## Hardware and method

Host: Mac16,10, Apple M4, 10 cores, 16 GiB. Fresh child process per run. One
warm-up and three measured runs except the read-only process-tree snapshot.
Peak RSS from `resource.getrusage` and `/usr/bin/time -l` matched. Retained
RSS from `ps -o rss` after `gc.collect()`. Canonical proving store was copied
to a temp file (1,789,808,640 bytes) and never opened writable. No daemon was
signalled.

## Ranked peaks

| Case | Peak RSS | Retained RSS | Median CPU | Removable vs import |
|---|---:|---:|---:|---:|
| D1 copied-store `max_writes=0` cycle | 2,000 MiB | 1,577 MiB | 251 s | 1,908 MiB |
| B10 one-event resolve on copied store | 1,907 MiB | 1,495 MiB | 509 s | 1,816 MiB |
| A4 idle Graphiti worker import | 109 MiB | 94 MiB | 14 s | 18 MiB |
| A2 import `newsroom.control_plane.cycle` | 92 MiB | 86 MiB | 11 s | baseline |
| A1 bare interpreter | 19 MiB | 19 MiB | 0 s | — |
| Live Neo4j (read-only `ps`) | 19 MiB | 19 MiB | — | — |

Fixture one-event work does not explain production RAM: solo 91 MiB, 10×
unrelated tiny observations 97 MiB. The copied 7-day corpus does: one event
still materialised **1,337** units from **6,850** poll observations, then kept
one. Cycle D1 formed **280** candidates, minted **0**, Graphiti **0**.

`graphiti-core` was not imported when `EvaluationGraphitiRunner` was
constructed. H1 is rejected as the dominant idle cost.

Malformed and empty inputs returned zero units and did not claim a queue
event.

Latest observation bodies are parsed more than once on the static path
(`YES_STATIC`).

Stage splits C1–C7 were run on the 10× fixture so the 1.7 GiB copy was not
rescanned 28 times. Production attribution is B10 versus D1 on that copy.

## Live process tree

Hermes Control Plane and the Graphiti worker were not in the snapshot. Observed
without signalling: newsroom-hub Python (~2 MiB), Neo4j helper (~4 MiB), Neo4j
Community (~19 MiB). Those idle RSS values are far below one-event
reconstruction.

## GO gate

1. Largest removable contributor: yes (H2/H3 corpus reconstruction).
2. ≥ 20% of peak or ≥ 64 MiB: yes (~1,908 MiB).
3. Bounded deterministic I/O: not for the current full-corpus scan.
4. Rust needs no authority write: would hold for a later bounded resolver.
5. Python authoritative without dual write: yes.
6. Not a microbenchmark: yes — end-to-end B10/D1 RSS is the same cost.

Gate 3 plus the H2 prescription keep the first Rust atom **`NO_GO`**.

## Next issue (draft only; not filed)

Title: `Fix — Resolve one Graphiti event without reconstructing the retained corpus`

Implementation-complete body should: select proving rows by the event’s
source/item/revision identity (or ingest refs) before parse/materialise;
keep rights checks; prove solo vs copied-store peak RSS no longer scales
with unrelated corpus size; forbid queue claims and canonical writes;
leave Rust until a measured remainder still clears the #898 floor.
