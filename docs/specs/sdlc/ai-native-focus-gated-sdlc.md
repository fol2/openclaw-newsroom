# AI-native Focus Gate SDLC

**Status:** Accepted
**Owner:** Product owner
**Accepted:** 2026-08-27
**Issues:** #799, #811
**Canonical language:** English (UK)
**Reference:** Anthropic, “The AI-Native Software Development Lifecycle”, 2026-08-21

## Decision

Newsroom uses an agent-owned, artefact-driven development loop. The ordinary
pull-request critical path is one deterministic Focus Gate job. Full repository
health, service qualification, research and irreversible operational controls
are independent conditional lanes.

The objective is not fewer checks. It is complete relevant evidence with
minimum wall time, model context and compute.

## Invariants

1. Agents own intent-to-merge for ordinary work; humans handle ambiguity,
   credentials, regulated or irreversible effects and explicit owner decisions.
2. Work enters the critical path only when it answers a concrete risk question
   for the exact change.
3. Unchanged context, environment setup, tests, review and remote observations
   are not repeated.
4. Every demonstrated failure mode and affected boundary remains covered.
   Unknown executable work escalates visibly and fail-closed.

## Intent and human boundary

The GitHub issue is the ordinary change-intent source of truth. A second intent,
plan or specification is justified only by a concrete ambiguity or an
independent durable decision boundary.

The owner defines outcomes, constraints, permitted effects, success measures
and stop conditions. Agents choose the implementation and relevant evidence.
This repository has no organisation ruleset, merge queue or required-status
platform enforcement. Ordinary non-F4 work may be merged by the agent after one
observed exact-head Focus Gate success and one clean feature-complete review.
F4 remains owner gated.

## Focus Gates

- **F0:** exact change integrity, including touched Python, JSON, TOML, YAML and
  shell syntax. Documentation-only changes stop here.
- **F1:** direct positive, negative, boundary and regression behaviour.
- **F2:** deterministic affected callers, consumers, public symbols and
  contracts. Stateful changes add bounded migration/authority sentinels rather
  than the complete authority inventory.
- **F3:** bounded actual-service evidence only when local evidence is
  insufficient or an actual-service consumer is selected.
- **F4:** credentials, security, migration/deletion, publication, deployment,
  admission, activation and release remain exact and fail-closed.

Internal publication implementation, retained payloads and tests are not public
publication effects merely because their filename contains `publication`.
Under the owner-approved #151 scope, ordinary engineering and private pipeline
operation are autonomous; the public exposure control belongs to newsroom-hub.
Do not add a per-story or per-batch human approval to that private pipeline.
Verify the actual destination and exposure boundary before connecting it.
This distinction does not authorise changing hub public exposure, bypassing
credentials or integrity controls, or treating a failed revision as covered.


## Machine route

`scripts/sdlc/focus_gate_v2.py` emits `newsroom.sdlc.focus-route.v1`, a canonical
content-addressed manifest containing:

- exact base/head and changed paths;
- selected gates and reasons;
- selected deterministic and actual-service tests;
- research and full-health routing;
- owner-authority and bootstrap requirements; and
- expected Focus Gate job/bootstrap counts.

The blocking selector is deterministic. An ML selector may be researched
separately but cannot enter the blocking path without promotion through this
contract.

## Event surfaces

| Workflow | Event surface | Purpose |
|---|---|---|
| `focus-gates.yml` | ordinary pull requests | F0-F4 route, at most one locked bootstrap |
| `ci.yml` | research paths, schedule, manual | provider-free Graphiti research |
| `evidence.yml` | push to `main`, schedule, manual | complete deterministic product health; research fixtures excluded |

A normal narrow PR never starts the retired eighteen-shard topology or the
Graphiti research campaign. Rapid consecutive main pushes cancel obsolete
full-health runs so only the newest main head consumes the complete inventory.

## Selection rules

Changed tests select themselves. Changed source selects tests through explicit
critical rules and repository import/package analysis. Public constants and
short re-exported symbols are resolved through AST analysis. Migration and
authority paths add two bounded sentinels. A discovered
`*_neo4j_service.py` consumer moves to F3 automatically. Dependency changes
truthfully select both full product health and the isolated Graphiti research
lane.

A focused failure may broaden to its implicated dependency. It does not
automatically broaden to the repository.

## Retained compatibility tooling

The pre-hardening `[focus]` table and `sdlc-v2.6` lane, receipt and timing
tables remain in `.sdlc/gates.toml` so historical receipts and dormant
diagnostic commands keep validating fail-closed. They are compatibility data,
not the current ordinary pull-request topology.
`[focus_hardening]` and `focus_gate_v2.py` are the current execution SSOT and
create no merge-queue dependency.

## Research, review and stop rule

Research starts from an explicit uncertainty and produces a compact promoted
contract, fixture, policy, benchmark or decision. Normal development consumes
that output rather than replaying the campaign. Provider calls always require
separate owner authority.

One feature-complete review is the default. Repeat only after a material change
or unresolved relevant finding. Stop after a coherent evidence set; report
exact runs, omissions and uncertainty. Never claim an unobserved workflow
result.

## Quantitative target

An ordinary documentation PR starts one Focus Gate evidence job and zero
project-dependency bootstraps. An ordinary executable PR starts one Focus Gate
evidence job and one locked bootstrap. The separate trusted PR Lifecycle
metadata check remains lightweight and installs no project dependencies.
Obsolete heads are cancelled. Post-merge/scheduled/manual full health and
research remain outside the ordinary critical path.

### Exact-head evidence after merge

Merge commits on `main` do not automatically re-run Focus Gates. When an
operation needs observed CI on the **exact tip SHA** (bounded live canary,
provider-backed proof, or other fail-closed exact-main gates):

1. Prefer `workflow_dispatch` of `focus-gates.yml` on that tip (with the
   correct non-head `base_sha` for routing).
2. Treat Full Repository Health as an independent inventory lane — not the
   default wait for that evidence.
3. Do not block live-canary wall time on full-health completion when Focus
   Gates can be obtained on the tip.

This preserves the wall-time objective: complete relevant evidence without
pulling the full-inventory lane into the ordinary or bounded-live critical
path.

## Non-effects

This contract grants no publication, provider call, production admission,
deployment, activation, spend or credential authority.
