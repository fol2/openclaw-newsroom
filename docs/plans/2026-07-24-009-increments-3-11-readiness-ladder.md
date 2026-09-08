# Increments 3–11 blocked readiness ladder

**Status:** Proposed — owner review required
**Owner:** Product owner
**Prepared:** 2026-07-24
**Canonical language:** UK English
**Review base:** `main@5fcd8bc862e552961b6b147572879e79c7266931`
**Parent programme:** [`2026-07-16-005-native-graphrag-production-implementation.md`](2026-07-16-005-native-graphrag-production-implementation.md)
**Immediate predecessor package:** [`2026-07-24-008-increment-2-complete-fixture-readiness.md`](2026-07-24-008-increment-2-complete-fixture-readiness.md)
**Canonical specification map:** [`../specs/editorial-automation/README.md`](../specs/editorial-automation/README.md)
**Implementation authority:** None. This document prepares dependency, review, evidence and stop boundaries. It does not authorise code, source access, Graphiti, models, embeddings, search, spending, shadow, canary, publication, production activation or legacy retirement.

**Currentness:** The Increment 11 section was reconciled on 2026-09-08 after
PRE151. Earlier increment snapshots remain historical; current issue intent,
Accepted ADRs/specifications and `AGENTS.md`/`REVIEW.md` govern current work.

## 1. Purpose

Prepare Increments 3–11 far enough that the next fresh implementation session can begin from an explicit boundary after its predecessor closes, without inventing requirements from the master roadmap or prematurely selecting evidence-dependent values.

This is a readiness ladder, not a claim that every later increment is implementation-ready today. Each increment is placed in one of three states:

1. **Prepared and dependency-blocked:** accepted requirements define the work; implementation waits for the preceding increment and a current-head readiness check.
2. **Prepared with owner-decision gate:** accepted requirements define the decision packet, but exact runtime versions, rights, budgets, thresholds or operational values require a separate owner-approved plan.
3. **Prepared with normative blocker:** relevant downstream specifications remain Draft or another authority is missing; only the decision template, inventory and stop conditions are prepared.

## 2. Global execution rules

1. Increment numbers are merge and verification boundaries, not independently activatable product stages.
2. No increment inherits authority from the prior increment. Exact source, adapter, model, prompt, embedding, provider, Profile and deployment versions qualify separately.
3. Separate ordinary readiness/local engineering work authorised by the current issue from its blocked live or production effects. A production entry gate is not a prohibition on reconciling evidence or preparing a scope proposal.
4. Use the current issue as change intent and reference affected Accepted contracts, exclusions, evidence and any relevant rollback; do not duplicate an implementation-complete issue into a new packet.
5. SQLite and governed objects remain authority. Graph, vector and full-text systems remain rebuildable projections.
6. Models, Graphiti, workers and retrieval systems propose or provide context. Deterministic or authorised controllers commit.
7. GraphRAG remains mandatory in complete live shadow, canary and production, but never becomes editorial or publication authority.
8. Failure, staleness, gaps and unavailable dependencies remain explicit; they never become no news, no prior match or editorial rejection.
9. Follow the current `AGENTS.md` and `REVIEW.md`: deterministic Focus routing, one observed exact-head Focus success and one clean feature-complete review for ordinary merges. Actual-service evidence is conditional on the changed boundary; Full Repository Health is independent.
10. Production activation, Operational Admission, Evidence Intake canary, publication authority and legacy retirement remain separate owner decisions.
11. Default to one issue, branch and ordinary PR per coherent change. Split only for an independent merge, rollback, owner, dependency or release boundary, not per gate or machine.

## 3. Dependency chain

```text
Increment 2 complete fixture
    ↓
Increment 3 source adapters and discovery lineage
    ↓
Increment 4 extraction, entity resolution and relation admission
    ↓
Increment 5 production hybrid retrieval and named tools
    ↓
Increment 6 full triage, Hypotheses, Candidates and Handoff
    ↓
Increment 7 Agenda, bounded search and Event-Scoped Local Watch
    ↓
Increment 8 evaluation, operations, recovery and security
    ↓ owner-approved exact shadow plan
Increment 9 production-equivalent integrated shadow
    ↓ accepted Evidence Intake authority and canary plan
Increment 10 governed Evidence Intake canary
    ↓ accepted publication/activation authority and explicit owner decision
Increment 11 bounded Hermes production activation
```

The accepted programme permits justified overlap after shared contracts merge. Current issue intent determines ordinary development scope; predecessor evidence and exact effect authority still govern live and production starts. Planning or merging an ordinary change does not activate an increment.

---

# Increment 3 — Source adapters and discovery lineage

**Readiness state:** Prepared and dependency-blocked
**Dependency:** Increment 2 completed on `main`
**Runtime authority:** Fixtures and approved replay only; named live sources remain disabled

## Objective

Implement repository-native generic source and discovery contracts from authorised trigger through Check, Source Item and Revision, observable transition, Signal and Lead, with structural graph projection and source, graph and coverage health.

## Normative basis

- `COV-*` coverage obligations and gaps;
- `FLOW-001`–`FLOW-045`, applicable failure and inspectability requirements;
- `DREC-001`–`DREC-037` and `DREC-060`–`DREC-077`;
- `SRC-*` source roles, portfolio functions and readiness;
- `CHG-*` and `AGEN-*` source-change, baseline and Agenda semantics;
- `DOUT-*` and applicable `DPRI-*` outcome separation;
- `GRAG-024`–`GRAG-028`, `GRAG-035` and `GRAG-042`–`GRAG-045`;
- ADR 0001, ADR 0002, ADR 0004 and ADR 0005.

## Prepared review units

### 3A — Source registry and immutable source contracts

Deliver:

- Source Definition and immutable Source Definition Version authority;
- source role, portfolio function, coverage mapping, rights reference and dependency records;
- observation-model and baseline-policy identities;
- stable Source Item, Revision, Representation and Occurrence contracts;
- versioned source-specific identity and canonicalisation interfaces;
- authenticated commands, checked migrations and startup integrity; and
- no silent source addition, removal, repurpose or inherited authority.

### 3B — Generic transport and parser adapter boundary

Deliver generic, fixture-driven interfaces for:

- RSS/Atom;
- structured JSON;
- mutable maintained documents;
- complete current-state snapshots;
- rolling listings; and
- Planned Agenda feeds.

The boundary includes strict TLS, redirect, egress, timeout, body, content-type, parser-resource and shape-drift controls. Empty, malformed, partial, unchanged, changed, blocked and failed outcomes remain distinct.

### 3C — Check, baseline and observable-transition authority

Deliver:

- Check Request, Attempt and Outcome authority;
- source-specific first-run and reset baselines;
- deterministic new, revised, re-observed, activation, escalation, de-escalation, clearance, expiry, cancellation, withdrawal, reactivation and ambiguous-absence semantics;
- no model in scheduling, access, parsing or unchanged detection;
- idempotent replay and exact lineage; and
- Operational Findings for integrity or source-contract failures.

### 3D — Signal, deterministic gate and Lead foundation

Deliver:

- Discovery Signal admission;
- Gate Decisions for duplicate, non-change, clear exclusion, promotion and operational hold;
- one Lead per promoted Signal by default;
- accepted coverage and urgency basis;
- immutable Lead lineage and Watch Condition seam;
- canonical outcome/reason mapping; and
- no Candidate, evidence or publication bypass.

### 3E — Discovery-lineage projection and health seam

Deliver:

- structural projection of Source Definition, Item, Revision, Representation, Signal and Lead lineage;
- exact checkpoint, gap, dead-letter and generation behaviour through the existing projection authority;
- source, parser, projection and coverage-health semantic records;
- bounded status commands and read-only inspection; and
- a complete fixture path through actual Neo4j.

## Entry gate

Do not start Increment 3 until:

- Increment 2 is closed as completed;
- the current `main` head is recorded;
- the Increment 2 relation and hybrid retrieval contracts are stable;
- fixture-only source content and parser cases are approved for repository retention; and
- the Increment 3 issue records exact PR boundaries and selected generic interfaces.

## Completion gate

Increment 3 closes when generic adapters and source semantics pass fixture, replay, fault, actual-Neo4j and full-repository evidence, while every named live source remains disabled and no source credential, schedule, external request or spending is introduced.

## Stop boundary

Do not begin Increment 4 until Increment 3 has a complete deterministic Source Revision → Signal → Lead lineage that can supply exact permitted inputs to extraction without importing legacy identity or live-source assumptions.

---

# Increment 4 — Extraction, entity resolution and relation admission

**Readiness state:** Prepared and dependency-blocked
**Dependency:** Increment 3 completed on `main`
**Runtime authority:** Deterministic fake extraction first; real Graphiti/model execution requires a separate rights, cost and Evaluation Plan decision

## Objective

Implement production-shaped Extraction Runs, explicit entity and alias authority, entity resolution decisions, merge/split/reversal, Relation Proposals and Admission Decisions, isolated Graphiti integration and admitted projection using bilingual fixtures.

## Normative basis

- `DREC-*` exact identity, version, lineage and time semantics;
- `GRAG-010`–`GRAG-016` trust, entities and reified relations;
- `GRAG-020`–`GRAG-028` proposal-only extraction, persistence, admission and rebuild;
- `GRAG-030`, `GRAG-034` and `GRAG-035`;
- `GRPROD-010`–`GRPROD-016` and applicable deployment/security requirements;
- Accepted discovery source, change and triage contracts for input and downstream boundaries.

## Prepared review units

### 4A — Extraction Run and proposal authority

Deliver immutable Extraction Runs binding exact input object, Source Revision, Representation, model/framework/prompt/code contract, structured output, timing, cost, partial/failure state and proposal set. Start with a deterministic fake extractor through the final interface.

### 4B — Entity mention, canonical entity and resolution decisions

Deliver:

- Entity Mention and Alias records;
- Canonical Entity identity and immutable versions;
- Resolution Proposals;
- accept, reject, hold and unresolved decisions;
- explicit merge, split and reversal decisions;
- bilingual English/Hong Kong Traditional Chinese fixtures; and
- guards preventing similarity or name equality from automatic canonicalisation.

### 4C — General relation proposal and admission authority

Extend the Increment 2 generic relation seam to versioned editorial predicates, exact evidence references, temporal scope, entity-resolution preconditions, rejection/hold/revocation/supersession and admitted projection.

### 4D — Isolated Graphiti adapter

Deliver a private Graphiti adapter in a logically isolated, disposable proposal workspace. It may create extraction output and proposals only. It holds no governed projection or authority credential. Every output is persisted before admission.

The exact Graphiti release, model, prompt, destination, retention and cost must be owner-approved before any real execution. Until then, only interface, fake and approved replay evidence runs.

### 4E — Bilingual proposal/admission proof

Prove:

- correct alias resolution without false merge;
- unresolved identity blocks dependent relation admission;
- unadmitted proposals remain proposal-scoped;
- admitted relations project with complete provenance;
- merge, split and reversal preserve predecessor history;
- replay/rebuild uses retained output and does not rerun stochastic extraction; and
- actual Neo4j contains only governed admitted state on the admitted surface.

## Entry gate

Do not start Increment 4 until Increment 3 produces authoritative, rights-permitted Source Revision and Representation inputs through the new discovery path.

Real Graphiti execution additionally requires:

- exact framework and model versions;
- rights and model-destination approval;
- prompt and output-schema approval;
- cost and rate budget;
- security and credential boundary;
- retention and replay rules; and
- an owner-approved Evaluation Plan or explicitly narrower qualification plan.

## Completion gate

Increment 4 closes when deterministic extraction, entity-resolution governance and relation admission are complete and tested, and either the exact Graphiti/model lane is qualified under owner authority or remains explicitly disabled with no claim of runtime completion.

## Stop boundary

Do not begin production hybrid retrieval in Increment 5 until admitted entity and relation semantics, trust scopes, invalidation and generation rebuild are stable and actual-Neo4j evidence proves no proposal leakage.

---

# Increment 5 — Production hybrid retrieval and named tools

**Readiness state:** Prepared and dependency-blocked with evidence-dependent choices
**Dependency:** Increment 4 completed on `main`
**Runtime authority:** No production embedding or protected-content index until exact rights, model, cost and Evaluation Plan decisions exist

## Objective

Turn the Increment 2 fixture retrieval path into production-shaped exact, full-text, vector and bounded graph retrieval with authoritative hydration, dependency-aware deduplication, named read-only tools, budgets, security and pre-registered ablation.

## Normative basis

- `GRAG-030`–`GRAG-035` hybrid retrieval, hydration, named tools and metadata;
- `GRAG-040`–`GRAG-046` discovery integration and degraded behaviour;
- `GRAG-050`–`GRAG-058` initial target and evaluation;
- `GRPROD-001`–`GRPROD-031` as applicable;
- `TRI-020`–`TRI-028` retrieval and collision boundaries;
- `DEVAL-*` for ablation, required slices and threshold discipline;
- `DOPS-*` for profiles, health, security, capacity and degradation.

## Prepared review units

### 5A — Production retrieval contract decision

Before code that creates production vectors, approve exact:

- embedding model and destination;
- chunking, normalisation and language behaviour;
- full-text index configuration;
- vector dimensions and similarity function;
- graph depth, predicates and temporal window;
- retrieval mode limits and budgets;
- fusion and dependency-deduplication method;
- freshness, watermark, gap and degraded policy;
- rights, retention and purge handling; and
- evaluation and rollback versions.

Calibration may inform values but cannot qualify the same thresholds.

### 5B — Production retriever implementations

Deliver exact, full-text, vector and graph retrievers behind typed interfaces with query receipts, limits, provenance, trust, projection metadata and deterministic failure outcomes.

### 5C — Named read-only tools

Implement the accepted initial tool family, beginning with:

- `find_related_event_candidates`;
- `get_event_or_process_timeline`;
- `find_source_revision_impact`;
- `find_shared_origin_dependencies`;
- `find_conflicting_relation_candidates`; and
- `get_candidate_provenance`.

Each tool fixes purpose, allowed types, trust, depth, fan-out, window, result count, timeout, freshness and mandatory provenance. No general write or unrestricted generated Cypher is introduced.

### 5D — Authoritative hydration and degraded operation

Deliver complete current hydration, rights filtering, context size accounting, gap enforcement, explicit unavailable/stale/partial outcomes and approved exact fallbacks. Graph or semantic retrieval outage never becomes no prior match.

### 5E — Ablation, security and actual-service qualification

Run pre-registered exact-only, full-text-only, vector-only, graph-only and hybrid comparisons on required English, Chinese, mixed-language, timeline, dependency, correction and distractor fixtures. Test query injection, scope amplification, credential isolation, performance bounds, purge and rebuild.

## Entry gate

Do not start Increment 5 implementation until Increment 4 closes and the exact production retrieval contract is owner-approved. Interface-only work may be prepared, but no real embedding call or protected-content vector is created before rights and Evaluation Plan authority.

## Completion gate

Increment 5 closes only when production-shaped named tools, authoritative hydration, gap/degraded behaviour, security and pre-registered ablation pass for exact bound versions against actual Neo4j.

## Stop boundary

Do not begin full triage in Increment 6 until retrieval returns stable, bounded, trust-labelled and authority-hydrated context with exact collision checks and explicit incomplete outcomes.

---

# Increment 6 — Full triage, Hypotheses, Candidates and Handoff

**Readiness state:** Prepared and dependency-blocked
**Dependency:** Increment 5 completed on `main`
**Runtime authority:** Evaluation Evidence Intake sink only; no evidence acquisition or publication

## Objective

Implement bounded Triage Work Items and execution batches, structured untrusted Triage Proposals, deterministic validation, immutable Lead dispositions, Event Hypotheses and versions, relationship decisions, Candidate admission/versioning and durable Evidence Handoff to an evaluation sink.

## Normative basis

- `FLOW-040`–`FLOW-075`, failure and inspectability requirements;
- `DREC-033`–`DREC-057` and exact lineage/time requirements;
- `TRI-*`;
- `DOUT-*` and `DPRI-*`;
- `GRAG-040`–`GRAG-046`;
- applicable `DOPS-*` queue, stale-work, dependency and Handoff rules.

## Prepared review units

### 6A — Canonical outcomes, reasons and priority lanes

Implement immutable canonical outcome mapping, structured reasons and basis classes, next-action separation, Watch Conditions, terminality and ordinal lanes `CONTAINMENT`, `URGENT`, `TIME_SENSITIVE`, `PLANNED_WINDOW`, `ROUTINE` and `OPTIONAL_EVALUATION`. No global eligibility score or quota is introduced.

### 6B — Work Items, execution batches and queue authority

Deliver exact decision Lead/context-only Lead manifests, bounded batches, urgent isolation, concurrency ownership, deadlines, fairness, starvation protection, stale-work revalidation and visible backpressure.

### 6C — Triage worker and proposal contract

Deliver a structured proposal schema, immutable attempts, exact input citations, allowed routes, uncertainty, relationship targets, Watch Conditions, supplemental actions and Candidate manifests. Start with deterministic or approved fixture workers; real model execution requires exact version, rights, budget and evaluation authority.

### 6D — Event Hypothesis and relationship decisions

Deliver creation, association, versioning, consolidation and split with append-only predecessor history. Preserve same state, development, correction/reversal, related distinct, no adequate match and uncertain relationship as separate unverified semantics.

### 6E — Candidate authority and versioning

Extend the existing Candidate foundation to multiple Candidates and immutable versions with exact Hypothesis, Lead, Signal, coverage, urgency, likely-new-information, uncertainty, evidence-objective and governing-version manifests. Collision remains relational and deterministic.

### 6F — Evaluation Handoff sink

Implement stable semantic Handoff identity, idempotent transport attempts, acknowledgement correlation, pending/ambiguous retry and structured feedback against an evaluation-only sink. Discovery records remain non-evidence.

## Entry gate

Do not start Increment 6 until Increment 5 provides qualified named retrieval tools and exact current collision checks. Any model-assisted triage additionally requires an exact worker/model version, rights, budget and Evaluation Plan.

## Completion gate

Increment 6 closes when the full fixture Source Revision → Signal → Lead → Work Item → Retrieval Context → Proposal → Hypothesis → Candidate Version → evaluation Handoff path passes replay, concurrency, failure, urgent and actual-service evidence.

## Stop boundary

Do not begin Agenda/search/local-watch runtime work in Increment 7 until the normal Signal-to-Candidate workflow and supplemental-discovery re-entry path are complete and cannot be bypassed.

---

# Increment 7 — Agenda, bounded search and Event-Scoped Local Watch

**Readiness state:** Prepared and dependency-blocked with provider and locality decisions deferred
**Dependency:** Increment 6 completed on `main`
**Runtime authority:** Providers and named localities remain disabled until exact rights, budget, evaluation and operations decisions

## Objective

Implement Planned Agenda lifecycle, bounded Search Purpose and request records, privacy and amplification controls, Coverage Audit, Event-Scoped Local Watch and prospective-versus-retrospective distinctions without selecting a generic search clock or systematic locality promise.

## Normative basis

- `AGEN-*` and applicable `CHG-*`;
- `SRCH-*` and `CAUD-*`;
- `LOC-*`;
- `FLOW-010`–`FLOW-015`, `FLOW-057` and `FLOW-070`–`FLOW-087`;
- `DREC-*` Agenda, search-channel, Gap and Watch lineage;
- applicable `DOUT-*`, `DPRI-*`, `DEVAL-*` and `DOPS-*`.

## Prepared review units

### 7A — Planned Agenda authority

Deliver stable Agenda Items and immutable Versions, time-zone and precision honesty, reschedule/cancel history, occurrence-confirmation paths, missed-expectation criteria, late resolution and no clock-generated Lead or Candidate.

### 7B — Search records and deterministic controls

Deliver Search Purpose, Request, Attempt, Outcome, Result Reference and Review Decision; purpose-specific templates; query privacy; hard result/page/variant/retry/branch/time/cost limits; provider-attributed result lineage; and no recursive or silent fallback.

### 7C — Coverage audit and Gap assessment

Deliver pre-registered prospective comparator records, separately labelled retrospective investigations, reviewed Gap decisions, dependency/timeliness/health review and isolated/systemic/Best-effort/deferred assessments.

### 7D — Event-Scoped Local Watch

Deliver bounded event purpose, locality/service boundary, source set, budget, owner, expiry, rights, Profile and closure/conversion conditions. One or repeated watch never creates permanent locality selection or all-local-news scope.

### 7E — Provider and locality qualification seams

Implement provider and Locality Coverage Proposal/Decision interfaces without enabling a provider or selected locality. GDELT remains Held, Brave remains Rights Review Required, SearXNG and unofficial wrappers remain Research unless later exact decisions change them.

## Entry gate

Do not start Increment 7 until Increment 6 completes the normal supplemental-discovery and watch re-entry boundaries.

No actual search provider, recurring query, local source set or Event-Scoped Watch external access executes without:

- exact accepted purpose;
- rights and query-data approval;
- provider or source version;
- gross budget and hard amplification limits;
- Operational Profile;
- Evaluation Plan where prospective claims are intended; and
- explicit owner authority for the bounded run.

## Completion gate

Increment 7 closes when Agenda, search, audit and local-watch semantics pass deterministic fixture and replay evidence while every provider and permanent locality remains disabled unless separately qualified.

## Stop boundary

Do not begin Increment 8 release qualification until Agenda, search, Gap and locality records are versioned, bounded, observable and attributable enough to evaluate without hindsight or hidden provider behaviour.

---

# Increment 8 — Evaluation, operations, recovery and security

**Readiness state:** Prepared and dependency-blocked with numerical objectives deferred
**Dependency:** Increment 7 completed on `main`
**Runtime authority:** Qualification infrastructure only; no live shadow until a separate owner-approved Increment 9 plan

## Objective

Implement Evaluation Plans and Epochs, rights-limited review and metrics, Operational Profiles, scheduling and queue controls, observability, incidents, security, backup/restore, deterministic reconciliation, purge, fault injection, capacity/licence qualification and production-readiness decisions.

## Normative basis

- `DEVAL-*`;
- `DOPS-*`;
- `COV-*`, `SRC-*`, `SRCH-*`, `CAUD-*`, `LOC-*`, `TRI-080`–`TRI-085`;
- `GRAG-054`–`GRAG-058` and native GraphRAG release gates;
- accepted SDLC v2 evidence contract for repository machine gates.

## Prepared review units

### 8A — Evaluation authority and frozen Epochs

Deliver immutable Evaluation Plans, Epochs, Runs, Units, Cases, Label Sets, Adjudication Decisions, Metric Reports and Release-Evidence Decisions. Calibration and qualification remain separate.

### 8B — Review, metrics, slices and ablation

Deliver event-level prospective universes, contemporaneous/later labels, blinding, second review, adjudication, stage-specific metrics, required language/geography/urgency/source/transition slices, source contribution and retrieval ablation. No provider is ground truth.

### 8C — Operational Profiles and execution controls

Deliver exact source/provider/worker/queue/Handoff Profiles, Schedule Occurrences, leases, due-work identity, jitter, catch-up, retries, circuit breakers, quarantine, contingencies, fairness and capacity controls.

### 8D — Health, observability, incidents and security

Deliver multidimensional health, coverage availability, consequence-based alerts, version-attributed metrics/logs, runbooks, incident records, least-privilege credentials, egress restrictions, strict input containment and audited manual actions.

### 8E — Reconciliation, backup, restore, purge and recovery

Deliver deterministic reconciliation of leases, attempts, transitions, queues, Handoffs and projections; backup/restore integrity; post-restore reconciliation; graph/index rebuild; rights/privacy purge; tombstone non-resurrection; fault injection; and tested rollback.

### 8F — Capacity, licence and readiness decision

Bind intended hardware, p50/p95 latency, memory, disk, growth, rebuild, backup, reviewer burden, cost, queue headroom, urgent reserve, Neo4j/Graphiti licence and deployment/upgrade evidence into explicit release and Operational Admission decisions. Admission remains separate from activation.

## Entry gate

Do not start Increment 8 as a release-qualification programme until Increments 3–7 close. Bounded test-harness work may be separately authorised earlier only when it cannot execute sources or create release authority.

Every numerical threshold, timing objective, budget, reviewer rule and capacity target is owner-approved before qualification results are reviewed.

## Completion gate

Increment 8 closes only when the exact candidate system versions have reproducible fixture, replay, fault, rights-purge, backup/restore, security, performance, capacity, licence, monitoring, runbook and Operational Admission evidence sufficient to decide whether an Increment 9 shadow plan may be approved.

## Stop boundary

Do not run production-equivalent live shadow until Increment 8 completes and a separate owner-approved Increment 9 plan binds every exact version, source, right, budget, Profile, reviewer and production-equivalence difference.

---

# Increment 9 — Production-equivalent integrated shadow

**Readiness state:** Prepared with mandatory owner-decision gate
**Dependency:** Increment 8 completed on `main`
**Runtime authority:** None until a separate immutable owner-approved Evaluation Plan and shadow execution plan exist

## Objective

Run a bounded production-equivalent integrated shadow using the production-target GraphRAG stack and exact candidate source, extraction, retrieval, triage and operational versions, with no public effect and no production authority mutation.

## Required owner decision packet

Before implementation or execution, the Increment 9 plan must bind:

- exact Source Definition Versions and rights decisions;
- SQLite schema, governed-object contract and retention;
- Neo4j image/server/driver and deployment topology;
- ontology, projectors, indexes and generation contracts;
- Graphiti, model, prompt and embedding versions;
- relation/entity admission policy;
- named tools, retrieval policy, thresholds and budgets;
- Operational Profiles, schedules, queues, alerts and runbooks;
- Evaluation Plan, frozen Epoch, reviewers, blinding and adjudication;
- prospective Comparators and Search Purposes;
- required slices, blockers, non-zero thresholds and exposure conditions;
- source, provider, model, storage, reviewer and gross monetary budgets;
- licence, backup, restore, purge and intended-hardware evidence;
- production-equivalence statement listing every difference;
- early-stop, containment, rollback and incident rules; and
- explicit statement of no public effect.

## Prepared review units

### 9R — Owner-approved shadow plan

Documentation and decision only. No run starts before approval.

### 9A — Isolated shadow authority and deployment

Create an evaluation authority scope, secret and credential separation, no-public-effect controls, protected artefact storage and production-equivalence validation.

### 9B — Frozen live prospective execution

Execute the approved source portfolio, GraphRAG, extraction, triage and evaluation sink under the frozen Epoch. Material changes close the Epoch.

### 9C — Prospective comparator and fault phases

Run only pre-registered provider/audit methods and approved fault/degraded cases within rights and budgets.

### 9D — Review, ablation and release-evidence decision

Complete human review, adjudication, metrics, slices, source/retrieval ablation, cost and operational reporting. Retain failed and inconclusive Runs.

## Entry gate

Increment 9 remains blocked until all required inputs above are owner-approved and current. Adapter-only live checks do not satisfy this gate.

## Completion gate

Increment 9 closes only with an explicit retained outcome such as failed, inconclusive, continue shadow, Comparator-only, scoped operational eligibility or blocked by an Active-coverage deficiency. It never silently graduates into canary or production.

## Stop boundary

Do not begin Increment 10 until the exact system receives the required shadow and Operational Admission outcomes, all zero-tolerance findings are remediated, and Evidence Intake authority is accepted or explicitly owner-authorised.

---

# Increment 10 — Governed Evidence Intake canary

**Readiness state:** Prepared with normative and owner-decision blockers
**Dependency:** Increment 9 completed with qualifying outcome
**Normative blocker:** Downstream evidence, rights, sensitive-content and publication specifications in the editorial suite remain Draft unless separately accepted or explicitly authorised

## Objective

Hand a bounded set of exact Candidate Versions from the production-target GraphRAG deployment to a governed Evidence Intake canary. Add no direct drafting or publication path.

## Required authority before implementation

The owner must accept or explicitly authorise the minimum Evidence Intake contract covering:

- stable intake identity and exact Candidate Version correlation;
- permitted source retrieval and rights authority;
- Source Observation and Evidence Package boundaries;
- evidence sufficiency and claim admission separation;
- sensitive-content and escalation handling;
- retention, audit and lawful deletion;
- intake outcomes and structured discovery feedback;
- credentials, tools and external effects;
- canary scope, reviewer authority and rollback; and
- proof that discovery material is not evidence.

Until that authority exists, only transport-neutral interface and decision-template preparation is permitted.

## Prepared review units

### 10R — Evidence Intake and canary authority

Accept or explicitly authorise the minimum normative contract and one owner-approved canary plan.

### 10A — Durable Handoff transport and reconciliation

Implement exact Candidate Version submission, semantic idempotency, ambiguous-response reconciliation, acknowledgement, pending state and structured feedback without evidence or publication bypass.

### 10B — Bounded canary scope and containment

Bind exact Candidates, intake versions, reviewers, budgets, credentials, allowed source/evidence actions, no-public-effect controls, stop conditions and rollback.

### 10C — Canary execution and review

Run the bounded canary, retain every Candidate/Handoff/intake outcome, reconcile failures and report evidence-stage defects without rewriting discovery history.

### 10D — Canary decision

Record failed, inconclusive, continue canary, eligible for activation planning or another explicit outcome. Canary success is not production activation.

## Entry gate

Do not implement or execute Increment 10 beyond interface preparation until:

- Increment 9 qualifying evidence exists;
- Operational Admission remains current;
- Evidence Intake requirements are Accepted or explicitly authorised;
- exact rights, sensitive-content, retention and reviewer controls exist; and
- the canary plan is owner-approved.

## Completion gate

Increment 10 closes only after bounded Candidate handoff, intake acknowledgement, failure reconciliation, feedback preservation, rights and sensitive-content controls, rollback and explicit owner decision pass without publication or public effect.

## Stop boundary

Do not begin production activation or legacy retirement until Increment 10 closes and the publication, autonomy, rights, sensitive-content, lifecycle, dispatch and activation authorities required for public effects are Accepted or explicitly owner-authorised.

---

# Increment 11 — Bounded Hermes production activation

**Current readiness delta and first-scope proposal:** [issue #151](https://github.com/fol2/newsroom/issues/151)
**PRE151 dependency:** [#895 complete](https://github.com/fol2/newsroom/issues/895#issuecomment-5583562931); this is not production admission
**Readiness state:** Documentary reconciliation authorised; live qualification and activation retain their exact entry controls
**Specification status:** All eight publication-facing specifications are Accepted as checked on `fe01aee037f85f5a25f727810b17b730a1007ee9`; production policy and evidence currentness are separate
**Owner gate:** Formal 11R and exact production activation remain subject to #564 and the applicable admission/activation decisions

## Objective

Activate one exact admitted Hermes deployment with mandatory GraphRAG and governed downstream authority. The first public target is the integrated app-serving system and native readers. [ADR 0009](../adr/0009-legacy-operational-newsroom-dead.md) declares the legacy operational stack dead: no compatibility period, dual-write or historical import. Separately authorised tree deletion does not block 11B. RSS/Atom remains a valid Source Definition transport.

## Reuse and actual remaining evidence

The eight publication-facing specifications no longer present a Draft-status blocker. Their Accepted requirements still govern autonomy, evidence, presentation, rights/visuals, sensitive content, lifecycle/audit, publication engineering and quality/change control. Do not confuse accepted text with qualified production values or a deployed adapter.

The #557 decision inventory and 9Q qualification mechanisms are complete; #760 implements the production-admission mechanism. PRE151 proves the bounded admitted-only Graphiti outcome. None supplies a production freeze, a qualifying shadow/live-canary closeout, current production admission or an activation instruction by inheritance.

Use #151 for the exact remaining delta: deployment/Operational Profile and seven evaluated identities; current runtime, rights/terms, credentials and egress; qualifying shadow/canary; production coverage and spend accounting; recovery and public-effect readiness; and the actual admission/activation decisions. Reuse evidence with valid identity/provenance. The twenty First I/O records and sixteen production-admission checks are separate inventories, not thirty-six new engineering milestones.

The completed PRE151 cohort does not turn visible historical holds into terminal production coverage or settle unrelated spend. Production coverage and accounting must satisfy their own typed contracts; scope changes require explicit qualification, not omitted rows or invented zero balances.

## Activation binding

One exact manifest and its evidence must bind:

- command/authentication versions, schema/migrations, authoritative objects and store identity;
- source portfolio, Operational Profile, rights, credentials and egress;
- ontology, real Neo4j, governed projectors, isolated Graphiti proposals, admission and indexes;
- models, prompts, embeddings, chunking, normalisation and production retrieval;
- triage/Candidate, Evidence Intake, publication adapters and handoff non-effect identities;
- the Accepted publication specifications and their applicable production policy values;
- deployment hardware, service roles, capacity, monitoring, incident controls and finite allocations;
- qualifying shadow/canary evidence and tested, identity-matched backup/restore/rollback; and
- exact activation scope, start/stop conditions, public-effect reconciliation and recovery target.

A manifest missing, disabling, faking or incompatibly replacing mandatory GraphRAG is invalid. A no-writer daemon cycle is only runtime qualification, not a complete GraphRAG shadow, canary or production slice. The existing complete path and #151's smallest scope proposal remain the reference; do not introduce a graphless first-production stage.

## Review and effect boundaries

These labels do not require a separate issue or PR per item:

- **11R — Formal activation-plan decision:** the separate owner decision after the qualifying evidence and production admission required by #564. Readiness reconciliation or a scope proposal is not this decision.
- **11A — Exact manifest and readiness enforcement:** reuse the implemented contracts; change only demonstrated gaps in identity, GraphRAG, role, schema or recovery enforcement.
- **11B — Bounded production activation:** activate only the exact approved scope with acknowledgement, reconciliation, containment, monitoring and rollback. No implicit expansion follows success.
- **11C — New-system pending-effect reconciliation:** not legacy drain, compatibility, dual-write or historical import.
- **11D — Remaining in-scope retirement decision:** only for components actually still in scope; dead legacy code deletion is separate and not a prerequisite to 11B.

## Entry and completion

The current owner instruction permits #151's readiness reconciliation and first-scope proposal without first approving the finished activation plan. Ordinary changes follow current Focus Gates and one feature-complete review, not repeated research/review councils or Full Repository Health in the critical path.

Actual live qualification and production entry retain the accepted #564 sequence: exact minimum Hermes runtime and freeze/First I/O records; a `SCOPED_OPERATIONAL_ELIGIBILITY` shadow; an `ELIGIBLE_FOR_ACTIVATION_PLANNING` live Evidence Intake canary; current production admission binding the seven identities; then a separate formal 11R decision and exact activation authority. Current rights, security, coverage, spend and tested recovery/public-effect controls remain required at their affected boundaries. The historical Increment 10 `BLOCKED_ACTIVE_COVERAGE` fixture disposition is not rewritten by PRE151.

Close #151 only after its exact approved deployment scope is activated, public effects reconcile, monitoring/recovery remain healthy and signed closeout accurately states active, held and excluded scope. Legacy resurrection or deletion is not an added completion requirement. Full seal under 60 seconds remains a non-blocking efficiency backlog, not a new planning or PRE151 gate.

## Stop boundary

This roadmap update performs no live/provider operation, production-admission mint, activation, publication, deletion or public dispatch. No successful bounded start grants automatic expansion to another source, locality, model, provider, publication surface or public-effect scope. Material changes return only to the affected accepted change-control, evaluation and effect boundaries.

---

## 4. Prepared issue posture

The earlier prepared issue snapshots are retained below, except for the reconciled Increment 11 entry. Current issues, not these historical snapshots, determine live status:

- Increment 2: open and ready for owner review; implementation blocked until the Increment 2 readiness package is accepted.
- Increment 3: open, `Prepared / blocked by Increment 2`.
- Increment 4: open, `Prepared / blocked by Increment 3`.
- Increment 5: open, `Prepared / blocked by Increment 4 and production retrieval contract decision`.
- Increment 6: open, `Prepared / blocked by Increment 5`.
- Increment 7: open, `Prepared / blocked by Increment 6 and provider/locality authority`.
- Increment 8: open, `Prepared / blocked by Increment 7 and numerical Evaluation/Operational Plan decisions`.
- Increment 9: open, `Decision packet prepared / blocked by Increment 8 and separate owner-approved shadow plan`.
- Increment 10: open, `Contract gate prepared / blocked by Increment 9 and accepted Evidence Intake authority`.
- Increment 11: open; current readiness delta and proposed first scope are in #151. PRE151 is complete; formal production qualification/admission and activation are distinct outstanding boundaries.

A blocked production effect must not be represented as authorised. Separately scoped readiness/local engineering may proceed under current issue intent without claiming the production gate has passed.

## 5. Programme-level completion rule

The accepted programme is not “finished” merely because documents and blocked issues exist. It is finished only when:

1. Increments 2–8 are implemented, reviewed and merged with exact evidence;
2. Increment 9 completes an owner-approved production-equivalent shadow with an explicit qualifying decision;
3. Increment 10 completes a governed Evidence Intake canary under accepted authority;
4. Increment 11 receives exact production activation authority and completes its approved Hermes scope, with any genuinely in-scope retirement handled under ADR 0009; and
5. every Accepted requirement traces to code, configuration, evidence, control or a documented retained procedure with no unresolved blocker hidden by aggregate success.
