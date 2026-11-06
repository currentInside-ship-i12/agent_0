# Agent 0 — Product brief

## Problem

Tata Steel needs a repeatable way to turn a bulk GST registry into a local
distributor-discovery funnel without confusing incomplete public information
with commercial proof.

The question is:

> Which entities near a target depot are plausible distributor opportunities,
> which need more data, and which have enough source-backed evidence to justify
> commercial outreach?

## Product contract

Agent 0 does not claim that every registry row is a verified distributor. It
routes each record into an explicit queue:

| Queue | Meaning | Commercial action |
|---|---|---|
| Verified recommendation | Eligibility passed and the ToT/S2T output has validated source citations | Grade A/B outreach |
| Prioritized lead | Eligibility passed, but web evidence is absent, weak, or outside budget | Field verification |
| Needs data | Identity, turnover, or location is uncertain | Data acquisition |
| Rejected | A known hard rule failed | Do not pursue without remediation |

## Design principles

- **Optimistic discovery:** missing information preserves a potential lead; it
  does not silently become a negative business fact.
- **Conservative recommendation:** only source-backed, citation-validated
  evidence can produce Grades A/B.
- **Deterministic policy:** gates, citation validation, scores, queue routing,
  and exports are ordinary Python rules.
- **Explicit uncertainty:** district and state-centroid locations cannot decide
  hard spatial eligibility.
- **Reproducibility:** every run writes a manifest with input hashes, config,
  options, depot, code version, and queue counts.

## Current scoring policy

`Profit score` combines logistics, turnover, and processing capability.
`Ease score` combines compliance, switching feasibility, and infrastructure.

Large distributors remain eligible and use the same top turnover plateau as
the established target band. A regional market-coverage strategy—such as
balancing size bands, territories, and white-space—is deliberately deferred to
a separate product iteration rather than hidden in the current score.

Known cancelled filing status and non-dealer verification are hard rejections.
Unknown filing status is not a positive compliance claim.

## Nested ToT and Stage 2 Thinking

The implementation intentionally uses one structured LLM request per candidate
for now:

1. `tot.brand_alignment`
2. `tot.infrastructure`
3. `tot.onboarding_risk`
4. `stage_2` adversarial reconciliation

Each branch must cite retained Firecrawl source IDs with exact excerpts. Stage 2
records supported claims, challenged claims, unresolved questions, and a final
rationale. The model may interpret evidence, but deterministic code checks that
quoted excerpts exist in the cited sources before the output is called
verified.

## Deliberately deferred

- A top-tier regional coverage/white-space strategy for large distributors.
- Database/CRM persistence and historical opportunity tracking.
- Parallel multi-call ToT; the current implementation uses one nested
  structured call to contain cost and preserve broad model compatibility.

## Operating modes

- `--offline`: zero network calls; generates leads, needs-data, and rejection
  artifacts without pretending they are verified.
- Connected mode: optional Nominatim geocoding, Firecrawl evidence collection,
  configured LLM verification, and bounded Google Maps route refinement.

See [`README.md`](./README.md) for commands and
[`ARCHITECTURE.md`](./ARCHITECTURE.md) for responsibility boundaries.
