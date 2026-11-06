# Agent 0 architecture

## System boundary

Agent 0 is a batch CLI for distributor discovery around one Tata Steel depot.
It accepts a GST-registry CSV and optionally an existing Tata-distributor CSV.
It writes auditable queue artifacts; it does not create CRM records, onboard
partners, or make autonomous commercial commitments.

```text
GST registrations
  │
  ├── normalize identity, turnover, address, explicit coordinates
  ├── geocode with provenance and quality
  ├── deterministic eligibility
  │     ├── eligible
  │     ├── needs data
  │     └── rejected
  ├── bounded Firecrawl source collection
  ├── single nested ToT + Stage 2 Thinking response
  ├── deterministic citation validation
  ├── deterministic scoring and queue routing
  └── CSV / JSON queues + reproducibility manifest
```

## Responsibility boundaries

| Layer | Module | Responsibility |
|---|---|---|
| Input normalization | `csv_loader.py` | GSTIN validation, CSV-header handling, turnover parsing, manual coordinate extraction |
| Spatial evidence | `geocoder.py`, `spatial.py` | Provenance-labelled geocoding and distance calculations |
| Eligibility | `prefilter.py` | Hard factual rejects versus records that need more data |
| Web retrieval | `enrichment.py` | Bounded Firecrawl search, source-preserving cache, citation validation |
| Qualitative analysis | `agent.py` | One nested structured ToT/S2T provider response |
| Commercial policy | `scoring.py` | Deterministic scoring, hard disqualification, grade and queue assignment |
| Orchestration | `pipeline.py` | Stage sequencing, budgets, manifests, and queue ordering |
| Presentation | `cli.py` | Argument validation and safe export |

## Queue model

### Deterministic eligibility

The prefilter evaluates all factual gates and produces one of:

- `ELIGIBLE`: validated GSTIN, active registration, product/activity fit,
  known qualifying turnover, and a trusted location that passes spatial gates.
- `NEEDS_DATA`: plausible record whose turnover or trusted location is absent.
  It is retained rather than pessimistically rejected.
- `REJECTED`: known invalid identity, inactive registration, product/activity
  mismatch, too-small turnover, configured turnover breach, cannibalization, or
  trusted-location distance breach.

Only `MANUAL`, `PIN_EXACT`, and `NOMINATIM` locations are trusted for hard
spatial gates. `PIN_DISTRICT` and `STATE_CENTROID` are explicitly approximate
and route to `NEEDS_DATA`.

### Evidence and nested ToT/S2T

Firecrawl search returns individual source records:

```text
source_id, URL, title, bounded content
```

The LLM receives those records as untrusted documents. It must return:

```text
tot.brand_alignment
tot.infrastructure
tot.onboarding_risk
stage_2.supported_claims
stage_2.challenged_claims
stage_2.unresolved_questions
stage_2.final_rationale
```

Each ToT branch carries source-ID citations with exact excerpts. Stage 2
adversarially reconciles the three branch views. This is intentionally one
structured call for compatibility and bounded cost, not parallel multi-call
reasoning.

`validate_evidence_citations` independently checks that cited source IDs exist
and quoted excerpts occur in retained source content. A verified result needs a
complete nested structure, at least two valid citations, and evidence in both
brand and infrastructure branches.

### Grades and commercial queues

Scores are deterministic once their inputs are fixed:

```text
profit = logistics × 0.45 + turnover × 0.35 + processing × 0.20
ease   = compliance × 0.40 + switching × 0.35 + infrastructure × 0.25
```

The grade ladder is applied after hard disqualification:

- `is_primary_steel_dealer == false` → rejected Grade D.
- `gst_filing_regularity == CANCELLED` → rejected Grade D.
- A/B require citation-validated evidence.
- Any non-verified A/B candidate is capped at C and becomes a prioritized lead.

The final ordering is queue first, then grade, then scores. An unverified
high-score lead cannot outrank a verified A/B recommendation as if the two had
equal commercial readiness.

## Network modes and spend boundaries

| Integration | Enabled in connected mode | Disabled by `--offline` |
|---|---:|---:|
| Nominatim address/PIN lookup | `--nominatim` | Yes |
| Firecrawl search | API key + no `--skip-enrichment` | Yes |
| LLM verification | provider key + no `--skip-llm` | Yes |
| Google Maps route refinement | API key + `gmaps_refine_top > 0` | Yes |

The enrichment budget limits Firecrawl and LLM candidates. The Maps budget
limits exact route calls. Source cache entries use a hashed identifier and
retain per-result provenance; stale or legacy flattened entries are refreshed.

## Reproducibility

Every run produces `agent0_run_manifest.json` containing:

- depot and run timestamp;
- SHA-256 of distributor and existing-network inputs;
- code revision when available;
- decision configuration without secrets;
- mode/options;
- queue counts.

Live evidence and model output remain time-dependent. The manifest and retained
source artifacts make a run reviewable, but they do not claim a web-connected
run is deterministic across time or providers.

## Configuration invariants

Configuration rejects:

- unknown keys;
- credentials in YAML;
- negative budgets or TTLs;
- invalid provider names;
- invalid score thresholds;
- weights that do not sum to one;
- invalid turnover ranges.

API credentials remain environment-only. The `config.yaml` file contains
decision parameters, not secrets.

## Deferred coverage strategy

All distributor sizes above the minimum threshold remain eligible. The
repository intentionally does not yet encode regional white-space, size-band,
territory, or supply-capacity optimization. That requires a separate validated
coverage strategy rather than an arbitrary upper-turnover penalty.
