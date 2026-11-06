# Agent 0 — Architecture & Product Overview

> Tata Steel GTM Distributor Search & Scoring Engine
> Generated from a full read of `src/agent_0/`, `tests/`, `config.yaml` and a
> live offline run against `data/distributors.csv` (500 rows).

---

## 1. What the product is

**Agent 0** is a command-line pipeline that answers one commercial question for
Tata Steel's go-to-market team:

> *"Which new steel distributors should we onboard near a given plant, and in
> what order?"*

It ingests a bulk GST (Goods and Services Tax) registry export of Indian
businesses, screens it down to entities that plausibly distribute steel,
enriches the survivors with public web evidence, uses an LLM to qualitatively
verify each one, and produces a **ranked, graded shortlist** with a full
audit trail of why each entity was kept or dropped.

The output is a **decision artefact**, not a report. Every row of
`agent0_results.csv` is a distributor with a letter grade (A–D), a profit
score, an ease-of-onboarding score, and the evidence that justifies both.

### Intended users

| Persona | How they use it |
|---|---|
| **GTM / channel manager** | Runs the pipeline for a depot, reads the top-20 rows, dispatches field teams to the A/B grades. |
| **Regional sales head** | Runs the pipeline per region, compares grade distributions across depots. |
| **Data/ops analyst** | Tunes `config.yaml` gates and weights, re-runs, diffs the shortlists. |

### The commercial logic it encodes

A candidate is worth onboarding when three things hold together:

1. **It is a real steel distributor** — right HSN chapters, right business
   activity, active GST registration.
2. **It is worth the logistics cost** — close enough to a Tata plant to serve
   economically.
3. **It is easy to onboard** — clean compliance record, no exclusivity lock-in
   with a competitor steel mill, real physical infrastructure.

Plus one channel-hygiene constraint: **do not cannibalise an existing Tata
distributor.** Candidates within a minimum radius of the current network are
rejected outright.

### What it is explicitly not

- Not a lead-scraping tool. It never invents candidates; it only ranks the
  ones in the input CSV.
- Not a replacement for field verification. Grades C and D are explicitly
  labelled "field verification needed" / "disqualified".
- Not autonomous. There is no scheduler, no database, no dashboard, no API
  server. It is a batch CLI that writes CSV/JSON.

---

## 2. The pipeline

```
 data/distributors.csv
          │
          ▼
 ┌────────────────────┐
 │ 1. INGEST          │  csv_loader.py
 │    parse rows      │  · header-variant resolution
 │    parse slabs     │  · turnover slab → ₹ Cr midpoint
 │    dedup by PAN    │  · keep highest-turnover record
 └─────────┬──────────┘
           ▼
 ┌────────────────────┐
 │ 2. GEOCODE         │  geocoder.py
 │    PIN → offline   │  · exact 6-digit table
 │    PIN → prefix    │  · 3-digit district table
 │    PIN → Nominatim │  · optional, 1 req/sec
 │    else centroid   │  · state capital fallback
 └─────────┬──────────┘
           ▼
 ┌────────────────────┐
 │ 3. PRE-FILTER      │  prefilter.py       ← zero LLM
 │    8 hard gates    │  · deterministic reject / pass
 └─────────┬──────────┘
           ▼  (passed candidates only)
 ┌────────────────────┐
 │ 4. PRIORITISE      │  pipeline.py
 │    top-N by        │  · top_n_for_enrichment (default 50)
 │    turnover        │
 │    GMaps budget    │  · gmaps_refine_top (default 5, closest N)
 └─────────┬──────────┘
           ▼
 ┌────────────────────┐
 │ 5. ENRICH          │  enrichment.py       ← Firecrawl
 │    web search      │  · 5 results/candidate, 3 workers
 │    disk cache      │  · .agent0_cache/{GSTIN}.json, 30d TTL
 └─────────┬──────────┘
           ▼
 ┌────────────────────┐
 │ 6. VERIFY          │  agent.py            ← LLM (ToT + S2T)
 │    3 branches      │  · Brand alignment
 │    + 3 S2T steps   │  · Infrastructure
 │                    │  · Onboarding risk
 │                    │  · Evidence grounding + falsification
 └─────────┬──────────┘
           ▼
 ┌────────────────────┐
 │ 7. SCORE & GRADE   │  scoring.py          ← zero LLM
 │    profit score    │  · 3 weighted sub-scores
 │    ease score      │  · 3 weighted sub-scores
 │    letter grade    │  · A/B/C/D thresholds
 └─────────┬──────────┘
           ▼
   agent0_results.csv / .json  +  rich console summary
```

### Design invariant

The LLM is confined to **one** stage. Every gate, every score, every grade is
computed by deterministic Python. This is a deliberate and correct separation:

- Gates and scores are **auditable** and **reproducible** — same input, same
  output, always.
- The LLM's job is judgement (is this actually a steel dealer? does it have a
  yard?), which is exactly what a rule cannot express.
- A failed LLM call degrades *qualitative confidence*, never the arithmetic.

> **Caveat — this invariant is currently violated in effect.** See
> [`REVIEW.md` § R1](./REVIEW.md#r1-critical-unverified-candidates-can-be-graded-b).
> The safe-default S2T values are optimistic enough that an entirely
> unverified candidate can reach Grade B, so the LLM stage is not yet
> load-bearing in the way the design intends.

---

## 3. Module map

| Module | Responsibility | LLM? | Network? |
|---|---|:--:|:--:|
| `cli.py` | Typer entry point, arg parsing, config load, export | – | – |
| `config.py` | YAML + `.env` → `PipelineConfig` (pydantic) | – | – |
| `models.py` | All data contracts (pydantic v2) | – | – |
| `csv_loader.py` | Raw CSV → `DistributorCandidate` | – | – |
| `geocoder.py` | Candidate → lat/lon + `GeoSource` | – | Nominatim |
| `spatial.py` | Haversine, tortuosity estimate, GMaps distance, cannibalisation | – | GMaps |
| `prefilter.py` | 8 hard gates → pass/fail + reasons | – | – |
| `enrichment.py` | Firecrawl web search + disk cache | – | Firecrawl |
| `agent.py` | ToT + S2T verification, 3 provider adapters | ✅ | Anthropic/OpenAI/Gemini |
| `scoring.py` | Profit/ease sub-scores → weighted score → grade | – | – |
| `pipeline.py` | Orchestrator, phase sequencing, summary output | – | – |

### Data contracts (`models.py`)

```
DistributorCandidate          normalised input record (48 fields)
  └─ PrefilterResult          pass/fail + rejection reasons + distances
       └─ FinalEvaluation     the shipped row: scores, grade, evidence
            └─ S2TVerificationOutput   the LLM's structured verdict
            └─ ScoreBreakdown          per-component score decomposition
```

Two design choices worth noting:

- **`S2TVerificationOutput` uses `Literal` types, never free-form floats.**
  The LLM can only return `"HIGH" | "MEDIUM" | "LOW"`, and confidence is a
  discrete bucket (`STRONG/MODERATE/WEAK/NONE`) mapped to a score via
  `CONFIDENCE_SCORE_MAP`. This eliminates calibration drift across runs and
  makes the JSON schema a valid strict-schema contract for all three
  providers.
- **`ScoreBreakdown` keeps every component.** A grade is never a black box:
  the CSV carries `S_Logistics`, `S_Turnover`, `S_Processing`, `E_Compliance`,
  `E_Switching`, `E_Infra` alongside the totals.

---

## 4. The scoring model

### 4.1 Profit score — "is this worth our logistics spend?"

```
S_profit = 0.45·S_logistics + 0.35·S_turnover + 0.20·S_processing
```

**S_logistics** — linear decay with distance to the depot:

```
S_logistics = clamp(100 · (1 − distance_km / 150), 0, 100)
```

`150 km` is `spatial.d_target_max_km`. Beyond it the sub-score is 0.

**S_turnover** — a piecewise curve over the turnover slab midpoint (₹ Cr):

| Turnover (₹ Cr) | S_turnover | Rationale |
|---|---|---|
| 0 – 15 | `25 · (T/15)` → 0 → 25 | ramp from zero |
| 15 – 50 | `25 + 75·((T−15)/35)` → 25 → 100 | the target band |
| 50 – 150 | 100 | plateau: ideal size |
| 150 – 300 | `max(40, 100 − (T−150)/3)` → 100 → 50 | taper: too big to need us |
| > 300 | 0 | unreachable via the gate (see REVIEW R5) |

**S_processing** — from the LLM's `processing_capability`:
`ADVANCED = 100`, `BASIC = 60`, `NONE = 20`.

### 4.2 Ease score — "how hard is this to onboard?"

```
S_ease = 0.40·E_compliance + 0.35·E_switching + 0.25·E_infrastructure
```

**E_compliance** — from `gst_filing_regularity` × `promoter_litigation_flag`:

| Filing | No litigation | Litigation |
|---|---|---|
| `REGULAR` | 100 | 30 |
| `IRREGULAR` | 50 | 50 |
| `CANCELLED` | 0 | 0 |
| `UNKNOWN` | 70 | 20 |

**E_switching** — from `switching_feasibility`:
`HIGH = 100`, `MEDIUM = 65`, `LOW = 30`.

**E_infrastructure** — stockyard × processing capability:

| Stockyard | ADVANCED | BASIC | NONE |
|---|---|---|---|
| Yes | 100 | 75 | 50 |
| No | 15 | 15 | 15 |

### 4.3 Grading

Applied in order; the first match wins.

| # | Condition | Grade |
|---|---|---|
| 1 | `profit ≤ 40` **and** `ease ≤ 40` | **D** |
| 2 | `profit ≥ 75` and `ease ≥ 75` and `confidence ≥ 0.80` and is a steel dealer | **A** |
| 3 | `profit ≥ 70` and `ease ≥ 50` and is a steel dealer | **B** |
| 4 | `profit ≥ 50` and is a steel dealer | **C** |
| 5 | everything else | **D** |

Hard disqualifiers that force D regardless of score:
`is_primary_steel_dealer == False`, `gst_filing_regularity == "CANCELLED"`.

R1 policy: `evidence_confidence == "NONE"` (skipped, beyond top-N, or all
providers failed) caps the grade at **C** and sets
`verification_status` to `UNVERIFIED` or `FAILED`. Exported as
`Verification Status`. Verified candidates with real Firecrawl enrichment
are `VERIFIED`; LLM-verified without enrichment are `PARTIAL`.

Confidence comes from the discrete bucket map:

| Bucket | Score |
|---|---|
| `STRONG` | 0.90 |
| `MODERATE` | 0.65 |
| `WEAK` | 0.35 |
| `NONE` | 0.10 |

---

## 5. The pre-filter gates

All eight gates run in `prefilter.py` with zero LLM involvement. A candidate
must clear all of them; every failure reason is retained (a candidate can fail
several gates).

| Gate | Rule | Failure message |
|---|---|---|
| a | GSTIN is exactly 15 chars | `Invalid GSTIN length` |
| b | `REGISTRATION STATUS == "Active"` | `Registration status not active` |
| c | At least one HSN starts with a required prefix, and not SAC-only | `Missing steel HSN …` |
| d | `NATURE OF CORE BUSINESS ACTIVITIES` ∈ allowed list | `Core business '…' not in allowed activities` |
| e | Turnover midpoint in `[15, 300]` ₹ Cr — **0.0 (unknown) passes** | `Turnover ₹…Cr outside allowed range` |
| f | Lat/lon present | `Missing geocoded coordinates` |
| g | ≥ 30 km from every existing Tata distributor | `Cannibalization conflict` |
| h | Coarse road distance to depot ≤ 180 km | `Coarse road distance … exceeds maximum` |

---

## 6. Configuration

Three layers, highest priority first:

```
CLI flags  ──►  .env / OS environment  ──►  config.yaml  ──►  code defaults
   (run-time)      (API keys only)          (parameters)       (Pydantic defaults)
```

`config.yaml` is the parameter surface. It is validated by
`PipelineConfig` with `extra="forbid"` at **every** level, so an unknown or
misspelled key is a hard error rather than a silent fallback to a default.

| Section | Controls |
|---|---|
| `gates` | Pre-filter thresholds — turnover range, max distance, cannibalisation radius, HSN prefixes, allowed business activities |
| `spatial` | Tortuosity factor, distance at which logistics hits 0 |
| `profit_weights` / `ease_weights` | Sub-score weighting (must sum to 1.0) |
| `grading` | A/B/C/D thresholds |
| `enrichment` | Top-N budget, GMaps refinement budget, cache dir + TTL |
| `llm` | Provider, model, temperature, max tokens |

API keys are read from the environment only (never from YAML), and are
excluded from `repr` so they cannot leak into logs or tracebacks.

---

## 7. The LLM stage in detail

### 7.1 Tree of Thoughts — three parallel branches

| Branch | Question | Output fields |
|---|---|---|
| **1. Brand alignment** | Which steel brands does it carry? Is it locked to a competitor? How easy to switch? | `brands_currently_handled`, `is_exclusive_to_competitor`, `switching_feasibility` |
| **2. Infrastructure** | Does it have a real yard? Does it process steel? | `has_physical_stockyard`, `processing_capability` |
| **3. Onboarding risk** | Is it a genuine stockist or a paper company? Any litigation, disqualification or irregular filing? | `promoter_litigation_flag`, `gst_filing_regularity` |

### 7.2 Stage 2 Thinking — three steps

1. **Evidence grounding** — every claim tied to verbatim quotes/URLs from the
   scraped text. No extrapolation.
2. **Adversarial falsification** — actively try to *disprove* the candidate:
   name collisions, scrap traders posing as stockists, pure logistics
   providers.
3. **Strict structured output** — emit the verdict.

### 7.3 Provider adapters

| Provider | Transport | Structured output |
|---|---|---|
| Anthropic | `anthropic` SDK | forced `tool_use` (`emit_verification`) |
| OpenAI | `openai` SDK | `response_format=json_schema`, `strict=true` |
| Gemini | `openai` SDK against Gemini's OpenAI-compatible endpoint | `response_format=json_schema`, `strict=true` |

All three converge on the same `S2TVerificationOutput` pydantic model, so
`scoring.py` is provider-agnostic.

Failure handling: if the primary provider fails (quota, outage, malformed
response), the remaining providers with a configured key are tried in turn.
If every provider fails, a conservative safe default is returned and the
failure is recorded in `verbatim_evidence`.

---

## 8. External dependencies & cost profile

| Service | Purpose | When called | Cost control |
|---|---|---|---|
| **Firecrawl** | Web search + scrape | Once per enriched candidate | `top_n_for_enrichment` (50), 5 results each, 3 concurrent workers, 30-day disk cache |
| **Anthropic / OpenAI / Gemini** | ToT + S2T verification | Once per enriched candidate | `top_n_for_enrichment` (50) |
| **Google Maps** | True driving distance | Only for the `gmaps_refine_top` (5) closest candidates | `gmaps_refine_top`, `0` disables |
| **Nominatim** | PIN → coordinates | Only when `--nominatim` is passed and the PIN is not in the offline table | 1 req/sec, disk cache, opt-in |

The two spend governors (`top_n_for_enrichment`, `gmaps_refine_top`) exist so
that a single run over a large CSV has a predictable, bounded cost.

---

## 9. Known architectural limitations

These are structural properties of the current design, not bugs. They are
expanded with evidence and proposed fixes in [`REVIEW.md`](./REVIEW.md).

| # | Limitation | Impact |
|---|---|---|
| L1 | **Geocoding is a hardcoded lookup table** — 12 exact PINs, 21 3-digit prefixes, then a state-capital centroid fallback | 56% of the sample dataset resolves to a *state centroid*, not a real location. Distances — worth 45% of the profit score — are therefore meaningless for most candidates, and the 180 km gate rejects most of India for any single depot. |
| L2 | **The geocode provenance is captured but never used** | The pipeline knows a candidate is a centroid guess (`geocode_source`) but the gate and the score ignore it. *(Partially addressed: `geocode_source` is now exported — see REVIEW R4.)* |
| L3 | **The safe-default S2T is optimistic** | An unverified candidate scores `ease = 54.5`, clearing the Grade B threshold of 50. A candidate that failed every LLM call can be labelled "B — target for multi-brand onboarding". |
| L4 | **Verification is serial** | The LLM loop runs one candidate at a time. 50 candidates ≈ 4–8 minutes of serial latency, while enrichment runs 3-wide in parallel. |
| L5 | **PAN deduplication happens before geocoding** | A multi-state company is represented by whichever registration has the highest turnover — which may be in a different state than the depot, so the candidate is dropped from that depot's shortlist. |
| L6 | **No run manifest** | The output records no config, no timestamp, no depot, no input hash, no provider. Two runs cannot be compared or reproduced. |
| L7 | **No persistence layer** | Every run starts from scratch. There is no database of past results, no candidate history, no pipeline-to-CRM hand-off. |
| L8 | **CLI only** | No API, no UI, no scheduling. Results are consumed by opening a CSV. |

---

## 10. Extension points

| To add… | Touch |
|---|---|
| A new LLM provider | `agent.py`: implement `_run_<provider>`, register in `_dispatch_provider` and `_providers_with_keys` |
| A new data source (e.g. Udyam, TradeIndia) | `csv_loader.py`: extend `parse_distributor_row` + `_col` header variants |
| A new gate | `prefilter.py::_evaluate_candidate` — append to `rejection_reasons`, add a label to `REASON_CATEGORIES` |
| A new score sub-component | `scoring.py`: add to `_compute_profit_score` / `_compute_ease_score`, add a weight in `config.yaml`, add a field to `ScoreBreakdown` |
| A new grade rule | `scoring.py::score_candidate` — extend the ordered `if/elif` chain, add thresholds to `GradingConfig` |
| Real geocoding | `geocoder.py`: replace the offline tables with a PIN database or a geocoding API; `GeoSource` already models the provenance |

---

## 11. Running it

```bash
pip install -e ".[dev]"
cp .env.example .env          # fill in API keys
agent-0 --help

# Fully offline — deterministic, no API spend
agent-0 run --distributor-csv data/distributors.csv \
            --depot-name "Delhi Hub" --depot-lat 28.6139 --depot-lon 77.2090 \
            --skip-enrichment --skip-llm --format csv --output out/

# Full pipeline
agent-0 run --distributor-csv data/distributors.csv \
            --depot-name "Jamshedpur" --depot-lat 22.80 --depot-lon 86.20 \
            --tata-csv existing_tata_distributors.csv \
            --format all --output out/

agent-0 info                   # effective configuration
```

`--verbose` surfaces LLM and enrichment failures on stderr — by default those
are logged at WARNING level and would otherwise be easy to miss.

---

## 12. Testing

`tests/test_agent_0.py` — 31 tests, plus `tests/stress_mocked.py` — 22 mocked stress tests, all deterministic and network-free. The
network boundary is stubbed; everything else is real code.

| Area | Covered |
|---|---|
| Turnover parsing | Unit inheritance, lakh/crore conversion, "and above" slabs, real Clear GST export formats |
| Geocoding | Offline PIN resolution, Haryana prefix table, manual-coordinate passthrough |
| Scoring | Continuity of the turnover curve, ramp anchors, grade lift with strong evidence |
| CSV parsing | Actual-data header variants (`EINVOICESTATUS` vs `EINVOICE STATUS`), field passthrough |
| Enrichment | Firecrawl request shape, response parsing, 5 000-char truncation, cache write, **no-evidence payloads are not cached** |
| LLM | Anthropic workspace header, Gemini endpoint/model/schema, provider dispatch, **provider fallback on failure**, **all-providers-failed safe default** |
| Pipeline | Happy path, GMaps refinement budget split, `gmaps_refine_top=0`, **invalid `--format` rejection** |
| Config | Gemini key precedence, **unknown-key rejection** |

---

## 13. Related documents

- [`REVIEW.md`](./REVIEW.md) — defect register from the architecture review,
  with severity, evidence, reproduction and remediation status.
