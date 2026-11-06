# Agent 0 — Tata Steel GTM Distributor Search & Scoring Engine

> **Resume / CV one-liner**
> *Built agent-0: an end-to-end LLM-augmented decision engine that screens a 500-row GST steel registry through deterministic hard gates, budgets paid enrichment APIs (Firecrawl, Google Maps), verifies shortlisted candidates via Tree-of-Thoughts LLM reasoning with structured output, and emits graded, fully auditable distributor recommendations with provenance columns, spend governors, and fail-safe defaults.*

## What it does

Agent 0 turns a bulk GST registry export into a ranked, graded distributor
shortlist for Tata Steel's go-to-market expansion. It answers:
**"Which registered entities should Tata Steel target as new steel
distributors around a given depot, and in what order?"**

The system is deliberately layered so that every recommendation carries its
audit trail: hard filters, distance math with spend budgets, public-web
evidence, LLM verification with self-critique, and a deterministic scoring
model — each stage's metadata lands in the output CSV/JSON.

## Pipeline at a glance

| Phase | Module | What happens |
|---|---|---|
| 1. Ingestion | `csv_loader.py` | Parse GST CSV (500 rows), decode turnover slabs / HSN / activity codes, **PAN-level deduplication** |
| 2. Geocoding | `geocoder.py` | Address → coordinates via an offline PINCODE table (Python pincode lib), optional Nominatim fallback |
| 3. Hard-gate prefilter | `prefilter.py` | 8 deterministic gates: active status, HSN 72/73 (steel), business activity (MFT/TRD:TWD/TRD:TRR), turnover 15–300 Cr, ≤180 km coarse road distance, ≥30 km cannibalization radius vs. existing distributors, mandatory coordinates |
| 4. Batch selection | `pipeline.py` | Sort passers by turnover (desc), take **top-N** for enrichment (`top_n_for_enrichment`), then choose the **closest-N** by coarse distance for GMaps refinement (`gmaps_refine_top`) — a spend governor, parallel to the enrichment budget |
| 5. Web enrichment | `enrichment.py` | Firecrawl searches per candidate with disk cache (`.agent0_cache`, 30-day TTL); graceful "no evidence" and MOCK fallbacks |
| 6. LLM verification | `agent.py` | Tree-of-Thoughts (brand / infrastructure / risk branches) → **S2T** stage (grounding + adversarial falsification), structured enum output with discrete confidence (`STRONG / MODERATE / WEAK / NONE`); providers: Anthropic Claude, OpenAI-compatible, Gemini; any failure → safe defaults, never raises |
| 7. Scoring & grading | `scoring.py` | Deterministic **Profit score** (logistics 0.45 / turnover 0.35 / processing 0.20) and **Ease score** (compliance 0.40 / switching 0.35 / infrastructure 0.25) → grade ladder **A / B / C / D** with a D-ceiling for critically low scores |
| 8. Export | `cli.py` | `agent0_results.csv` + `.json` (+ rich tables): grade, every sub-score, distance *and its source* (GMAPS vs HAVERSINE), geocode provenance, LLM verdict with verbatim evidence, rejection reasons |

## Key design decisions

- **Spend governors at two levels.** `top_n_for_enrichment` budgets Firecrawl
  calls; `gmaps_refine_top` (default 5, CLI `--gmaps-top`, `0` disables)
  budgets Google Maps Distance-Matrix calls to only the closest candidates.
  The rest stay on free Haversine (×1.30 tortuosity) — distance provenance is
  always visible per row.
- **Fail-safe by default.** LLM or enrichment failures degrade to deterministic
  safe defaults with `Confidence: NONE`, and the batch summary explicitly warns
  how many verifications failed — grades are never recomputed on hidden data.
- **Structured LLM output.** Confidence is a 4-bucket enum mapped to the score
  (`models.py`), so the LLM cannot smuggle arbitrary numbers into grading.
- **Grade A is evidence-qualified.** A requires profit ≥75, ease ≥75,
  confidence ≥0.80, *and* LLM-verified primary-steel-dealer status
  (`scoring.py:185-217`). In every recorded live batch to date, no grade A was
  produced — by design when evidence is thin.
- **Unknown-config rejection.** `config.yaml` sets `extra="forbid"` semantics:
  a YAML typo fails loudly instead of silently mirroring a code default.

## Proven (Delhi batch, depot 28.6139 / 77.2090)

Run: `full_delhi_batch/` — full pipeline with Gemini LLM, 29 passers
surviving from 500 raw rows:

| Metric | Result |
|---|---|
| Passers | 29 / 475 post-dedup (≈6%) |
| Grades | B: 5 · C: 16 · D: 8 |
| Enrichment | FIRECRAWL: 17 · MOCK fallback: 12 |
| Distance | GMAPS: 5 (budgeted) · HAVERSINE: 24 |
| LLM confidence | NONE: 29 (provider quota-hard-limited → safe defaults) |

The same entities re-run in `final_delhi_batch/` produced C: 21 / B: 5 / D: 3 —
**a caution the docs treat as a feature**: distance-source budget changes and
LLM failures visibly shift grades rather than silently altering rank quality;
this is why grades are framed as *decision-support bounded*, not autonomous.

## CLI

```bash
# fully offline, zero cost
agent-0 run --distributor-csv data/distributors.csv \
  --depot-name "Tata Steel Delhi Hub" --depot-lat 28.6139 --depot-lon 77.2090 \
  --skip-enrichment --skip-llm --format csv --output out/

# full pipeline: top-30 enrichment, top-5 GMaps, Gemini LLM
agent-0 run --distributor-csv data/distributors.csv \
  --depot-name "Tata Steel Delhi Hub" --depot-lat 28.6139 --depot-lon 77.2090 \
  --top-n 30 --gmaps-top 5 --llm gemini --format all --output full_delhi_batch/

agent-0 info   # effective configuration
```

Flags: `--top-n`, `--gmaps-top` (0 disables), `--max-distance`, `--llm-provider`,
`--use-nominatim`, `--tata-csv` (anti-cannibalization reference set),
`--skip-enrichment`, `--skip-llm`, `--format table|csv|json|all`.

## Status & verification

- Python ≥3.10, stdlib-plus typer/pydantic/pandas/rich/openai stack
  (`pyproject.toml`); run under pyenv 3.10.13.
- **53 deterministic pytest tests pass offline** (`tests/test_agent_0.py`
  31 + `tests/stress_mocked.py` 22), including scoring-grid and gate
  regression, turnover-parser adversarial cases, R1 unverified-cap
  regressions, and GMaps-budget split tests with mocked maps calls
  (no API spend).
- Automated suite covers parser and scoring invariants; live batch evidence
  covers the integration path (see table above).

## Known limitations (decision-support, not decision-autonomy)

- Turnover is slab-midpoint (±50%); geocoding falls back to state centroids
  (tens-of-km error can pass/miss the 180 km gate).
- Cannibalization radius is straight-line, not road-aware.
- Grades A/B require LLM confidence beyond `NONE`: unverified (`--skip-llm`,
  beyond top-N, or failed providers) is capped at C and marked
  `Verification Status` (`UNVERIFIED/FAILED`). B means "verify in field
  in this order", not "onboard"; only LLM-verified candidates reach A.
- Thresholds/weights are asserted (config-checked) but not backtested against
  realized distributor performance.
- `.venv` environment shadowing is a known documented hazard: bare `python3`
  may resolve to a stale editable install; use the pyenv 3.10.13 interpreter.

## Repo map

```
src/agent_0/   cli, config, models, csv_loader, geocoder, spatial, prefilter,
               enrichment, agent (LLM), scoring, pipeline
data/          distributors.csv (500-row GST export)
tests/         test_agent_0.py — 28 offline tests
config.yaml    all gate/score/grade/budget defaults (mirrors PipelineConfig)
final_delhi_batch/, full_delhi_batch/ : recorded live-run outputs
```
