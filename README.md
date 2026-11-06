# Agent 0 — Tata Steel distributor discovery

Agent 0 turns a GST-registry export into four explicit commercial queues:

1. **Verified recommendations** — the only candidates eligible for Grades A/B.
   Their nested ToT and Stage 2 Thinking output contains citations that
   deterministic code verified against retained web-source text.
2. **Prioritized leads** — plausible businesses ranked for field verification;
   incomplete evidence can never promote them to A/B.
3. **Needs data** — plausible records with uncertain turnover, identity, or
   location that require data collection rather than a guessed decision.
4. **Rejected** — records excluded by known deterministic rules, with reasons.

The tool is decision support, not autonomous onboarding. It keeps discovery
optimistic while treating missing or weak evidence as uncertainty rather than a
positive signal.

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env  # fill in only the services you intend to use
agent-0 --help

# Truly offline: no Firecrawl, LLM, Nominatim, or Google Maps calls.
agent-0 run --distributor-csv data/distributors.csv \
  --depot-name "Jamshedpur" --depot-lat 22.80 --depot-lon 86.20 \
  --offline --format all --output out/

# Connected verification run.
agent-0 run --distributor-csv data/distributors.csv \
  --depot-name "Jamshedpur" --depot-lat 22.80 --depot-lon 86.20 \
  --tata-csv existing_tata.csv --nominatim \
  --format all --output out/
```

Use `--gmaps-top 0` to disable route refinement in a connected run. Explicit
`--config`, `--env`, and `--tata-csv` paths must exist; Agent 0 never silently
falls back when an operator supplied one.

## Evidence workflow

For candidates inside the configured enrichment budget, Firecrawl search results
are retained individually with source IDs, URLs, titles, and bounded content.
The LLM makes **one structured response** with:

- three nested ToT branches: brand alignment, infrastructure, and onboarding
  risk;
- exact source-ID citations per branch;
- Stage 2 Thinking: adversarial reconciliation, challenged claims, unresolved
  questions, and final rationale.

The system validates every cited excerpt against the retained source content.
No citation-validated evidence means no A/B recommendation.

## Output artifacts

`--format csv`, `json`, or `all` emits queue-specific artifacts:

- `agent0_results.*` — all scored candidates;
- `agent0_verified_recommendations.*`;
- `agent0_prioritized_leads.*`;
- `agent0_needs_data.*`;
- `agent0_prefilter_rejected.*`;
- `agent0_post_verification_rejected.*`;
- `agent0_run_manifest.json` — input hashes, decision config, options, depot,
  code version, and queue counts.

CSV exports neutralize spreadsheet formulas in source-derived text.

## Location policy

Coordinates are labelled as `MANUAL`, `PIN_EXACT`, `PIN_DISTRICT`, `NOMINATIM`,
`STATE_CENTROID`, or `NONE`.

Only manual, exact-PIN, and Nominatim coordinates can drive hard distance and
anti-cannibalization gates. District and state-centroid values remain useful
for discovery, but land in `Needs data` until location verification is
available.

## Validation

```bash
pytest -q
ruff format --check src tests
ruff check src tests
python -m pip wheel --no-deps .
```

The current suite is mocked and network-free. Run a controlled connected smoke
test only with approved credentials and a bounded enrichment budget.
