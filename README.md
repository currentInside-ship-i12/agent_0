# Agent 0 — Tata Steel GTM Distributor Search & Scoring Engine

Search → Filter → Enrich → Score → Grade pipeline for Tata Steel distributor
expansion. Ingests a bulk GST registry export, screens it to entities that
plausibly distribute steel, enriches them with public web evidence, verifies
them with an LLM (Tree of Thoughts + Stage 2 Thinking), and emits a ranked,
graded shortlist with a full audit trail.

## Documentation

| Document | Contents |
|---|---|
| [`PROJECT.md`](./PROJECT.md) | Project overview with resume/CV one-liner: pipeline, design decisions, proven Delhi-batch results, CLI usage, status & known limitations |
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Product overview, pipeline architecture, module map, the full scoring & grading model, configuration, known limitations |
| [`REVIEW.md`](./REVIEW.md) | Architecture review: 28 findings by severity, with evidence, reproduction and remediation status |

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env  # fill in API keys
agent-0 --help

# Fully offline, deterministic, no API spend
agent-0 run --distributor-csv data.csv --depot-name "Jamshedpur" \
            --depot-lat 22.80 --depot-lon 86.20 \
            --skip-enrichment --skip-llm --format table

# Full pipeline
agent-0 run --distributor-csv data.csv --depot-name "Jamshedpur" \
            --depot-lat 22.80 --depot-lon 86.20 \
            --tata-csv existing_tata.csv --format all --output out/

agent-0 info     # effective configuration
```

See `config.yaml` for gate/threshold defaults. Add `--verbose` to surface
LLM and enrichment failures on stderr.

## Output

`agent0_results.csv` / `agent0_results.json` — one row per candidate with the
letter grade, profit and ease scores, every sub-score component, the
distance and where that distance came from, the geocode provenance, the LLM's
structured verdict with verbatim evidence, and the reasoning for any
rejection.

## Tests

```bash
pytest -q          # 53 deterministic tests (offline + mocked stress), no network
ruff check src tests
```
