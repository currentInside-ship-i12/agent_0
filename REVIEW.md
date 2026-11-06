# Agent 0 release-readiness register

This is a current-state register, not a historical review transcript. Each
item records the implemented decision or the explicitly deferred work so that
the document remains useful after later changes.

## Implemented in the current release

| Area | Decision and evidence |
|---|---|
| Unverified grade inflation | Grades A/B now require citation-validated evidence. Otherwise an apparent A/B is capped at C and routed to `PRIORITIZED_LEAD`. |
| Cancelled compliance | `gst_filing_regularity == CANCELLED` is an explicit hard Grade-D rejection. |
| Offline correctness | `--offline` disables Firecrawl, LLMs, Nominatim, and Google Maps even when keys are configured. |
| Path safety | Explicit CLI config, environment, and existing-network paths must exist. GSTINs are format/checksum validated and cache filenames are SHA-256 hashes. |
| Spatial provenance | Manual, exact-PIN, district-PIN, Nominatim, centroid, and missing locations are distinct. Approximate locations route to `NEEDS_DATA` instead of deciding hard spatial gates. |
| Existing network input | Explicit coordinate columns are parsed. Only trusted existing-network locations participate in anti-cannibalization. |
| ToT and Stage 2 Thinking | A single strict nested response records three named ToT branches and a Stage 2 adversarial reconciliation. Source-ID excerpts are validated deterministically. |
| Auditing | Separate scored, verified, prioritized-lead, needs-data, and rejected artifacts are emitted with a reproducibility manifest. |
| Spreadsheet safety | CSV cells beginning with formula prefixes are escaped. |
| Configuration | Decision configs validate weight sums, threshold ranges, budgets, provider names, and reject secrets in YAML. |

## Accepted product decisions

| Decision | Rationale |
|---|---|
| Single-call nested ToT/S2T | Preserves explicit thinking structure while limiting provider cost and maintaining broad model compatibility. |
| Existing full dossier and provider fallback | Kept intentionally. The current product accepts existing data flow and fallback behavior rather than changing commercial inputs mid-release. |
| Large-distributor treatment | All sizes remain eligible after the minimum turnover threshold. Detailed coverage strategy is deferred. |
| Web geocoding | Optional Nominatim address/PIN lookup is the simple approximation fallback. It is labelled and never treated as equivalent to an exact source without provenance. |

## Deferred work

### Regional coverage strategy

The product still needs a dedicated strategy for territory white-space, depot
capacity, distributor size bands, overlap, and supply economics. This must be
modelled and validated as a separate initiative rather than hidden inside
turnover scoring.

### Live-provider acceptance test

The automated suite is network-free. Before enabling a connected production
workflow, run a bounded smoke test against the approved Firecrawl, Maps, and
LLM accounts, then inspect retained sources, citations, provider/model
metadata, and cost telemetry.

### Data governance

The current product decision retains the existing dossier and provider fallback
behavior. If data-sharing requirements change, introduce provider allowlists,
dossier minimization, and retention controls as a separately approved change.

## Required release checks

```bash
pytest -q
ruff format --check src tests
ruff check src tests
python -m pip wheel --no-deps .
```

For a no-cost verification of core behavior:

```bash
agent-0 run --distributor-csv data/distributors.csv \
  --depot-name "Delhi Hub" --depot-lat 28.6139 --depot-lon 77.2090 \
  --offline --format all --output out/
```
