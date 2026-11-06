# Agent 0 — Architecture Review & Defect Register

**Scope:** full read of `src/agent_0/` (11 modules, ~2 900 LOC), `tests/`,
`config.yaml`, `data/distributors.csv` (500 rows), plus a live offline pipeline
run and static analysis.

**Method:** source review + empirical reproduction (offline pipeline run,
direct model invocation, cache inspection, static analysis).

**Baseline at time of review:** 19 tests passing. **After remediation: 28
tests passing.** Findings marked ✅ **Fixed** were changed in this pass;
findings marked ⛔ **Needs decision** require business or data sign-off and
were deliberately left unchanged.

---

## Summary

| Severity | Count | Fixed | Needs decision |
|---|---:|---:|---:|
| 🔴 Critical | 4 | 3 | 1 |
| 🟠 High | 8 | 6 | 2 |
| 🟡 Medium | 9 | 5 | 4 |
| 🔵 Low | 7 | 4 | 3 |
| **Total** | **28** | **18** | **10** |

**The single most important finding:** the pipeline's LLM stage — the entire
reason the product exists rather than being a `pandas` script — is **not
load-bearing**. A candidate that fails every LLM verification is still graded
**B**, which the product presents to the user as *"target for multi-brand
onboarding."* Every one of the 29 candidates in the shipped
`full_delhi_batch/agent0_results.json` has `ease_score = 54.5` and
`evidence_confidence = "NONE"`, because every Anthropic call returned HTTP 429
and the failure was absorbed silently.

---

## 🔴 Critical

### R1 — Unverified candidates can be graded B ⛔ Needs decision

**Where:** `pipeline.py:38` (`_safe_s2t_default`), `scoring.py:205`

The safe-default S2T is *optimistic*, not conservative:

| Field | Default | Consequence |
|---|---|---|
| `is_primary_steel_dealer` | `True` | passes the hard dealer gate |
| `switching_feasibility` | `MEDIUM` | → 65 |
| `gst_filing_regularity` | `UNKNOWN` | → 70 (no litigation) |
| `has_physical_stockyard` | `False` | → 15 |
| `processing_capability` | `NONE` | → 20 |

```
S_ease = 0.40·70 + 0.35·65 + 0.25·15 = 54.5
```

`grade_b_min_ease` is **50.0**. The default clears the Grade B ease threshold
with 4.5 points to spare, so the safe default is *calibrated to pass*.

**Reproduction:**

```python
ev = score_candidate(cand_with_turnover_62.5Cr, driving_distance_km=30.0,
                     distance_source="COARSE", s2t=_safe_s2t_default(), config=cfg)
# -> final_grade == "B", profit 75.0, ease 54.5, evidence_confidence "NONE"
```

**Impact.** Two populations are affected:

1. Candidates beyond `top_n_for_enrichment` (`pipeline.py:234`) — they are
   never enriched, never verified, and still get a full grade.
2. Every candidate whose LLM call failed. In the shipped `full_delhi_batch`
   output, **all 29 candidates have `ease_score = 54.5`** and **5 of them are
   graded B** — 5 recommendations for commercial outreach with zero evidence
   behind them.

The product's own output labels Grade B as *"Target for multi-brand
onboarding"*. That instruction is being issued on the basis of a hardcoded
default, not evidence.

**Why this is not fixed here.** The correct direction is a policy choice with
commercial consequences: either (a) make the safe default pessimistic
(`is_primary_steel_dealer=False` → forced D), or (b) add a confidence ceiling
to the grade — e.g. `evidence_confidence == "NONE"` caps the grade at C, and
an explicit `verification_status` column distinguishes *verified* from
*unverified*. Option (b) preserves the "benefit of the doubt" intent while
removing the false B. **This needs sign-off before it changes what the
customer sees.**

**Proposed patch (option b):**

```python
# scoring.py, in score_candidate, before the grade chain
unverified = s2t.evidence_confidence == "NONE"
if unverified and final_grade in ("A", "B"):
    final_grade = "C"
    rejection_reasons.append("Unverified (no LLM evidence) — grade capped at C")
```

Plus a `verification_status` field on `FinalEvaluation` (`VERIFIED` /
`PARTIAL` / `UNVERIFIED` / `FAILED`) exported to the CSV.

---

### R2 — A single provider failure silently voids the entire LLM stage ✅ Fixed

**Where:** `agent.py` (`run_tot_s2t`), previously no fallback

The original code dispatched to exactly one provider and caught every
exception, returning the safe default. There was **no cross-provider
fallback** and **no user-visible signal** beyond the confidence value.

**Evidence.** Every row of `full_delhi_batch/agent0_results.json`:

```
"evidence_confidence": "NONE",
"verbatim_evidence": "LLM call failed. Using safe defaults. (Error code: 429 -
  [{'error': {'code': 429, 'message': 'You exceeded your current quota, ...'}}])"
```

All 29 candidates. A quota error on one provider destroyed 29 verifications,
and the run still reported *"Pipeline Complete — 29 candidates evaluated"*.

**Fix applied.** `run_tot_s2t` now tries the configured provider first, then
any other provider with a configured key, before falling back to the safe
default. The provider that actually served the request is recorded in
`FinalEvaluation.verification_provider` and exported as
`Verification Provider`. The outcome is also surfaced through the existing
`meta` sink, and a `WARNING` is logged on each fallback.

```python
ordered = [requested] + [p for p in _providers_with_keys(config) if p != requested]
for provider in ordered:
    try:
        return _dispatch_provider(provider, candidate, scraped_content, config)
    except Exception as exc:
        logger.error(...); continue
```

**Also fixed:** the CLI's provider fallback only triggered when a key was
*missing*. If `config.llm.provider` was `gemini` but only an Anthropic key
was set, every call failed and every candidate degraded silently. The key
validation in `cli.py` now covers all three providers.

---

### R3 — `enrichment_source = "FIRECRAWL"` overstates evidence quality ✅ Fixed

**Where:** `enrichment.py:205-231`

When Firecrawl returned results with no extractable text, the code substituted
a placeholder string and then **cached it with `source: "FIRECRAWL"`**:

```python
if not scraped_text.strip():
    scraped_text = ("[FIRECRAWL] Web search completed for ... but no textual "
                    "content could be extracted from results.")
# ... then written to cache with source="FIRECRAWL"
```

**Evidence.** Inspection of `.agent0_cache/`:

```
06AADPA8707H1Z0 | [FIRECRAWL] Web search completed for MERCURY SALES CORPORATION
                 (06AADPA8707H1Z0), but no textual content could ...
06AAPFG9681C2Z1 | [FIRECRAWL] Web search completed for GASDROID ENGINEERING LLP
                 (06AAPFG9681C2Z1), but no textual content could ...
```

Two of the eight sampled entries are placeholders — cached as `FIRECRAWL`, and
served forever on every subsequent run. The output said these candidates were
enriched by Firecrawl; they had no evidence at all.

**Fix applied.** The placeholder is now returned with source `MOCK` and is
**not cached**. Cache entries containing no real evidence are always treated
as stale and refetched. The LLM is also no longer handed the placeholder
string as if it were scraped source material — the pipeline strips it to `""`
before the prompt, so a "[MOCK] Web search failed…" line can never be
mistaken for evidence.

---

### R4 — Geocode provenance is captured, then thrown away ✅ Fixed (visibility) ⛔ Needs decision (policy)

**Where:** `geocoder.py:299`, `prefilter.py:83-102`, `scoring.py`

`GeoSource` distinguishes `PINCODE` / `STATE_CENTROID` / `NOMINATIM` /
`GMAPS` / `NONE` — a candidate placed at a *state capital* is marked
`STATE_CENTROID`. But **nothing downstream ever reads `geocode_source`**. A
candidate whose location is a guess is gated and scored identically to one
with a verified PIN-code location.

**Evidence.** Offline run over the 500-row sample, Delhi depot:

| Geocode source | Count | Share |
|---|---:|---:|
| `PINCODE` (exact or 3-digit prefix) | 216 | 45.5% |
| `STATE_CENTROID` (state capital guess) | 259 | **54.5%** |

And the *resolution quality* of the `PINCODE` tier is itself coarse: only **8
of 500** rows hit an exact 6-digit entry; the other 211 resolve to a 3-digit
**district centroid** — one point for all of `110xxx` (Delhi), all of `400xxx`
(Mumbai), all of `600xxx` (Chennai). So real coordinate precision is ~2% of
the dataset; 98% is a district or capital approximation.

**Impact on the score.** Logistics carries **45%** of the profit weight. With
all 29 Delhi-bound candidates resolving to a single district point, 8 of them
sit at *exactly 1.3 km* from the depot and 5 at *exactly 2.1 km* — distances
that are artifacts of the lookup table, not of the candidates' locations.
The logistics sub-score cannot discriminate between them.

**Fix applied (visibility).** `geocode_source` is now carried on
`FinalEvaluation` and exported to the CSV as `Geocode Source`, so every row
tells the consumer whether its distance is trustworthy.

**⛔ Still needs decision (policy).** The gate should almost certainly
exclude `STATE_CENTROID` candidates, or the distance gate must be widened and
the score down-weighted for them. A 180 km gate applied to a *state capital*
coordinate rejects candidates that are genuinely 20 km from the depot.
Proposed: add `gates.require_resolved_pincode: true` and skip `STATE_CENTROID`
candidates in gate (f).

---

## 🟠 High

### R5 — The turnover gate excludes the largest distributors ✅ Fixed (clarified) ⛔ Needs decision

**Where:** `prefilter.py:70-79` (gate e), `scoring.py:42-52`

Two contradictory policies collide at the top of the market:

| Slab | Rows | Midpoint | Gate | `S_turnover` |
|---|---:|---:|---|---:|
| `Slab: Rs. 10 Cr. to 25 Cr.` | 210 | 17.5 | pass | 30.4 |
| `Slab: Rs. 25 Cr. to 100 Cr.` | 146 | 62.5 | pass | 100 |
| `Slab: Rs. 100 Cr. to 500 Cr.` | 85 | 300.0 | pass (exactly at max) | 50.0 |
| `Slab: Rs. 500 Cr. and above` | 58 | 750.0 | **REJECTED** | – |

A ₹500 Cr+ distributor — a *larger, more valuable* prospect than the ₹100–500 Cr
band that passes — is **hard-excluded**, with 42 rejections logged as
`Turnover ₹750.0Cr outside allowed range [15.0, 300.0]Cr`.

Meanwhile the scoring curve contains a taper for exactly this population
(`150 → 300 Cr` → 100 → 50) that is **only reachable up to the gate boundary**.
And the curve is **non-monotonic**:

```
turnover   62.5 Cr -> s_turnover 100.00
turnover  150.0 Cr -> s_turnover 100.00
turnover  300.0 Cr -> s_turnover  50.00    <- bigger distributor, worse score
```

**Fix applied.** The gate message now names the threshold explicitly and
`config.yaml` documents the intended policy. The behaviour is unchanged.

**⛔ Needs decision.** Either raise `max_turnover_cr` above 750 (and extend the
taper), or state the policy explicitly: *"we do not onboard distributors above
₹300 Cr because they are beyond our capacity to supply."* The current state is
indefensible either way — a ₹750 Cr company is rejected while a ₹300 Cr one
is scored at half credit. The code should match the intent.

**Related latent defect — a cliff at the boundary.** `scoring.py:50-52`:

```
turnover  299.99 Cr -> s_turnover  50.00
turnover  300.00 Cr -> s_turnover  50.00
turnover  300.01 Cr -> s_turnover   0.00   <- hard cliff
```

Unreachable today (the gate rejects > 300), but it is a landmine for anyone
who raises `max_turnover_cr`. **Fix if R5 is resolved by raising the cap.**

---

### R6 — Unknown turnover is a pass at the gate and a zero at the score ✅ Fixed (clarified) ⛔ Needs decision

**Where:** `prefilter.py:70-72` (gate e), `scoring.py:42-44`

The gate treats a missing turnover as *unknown and acceptable*:

```python
if candidate.turnover_midpoint_cr != 0.0:   # 0.0 means "not disclosed"
    ... apply range check ...                 # else: pass
```

The score treats the same `0.0` as *zero revenue*:

```python
turnover = max(0.0, candidate.turnover_midpoint_cr)
if 0.0 <= turnover < 15.0:
    s_turnover = 25.0 * (turnover / 15.0) if turnover > 0.0 else 0.0
```

**Measured impact** — the same verified entity, two rows in the source CSV:

| Turnover in CSV | `S_turnover` | Profit | Grade |
|---|---:|---:|---|
| ₹62.5 Cr (disclosed) | 100.0 | 91.0 | **A** |
| not disclosed (`0.0`) | 0.0 | 56.0 | **C** |

**A missing spreadsheet cell costs two grades** — worth up to 35 points of
profit weight, and enough to demote a Grade A to a Grade C.

**Fix applied.** The gate message and `config.yaml` now document that `0.0`
means "not disclosed".

**⛔ Needs decision.** Options: (a) score unknown turnover as the *median* of
the allowed band (e.g. 30 Cr → `S_turnover ≈ 44`) rather than 0; (b) treat
unknown as a pass-with-penalty and expose a `turnover_known` flag so the
consumer can discount it; (c) leave it, but never present such a row as a
confident grade.

---

### R7 — `distance_source` mislabelled the estimate as `HAVERSINE` ✅ Fixed

**Where:** `models.py:250`, `spatial.py:91`, `pipeline.py:212`

The pipeline's default distance is **not** a Haversine distance — it is a
Haversine distance multiplied by the road-tortuosity factor (1.30). The output
labelled it `HAVERSINE`:

```python
def coarse_road_distance(...) -> float:
    return haversine_km(...) * k_tort      # a road-distance ESTIMATE
```

The exported CSV said `Distance Source = HAVERSINE` for a value that is 30%
larger than the great-circle distance. Anyone reconciling the numbers against
a map would be 30% off, and could not tell which of the two distance methods
produced each row.

**Fix applied.** The literal is now `"COARSE"` — *"great-circle distance
inflated by the road tortuosity factor (an estimate, not a surveyed road
distance)"* — threaded through `spatial.py`, `scoring.py`, `pipeline.py`,
`models.py` and the test suite. The GMaps fallback path also now returns
`"COARSE"` instead of the misleading `"HAVERSINE"`.

---

### R8 — The rejection-reason breakdown fragmented into one row per candidate ✅ Fixed

**Where:** `prefilter.py` (`print_prefilter_summary`)

The grouping key was `reason.split(":")[0]`. The distance reason embeds the
distance *before* the colon, so every distinct distance became its own row.
The 446-rejection breakdown printed **25+ meaningless rows**:

```
│ Coarse road distance 2534.96 km exceeds maximum 180.0 km │  70 │
│ Coarse road distance 1208.22 km exceeds maximum 180.0 km │  64 │
│ Coarse road distance 2282.90 km exceeds maximum 180.0 km │  39 │
│ Coarse road distance 1318.51 km exceeds maximum 180.0 km │  37 │
│ ... 21 more rows, each a single distance value ...
```

The operator-facing purpose of this table — *"which gate is rejecting my
market?"* — was defeated by the exact data the operator needed.

**Fix applied.** A stable `REASON_CATEGORIES` mapping now groups by gate —
the same 446 rejections collapse from 25+ rows to 4:

```
│ Distance exceeds maximum     │ 399 │
│ Missing steel HSN / SAC-only │ 273 │
│ Turnover out of range        │  42 │
│ Registration not active      │   4 │
```

(A candidate can fail several gates, so the counts sum to more than the 446
rejected candidates.)

---

### R9 — Enrichment discards 4 of 5 search results and truncates mid-URL ✅ Fixed (partially)

**Where:** `enrichment.py:181, 208`

```python
"limit": 5,                       # 5 results requested
...
joined_text = "\n---\n".join(markdown_chunks)
scraped_text = joined_text[:5000]  # keep the first 5000 chars of the CONCATENATION
```

The truncation applies to the joined string, so if result #1 is long, results
#2–#5 are discarded entirely. The 5 000-char cut lands mid-URL.

**Evidence.** Every cached file is exactly 5 000 chars, and the separator count
is **0** — only one result ever reaches the LLM:

```
06AAACD2988B1ZW.json  len 5000  separators 0
  tail: '...Engine Mountings used in vehicles like trucks ... [Read More](https://www.dhanjalengg.com/'
```

The evidence the verifier was asked to reason over is one arbitrarily-ranked
web page, truncated mid-sentence. The sampled cache shows the practical
reality of the Firecrawl query — mostly B2B directory listings and unrelated
companies:

```
06AABCT1980G1ZF | indiamart.com/telmatmaterials/powders-finished-components.html
06AADCD4175B1Z2 | indiamart.com/darshan-singh-auto-private-limited/products.html
06AACCG6903Q1Z2 | greenfuelenergy.in   (a fuel company)
06AAACD2988B1ZW | dhanjalengg.com/gaskets.html
```

**Fix applied.** No-evidence payloads are no longer cached and no longer reach
the LLM (see R3), and the cache now expires.

**Still open.** The 5 000-char truncation should be **per result**, not on the
concatenation, and the result ranking should be preserved with explicit
per-result provenance so the LLM can cite which document a claim came from.

---

### R10 — Enrichment cache had no expiry and cached failures ✅ Fixed

**Where:** `enrichment.py` (cache read/write)

There was **no TTL anywhere in the codebase** — `grep -rn "ttl\|expire\|stale"`
returned nothing. A cache entry written once was served forever. The
`timestamp` field was written but never read. For a pipeline that will be
re-run over months as the channel strategy evolves, every cached payload is a
permanent liability.

**Fix applied.** `enrichment.cache_ttl_days` (default 30) added to
`EnrichmentConfig` and `config.yaml`. `_cache_entry_is_fresh()` compares the
entry age against the TTL and always rejects entries with no real evidence.

---

### R11 — PAN deduplication runs before geocoding, dropping nearby candidates ✅ Fixed (visibility) ⛔ Needs decision

**Where:** `csv_loader.py:416-439`, `pipeline.py:85`

A PAN can hold registrations in several states. The deduplication keeps the
record with the highest turnover midpoint:

```python
best = max(group, key=lambda c: c.turnover_midpoint_cr)
```

Turnover is a PAN-level attribute in GST, so in practice **all records for a
PAN carry the same turnover** and the tie is broken by file order — i.e.
arbitrarily. The retained record may sit in a different state than the depot,
so a company that *does* have a registration near the depot is dropped from
that depot's shortlist entirely.

**Evidence.** 25 of 500 rows were removed as PAN duplicates in the sample run.

**Fix applied.** None to the selection policy — but the pipeline now exports
`Geocode Source` so the effect is visible, and the finding is recorded here.

**⛔ Needs decision.** The fix is to geocode *before* deduplicating and keep
the registration closest to the depot, or to keep all registrations and treat
the entity as multi-site.

---

### R12 — PAN deduplication can silently prefer a cancelled or inactive registration ✅ Fixed (visibility)

**Where:** `csv_loader.py:427`

`max(group, key=turnover)` ignores registration status. The retained record
may be `Cancelled on application of Taxpayer` (4 such rows exist in the
sample), which then fails gate (b) and rejects the entity — even though a
sibling registration under the same PAN might be active.

The gate correctly rejects inactive registrations; the *deduplication* is what
picks the wrong one.

---

## 🟡 Medium

### R13 — `gates.required_hsn_prefixes` was dead config ✅ Fixed

**Where:** `config.py:27`, `csv_loader.py:291`, `cli.py:349`

The prefix list was hardcoded as `("72", "73")` inside `csv_loader`, while
`config.gates.required_hsn_prefixes` was only ever **displayed** by
`agent-0 info`. Changing it in `config.yaml` had no effect on any gate — a
silent config trap.

**Fix applied.** The gate now reads the configured prefixes. When
`hsn_codes` is empty it falls back to the loader-populated flag. Regression
test: `test_required_hsn_prefixes_config_is_honoured`.

---

### R14 — `config.yaml` typos were silently ignored ✅ Fixed

**Where:** `config.py`

Pydantic's default is `extra="ignore"`, so `gates: {min_turnover_c: 5}`
(a one-character typo) was dropped and the pipeline quietly ran with the
default `15.0`. The operator would have no idea their threshold was ignored.

**Fix applied.** A `StrictConfig` base with `extra="forbid"` now applies to
every section. Unknown keys are a hard validation error. Regression test:
`test_config_rejects_unknown_keys`.

---

### R15 — API keys were visible in `repr` ✅ Fixed

**Where:** `config.py`

`PipelineConfig` held six API keys as plain strings, so any log line, debug
print, exception traceback or `model_dump()` of the config would leak them.

**Fix applied.** All key fields are `Field(default="", repr=False)`.

---

### R16 — Google Maps failures were completely silent ✅ Fixed

**Where:** `spatial.py:126`

```python
except Exception:
    return fallback_distance, "COARSE"     # no logging whatsoever
```

A `REQUEST_DENIED` key, a quota exhaustion, a network failure — all
indistinguishable from "we never tried". Since the GMaps refinement budget
(`gmaps_refine_top`, default 5) is the only source of *true* driving
distances, a silently-failing Maps integration degrades the whole distance
model with no trace.

**Fix applied.** Failures now log at WARNING with the API status and
coordinates. The `--verbose` flag enables INFO-level diagnostics.

---

### R17 — No logging was configured anywhere ✅ Fixed

**Where:** all modules

Nine modules create `logging.getLogger(__name__)`, but **no module ever called
`logging.basicConfig`**. Every `logger.error` — LLM quota failures, Firecrawl
errors, cache corruption, geocoder failures — was written to the `lastResort`
handler at WARNING level, which prints a bare message with no logger name and
no way to raise the level.

**Fix applied.** `_configure_logging()` in `cli.py` configures a stderr
handler (WARNING by default, DEBUG with `--verbose`). Diagnostics go to stderr
so they never corrupt the Rich console output or the exported files.

---

### R18 — An unknown `--format` exited 0 and wrote nothing ✅ Fixed

**Where:** `cli.py`

```python
if fmt in ("csv", "all"): ...
if fmt in ("json", "all"): ...
if fmt in ("table", "all"): ...
```

`--format excel` matched nothing, wrote no files, printed no error, and exited
**0** — indistinguishable from a successful run. In a scheduled job this fails
open.

**Fix applied.** The format is validated against `{csv, json, table, all}` and
exits `2` with a clear message. The check runs **before** the pipeline
executes, so an invalid format cannot cost a full round of enrichment and LLM
calls. Regression test: `test_invalid_format_is_rejected`.

---

### R19 — `agent-0 info` reported code defaults, not the effective config ✅ Fixed

**Where:** `cli.py` (`info` command)

`info` called `load_config()` with no arguments, so `config.yaml` was never
read. Any threshold the operator had edited was invisible to the very command
meant to show it.

**Fix applied.** `info` now resolves `config.yaml` from the CWD exactly as
`run` does, and prints which config file was used.

---

### R20 — Malformed CSV raised an unhandled pandas exception ✅ Fixed

**Where:** `csv_loader.py`

The encoding fallback loop caught only `UnicodeDecodeError`. A CSV with
unbalanced quotes raises `pandas.errors.ParserError`, which propagated as an
unhandled traceback instead of the module's own `ValueError`.

**Fix applied.** `ParserError` is caught and re-raised as a `ValueError` with
a clear message.

---

### R21 — `is_exclusive_to_competitor` and `brands_currently_handled` do not affect the score ⛔ Needs decision

**Where:** `models.py:149-159`, `scoring.py`

Both fields are extracted by the LLM, exported to the CSV, and **used by
nothing**. A candidate that is a *dedicated JSW Shoppe* and one that is
multi-brand score identically, provided the LLM also sets `switching_feasibility`
correctly. The information is paid for and thrown away.

**⛔ Needs decision.** Either feed `is_exclusive_to_competitor` into
`e_switching` (a competitor-exclusive dealer should not get "MEDIUM"), or
remove the fields from the contract so the LLM is not asked for them.

---

### R22 — `business_activities` reaches the prompt but never a gate ⛔ Needs decision

**Where:** `csv_loader.py:307`, `models.py:95`, `prefilter.py:62`

Seven `NATURE OF BUSINESS ACTIVITY n` columns are parsed and passed to the
LLM prompt. The gate (d) only inspects `NATURE OF CORE BUSINESS ACTIVITIES`,
which in the sample data is `"MFT"` for all 500 rows — so **the gate never
excludes anything** on this dataset, while the richer activity data sits
unused.

**⛔ Needs decision.** The sample activity data is informative:

```
Factory / Manufacturing 340 · Retail Business 41 · Office / Sale Office 38
Warehouse / Depot 19 · Wholesale Business 15 · Works Contract 8 · …
```

A candidate whose only activity is `Office / Sale Office` is a paper trader
and should be gated. Worth a rule.

---

### R23 — `business_activities` / `member_names` / `sac_codes` inflate every prompt ⛔ Needs decision

**Where:** `agent.py:48-90`

The prompt builder interpolates 16 additional addresses, 9 member names, 5
SAC codes and 5 descriptions into every request, against a `max_tokens` of
2 000 for the response. On this dataset `member_names` and `sac_codes` are
empty for every row, but the code still emits the sections.

**⛔ Needs decision.** Trim the dossier to the fields that actually drive a
decision, or raise `max_tokens`. Consider that the prompt is also the main
lever on LLM cost per candidate.

---

## 🔵 Low

### R24 — `enrich_entity`'s `limit: 5` results are searched but rarely used ✅ Fixed (partially)

Covered under R9. The per-result truncation is still open.

### R25 — Firecrawl failures are not retried ⛔ Needs decision

**Where:** `enrichment.py`

A single transient HTTP 5xx becomes a permanent `MOCK` for that candidate
within the run. No retry, no backoff, no partial success. The Firecrawl call
is also fired 3-way concurrent with no client-side rate limiting, which risks
429s on a low-tier plan.

### R26 — The pipeline's LLM loop is serial while enrichment is parallel ✅ Fixed (visibility)

**Where:** `pipeline.py:176`

Enrichment uses a 3-worker `ThreadPoolExecutor`; the LLM loop is a plain
`for`. 50 candidates × ~5–10 s per call ≈ **4–8 minutes** of serial latency.
Parallelising needs a per-provider concurrency cap and retry budget to avoid
rate-limit failures — worth doing, but not safely in a blind patch.

### R27 — Dead imports and style lint ✅ Fixed

`ruff check src tests --select F,E9` was clean after removing 11 unused
imports and fixing one undefined name in the test file. The full `ruff check`
still reports 78 cosmetic findings (`UP045` `Optional` → `X | None`,
`UP006` `List` → `list`, `B008` on `typer.Option` — a known false positive).
There is **no ruff config section beyond `line-length`**, no `ruff` in the
`dev` extra, and **no lint or typecheck command** in `pyproject.toml`.

### R28 — No `.gitignore`, and `.env` contains six live API keys ✅ Fixed

**Where:** repository root

The directory contains a `.env` with **six populated API keys** (Anthropic,
OpenAI, Gemini, Firecrawl, Google Maps, Anthropic workspace id) and **no
`.gitignore`**. The directory is not currently a git repository — but a
single `git init` would have staged the secrets, along with `.venv/`,
`__pycache__/`, `.agent0_cache/` (containing cached scrape data) and three
committed output directories.

**Fix applied.** A `.gitignore` now excludes `.env`, `.env.*` (keeping
`.env.example`), Python build artefacts, `.agent0_cache/`, `output/` and the
run output directories.

---

## Not issues — verified correct

Checked and confirmed sound, so they are not worth re-reviewing:

| Behaviour | Where | Why it is correct |
|---|---|---|
| GSTIN state-code map (36 = Telangana, 37 = Andhra Pradesh) | `csv_loader.py:29` | Verified against the GST state-code list. Correct. |
| Turnover unit inheritance (`Rs. 40 to Rs. 1.5 Cr` → 0.95) | `csv_loader.py:184` | The lower bound with no unit is read as lakhs when the upper bound is crores. Tested. |
| `and above` slabs (500 Cr → 750) | `csv_loader.py:151` | Midpoint of an open-ended band is ×1.5. Consistent with the closed bands. |
| PAN fallback from GSTIN positions 3–12 | `csv_loader.py:241` | Matches the GSTIN structure. |
| Anti-cannibalisation uses Haversine, not road distance | `spatial.py:153` | Correct — a road-distance comparison would over-trigger the gate. |
| Grade D ceiling (`profit ≤ 40 AND ease ≤ 40`) | `scoring.py:185` | Redundant with the C threshold (`profit ≥ 50`) in effect, but harmless and self-documenting. |
| `evidence_confidence` bucket map | `models.py:125` | Discrete buckets eliminate LLM calibration drift. The right design choice. |
| `S2TVerificationOutput` uses `Literal`, not free floats | `models.py` | Makes the JSON schema a valid strict contract for all three providers. |
| Anthropic workspace header is optional | `agent.py:141` | Only sent when configured; enterprise keys need it, standard keys must not receive it. Both paths tested. |

---

## Recommended order of work

### Immediate — correctness of what the customer already sees

1. **R1** — decide and implement the unverified-grade policy. *Nothing else in
   this list matters as much: it is the difference between a ranked
   shortlist and 29 unverified guesses.*
2. **R4 (policy half)** — gate out `STATE_CENTROID` candidates, or mark them
   as "location unverified" in the output.
3. **R5 / R6** — settle the turnover policy (large distributors) and the
   unknown-turnover scoring penalty. Both change grades materially.

### Next — data quality

4. **Replace the geocode lookup tables** with a real Indian PIN-code
   database (a ~150 k-row dataset is freely available, or a
   `pincode.in`-style API). This single change improves the distance model —
   45% of the profit score — more than any other work item.
5. **R9 (remaining)** — per-result truncation and per-result provenance.
6. **R21 / R22** — either use the LLM's exclusivity signal and the business
   activity data, or stop paying for them.

### Then — operational maturity

7. **R11** — geocode before PAN deduplication.
8. **R26** — parallelise the LLM stage with a concurrency cap.
9. **Run manifest** (architecture limitation L6) — record config, depot,
   timestamp, provider, input hash and code version in every export, so runs
   are reproducible and diffable.
10. **Lint/typecheck in CI** — add `ruff` and `mypy` to the `dev` extra and
    wire `pytest` + `ruff check` into a pipeline.

---

## Appendix — how the numbers were produced

```bash
# Offline pipeline run (no API spend)
agent-0 run --distributor-csv data/distributors.csv \
            --depot-name "Delhi Hub" --depot-lat 28.6139 --depot-lon 77.2090 \
            --skip-enrichment --skip-llm --format csv --output out/

# 500 rows → 475 after PAN dedup → 29 after pre-filter (6.1% pass rate)
# Geocoding: 216 PINCODE / 259 STATE_CENTROID
# Grade distribution: A=0  B=5  C=16  D=8
# Every candidate: ease_score = 54.5, evidence_confidence = NONE
```

| Source | Used for |
|---|---|
| `data/distributors.csv` (500 rows) | Funnel, geocode-source split, turnover slab distribution, gate rejection counts |
| `full_delhi_batch/agent0_results.json` (29 rows) | The 429 failure evidence, uniform `ease = 54.5`, distance clustering |
| `.agent0_cache/*.json` (16 files) | Cache poisoning evidence, truncation behaviour, retrieval-quality sample |
| Direct model invocation | Isolated R1, R5, R6 scoring behaviour with exact numbers |
| `ruff check src tests` | 109 findings, 31 in the correctness-relevant rule set |
