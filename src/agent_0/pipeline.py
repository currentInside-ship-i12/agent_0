"""End-to-end pipeline orchestrator for Agent 0.

Wires together: CSV loading → geocoding → pre-filtering →
web enrichment → Claude ToT/S2T → deterministic scoring → grading.

LLM is used ONLY for qualitative disambiguation (ToT + S2T).
All math lives in scoring.py. All gates live in prefilter.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_0.agent import run_tot_s2t
from agent_0.config import PipelineConfig
from agent_0.csv_loader import load_distributor_csv
from agent_0.enrichment import batch_enrich, is_no_evidence
from agent_0.geocoder import geocode_candidates
from agent_0.models import (
    EnrichmentSource,
    FinalEvaluation,
    S2TVerificationOutput,
    TataDepot,
)
from agent_0.prefilter import prefilter
from agent_0.scoring import score_candidate
from agent_0.spatial import coarse_road_distance, google_maps_distance

console = Console()


def _safe_s2t_default() -> S2TVerificationOutput:
    """Return a conservative default when no enrichment/LLM is performed."""
    return S2TVerificationOutput(
        is_primary_steel_dealer=True,
        brands_currently_handled=[],
        is_exclusive_to_competitor=False,
        switching_feasibility="MEDIUM",
        has_physical_stockyard=False,
        processing_capability="NONE",
        promoter_litigation_flag=False,
        gst_filing_regularity="UNKNOWN",
        evidence_confidence="NONE",
        verbatim_evidence="No enrichment performed. Using CSV-only defaults.",
        falsification_notes="No verification performed.",
    )


def run_pipeline(
    distributor_csv: Path,
    tata_depot: TataDepot,
    config: PipelineConfig,
    existing_tata_csv: Optional[Path] = None,
    skip_enrichment: bool = False,
    skip_llm: bool = False,
    use_nominatim: bool = False,
) -> list[FinalEvaluation]:
    """Execute the full Agent 0 pipeline.

    Args:
        distributor_csv: Path to the distributor CSV file.
        tata_depot: Target Tata Steel depot/plant location.
        config: Pipeline configuration.
        existing_tata_csv: Optional CSV of existing Tata distributors (must have
            GSTIN, PERMANENT ADDRESS or lat/lon columns).
        skip_enrichment: If True, skip Firecrawl web enrichment.
        skip_llm: If True, skip Claude/OpenAI ToT+S2T (uses safe defaults).
        use_nominatim: If True, use Nominatim API for geocoding.

    Returns:
        List of FinalEvaluation results sorted by profit_score desc.
    """
    t0 = time.time()

    # ── Phase 1: Data Ingestion ────────────────────────
    console.print(Panel("[bold cyan]Phase 1: Data Ingestion[/]", expand=False))
    candidates = load_distributor_csv(distributor_csv, deduplicate_by_pan=True)

    if not candidates:
        console.print("[red]No valid candidates loaded from CSV. Aborting.[/]")
        return []

    # ── Phase 2: Geocoding ─────────────────────────────
    console.print(Panel("[bold cyan]Phase 2: Geocoding[/]", expand=False))
    candidates = geocode_candidates(candidates, use_nominatim=use_nominatim)

    # ── Load existing Tata distributors (for anti-cannibalization) ──
    existing_coords: list[tuple[float, float]] = []
    if existing_tata_csv and existing_tata_csv.exists():
        console.print(f"  Loading existing Tata distributors from [bold]{existing_tata_csv}[/]")
        existing_distributors = load_distributor_csv(existing_tata_csv, deduplicate_by_pan=False)
        existing_distributors = geocode_candidates(existing_distributors, use_nominatim=use_nominatim)
        existing_coords = [
            (d.latitude, d.longitude)
            for d in existing_distributors
            if d.latitude is not None and d.longitude is not None
        ]
        console.print(f"  Loaded {len(existing_coords)} existing distributor locations.")

    # ── Phase 3: Deterministic Pre-filtering ───────────
    console.print(Panel("[bold cyan]Phase 3: Deterministic Pre-filtering[/]", expand=False))
    filter_results = prefilter(candidates, tata_depot, existing_coords, config)
    passed_results = [r for r in filter_results if r.passed]
    passed_candidates = [r.candidate for r in passed_results]

    if not passed_candidates:
        console.print("[red]No candidates passed pre-filtering. Aborting.[/]")
        return []

    # ── Phase 4: Select Top-N for Enrichment ───────────
    # Sort by turnover (descending) as a coarse priority signal before enrichment.
    passed_candidates.sort(key=lambda c: c.turnover_midpoint_cr, reverse=True)
    top_n = config.enrichment.top_n_for_enrichment
    enrichment_batch = passed_candidates[:top_n]
    remaining = passed_candidates[top_n:]

    console.print(
        f"  Selected top {len(enrichment_batch)} candidates for enrichment "
        f"(of {len(passed_candidates)} passed)."
    )

    # ── GMaps refinement budget: closest N of the batch by coarse distance ──
    # (Missing coordinates rank last and never consume budget.)
    refine_n = max(0, config.enrichment.gmaps_refine_top)
    if refine_n > 0 and config.gmaps_api_key:
        ranked = sorted(
            enrichment_batch,
            key=lambda c: (
                coarse_road_distance(
                    c.latitude,
                    c.longitude,
                    tata_depot.latitude,
                    tata_depot.longitude,
                    config.spatial.road_tortuosity_factor,
                )
                if c.latitude is not None and c.longitude is not None
                else float("inf")
            ),
        )
        gmaps_refine_set = {c.gstin for c in ranked[:refine_n]}
        console.print(
            f"  GMaps refinement budget: {len(gmaps_refine_set)} of "
            f"{len(enrichment_batch)} closest candidates."
        )
    else:
        gmaps_refine_set = set()

    # ── Phase 5: Web Enrichment (Firecrawl) ────────────
    enrichment_data: dict[str, tuple[str, str]] = {}
    if not skip_enrichment and config.firecrawl_api_key:
        console.print(Panel("[bold cyan]Phase 5: Web Enrichment (Firecrawl)[/]", expand=False))
        enrichment_data = batch_enrich(
            enrichment_batch,
            firecrawl_api_key=config.firecrawl_api_key,
            cache_dir=config.enrichment.cache_dir,
            cache_ttl_days=config.enrichment.cache_ttl_days,
        )
    else:
        if skip_enrichment:
            console.print("[yellow]  Skipping web enrichment (--skip-enrichment flag).[/]")
        else:
            console.print("[yellow]  No Firecrawl API key. Skipping enrichment.[/]")

    # ── Phase 6: Claude/OpenAI ToT + S2T ───────────────
    evaluations: list[FinalEvaluation] = []

    if not skip_llm:
        console.print(Panel("[bold cyan]Phase 6: LLM Verification (ToT + S2T)[/]", expand=False))

    for i, cand in enumerate(enrichment_batch):
        # Get scraped content (or empty)
        scraped_text, enrichment_source = enrichment_data.get(
            cand.gstin, ("", "NONE")
        )

        # Never hand a placeholder / error string to the LLM as if it were
        # scraped source material — it would be treated as evidence.
        llm_evidence = "" if is_no_evidence(scraped_text) else scraped_text

        # Run ToT + S2T (or use defaults)
        verification_meta: dict = {}
        if skip_llm:
            s2t_result = _safe_s2t_default()
        else:
            console.print(
                f"  [{i + 1}/{len(enrichment_batch)}] Verifying: "
                f"[bold]{cand.display_name}[/] ({cand.gstin})"
            )
            s2t_result = run_tot_s2t(cand, llm_evidence, config, meta=verification_meta)

        # Get distance: GMaps refinement only inside the budget set,
        # otherwise the coarse tortuosity estimate (or 999.0 fallback).
        if cand.latitude is not None and cand.longitude is not None:
            if cand.gstin in gmaps_refine_set:
                dist_km, dist_source = google_maps_distance(
                    cand.latitude,
                    cand.longitude,
                    tata_depot.latitude,
                    tata_depot.longitude,
                    config.gmaps_api_key,
                )
            else:
                dist_km = coarse_road_distance(
                    cand.latitude,
                    cand.longitude,
                    tata_depot.latitude,
                    tata_depot.longitude,
                    config.spatial.road_tortuosity_factor,
                )
                dist_source: Literal["GMAPS", "COARSE"] = "COARSE"
        else:
            dist_km = 999.0
            dist_source = "COARSE"

        # Score and grade
        evaluation = score_candidate(
            candidate=cand,
            driving_distance_km=dist_km,
            distance_source=dist_source,
            s2t=s2t_result,
            config=config,
            enrichment_source=EnrichmentSource(enrichment_source),
            verification_provider=verification_meta.get("provider"),
        )
        evaluations.append(evaluation)

    # ── Handle remaining candidates (beyond top-N) with defaults ──
    for cand in remaining:
        s2t_result = _safe_s2t_default()
        if cand.latitude is not None and cand.longitude is not None:
            dist_km = coarse_road_distance(
                cand.latitude,
                cand.longitude,
                tata_depot.latitude,
                tata_depot.longitude,
                config.spatial.road_tortuosity_factor,
            )
        else:
            dist_km = 999.0

        evaluation = score_candidate(
            candidate=cand,
            driving_distance_km=dist_km,
            distance_source="COARSE",
            s2t=s2t_result,
            config=config,
            enrichment_source=EnrichmentSource.NONE,
        )
        evaluations.append(evaluation)

    # ── LLM outcome notice ───────────────────────────────
    hint = summarize_llm_outcome(
        evaluations, llm_attempted=len(enrichment_batch), skip_llm=skip_llm
    )
    if hint:
        console.print(f"[red]{hint}[/]")

    # ── Sort by profit score (desc), then ease score (desc) ──
    evaluations.sort(
        key=lambda e: (e.scores.profit_score, e.scores.ease_score),
        reverse=True,
    )

    elapsed = time.time() - t0

    # ── Summary ────────────────────────────────────────
    _print_summary(evaluations, elapsed)

    return evaluations


def summarize_llm_outcome(
    evaluations: list[FinalEvaluation],
    llm_attempted: int,
    skip_llm: bool,
) -> str | None:
    """Return a warning when every attempted LLM verification failed.

    A failed call is a verification whose confidence is NONE *and* whose
    verbatim evidence records an "LLM call failed" error (the --skip-llm
    safe default has different verbatim text and is not a failure).
    Returns None when verification was skipped, unattempted, or at least
    partially successful.
    """
    if skip_llm or llm_attempted <= 0:
        return None
    failures = sum(
        1
        for e in evaluations
        if e.verification is not None
        and e.verification.evidence_confidence == "NONE"
        and e.verification.verbatim_evidence.startswith("LLM call failed")
    )
    if failures >= llm_attempted:
        return (
            f"All {llm_attempted} LLM verifications failed "
            "(confidence NONE). Check provider credits/keys, retry with "
            "--llm <funded provider>, or use --skip-llm for CSV-only scoring."
        )
    return None


def _print_summary(evaluations: list[FinalEvaluation], elapsed: float) -> None:
    """Print a rich summary table of results."""
    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for e in evaluations:
        grade_counts[e.final_grade] += 1

    console.print()
    console.print(
        Panel(
            f"[bold green]Pipeline Complete[/] — {len(evaluations)} candidates evaluated "
            f"in {elapsed:.1f}s",
            expand=False,
        )
    )

    # Grade distribution
    grade_table = Table(title="Grade Distribution", show_lines=True)
    grade_table.add_column("Grade", style="bold")
    grade_table.add_column("Count", justify="right")
    grade_table.add_column("Action")
    grade_table.add_row(
        "[green]A[/]", str(grade_counts["A"]), "Immediate commercial dispatch"
    )
    grade_table.add_row(
        "[yellow]B[/]", str(grade_counts["B"]), "Target for multi-brand onboarding"
    )
    grade_table.add_row(
        "[blue]C[/]", str(grade_counts["C"]), "Retain in pool — field verification needed"
    )
    grade_table.add_row(
        "[red]D[/]", str(grade_counts["D"]), "Disqualified / Rejected"
    )
    console.print(grade_table)

    # Top 10 results
    if evaluations:
        top_table = Table(title="Top 10 Candidates", show_lines=True)
        top_table.add_column("#", style="bold", width=3)
        top_table.add_column("Grade", width=5)
        top_table.add_column("Name", max_width=30)
        top_table.add_column("GSTIN", width=16)
        top_table.add_column("State", width=15)
        top_table.add_column("Dist (km)", justify="right", width=10)
        top_table.add_column("Profit", justify="right", width=8)
        top_table.add_column("Ease", justify="right", width=8)
        top_table.add_column("Confidence", width=10)

        grade_colors = {"A": "green", "B": "yellow", "C": "blue", "D": "red"}

        for i, e in enumerate(evaluations[:10], 1):
            color = grade_colors.get(e.final_grade, "white")
            conf = e.verification.evidence_confidence if e.verification else "N/A"
            top_table.add_row(
                str(i),
                f"[{color}]{e.final_grade}[/]",
                e.trade_name or e.legal_name,
                e.gstin,
                e.state,
                f"{e.driving_distance_km:.1f}",
                f"{e.scores.profit_score:.1f}",
                f"{e.scores.ease_score:.1f}",
                str(conf),
            )

        console.print(top_table)
