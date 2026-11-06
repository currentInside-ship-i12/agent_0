"""End-to-end, evidence-aware distributor discovery pipeline.

The pipeline intentionally separates deterministic eligibility, incomplete-data
queues, and scored commercial leads. A language model can interpret public
evidence, but citation validation and recommendation state remain deterministic.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_0.agent import run_tot_s2t
from agent_0.config import PipelineConfig
from agent_0.csv_loader import load_distributor_csv
from agent_0.enrichment import batch_enrich, validate_evidence_citations
from agent_0.geocoder import geocode_candidates
from agent_0.models import (
    CandidateAuditRecord,
    CandidateDisposition,
    EnrichmentResult,
    EnrichmentSource,
    EvidenceValidation,
    FinalEvaluation,
    PipelineRunResult,
    RecommendationState,
    RunManifest,
    S2TVerificationOutput,
    TataDepot,
)
from agent_0.prefilter import TRUSTED_SPATIAL_SOURCES, prefilter
from agent_0.scoring import score_candidate
from agent_0.spatial import coarse_road_distance, google_maps_distance

console = Console()


def _safe_s2t_default(
    reason: str = "No source-level verification was performed.",
) -> S2TVerificationOutput:
    """Return an explicit unknown verdict when verification cannot run."""
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
        verbatim_evidence=reason,
        falsification_notes="No verification performed.",
    )


def _file_sha256(path: Path) -> str:
    """Hash an input artifact so a manifest can identify the exact run input."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_version() -> str | None:
    """Return the checked-out commit without making Git availability mandatory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _config_snapshot(config: PipelineConfig) -> dict:
    """Serialize decision parameters without exporting API credentials."""
    return {
        "gates": config.gates.model_dump(mode="json"),
        "spatial": config.spatial.model_dump(mode="json"),
        "profit_weights": config.profit_weights.model_dump(mode="json"),
        "ease_weights": config.ease_weights.model_dump(mode="json"),
        "grading": config.grading.model_dump(mode="json"),
        "enrichment": config.enrichment.model_dump(mode="json"),
        "llm": config.llm.model_dump(mode="json"),
    }


def _audit_record_from_prefilter(result) -> CandidateAuditRecord:
    """Convert a deterministic prefilter outcome into an exportable audit row."""
    reasons = (
        result.rejection_reasons
        if result.disposition == CandidateDisposition.REJECTED
        else result.needs_data_reasons
    )
    return CandidateAuditRecord(
        candidate=result.candidate,
        disposition=result.disposition,
        reasons=reasons,
        coarse_road_distance_km=result.coarse_road_distance_km,
        nearest_existing_distributor_km=result.nearest_existing_distributor_km,
    )


def _build_manifest(
    distributor_csv: Path,
    tata_depot: TataDepot,
    config: PipelineConfig,
    existing_tata_csv: Optional[Path],
    *,
    skip_enrichment: bool,
    skip_llm: bool,
    use_nominatim: bool,
    offline: bool,
    counts: dict[str, int],
) -> RunManifest:
    """Build reproducibility metadata after the final queue counts are known."""
    return RunManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        depot=tata_depot,
        distributor_csv=str(distributor_csv),
        distributor_csv_sha256=_file_sha256(distributor_csv),
        existing_tata_csv=str(existing_tata_csv) if existing_tata_csv else None,
        existing_tata_csv_sha256=(
            _file_sha256(existing_tata_csv) if existing_tata_csv is not None else None
        ),
        code_version=_git_version(),
        config=_config_snapshot(config),
        options={
            "skip_enrichment": skip_enrichment,
            "skip_llm": skip_llm,
            "use_nominatim": use_nominatim,
            "offline": offline,
        },
        counts=counts,
    )


def run_pipeline(
    distributor_csv: Path,
    tata_depot: TataDepot,
    config: PipelineConfig,
    existing_tata_csv: Optional[Path] = None,
    skip_enrichment: bool = False,
    skip_llm: bool = False,
    use_nominatim: bool = False,
    offline: bool = False,
) -> PipelineRunResult:
    """Run discovery, verification, scoring, and audit-queue construction.

    `offline=True` is the only no-network mode: it disables Firecrawl, model
    calls, Nominatim, and Google Maps regardless of configured API keys.
    """
    if existing_tata_csv is not None and not existing_tata_csv.is_file():
        raise FileNotFoundError(
            f"Existing Tata distributor CSV was explicitly supplied but does not exist: "
            f"{existing_tata_csv}"
        )

    if offline:
        skip_enrichment = True
        skip_llm = True
        use_nominatim = False

    started_at = time.time()

    console.print(Panel("[bold cyan]Phase 1: Data Ingestion[/]", expand=False))
    # Keep distinct GST registrations: a PAN may legitimately have different
    # depot-relevant sites, active status, and locations.
    candidates = load_distributor_csv(distributor_csv, deduplicate_by_pan=False)

    console.print(Panel("[bold cyan]Phase 2: Geocoding[/]", expand=False))
    candidates = geocode_candidates(candidates, use_nominatim=use_nominatim)

    existing_coords: list[tuple[float, float]] = []
    skipped_existing_locations = 0
    if existing_tata_csv is not None:
        console.print(f"  Loading existing Tata distributors from [bold]{existing_tata_csv}[/]")
        existing_distributors = load_distributor_csv(
            existing_tata_csv,
            deduplicate_by_pan=False,
        )
        existing_distributors = geocode_candidates(
            existing_distributors,
            use_nominatim=use_nominatim,
        )
        for distributor in existing_distributors:
            if (
                distributor.latitude is not None
                and distributor.longitude is not None
                and distributor.geocode_source in TRUSTED_SPATIAL_SOURCES
            ):
                existing_coords.append((distributor.latitude, distributor.longitude))
            else:
                skipped_existing_locations += 1
        console.print(f"  Loaded {len(existing_coords)} trusted existing distributor locations.")
        if skipped_existing_locations:
            console.print(
                f"[yellow]  {skipped_existing_locations} existing locations were not used "
                "for cannibalization because their coordinates need verification.[/]"
            )

    console.print(Panel("[bold cyan]Phase 3: Deterministic Eligibility[/]", expand=False))
    filter_results = prefilter(candidates, tata_depot, existing_coords, config)
    eligible_results = [
        result for result in filter_results if result.disposition == CandidateDisposition.ELIGIBLE
    ]
    needs_data = [
        _audit_record_from_prefilter(result)
        for result in filter_results
        if result.disposition == CandidateDisposition.NEEDS_DATA
    ]
    rejected = [
        _audit_record_from_prefilter(result)
        for result in filter_results
        if result.disposition == CandidateDisposition.REJECTED
    ]
    eligible_candidates = [result.candidate for result in eligible_results]

    if not eligible_candidates:
        manifest = _build_manifest(
            distributor_csv,
            tata_depot,
            config,
            existing_tata_csv,
            skip_enrichment=skip_enrichment,
            skip_llm=skip_llm,
            use_nominatim=use_nominatim,
            offline=offline,
            counts={
                "input_candidates": len(candidates),
                "eligible": 0,
                "needs_data": len(needs_data),
                "rejected": len(rejected),
                "evaluated": 0,
                "existing_locations_skipped": skipped_existing_locations,
            },
        )
        _print_summary(
            PipelineRunResult(needs_data=needs_data, rejected=rejected, manifest=manifest),
            time.time() - started_at,
        )
        return PipelineRunResult(
            needs_data=needs_data,
            rejected=rejected,
            manifest=manifest,
        )

    # Large distributors remain eligible under the approved hybrid policy.
    # A more nuanced coverage strategy belongs in its own product iteration.
    eligible_candidates.sort(
        key=lambda candidate: (
            candidate.turnover_known,
            candidate.turnover_midpoint_cr,
        ),
        reverse=True,
    )
    top_n = config.enrichment.top_n_for_enrichment
    enrichment_batch = eligible_candidates[:top_n]
    remaining = eligible_candidates[top_n:]
    console.print(
        f"  Selected {len(enrichment_batch)} eligible candidates for enrichment "
        f"(of {len(eligible_candidates)} eligible)."
    )

    refine_n = 0 if offline else config.enrichment.gmaps_refine_top
    if refine_n > 0 and config.gmaps_api_key:
        ranked = sorted(
            enrichment_batch,
            key=lambda candidate: coarse_road_distance(
                candidate.latitude,
                candidate.longitude,
                tata_depot.latitude,
                tata_depot.longitude,
                config.spatial.road_tortuosity_factor,
            ),
        )
        gmaps_refine_set = {candidate.gstin for candidate in ranked[:refine_n]}
        console.print(
            f"  GMaps refinement budget: {len(gmaps_refine_set)} of "
            f"{len(enrichment_batch)} candidates."
        )
    else:
        gmaps_refine_set: set[str] = set()

    enrichment_data: dict[str, EnrichmentResult] = {}
    if not skip_enrichment and config.firecrawl_api_key:
        console.print(Panel("[bold cyan]Phase 4: Web Enrichment (Firecrawl)[/]", expand=False))
        enrichment_data = batch_enrich(
            enrichment_batch,
            firecrawl_api_key=config.firecrawl_api_key,
            cache_dir=config.enrichment.cache_dir,
            max_workers=config.enrichment.max_workers,
            cache_ttl_days=config.enrichment.cache_ttl_days,
            max_chars_per_source=config.enrichment.max_chars_per_source,
            max_retries=config.enrichment.max_retries,
        )
    elif skip_enrichment:
        console.print("[yellow]  Web enrichment disabled for this run.[/]")
    else:
        console.print("[yellow]  No Firecrawl API key. Web enrichment skipped.[/]")

    evaluations: list[FinalEvaluation] = []
    if not skip_llm:
        console.print(Panel("[bold cyan]Phase 5: Nested ToT + Stage 2 Thinking[/]", expand=False))

    for index, candidate in enumerate(enrichment_batch, start=1):
        enrichment = enrichment_data.get(candidate.gstin, EnrichmentResult())
        verification_meta: dict = {}
        if skip_llm:
            verification = _safe_s2t_default("LLM verification disabled for this run.")
        elif enrichment.source == EnrichmentSource.FIRECRAWL and enrichment.sources:
            console.print(
                f"  [{index}/{len(enrichment_batch)}] Verifying: "
                f"[bold]{candidate.display_name}[/] ({candidate.gstin})"
            )
            verification = run_tot_s2t(
                candidate,
                enrichment.sources,
                config,
                meta=verification_meta,
            )
        else:
            verification = _safe_s2t_default(
                "No source-level web evidence was available for model verification."
            )

        if candidate.gstin in gmaps_refine_set:
            distance_km, distance_source = google_maps_distance(
                candidate.latitude,
                candidate.longitude,
                tata_depot.latitude,
                tata_depot.longitude,
                config.gmaps_api_key,
            )
        else:
            distance_km = coarse_road_distance(
                candidate.latitude,
                candidate.longitude,
                tata_depot.latitude,
                tata_depot.longitude,
                config.spatial.road_tortuosity_factor,
            )
            distance_source: Literal["GMAPS", "COARSE"] = "COARSE"

        evidence_validation = validate_evidence_citations(
            verification,
            enrichment.sources,
        )
        evaluations.append(
            score_candidate(
                candidate=candidate,
                driving_distance_km=distance_km,
                distance_source=distance_source,
                s2t=verification,
                config=config,
                enrichment_source=enrichment.source,
                verification_provider=verification_meta.get("provider"),
                verification_model=verification_meta.get("model"),
                evidence_validation=evidence_validation,
            )
        )

    for candidate in remaining:
        verification = _safe_s2t_default("Candidate was outside the configured enrichment budget.")
        distance_km = coarse_road_distance(
            candidate.latitude,
            candidate.longitude,
            tata_depot.latitude,
            tata_depot.longitude,
            config.spatial.road_tortuosity_factor,
        )
        evaluations.append(
            score_candidate(
                candidate=candidate,
                driving_distance_km=distance_km,
                distance_source="COARSE",
                s2t=verification,
                config=config,
                evidence_validation=EvidenceValidation(),
            )
        )

    hint = summarize_llm_outcome(
        evaluations,
        llm_attempted=len(enrichment_batch) if not skip_llm else 0,
        skip_llm=skip_llm,
    )
    if hint:
        console.print(f"[yellow]{hint}[/]")

    state_rank = {
        RecommendationState.VERIFIED_RECOMMENDATION: 0,
        RecommendationState.PRIORITIZED_LEAD: 1,
        RecommendationState.REJECTED: 2,
    }
    grade_rank = {"A": 0, "B": 1, "C": 2, "D": 3}
    evaluations.sort(
        key=lambda evaluation: (
            state_rank[evaluation.recommendation_state],
            grade_rank[evaluation.final_grade],
            -evaluation.scores.profit_score,
            -evaluation.scores.ease_score,
        )
    )

    manifest = _build_manifest(
        distributor_csv,
        tata_depot,
        config,
        existing_tata_csv,
        skip_enrichment=skip_enrichment,
        skip_llm=skip_llm,
        use_nominatim=use_nominatim,
        offline=offline,
        counts={
            "input_candidates": len(candidates),
            "eligible": len(eligible_candidates),
            "needs_data": len(needs_data),
            "rejected": len(rejected),
            "evaluated": len(evaluations),
            "verified_recommendations": sum(
                evaluation.recommendation_state == RecommendationState.VERIFIED_RECOMMENDATION
                for evaluation in evaluations
            ),
            "prioritized_leads": sum(
                evaluation.recommendation_state == RecommendationState.PRIORITIZED_LEAD
                for evaluation in evaluations
            ),
            "post_verification_rejections": sum(
                evaluation.recommendation_state == RecommendationState.REJECTED
                for evaluation in evaluations
            ),
            "existing_locations_skipped": skipped_existing_locations,
        },
    )
    result = PipelineRunResult(
        evaluations=evaluations,
        needs_data=needs_data,
        rejected=rejected,
        manifest=manifest,
    )
    _print_summary(result, time.time() - started_at)
    return result


def summarize_llm_outcome(
    evaluations: list[FinalEvaluation],
    llm_attempted: int,
    skip_llm: bool,
) -> str | None:
    """Summarize model coverage without concealing partial degradation."""
    if skip_llm or llm_attempted <= 0:
        return None
    failures = sum(evaluation.verification_status == "FAILED" for evaluation in evaluations)
    unverified = sum(evaluation.verification_status == "UNVERIFIED" for evaluation in evaluations)
    if failures:
        return (
            f"{failures}/{llm_attempted} LLM verifications failed; affected candidates "
            "remain prioritized leads, not verified recommendations."
        )
    if unverified:
        return (
            f"{unverified}/{llm_attempted} candidates have no citation-validated evidence; "
            "they remain prioritized leads."
        )
    return None


def _print_summary(result: PipelineRunResult, elapsed: float) -> None:
    """Print queue-aware output that does not equate leads with recommendations."""
    state_counts = {
        RecommendationState.VERIFIED_RECOMMENDATION: 0,
        RecommendationState.PRIORITIZED_LEAD: 0,
        RecommendationState.REJECTED: 0,
    }
    for evaluation in result.evaluations:
        state_counts[evaluation.recommendation_state] += 1

    console.print()
    console.print(
        Panel(
            f"[bold green]Pipeline Complete[/] — {len(result.evaluations)} scored, "
            f"{len(result.needs_data)} need data, and {len(result.rejected)} "
            f"deterministically rejected in {elapsed:.1f}s",
            expand=False,
        )
    )

    queue_table = Table(title="Commercial Queues", show_lines=True)
    queue_table.add_column("Queue", style="bold")
    queue_table.add_column("Count", justify="right")
    queue_table.add_column("Action")
    queue_table.add_row(
        "Verified recommendations",
        str(state_counts[RecommendationState.VERIFIED_RECOMMENDATION]),
        "A/B only: source-backed commercial action",
    )
    queue_table.add_row(
        "Prioritized leads",
        str(state_counts[RecommendationState.PRIORITIZED_LEAD]),
        "Collect field evidence before onboarding",
    )
    queue_table.add_row(
        "Needs data",
        str(len(result.needs_data)),
        "Resolve location, turnover, or identity uncertainty",
    )
    queue_table.add_row(
        "Rejected",
        str(len(result.rejected) + state_counts[RecommendationState.REJECTED]),
        "Known deterministic or verification hard failure",
    )
    console.print(queue_table)

    if result.evaluations:
        top_table = Table(title="Top Commercial Candidates", show_lines=True)
        top_table.add_column("#", style="bold", width=3)
        top_table.add_column("Queue", width=22)
        top_table.add_column("Grade", width=5)
        top_table.add_column("Name", max_width=28)
        top_table.add_column("GSTIN", width=16)
        top_table.add_column("Profit", justify="right", width=8)
        top_table.add_column("Evidence", justify="right", width=9)

        for index, evaluation in enumerate(result.evaluations[:10], start=1):
            top_table.add_row(
                str(index),
                evaluation.recommendation_state.value,
                evaluation.final_grade,
                evaluation.trade_name or evaluation.legal_name,
                evaluation.gstin,
                f"{evaluation.scores.profit_score:.1f}",
                str(evaluation.evidence_validation.valid_citation_count),
            )
        console.print(top_table)
