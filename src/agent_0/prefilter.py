"""Deterministic pre-screening engine with hard boolean gates.

Applies strict business, compliance, turnover, and spatial gates to
filter distributor candidates before downstream enrichment and scoring.
Zero LLM involvement.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from rich.console import Console
from rich.table import Table

from agent_0.config import PipelineConfig
from agent_0.csv_loader import is_valid_gstin
from agent_0.models import (
    CandidateDisposition,
    DistributorCandidate,
    GeoSource,
    PrefilterResult,
    TataDepot,
)
from agent_0.spatial import check_cannibalization, coarse_road_distance

TRUSTED_SPATIAL_SOURCES = {
    GeoSource.MANUAL,
    GeoSource.PIN_EXACT,
    GeoSource.NOMINATIM,
}


def _evaluate_candidate(
    candidate: DistributorCandidate,
    tata_depot: TataDepot,
    existing_distributor_coords: list[tuple[float, float]],
    config: PipelineConfig,
) -> PrefilterResult:
    """Evaluate a single candidate against all hard gates in sequence."""
    rejection_reasons: list[str] = []
    needs_data_reasons: list[str] = []
    coarse_dist: Optional[float] = None
    nearest_dist: Optional[float] = None

    # Gate a: GSTIN validity
    gstin = candidate.gstin.strip() if candidate.gstin else ""
    if not is_valid_gstin(gstin):
        rejection_reasons.append(f"Invalid GSTIN format or checksum: '{candidate.gstin}'")

    # Gate b: Registration status
    status = (candidate.registration_status or "").strip().lower()
    if status != "active":
        rejection_reasons.append(
            f"Registration status not active: '{candidate.registration_status}'"
        )

    # Gate c: Steel HSN gate
    # The prefix list is read from config (not hardcoded) so that changing
    # `gates.required_hsn_prefixes` actually changes the gate. The
    # loader-populated flag is only used as a fallback for candidates
    # built without an explicit hsn_codes list.
    hsn_prefixes = tuple(config.gates.required_hsn_prefixes)
    has_steel_hsn = (
        any(code.startswith(hsn_prefixes) for code in candidate.hsn_codes)
        if candidate.hsn_codes
        else candidate.has_steel_hsn
    )
    if not has_steel_hsn or candidate.is_sac_only:
        rejection_reasons.append(
            f"Missing steel HSN (chapter {'/'.join(hsn_prefixes)}) or entity is SAC-only"
        )

    # Gate d: Business activity gate
    core_business = (candidate.nature_of_core_business or "").strip().upper()
    allowed_activities = {
        activity.strip().upper() for activity in config.gates.allowed_business_activities
    }
    if core_business and core_business not in allowed_activities:
        rejection_reasons.append(
            f"Core business '{candidate.nature_of_core_business}' not in allowed activities "
            f"{config.gates.allowed_business_activities}"
        )

    # Gate e: Turnover gate
    if not candidate.turnover_known:
        needs_data_reasons.append("Turnover is not disclosed or could not be parsed")
    elif candidate.turnover_midpoint_cr < config.gates.min_turnover_cr:
        rejection_reasons.append(
            f"Turnover ₹{candidate.turnover_midpoint_cr:.1f}Cr below minimum "
            f"{config.gates.min_turnover_cr}Cr"
        )
    elif (
        config.gates.max_turnover_cr is not None
        and candidate.turnover_midpoint_cr > config.gates.max_turnover_cr
    ):
        rejection_reasons.append(
            f"Turnover ₹{candidate.turnover_midpoint_cr:.1f}Cr exceeds configured maximum "
            f"{config.gates.max_turnover_cr}Cr"
        )

    # Gate f: Geocoding check
    if candidate.latitude is None or candidate.longitude is None:
        needs_data_reasons.append("Missing geocoded coordinates (latitude/longitude)")
    elif candidate.geocode_source not in TRUSTED_SPATIAL_SOURCES:
        needs_data_reasons.append(
            f"Location requires verification (source: {candidate.geocode_source.value})"
        )
    else:
        # Gate g: Anti-cannibalization
        too_close, nearest_dist = check_cannibalization(
            candidate.latitude,
            candidate.longitude,
            existing_distributor_coords,
            config.gates.min_cannibalization_radius_km,
        )
        if too_close:
            rejection_reasons.append(
                f"Cannibalization conflict: {nearest_dist:.2f} km to nearest distributor "
                f"(min {config.gates.min_cannibalization_radius_km} km required)"
            )

        # Gate h: Coarse distance to Tata depot
        coarse_dist = coarse_road_distance(
            candidate.latitude,
            candidate.longitude,
            tata_depot.latitude,
            tata_depot.longitude,
            k_tort=config.spatial.road_tortuosity_factor,
        )
        if coarse_dist > config.gates.max_coarse_distance_km:
            rejection_reasons.append(
                f"Coarse road distance {coarse_dist:.2f} km exceeds maximum "
                f"{config.gates.max_coarse_distance_km} km"
            )

    if rejection_reasons:
        disposition = CandidateDisposition.REJECTED
    elif needs_data_reasons:
        disposition = CandidateDisposition.NEEDS_DATA
    else:
        disposition = CandidateDisposition.ELIGIBLE

    return PrefilterResult(
        candidate=candidate,
        disposition=disposition,
        passed=disposition == CandidateDisposition.ELIGIBLE,
        rejection_reasons=rejection_reasons,
        needs_data_reasons=needs_data_reasons,
        coarse_road_distance_km=coarse_dist,
        nearest_existing_distributor_km=nearest_dist,
    )


# ────────────────────────────────────────────────────────
# Rejection Reason Categorisation
# ────────────────────────────────────────────────────────

# Stable gate identifiers used to group rejection reasons in the summary.
# The human-readable reason strings embed variable data (distances, turnover
# values) which must not fragment the breakdown into one row per candidate.
REASON_CATEGORIES: list[tuple[str, str]] = [
    ("Invalid GSTIN format", "GSTIN invalid"),
    ("Registration status not active", "Registration not active"),
    ("Missing steel HSN", "Missing steel HSN / SAC-only"),
    ("Core business", "Core business not allowed"),
    ("Turnover", "Turnover outside policy"),
    ("Missing geocoded coordinates", "Missing coordinates"),
    ("Location requires verification", "Location needs verification"),
    ("Cannibalization conflict", "Anti-cannibalization conflict"),
    ("Coarse road distance", "Distance exceeds maximum"),
]


def categorise_rejection(reason: str) -> str:
    """Map a raw rejection reason to its stable gate category label."""
    for prefix, label in REASON_CATEGORIES:
        if reason.startswith(prefix):
            return label
    return reason.split(":")[0] if ":" in reason else reason


def print_prefilter_summary(
    results: list[PrefilterResult],
    console: Optional[Console] = None,
) -> None:
    """Print a rich terminal summary of pre-filter results."""
    con = console or Console()
    total = len(results)
    eligible_count = sum(
        1 for result in results if result.disposition == CandidateDisposition.ELIGIBLE
    )
    needs_data_count = sum(
        1 for result in results if result.disposition == CandidateDisposition.NEEDS_DATA
    )
    rejected_count = sum(
        1 for result in results if result.disposition == CandidateDisposition.REJECTED
    )

    summary_table = Table(title="Pre-Filter Stage Summary", show_header=True)
    summary_table.add_column("Metric", style="bold cyan")
    summary_table.add_column("Count", justify="right", style="bold")
    summary_table.add_column("Percentage", justify="right")

    summary_table.add_row("Total Evaluated", str(total), "100.0%")
    pct_eligible = (eligible_count / total * 100.0) if total > 0 else 0.0
    pct_needs_data = (needs_data_count / total * 100.0) if total > 0 else 0.0
    pct_fail = (rejected_count / total * 100.0) if total > 0 else 0.0
    summary_table.add_row(
        "Eligible",
        f"[green]{eligible_count}[/green]",
        f"{pct_eligible:.1f}%",
    )
    summary_table.add_row(
        "Needs Data",
        f"[yellow]{needs_data_count}[/yellow]",
        f"{pct_needs_data:.1f}%",
    )
    summary_table.add_row("Rejected", f"[red]{rejected_count}[/red]", f"{pct_fail:.1f}%")
    con.print(summary_table)

    if rejected_count > 0 or needs_data_count > 0:
        reason_counts: Counter[str] = Counter()
        for r in results:
            for reason in r.rejection_reasons:
                reason_counts[categorise_rejection(reason)] += 1
            for reason in r.needs_data_reasons:
                reason_counts[categorise_rejection(reason)] += 1

        reason_table = Table(title="Rejection Reasons Breakdown", show_header=True)
        reason_table.add_column("Gate / Reason Category", style="yellow")
        reason_table.add_column("Failures", justify="right", style="bold red")

        for reason, count in reason_counts.most_common():
            reason_table.add_row(reason, str(count))

        con.print(reason_table)


def prefilter(
    candidates: list[DistributorCandidate],
    tata_depot: TataDepot,
    existing_distributor_coords: list[tuple[float, float]],
    config: PipelineConfig,
) -> list[PrefilterResult]:
    """Run deterministic pre-screening on all distributor candidates.

    Args:
        candidates: List of normalized candidate records.
        tata_depot: Origin depot or plant location.
        existing_distributor_coords: Coordinates of current network distributors.
        config: Pipeline configuration containing gate thresholds.

    Returns:
        List of PrefilterResult objects, with eligible candidates first,
        followed by Needs Data and rejected records.
    """
    evaluated: list[PrefilterResult] = [
        _evaluate_candidate(
            candidate,
            tata_depot,
            existing_distributor_coords,
            config,
        )
        for candidate in candidates
    ]

    print_prefilter_summary(evaluated)

    eligible_results = [
        result for result in evaluated if result.disposition == CandidateDisposition.ELIGIBLE
    ]
    needs_data_results = [
        result for result in evaluated if result.disposition == CandidateDisposition.NEEDS_DATA
    ]
    rejected_results = [
        result for result in evaluated if result.disposition == CandidateDisposition.REJECTED
    ]
    return eligible_results + needs_data_results + rejected_results
