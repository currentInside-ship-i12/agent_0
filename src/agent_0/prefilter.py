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
from agent_0.models import DistributorCandidate, PrefilterResult, TataDepot
from agent_0.spatial import check_cannibalization, coarse_road_distance


def _evaluate_candidate(
    candidate: DistributorCandidate,
    tata_depot: TataDepot,
    existing_distributor_coords: list[tuple[float, float]],
    config: PipelineConfig,
) -> PrefilterResult:
    """Evaluate a single candidate against all hard gates in sequence."""
    rejection_reasons: list[str] = []
    coarse_dist: Optional[float] = None
    nearest_dist: Optional[float] = None

    # Gate a: GSTIN validity
    gstin = candidate.gstin.strip() if candidate.gstin else ""
    if len(gstin) != 15:
        rejection_reasons.append(
            f"Invalid GSTIN length ({len(gstin)} != 15): '{candidate.gstin}'"
        )

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
    if (
        candidate.nature_of_core_business is not None
        and candidate.nature_of_core_business not in config.gates.allowed_business_activities
    ):
        rejection_reasons.append(
            f"Core business '{candidate.nature_of_core_business}' not in allowed activities "
            f"{config.gates.allowed_business_activities}"
        )

    # Gate e: Turnover gate
    # If turnover_midpoint_cr == 0.0, PASS (unknown / not-disclosed turnover)
    if candidate.turnover_midpoint_cr != 0.0 and not (
        config.gates.min_turnover_cr
        <= candidate.turnover_midpoint_cr
        <= config.gates.max_turnover_cr
    ):
        rejection_reasons.append(
            f"Turnover ₹{candidate.turnover_midpoint_cr:.1f}Cr outside allowed range "
            f"[{config.gates.min_turnover_cr}, {config.gates.max_turnover_cr}]Cr"
        )

    # Gate f: Geocoding check
    if candidate.latitude is None or candidate.longitude is None:
        rejection_reasons.append("Missing geocoded coordinates (latitude/longitude)")
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

    passed = len(rejection_reasons) == 0
    return PrefilterResult(
        candidate=candidate,
        passed=passed,
        rejection_reasons=rejection_reasons,
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
    ("Invalid GSTIN length", "GSTIN invalid"),
    ("Registration status not active", "Registration not active"),
    ("Missing steel HSN", "Missing steel HSN / SAC-only"),
    ("Core business", "Core business not allowed"),
    ("Turnover", "Turnover out of range"),
    ("Missing geocoded coordinates", "Missing coordinates"),
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
    passed_count = sum(1 for r in results if r.passed)
    rejected_count = total - passed_count

    summary_table = Table(title="Pre-Filter Stage Summary", show_header=True)
    summary_table.add_column("Metric", style="bold cyan")
    summary_table.add_column("Count", justify="right", style="bold")
    summary_table.add_column("Percentage", justify="right")

    summary_table.add_row("Total Evaluated", str(total), "100.0%")
    pct_pass = (passed_count / total * 100.0) if total > 0 else 0.0
    pct_fail = (rejected_count / total * 100.0) if total > 0 else 0.0
    summary_table.add_row("Passed", f"[green]{passed_count}[/green]", f"{pct_pass:.1f}%")
    summary_table.add_row("Rejected", f"[red]{rejected_count}[/red]", f"{pct_fail:.1f}%")
    con.print(summary_table)

    if rejected_count > 0:
        reason_counts: Counter[str] = Counter()
        for r in results:
            for reason in r.rejection_reasons:
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
        List of PrefilterResult objects, with passed candidates sorted first.
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

    passed_results = [r for r in evaluated if r.passed]
    failed_results = [r for r in evaluated if not r.passed]
    return passed_results + failed_results
