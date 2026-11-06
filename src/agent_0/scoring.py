"""Deterministic scoring and grading engine for Agent 0.

Computes multi-dimensional profit and ease scores, and assigns deterministic
letter grades (A, B, C, D) using strict mathematical formulas.
Zero LLM involvement.
"""

from __future__ import annotations

from typing import Literal, Optional

from agent_0.config import PipelineConfig
from agent_0.models import (
    DistributorCandidate,
    EnrichmentSource,
    FinalEvaluation,
    S2TVerificationOutput,
    ScoreBreakdown,
)


def _compute_profit_score(
    candidate: DistributorCandidate,
    driving_distance_km: float,
    s2t: S2TVerificationOutput,
    config: PipelineConfig,
) -> tuple[float, float, float, float]:
    """Calculate Profit Score (S_profit) and its sub-components in [0, 100]."""
    # 1. Logistics score
    d_max = config.spatial.d_target_max_km
    if d_max > 0:
        s_logistics = max(0.0, min(100.0, 100.0 * (1.0 - (driving_distance_km / d_max))))
    else:
        s_logistics = 0.0

    # 2. Turnover score (smooth bell curve on midpoint in ₹ Crores).
    # Continuous ramps — no cliff at 15 Cr:
    #   0 -> 15 Cr   : 0 -> 25
    #   15 -> 50 Cr  : 25 -> 100
    #   50 -> 150 Cr : 100 (plateau)
    #   150 -> 300 Cr: 100 -> 50 (taper, floored at 40)
    turnover = max(0.0, candidate.turnover_midpoint_cr)
    if 0.0 <= turnover < 15.0:
        s_turnover = 25.0 * (turnover / 15.0) if turnover > 0.0 else 0.0
    elif 15.0 <= turnover < 50.0:
        s_turnover = 25.0 + 75.0 * ((turnover - 15.0) / 35.0)
    elif 50.0 <= turnover <= 150.0:
        s_turnover = 100.0
    elif 150.0 < turnover <= 300.0:
        s_turnover = max(40.0, 100.0 - (turnover - 150.0) / 3.0)
    else:
        s_turnover = 0.0

    # 3. Processing capability score
    proc_map: dict[str, float] = {
        "ADVANCED": 100.0,
        "BASIC": 60.0,
        "NONE": 20.0,
    }
    s_processing = proc_map.get(s2t.processing_capability, 20.0)

    # Weighted combination
    w = config.profit_weights
    profit_score = (
        w.logistics * s_logistics
        + w.turnover * s_turnover
        + w.processing * s_processing
    )

    return s_logistics, s_turnover, s_processing, profit_score


def _compute_ease_score(
    s2t: S2TVerificationOutput,
    config: PipelineConfig,
) -> tuple[float, float, float, float]:
    """Calculate Ease Score (S_ease) and its sub-components in [0, 100]."""
    # 1. Compliance score
    reg = s2t.gst_filing_regularity
    lit = s2t.promoter_litigation_flag

    if reg == "REGULAR" and not lit:
        e_compliance = 100.0
    elif reg == "REGULAR" and lit:
        e_compliance = 30.0
    elif reg == "IRREGULAR":
        e_compliance = 50.0
    elif reg == "CANCELLED":
        e_compliance = 0.0
    elif reg == "UNKNOWN" and not lit:
        e_compliance = 70.0
    elif reg == "UNKNOWN" and lit:
        e_compliance = 20.0
    else:
        e_compliance = 0.0

    # 2. Switching feasibility score
    switch_map: dict[str, float] = {
        "HIGH": 100.0,
        "MEDIUM": 65.0,
        "LOW": 30.0,
    }
    e_switching = switch_map.get(s2t.switching_feasibility, 30.0)

    # 3. Infrastructure score
    if s2t.has_physical_stockyard:
        if s2t.processing_capability == "ADVANCED":
            e_infra = 100.0
        elif s2t.processing_capability == "BASIC":
            e_infra = 75.0
        else:
            e_infra = 50.0
    else:
        e_infra = 15.0

    # Weighted combination
    u = config.ease_weights
    ease_score = (
        u.compliance * e_compliance
        + u.switching * e_switching
        + u.infrastructure * e_infra
    )

    return e_compliance, e_switching, e_infra, ease_score


def _verification_status(
    s2t: S2TVerificationOutput,
    enrichment_source: EnrichmentSource,
) -> Literal["VERIFIED", "PARTIAL", "UNVERIFIED", "FAILED"]:
    if s2t.verbatim_evidence.startswith("LLM call failed"):
        return "FAILED"
    if s2t.evidence_confidence == "NONE":
        return "UNVERIFIED"
    if enrichment_source != EnrichmentSource.FIRECRAWL:
        return "PARTIAL"
    return "VERIFIED"


def score_candidate(
    candidate: DistributorCandidate,
    driving_distance_km: float,
    distance_source: Literal["GMAPS", "COARSE"],
    s2t: S2TVerificationOutput,
    config: PipelineConfig,
    enrichment_source: EnrichmentSource = EnrichmentSource.NONE,
    verification_provider: Optional[str] = None,
) -> FinalEvaluation:
    """Evaluate and grade a single distributor candidate.

    Args:
        candidate: Normalized candidate data.
        driving_distance_km: Driving road distance to Tata depot.
        distance_source: Source of driving distance ('GMAPS' or 'COARSE').
        s2t: Structured verification primitives.
        config: Pipeline configuration with weights and grading thresholds.
        enrichment_source: Source used for evidence enrichment.
        verification_provider: LLM provider that served the verification.

    Returns:
        FinalEvaluation with complete breakdown and letter grade.
    """
    s_logistics, s_turnover, s_processing, profit_score = _compute_profit_score(
        candidate=candidate,
        driving_distance_km=driving_distance_km,
        s2t=s2t,
        config=config,
    )

    e_compliance, e_switching, e_infra, ease_score = _compute_ease_score(
        s2t=s2t,
        config=config,
    )

    breakdown = ScoreBreakdown(
        s_logistics=round(s_logistics, 2),
        s_turnover=round(s_turnover, 2),
        s_processing=round(s_processing, 2),
        profit_score=round(profit_score, 2),
        e_compliance=round(e_compliance, 2),
        e_switching=round(e_switching, 2),
        e_infra=round(e_infra, 2),
        ease_score=round(ease_score, 2),
    )

    # Deterministic grading
    g = config.grading
    rejection_reasons: list[str] = []

    if not s2t.is_primary_steel_dealer:
        rejection_reasons.append("Entity is not a primary steel dealer")

    if s2t.gst_filing_regularity == "CANCELLED":
        rejection_reasons.append("GST filing regularity is CANCELLED (compliance hard fail)")

    confidence = s2t.confidence_score

    # Enforce Grade D ceiling: critically low profit AND ease can never
    # grade above D, regardless of other signals.
    if (
        profit_score <= g.grade_d_max_profit
        and ease_score <= g.grade_d_max_ease
    ):
        final_grade: Literal["A", "B", "C", "D"] = "D"
        rejection_reasons.append(
            f"Critically low scores: profit {profit_score:.1f} <= {g.grade_d_max_profit} "
            f"and ease {ease_score:.1f} <= {g.grade_d_max_ease}"
        )
    elif (
        profit_score >= g.grade_a_min_profit
        and ease_score >= g.grade_a_min_ease
        and confidence >= g.grade_a_min_confidence
        and s2t.is_primary_steel_dealer
    ):
        final_grade = "A"
    elif (
        profit_score >= g.grade_b_min_profit
        and ease_score >= g.grade_b_min_ease
        and s2t.is_primary_steel_dealer
    ):
        final_grade = "B"
    elif (
        profit_score >= g.grade_c_min_profit
        and s2t.is_primary_steel_dealer
    ):
        final_grade = "C"
    else:
        final_grade = "D"
        if s2t.is_primary_steel_dealer and not rejection_reasons:
            rejection_reasons.append(
                f"Below Grade C profit threshold: {profit_score:.1f} < {g.grade_c_min_profit}"
            )

    verification_status = _verification_status(s2t, enrichment_source)
    unverified = s2t.evidence_confidence == "NONE"
    if unverified and final_grade in ("A", "B"):
        final_grade = "C"
        rejection_reasons.append("Unverified (no LLM evidence) — grade capped at C")

    return FinalEvaluation(
        gstin=candidate.gstin,
        pan=candidate.pan,
        legal_name=candidate.legal_name,
        trade_name=candidate.trade_name,
        state=candidate.state_name or candidate.state_code,
        driving_distance_km=driving_distance_km,
        distance_source=distance_source,
        geocode_source=candidate.geocode_source,
        scores=breakdown,
        final_grade=final_grade,
        verification=s2t,
        enrichment_source=enrichment_source,
        verification_provider=verification_provider,
        verification_status=verification_status,
        rejection_reasons=rejection_reasons,
        turnover_slab=candidate.turnover_slab,
        permanent_address=candidate.permanent_address,
        ekyc_status=candidate.ekyc_status,
        aadhaar_verified=candidate.aadhaar_verified,
    )
