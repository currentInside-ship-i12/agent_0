"""Mocked scale, boundary, and failure tests for Agent 0."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_0.agent import _get_safe_default_output
from agent_0.config import PipelineConfig
from agent_0.csv_loader import gstin_check_digit, load_distributor_csv
from agent_0.models import (
    DistributorCandidate,
    EnrichmentSource,
    EvidenceSource,
    GeoSource,
    RecommendationState,
    S2TVerificationOutput,
    TataDepot,
)
from agent_0.pipeline import run_pipeline
from agent_0.scoring import score_candidate

DELHI = TataDepot(name="Delhi Hub", latitude=28.6139, longitude=77.2090)
HEADER = [
    "GSTIN",
    "PAN",
    "LEGAL NAME",
    "TRADE NAME",
    "REGISTRATION STATUS",
    "NATURE OF CORE BUSINESS ACTIVITIES",
    "AGGRE TURNOVER",
    "PERMANENT ADDRESS",
    "HSN CODE 1",
    "LATITUDE",
    "LONGITUDE",
]


def _gstin(index: int, state: str = "07") -> str:
    prefix = f"{state}ABCDE{index:04d}F1Z"
    return prefix + gstin_check_digit(prefix)


def _row(
    index: int,
    *,
    pin: str = "110020",
    turnover: str = "Rs. 25 Cr to Rs. 100 Cr",
    hsn: str = "7214",
    core: str = "MFT",
    legal: str | None = None,
    gstin: str | None = None,
    latitude: str = "28.6139",
    longitude: str = "77.2090",
) -> list[str]:
    return [
        gstin or _gstin(index),
        f"ABCDE{index:04d}F",
        legal or f"Steel {index}",
        f"Steel {index}",
        "Active",
        core,
        turnover,
        f"Plot {index}, Delhi {pin}",
        hsn,
        latitude,
        longitude,
    ]


def _write_csv(rows: list[list[str]]) -> Path:
    path = Path(tempfile.mkdtemp()) / "distributors.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def _strong() -> S2TVerificationOutput:
    return S2TVerificationOutput(
        is_primary_steel_dealer=True,
        brands_currently_handled=["SAIL"],
        switching_feasibility="HIGH",
        has_physical_stockyard=True,
        processing_capability="ADVANCED",
        gst_filing_regularity="REGULAR",
        evidence_confidence="STRONG",
    )


def test_scale_1500_rows_offline() -> None:
    rows = [_row(index) for index in range(1, 1501)]
    result = run_pipeline(
        distributor_csv=_write_csv(rows),
        tata_depot=DELHI,
        config=PipelineConfig(),
        offline=True,
    )
    assert len(result.evaluations) == 1500
    assert not result.needs_data
    assert all(
        evaluation.recommendation_state == RecommendationState.PRIORITIZED_LEAD
        for evaluation in result.evaluations
    )


def test_cannibalization_many_existing_records() -> None:
    candidate = _write_csv([_row(1)])
    existing = _write_csv([_row(index) for index in range(1000, 1500)])
    result = run_pipeline(
        distributor_csv=candidate,
        tata_depot=DELHI,
        config=PipelineConfig(),
        existing_tata_csv=existing,
        offline=True,
    )
    assert not result.evaluations
    assert len(result.rejected) == 1


@pytest.mark.parametrize(
    ("turnover", "expected"),
    [
        (0.0, 0.0),
        (14.9, 24.83),
        (15.0, 25.0),
        (50.0, 100.0),
        (150.0, 100.0),
        (300.0, 100.0),
        (750.0, 100.0),
    ],
)
def test_turnover_score_boundaries(turnover: float, expected: float) -> None:
    candidate = DistributorCandidate(
        gstin=_gstin(2),
        pan="ABCDE0002F",
        legal_name="T",
        turnover_midpoint_cr=turnover,
        turnover_known=turnover > 0,
    )
    evaluation = score_candidate(
        candidate,
        50.0,
        "COARSE",
        _strong(),
        PipelineConfig(),
    )
    assert abs(evaluation.scores.s_turnover - expected) < 0.5


def test_malformed_and_missing_csv_raise_clear_errors() -> None:
    bad = Path(tempfile.mkdtemp()) / "bad.csv"
    bad.write_text('GSTIN,"unterminated\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse CSV"):
        load_distributor_csv(bad)
    with pytest.raises(FileNotFoundError):
        load_distributor_csv(Path(tempfile.mkdtemp()) / "missing.csv")


def test_gmaps_budget_only_applies_to_eligible_candidates() -> None:
    import agent_0.pipeline as pipeline

    rows = [
        _row(3, latitude="28.6139", longitude="77.2090"),
        _row(4, latitude="28.7139", longitude="77.2090"),
        _row(5, latitude="28.8139", longitude="77.2090"),
    ]
    config = PipelineConfig()
    config.gmaps_api_key = "maps-key"
    config.enrichment.gmaps_refine_top = 1
    calls: list[tuple] = []

    def fake_maps(*args, **kwargs):
        calls.append(args)
        return 42.0, "GMAPS"

    with patch.object(pipeline, "google_maps_distance", side_effect=fake_maps):
        result = run_pipeline(
            _write_csv(rows),
            DELHI,
            config,
            skip_enrichment=True,
            skip_llm=True,
        )
    assert len(calls) == 1
    assert sum(evaluation.distance_source == "GMAPS" for evaluation in result.evaluations) == 1


def test_offline_results_are_deterministic() -> None:
    path = _write_csv([_row(6), _row(7, latitude="28.7")])
    first = run_pipeline(path, DELHI, PipelineConfig(), offline=True)
    second = run_pipeline(path, DELHI, PipelineConfig(), offline=True)
    assert [item.model_dump(mode="json") for item in first.evaluations] == [
        item.model_dump(mode="json") for item in second.evaluations
    ]


def test_unknown_turnover_and_district_location_need_data() -> None:
    rows = [
        _row(8, turnover="", latitude="", longitude=""),
        _row(9, latitude="", longitude=""),
    ]
    result = run_pipeline(_write_csv(rows), DELHI, PipelineConfig(), offline=True)
    assert not result.evaluations
    assert len(result.needs_data) == 2
    assert any("Turnover" in reason for reason in result.needs_data[0].reasons)
    assert any("Location" in reason for reason in result.needs_data[1].reasons)


def test_failed_llm_is_a_prioritized_lead_not_a_recommendation() -> None:
    candidate = DistributorCandidate(
        gstin=_gstin(10),
        pan="ABCDE0010F",
        legal_name="T",
        turnover_midpoint_cr=62.5,
        turnover_known=True,
        geocode_source=GeoSource.MANUAL,
    )
    evaluation = score_candidate(
        candidate,
        30.0,
        "COARSE",
        _get_safe_default_output("all providers failed"),
        PipelineConfig(),
        enrichment_source=EnrichmentSource.NONE,
    )
    assert evaluation.final_grade == "C"
    assert evaluation.recommendation_state == RecommendationState.PRIORITIZED_LEAD


def test_untrusted_coordinates_are_not_used_for_hard_cannibalization() -> None:
    candidate = DistributorCandidate(
        gstin=_gstin(11),
        pan="ABCDE0011F",
        legal_name="T",
        latitude=28.61,
        longitude=77.20,
        geocode_source=GeoSource.STATE_CENTROID,
        turnover_midpoint_cr=62.5,
        turnover_known=True,
        registration_status="Active",
        hsn_codes=["7214"],
        has_steel_hsn=True,
        nature_of_core_business="MFT",
    )
    from agent_0.prefilter import _evaluate_candidate

    outcome = _evaluate_candidate(candidate, DELHI, [(28.61, 77.20)], PipelineConfig())
    assert outcome.disposition.value == "NEEDS_DATA"
    assert not any("Cannibalization" in reason for reason in outcome.rejection_reasons)


def test_formula_values_survive_pipeline_as_data() -> None:
    result = run_pipeline(
        _write_csv([_row(12, legal='=HYPERLINK("evil")')]),
        DELHI,
        PipelineConfig(),
        offline=True,
    )
    assert result.evaluations[0].legal_name.startswith("=")


def test_citation_validation_requires_actual_source_text() -> None:
    source = EvidenceSource(source_id="s1", content="warehouse evidence")
    output = S2TVerificationOutput(
        is_primary_steel_dealer=True,
        switching_feasibility="HIGH",
        has_physical_stockyard=True,
        processing_capability="ADVANCED",
        evidence_confidence="STRONG",
        tot={
            "brand_alignment": {
                "conclusion": "x",
                "citations": [{"source_id": "s1", "quote": "warehouse evidence"}],
            },
            "infrastructure": {
                "conclusion": "x",
                "citations": [{"source_id": "s1", "quote": "warehouse evidence"}],
            },
            "onboarding_risk": {"conclusion": "unknown"},
        },
        stage_2={"final_rationale": "x"},
    )
    from agent_0.enrichment import validate_evidence_citations

    assert validate_evidence_citations(output, [source]).verified
