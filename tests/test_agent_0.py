"""Regression tests for Agent 0's deterministic and evidence boundaries."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from agent_0.agent import _run_claude, _run_gemini, run_tot_s2t
from agent_0.cli import _evaluations_to_df, _export_run, app
from agent_0.config import PipelineConfig, load_config
from agent_0.csv_loader import (
    gstin_check_digit,
    is_valid_gstin,
    parse_distributor_row,
    parse_turnover_slab,
)
from agent_0.enrichment import (
    _cache_entry_is_fresh,
    _cache_filename,
    enrich_entity,
    validate_evidence_citations,
)
from agent_0.geocoder import geocode_candidates
from agent_0.models import (
    CandidateDisposition,
    DistributorCandidate,
    EnrichmentResult,
    EnrichmentSource,
    EvidenceSource,
    GeoSource,
    RecommendationState,
    S2TVerificationOutput,
    TataDepot,
)
from agent_0.pipeline import run_pipeline
from agent_0.prefilter import _evaluate_candidate
from agent_0.scoring import score_candidate


def _gstin(index: int, state: str = "07") -> str:
    """Build a valid GSTIN-shaped test identifier with a real checksum."""
    prefix = f"{state}ABCDE{index:04d}F1Z"
    return prefix + gstin_check_digit(prefix)


def _plain_cli_output(output: str) -> str:
    """Remove Rich framing before asserting terminal diagnostics."""
    without_box_drawing = re.sub(r"[\u2500-\u257f]", " ", output)
    return " ".join(without_box_drawing.split())


def _candidate(
    index: int = 1,
    *,
    latitude: float | None = 28.6139,
    longitude: float | None = 77.2090,
    geocode_source: GeoSource = GeoSource.MANUAL,
    turnover: float = 62.5,
) -> DistributorCandidate:
    return DistributorCandidate(
        gstin=_gstin(index),
        pan=f"ABCDE{index:04d}F",
        legal_name=f"Steel {index}",
        state_code="07",
        state_name="Delhi",
        registration_status="Active",
        nature_of_core_business="MFT",
        hsn_codes=["7214"],
        has_steel_hsn=True,
        latitude=latitude,
        longitude=longitude,
        geocode_source=geocode_source,
        turnover_midpoint_cr=turnover,
        turnover_known=turnover > 0,
    )


def _sources() -> list[EvidenceSource]:
    return [
        EvidenceSource(
            source_id="firecrawl-1",
            url="https://example.test/yard",
            title="Acme Steel yard",
            content=(
                "Acme Steel operates a multi-brand stockyard with a gantry crane "
                "in Delhi for steel coils and bars."
            ),
        ),
        EvidenceSource(
            source_id="firecrawl-2",
            url="https://example.test/brands",
            title="Acme Steel brands",
            content="Acme Steel stocks SAIL and JSW products for regional customers.",
        ),
    ]


def _s2t_payload() -> dict:
    return {
        "is_primary_steel_dealer": True,
        "brands_currently_handled": ["SAIL", "JSW"],
        "is_exclusive_to_competitor": False,
        "switching_feasibility": "HIGH",
        "has_physical_stockyard": True,
        "processing_capability": "ADVANCED",
        "promoter_litigation_flag": False,
        "gst_filing_regularity": "REGULAR",
        "evidence_confidence": "STRONG",
        "verbatim_evidence": "Evidence is retained in source-level citations.",
        "falsification_notes": "Name and location still require field confirmation.",
        "tot": {
            "brand_alignment": {
                "conclusion": "The company handles multiple steel brands.",
                "confidence": "STRONG",
                "citations": [
                    {
                        "source_id": "firecrawl-2",
                        "quote": "Acme Steel stocks SAIL and JSW products for regional customers.",
                    }
                ],
                "falsification": "The web page could refer to a namesake business.",
            },
            "infrastructure": {
                "conclusion": "The company has a physical stockyard.",
                "confidence": "STRONG",
                "citations": [
                    {
                        "source_id": "firecrawl-1",
                        "quote": "Acme Steel operates a multi-brand stockyard with a gantry crane",
                    }
                ],
                "falsification": "The yard description could be outdated.",
            },
            "onboarding_risk": {
                "conclusion": "No authoritative compliance source was retrieved.",
                "confidence": "NONE",
                "citations": [],
                "falsification": "Compliance must be checked against authoritative records.",
            },
        },
        "stage_2": {
            "supported_claims": [
                "The candidate handles SAIL and JSW products.",
                "The candidate operates a stockyard.",
            ],
            "challenged_claims": ["Dealer identity needs GSTIN/location confirmation."],
            "unresolved_questions": ["No authoritative filing record was retrieved."],
            "final_rationale": "Two retained public sources support the commercial lead.",
        },
    }


def _strong_s2t() -> S2TVerificationOutput:
    return S2TVerificationOutput.model_validate(_s2t_payload())


def _write_csv(
    tmp_path: Path,
    rows: list[dict[str, str]],
    filename: str = "distributors.csv",
) -> Path:
    path = tmp_path / filename
    headers = [
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
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(
    index: int,
    *,
    gstin: str | None = None,
    pin: str = "110020",
    turnover: str = "Rs. 25 Cr to Rs. 100 Cr",
    latitude: str = "28.6139",
    longitude: str = "77.2090",
    status: str = "Active",
) -> dict[str, str]:
    return {
        "GSTIN": gstin or _gstin(index),
        "PAN": f"ABCDE{index:04d}F",
        "LEGAL NAME": f"Steel {index}",
        "TRADE NAME": f"Steel {index}",
        "REGISTRATION STATUS": status,
        "NATURE OF CORE BUSINESS ACTIVITIES": "MFT",
        "AGGRE TURNOVER": turnover,
        "PERMANENT ADDRESS": f"Plot {index}, Delhi {pin}",
        "HSN CODE 1": "7214",
        "LATITUDE": latitude,
        "LONGITUDE": longitude,
    }


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output


def test_gstin_format_and_checksum_validation() -> None:
    assert is_valid_gstin(_gstin(1))
    assert not is_valid_gstin("../outside-file")
    assert not is_valid_gstin(_gstin(1)[:-1] + "0")


def test_turnover_slab_parsing() -> None:
    assert parse_turnover_slab("Rs. 40 to Rs. 1.5 Cr") == 0.95
    assert parse_turnover_slab("Slab: Rs. 100 Cr. to 500 Cr.") == 300.0
    assert parse_turnover_slab("Slab: Rs. 500 Cr. and above") == 750.0
    assert parse_turnover_slab("Not Available") == 0.0


def test_parser_reads_manual_coordinates_and_normalizes_headers() -> None:
    row = pd.Series(
        {
            "GSTIN": _gstin(2),
            "LAT": "12.9716",
            "LNG": "77.5946",
            "AGGRE TURNOVER": "Rs. 25 Cr to Rs. 100 Cr",
        }
    )
    candidate = parse_distributor_row(row)
    assert candidate is not None
    assert candidate.latitude == 12.9716
    assert candidate.longitude == 77.5946
    assert candidate.geocode_source == GeoSource.MANUAL
    assert candidate.turnover_known


def test_exact_and_district_pin_provenance_are_distinct() -> None:
    exact = _candidate(3, latitude=None, longitude=None, geocode_source=GeoSource.NONE)
    exact.pincode = "831001"
    district = _candidate(4, latitude=None, longitude=None, geocode_source=GeoSource.NONE)
    district.pincode = "110020"
    result = geocode_candidates([exact, district])
    assert result[0].geocode_source == GeoSource.PIN_EXACT
    assert result[1].geocode_source == GeoSource.PIN_DISTRICT


def test_nominatim_address_lookup_precedes_district_fallback(tmp_path: Path) -> None:
    import agent_0.geocoder as geocoder

    candidate = _candidate(41, latitude=None, longitude=None, geocode_source=GeoSource.NONE)
    candidate.pincode = "110020"
    candidate.permanent_address = "Plot 1, Delhi 110020"
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: [{"lat": "28.6200", "lon": "77.2200"}],
    )
    with (
        patch.object(geocoder.requests, "get", return_value=response) as get,
        patch.object(geocoder.time, "sleep"),
    ):
        result = geocode_candidates(
            [candidate],
            use_nominatim=True,
            cache_file=tmp_path / "nominatim-cache.json",
        )
    assert result[0].geocode_source == GeoSource.NOMINATIM
    assert get.call_args.kwargs["params"]["q"].startswith("Plot 1, Delhi")


def test_district_location_routes_to_needs_data() -> None:
    candidate = _candidate(
        5,
        latitude=28.61,
        longitude=77.20,
        geocode_source=GeoSource.PIN_DISTRICT,
    )
    result = _evaluate_candidate(
        candidate,
        TataDepot(name="Delhi", latitude=28.6139, longitude=77.2090),
        [],
        PipelineConfig(),
    )
    assert result.disposition == CandidateDisposition.NEEDS_DATA
    assert not result.passed
    assert "Location requires verification" in result.needs_data_reasons[0]


def test_large_turnover_remains_eligible_under_hybrid_policy() -> None:
    candidate = _candidate(51, turnover=750.0)
    result = _evaluate_candidate(
        candidate,
        TataDepot(name="Delhi", latitude=28.6139, longitude=77.2090),
        [],
        PipelineConfig(),
    )
    assert result.disposition == CandidateDisposition.ELIGIBLE


def test_pipeline_offline_disables_every_network_boundary(tmp_path: Path) -> None:
    import agent_0.pipeline as pipeline

    csv_path = _write_csv(tmp_path, [_row(6)])
    config = PipelineConfig()
    config.gmaps_api_key = "maps-key"
    config.firecrawl_api_key = "firecrawl-key"
    config.anthropic_api_key = "anthropic-key"
    with (
        patch.object(pipeline, "google_maps_distance") as maps,
        patch.object(pipeline, "batch_enrich") as enrich,
        patch.object(pipeline, "run_tot_s2t") as verify,
    ):
        result = run_pipeline(
            csv_path,
            TataDepot(name="Delhi", latitude=28.6139, longitude=77.2090),
            config,
            offline=True,
        )
    assert len(result.evaluations) == 1
    assert result.evaluations[0].recommendation_state == RecommendationState.PRIORITIZED_LEAD
    assert maps.call_count == enrich.call_count == verify.call_count == 0
    assert result.manifest and result.manifest.options["offline"] is True


def test_manifest_never_contains_configured_api_keys(tmp_path: Path) -> None:
    config = PipelineConfig()
    config.anthropic_api_key = "secret"
    config.firecrawl_api_key = "secret"
    result = run_pipeline(
        _write_csv(tmp_path, [_row(61)]),
        TataDepot(name="Delhi", latitude=28.6139, longitude=77.2090),
        config,
        offline=True,
    )
    manifest_json = json.dumps(result.manifest.model_dump(mode="json"))
    assert "secret" not in manifest_json
    assert "anthropic_api_key" not in manifest_json


def test_pipeline_exports_needs_data_and_rejections(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        [
            _row(7),
            _row(8, latitude="", longitude=""),
            _row(9, gstin="INVALID-GSTIN-1"),
        ],
    )
    result = run_pipeline(
        csv_path,
        TataDepot(name="Delhi", latitude=28.6139, longitude=77.2090),
        PipelineConfig(),
        offline=True,
    )
    assert len(result.evaluations) == 1
    assert len(result.needs_data) == 1
    assert len(result.rejected) == 1


def test_existing_manual_locations_enforce_cannibalization(tmp_path: Path) -> None:
    candidate_csv = _write_csv(tmp_path, [_row(10)], "candidate.csv")
    existing_csv = _write_csv(tmp_path, [_row(11)], "existing.csv")
    result = run_pipeline(
        candidate_csv,
        TataDepot(name="Delhi", latitude=28.6139, longitude=77.2090),
        PipelineConfig(),
        existing_tata_csv=existing_csv,
        offline=True,
    )
    assert not result.evaluations
    assert len(result.rejected) == 1
    assert "Cannibalization conflict" in result.rejected[0].reasons[0]


def test_cancelled_verification_is_always_rejected() -> None:
    verification = _strong_s2t()
    verification.gst_filing_regularity = "CANCELLED"
    evaluation = score_candidate(
        _candidate(12),
        0.0,
        "COARSE",
        verification,
        PipelineConfig(),
        enrichment_source=EnrichmentSource.FIRECRAWL,
        evidence_validation=validate_evidence_citations(verification, _sources()),
    )
    assert evaluation.final_grade == "D"
    assert evaluation.recommendation_state == RecommendationState.REJECTED


def test_valid_nested_citations_enable_verified_grade() -> None:
    verification = _strong_s2t()
    validation = validate_evidence_citations(verification, _sources())
    evaluation = score_candidate(
        _candidate(13),
        10.0,
        "COARSE",
        verification,
        PipelineConfig(),
        enrichment_source=EnrichmentSource.FIRECRAWL,
        evidence_validation=validation,
    )
    assert validation.verified
    assert evaluation.final_grade == "A"
    assert evaluation.recommendation_state == RecommendationState.VERIFIED_RECOMMENDATION


def test_invalid_citation_caps_grade_at_c() -> None:
    payload = _s2t_payload()
    payload["tot"]["brand_alignment"]["citations"][0]["quote"] = "Not in the source"
    verification = S2TVerificationOutput.model_validate(payload)
    validation = validate_evidence_citations(verification, _sources())
    evaluation = score_candidate(
        _candidate(14),
        10.0,
        "COARSE",
        verification,
        PipelineConfig(),
        enrichment_source=EnrichmentSource.FIRECRAWL,
        evidence_validation=validation,
    )
    assert not validation.verified
    assert evaluation.final_grade == "C"
    assert evaluation.recommendation_state == RecommendationState.PRIORITIZED_LEAD


def test_pipeline_promotes_only_citation_validated_nested_evidence(tmp_path: Path) -> None:
    import agent_0.pipeline as pipeline

    csv_path = _write_csv(tmp_path, [_row(141)])
    config = PipelineConfig()
    config.firecrawl_api_key = "firecrawl-key"
    config.anthropic_api_key = "anthropic-key"
    evidence = EnrichmentResult(
        source=EnrichmentSource.FIRECRAWL,
        sources=_sources(),
    )

    def fake_llm(candidate, sources, config, meta=None):
        if meta is not None:
            meta.update({"provider": "anthropic", "model": "claude-test", "fallback": False})
        return _strong_s2t()

    with (
        patch.object(pipeline, "batch_enrich", return_value={_gstin(141): evidence}),
        patch.object(pipeline, "run_tot_s2t", side_effect=fake_llm),
    ):
        result = run_pipeline(
            csv_path,
            TataDepot(name="Delhi", latitude=28.6139, longitude=77.2090),
            config,
        )
    assert result.evaluations[0].recommendation_state == RecommendationState.VERIFIED_RECOMMENDATION
    assert result.evaluations[0].verification_model == "claude-test"


def test_pipeline_does_not_call_llm_without_retained_sources(tmp_path: Path) -> None:
    import agent_0.pipeline as pipeline

    csv_path = _write_csv(tmp_path, [_row(142)])
    config = PipelineConfig()
    config.firecrawl_api_key = "firecrawl-key"
    config.anthropic_api_key = "anthropic-key"
    with (
        patch.object(
            pipeline,
            "batch_enrich",
            return_value={
                _gstin(142): EnrichmentResult(
                    source=EnrichmentSource.MOCK,
                    error="No source text",
                )
            },
        ),
        patch.object(pipeline, "run_tot_s2t") as verify,
    ):
        result = run_pipeline(
            csv_path,
            TataDepot(name="Delhi", latitude=28.6139, longitude=77.2090),
            config,
        )
    assert not verify.called
    assert result.evaluations[0].recommendation_state == RecommendationState.PRIORITIZED_LEAD


def test_llm_output_contract_is_strict() -> None:
    payload = _s2t_payload()
    payload["is_primary_steel_dealer"] = "true"
    with pytest.raises(ValidationError):
        S2TVerificationOutput.model_validate(payload)
    payload = _s2t_payload()
    payload["unexpected"] = "ignored previously"
    with pytest.raises(ValidationError):
        S2TVerificationOutput.model_validate(payload)


def test_firecrawl_preserves_per_source_provenance_and_safe_cache(tmp_path: Path) -> None:
    import agent_0.enrichment as enrichment

    response = {
        "data": [
            {"url": "https://one.test", "title": "One", "markdown": "one " * 2000},
            {"url": "https://two.test", "title": "Two", "markdown": "two evidence"},
        ]
    }
    with patch.object(enrichment, "_post_firecrawl_http", return_value=response):
        result = enrich_entity(
            "Test Steel",
            _gstin(15),
            "Delhi",
            firecrawl_api_key="test-key",
            cache_dir=str(tmp_path),
            max_chars_per_source=100,
        )
    assert result.source == EnrichmentSource.FIRECRAWL
    assert len(result.sources) == 2
    assert len(result.sources[0].content) == 100
    assert result.sources[1].url == "https://two.test"
    assert (tmp_path / _cache_filename(_gstin(15))).is_file()
    assert "/" not in _cache_filename("../outside-file")


def test_firecrawl_cache_expires_and_legacy_payload_is_stale() -> None:
    old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    current = {
        "schema_version": 2,
        "timestamp": old,
        "sources": [{"source_id": "s", "content": "Evidence text"}],
    }
    assert not _cache_entry_is_fresh(current, ttl_days=30)
    assert _cache_entry_is_fresh(current, ttl_days=365)
    assert not _cache_entry_is_fresh({"text": "old flattened cache"}, ttl_days=30)


def test_firecrawl_failures_do_not_create_evidence(tmp_path: Path) -> None:
    import agent_0.enrichment as enrichment

    with patch.object(enrichment, "_post_firecrawl_http", side_effect=RuntimeError("429")):
        result = enrich_entity(
            "Test Steel",
            _gstin(16),
            "Delhi",
            firecrawl_api_key="test-key",
            cache_dir=str(tmp_path),
            max_retries=0,
        )
    assert result.source == EnrichmentSource.MOCK
    assert not result.sources


def test_claude_receives_source_addressable_prompt() -> None:
    captured: dict = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="emit_verification",
                        input=_s2t_payload(),
                    )
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    config = PipelineConfig()
    config.anthropic_api_key = "key"
    with patch("agent_0.agent.anthropic.Anthropic", FakeClient):
        result = _run_claude(_candidate(17), _sources(), config)
    assert result.tot.infrastructure.citations[0].source_id == "firecrawl-1"
    assert "[SOURCE firecrawl-1]" in captured["messages"][0]["content"]


def test_gemini_uses_nested_schema() -> None:
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=json.dumps(_s2t_payload())))
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    config = PipelineConfig()
    config.gemini_api_key = "key"
    with patch("agent_0.agent.openai.OpenAI", FakeClient):
        result = _run_gemini(_candidate(18), _sources(), config)
    assert result.stage_2.final_rationale
    assert captured["response_format"]["json_schema"]["strict"] is True


def test_provider_fallback_keeps_existing_behavior() -> None:
    import agent_0.agent as agent

    config = PipelineConfig()
    config.anthropic_api_key = "primary"
    config.openai_api_key = "fallback"
    metadata: dict = {}
    with (
        patch.object(agent, "_run_claude", side_effect=RuntimeError("quota")),
        patch.object(agent, "_run_openai", return_value=_strong_s2t()),
    ):
        result = run_tot_s2t(_candidate(19), _sources(), config, metadata)
    assert result.evidence_confidence == "STRONG"
    assert metadata == {"provider": "openai", "fallback": True, "model": "gpt-4o"}


def test_config_rejects_semantic_errors_and_yaml_secrets(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(profit_weights={"logistics": 1, "turnover": 1, "processing": 1})
    with pytest.raises(ValidationError):
        PipelineConfig(enrichment={"top_n_for_enrichment": -1})
    with pytest.raises(ValidationError):
        PipelineConfig(unknown_key=True)

    config_file = tmp_path / "config.yaml"
    config_file.write_text("anthropic_api_key: do-not-store-secrets-here\n")
    with pytest.raises(ValueError, match="API credentials"):
        load_config(config_path=config_file, env_path=tmp_path / "missing.env")


def test_explicit_missing_cli_paths_are_rejected(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, [_row(20)])
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--distributor-csv",
            str(csv_path),
            "--depot-name",
            "Delhi",
            "--depot-lat",
            "28.6139",
            "--depot-lon",
            "77.2090",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--offline",
        ],
    )
    assert result.exit_code == 2
    output = _plain_cli_output(result.output)
    assert "Invalid value" in output
    assert "does not exist" in output


def test_explicit_missing_existing_network_path_is_rejected(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, [_row(71)])
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--distributor-csv",
            str(csv_path),
            "--depot-name",
            "Delhi",
            "--depot-lat",
            "28.6139",
            "--depot-lon",
            "77.2090",
            "--tata-csv",
            str(tmp_path / "missing-existing.csv"),
            "--offline",
        ],
    )
    assert result.exit_code == 2
    output = _plain_cli_output(result.output)
    assert "Invalid value" in output
    assert "does not exist" in output


def test_csv_export_escapes_formulas() -> None:
    candidate = _candidate(21)
    candidate.legal_name = '=HYPERLINK("https://evil.test")'
    evaluation = score_candidate(
        candidate,
        10.0,
        "COARSE",
        _strong_s2t(),
        PipelineConfig(),
        enrichment_source=EnrichmentSource.FIRECRAWL,
        evidence_validation=validate_evidence_citations(_strong_s2t(), _sources()),
    )
    dataframe = _evaluations_to_df([evaluation])
    assert dataframe.loc[0, "Legal Name"].startswith("'=")


def test_queue_and_manifest_artifacts_are_exported(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, [_row(22)])
    result = run_pipeline(
        csv_path,
        TataDepot(name="Delhi", latitude=28.6139, longitude=77.2090),
        PipelineConfig(),
        offline=True,
    )
    _export_run(result, tmp_path / "out", "all")
    expected = {
        "agent0_results.csv",
        "agent0_prioritized_leads.csv",
        "agent0_needs_data.csv",
        "agent0_prefilter_rejected.csv",
        "agent0_run_manifest.json",
    }
    assert expected.issubset({path.name for path in (tmp_path / "out").iterdir()})
    manifest = json.loads((tmp_path / "out" / "agent0_run_manifest.json").read_text())
    assert manifest["distributor_csv_sha256"] == result.manifest.distributor_csv_sha256
