"""Deterministic regression tests for Agent 0 fixes."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from agent_0.cli import app
from agent_0.config import PipelineConfig
from agent_0.csv_loader import parse_distributor_row, parse_turnover_slab
from agent_0.geocoder import OFFLINE_PINCODE_COORDS, geocode_candidates
from agent_0.models import (
    DistributorCandidate,
    EnrichmentSource,
    FinalEvaluation,
    GeoSource,
    S2TVerificationOutput,
    TataDepot,
)
from agent_0.pipeline import run_pipeline
from agent_0.scoring import score_candidate


def _s2t_default() -> S2TVerificationOutput:
    return S2TVerificationOutput(
        is_primary_steel_dealer=True,
        switching_feasibility="MEDIUM",
        processing_capability="BASIC",
        has_physical_stockyard=True,
        gst_filing_regularity="UNKNOWN",
        evidence_confidence="NONE",
    )


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "agent" in result.output.lower() or "Usage" in result.output


def test_turnover_slab_unit_inheritance():
    # Bare "40" with upper "1.5 Cr" must be read as 40 Lakhs, not 40 Cr.
    assert parse_turnover_slab("Rs. 40 to Rs. 1.5 Cr") == 0.95
    assert parse_turnover_slab("Rs. 40 lakhs to Rs. 1.5 Cr") == 0.95
    # Sanity: explicit Cr on both sides is unchanged.
    assert parse_turnover_slab("Rs. 1.5 Cr to Rs. 5 Cr") == 3.25
    # Real-data slab formats (Clear GST export with trailing periods).
    assert parse_turnover_slab("Slab: Rs. 100 Cr. to 500 Cr.") == 300.0
    assert parse_turnover_slab("Slab: Rs. 10 Cr. to 25 Cr.") == 17.5
    assert parse_turnover_slab("Slab: Rs. 500 Cr. and above") == 750.0


def test_offline_geocoding_831001():
    cand = DistributorCandidate(
        gstin="20ABCDE1234F1Z5",
        pan="ABCDE1234F",
        legal_name="Test Steel",
        state_code="20",
        state_name="Jharkhand",
        pincode="831001",
    )
    out = geocode_candidates([cand], use_nominatim=False)
    assert out[0].geocode_source == GeoSource.PINCODE
    assert out[0].geocode_source == "PINCODE"  # str-Enum compat
    exp_lat, exp_lon = OFFLINE_PINCODE_COORDS["831001"]
    assert out[0].latitude == exp_lat
    assert out[0].longitude == exp_lon

    # Other steel-hub pincodes resolve offline too.
    for pin in ("755026", "410208", "421302"):
        c = DistributorCandidate(
            gstin="20ABCDE1234F1Z5",
            pan="ABCDE1234F",
            legal_name="X",
            state_code="20",
            pincode=pin,
        )
        res = geocode_candidates([c], use_nominatim=False)
        assert res[0].geocode_source == GeoSource.PINCODE, pin


def test_turnover_score_continuity():
    cfg = PipelineConfig()

    def _turnover_score(turnover: float) -> float:
        cand = DistributorCandidate(
            gstin="20ABCDE1234F1Z5",
            pan="ABCDE1234F",
            legal_name="T",
            turnover_midpoint_cr=turnover,
            has_steel_hsn=True,
        )
        ev = score_candidate(
            candidate=cand,
            driving_distance_km=50.0,
            distance_source="COARSE",
            s2t=_s2t_default(),
            config=cfg,
        )
        return ev.scores.s_turnover

    # No cliff at the old 14.9 -> 15.0 boundary.
    assert abs(_turnover_score(15.0) - _turnover_score(14.9)) < 5.0
    # Ramp anchors: 0 -> 0, 15 -> 25, 50 -> 100.
    assert _turnover_score(0.0) == 0.0
    assert _turnover_score(15.0) == 25.0
    assert _turnover_score(50.0) == 100.0


def test_pipeline_happy_path(tmp_path: Path):
    csv_path = tmp_path / "distributors.csv"
    csv_path.write_text(
        "GSTIN,PAN,LEGAL NAME,TRADE NAME,REGISTRATION STATUS,"
        "NATURE OF CORE BUSINESS ACTIVITIES,AGGRE TURNOVER,"
        "PERMANENT ADDRESS,HSN CODE 1,DESCRIPTION OF GOODS 1,"
        "NATURE OF BUSINESS ACTIVITY 1\n"
        '20ABCDE1234F1Z5,ABCDE1234F,Test Steel Pvt Ltd,Test Steel,Active,MFT,'
        '"Rs. 25 Cr to Rs. 100 Cr","Plot 1, Sakchi, Jamshedpur 831001",'
        '7214,Iron bars,MFT\n',
        encoding="utf-8",
    )
    cfg = PipelineConfig()
    cfg.enrichment.top_n_for_enrichment = 10
    depot = TataDepot(name="Jamshedpur", latitude=22.80, longitude=86.20)
    evaluations = run_pipeline(
        distributor_csv=csv_path,
        tata_depot=depot,
        config=cfg,
        skip_enrichment=True,
        skip_llm=True,
    )
    assert len(evaluations) == 1
    ev = evaluations[0]
    assert ev.gstin == "20ABCDE1234F1Z5"
    assert ev.enrichment_source == EnrichmentSource.NONE
    # Offline 831001 geocoding keeps the depot distance small.
    assert ev.driving_distance_km < 30.0


def test_actual_csv_headers():
    # Actual-data headers: EINVOICESTATUS (no space), EKYC_STATUS, AADHAR VERIFIED.
    row = pd.Series(
        {
            "GSTIN": "20ABCDE1234F1Z5",
            "EINVOICESTATUS": "Y",
            "EKYC_STATUS": "Yes",
            "AADHAR VERIFIED": "Yes",
        }
    )
    cand = parse_distributor_row(row)
    assert cand is not None
    assert cand.einvoice_enabled is True
    assert cand.ekyc_status == "Yes"
    assert cand.aadhaar_verified == "Yes"

    # Legacy legend header still works as fallback.
    legacy = pd.Series({"GSTIN": "20ABCDE1234F1Z5", "EINVOICE STATUS": "Y"})
    assert parse_distributor_row(legacy).einvoice_enabled is True

    # Passthrough to FinalEvaluation.
    cfg = PipelineConfig()
    ev = score_candidate(
        candidate=cand,
        driving_distance_km=50.0,
        distance_source="COARSE",
        s2t=_s2t_default(),
        config=cfg,
    )
    assert ev.ekyc_status == "Yes"
    assert ev.aadhaar_verified == "Yes"


def test_firecrawl_payload_and_parse(tmp_path: Path):
    """Live-path boundary test: Firecrawl request shape + response parsing.

    HTTP layer is stubbed (no network, no cost); everything else is real code.
    """
    import agent_0.enrichment as enr

    captured: dict = {}
    big_md = "# Steel Co\n" + "yard " * 3000  # >5000 chars to prove truncation

    def fake_post(url, headers, payload, timeout_seconds=20.0):
        captured["url"] = url
        captured["payload"] = payload
        captured["auth"] = headers.get("Authorization", "")
        return {"data": [{"markdown": big_md}, {"description": "short desc"}]}

    with patch.object(enr, "_post_firecrawl_http", side_effect=fake_post):
        text, source = enr.enrich_entity(
            "Test Steel", "20ABCDE1234F1Z5", "Jharkhand",
            firecrawl_api_key="test-key", cache_dir=str(tmp_path),
        )
    assert captured["url"] == "https://api.firecrawl.dev/v1/search"
    assert "Test Steel" in captured["payload"]["query"]
    assert "20ABCDE1234F1Z5" in captured["payload"]["query"]
    assert "Jharkhand" in captured["payload"]["query"]
    assert captured["payload"]["limit"] == 5
    assert captured["auth"] == "Bearer test-key"
    assert source == "FIRECRAWL"
    assert len(text) <= 5000  # truncation enforced
    assert (tmp_path / "20ABCDE1234F1Z5.json").is_file()  # cache written


def test_s2t_schema_accepts_llm_shaped_output():
    """An LLM-shaped tool payload must validate into S2TVerificationOutput."""
    from agent_0.models import S2TVerificationOutput

    tool_input = {
        "is_primary_steel_dealer": True,
        "brands_currently_handled": ["SAIL", "JSW"],
        "is_exclusive_to_competitor": False,
        "switching_feasibility": "HIGH",
        "has_physical_stockyard": True,
        "processing_capability": "ADVANCED",
        "promoter_litigation_flag": False,
        "gst_filing_regularity": "REGULAR",
        "evidence_confidence": "STRONG",
        "verbatim_evidence": "Source X: '...stockyard...' (url)",
        "falsification_notes": "Name collision checked; no match found.",
    }
    out = S2TVerificationOutput.model_validate(tool_input)
    assert out.confidence_score == 0.90
    assert out.processing_capability == "ADVANCED"


def test_grade_lift_with_strong_evidence(tmp_path: Path):
    """Checklist item 1: STRONG evidence must lift the top candidate to Grade A.

    LLM + Firecrawl are stubbed at the boundary (no network, no cost);
    grading math, wiring, and CSV export are all real code.
    """
    import agent_0.pipeline as pipe
    from agent_0.cli import _evaluations_to_df

    csv_path = tmp_path / "probe.csv"
    csv_path.write_text(
        "GSTIN,PAN,LEGAL NAME,TRADE NAME,REGISTRATION STATUS,"
        "NATURE OF CORE BUSINESS ACTIVITIES,AGGRE TURNOVER,"
        "PERMANENT ADDRESS,HSN CODE 1,EINVOICESTATUS,EKYC_STATUS,AADHAR VERIFIED\n"
        '07ADQPA9410K1ZE,ADQPA9410K,Probe Steel,Probe Steel,Active,MFT,'
        '"Slab: Rs. 25 Cr. to 100 Cr.","Plot 1, Delhi 110020",7214,Y,Yes,Yes\n',
        encoding="utf-8",
    )
    strong = S2TVerificationOutput(
        is_primary_steel_dealer=True,
        brands_currently_handled=["SAIL"],
        is_exclusive_to_competitor=False,
        switching_feasibility="HIGH",
        has_physical_stockyard=True,
        processing_capability="ADVANCED",
        promoter_litigation_flag=False,
        gst_filing_regularity="REGULAR",
        evidence_confidence="STRONG",
        verbatim_evidence="Firecrawl doc: 'Probe Steel operates a 2-acre stockyard' (example.com)",
        falsification_notes="Checked name collision vs Probe Steels Pvt Ltd; distinct GSTIN.",
    )
    seen: dict = {}

    def fake_llm(candidate, scraped_content, config, meta=None):
        seen["scraped"] = scraped_content
        if meta is not None:
            meta["provider"] = "anthropic"
            meta["fallback"] = False
        return strong

    def fake_enrich(candidates, firecrawl_api_key, cache_dir, cache_ttl_days=30):
        assert firecrawl_api_key == "test-key"
        return {c.gstin: ("canned web text about stockyard", "FIRECRAWL") for c in candidates}

    cfg = PipelineConfig()
    cfg.firecrawl_api_key = "test-key"
    cfg.enrichment.cache_dir = str(tmp_path / "cache")
    cfg.enrichment.top_n_for_enrichment = 1
    with patch.object(pipe, "run_tot_s2t", side_effect=fake_llm), patch.object(
        pipe, "batch_enrich", side_effect=fake_enrich
    ):
        evs = run_pipeline(
            distributor_csv=csv_path,
            tata_depot=TataDepot(name="Tata Steel Delhi Hub", latitude=28.6139, longitude=77.2090),
            config=cfg,
            skip_enrichment=False,
            skip_llm=False,
        )
    assert len(evs) == 1
    ev = evs[0]
    assert seen["scraped"] == "canned web text about stockyard"  # enrichment -> LLM wired
    assert ev.enrichment_source == EnrichmentSource.FIRECRAWL
    assert ev.verification and ev.verification.evidence_confidence == "STRONG"
    assert ev.final_grade == "A"  # the grade lift
    df = _evaluations_to_df(evs)
    assert df.loc[0, "Verbatim Evidence"].startswith("Firecrawl doc:")
    assert "collision" in df.loc[0, "Falsification Notes"]


def _s2t_tool_input() -> dict:
    return {
        "is_primary_steel_dealer": True,
        "brands_currently_handled": ["SAIL"],
        "is_exclusive_to_competitor": False,
        "switching_feasibility": "HIGH",
        "has_physical_stockyard": True,
        "processing_capability": "ADVANCED",
        "promoter_litigation_flag": False,
        "gst_filing_regularity": "REGULAR",
        "evidence_confidence": "STRONG",
        "verbatim_evidence": "doc",
        "falsification_notes": "notes",
    }


def test_claude_workspace_header_sent():
    """Enterprise keys need anthropic-workspace-id header; assert it is sent."""
    from types import SimpleNamespace

    import agent_0.agent as ag

    captured: dict = {}

    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="emit_verification",
                        input=_s2t_tool_input(),
                    )
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.messages = FakeMessages()

    cfg = PipelineConfig()
    cfg.anthropic_api_key = "k"
    cfg.anthropic_workspace_id = "ws-123"
    cand = DistributorCandidate(
        gstin="20ABCDE1234F1Z5", pan="ABCDE1234F", legal_name="X"
    )
    with patch("agent_0.agent.anthropic.Anthropic", FakeClient):
        out = ag._run_claude(cand, "text", cfg)
    assert captured["default_headers"] == {"anthropic-workspace-id": "ws-123"}
    assert out.evidence_confidence == "STRONG"


def test_claude_no_workspace_header_by_default():
    """Without a workspace id, no extra header is sent (legacy behavior)."""
    from types import SimpleNamespace

    import agent_0.agent as ag

    captured: dict = {}

    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="emit_verification",
                        input=_s2t_tool_input(),
                    )
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.messages = FakeMessages()

    cfg = PipelineConfig()
    cfg.anthropic_api_key = "k"
    cand = DistributorCandidate(
        gstin="20ABCDE1234F1Z5", pan="ABCDE1234F", legal_name="X"
    )
    with patch("agent_0.agent.anthropic.Anthropic", FakeClient):
        ag._run_claude(cand, "text", cfg)
    assert captured["default_headers"] == {}


def test_haryana_prefixes_resolve_offline():
    """All shortlist Haryana prefixes must hit PINCODE, not STATE_CENTROID."""
    from agent_0.geocoder import PIN_PREFIX_3_COORDS

    for prefix in ("121", "122", "123", "131", "132", "134"):
        assert prefix in PIN_PREFIX_3_COORDS, prefix
        cand = DistributorCandidate(
            gstin="06AAAA0000A1Z5",
            pan="AAAA0000A",
            legal_name="H",
            state_code="06",
            pincode=prefix + "001",
        )
        out = geocode_candidates([cand], use_nominatim=False)
        assert out[0].geocode_source == GeoSource.PINCODE, prefix


def _gemini_fake_client(captured: dict, content: str):
    from types import SimpleNamespace

    class FakeCompletions:
        def create(self, **kwargs):
            captured["create_kwargs"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = FakeChat()

    return FakeClient


def test_gemini_client_construction_and_parse():
    """Gemini must use its OpenAI-compat endpoint + default model + strict schema."""
    import json

    import agent_0.agent as ag

    captured: dict = {}
    payload = dict(_s2t_tool_input())
    payload["evidence_confidence"] = "MODERATE"
    with patch(
        "agent_0.agent.openai.OpenAI",
        _gemini_fake_client(captured, json.dumps(payload)),
    ):
        cfg = PipelineConfig()
        cfg.gemini_api_key = "gk"
        # Default config model is Claude-based -> must fall back to Gemini default.
        assert "gemini" not in cfg.llm.model.lower()
        cand = DistributorCandidate(
            gstin="20ABCDE1234F1Z5", pan="ABCDE1234F", legal_name="X"
        )
        out = ag._run_gemini(cand, "web text", cfg)
    assert captured["client_kwargs"]["base_url"] == ag.GEMINI_BASE_URL
    assert captured["client_kwargs"]["api_key"] == "gk"
    assert captured["client_kwargs"]["timeout"] == 120.0
    assert captured["create_kwargs"]["model"] == ag.GEMINI_DEFAULT_MODEL
    assert captured["create_kwargs"]["response_format"]["json_schema"]["strict"] is True
    assert out.evidence_confidence == "MODERATE"
    assert out.confidence_score == 0.65


def test_gemini_respects_configured_model():
    """An explicit gemini-* model in config must be used as-is."""
    import json

    import agent_0.agent as ag

    captured: dict = {}
    with patch(
        "agent_0.agent.openai.OpenAI",
        _gemini_fake_client(captured, json.dumps(_s2t_tool_input())),
    ):
        cfg = PipelineConfig()
        cfg.gemini_api_key = "gk"
        cfg.llm.model = "gemini-1.5-flash"
        cand = DistributorCandidate(
            gstin="20ABCDE1234F1Z5", pan="ABCDE1234F", legal_name="X"
        )
        ag._run_gemini(cand, "web text", cfg)
    assert captured["create_kwargs"]["model"] == "gemini-1.5-flash"


def test_run_tot_s2t_dispatches_gemini(monkeypatch):
    """Provider 'gemini' must route to _run_gemini (mocked, no network)."""
    import agent_0.agent as ag

    sentinel = S2TVerificationOutput(**_s2t_tool_input())
    with patch.object(ag, "_run_gemini", return_value=sentinel) as m:
        cfg = PipelineConfig()
        cfg.llm.provider = "gemini"
        cand = DistributorCandidate(
            gstin="20ABCDE1234F1Z5", pan="ABCDE1234F", legal_name="X"
        )
        out = ag.run_tot_s2t(cand, "text", cfg)
    assert out is sentinel
    m.assert_called_once()


def test_load_config_gemini_keys(monkeypatch):
    """GEMINI_API_KEY wins; GOOGLE_API_KEY is the fallback.

    Both vars are set explicitly in each phase so the repo's real .env
    can never leak into the assertion (load_dotenv reads it otherwise).
    """
    from agent_0.config import load_config

    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "other-key")
    assert load_config().gemini_api_key == "gem-key"
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-key")
    assert load_config().gemini_api_key == "goog-key"


def _failed_eval() -> FinalEvaluation:
    from agent_0.models import S2TVerificationOutput
    return FinalEvaluation(
        gstin="20ABCDE1234F1Z5",
        pan="ABCDE1234F",
        legal_name="X",
        verification=S2TVerificationOutput(
            is_primary_steel_dealer=True,
            switching_feasibility="MEDIUM",
            processing_capability="NONE",
            has_physical_stockyard=False,
            evidence_confidence="NONE",
            verbatim_evidence="LLM call failed. Using safe defaults. (400)",
        ),
    )


def test_llm_failure_hint():
    from agent_0.pipeline import summarize_llm_outcome

    assert summarize_llm_outcome([], 0, False) is None
    assert summarize_llm_outcome([_failed_eval()], 1, True) is None  # skipped, not failed
    hint = summarize_llm_outcome([_failed_eval(), _failed_eval()], 2, False)
    assert hint is not None and "--llm" in hint
    good = _failed_eval()
    good.verification.verbatim_evidence = "Real grounded evidence."
    assert summarize_llm_outcome([_failed_eval(), good], 2, False) is None


def _probe_pin_rows():
    """Four gate-passing rows with comfortably separated coarse distances.

    Pincodes resolve via the offline table: 110xxx (Delhi, ~1km) <
    121xxx (Faridabad, ~25km) < 131xxx (Sonipat, ~46km) <
    123xxx (Rewari, ~74km) from the Delhi test depot.
    """
    return [
        ("07PIN1111111Z11", "P111111111", "110020"),
        ("07PIN2222222Z22", "P222222222", "121001"),
        ("07PIN3333333Z33", "P333333333", "131001"),
        ("07PIN4444444Z44", "P444444444", "123001"),
    ]


def _write_probe_csv(rows):
    import csv as csvmod
    import tempfile

    p = Path(tempfile.mkdtemp()) / "probe.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csvmod.writer(f)
        w.writerow(
            [
                "GSTIN",
                "PAN",
                "REGISTRATION STATUS",
                "NATURE OF CORE BUSINESS ACTIVITIES",
                "AGGRE TURNOVER",
                "PERMANENT ADDRESS",
                "HSN CODE 1",
            ]
        )
        for gstin, pan, pin in rows:
            w.writerow(
                [
                    gstin,
                    pan,
                    "Active",
                    "MFT",
                    "Rs. 25 Cr to Rs. 100 Cr",
                    f"Plot 1, Delhi {pin}",
                    "7214",
                ]
            )
    return p


def test_gmaps_refine_budget_split():
    """Only the N closest batch members get GMaps; rest stay coarse."""
    import agent_0.pipeline as pipe

    depot = TataDepot(name="D", latitude=28.6139, longitude=77.2090)
    calls: list = []

    def fake_gmaps(lat1, lon1, lat2, lon2, key):
        calls.append((lat1, lon1))
        return 42.0, "GMAPS"

    cfg = PipelineConfig()
    cfg.gmaps_api_key = "gk"
    cfg.enrichment.top_n_for_enrichment = 10
    cfg.enrichment.gmaps_refine_top = 2
    # run_pipeline directly (skip flags keep everything offline except mocked GMaps)
    from agent_0.pipeline import run_pipeline as rp

    with patch.object(pipe, "google_maps_distance", side_effect=fake_gmaps):
        evs = rp(
            distributor_csv=_write_probe_csv(_probe_pin_rows()),
            tata_depot=depot,
            config=cfg,
            skip_enrichment=True,
            skip_llm=True,
        )
    assert len(evs) == 4
    by_gstin = {e.gstin: e for e in evs}
    assert by_gstin["07PIN1111111Z11"].distance_source == "GMAPS"
    assert by_gstin["07PIN2222222Z22"].distance_source == "GMAPS"
    assert by_gstin["07PIN3333333Z33"].distance_source == "COARSE"
    assert by_gstin["07PIN4444444Z44"].distance_source == "COARSE"
    assert by_gstin["07PIN1111111Z11"].driving_distance_km == 42.0
    assert len(calls) == 2


def test_gmaps_refine_zero_disables():
    """gmaps_refine_top=0 must make zero Maps calls even with a key set."""
    import agent_0.pipeline as pipe
    from agent_0.pipeline import run_pipeline as rp

    depot = TataDepot(name="D", latitude=28.6139, longitude=77.2090)
    calls: list = []

    def fake_gmaps(*a, **k):
        calls.append(a)
        return 1.0, "GMAPS"

    cfg = PipelineConfig()
    cfg.gmaps_api_key = "gk"
    cfg.enrichment.gmaps_refine_top = 0
    with patch.object(pipe, "google_maps_distance", side_effect=fake_gmaps):
        evs = rp(
            distributor_csv=_write_probe_csv(_probe_pin_rows()[:1]),
            tata_depot=depot,
            config=cfg,
            skip_enrichment=True,
            skip_llm=True,
        )
    assert calls == []
    assert evs[0].distance_source == "COARSE"


# ────────────────────────────────────────────────────────
# Regression tests for the review findings
# ────────────────────────────────────────────────────────


def test_required_hsn_prefixes_config_is_honoured():
    """`gates.required_hsn_prefixes` must actually drive the steel HSN gate.

    Previously the prefix list was hardcoded in csv_loader, so changing it
    in config.yaml had no effect on the gate.
    """
    from agent_0.prefilter import _evaluate_candidate

    cand = DistributorCandidate(
        gstin="20ABCDE1234F1Z5",
        pan="ABCDE1234F",
        legal_name="X",
        hsn_codes=["7214"],  # chapter 72 — steel
        has_steel_hsn=True,
        registration_status="Active",
        latitude=22.80,
        longitude=86.20,
    )
    depot = TataDepot(name="Jamshedpur", latitude=22.80, longitude=86.20)

    cfg = PipelineConfig()
    cfg.gates.required_hsn_prefixes = ["72", "73"]
    assert _evaluate_candidate(cand, depot, [], cfg).passed

    # Requiring chapter 73 only must reject the chapter-72 candidate.
    cfg.gates.required_hsn_prefixes = ["73"]
    result = _evaluate_candidate(cand, depot, [], cfg)
    assert not result.passed
    assert any("Missing steel HSN" in r for r in result.rejection_reasons)


def test_rejection_breakdown_groups_by_gate_not_by_value():
    """Distance rejections must collapse into one row, not one per value."""
    from agent_0.prefilter import categorise_rejection

    assert categorise_rejection("Coarse road distance 2534.96 km exceeds maximum 180.0 km") == (
        "Distance exceeds maximum"
    )
    assert categorise_rejection("Coarse road distance 1208.22 km exceeds maximum 180.0 km") == (
        "Distance exceeds maximum"
    )
    assert categorise_rejection("Turnover ₹750.0Cr outside allowed range [15.0, 300.0]Cr") == (
        "Turnover out of range"
    )
    assert categorise_rejection("Registration status not active: 'Cancelled'") == (
        "Registration not active"
    )


def test_config_rejects_unknown_keys():
    """A typo in config.yaml must fail loudly instead of being ignored."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        PipelineConfig(gates={"min_turnover_c": 5})
    with pytest.raises(pydantic.ValidationError):
        PipelineConfig(typo_key=1)


def test_invalid_format_is_rejected():
    """An unknown --format must fail fast, before any pipeline work is done."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--distributor-csv",
            str(Path("data/distributors.csv")),
            "--depot-name",
            "D",
            "--depot-lat",
            "28.61",
            "--depot-lon",
            "77.20",
            "--skip-llm",
            "--skip-enrichment",
            "--format",
            "excel",
        ],
    )
    assert result.exit_code == 2
    assert "excel" in result.output
    # Must fail before the pipeline runs (no candidate summary produced).
    assert "Pipeline Complete" not in result.output


def test_llm_provider_falls_back_when_primary_fails():
    """A failing primary provider must fall through to a keyed provider."""
    import agent_0.agent as ag

    sentinel = S2TVerificationOutput(**_s2t_tool_input())

    cfg = PipelineConfig()
    cfg.llm.provider = "anthropic"
    cfg.anthropic_api_key = "primary-key"
    cfg.openai_api_key = "fallback-key"
    cand = DistributorCandidate(
        gstin="20ABCDE1234F1Z5", pan="ABCDE1234F", legal_name="X"
    )

    def boom(*a, **k):
        raise RuntimeError("quota exceeded")

    meta: dict = {}
    with patch.object(ag, "_run_claude", side_effect=boom), patch.object(
        ag, "_run_openai", return_value=sentinel
    ):
        out = ag.run_tot_s2t(cand, "text", cfg, meta=meta)

    assert out is sentinel
    assert meta["provider"] == "openai"
    assert meta["fallback"] is True


def test_llm_all_providers_failing_returns_safe_default():
    """When every provider fails, the safe default is returned — not a crash."""
    import agent_0.agent as ag

    cfg = PipelineConfig()
    cfg.llm.provider = "anthropic"
    cfg.anthropic_api_key = "k"
    cfg.openai_api_key = "k2"
    cand = DistributorCandidate(
        gstin="20ABCDE1234F1Z5", pan="ABCDE1234F", legal_name="X"
    )

    def boom(*a, **k):
        raise RuntimeError("nope")

    with patch.object(ag, "_run_claude", side_effect=boom), patch.object(
        ag, "_run_openai", side_effect=boom
    ):
        out = ag.run_tot_s2t(cand, "text", cfg)

    assert out.evidence_confidence == "NONE"
    assert out.verbatim_evidence.startswith("LLM call failed")


def test_no_evidence_payloads_are_not_cached_and_not_sent_to_llm():
    """Placeholder enrichment must not be cached nor passed to the LLM."""
    import agent_0.enrichment as enr

    # 1. A Firecrawl response with no extractable text is NOT cached.
    def fake_post(url, headers, payload, timeout_seconds=20.0):
        return {"data": []}

    with tempfile.TemporaryDirectory() as td:
        with patch.object(enr, "_post_firecrawl_http", side_effect=fake_post):
            text, source = enr.enrich_entity(
                "Test Steel", "20ABCDE1234F1Z5", "Jharkhand",
                firecrawl_api_key="k", cache_dir=td,
            )
        assert source == "MOCK"
        assert not (Path(td) / "20ABCDE1234F1Z5.json").exists()
        assert enr.is_no_evidence(text)

    # 2. The pipeline strips such payloads before calling the LLM.
    import agent_0.pipeline as pipe

    seen: dict = {}

    def fake_llm(candidate, scraped_content, config, meta=None):
        seen["scraped"] = scraped_content
        return S2TVerificationOutput(**_s2t_tool_input())

    def fake_enrich(candidates, firecrawl_api_key, cache_dir, cache_ttl_days=30):
        return {c.gstin: ("[MOCK] Web search failed", "MOCK") for c in candidates}

    cfg = PipelineConfig()
    cfg.firecrawl_api_key = "k"
    cfg.enrichment.top_n_for_enrichment = 1
    with patch.object(pipe, "run_tot_s2t", side_effect=fake_llm), patch.object(
        pipe, "batch_enrich", side_effect=fake_enrich
    ):
        pipe.run_pipeline(
            distributor_csv=_write_probe_csv(_probe_pin_rows()[:1]),
            tata_depot=TataDepot(name="D", latitude=28.6139, longitude=77.2090),
            config=cfg,
            skip_enrichment=False,
            skip_llm=False,
        )
    assert seen["scraped"] == ""


def test_enrichment_cache_expires():
    """Cache entries older than the TTL are refetched."""
    import json as jsonmod

    import agent_0.enrichment as enr

    old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    with tempfile.TemporaryDirectory() as td:
        cache_file = Path(td) / "20ABCDE1234F1Z5.json"
        cache_file.write_text(
            jsonmod.dumps(
                {
                    "gstin": "20ABCDE1234F1Z5",
                    "text": "real evidence about the company",
                    "timestamp": old,
                    "source": "FIRECRAWL",
                }
            )
        )
        assert not enr._cache_entry_is_fresh(
            jsonmod.loads(cache_file.read_text()), ttl_days=30
        )
        assert enr._cache_entry_is_fresh(
            jsonmod.loads(cache_file.read_text()), ttl_days=365
        )


def test_geocode_source_is_exported():
    """The output must expose where each coordinate came from."""
    import agent_0.pipeline as pipe
    from agent_0.cli import _evaluations_to_df

    cfg = PipelineConfig()
    cfg.enrichment.top_n_for_enrichment = 5
    evs = pipe.run_pipeline(
        distributor_csv=_write_probe_csv(_probe_pin_rows()[:1]),
        tata_depot=TataDepot(name="D", latitude=28.6139, longitude=77.2090),
        config=cfg,
        skip_enrichment=True,
        skip_llm=True,
    )
    df = _evaluations_to_df(evs)
    assert "Geocode Source" in df.columns
    assert df.loc[0, "Geocode Source"] == "PINCODE"
    assert df.loc[0, "Distance Source"] == "COARSE"


def test_unverified_none_capped_at_c():
    """R1: a NONE-confidence candidate must not grade above C."""
    cfg = PipelineConfig()
    cand = DistributorCandidate(
        gstin="20ABCDE1234F1Z5",
        pan="ABCDE1234F",
        legal_name="T",
        turnover_midpoint_cr=62.5,
        has_steel_hsn=True,
    )
    from agent_0.pipeline import _safe_s2t_default

    ev = score_candidate(
        candidate=cand,
        driving_distance_km=30.0,
        distance_source="COARSE",
        s2t=_safe_s2t_default(),
        config=cfg,
    )
    assert ev.final_grade == "C"
    assert ev.verification_status == "UNVERIFIED"
    assert any("capped at C" in r for r in ev.rejection_reasons)
    assert ev.scores.ease_score == 54.5


def test_failed_llm_status_and_cap():
    """All-providers-failed must be FAILED and capped at C."""
    cfg = PipelineConfig()
    cand = DistributorCandidate(
        gstin="20ABCDE1234F1Z5",
        pan="ABCDE1234F",
        legal_name="T",
        turnover_midpoint_cr=62.5,
    )
    from agent_0.agent import _get_safe_default_output

    ev = score_candidate(
        candidate=cand,
        driving_distance_km=30.0,
        distance_source="COARSE",
        s2t=_get_safe_default_output("anthropic: quota"),
        config=cfg,
    )
    assert ev.final_grade == "C"
    assert ev.verification_status == "FAILED"


def test_strong_evidence_can_still_reach_a():
    """R1 cap must not block verified evidence from reaching A."""
    cfg = PipelineConfig()
    cand = DistributorCandidate(
        gstin="20ABCDE1234F1Z5",
        pan="ABCDE1234F",
        legal_name="T",
        turnover_midpoint_cr=62.5,
    )
    ev = score_candidate(
        candidate=cand,
        driving_distance_km=30.0,
        distance_source="COARSE",
        s2t=S2TVerificationOutput(**_s2t_tool_input()),
        config=cfg,
        enrichment_source=EnrichmentSource.FIRECRAWL,
    )
    assert ev.final_grade == "A"
    assert ev.verification_status == "VERIFIED"
