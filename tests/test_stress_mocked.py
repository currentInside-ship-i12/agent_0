"""Extreme mocked stress tests — zero network, zero API cost.

Covers scale, adversarial CSV, boundaries, chaos, cost governors,
determinism and export integrity. Run: pytest tests/stress_mocked.py -v
"""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from agent_0.cli import _evaluations_to_df, app
from agent_0.config import PipelineConfig
from agent_0.models import DistributorCandidate, S2TVerificationOutput, TataDepot
from agent_0.pipeline import _safe_s2t_default, run_pipeline
from agent_0.scoring import score_candidate

DELHI = TataDepot(name="Delhi Hub", latitude=28.6139, longitude=77.2090)
HEADER = [
    "GSTIN", "PAN", "LEGAL NAME", "TRADE NAME", "REGISTRATION STATUS",
    "NATURE OF CORE BUSINESS ACTIVITIES", "AGGRE TURNOVER",
    "PERMANENT ADDRESS", "HSN CODE 1",
]


def _write_csv(rows):
    p = Path(tempfile.mkdtemp()) / "s.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    return p


def _row(i, pin="110020", turnover="Rs. 25 Cr to Rs. 100 Cr", hsn="7214",
         core="MFT", legal=None, gstin=None):
    gstin = gstin or f"07ABCDE{i:04d}F1Z{i % 10}"
    return [gstin, f"PAN{i:07d}", legal or f"Steel {i}", f"Steel {i}", "Active",
            core, turnover, f"Plot {i}, Delhi {pin}", hsn]


def _strong():
    return S2TVerificationOutput(
        is_primary_steel_dealer=True, brands_currently_handled=["SAIL"],
        switching_feasibility="HIGH", has_physical_stockyard=True,
        processing_capability="ADVANCED", gst_filing_regularity="REGULAR",
        evidence_confidence="STRONG", verbatim_evidence="yard photo (x.com)",
        falsification_notes="checked",
    )


# Scale
def test_scale_1500_rows_offline():
    rows = [_row(i, pin=["110020", "121001", "131001", "123001"][i % 4]) for i in range(1500)]
    cfg = PipelineConfig()
    cfg.enrichment.top_n_for_enrichment = 50
    evs = run_pipeline(distributor_csv=_write_csv(rows), tata_depot=DELHI, config=cfg,
                       skip_enrichment=True, skip_llm=True)
    assert len(evs) == 1500
    assert all(e.final_grade in ("C", "D") for e in evs)  # R1 cap
    assert all(e.verification_status == "UNVERIFIED" for e in evs)


def test_cannibalization_many_existing():
    rows = [_row(1)]
    existing = Path(tempfile.mkdtemp()) / "tata.csv"
    with open(existing, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i in range(500):
            w.writerow(_row(1000 + i, pin="110020",
                            turnover="Rs. 25 Cr to Rs. 100 Cr"))
    cfg = PipelineConfig()
    evs = run_pipeline(distributor_csv=_write_csv(rows), tata_depot=DELHI, config=cfg,
                       existing_tata_csv=existing, skip_enrichment=True, skip_llm=True)
    assert evs == []


# Boundaries / adversarial
@pytest.mark.parametrize("turnover,exp_s", [
    (0.0, 0.0), (14.9, 24.83), (15.0, 25.0), (50.0, 100.0), (150.0, 100.0),
    (300.0, 50.0), (300.01, 0.0), (750.0, 0.0),
])
def test_turnover_score_boundaries(turnover, exp_s):
    cand = DistributorCandidate(gstin="20ABCDE1234F1Z5", pan="ABCDE1234F",
                                legal_name="T", turnover_midpoint_cr=turnover)
    ev = score_candidate(candidate=cand, driving_distance_km=50.0,
                         distance_source="COARSE", s2t=_strong(), config=PipelineConfig())
    assert abs(ev.scores.s_turnover - exp_s) < 0.5


def test_adversarial_rows_rejected_not_crash():
    rows = [
        _row(1, gstin="SHORT", turnover="Rs. 1 Cr to Rs. 2 Cr"),
        _row(2, turnover="Rs. 500 Cr. and above", hsn="7214"),
        _row(3, turnover="", hsn=""),
        _row(4, core="SPO"),
        _row(5, legal="=HYPERLINK(\"evil\")"),
        _row(6, legal="Unicode élan ⚙️ yard"),
    ]
    cfg = PipelineConfig()
    evs = run_pipeline(distributor_csv=_write_csv(rows), tata_depot=DELHI, config=cfg,
                       skip_enrichment=True, skip_llm=True)
    assert isinstance(evs, list)  # no crash


def test_malformed_csv_and_missing_file():
    from agent_0.csv_loader import load_distributor_csv
    bad = Path(tempfile.mkdtemp()) / "bad.csv"
    bad.write_text('GSTIN,LEGAL NAME\n"unclosed,quote\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_distributor_csv(bad)
    with pytest.raises(FileNotFoundError):
        load_distributor_csv(Path("/tmp/does-not-exist-12345.csv"))


# Chaos: enrichment / maps / LLM
def test_firecrawl_chaos_all_degrade_to_mock():
    import agent_0.enrichment as enr
    for side in [RuntimeError("429"), RuntimeError("500"), ValueError("bad json")]:
        with patch.object(enr, "_post_firecrawl_http", side_effect=side):
            text, src = enr.enrich_entity("X", "20ABCDE1234F1Z5", "Jharkhand",
                                          firecrawl_api_key="k", cache_dir=tempfile.mkdtemp())
        assert src == "MOCK"
        assert enr.is_no_evidence(text)


def test_firecrawl_giant_and_empty():
    import agent_0.enrichment as enr
    with patch.object(enr, "_post_firecrawl_http",
                      return_value={"data": [{"markdown": "x" * 50000}]}):
        text, src = enr.enrich_entity("X", "20ABCDE1234F1Z5", "Jharkhand",
                                      firecrawl_api_key="k", cache_dir=tempfile.mkdtemp())
    assert src == "FIRECRAWL" and len(text) <= 5000
    with patch.object(enr, "_post_firecrawl_http", return_value={"data": []}):
        text, src = enr.enrich_entity("X", "20ABCDE1234F1Z5", "Jharkhand",
                                      firecrawl_api_key="k", cache_dir=tempfile.mkdtemp())
    assert src == "MOCK"


def test_corrupt_cache_refetched_not_crash():
    import agent_0.enrichment as enr
    td = tempfile.mkdtemp()
    Path(td, "20ABCDE1234F1Z5.json").write_text("{not json", encoding="utf-8")
    with patch.object(enr, "_post_firecrawl_http",
                      return_value={"data": [{"markdown": "real text"}]}):
        text, src = enr.enrich_entity("X", "20ABCDE1234F1Z5", "S", firecrawl_api_key="k",
                                      cache_dir=td)
    assert src == "FIRECRAWL" and "real text" in text


def test_gmaps_denied_falls_back_coarse():
    import agent_0.spatial as sp
    with patch.object(sp.requests, "get") as m:
        m.return_value.json.return_value = {"status": "REQUEST_DENIED"}
        m.return_value.raise_for_status.return_value = None
        dist, src = sp.google_maps_distance(28.61, 77.20, 28.62, 77.21, "k")
    assert src == "COARSE" and dist > 0


def test_llm_all_providers_fail_no_crash():
    import agent_0.agent as ag
    cfg = PipelineConfig()
    cfg.llm.provider = "anthropic"
    cfg.anthropic_api_key = "k"
    cfg.openai_api_key = "k2"
    cand = DistributorCandidate(gstin="20ABCDE1234F1Z5", pan="ABCDE1234F", legal_name="X")
    def boom(*a, **k):
        raise RuntimeError("down")
    with patch.object(ag, "_run_claude", side_effect=boom), \
         patch.object(ag, "_run_openai", side_effect=boom):
        out = ag.run_tot_s2t(cand, "text", cfg)
    assert out.evidence_confidence == "NONE"


# Cost governors
def test_gmaps_budget_respected():
    import agent_0.pipeline as pipe
    rows = [_row(1, pin="110020"), _row(2, pin="121001"), _row(3, pin="131001")]
    cfg = PipelineConfig()
    cfg.gmaps_api_key = "gk"
    cfg.enrichment.gmaps_refine_top = 1
    calls = []
    def fake(*a, **k):
        calls.append(a)
        return 42.0, "GMAPS"
    with patch.object(pipe, "google_maps_distance", side_effect=fake):
        evs = pipe.run_pipeline(distributor_csv=_write_csv(rows), tata_depot=DELHI,
                                config=cfg, skip_enrichment=True, skip_llm=True)
    assert len(calls) == 1
    assert sum(1 for e in evs if e.distance_source == "GMAPS") == 1


# Determinism + export
def test_offline_determinism_twice():
    rows = [_row(1), _row(2, pin="121001")]
    cfg = PipelineConfig()
    a = run_pipeline(distributor_csv=_write_csv(rows), tata_depot=DELHI, config=cfg,
                     skip_enrichment=True, skip_llm=True)
    b = run_pipeline(distributor_csv=_write_csv(rows), tata_depot=DELHI, config=cfg,
                     skip_enrichment=True, skip_llm=True)
    assert [e.gstin for e in a] == [e.gstin for e in b]
    assert [e.final_grade for e in a] == [e.final_grade for e in b]
    assert [e.scores.profit_score for e in a] == [e.scores.profit_score for e in b]


def test_export_schema_and_status_column():
    rows = [_row(1)]
    cfg = PipelineConfig()
    evs = run_pipeline(distributor_csv=_write_csv(rows), tata_depot=DELHI, config=cfg,
                       skip_enrichment=True, skip_llm=True)
    df = _evaluations_to_df(evs)
    for col in ["Grade", "Profit Score", "Distance Source", "Geocode Source",
                "Verification Status", "Confidence"]:
        assert col in df.columns
    assert df.loc[0, "Verification Status"] == "UNVERIFIED"
    data = json.loads(json.dumps([e.model_dump(mode="json") for e in evs]))
    assert data[0]["verification_status"] == "UNVERIFIED"


def test_cli_invalid_format_fails_before_pipeline():
    runner = CliRunner()
    r = runner.invoke(app, ["run", "--distributor-csv", "data/distributors.csv",
                            "--depot-name", "D", "--depot-lat", "28.61",
                            "--depot-lon", "77.20", "--skip-llm",
                            "--skip-enrichment", "--format", "excel"])
    assert r.exit_code == 2
    assert "Pipeline Complete" not in r.output


def test_safe_default_is_capped_not_b():
    cand = DistributorCandidate(gstin="20ABCDE1234F1Z5", pan="ABCDE1234F",
                                legal_name="T", turnover_midpoint_cr=62.5)
    ev = score_candidate(candidate=cand, driving_distance_km=30.0,
                         distance_source="COARSE", s2t=_safe_s2t_default(),
                         config=PipelineConfig())
    assert ev.final_grade == "C" and ev.scores.ease_score == 54.5
