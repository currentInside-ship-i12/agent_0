"""Typer CLI for Agent 0 distributor discovery and evidence queues."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import typer
from pydantic import ValidationError
from rich.console import Console

from agent_0.config import load_config
from agent_0.models import (
    CandidateAuditRecord,
    FinalEvaluation,
    PipelineRunResult,
    RecommendationState,
    TataDepot,
)
from agent_0.pipeline import run_pipeline

app = typer.Typer(
    name="agent0",
    help="Agent 0 — Tata Steel GTM distributor discovery and verification",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _configure_logging(verbose: bool = False) -> None:
    """Send diagnostic logs to stderr without corrupting exported artifacts."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _spreadsheet_safe(value: Any) -> Any:
    """Prevent spreadsheet applications from evaluating source text as formulas."""
    if not isinstance(value, str):
        return value
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def _safe_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build an export dataframe and neutralize spreadsheet formula cells."""
    dataframe = pd.DataFrame(rows)
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].map(_spreadsheet_safe)
    return dataframe


def _evaluations_to_df(evaluations: list[FinalEvaluation]) -> pd.DataFrame:
    """Flatten scored outcomes into a spreadsheet-safe dataframe."""
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        verification = evaluation.verification
        row = {
            "Queue": evaluation.recommendation_state.value,
            "Action": evaluation.action,
            "Grade": evaluation.final_grade,
            "GSTIN": evaluation.gstin,
            "PAN": evaluation.pan,
            "Legal Name": evaluation.legal_name,
            "Trade Name": evaluation.trade_name or "",
            "State": evaluation.state,
            "Turnover Slab": evaluation.turnover_slab or "",
            "Distance (km)": round(evaluation.driving_distance_km, 1),
            "Distance Source": evaluation.distance_source,
            "Geocode Source": evaluation.geocode_source.value,
            "Profit Score": round(evaluation.scores.profit_score, 2),
            "Ease Score": round(evaluation.scores.ease_score, 2),
            "S_Logistics": round(evaluation.scores.s_logistics, 2),
            "S_Turnover": round(evaluation.scores.s_turnover, 2),
            "S_Processing": round(evaluation.scores.s_processing, 2),
            "E_Compliance": round(evaluation.scores.e_compliance, 2),
            "E_Switching": round(evaluation.scores.e_switching, 2),
            "E_Infra": round(evaluation.scores.e_infra, 2),
            "Enrichment Source": evaluation.enrichment_source.value,
            "Verification Provider": evaluation.verification_provider or "",
            "Verification Model": evaluation.verification_model or "",
            "Verification Status": evaluation.verification_status,
            "Evidence Sources": evaluation.evidence_validation.source_count,
            "Valid Citations": evaluation.evidence_validation.valid_citation_count,
            "Citation Validated": evaluation.evidence_validation.verified,
            "Citation Errors": " | ".join(evaluation.evidence_validation.invalid_citations),
            "Address": evaluation.permanent_address or "",
            "eKYC Status": evaluation.ekyc_status or "",
            "Aadhaar Verified": evaluation.aadhaar_verified or "",
            "Rejection Reasons": " | ".join(evaluation.rejection_reasons),
        }
        if verification:
            row.update(
                {
                    "Is Steel Dealer": verification.is_primary_steel_dealer,
                    "Brands Handled": ", ".join(verification.brands_currently_handled),
                    "Exclusive To Competitor": verification.is_exclusive_to_competitor,
                    "Switching Feasibility": verification.switching_feasibility,
                    "Has Stockyard": verification.has_physical_stockyard,
                    "Processing": verification.processing_capability,
                    "Litigation Flag": verification.promoter_litigation_flag,
                    "GST Filing": verification.gst_filing_regularity,
                    "Confidence": verification.evidence_confidence,
                    "Verbatim Evidence": verification.verbatim_evidence,
                    "Falsification Notes": verification.falsification_notes,
                    "ToT Brand Conclusion": verification.tot.brand_alignment.conclusion,
                    "ToT Infrastructure Conclusion": verification.tot.infrastructure.conclusion,
                    "ToT Risk Conclusion": verification.tot.onboarding_risk.conclusion,
                    "Stage 2 Rationale": verification.stage_2.final_rationale,
                    "Stage 2 Challenges": " | ".join(verification.stage_2.challenged_claims),
                    "Stage 2 Unresolved Questions": " | ".join(
                        verification.stage_2.unresolved_questions
                    ),
                }
            )
        rows.append(row)
    return _safe_dataframe(rows)


def _audit_records_to_df(records: list[CandidateAuditRecord]) -> pd.DataFrame:
    """Flatten Needs Data and deterministic-rejection records for export."""
    rows = [
        {
            "Queue": record.disposition.value,
            "GSTIN": record.candidate.gstin,
            "PAN": record.candidate.pan,
            "Legal Name": record.candidate.legal_name,
            "Trade Name": record.candidate.trade_name or "",
            "State": record.candidate.state_name,
            "Turnover Slab": record.candidate.turnover_slab or "",
            "Turnover Known": record.candidate.turnover_known,
            "Geocode Source": record.candidate.geocode_source.value,
            "Address": record.candidate.permanent_address or "",
            "Coarse Distance (km)": record.coarse_road_distance_km,
            "Nearest Existing Distributor (km)": record.nearest_existing_distributor_km,
            "Reasons": " | ".join(record.reasons),
        }
        for record in records
    ]
    return _safe_dataframe(rows)


def _export_csv(dataframe: pd.DataFrame, output: Path) -> None:
    dataframe.to_csv(output, index=False, encoding="utf-8-sig")
    console.print(f"[green]CSV exported to:[/] {output}")


def _export_json(data: Any, output: Path) -> None:
    if isinstance(data, list):
        payload = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in data
        ]
    elif hasattr(data, "model_dump"):
        payload = data.model_dump(mode="json")
    else:
        payload = data
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    console.print(f"[green]JSON exported to:[/] {output}")


def _export_table(evaluations: list[FinalEvaluation]) -> None:
    """Print scored rows; audit queues are written to structured artifacts."""
    from rich.table import Table

    table = Table(title="Agent 0 — Scored Candidates", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Queue", max_width=24)
    table.add_column("Grade", width=5)
    table.add_column("Name", max_width=28)
    table.add_column("GSTIN", width=16)
    table.add_column("Profit", justify="right", width=7)
    table.add_column("Evidence", justify="right", width=8)

    for index, evaluation in enumerate(evaluations, start=1):
        table.add_row(
            str(index),
            evaluation.recommendation_state.value,
            evaluation.final_grade,
            (evaluation.trade_name or evaluation.legal_name)[:28],
            evaluation.gstin,
            f"{evaluation.scores.profit_score:.1f}",
            str(evaluation.evidence_validation.valid_citation_count),
        )
    console.print(table)


def _export_run(result: PipelineRunResult, output_dir: Path, fmt: str) -> None:
    """Write every commercial queue and the reproducibility manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    verified = [
        evaluation
        for evaluation in result.evaluations
        if evaluation.recommendation_state == RecommendationState.VERIFIED_RECOMMENDATION
    ]
    prioritized_leads = [
        evaluation
        for evaluation in result.evaluations
        if evaluation.recommendation_state == RecommendationState.PRIORITIZED_LEAD
    ]
    post_verification_rejections = [
        evaluation
        for evaluation in result.evaluations
        if evaluation.recommendation_state == RecommendationState.REJECTED
    ]

    if fmt in ("csv", "all"):
        _export_csv(_evaluations_to_df(result.evaluations), output_dir / "agent0_results.csv")
        _export_csv(
            _evaluations_to_df(verified),
            output_dir / "agent0_verified_recommendations.csv",
        )
        _export_csv(
            _evaluations_to_df(prioritized_leads),
            output_dir / "agent0_prioritized_leads.csv",
        )
        _export_csv(_audit_records_to_df(result.needs_data), output_dir / "agent0_needs_data.csv")
        _export_csv(
            _audit_records_to_df(result.rejected),
            output_dir / "agent0_prefilter_rejected.csv",
        )
        _export_csv(
            _evaluations_to_df(post_verification_rejections),
            output_dir / "agent0_post_verification_rejected.csv",
        )

    if fmt in ("json", "all"):
        _export_json(result.evaluations, output_dir / "agent0_results.json")
        _export_json(verified, output_dir / "agent0_verified_recommendations.json")
        _export_json(prioritized_leads, output_dir / "agent0_prioritized_leads.json")
        _export_json(result.needs_data, output_dir / "agent0_needs_data.json")
        _export_json(result.rejected, output_dir / "agent0_prefilter_rejected.json")
        _export_json(
            post_verification_rejections,
            output_dir / "agent0_post_verification_rejected.json",
        )
        _export_json(result.manifest, output_dir / "agent0_run_manifest.json")

    if fmt in ("table", "all"):
        _export_table(result.evaluations)


@app.command()
def run(
    distributor_csv: Path = typer.Option(
        ...,
        "--distributor-csv",
        "-d",
        help="Path to distributor CSV file.",
        exists=True,
        readable=True,
        dir_okay=False,
    ),
    depot_name: str = typer.Option(..., "--depot-name", help="Target Tata Steel depot/plant."),
    depot_lat: float = typer.Option(..., "--depot-lat", help="Depot latitude."),
    depot_lon: float = typer.Option(..., "--depot-lon", help="Depot longitude."),
    tata_csv: Optional[Path] = typer.Option(
        None,
        "--tata-csv",
        "-t",
        help="Existing Tata distributor CSV used for anti-cannibalization.",
        exists=True,
        readable=True,
        dir_okay=False,
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration YAML. Defaults to ./config.yaml when present.",
        exists=True,
        readable=True,
        dir_okay=False,
    ),
    env_file: Optional[Path] = typer.Option(
        None,
        "--env",
        help="Environment file. Defaults to ./.env when present.",
        exists=True,
        readable=True,
        dir_okay=False,
    ),
    output_format: str = typer.Option(
        "all",
        "--format",
        "-f",
        help="Output format: csv, json, table, or all.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory for queue and manifest artifacts. Defaults to ./output/.",
    ),
    skip_enrichment: bool = typer.Option(False, "--skip-enrichment"),
    skip_llm: bool = typer.Option(False, "--skip-llm"),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Disable every network integration: Firecrawl, LLMs, Nominatim, and Maps.",
    ),
    use_nominatim: bool = typer.Option(
        False,
        "--nominatim",
        help="Use Nominatim address/PIN lookup for unresolved locations.",
    ),
    max_distance: Optional[float] = typer.Option(
        None,
        "--max-distance",
        help="Override trusted-location coarse-distance gate in km.",
    ),
    top_n: Optional[int] = typer.Option(
        None,
        "--top-n",
        help="Override number of eligible candidates to enrich.",
    ),
    gmaps_top: Optional[int] = typer.Option(
        None,
        "--gmaps-top",
        help="Limit Google Maps refinement to N eligible candidates; 0 disables.",
    ),
    llm_provider: Optional[str] = typer.Option(
        None,
        "--llm",
        help="LLM provider override: anthropic, openai, or gemini.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run discovery → evidence → nested ToT/S2T → commercial queue export."""
    _configure_logging(verbose=verbose)
    fmt = output_format.lower()
    if fmt not in ("csv", "json", "table", "all"):
        console.print(
            f"[red]Unknown --format '{output_format}'. Expected csv, json, table, or all.[/]"
        )
        raise typer.Exit(2)

    cfg_path = config_file or Path("config.yaml")
    env_path = env_file or Path(".env")
    try:
        config = load_config(
            config_path=cfg_path if cfg_path.exists() else None,
            env_path=env_path if env_path.exists() else None,
        )
        if max_distance is not None:
            config.gates.max_coarse_distance_km = max_distance
        if top_n is not None:
            config.enrichment.top_n_for_enrichment = top_n
        if gmaps_top is not None:
            config.enrichment.gmaps_refine_top = gmaps_top
        if llm_provider:
            config.llm.provider = llm_provider.lower()
    except (ValidationError, ValueError) as exc:
        console.print(f"[red]Invalid configuration: {exc}[/]")
        raise typer.Exit(2) from exc

    if (
        not offline
        and not skip_llm
        and not config.anthropic_api_key
        and not config.openai_api_key
        and not config.gemini_api_key
    ):
        console.print(
            "[red]No LLM API key found. Set an API key, use --skip-llm, or use --offline.[/]"
        )
        raise typer.Exit(1)

    # Preserve the approved existing fallback behavior when the configured
    # primary key is absent. Runtime provider failures still fall through in
    # run_tot_s2t.
    if not offline and config.llm.provider == "anthropic" and not config.anthropic_api_key:
        if config.openai_api_key:
            console.print("[yellow]Anthropic key not found. Falling back to OpenAI.[/]")
            config.llm.provider = "openai"
        elif config.gemini_api_key:
            console.print("[yellow]Anthropic key not found. Falling back to Gemini.[/]")
            config.llm.provider = "gemini"

    try:
        depot = TataDepot(name=depot_name, latitude=depot_lat, longitude=depot_lon)
    except ValidationError as exc:
        console.print(f"[red]Invalid depot coordinates: {exc}[/]")
        raise typer.Exit(2) from exc
    console.print()
    console.print("[bold magenta]Agent 0 — Tata Steel distributor discovery[/]")
    console.print(f"  Target Depot: [bold]{depot.name}[/] ({depot.latitude}, {depot.longitude})")
    console.print(
        f"  Mode: {'Offline (no network)' if offline else 'Connected'} | "
        f"LLM: {config.llm.provider}/{config.llm.model}"
    )
    if offline:
        console.print(
            "[yellow]  Offline mode disables Firecrawl, LLMs, Nominatim, and Google Maps.[/]"
        )

    result = run_pipeline(
        distributor_csv=distributor_csv,
        tata_depot=depot,
        config=config,
        existing_tata_csv=tata_csv,
        skip_enrichment=skip_enrichment,
        skip_llm=skip_llm,
        use_nominatim=use_nominatim,
        offline=offline,
    )
    out_dir = output_dir or Path("output")
    _export_run(result, out_dir, fmt)

    if not result.evaluations and not result.needs_data and not result.rejected:
        console.print("[red]No valid candidate rows were produced from the input CSV.[/]")
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Show effective local configuration without displaying secret values."""
    _configure_logging()
    cfg_path = Path("config.yaml")
    try:
        config = load_config(config_path=cfg_path if cfg_path.exists() else None)
    except (ValidationError, ValueError) as exc:
        console.print(f"[red]Invalid configuration: {exc}[/]")
        raise typer.Exit(2) from exc

    console.print("[bold]Agent 0 Configuration:[/]")
    console.print(
        f"  Config File: {cfg_path if cfg_path.exists() else 'defaults only (no config.yaml found)'}"
    )
    console.print(f"  LLM Provider: {config.llm.provider}")
    console.print(f"  LLM Model: {config.llm.model}")
    console.print(f"  Anthropic Key: {'✓ Set' if config.anthropic_api_key else '✗ Missing'}")
    console.print(f"  OpenAI Key: {'✓ Set' if config.openai_api_key else '✗ Missing'}")
    console.print(f"  Gemini Key: {'✓ Set' if config.gemini_api_key else '✗ Missing'}")
    console.print(f"  Firecrawl Key: {'✓ Set' if config.firecrawl_api_key else '✗ Missing'}")
    console.print(f"  Google Maps Key: {'✓ Set' if config.gmaps_api_key else '✗ Missing'}")
    console.print()
    console.print("[bold]Eligibility:[/]")
    console.print(f"  Minimum Turnover: ₹{config.gates.min_turnover_cr} Cr")
    console.print(
        f"  Maximum Turnover: "
        f"{'None (all sizes retained)' if config.gates.max_turnover_cr is None else config.gates.max_turnover_cr}"
    )
    console.print(f"  Max Trusted-Location Distance: {config.gates.max_coarse_distance_km} km")
    console.print(f"  Anti-Cannibalization Radius: {config.gates.min_cannibalization_radius_km} km")
    console.print(f"  Top-N for Enrichment: {config.enrichment.top_n_for_enrichment}")
    console.print("  Offline Mode: --offline disables all network integrations")


if __name__ == "__main__":
    app()
