"""Typer CLI for Agent 0 — Tata Steel GTM Distributor Search Engine.

Usage:
    agent0 run --distributor-csv data.csv --depot-name "Jamshedpur" --depot-lat 22.80 --depot-lon 86.20
    agent0 run --distributor-csv data.csv --tata-csv existing.csv --config config.yaml --format table
    agent0 run --distributor-csv data.csv --depot-name "Kalinganagar" --depot-lat 20.93 --depot-lon 86.08 --skip-llm --format csv --output results.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console

from agent_0.config import load_config
from agent_0.models import FinalEvaluation, TataDepot
from agent_0.pipeline import run_pipeline

app = typer.Typer(
    name="agent0",
    help="Agent 0 — Tata Steel GTM Distributor Search & Scoring Engine",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _configure_logging(verbose: bool = False) -> None:
    """Enable module logging.

    Without this, every ``logger.error`` in the pipeline (LLM failures,
    enrichment errors, cache problems) is silently discarded and the run
    looks clean while actually having degraded.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


# ────────────────────────────────────────────────────────
# Export Helpers
# ────────────────────────────────────────────────────────


def _evaluations_to_df(evaluations: list[FinalEvaluation]) -> pd.DataFrame:
    """Flatten evaluations into a pandas DataFrame for CSV/table export."""
    rows = []
    for e in evaluations:
        row = {
            "Grade": e.final_grade,
            "GSTIN": e.gstin,
            "PAN": e.pan,
            "Legal Name": e.legal_name,
            "Trade Name": e.trade_name or "",
            "State": e.state,
            "Turnover Slab": e.turnover_slab or "",
            "Distance (km)": round(e.driving_distance_km, 1),
            "Distance Source": e.distance_source,
            "Geocode Source": e.geocode_source,
            "Profit Score": round(e.scores.profit_score, 2),
            "Ease Score": round(e.scores.ease_score, 2),
            "S_Logistics": round(e.scores.s_logistics, 2),
            "S_Turnover": round(e.scores.s_turnover, 2),
            "S_Processing": round(e.scores.s_processing, 2),
            "E_Compliance": round(e.scores.e_compliance, 2),
            "E_Switching": round(e.scores.e_switching, 2),
            "E_Infra": round(e.scores.e_infra, 2),
            "Enrichment Source": e.enrichment_source,
            "Verification Provider": e.verification_provider or "",
            "Verification Status": e.verification_status,
            "Address": e.permanent_address or "",
            "eKYC Status": e.ekyc_status or "",
            "Aadhaar Verified": e.aadhaar_verified or "",
        }
        if e.verification:
            row.update(
                {
                    "Is Steel Dealer": e.verification.is_primary_steel_dealer,
                    "Brands Handled": ", ".join(e.verification.brands_currently_handled),
                    "Switching Feasibility": e.verification.switching_feasibility,
                    "Has Stockyard": e.verification.has_physical_stockyard,
                    "Processing": e.verification.processing_capability,
                    "Litigation Flag": e.verification.promoter_litigation_flag,
                    "GST Filing": e.verification.gst_filing_regularity,
                    "Confidence": e.verification.evidence_confidence,
                    "Verbatim Evidence": e.verification.verbatim_evidence,
                    "Falsification Notes": e.verification.falsification_notes,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _export_csv(evaluations: list[FinalEvaluation], output: Path) -> None:
    df = _evaluations_to_df(evaluations)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    console.print(f"[green]CSV exported to:[/] {output}")


def _export_json(evaluations: list[FinalEvaluation], output: Path) -> None:
    data = [e.model_dump(mode="json") for e in evaluations]
    with open(output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    console.print(f"[green]JSON exported to:[/] {output}")


def _export_table(evaluations: list[FinalEvaluation]) -> None:
    from rich.table import Table

    table = Table(title="Agent 0 — Full Results", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Grade", width=5)
    table.add_column("Name", max_width=28)
    table.add_column("GSTIN", width=16)
    table.add_column("State", width=14)
    table.add_column("Turnover", width=12)
    table.add_column("Dist", justify="right", width=7)
    table.add_column("Profit", justify="right", width=7)
    table.add_column("Ease", justify="right", width=7)
    table.add_column("Conf", width=8)

    grade_colors = {"A": "green", "B": "yellow", "C": "blue", "D": "red"}

    for i, e in enumerate(evaluations, 1):
        color = grade_colors.get(e.final_grade, "white")
        conf = e.verification.evidence_confidence if e.verification else "N/A"
        name = (e.trade_name or e.legal_name)[:28]
        table.add_row(
            str(i),
            f"[{color}]{e.final_grade}[/]",
            name,
            e.gstin,
            e.state,
            e.turnover_slab or "N/A",
            f"{e.driving_distance_km:.0f}",
            f"{e.scores.profit_score:.1f}",
            f"{e.scores.ease_score:.1f}",
            str(conf),
        )

    console.print(table)


# ────────────────────────────────────────────────────────
# CLI Commands
# ────────────────────────────────────────────────────────


@app.command()
def run(
    distributor_csv: Path = typer.Option(
        ...,
        "--distributor-csv",
        "-d",
        help="Path to distributor CSV file (Clear GST format).",
        exists=True,
        readable=True,
    ),
    depot_name: str = typer.Option(
        ...,
        "--depot-name",
        help="Name of the target Tata Steel depot/plant.",
    ),
    depot_lat: float = typer.Option(
        ...,
        "--depot-lat",
        help="Latitude of the target depot.",
    ),
    depot_lon: float = typer.Option(
        ...,
        "--depot-lon",
        help="Longitude of the target depot.",
    ),
    tata_csv: Optional[Path] = typer.Option(
        None,
        "--tata-csv",
        "-t",
        help="Optional CSV of existing Tata distributors (same format).",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml. Defaults to ./config.yaml.",
    ),
    env_file: Optional[Path] = typer.Option(
        None,
        "--env",
        help="Path to .env file. Defaults to ./.env.",
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
        help="Output directory for csv/json files. Defaults to ./output/.",
    ),
    skip_enrichment: bool = typer.Option(
        False,
        "--skip-enrichment",
        help="Skip Firecrawl web enrichment.",
    ),
    skip_llm: bool = typer.Option(
        False,
        "--skip-llm",
        help="Skip Claude/OpenAI ToT+S2T verification (use safe defaults).",
    ),
    use_nominatim: bool = typer.Option(
        False,
        "--nominatim",
        help="Use Nominatim API for geocoding (slow, 1 req/sec).",
    ),
    max_distance: Optional[float] = typer.Option(
        None,
        "--max-distance",
        help="Override max coarse distance gate (km).",
    ),
    top_n: Optional[int] = typer.Option(
        None,
        "--top-n",
        help="Override number of candidates to enrich via Firecrawl+LLM.",
    ),
    gmaps_top: Optional[int] = typer.Option(
        None,
        "--gmaps-top",
        help="Limit GMaps refinement to the N closest candidates (0 disables).",
    ),
    llm_provider: Optional[str] = typer.Option(
        None,
        "--llm",
        help="LLM provider override: 'anthropic', 'openai', or 'gemini'.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Emit detailed diagnostics (LLM failures, enrichment errors) to stderr.",
    ),
) -> None:
    """Run the full Agent 0 pipeline: Search → Filter → Enrich → Score → Grade."""

    _configure_logging(verbose=verbose)

    # Validate the output format BEFORE running the pipeline: an invalid
    # format must not cost a full run of enrichment + LLM calls.
    fmt = output_format.lower()
    if fmt not in ("csv", "json", "table", "all"):
        console.print(
            f"[red]Unknown --format '{output_format}'. "
            "Expected one of: csv, json, table, all.[/]"
        )
        raise typer.Exit(2)

    console.print()
    console.print(
        "[bold magenta]╔══════════════════════════════════════════════╗[/]"
    )
    console.print(
        "[bold magenta]║   Agent 0 — Tata Steel GTM Search Engine    ║[/]"
    )
    console.print(
        "[bold magenta]╚══════════════════════════════════════════════╝[/]"
    )
    console.print()

    # ── Load config ────────────────────────────────────
    cfg_path = config_file or Path("config.yaml")
    env_path_resolved = env_file or Path(".env")
    config = load_config(
        config_path=cfg_path if cfg_path.exists() else None,
        env_path=env_path_resolved if env_path_resolved.exists() else None,
    )

    # Apply CLI overrides
    if max_distance is not None:
        config.gates.max_coarse_distance_km = max_distance
    if top_n is not None:
        config.enrichment.top_n_for_enrichment = top_n
    if gmaps_top is not None:
        config.enrichment.gmaps_refine_top = gmaps_top
    if llm_provider:
        config.llm.provider = llm_provider

    # Validate API keys
    if (
        not skip_llm
        and not config.anthropic_api_key
        and not config.openai_api_key
        and not config.gemini_api_key
    ):
        console.print(
            "[red]No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY "
            "in .env, or use --skip-llm.[/]"
        )
        raise typer.Exit(1)

    if config.llm.provider == "anthropic" and not config.anthropic_api_key:
        if config.openai_api_key:
            console.print(
                "[yellow]Anthropic key not found. Falling back to OpenAI.[/]"
            )
            config.llm.provider = "openai"
        elif config.gemini_api_key:
            console.print(
                "[yellow]Anthropic key not found. Falling back to Gemini.[/]"
            )
            config.llm.provider = "gemini"
        elif not skip_llm:
            console.print("[red]ANTHROPIC_API_KEY not set. Use --skip-llm or provide the key.[/]")
            raise typer.Exit(1)

    # ── Build depot ────────────────────────────────────
    if skip_llm:
        console.print(
            "[yellow]Warning: --skip-llm passed. LLM verification is skipped, "
            "so evidence confidence defaults to NONE (0.10) and "
            "Grades A/B are unreachable without LLM verification (capped at C).[/]"
        )
    depot = TataDepot(name=depot_name, latitude=depot_lat, longitude=depot_lon)
    console.print(f"  Target Depot: [bold]{depot.name}[/] ({depot.latitude}, {depot.longitude})")
    console.print(f"  Config: {cfg_path}")
    console.print(f"  LLM: {config.llm.provider}/{config.llm.model}")
    console.print(f"  Enrichment: {'Enabled' if not skip_enrichment and config.firecrawl_api_key else 'Disabled'}")
    console.print()

    # ── Run pipeline ───────────────────────────────────
    evaluations = run_pipeline(
        distributor_csv=distributor_csv,
        tata_depot=depot,
        config=config,
        existing_tata_csv=tata_csv,
        skip_enrichment=skip_enrichment,
        skip_llm=skip_llm,
        use_nominatim=use_nominatim,
    )

    if not evaluations:
        console.print("[red]No results produced.[/]")
        raise typer.Exit(1)

    # ── Export results ─────────────────────────────────
    out_dir = output_dir or Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt in ("csv", "all"):
        _export_csv(evaluations, out_dir / "agent0_results.csv")
    if fmt in ("json", "all"):
        _export_json(evaluations, out_dir / "agent0_results.json")
    if fmt in ("table", "all"):
        _export_table(evaluations)


@app.command()
def info() -> None:
    """Show Agent 0 configuration and system info."""
    _configure_logging()
    # Resolve config.yaml relative to the CWD, exactly like `run` does,
    # so the reported thresholds match what a pipeline run would use.
    cfg_path = Path("config.yaml")
    config = load_config(config_path=cfg_path if cfg_path.exists() else None)
    console.print("[bold]Agent 0 Configuration:[/]")
    console.print(f"  Config File: {cfg_path if cfg_path.exists() else 'defaults only (no config.yaml found)'}")
    console.print(f"  LLM Provider: {config.llm.provider}")
    console.print(f"  LLM Model: {config.llm.model}")
    console.print(f"  Anthropic Key: {'✓ Set' if config.anthropic_api_key else '✗ Missing'}")
    console.print(f"  Anthropic Workspace: {'✓ Set' if config.anthropic_workspace_id else '— Unset (only needed for unscoped enterprise keys)'}")
    console.print(f"  OpenAI Key: {'✓ Set' if config.openai_api_key else '✗ Missing'}")
    console.print(f"  Gemini Key: {'✓ Set' if config.gemini_api_key else '✗ Missing'}")
    console.print(f"  Firecrawl Key: {'✓ Set' if config.firecrawl_api_key else '✗ Missing'}")
    console.print(f"  Google Maps Key: {'✓ Set' if config.gmaps_api_key else '✗ Missing'}")
    console.print()
    console.print("[bold]Gate Thresholds:[/]")
    console.print(f"  Turnover Range: ₹{config.gates.min_turnover_cr}–{config.gates.max_turnover_cr} Cr")
    console.print(f"  Max Distance: {config.gates.max_coarse_distance_km} km")
    console.print(f"  Anti-Cannibalization Radius: {config.gates.min_cannibalization_radius_km} km")
    console.print(f"  HSN Prefixes: {config.gates.required_hsn_prefixes}")
    console.print(f"  Top-N for Enrichment: {config.enrichment.top_n_for_enrichment}")


if __name__ == "__main__":
    app()
