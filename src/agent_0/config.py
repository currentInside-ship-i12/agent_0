"""Configuration loader for Agent 0.

Loads from config.yaml (defaults) → .env (API keys) → CLI overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# ────────────────────────────────────────────────────────
# Config Sub-models
# ────────────────────────────────────────────────────────


class StrictConfig(BaseModel):
    """Base for all config sections.

    ``extra="forbid"`` makes an unknown key in config.yaml a hard error
    instead of a silently ignored typo.
    """

    model_config = ConfigDict(extra="forbid")


class GateConfig(StrictConfig):
    min_turnover_cr: float = 15.0
    max_turnover_cr: float = 300.0
    max_coarse_distance_km: float = 180.0
    min_cannibalization_radius_km: float = 30.0
    required_hsn_prefixes: list[str] = Field(default_factory=lambda: ["72", "73"])
    allowed_business_activities: list[str] = Field(
        default_factory=lambda: ["MFT", "TRD:TWD", "TRD:TRR"]
    )


class SpatialConfig(StrictConfig):
    road_tortuosity_factor: float = 1.30
    d_target_max_km: float = 150.0


class ProfitWeights(StrictConfig):
    logistics: float = 0.45
    turnover: float = 0.35
    processing: float = 0.20


class EaseWeights(StrictConfig):
    compliance: float = 0.40
    switching: float = 0.35
    infrastructure: float = 0.25


class GradingConfig(StrictConfig):
    grade_a_min_profit: float = 75.0
    grade_a_min_ease: float = 75.0
    grade_a_min_confidence: float = 0.80
    grade_b_min_profit: float = 70.0
    grade_b_min_ease: float = 50.0
    grade_c_min_profit: float = 50.0
    grade_d_max_profit: float = 40.0
    grade_d_max_ease: float = 40.0


class EnrichmentConfig(StrictConfig):
    top_n_for_enrichment: int = 50
    # Spend governor: Google Maps refinement applies only to the N closest
    # candidates (by coarse distance) in the enrichment batch. 0 disables
    # refinement entirely; values >= batch size refine everything.
    gmaps_refine_top: int = 5
    cache_dir: str = ".agent0_cache"
    # Maximum age of a cached Firecrawl payload before it is refetched.
    # 0 disables expiry. Cached payloads with no real evidence are always
    # refetched regardless of this value.
    cache_ttl_days: int = 30


class LLMConfig(StrictConfig):
    provider: str = "anthropic"  # "anthropic" or "openai"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 2000


# ────────────────────────────────────────────────────────
# Root Config
# ────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    # Reject unknown keys instead of silently ignoring them: a typo in
    # config.yaml would otherwise be dropped without any warning and the
    # pipeline would quietly run with code defaults.
    model_config = ConfigDict(extra="forbid")

    gates: GateConfig = Field(default_factory=GateConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    profit_weights: ProfitWeights = Field(default_factory=ProfitWeights)
    ease_weights: EaseWeights = Field(default_factory=EaseWeights)
    grading: GradingConfig = Field(default_factory=GradingConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # API Keys (loaded from .env or environment).
    # repr=False keeps them out of logs, tracebacks and model dumps.
    anthropic_api_key: str = Field(default="", repr=False)
    anthropic_workspace_id: str = Field(default="", repr=False)
    openai_api_key: str = Field(default="", repr=False)
    gemini_api_key: str = Field(default="", repr=False)
    firecrawl_api_key: str = Field(default="", repr=False)
    gmaps_api_key: str = Field(default="", repr=False)


def load_config(
    config_path: Optional[Path] = None,
    env_path: Optional[Path] = None,
) -> PipelineConfig:
    """Load pipeline configuration from YAML + .env files.

    Priority: .env / OS environment (API keys) > YAML (parameters) > defaults.
    """
    # Load .env file for API keys
    if env_path and env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # Try default .env in cwd

    # Load YAML config
    yaml_data: dict = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}

    # Build config from YAML
    config = PipelineConfig(**yaml_data) if yaml_data else PipelineConfig()

    # Override API keys from environment
    config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", config.anthropic_api_key)
    config.anthropic_workspace_id = os.getenv(
        "ANTHROPIC_WORKSPACE_ID", config.anthropic_workspace_id
    )
    config.openai_api_key = os.getenv("OPENAI_API_KEY", config.openai_api_key)
    config.gemini_api_key = os.getenv(
        "GEMINI_API_KEY", config.gemini_api_key
    ) or os.getenv("GOOGLE_API_KEY", config.gemini_api_key)
    config.firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", config.firecrawl_api_key)
    config.gmaps_api_key = os.getenv("GMAPS_API_KEY", config.gmaps_api_key)

    return config
