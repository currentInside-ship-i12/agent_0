"""Configuration loader for Agent 0.

Loads from config.yaml (defaults) → .env (API keys) → CLI overrides.
"""

from __future__ import annotations

import os
from math import isclose
from pathlib import Path
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ────────────────────────────────────────────────────────
# Config Sub-models
# ────────────────────────────────────────────────────────


class StrictConfig(BaseModel):
    """Base for all config sections.

    ``extra="forbid"`` makes an unknown key in config.yaml a hard error
    instead of a silently ignored typo.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class GateConfig(StrictConfig):
    min_turnover_cr: float = Field(default=15.0, ge=0.0)
    # Large distributors remain in the lead pool. Coverage strategy is
    # deliberately separate from deterministic eligibility.
    max_turnover_cr: Optional[float] = Field(default=None, ge=0.0)
    max_coarse_distance_km: float = Field(default=180.0, gt=0.0)
    min_cannibalization_radius_km: float = Field(default=30.0, ge=0.0)
    required_hsn_prefixes: list[str] = Field(
        default_factory=lambda: ["72", "73"],
        min_length=1,
    )
    allowed_business_activities: list[str] = Field(
        default_factory=lambda: ["MFT", "TRD:TWD", "TRD:TRR"],
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_turnover_range(self) -> "GateConfig":
        if self.max_turnover_cr is not None and self.max_turnover_cr < self.min_turnover_cr:
            raise ValueError("max_turnover_cr must be greater than or equal to min_turnover_cr")
        return self


class SpatialConfig(StrictConfig):
    road_tortuosity_factor: float = Field(default=1.30, ge=1.0)
    d_target_max_km: float = Field(default=150.0, gt=0.0)


class ProfitWeights(StrictConfig):
    logistics: float = Field(default=0.45, ge=0.0)
    turnover: float = Field(default=0.35, ge=0.0)
    processing: float = Field(default=0.20, ge=0.0)

    @model_validator(mode="after")
    def validate_total(self) -> "ProfitWeights":
        if not isclose(self.logistics + self.turnover + self.processing, 1.0, abs_tol=1e-9):
            raise ValueError("profit weights must sum to 1.0")
        return self


class EaseWeights(StrictConfig):
    compliance: float = Field(default=0.40, ge=0.0)
    switching: float = Field(default=0.35, ge=0.0)
    infrastructure: float = Field(default=0.25, ge=0.0)

    @model_validator(mode="after")
    def validate_total(self) -> "EaseWeights":
        if not isclose(self.compliance + self.switching + self.infrastructure, 1.0, abs_tol=1e-9):
            raise ValueError("ease weights must sum to 1.0")
        return self


class GradingConfig(StrictConfig):
    grade_a_min_profit: float = Field(default=75.0, ge=0.0, le=100.0)
    grade_a_min_ease: float = Field(default=75.0, ge=0.0, le=100.0)
    grade_a_min_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    grade_b_min_profit: float = Field(default=70.0, ge=0.0, le=100.0)
    grade_b_min_ease: float = Field(default=50.0, ge=0.0, le=100.0)
    grade_c_min_profit: float = Field(default=50.0, ge=0.0, le=100.0)
    grade_d_max_profit: float = Field(default=40.0, ge=0.0, le=100.0)
    grade_d_max_ease: float = Field(default=40.0, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def validate_ladder(self) -> "GradingConfig":
        if not (self.grade_a_min_profit >= self.grade_b_min_profit >= self.grade_c_min_profit):
            raise ValueError("grade profit thresholds must descend from A to C")
        if self.grade_d_max_profit >= self.grade_c_min_profit:
            raise ValueError("grade_d_max_profit must remain below grade_c_min_profit")
        return self


class EnrichmentConfig(StrictConfig):
    top_n_for_enrichment: int = Field(default=50, ge=0)
    # Spend governor: Google Maps refinement applies only to the N closest
    # candidates (by coarse distance) in the enrichment batch. 0 disables
    # refinement entirely; values >= batch size refine everything.
    gmaps_refine_top: int = Field(default=5, ge=0)
    cache_dir: str = ".agent0_cache"
    # Maximum age of a cached Firecrawl payload before it is refetched.
    # 0 disables expiry. Cached payloads with no real evidence are always
    # refetched regardless of this value.
    cache_ttl_days: int = Field(default=30, ge=0)
    max_workers: int = Field(default=3, ge=1)
    max_chars_per_source: int = Field(default=3000, ge=250)
    max_retries: int = Field(default=1, ge=0, le=5)


class LLMConfig(StrictConfig):
    provider: Literal["anthropic", "openai", "gemini"] = "anthropic"
    model: str = Field(default="claude-sonnet-4-20250514", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=256)


# ────────────────────────────────────────────────────────
# Root Config
# ────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    # Reject unknown keys instead of silently ignoring them: a typo in
    # config.yaml would otherwise be dropped without any warning and the
    # pipeline would quietly run with code defaults.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

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

    # Load YAML config. API keys belong in the environment, never in a
    # potentially committed configuration file.
    yaml_data: dict = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}
        if not isinstance(yaml_data, dict):
            raise ValueError("Configuration YAML must contain a mapping at its root")
        secret_keys = {
            "anthropic_api_key",
            "anthropic_workspace_id",
            "openai_api_key",
            "gemini_api_key",
            "firecrawl_api_key",
            "gmaps_api_key",
        }
        supplied_secrets = sorted(secret_keys.intersection(yaml_data))
        if supplied_secrets:
            joined = ", ".join(supplied_secrets)
            raise ValueError(
                f"API credentials must be supplied via environment variables, not YAML: {joined}"
            )

    # Build config from YAML
    config = PipelineConfig(**yaml_data) if yaml_data else PipelineConfig()

    # Override API keys from environment
    config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", config.anthropic_api_key)
    config.anthropic_workspace_id = os.getenv(
        "ANTHROPIC_WORKSPACE_ID", config.anthropic_workspace_id
    )
    config.openai_api_key = os.getenv("OPENAI_API_KEY", config.openai_api_key)
    config.gemini_api_key = os.getenv("GEMINI_API_KEY", config.gemini_api_key) or os.getenv(
        "GOOGLE_API_KEY", config.gemini_api_key
    )
    config.firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", config.firecrawl_api_key)
    config.gmaps_api_key = os.getenv("GMAPS_API_KEY", config.gmaps_api_key)

    return config
