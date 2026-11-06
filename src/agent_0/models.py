"""Core data models for Agent 0 pipeline.

Every struct in this module is a strict Pydantic v2 model.
LLM outputs are constrained to Literal types (no free-form floats)
to maximise determinism across runs.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class GeoSource(str, Enum):
    """Where a candidate's coordinates came from."""

    PINCODE = "PINCODE"
    STATE_CENTROID = "STATE_CENTROID"
    GMAPS = "GMAPS"
    NOMINATIM = "NOMINATIM"
    MANUAL = "MANUAL"
    NONE = "NONE"


class EnrichmentSource(str, Enum):
    """Where verification evidence text came from."""

    FIRECRAWL = "FIRECRAWL"
    MOCK = "MOCK"
    NONE = "NONE"


# ────────────────────────────────────────────────────────
# Input Models
# ────────────────────────────────────────────────────────


class TataDepot(BaseModel):
    """A Tata Steel plant, mother depot, or consignment agent location."""

    name: str
    latitude: float
    longitude: float


class DistributorCandidate(BaseModel):
    """Normalised distributor record, produced by csv_loader from raw CSV.

    This model contains every field consumed by downstream stages
    (pre-filter, enrichment, scoring).  The csv_loader is the sole
    producer; downstream code never mutates these fields.
    """

    # ── Identity ───────────────────────────────────────
    gstin: str
    pan: str
    legal_name: str
    trade_name: Optional[str] = None
    display_name: str = ""  # trade_name if available, else legal_name

    # ── Location ───────────────────────────────────────
    state_code: str = ""  # First 2 digits of GSTIN
    state_name: str = ""
    permanent_address: Optional[str] = None
    pincode: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    geocode_source: GeoSource = GeoSource.NONE

    # ── Turnover ───────────────────────────────────────
    turnover_slab: Optional[str] = None  # Raw slab string from CSV
    turnover_midpoint_cr: float = 0.0  # Midpoint of slab in ₹ Crores

    # ── Registration & Compliance ──────────────────────
    registration_status: str = "Unknown"
    nature_of_core_business: Optional[str] = None  # MFT, TRD:TWD, TRD:TRR, SPO
    constitution: Optional[str] = None
    einvoice_enabled: bool = False
    registration_type: Optional[str] = None
    cancelation_date: Optional[str] = None
    ekyc_status: Optional[str] = None  # Yes / No / Not Applicable
    aadhaar_verified: Optional[str] = None  # Yes / No

    # ── Product Classification ─────────────────────────
    hsn_codes: List[str] = Field(default_factory=list)
    hsn_descriptions: List[str] = Field(default_factory=list)
    sac_codes: List[str] = Field(default_factory=list)
    sac_descriptions: List[str] = Field(default_factory=list)
    has_steel_hsn: bool = False  # True if any HSN starts with 72 or 73
    is_sac_only: bool = False  # True if entity has SAC codes but no HSN codes

    # ── Business Activities & Infrastructure ───────────
    business_activities: List[str] = Field(default_factory=list)

    # ── People ─────────────────────────────────────────
    member_names: List[str] = Field(default_factory=list)

    # ── Additional Addresses ───────────────────────────
    additional_addresses: List[str] = Field(default_factory=list)
    num_additional_addresses: int = 0


# ────────────────────────────────────────────────────────
# Pre-filter Output
# ────────────────────────────────────────────────────────


class PrefilterResult(BaseModel):
    """Result of the deterministic pre-screening stage."""

    candidate: DistributorCandidate
    passed: bool
    rejection_reasons: List[str] = Field(default_factory=list)
    coarse_road_distance_km: Optional[float] = None
    nearest_existing_distributor_km: Optional[float] = None


# ────────────────────────────────────────────────────────
# Claude ToT + S2T Output
# ────────────────────────────────────────────────────────

# Discrete confidence buckets — eliminates calibration drift.
CONFIDENCE_SCORE_MAP: dict[str, float] = {
    "STRONG": 0.90,
    "MODERATE": 0.65,
    "WEAK": 0.35,
    "NONE": 0.10,
}


class S2TVerificationOutput(BaseModel):
    """Structured primitives emitted by Claude after ToT + S2T reasoning.

    Every field is a strict Literal / bool / List — no free-form
    floats except ``evidence_confidence`` which is mapped from
    a discrete bucket via ``CONFIDENCE_SCORE_MAP``.
    """

    # ── Branch 1: Brand Alignment ──────────────────────
    is_primary_steel_dealer: bool = Field(
        description=(
            "True if the entity is a genuine steel stockist, distributor, "
            "or manufacturer. False if scrap trader, logistics broker, "
            "or unrelated subsidiary."
        )
    )
    brands_currently_handled: List[str] = Field(
        default_factory=list,
        description="Steel brands currently distributed (e.g., JSW, SAIL, Jindal, local mills).",
    )
    is_exclusive_to_competitor: bool = Field(
        default=False,
        description=(
            "True if entity is known to be an exclusive tier-1 dealer "
            "for a competing steel manufacturer."
        ),
    )
    switching_feasibility: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description=(
            "HIGH = multi-brand / unbranded / easy to onboard. "
            "MEDIUM = competitor tier-2, some friction. "
            "LOW = competitor exclusive, high resistance."
        )
    )

    # ── Branch 2: Infrastructure ───────────────────────
    has_physical_stockyard: bool = Field(
        description="Evidence of a dedicated storage yard, warehouse, or depot.",
    )
    processing_capability: Literal["ADVANCED", "BASIC", "NONE"] = Field(
        description=(
            "ADVANCED = slitting / cut-to-length / decoiling / fabrication. "
            "BASIC = stockyard with crane or manual handling. "
            "NONE = office-only or pure drop-ship trader."
        )
    )

    # ── Branch 3: Onboarding Risk ──────────────────────
    promoter_litigation_flag: bool = Field(
        default=False,
        description="True if active legal disputes, tax evasion flags, or director disqualification found.",
    )
    gst_filing_regularity: Literal["REGULAR", "IRREGULAR", "CANCELLED", "UNKNOWN"] = Field(
        default="UNKNOWN",
        description="GST return filing pattern inferred from available data.",
    )

    # ── S2T Meta ───────────────────────────────────────
    evidence_confidence: Literal["STRONG", "MODERATE", "WEAK", "NONE"] = Field(
        description=(
            "STRONG = 3+ independent sources confirming claims. "
            "MODERATE = 1-2 sources, partially verified. "
            "WEAK = single unverified mention. "
            "NONE = no evidence found."
        )
    )
    verbatim_evidence: str = Field(
        default="",
        description="Exact quotes, URLs, and text snippets from sources supporting claims. Zero extrapolation.",
    )
    falsification_notes: str = Field(
        default="",
        description="Adversarial red-team analysis: how this candidate could be a false positive.",
    )

    @property
    def confidence_score(self) -> float:
        """Map discrete confidence bucket to a numeric score."""
        return CONFIDENCE_SCORE_MAP.get(self.evidence_confidence, 0.10)


# ────────────────────────────────────────────────────────
# Scoring Output
# ────────────────────────────────────────────────────────


class ScoreBreakdown(BaseModel):
    """Component-level score decomposition for auditability."""

    # Profit components
    s_logistics: float = 0.0
    s_turnover: float = 0.0
    s_processing: float = 0.0
    profit_score: float = 0.0

    # Ease components
    e_compliance: float = 0.0
    e_switching: float = 0.0
    e_infra: float = 0.0
    ease_score: float = 0.0


class FinalEvaluation(BaseModel):
    """Complete evaluation result for a single distributor candidate."""

    # ── Identity ───────────────────────────────────────
    gstin: str
    pan: str
    legal_name: str
    trade_name: Optional[str] = None
    state: str = ""

    # ── Spatial ────────────────────────────────────────
    driving_distance_km: float = 0.0
    # "COARSE" = great-circle distance inflated by the road tortuosity
    # factor (an estimate, not a surveyed road distance).
    # "GMAPS"  = driving distance from the Google Distance Matrix API.
    distance_source: Literal["GMAPS", "COARSE"] = "COARSE"
    # Provenance of the coordinates the distance was derived from.
    # STATE_CENTROID means the candidate was placed at its state capital
    # because its PIN code could not be resolved — treat its distance
    # as unreliable.
    geocode_source: GeoSource = GeoSource.NONE

    # ── Scores ─────────────────────────────────────────
    scores: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    final_grade: Literal["A", "B", "C", "D"] = "D"

    # ── Verification ───────────────────────────────────
    verification: Optional[S2TVerificationOutput] = None
    enrichment_source: EnrichmentSource = EnrichmentSource.NONE
    # Which LLM provider actually served this verification. Differs from the
    # configured provider when a fallback was used.
    verification_provider: Optional[str] = None
    # VERIFIED = LLM returned evidence beyond NONE; PARTIAL = LLM ran with
    # no real enrichment; UNVERIFIED = no LLM evidence; FAILED = all
    # providers failed.
    verification_status: Literal[
        "VERIFIED", "PARTIAL", "UNVERIFIED", "FAILED"
    ] = "UNVERIFIED"

    # ── Metadata ───────────────────────────────────────
    rejection_reasons: List[str] = Field(default_factory=list)
    turnover_slab: Optional[str] = None
    permanent_address: Optional[str] = None
    ekyc_status: Optional[str] = None
    aadhaar_verified: Optional[str] = None
