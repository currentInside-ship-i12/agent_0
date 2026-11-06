"""Core data models for Agent 0 pipeline.

Every struct in this module is a strict Pydantic v2 model.
LLM outputs are constrained to Literal types (no free-form floats)
to maximise determinism across runs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GeoSource(str, Enum):
    """Where a candidate's coordinates came from."""

    MANUAL = "MANUAL"
    PIN_EXACT = "PIN_EXACT"
    PIN_DISTRICT = "PIN_DISTRICT"
    NOMINATIM = "NOMINATIM"
    STATE_CENTROID = "STATE_CENTROID"
    NONE = "NONE"


class EnrichmentSource(str, Enum):
    """Where verification evidence text came from."""

    FIRECRAWL = "FIRECRAWL"
    MOCK = "MOCK"
    NONE = "NONE"


class CandidateDisposition(str, Enum):
    """Deterministic outcome of the pre-filtering stage."""

    ELIGIBLE = "ELIGIBLE"
    NEEDS_DATA = "NEEDS_DATA"
    REJECTED = "REJECTED"


class RecommendationState(str, Enum):
    """Commercial queue to which a scored candidate belongs."""

    VERIFIED_RECOMMENDATION = "VERIFIED_RECOMMENDATION"
    PRIORITIZED_LEAD = "PRIORITIZED_LEAD"
    REJECTED = "REJECTED"


class StrictLLMModel(BaseModel):
    """Strict contract for model-produced structured data.

    Input records remain permissive enough to normalize messy CSV exports.
    Model output must not silently coerce booleans or discard unexpected
    fields, because those values influence commercial recommendations.
    """

    model_config = ConfigDict(extra="forbid", strict=True)


class EvidenceCitation(StrictLLMModel):
    """A model claim tied to an immutable source record."""

    source_id: str = ""
    quote: str = ""


class ToTBranchOutput(StrictLLMModel):
    """One explicit branch of the single-call Tree of Thoughts analysis."""

    conclusion: str = ""
    confidence: Literal["STRONG", "MODERATE", "WEAK", "NONE"] = "NONE"
    citations: List[EvidenceCitation] = Field(default_factory=list)
    falsification: str = ""


class ToTBranches(StrictLLMModel):
    """Independent perspectives that Stage 2 must reconcile."""

    brand_alignment: ToTBranchOutput = Field(default_factory=ToTBranchOutput)
    infrastructure: ToTBranchOutput = Field(default_factory=ToTBranchOutput)
    onboarding_risk: ToTBranchOutput = Field(default_factory=ToTBranchOutput)


class Stage2ThinkingOutput(StrictLLMModel):
    """Adversarial reconciliation of the nested ToT branches."""

    supported_claims: List[str] = Field(default_factory=list)
    challenged_claims: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    final_rationale: str = ""


class EvidenceSource(BaseModel):
    """A retained web result supplied to the model as untrusted evidence."""

    source_id: str
    url: str = ""
    title: str = ""
    content: str = ""


class EnrichmentResult(BaseModel):
    """Source-preserving result of one enrichment request."""

    source: EnrichmentSource = EnrichmentSource.NONE
    sources: List[EvidenceSource] = Field(default_factory=list)
    error: Optional[str] = None


class EvidenceValidation(BaseModel):
    """Deterministic validation of model citations against retained sources."""

    source_count: int = 0
    cited_source_ids: List[str] = Field(default_factory=list)
    valid_citation_count: int = 0
    invalid_citations: List[str] = Field(default_factory=list)
    branch_citation_counts: dict[str, int] = Field(default_factory=dict)
    structured_tot_complete: bool = False
    verified: bool = False


# ────────────────────────────────────────────────────────
# Input Models
# ────────────────────────────────────────────────────────


class TataDepot(BaseModel):
    """A Tata Steel plant, mother depot, or consignment agent location."""

    name: str
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)


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
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    geocode_source: GeoSource = GeoSource.NONE

    # ── Turnover ───────────────────────────────────────
    turnover_slab: Optional[str] = None  # Raw slab string from CSV
    turnover_midpoint_cr: float = Field(default=0.0, ge=0.0)  # Midpoint of slab in ₹ Crores
    turnover_known: bool = False

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
    disposition: CandidateDisposition = CandidateDisposition.REJECTED
    passed: bool
    rejection_reasons: List[str] = Field(default_factory=list)
    needs_data_reasons: List[str] = Field(default_factory=list)
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


class S2TVerificationOutput(StrictLLMModel):
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
    tot: ToTBranches = Field(
        default_factory=ToTBranches,
        description="Nested ToT branch assessments, each with citation IDs and falsification.",
    )
    stage_2: Stage2ThinkingOutput = Field(
        default_factory=Stage2ThinkingOutput,
        description="Stage 2 adversarial reconciliation of the three ToT branches.",
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
    verification_model: Optional[str] = None
    # VERIFIED = LLM returned evidence beyond NONE; PARTIAL = LLM ran with
    # no real enrichment; UNVERIFIED = no LLM evidence; FAILED = all
    # providers failed.
    verification_status: Literal["VERIFIED", "PARTIAL", "UNVERIFIED", "FAILED"] = "UNVERIFIED"
    evidence_validation: EvidenceValidation = Field(default_factory=EvidenceValidation)
    recommendation_state: RecommendationState = RecommendationState.PRIORITIZED_LEAD
    action: str = "Field verification required"

    # ── Metadata ───────────────────────────────────────
    rejection_reasons: List[str] = Field(default_factory=list)
    turnover_slab: Optional[str] = None
    permanent_address: Optional[str] = None
    ekyc_status: Optional[str] = None
    aadhaar_verified: Optional[str] = None


class CandidateAuditRecord(BaseModel):
    """Audit record for a candidate that never reaches commercial scoring."""

    candidate: DistributorCandidate
    disposition: CandidateDisposition
    reasons: List[str] = Field(default_factory=list)
    coarse_road_distance_km: Optional[float] = None
    nearest_existing_distributor_km: Optional[float] = None


class RunManifest(BaseModel):
    """Reproducibility metadata for one pipeline execution."""

    created_at: str
    depot: TataDepot
    distributor_csv: str
    distributor_csv_sha256: str
    existing_tata_csv: Optional[str] = None
    existing_tata_csv_sha256: Optional[str] = None
    code_version: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)


class PipelineRunResult(BaseModel):
    """All queues and reproducibility data produced by one pipeline run."""

    evaluations: List[FinalEvaluation] = Field(default_factory=list)
    needs_data: List[CandidateAuditRecord] = Field(default_factory=list)
    rejected: List[CandidateAuditRecord] = Field(default_factory=list)
    manifest: Optional[RunManifest] = None

    # Preserve iteration over scored evaluations for callers migrating from the
    # earlier list-returning run_pipeline API.
    def __iter__(self):
        return iter(self.evaluations)

    def __len__(self) -> int:
        return len(self.evaluations)

    def __getitem__(self, index: int) -> FinalEvaluation:
        return self.evaluations[index]
