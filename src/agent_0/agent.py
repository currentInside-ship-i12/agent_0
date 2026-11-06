"""LLM evaluation module implementing Tree of Thoughts and Stage 2 Thinking.

Provides deterministic candidate evaluation using Anthropic Claude (primary)
and OpenAI (secondary/fallback) structured outputs for Tata Steel GTM intelligence.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

import anthropic
import openai

from agent_0.config import PipelineConfig
from agent_0.models import DistributorCandidate, S2TVerificationOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Agent 0 for Tata Steel GTM intelligence. Your task is to evaluate distributor candidates using ToT + S2T. Be deterministic. Never speculate beyond evidence. The Tata Steel context: Tata Steel is India's largest steel producer manufacturing flat products (HR/CR coils, galvanized sheets), long products (TMT bars/rebars like Tiscon, wire rods), and tubes & pipes. They sell through a distributor network across India. We are looking for new distributors to expand the GTM channel.

Execute evaluation strictly through the following framework:

### Tree of Thoughts (ToT): 3 Parallel Evaluation Branches
1. Branch 1 (Brand Alignment):
   - What steel brands does this entity currently distribute or stock (e.g., JSW, SAIL, Jindal, AM/NS, Vizag Steel, local rerollers)?
   - Are they exclusive to a competitor (e.g., dedicated JSW Shoppe / Jindal steel center)?
   - How feasible is switching or adding Tata Steel? (HIGH = unbranded/multi-brand; MEDIUM = tier-2 competitor dealer; LOW = competitor exclusive).

2. Branch 2 (Infrastructure & Processing):
   - Does the entity possess a dedicated physical stockyard, warehouse, or depot?
   - Does it possess steel processing equipment (slitting, cut-to-length / CTL lines, decoiling, overhead gantry cranes, shears)?
   - Rate processing capability: ADVANCED (CTL/slitting/decoiling), BASIC (stockyard with crane/manual handling), or NONE (office-only/trader).

3. Branch 3 (Onboarding Risk & Compliance):
   - Is this a genuine primary steel dealer/stockist, or a scrap trader, logistics broker, or paper company?
   - Are there litigation flags, director disqualifications, tax disputes, or irregular GST filing signals?

### Stage 2 Thinking (S2T): Evidence Grounding & Falsification
- Step 1 (Evidence Grounding): Ground all claims in verbatim evidence and citations from the provided scraped web text and statutory profile. Do not extrapolate.
- Step 2 (Adversarial Falsification): Actively attempt to falsify and disprove claims. Analyze how this candidate could be a false positive (e.g., identical name collision, scrap dealer pretending to be primary stockist, purely logistics provider).
- Step 3 (Strict Output): Formulate and emit the final structured verification decision.
"""


def _build_user_prompt(
    candidate: DistributorCandidate,
    scraped_content: str,
) -> str:
    """Construct structured user prompt containing candidate data and scraped web text."""
    hsn_codes_str = ", ".join(candidate.hsn_codes) if candidate.hsn_codes else "None"
    hsn_desc_str = (
        "; ".join(candidate.hsn_descriptions)
        if candidate.hsn_descriptions
        else "None"
    )
    sac_codes_str = ", ".join(candidate.sac_codes) if candidate.sac_codes else "None"
    activities_str = (
        ", ".join(candidate.business_activities)
        if candidate.business_activities
        else "None"
    )
    members_str = (
        ", ".join(candidate.member_names) if candidate.member_names else "None"
    )

    return f"""### Distributor Candidate Dossier
- Legal Name: {candidate.legal_name}
- Trade Name: {candidate.trade_name or "N/A"}
- GSTIN: {candidate.gstin}
- Constitution: {candidate.constitution or "N/A"}
- Nature of Core Business: {candidate.nature_of_core_business or "N/A"}
- State: {candidate.state_name}
- Registered Address: {candidate.permanent_address or "N/A"}
- Turnover Slab: {candidate.turnover_slab or "N/A"}
- Promoters / Members: {members_str}
- HSN Codes: {hsn_codes_str}
- HSN Descriptions: {hsn_desc_str}
- SAC Codes: {sac_codes_str}
- Declared Business Activities: {activities_str}

### Scraped Web & OSINT Intelligence Content:
\"\"\"
{scraped_content.strip()}
\"\"\"

Conduct your Tree of Thoughts (Branches 1-3) followed by Stage 2 Thinking (Evidence Grounding + Adversarial Falsification), and emit the structured verification output.
"""


def _get_safe_default_output(reason: str) -> S2TVerificationOutput:
    """Return conservative safe default verification output upon failure."""
    return S2TVerificationOutput(
        is_primary_steel_dealer=True,  # Benefit of doubt
        brands_currently_handled=[],
        is_exclusive_to_competitor=False,
        switching_feasibility="MEDIUM",
        has_physical_stockyard=False,
        processing_capability="NONE",
        promoter_litigation_flag=False,
        gst_filing_regularity="UNKNOWN",
        evidence_confidence="NONE",
        verbatim_evidence=f"LLM call failed. Using safe defaults. ({reason})",
        falsification_notes="No verification performed.",
    )


def _prepare_openai_strict_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure JSON schema conforms strictly to OpenAI structured output requirements."""
    schema_copy = copy.deepcopy(schema)

    def _make_strict(subschema: Dict[str, Any]) -> None:
        if not isinstance(subschema, dict):
            return
        if subschema.get("type") == "object":
            subschema["additionalProperties"] = False
            props = subschema.get("properties", {})
            subschema["required"] = list(props.keys())
            for prop in props.values():
                _make_strict(prop)
        elif subschema.get("type") == "array":
            items = subschema.get("items")
            if isinstance(items, dict):
                _make_strict(items)
        if "$defs" in subschema:
            for def_val in subschema["$defs"].values():
                _make_strict(def_val)

    _make_strict(schema_copy)
    return schema_copy


def _run_claude(
    candidate: DistributorCandidate,
    scraped_content: str,
    config: PipelineConfig,
) -> S2TVerificationOutput:
    """Execute ToT + S2T verification using Anthropic Claude API."""
    extra_headers: dict[str, str] = {}
    workspace_id = (config.anthropic_workspace_id or "").strip()
    if workspace_id:
        # Enterprise keys not scoped to a workspace are rejected (400)
        # unless this header identifies the workspace to bill against.
        extra_headers["anthropic-workspace-id"] = workspace_id
    client = anthropic.Anthropic(
        api_key=config.anthropic_api_key, default_headers=extra_headers
    )
    user_prompt = _build_user_prompt(candidate, scraped_content)

    tool_definition = {
        "name": "emit_verification",
        "description": "Emit strict structured S2T verification output for distributor candidate.",
        "input_schema": S2TVerificationOutput.model_json_schema(),
    }

    response = client.messages.create(
        model=config.llm.model,
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[tool_definition],
        tool_choice={"type": "tool", "name": "emit_verification"},
    )

    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "emit_verification"
        ):
            tool_input = getattr(block, "input", {})
            return S2TVerificationOutput.model_validate(tool_input)

    raise ValueError("Claude response did not contain expected emit_verification tool call")


def _run_openai(
    candidate: DistributorCandidate,
    scraped_content: str,
    config: PipelineConfig,
) -> S2TVerificationOutput:
    """Execute ToT + S2T verification using OpenAI API."""
    client = openai.OpenAI(api_key=config.openai_api_key)
    user_prompt = _build_user_prompt(candidate, scraped_content)

    model_name = config.llm.model
    # If config model still points to Claude default, fall back to gpt-4o for OpenAI
    if not model_name or "claude" in model_name.lower():
        model_name = "gpt-4o"

    schema = _prepare_openai_strict_schema(S2TVerificationOutput.model_json_schema())

    response = client.chat.completions.create(
        model=model_name,
        temperature=config.llm.temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "s2t_verification",
                "schema": schema,
                "strict": True,
            },
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI completion returned empty content")

    return S2TVerificationOutput.model_validate_json(content)


# Gemini is reached through its OpenAI-compatible endpoint, so no new
# SDK dependency is required (reuses the installed `openai` package).
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Default model is set by the live API itself: gemini-2.0-flash was retired
# (404: "no longer available ... use models/gemini-3.8-flash").
GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"


def _run_gemini(
    candidate: DistributorCandidate,
    scraped_content: str,
    config: PipelineConfig,
) -> S2TVerificationOutput:
    """Execute ToT + S2T verification using Google Gemini API.

    Uses Gemini's OpenAI-compatible chat-completions endpoint with the same
    strict structured-output contract as the OpenAI path. If Gemini ever
    rejects strict ``json_schema`` mode, the dispatcher's safe-default
    fallback engages and the error is recorded in verbatim evidence.
    """
    client = openai.OpenAI(
        api_key=config.gemini_api_key,
        base_url=GEMINI_BASE_URL,
        # Bound the worst case: the pipeline loop is sequential, so one
        # hanging call would stall the whole batch. Timeouts fall through
        # to the safe-default fallback like any other API error.
        timeout=120.0,
    )
    user_prompt = _build_user_prompt(candidate, scraped_content)

    model_name = config.llm.model
    # If config model still points elsewhere, fall back to the Gemini default.
    if not model_name or "gemini" not in model_name.lower():
        model_name = GEMINI_DEFAULT_MODEL

    schema = _prepare_openai_strict_schema(S2TVerificationOutput.model_json_schema())

    response = client.chat.completions.create(
        model=model_name,
        temperature=config.llm.temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "s2t_verification",
                "schema": schema,
                "strict": True,
            },
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Gemini completion returned empty content")

    return S2TVerificationOutput.model_validate_json(content)


def _dispatch_provider(
    provider: str,
    candidate: DistributorCandidate,
    scraped_content: str,
    config: PipelineConfig,
) -> S2TVerificationOutput:
    """Route a verification request to the implementation for `provider`."""
    if provider == "anthropic":
        return _run_claude(candidate, scraped_content, config)
    if provider == "openai":
        return _run_openai(candidate, scraped_content, config)
    if provider == "gemini":
        return _run_gemini(candidate, scraped_content, config)
    raise ValueError(f"Unknown LLM provider '{provider}'")


def _providers_with_keys(config: PipelineConfig) -> list[str]:
    """Return the LLM providers that have a usable API key, in fallback order."""
    keyed = [
        ("anthropic", config.anthropic_api_key),
        ("openai", config.openai_api_key),
        ("gemini", config.gemini_api_key),
    ]
    return [name for name, key in keyed if key and key.strip()]


def run_tot_s2t(
    candidate: DistributorCandidate,
    scraped_content: str,
    config: PipelineConfig,
    meta: Optional[Dict[str, Any]] = None,
) -> S2TVerificationOutput:
    """Run Tree of Thoughts and Stage 2 Thinking evaluation for a distributor candidate.

    Dispatches to Anthropic Claude, OpenAI, or Google Gemini based on
    config.llm.provider. If the primary provider fails (quota, outage,
    malformed response) the remaining providers with a configured key are
    tried in turn, so a single provider failure degrades one candidate
    rather than the entire run. If every provider fails, a safe default is
    returned.

    Args:
        candidate: DistributorCandidate record.
        scraped_content: Enriched web/OSINT text.
        config: Pipeline configuration containing LLM parameters and API keys.
        meta: Optional dict that receives run metadata: "provider" (the
            provider that actually served the request) and "fallback"
            (True when it was not the configured provider).

    Returns:
        Structured S2TVerificationOutput.
    """
    requested = (config.llm.provider or "anthropic").strip().lower()

    # Primary provider first, then any other provider with a usable key.
    ordered: List[str] = [requested]
    for provider in _providers_with_keys(config):
        if provider != requested:
            ordered.append(provider)

    last_exc: Optional[Exception] = None

    for provider in ordered:
        try:
            result = _dispatch_provider(provider, candidate, scraped_content, config)
        except Exception as exc:
            last_exc = exc
            logger.error(
                "run_tot_s2t failed for candidate %s (%s) with provider %s: %s",
                candidate.gstin,
                candidate.legal_name,
                provider,
                exc,
                exc_info=True,
            )
            if provider == requested:
                logger.warning(
                    "Primary provider '%s' failed for %s; falling back to other "
                    "configured providers.",
                    provider,
                    candidate.gstin,
                )
            continue

        if meta is not None:
            meta["provider"] = provider
            meta["fallback"] = provider != requested
        return result

    reason = f"{requested}: {last_exc}" if last_exc else f"{requested}: no configured provider"
    return _get_safe_default_output(reason)
