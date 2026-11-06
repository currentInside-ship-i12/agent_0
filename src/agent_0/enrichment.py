"""Source-preserving Firecrawl enrichment for Agent 0.

Search results are retained as individual evidence records instead of being
flattened into an anonymous text blob. This lets the LLM cite stable source
IDs and lets deterministic code validate those citations before a candidate is
called verified.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
except ImportError:
    Progress = None  # type: ignore[assignment]

from agent_0.models import (
    DistributorCandidate,
    EnrichmentResult,
    EnrichmentSource,
    EvidenceSource,
    EvidenceValidation,
    S2TVerificationOutput,
)

logger = logging.getLogger(__name__)

NO_EVIDENCE_MARKERS = (
    "[MOCK]",
    "[FIRECRAWL] Web search completed for",
)
CACHE_SCHEMA_VERSION = 2


def _is_no_evidence(text: str) -> bool:
    """True when a legacy text payload contains no genuine scraped content."""
    stripped = text.strip()
    return not stripped or stripped.startswith(NO_EVIDENCE_MARKERS)


def is_no_evidence(text: str) -> bool:
    """Public compatibility wrapper for legacy text callers."""
    return _is_no_evidence(text)


def _has_real_sources(result: EnrichmentResult) -> bool:
    return bool(result.sources and any(source.content.strip() for source in result.sources))


def _cache_filename(gstin: str) -> str:
    """Return a traversal-safe cache filename without exposing raw GSTINs."""
    digest = hashlib.sha256(gstin.strip().upper().encode("utf-8")).hexdigest()
    return f"{digest}.json"


def _cache_entry_is_fresh(cached_data: dict[str, Any], ttl_days: int) -> bool:
    """Return whether a source-preserving cache entry is still usable."""
    if (
        not isinstance(cached_data, dict)
        or cached_data.get("schema_version") != CACHE_SCHEMA_VERSION
    ):
        return False
    sources = cached_data.get("sources")
    if not isinstance(sources, list) or not any(
        isinstance(source, dict) and str(source.get("content", "")).strip() for source in sources
    ):
        return False
    if ttl_days == 0:
        return True

    timestamp = cached_data.get("timestamp")
    if not timestamp:
        return False
    try:
        cached_at = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return False
    if cached_at.tzinfo is None:
        cached_at = cached_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - cached_at <= timedelta(days=ttl_days)


def _post_firecrawl_http(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Execute a Firecrawl search request using requests or urllib."""
    if requests is not None:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_sources(
    response_data: dict[str, Any],
    max_chars_per_source: int,
) -> list[EvidenceSource]:
    """Convert Firecrawl v1/v2 search response items into bounded source records."""
    raw_items = response_data.get("data") or response_data.get("web") or []
    if not isinstance(raw_items, list):
        return []

    sources: list[EvidenceSource] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        content = item.get("markdown") or item.get("content") or item.get("description") or ""
        if not isinstance(content, str) or not content.strip():
            continue
        url = str(item.get("url") or metadata.get("sourceURL") or metadata.get("url") or "")
        title = str(item.get("title") or metadata.get("title") or "")
        sources.append(
            EvidenceSource(
                source_id=f"firecrawl-{index}",
                url=url,
                title=title,
                content=content.strip()[:max_chars_per_source],
            )
        )
    return sources


def _load_cached_result(cache_file: Path, cache_ttl_days: int) -> EnrichmentResult | None:
    """Load a valid cache entry or return None when it should be refreshed."""
    if not cache_file.exists():
        return None
    try:
        cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
        if not _cache_entry_is_fresh(cached_data, cache_ttl_days):
            return None
        return EnrichmentResult(
            source=EnrichmentSource(cached_data.get("source", EnrichmentSource.FIRECRAWL)),
            sources=[EvidenceSource.model_validate(source) for source in cached_data["sources"]],
        )
    except Exception as exc:
        logger.warning("Failed to read enrichment cache %s: %s", cache_file, exc)
        return None


def _write_cached_result(cache_file: Path, gstin: str, result: EnrichmentResult) -> None:
    """Persist only genuine source records in the versioned cache format."""
    if not _has_real_sources(result):
        return
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "gstin": gstin,
            "source": result.source.value,
            "sources": [source.model_dump(mode="json") for source in result.sources],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write enrichment cache %s: %s", cache_file, exc)


def enrich_entity(
    company_name: str,
    gstin: str,
    state: str,
    firecrawl_api_key: str,
    cache_dir: str = ".agent0_cache",
    cache_ttl_days: int = 30,
    max_chars_per_source: int = 3000,
    max_retries: int = 1,
) -> EnrichmentResult:
    """Discover public evidence for one candidate through Firecrawl search.

    The product needs public pages about distributor identity, brands,
    infrastructure, and operations. Search is the appropriate endpoint because
    the workflow starts with a company/GSTIN query rather than a known URL.
    """
    cache_path = Path(cache_dir)
    cache_file = cache_path / _cache_filename(gstin)
    cached_result = _load_cached_result(cache_file, cache_ttl_days)
    if cached_result is not None:
        logger.info("Loaded cached enrichment data for GSTIN %s", gstin)
        return cached_result

    if not firecrawl_api_key or not firecrawl_api_key.strip():
        return EnrichmentResult(
            source=EnrichmentSource.MOCK,
            error="Firecrawl API key is not configured",
        )

    endpoint = "https://api.firecrawl.dev/v1/search"
    headers = {
        "Authorization": f"Bearer {firecrawl_api_key.strip()}",
        "Content-Type": "application/json",
    }
    query = f'"{company_name}" OR "{gstin}" steel distributor dealer warehouse stockyard {state}'
    payload: dict[str, Any] = {
        "query": query,
        "limit": 5,
        "scrapeOptions": {"formats": ["markdown"]},
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            data = _post_firecrawl_http(
                url=endpoint,
                headers=headers,
                payload=payload,
                timeout_seconds=20.0,
            )
            sources = _extract_sources(data, max_chars_per_source)
            if not sources:
                logger.warning(
                    "Firecrawl returned no extractable sources for %s (%s)",
                    company_name,
                    gstin,
                )
                return EnrichmentResult(
                    source=EnrichmentSource.MOCK,
                    error="Firecrawl returned no extractable text",
                )
            result = EnrichmentResult(source=EnrichmentSource.FIRECRAWL, sources=sources)
            _write_cached_result(cache_file, gstin, result)
            return result
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                delay = 2**attempt
                logger.warning(
                    "Firecrawl attempt %d/%d failed for %s; retrying in %ds: %s",
                    attempt + 1,
                    max_retries + 1,
                    gstin,
                    delay,
                    exc,
                )
                time.sleep(delay)

    logger.error(
        "Firecrawl enrichment failed for %s (%s): %s",
        company_name,
        gstin,
        last_error,
    )
    return EnrichmentResult(
        source=EnrichmentSource.MOCK,
        error=f"Firecrawl search failed: {last_error}",
    )


def batch_enrich(
    candidates: list[DistributorCandidate],
    firecrawl_api_key: str,
    cache_dir: str = ".agent0_cache",
    max_workers: int = 3,
    cache_ttl_days: int = 30,
    max_chars_per_source: int = 3000,
    max_retries: int = 1,
) -> dict[str, EnrichmentResult]:
    """Enrich candidates concurrently while preserving source-level provenance."""
    results: dict[str, EnrichmentResult] = {}
    if not candidates:
        return results

    def worker(candidate: DistributorCandidate) -> tuple[str, EnrichmentResult]:
        company_name = candidate.display_name or candidate.trade_name or candidate.legal_name
        result = enrich_entity(
            company_name=company_name,
            gstin=candidate.gstin,
            state=candidate.state_name,
            firecrawl_api_key=firecrawl_api_key,
            cache_dir=cache_dir,
            cache_ttl_days=cache_ttl_days,
            max_chars_per_source=max_chars_per_source,
            max_retries=max_retries,
        )
        return candidate.gstin, result

    total_count = len(candidates)
    progress_context = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        if Progress is not None
        else None
    )

    if progress_context is not None:
        with progress_context:
            task_id = progress_context.add_task(
                f"[cyan]Enriching {total_count} candidates...",
                total=total_count,
            )
            _collect_batch_results(
                candidates,
                worker,
                max_workers,
                results,
                lambda: progress_context.advance(task_id, 1),
            )
    else:
        _collect_batch_results(candidates, worker, max_workers, results, lambda: None)

    return results


def _collect_batch_results(
    candidates: list[DistributorCandidate],
    worker: Callable[[DistributorCandidate], tuple[str, EnrichmentResult]],
    max_workers: int,
    results: dict[str, EnrichmentResult],
    on_complete: Callable[[], None],
) -> None:
    """Execute workers once; shared by rich and non-rich progress paths."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_candidate = {
            executor.submit(worker, candidate): candidate for candidate in candidates
        }
        for future in as_completed(future_to_candidate):
            candidate = future_to_candidate[future]
            try:
                gstin, enrichment = future.result()
                results[gstin] = enrichment
            except Exception as exc:
                logger.error(
                    "Unexpected enrichment worker error for candidate %s: %s",
                    candidate.gstin,
                    exc,
                )
                results[candidate.gstin] = EnrichmentResult(
                    source=EnrichmentSource.MOCK,
                    error=f"Unexpected enrichment worker error: {exc}",
                )
            on_complete()


def _normalized_text(value: str) -> str:
    """Normalize whitespace and quotes for deterministic excerpt matching."""
    return " ".join(value.replace("“", '"').replace("”", '"').split()).casefold()


def validate_evidence_citations(
    verification: S2TVerificationOutput,
    sources: list[EvidenceSource],
) -> EvidenceValidation:
    """Validate nested ToT citations against immutable source content.

    A model may summarize evidence, but it cannot promote an unsupported quote
    into a verified commercial recommendation. Verified evidence requires two
    valid citations spanning the brand and infrastructure branches plus a
    complete Stage 2 reconciliation.
    """
    sources_by_id = {source.source_id: source for source in sources}
    branch_outputs = {
        "brand_alignment": verification.tot.brand_alignment,
        "infrastructure": verification.tot.infrastructure,
        "onboarding_risk": verification.tot.onboarding_risk,
    }
    branch_citation_counts = {name: 0 for name in branch_outputs}
    cited_source_ids: list[str] = []
    invalid_citations: list[str] = []
    valid_citation_count = 0

    for branch_name, branch in branch_outputs.items():
        for citation in branch.citations:
            source = sources_by_id.get(citation.source_id)
            normalized_quote = _normalized_text(citation.quote)
            if not source:
                invalid_citations.append(f"{branch_name}: unknown source_id '{citation.source_id}'")
                continue
            if len(normalized_quote) < 12:
                invalid_citations.append(
                    f"{branch_name}: citation from '{citation.source_id}' is too short"
                )
                continue
            if normalized_quote not in _normalized_text(source.content):
                invalid_citations.append(
                    f"{branch_name}: quote not found in '{citation.source_id}'"
                )
                continue
            valid_citation_count += 1
            branch_citation_counts[branch_name] += 1
            if citation.source_id not in cited_source_ids:
                cited_source_ids.append(citation.source_id)

    structured_tot_complete = bool(
        verification.tot.brand_alignment.conclusion.strip()
        and verification.tot.infrastructure.conclusion.strip()
        and verification.tot.onboarding_risk.conclusion.strip()
        and verification.stage_2.final_rationale.strip()
    )
    verified = bool(
        sources
        and structured_tot_complete
        and valid_citation_count >= 2
        and branch_citation_counts["brand_alignment"] > 0
        and branch_citation_counts["infrastructure"] > 0
        and verification.evidence_confidence in {"MODERATE", "STRONG"}
    )
    return EvidenceValidation(
        source_count=len(sources),
        cited_source_ids=cited_source_ids,
        valid_citation_count=valid_citation_count,
        invalid_citations=invalid_citations,
        branch_citation_counts=branch_citation_counts,
        structured_tot_complete=structured_tot_complete,
        verified=verified,
    )
