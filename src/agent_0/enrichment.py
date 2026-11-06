"""Web enrichment module for Agent 0 using Firecrawl with disk caching.

This module provides web search and scraping integration via Firecrawl's API
to enrich distributor candidates with online presence, infrastructure, and
brand association intelligence.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from agent_0.models import DistributorCandidate

logger = logging.getLogger(__name__)

# Marker used for enrichment payloads that carry no real evidence. These are
# never cached (they would otherwise be served forever) and are never handed
# to the LLM as if they were scraped source material.
NO_EVIDENCE_MARKERS = (
    "[MOCK]",
    "[FIRECRAWL] Web search completed for",
)


def _is_no_evidence(text: str) -> bool:
    """True when the payload contains no genuine scraped content."""
    stripped = text.strip()
    if not stripped:
        return True
    return stripped.startswith(NO_EVIDENCE_MARKERS)


def is_no_evidence(text: str) -> bool:
    """Public wrapper: True when the payload contains no genuine scraped content."""
    return _is_no_evidence(text)


def _cache_entry_is_fresh(cached_data: Dict[str, Any], ttl_days: int) -> bool:
    """Check whether a cache entry is still usable.

    Entries with no real evidence are always considered stale. Otherwise the
    age is compared against the TTL (0 disables expiry).
    """
    if not isinstance(cached_data, dict):
        return False
    if _is_no_evidence(str(cached_data.get("text", ""))):
        return False
    if ttl_days <= 0:
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
    age = datetime.now(timezone.utc) - cached_at
    return age <= timedelta(days=ttl_days)


def _post_firecrawl_http(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    """Execute HTTP POST request to Firecrawl API using requests or urllib."""
    payload_bytes = json.dumps(payload).encode("utf-8")

    if requests is not None:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def enrich_entity(
    company_name: str,
    gstin: str,
    state: str,
    firecrawl_api_key: str,
    cache_dir: str = ".agent0_cache",
    cache_ttl_days: int = 30,
) -> Tuple[str, str]:
    """Enrich candidate entity with Firecrawl web search results and caching.

    Args:
        company_name: Entity legal or trade name.
        gstin: 15-character Goods and Services Tax Identification Number.
        state: Indian state of entity operations.
        firecrawl_api_key: API authorization key for Firecrawl.
        cache_dir: Directory where JSON cache files are persisted.
        cache_ttl_days: Maximum age of a cache entry before it is refetched.
            0 disables expiry.

    Returns:
        Tuple of (scraped_text, source) where source is 'FIRECRAWL' or 'MOCK'.
    """
    cache_path = Path(cache_dir)
    cache_file = cache_path / f"{gstin}.json"

    # 1. Check local cache (honouring the TTL)
    if cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
            if _cache_entry_is_fresh(cached_data, cache_ttl_days):
                cached_text = cached_data.get("text", "")
                cached_source = cached_data.get("source", "FIRECRAWL")
                logger.info("Loaded cached enrichment data for GSTIN %s", gstin)
                return cached_text, cached_source
            logger.info("Cache entry for GSTIN %s is stale; refetching", gstin)
        except Exception as exc:
            logger.warning(
                "Failed to read cache file %s for GSTIN %s: %s",
                cache_file,
                gstin,
                exc,
            )

    # 2. Check for mock mode
    if not firecrawl_api_key or not firecrawl_api_key.strip():
        mock_text = (
            f"[MOCK] No web data available for {company_name} ({gstin}). "
            "Scoring will rely on CSV data only."
        )
        return mock_text, "MOCK"

    # 3. Firecrawl API integration
    endpoint = "https://api.firecrawl.dev/v1/search"
    headers = {
        "Authorization": f"Bearer {firecrawl_api_key.strip()}",
        "Content-Type": "application/json",
    }
    query = (
        f'"{company_name}" OR "{gstin}" steel distributor dealer warehouse '
        f"stockyard {state}"
    )
    payload: Dict[str, Any] = {
        "query": query,
        "limit": 5,
        "scrapeOptions": {"formats": ["markdown"]},
    }

    try:
        data = _post_firecrawl_http(
            url=endpoint,
            headers=headers,
            payload=payload,
            timeout_seconds=20.0,
        )

        markdown_chunks: List[str] = []
        raw_items = data.get("data", []) if isinstance(data, dict) else []
        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, dict):
                    md = (
                        item.get("markdown")
                        or item.get("content")
                        or item.get("description")
                        or ""
                    )
                    if isinstance(md, str) and md.strip():
                        markdown_chunks.append(md.strip())

        joined_text = "\n---\n".join(markdown_chunks)
        scraped_text = joined_text[:5000]

        if not scraped_text.strip():
            # No usable content. Do NOT cache this placeholder: caching it
            # would make the candidate permanently "enriched" with nothing.
            logger.warning(
                "Firecrawl returned no extractable text for %s (%s)", company_name, gstin
            )
            placeholder = (
                f"[FIRECRAWL] Web search completed for {company_name} ({gstin}), "
                "but no textual content could be extracted from results."
            )
            return placeholder, "MOCK"

        # Write to cache (only genuine evidence is persisted)
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_payload = {
                "gstin": gstin,
                "text": scraped_text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "FIRECRAWL",
            }
            cache_file.write_text(
                json.dumps(cache_payload, indent=2),
                encoding="utf-8",
            )
        except Exception as cache_exc:
            logger.warning(
                "Failed to write cache for GSTIN %s: %s",
                gstin,
                cache_exc,
            )

        return scraped_text, "FIRECRAWL"

    except Exception as exc:
        logger.error(
            "Firecrawl enrichment failed for %s (%s): %s",
            company_name,
            gstin,
            exc,
            exc_info=True,
        )
        fallback_text = (
            f"[MOCK] Web search failed for {company_name} ({gstin}): {exc}. "
            "Scoring will rely on CSV data only."
        )
        return fallback_text, "MOCK"


def batch_enrich(
    candidates: List[DistributorCandidate],
    firecrawl_api_key: str,
    cache_dir: str = ".agent0_cache",
    max_workers: int = 3,
    cache_ttl_days: int = 30,
) -> Dict[str, Tuple[str, str]]:
    """Enrich multiple distributor candidates concurrently with progress tracking.

    Args:
        candidates: List of DistributorCandidate records to enrich.
        firecrawl_api_key: API authorization key for Firecrawl.
        cache_dir: Directory where JSON cache files are persisted.
        max_workers: Maximum thread worker pool concurrency.
        cache_ttl_days: Maximum age of a cache entry before it is refetched.

    Returns:
        Mapping of GSTIN to (scraped_text, source).
    """
    results: Dict[str, Tuple[str, str]] = {}
    if not candidates:
        return results

    def _worker(candidate: DistributorCandidate) -> Tuple[str, Tuple[str, str]]:
        company_name = (
            candidate.display_name
            or candidate.trade_name
            or candidate.legal_name
        )
        res = enrich_entity(
            company_name=company_name,
            gstin=candidate.gstin,
            state=candidate.state_name,
            firecrawl_api_key=firecrawl_api_key,
            cache_dir=cache_dir,
            cache_ttl_days=cache_ttl_days,
        )
        return candidate.gstin, res

    total_count = len(candidates)

    if Progress is not None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        with progress:
            task_id = progress.add_task(
                f"[cyan]Enriching {total_count} candidates...",
                total=total_count,
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_cand = {
                    executor.submit(_worker, cand): cand for cand in candidates
                }
                for future in as_completed(future_to_cand):
                    cand = future_to_cand[future]
                    try:
                        gstin, enrichment = future.result()
                        results[gstin] = enrichment
                    except Exception as exc:
                        logger.error(
                            "Unexpected batch worker error for candidate %s: %s",
                            cand.gstin,
                            exc,
                        )
                        results[cand.gstin] = (
                            f"[MOCK] Enrichment error: {exc}",
                            "MOCK",
                        )
                    progress.advance(task_id, 1)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cand = {
                executor.submit(_worker, cand): cand for cand in candidates
            }
            for future in as_completed(future_to_cand):
                cand = future_to_cand[future]
                try:
                    gstin, enrichment = future.result()
                    results[gstin] = enrichment
                except Exception as exc:
                    logger.error(
                        "Unexpected batch worker error for candidate %s: %s",
                        cand.gstin,
                        exc,
                    )
                    results[cand.gstin] = (
                        f"[MOCK] Enrichment error: {exc}",
                        "MOCK",
                    )

    return results
