"""Geocoding module for Distributor Candidates.

Assigns spatial coordinates (latitude and longitude) to distributor candidates
using GSTIN state centroids, PIN code caching, and optional Nominatim API
lookups with strict rate-limiting.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console
from rich.table import Table

from agent_0.models import DistributorCandidate, GeoSource

logger = logging.getLogger(__name__)
console = Console()

# ────────────────────────────────────────────────────────
# Offline PIN-code coordinate table (no network required)
#
# Exact 6-digit hits are preferred; otherwise the first 3 digits
# (postal sorting district) give an approximate location. This runs
# BEFORE the use_nominatim gate so extracted pincodes resolve to
# GeoSource.PINCODE instead of falling back to STATE_CENTROID.
# ────────────────────────────────────────────────────────

OFFLINE_PINCODE_COORDS: dict[str, tuple[float, float]] = {
    # Tata Steel hubs (exact)
    "831001": (22.8046, 86.2029),  # Jamshedpur (Sakchi)
    "831002": (22.7980, 86.1830),  # Jamshedpur (Bistupur)
    "755026": (20.9580, 86.0650),  # Kalinganagar (Jajpur, Odisha)
    "410208": (19.0330, 73.0980),  # Kalamboli (Navi Mumbai)
    "421302": (19.2960, 73.0650),  # Bhiwandi (Thane)
    # Major metros (exact anchors)
    "700001": (22.5720, 88.3500),  # Kolkata
    "400001": (18.9350, 72.8350),  # Mumbai (Fort)
    "411001": (18.5200, 73.8550),  # Pune
    "560001": (12.9780, 77.5720),  # Bengaluru
    "110001": (28.6320, 77.2190),  # New Delhi (Connaught Place)
    "380001": (23.0300, 72.5800),  # Ahmedabad
    "600001": (13.0820, 80.2750),  # Chennai
}

PIN_PREFIX_3_COORDS: dict[str, tuple[float, float]] = {
    "831": (22.8000, 86.2000),  # Jamshedpur district
    "755": (20.9500, 86.0600),  # Jajpur / Kalinganagar district
    "410": (19.0300, 73.1000),  # Navi Mumbai / Raigad district
    "421": (19.2900, 73.0600),  # Thane / Bhiwandi district
    "700": (22.5700, 88.3600),  # Kolkata district
    "400": (19.0700, 72.8700),  # Mumbai district
    "411": (18.5200, 73.8500),  # Pune district
    "560": (12.9700, 77.5900),  # Bengaluru district
    "110": (28.6100, 77.2000),  # Delhi district
    "380": (23.0200, 72.5700),  # Ahmedabad district
    "600": (13.0800, 80.2700),  # Chennai district
    "500": (17.3800, 78.4800),  # Hyderabad district
    # Haryana districts (observed in shortlist; district-level precision)
    "121": (28.4100, 77.3100),  # Faridabad district
    "122": (28.4600, 77.0300),  # Gurugram district
    "123": (28.1900, 76.6200),  # Rewari district
    "131": (28.9900, 77.0200),  # Sonipat district
    "132": (29.6900, 76.9800),  # Karnal district
    "134": (30.3800, 76.7800),  # Ambala district
}


def lookup_offline_pincode(pincode: str | None) -> tuple[float, float] | None:
    """Resolve a pincode offline via exact then 3-digit prefix tables."""
    if not pincode:
        return None
    clean = pincode.strip()
    if clean in OFFLINE_PINCODE_COORDS:
        return OFFLINE_PINCODE_COORDS[clean]
    if len(clean) >= 3 and clean[:3] in PIN_PREFIX_3_COORDS:
        return PIN_PREFIX_3_COORDS[clean[:3]]
    return None

# ────────────────────────────────────────────────────────
# GSTIN 2-digit State Code to Geographic Centroid Table
# ────────────────────────────────────────────────────────

GSTIN_STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "01": (34.08, 74.80),
    "02": (31.10, 77.17),
    "03": (31.15, 75.34),
    "04": (30.73, 76.78),
    "05": (30.07, 79.02),
    "06": (28.47, 76.10),
    "07": (28.61, 77.21),
    "08": (26.85, 75.76),
    "09": (26.85, 80.95),
    "10": (25.59, 85.14),
    "11": (27.53, 88.51),
    "12": (28.22, 94.73),
    "13": (26.16, 94.56),
    "14": (24.66, 93.91),
    "15": (23.16, 92.94),
    "16": (23.94, 91.99),
    "17": (25.47, 91.37),
    "18": (26.20, 92.94),
    "19": (22.99, 87.85),
    "20": (23.61, 85.28),
    "21": (20.94, 84.80),
    "22": (21.28, 81.87),
    "23": (24.63, 78.55),
    "24": (22.26, 71.19),
    "25": (20.42, 72.83),
    "26": (20.27, 73.01),
    "27": (19.66, 75.30),
    "28": (15.91, 79.74),
    "29": (15.32, 75.71),
    "30": (15.30, 74.09),
    "31": (10.57, 72.64),
    "32": (10.85, 76.27),
    "33": (11.13, 78.66),
    "34": (11.94, 79.81),
    "35": (11.74, 92.66),
    "36": (18.11, 79.02),
    "37": (15.91, 79.74),
    "38": (34.15, 77.58),
}

DEFAULT_CACHE_DIR = Path(".agent0_cache")
PINCODE_CACHE_FILE = DEFAULT_CACHE_DIR / "pincode_cache.json"


# ────────────────────────────────────────────────────────
# Cache Management
# ────────────────────────────────────────────────────────


def _load_pincode_cache(cache_file: Path = PINCODE_CACHE_FILE) -> dict[str, list[float]]:
    """Load cached pincode coordinates from local JSON file."""
    if cache_file.is_file():
        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            logger.warning("Failed to read pincode cache at %s: %s", cache_file, e)
    return {}


def _save_pincode_cache(
    cache: dict[str, list[float]],
    cache_file: Path = PINCODE_CACHE_FILE,
) -> None:
    """Save cached pincode coordinates to local JSON file."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save pincode cache to %s: %s", cache_file, e)


# ────────────────────────────────────────────────────────
# Nominatim Geocoding
# ────────────────────────────────────────────────────────


def geocode_pincode_nominatim(
    pincode: str,
    cache: dict[str, list[float]],
    cache_file: Path = PINCODE_CACHE_FILE,
) -> Optional[tuple[float, float]]:
    """Lookup latitude/longitude for an Indian 6-digit PIN code via Nominatim.

    Uses caching to avoid repeated network calls. Applies a strict 1-second
    rate-limit between live API requests per OSM acceptable use policy.

    Args:
        pincode: 6-digit Indian postal code.
        cache: In-memory cache mapping pincode strings to [lat, lon].
        cache_file: Path to persistent cache JSON file.

    Returns:
        Tuple of (latitude, longitude) if resolved, else None.
    """
    clean_pincode = pincode.strip()
    if not clean_pincode:
        return None

    # Check cache first
    if clean_pincode in cache:
        coords = cache[clean_pincode]
        if coords and len(coords) == 2:
            return coords[0], coords[1]
        return None

    url = (
        f"https://nominatim.openstreetmap.org/search?"
        f"postalcode={clean_pincode}&country=India&format=json"
    )
    headers = {"User-Agent": "Agent0-TataSteelDistributorPipeline/1.0"}

    try:
        time.sleep(1.0)  # Nominatim strict rate-limit requirement: 1 request/sec
        resp = requests.get(url, headers=headers, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()

        if data and isinstance(data, list) and len(data) > 0:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            cache[clean_pincode] = [lat, lon]
            _save_pincode_cache(cache, cache_file)
            return lat, lon

        # Cache misses as empty list to prevent repeated 404/empty queries
        cache[clean_pincode] = []
        _save_pincode_cache(cache, cache_file)
    except Exception as e:
        logger.warning("Nominatim geocoding failed for pincode %s: %s", clean_pincode, e)

    return None


# ────────────────────────────────────────────────────────
# Main Geocoding Orchestrator
# ────────────────────────────────────────────────────────


def geocode_candidates(
    candidates: list[DistributorCandidate],
    use_nominatim: bool = False,
    cache_file: Path = PINCODE_CACHE_FILE,
) -> list[DistributorCandidate]:
    """Assign latitude, longitude, and geocode_source to DistributorCandidate records.

    Resolution hierarchy:
      1. Pre-existing coordinates -> geocode_source = GeoSource.MANUAL (retained).
      2. Offline pincode table (exact 6-digit, then 3-digit prefix)
         -> geocode_source = GeoSource.PINCODE (no network required).
      3. If pincode is available and use_nominatim=True -> geocoded via Nominatim/cache.
      4. Fallback to state centroid matching candidate.state_code.
      5. If unknown state code, remains geocode_source = GeoSource.NONE.

    Args:
        candidates: List of DistributorCandidate models to geocode in-place.
        use_nominatim: When True, queries OSM Nominatim for PIN code coordinates
            not covered by the offline table.
        cache_file: Local cache path for resolved pincodes.

    Returns:
        The mutated candidates list.
    """
    pincode_cache = _load_pincode_cache(cache_file) if use_nominatim else {}
    source_counts: dict[str, int] = {
        "MANUAL": 0,
        "PINCODE": 0,
        "NOMINATIM": 0,
        "STATE_CENTROID": 0,
        "NONE": 0,
    }

    for cand in candidates:
        # Step a: If candidate already has lat/lon set, keep as MANUAL
        if cand.latitude is not None and cand.longitude is not None:
            cand.geocode_source = GeoSource.MANUAL
            source_counts["MANUAL"] += 1
            continue

        resolved = False

        # Step b: Offline pincode lookup (exact, then 3-digit prefix) — no network.
        if cand.pincode:
            offline_coords = lookup_offline_pincode(cand.pincode)
            if offline_coords:
                cand.latitude, cand.longitude = offline_coords
                cand.geocode_source = GeoSource.PINCODE
                source_counts["PINCODE"] += 1
                resolved = True

        # Step c: Try PIN code geocoding via Nominatim if requested and still unresolved
        if not resolved and use_nominatim and cand.pincode:
            coords = geocode_pincode_nominatim(cand.pincode, pincode_cache, cache_file)
            if coords:
                cand.latitude, cand.longitude = coords
                cand.geocode_source = GeoSource.NOMINATIM
                source_counts["NOMINATIM"] += 1
                resolved = True

        # Step d: Fallback to state centroid from GSTIN_STATE_CENTROIDS
        if not resolved:
            state_code = cand.state_code.zfill(2) if cand.state_code else ""
            if state_code in GSTIN_STATE_CENTROIDS:
                lat, lon = GSTIN_STATE_CENTROIDS[state_code]
                cand.latitude = lat
                cand.longitude = lon
                cand.geocode_source = GeoSource.STATE_CENTROID
                source_counts["STATE_CENTROID"] += 1
            else:
                cand.geocode_source = GeoSource.NONE
                source_counts["NONE"] += 1

    # Rich summary table
    table = Table(title="Agent 0 Geocoding Summary")
    table.add_column("Geocode Source", style="cyan", no_wrap=True)
    table.add_column("Count", style="magenta")

    total_geocoded = sum(
        count for src, count in source_counts.items() if src != "NONE"
    )

    for src, count in source_counts.items():
        table.add_row(src, str(count))
    table.add_section()
    table.add_row("Total Geocoded Candidates", str(total_geocoded))

    console.print(table)

    return candidates
