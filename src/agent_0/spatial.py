"""Spatial distance engine with Haversine and Google Maps.

Provides distance calculations and anti-cannibalization checks.
Functions in this module are stateless and use the standard math library
along with requests for remote distance matrix queries.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_EARTH_RADIUS_KM: float = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two coordinates in kilometers.

    Args:
        lat1: Latitude of origin in degrees.
        lon1: Longitude of origin in degrees.
        lat2: Latitude of destination in degrees.
        lon2: Longitude of destination in degrees.

    Returns:
        Great-circle distance in kilometers.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    sin_half_dphi = math.sin(delta_phi / 2.0)
    sin_half_dlambda = math.sin(delta_lambda / 2.0)

    a = (
        sin_half_dphi * sin_half_dphi
        + math.cos(phi1) * math.cos(phi2) * sin_half_dlambda * sin_half_dlambda
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return _EARTH_RADIUS_KM * c


def coarse_road_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    k_tort: float = 1.30,
) -> float:
    """Estimate road distance using great-circle distance and tortuosity factor.

    Args:
        lat1: Latitude of origin in degrees.
        lon1: Longitude of origin in degrees.
        lat2: Latitude of destination in degrees.
        lon2: Longitude of destination in degrees.
        k_tort: Road tortuosity multiplier (default: 1.30).

    Returns:
        Estimated road distance in kilometers.
    """
    return haversine_km(lat1, lon1, lat2, lon2) * k_tort


def google_maps_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    api_key: str,
) -> tuple[float, Literal["GMAPS", "COARSE"]]:
    """Calculate driving distance using Google Distance Matrix API with fallback.

    Args:
        lat1: Latitude of origin in degrees.
        lon1: Longitude of origin in degrees.
        lat2: Latitude of destination in degrees.
        lon2: Longitude of destination in degrees.
        api_key: Google Maps API key.

    Returns:
        Tuple of (distance_km, source) where source is 'GMAPS' on success
        or 'COARSE' on failure/fallback.
    """
    fallback_distance = coarse_road_distance(lat1, lon1, lat2, lon2)

    if requests is None or not api_key or not api_key.strip():
        return fallback_distance, "COARSE"

    endpoint = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": f"{lat1},{lon1}",
        "destinations": f"{lat2},{lon2}",
        "mode": "driving",
        "key": api_key.strip(),
    }

    try:
        response = requests.get(endpoint, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "OK":
            logger.warning(
                "Google Maps Distance Matrix returned status '%s' for "
                "(%s,%s)->(%s,%s); using coarse estimate",
                data.get("status"),
                lat1,
                lon1,
                lat2,
                lon2,
            )
            return fallback_distance, "COARSE"

        rows = data.get("rows")
        if not rows:
            return fallback_distance, "COARSE"

        elements = rows[0].get("elements")
        if not elements or elements[0].get("status") != "OK":
            return fallback_distance, "COARSE"

        distance_meters = elements[0].get("distance", {}).get("value")
        if distance_meters is None:
            return fallback_distance, "COARSE"

        distance_km = float(distance_meters) / 1000.0
        return distance_km, "GMAPS"
    except Exception as exc:
        # Previously silent: the caller could not tell a GMaps failure apart
        # from a request that was never made.
        logger.warning(
            "Google Maps distance lookup failed for (%s,%s)->(%s,%s): %s",
            lat1,
            lon1,
            lat2,
            lon2,
            exc,
        )
        return fallback_distance, "COARSE"


def check_cannibalization(
    candidate_lat: float,
    candidate_lon: float,
    existing_coords: list[tuple[float, float]],
    min_radius_km: float,
) -> tuple[bool, float]:
    """Check whether a candidate is within the cannibalization radius of existing distributors.

    Args:
        candidate_lat: Latitude of candidate.
        candidate_lon: Longitude of candidate.
        existing_coords: List of (latitude, longitude) tuples for existing distributors.
        min_radius_km: Minimum allowed distance buffer in kilometers.

    Returns:
        Tuple of (too_close, nearest_distance_km). If existing_coords is empty,
        returns (False, infinity).
    """
    if not existing_coords:
        return False, float("inf")

    min_distance = float("inf")
    for ex_lat, ex_lon in existing_coords:
        dist = haversine_km(candidate_lat, candidate_lon, ex_lat, ex_lon)
        if dist < min_distance:
            min_distance = dist

    too_close = min_distance < min_radius_km
    return too_close, min_distance
