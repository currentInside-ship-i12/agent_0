"""CSV ingestion and normalization pipeline for Distributor Candidates.

Loads raw distributor GST export CSVs, parses turnover slabs, normalizes
HSN/SAC product codes, extracts geolocation hints (state, pincode), and
performs PAN-level deduplication.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

from agent_0.models import DistributorCandidate

logger = logging.getLogger(__name__)
console = Console()

# ────────────────────────────────────────────────────────
# State Code Mapping
# ────────────────────────────────────────────────────────

GSTIN_STATE_MAP: dict[str, str] = {
    "01": "Jammu & Kashmir",
    "02": "Himachal Pradesh",
    "03": "Punjab",
    "04": "Chandigarh",
    "05": "Uttarakhand",
    "06": "Haryana",
    "07": "Delhi",
    "08": "Rajasthan",
    "09": "Uttar Pradesh",
    "10": "Bihar",
    "11": "Sikkim",
    "12": "Arunachal Pradesh",
    "13": "Nagaland",
    "14": "Manipur",
    "15": "Mizoram",
    "16": "Tripura",
    "17": "Meghalaya",
    "18": "Assam",
    "19": "West Bengal",
    "20": "Jharkhand",
    "21": "Odisha",
    "22": "Chhattisgarh",
    "23": "Madhya Pradesh",
    "24": "Gujarat",
    "25": "Dadra & Nagar Haveli and Daman & Diu",
    "26": "Dadra & Nagar Haveli and Daman & Diu",
    "27": "Maharashtra",
    "28": "Andhra Pradesh",
    "29": "Karnataka",
    "30": "Goa",
    "31": "Lakshadweep",
    "32": "Kerala",
    "33": "Tamil Nadu",
    "34": "Puducherry",
    "35": "Andaman & Nicobar Islands",
    "36": "Telangana",
    "37": "Andhra Pradesh",
    "38": "Ladakh",
}

PINCODE_REGEX = re.compile(r"\b([1-9]\d{5})\b")


# ────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────


def _clean_str(val: Any) -> Optional[str]:
    """Clean and normalize string values, returning None for empty/null."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null", "not available"}:
        return None
    return s


def _col(row: pd.Series, *names: str) -> Any:
    """Return the raw cell for the first matching header variant.

    Real GST exports vary in header spelling/spacing (e.g. "EINVOICESTATUS"
    vs "EINVOICE STATUS"). The first name is the actual-data header; the
    rest are legacy/legend fallbacks. Returns None when no variant exists.
    """
    index = row.index
    for name in names:
        if name in index:
            return row.get(name)
    return None


def _extract_unit_and_value(text: str, is_lower: bool = False) -> Optional[tuple[float, str]]:
    """Extract numeric value and unit ('lakh' or 'cr') from a fragment.

    If is_lower is True and multiple numbers exist (e.g. 'Slab 1 - ₹0'), the last
    number immediately preceding the upper bound separator is selected.
    """
    cleaned = text.replace(",", "").replace("₹", "").strip()
    matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(lakhs?|cr(?:ore)?s?)?", cleaned, re.IGNORECASE))
    if not matches:
        return None
    target = matches[-1] if is_lower else matches[0]
    num = float(target.group(1))
    unit = target.group(2)
    return num, (unit.lower() if unit else "")


def parse_turnover_slab(slab_str: Optional[str]) -> float:
    """Extract midpoint turnover in ₹ Crores from a slab description.

    Handles variations such as:
      - "Slab 1 - ₹0 to ₹40 lakhs" -> 0.20
      - "Rs. 40 lakhs to Rs. 1.5 Cr" -> 0.95
      - "Above Rs. 40 lakhs upto Rs. 1.5 Cr" -> 0.95
      - "Rs. 1.5 Cr to Rs. 5 Cr" -> 3.25
      - "Rs. 5 Cr to Rs. 25 Cr" -> 15.0
      - "Rs. 25 Cr to Rs. 100 Cr" -> 62.5
      - "Rs. 100 Cr to Rs. 500 Cr" -> 300.0
      - "Above Rs. 500 Cr" -> 750.0
      - "Not Available" / NaN / None -> 0.0

    Args:
        slab_str: Raw turnover slab string from GST portal.

    Returns:
        Estimated turnover midpoint in ₹ Crores.
    """
    cleaned = _clean_str(slab_str)
    if not cleaned:
        return 0.0

    s = cleaned.replace(",", "").replace("₹", "").strip()

    # Case: "Above Rs. 500 Cr" without upper bound
    if re.search(r"above\s+(?:rs\.?\s*)?500\s*(?:cr|crore)", s, re.IGNORECASE) and not re.search(
        r"\b(?:to|upto)\b", s, re.IGNORECASE
    ):
        return 750.0

    # Case: "<X> and above" (e.g. "Slab: Rs. 500 Cr. and above")
    and_above_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(lakhs?|cr(?:ore)?s?)?\.?\s*and\s+above",
        s,
        re.IGNORECASE,
    )
    if and_above_match:
        val = float(and_above_match.group(1))
        unit = (and_above_match.group(2) or "").lower()
        val_cr = (val * 0.01) if "lakh" in unit else val
        return round(val_cr * 1.5, 2)

    # Range pattern: [optional prefix] <lower> (to|upto) <upper>
    range_match = re.search(
        r"(?:above\s+)?(.*?)\b(?:to|upto)\b\s*(.*)",
        s,
        re.IGNORECASE,
    )
    if range_match:
        part1 = range_match.group(1).strip()
        part2 = range_match.group(2).strip()

        val1 = _extract_unit_and_value(part1, is_lower=True)
        val2 = _extract_unit_and_value(part2, is_lower=False)

        if val1 and val2:
            num1, unit1 = val1
            num2, unit2 = val2

            # If lower bound has no explicit unit, inherit upper bound unit —
            # except when the upper unit is Cr and the lower number is larger
            # (e.g. "Rs. 40 to Rs. 1.5 Cr"): a bare "40 Cr" would exceed the
            # upper bound, so the lower value must be in Lakhs.
            if not unit1 and unit2:
                if "cr" in unit2 and num1 > num2:
                    # Treat lower bound as Lakhs (num1 / 100 Cr).
                    cr1 = num1 * 0.01
                    cr2 = num2  # unit2 is already Cr
                    return round((cr1 + cr2) / 2.0, 2)
                unit1 = unit2
            elif not unit2 and unit1:
                unit2 = unit1

            # Convert to Crores
            cr1 = (num1 * 0.01) if "lakh" in unit1 else num1
            cr2 = (num2 * 0.01) if "lakh" in unit2 else num2

            return round((cr1 + cr2) / 2.0, 2)

    # Fallback: check for single "Above X Cr" or similar
    above_match = re.search(r"above\s+(?:rs\.?\s*)?(\d+(?:\.\d+)?)\s*(lakhs?|cr(?:ore)?s?)?", s, re.IGNORECASE)
    if above_match:
        val = float(above_match.group(1))
        unit = (above_match.group(2) or "").lower()
        val_cr = (val * 0.01) if "lakh" in unit else val
        return round(val_cr * 1.5, 2)

    return 0.0


def _normalize_code(code: Any) -> Optional[str]:
    """Normalize HSN/SAC code by removing spaces, dots, and hyphens."""
    val = _clean_str(code)
    if not val:
        return None
    normalized = re.sub(r"[\s.\-]", "", val)
    return normalized if normalized else None


# ────────────────────────────────────────────────────────
# Row Parsing
# ────────────────────────────────────────────────────────


def parse_distributor_row(row: pd.Series) -> Optional[DistributorCandidate]:
    """Transform a single pandas CSV row into a DistributorCandidate model.

    Args:
        row: Pandas Series representing one record from the distributor CSV.

    Returns:
        Validated DistributorCandidate model, or None if gstin is absent.
    """
    gstin = _clean_str(row.get("GSTIN"))
    if not gstin:
        return None

    gstin = gstin.upper().strip()
    pan = _clean_str(row.get("PAN"))
    if not pan:
        # Fallback to PAN from GSTIN if valid structure
        pan = gstin[2:12].upper() if len(gstin) >= 12 else gstin
    else:
        pan = pan.upper().strip()

    legal_name = _clean_str(row.get("LEGAL NAME")) or gstin
    trade_name = _clean_str(row.get("TRADE NAME"))
    display_name = trade_name if trade_name else legal_name

    # State extraction from GSTIN
    state_code = gstin[:2] if len(gstin) >= 2 else ""
    state_name = GSTIN_STATE_MAP.get(state_code, "")

    # Permanent address and pincode extraction
    perm_addr = _clean_str(row.get("PERMANENT ADDRESS"))
    pincode: Optional[str] = None
    if perm_addr:
        pincode_match = PINCODE_REGEX.search(perm_addr)
        if pincode_match:
            pincode = pincode_match.group(1)

    # Turnover
    turnover_slab = _clean_str(row.get("AGGRE TURNOVER"))
    turnover_midpoint_cr = parse_turnover_slab(turnover_slab)

    # Registration & status
    reg_status = _clean_str(row.get("REGISTRATION STATUS")) or "Unknown"
    nature_core = _clean_str(row.get("NATURE OF CORE BUSINESS ACTIVITIES"))
    constitution = _clean_str(row.get("CONSTITUTION"))
    einvoice_raw = _clean_str(_col(row, "EINVOICESTATUS", "EINVOICE STATUS"))
    einvoice_enabled = bool(einvoice_raw and einvoice_raw.upper() == "Y")
    reg_type = _clean_str(row.get("REGISTRATION TYPE"))
    cancelation_date = _clean_str(row.get("CANCELATION DATE"))
    ekyc_status = _clean_str(_col(row, "EKYC_STATUS", "eKYC_Status"))
    aadhaar_verified = _clean_str(_col(row, "AADHAR VERIFIED", "AADHA VERIFIED"))

    # HSN Codes & Descriptions
    hsn_codes: list[str] = []
    hsn_descriptions: list[str] = []
    for i in range(1, 6):
        hsn_val = _normalize_code(row.get(f"HSN CODE {i}"))
        if hsn_val:
            hsn_codes.append(hsn_val)
        desc_val = _clean_str(row.get(f"DESCRIPTION OF GOODS {i}"))
        if desc_val:
            hsn_descriptions.append(desc_val)

    # Steel HSN check (Chapters 72: Iron/Steel, 73: Articles of Iron/Steel)
    has_steel_hsn = any(c.startswith("72") or c.startswith("73") for c in hsn_codes)

    # SAC Codes & Descriptions
    sac_codes: list[str] = []
    sac_descriptions: list[str] = []
    for i in range(1, 6):
        sac_val = _normalize_code(row.get(f"SAC CODE {i}"))
        if sac_val:
            sac_codes.append(sac_val)
        sac_desc = _clean_str(row.get(f"DESCRIPTION OF SAC {i}"))
        if sac_desc:
            sac_descriptions.append(sac_desc)

    is_sac_only = bool(sac_codes and not hsn_codes)

    # Nature of Business Activities
    business_activities: list[str] = []
    for i in range(1, 8):
        activity = _clean_str(row.get(f"NATURE OF BUSINESS ACTIVITY {i}"))
        if activity:
            business_activities.append(activity)

    # Member Details
    member_names: list[str] = []
    for i in range(1, 10):
        member = _clean_str(row.get(f"MEMBER DETAILS {i}"))
        if member:
            member_names.append(member)

    # Additional Addresses
    additional_addresses: list[str] = []
    for i in range(1, 17):
        addr = _clean_str(row.get(f"ADDITIONAL ADDRESS {i}"))
        if addr:
            additional_addresses.append(addr)

    num_additional_addresses = len(additional_addresses)

    return DistributorCandidate(
        gstin=gstin,
        pan=pan,
        legal_name=legal_name,
        trade_name=trade_name,
        display_name=display_name,
        state_code=state_code,
        state_name=state_name,
        permanent_address=perm_addr,
        pincode=pincode,
        latitude=None,
        longitude=None,
        geocode_source="NONE",
        turnover_slab=turnover_slab,
        turnover_midpoint_cr=turnover_midpoint_cr,
        registration_status=reg_status,
        nature_of_core_business=nature_core,
        constitution=constitution,
        einvoice_enabled=einvoice_enabled,
        registration_type=reg_type,
        cancelation_date=cancelation_date,
        ekyc_status=ekyc_status,
        aadhaar_verified=aadhaar_verified,
        hsn_codes=hsn_codes,
        hsn_descriptions=hsn_descriptions,
        sac_codes=sac_codes,
        sac_descriptions=sac_descriptions,
        has_steel_hsn=has_steel_hsn,
        is_sac_only=is_sac_only,
        business_activities=business_activities,
        member_names=member_names,
        additional_addresses=additional_addresses,
        num_additional_addresses=num_additional_addresses,
    )


# ────────────────────────────────────────────────────────
# Main Loader & Deduplication
# ────────────────────────────────────────────────────────


def load_distributor_csv(
    csv_path: Path,
    deduplicate_by_pan: bool = True,
) -> list[DistributorCandidate]:
    """Load distributor CSV and transform into DistributorCandidate models.

    Args:
        csv_path: Filesystem path to the input distributor CSV file.
        deduplicate_by_pan: When True, groups GSTINs sharing the same PAN and
            retains the record with the highest turnover slab midpoint.

    Returns:
        List of parsed and deduplicated DistributorCandidate objects.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If file cannot be decoded with utf-8 or latin-1.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Distributor CSV file not found at: {path}")

    # Encoding fallback strategy
    df: Optional[pd.DataFrame] = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=encoding, dtype=str)
            logger.info("Successfully read %s using encoding %s", path.name, encoding)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError as exc:
            # Malformed CSV (e.g. unbalanced quotes) is not a decoding
            # problem — retrying other encodings cannot help.
            raise ValueError(f"Failed to parse CSV {path} as CSV: {exc}") from exc

    if df is None:
        raise ValueError(f"Failed to decode CSV {path} with utf-8, latin-1, or cp1252.")

    total_rows = len(df)
    candidates: list[DistributorCandidate] = []

    for _, row in df.iterrows():
        candidate = parse_distributor_row(row)
        if candidate:
            candidates.append(candidate)

    valid_rows = len(candidates)
    dedup_count = 0

    if deduplicate_by_pan:
        # Group by PAN and retain candidate with highest turnover midpoint
        pan_groups: dict[str, list[DistributorCandidate]] = {}
        for cand in candidates:
            pan_groups.setdefault(cand.pan, []).append(cand)

        deduped_candidates: list[DistributorCandidate] = []
        for pan, group in pan_groups.items():
            if len(group) > 1:
                dedup_count += len(group) - 1
                # Sort descending by turnover_midpoint_cr; keep first
                best = max(group, key=lambda c: c.turnover_midpoint_cr)
                logger.info(
                    "Deduplicated PAN %s: selected GSTIN %s (Turnover: %.2f Cr) from %d records",
                    pan,
                    best.gstin,
                    best.turnover_midpoint_cr,
                    len(group),
                )
                deduped_candidates.append(best)
            else:
                deduped_candidates.append(group[0])

        candidates = deduped_candidates

    steel_hsn_count = sum(1 for c in candidates if c.has_steel_hsn)

    # Rich summary output
    table = Table(title="Agent 0 Distributor CSV Ingestion Summary")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Count", style="magenta")

    table.add_row("Total Raw Rows", str(total_rows))
    table.add_row("Valid Records Parsed", str(valid_rows))
    table.add_row("PAN Deduplications Removed", str(dedup_count))
    table.add_row("Final Candidates", str(len(candidates)))
    table.add_row("Entities with Steel HSN (72/73)", str(steel_hsn_count))

    console.print(table)

    return candidates
