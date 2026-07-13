"""
validators.py — Deterministic validators for structured PII entities.
Each validator returns True (pass), False (fail), or None (not applicable).
"""

import ipaddress
import re
from datetime import date, datetime
from typing import Optional

from pii_detector.config.settings import MIN_DOB_AGE, MAX_DOB_AGE

# ─── Verhoeff tables (Aadhaar checksum) ──────────────────────────────────────
_D = [
    [0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0],
]
_P = [
    [0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8],
]


def digits_only(value: str) -> str:
    return re.sub(r"\D", "", str(value))


def compact_alnum(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(value))


# ─── Checksums ────────────────────────────────────────────────────────────────

def luhn_valid(value: str) -> bool:
    """Luhn algorithm — credit card numbers."""
    d = digits_only(value)
    if not 13 <= len(d) <= 19 or len(set(d)) == 1:
        return False
    total, parity = 0, len(d) % 2
    for i, ch in enumerate(d):
        n = int(ch)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def verhoeff_valid(value: str) -> bool:
    """Verhoeff algorithm — Aadhaar (12 digits, first digit 2-9)."""
    d = digits_only(value)
    if len(d) != 12 or d[0] not in "23456789":
        return False
    c = 0
    for i, item in enumerate(reversed(d)):
        c = _D[c][_P[i % 8][int(item)]]
    return c == 0


def iban_valid(value: str) -> bool:
    """ISO 7064 MOD-97-10 — IBAN."""
    iban = re.sub(r"\s+", "", value).upper()
    if not 15 <= len(iban) <= 34 or not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", iban):
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    remainder = 0
    for ch in numeric:
        remainder = (remainder * 10 + int(ch)) % 97
    return remainder == 1


def nhs_valid(value: str) -> bool:
    """NHS number checksum (England/Wales) — 10 digits."""
    d = digits_only(value)
    if len(d) != 10:
        return False
    total = sum(int(d[i]) * (10 - i) for i in range(9))
    check = 11 - (total % 11)
    if check == 11:
        check = 0
    return check != 10 and check == int(d[9])


# ─── Format validators ────────────────────────────────────────────────────────

def ip_valid(value: str) -> bool:
    try:
        ipaddress.ip_address(value.strip())
        return True
    except ValueError:
        return False


def pan_valid(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", value.strip().upper()))


def ifsc_valid(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", value.strip().upper()))


def mac_valid(value: str) -> bool:
    return bool(re.fullmatch(r"(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}", value.strip()))


def email_valid(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", value.strip()))


def ssn_valid(value: str) -> bool:
    """Structural SSN validation (not area-code ranges, just exclusions)."""
    d = digits_only(value)
    if len(d) != 9:
        return False
    if d[:3] in {"000", "666"} or d[:3].startswith("9"):
        return False
    if d[3:5] == "00" or d[5:] == "0000":
        return False
    return True


def ni_valid(value: str) -> bool:
    """UK National Insurance Number structural check."""
    clean = compact_alnum(value).upper()
    forbidden_prefix = {"D", "F", "I", "Q", "U", "V"}
    if len(clean) != 9:
        return False
    if clean[0] in forbidden_prefix or clean[1] in forbidden_prefix:
        return False
    if clean[:2] in {"BG", "GB", "KN", "NK", "NT", "TN", "ZZ"}:
        return False
    if not clean[:2].isalpha() or not clean[2:8].isdigit() or clean[8] not in "ABCD":
        return False
    return True


def itin_valid(value: str) -> bool:
    """US ITIN: 9XX-7X-XXXX or 9XX-8X-XXXX."""
    d = digits_only(value)
    if len(d) != 9:
        return False
    if d[0] != "9":
        return False
    if d[3] not in "7":  # second group starts 70-88, excluding 89,93
        return d[3] == "8" and d[4] in "012345678"
    return True


def phone_valid(value: str) -> bool:
    raw = value.strip()
    d = digits_only(raw)
    if raw.startswith("+"):
        return bool(re.fullmatch(r"\+[1-9]\d{6,14}", raw))
    if len(d) == 10:
        return d[0] != "0"
    if len(d) == 12 and d.startswith("91"):
        return d[2] in "6789"
    return False


# ─── Date validators ─────────────────────────────────────────────────────────

_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
    "%m-%d-%Y", "%m/%d/%Y", "%d.%m.%Y", "%Y.%m.%d",
)


def parse_date(value: str) -> Optional[date]:
    text = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def valid_date(value: str) -> bool:
    return parse_date(value) is not None


def plausible_dob(value: str) -> bool:
    parsed = parse_date(value)
    if parsed is None or parsed > date.today():
        return False
    age = (date.today() - parsed).days / 365.2425
    return MIN_DOB_AGE <= age <= MAX_DOB_AGE


# ─── Dispatch ─────────────────────────────────────────────────────────────────

def validator_for(entity: str, value: str) -> Optional[bool]:
    """Return True/False/None for the given entity and raw value."""
    dispatch = {
        "Credit Card Number":        luhn_valid,
        "Aadhaar Number":            verhoeff_valid,
        "IBAN":                      iban_valid,
        "UK NHS Number":             nhs_valid,
        "IP Address":                ip_valid,
        "PAN Number":                pan_valid,
        "IFSC Code":                 ifsc_valid,
        "MAC Address":               mac_valid,
        "Email Address":             email_valid,
        "Social Security Number":    ssn_valid,
        "National Insurance Number": ni_valid,
        "Tax Identification Number": itin_valid,
        "Phone Number":              phone_valid,
        "Date of Birth":             plausible_dob,
    }
    fn = dispatch.get(entity)
    return fn(value) if fn else None


# ─── Strength Registry (exported for aggregator) ──────────────────────────────

def validator_strength(entity: str) -> tuple:
    """Return (kind_str, strength_float) for a given entity.

    kind_str  — one of CHECKSUM | STRICT_FORMAT | STANDARD_PARSE |
                        PLAUSIBILITY | BASIC_STRUCTURE | NONE
    strength  — [0,1] relative confidence when the validator passes
    """
    from pii_detector.config.settings import VALIDATOR_STRENGTH
    return VALIDATOR_STRENGTH.get(entity, ("NONE", 0.0))

