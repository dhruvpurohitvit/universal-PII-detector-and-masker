"""
patterns.py — Compiled regex patterns for PII detection.
All patterns are compiled once at import time.
"""

import re
from typing import List, Tuple

from pii_detector.config.settings import PATTERN_RELIABILITY, STRICT_PATTERN_NAMES, MODERATE_PATTERN_NAMES, BROAD_PATTERN_NAMES
from pii_detector.core.validators import validator_for

# (entity_name, pattern_name, raw_pattern, base_score)
_REGEX_SPECS: List[Tuple[str, str, str, float]] = [
    # ── High-precision structured IDs ────────────────────────────────────────
    ("PAN Number",            "pan_strict",       r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",                               0.85),
    ("Aadhaar Number",        "aadhaar",          r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b",                  0.80),
    ("Email Address",         "email",            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",   0.95),
    ("Social Security Number","ssn_dashes",       r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",  0.90),
    ("Social Security Number","ssn_compact",      r"\b(?!000|666|9\d{2})(?!000000000)\d{9}\b",                 0.55),
    ("IP Address",            "ipv4",
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",                     0.85),
    ("IP Address",            "ipv6_full",        r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b",           0.85),
    ("Credit Card Number",    "cc_groups",
        r"\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{3,4}\b",               0.82),
    # ── Phone numbers ─────────────────────────────────────────────────────────
    ("Phone Number",          "phone_intl",       r"\+[1-9]\d{6,14}\b",                                        0.78),
    ("Phone Number",          "phone_in_mobile",  r"\b[6-9]\d{9}\b",                                           0.75),
    ("Phone Number",          "phone_us",         r"\b(?:\+1[\s\-]?)?\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}\b",   0.72),
    # ── Documents ─────────────────────────────────────────────────────────────
    ("Passport Number",       "passport_in",      r"\b[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9]\b",                0.78),
    ("Passport Number",       "passport_generic", r"\b[A-Z]{1,2}\d{6,9}\b",                                    0.52),
    ("Driver License Number", "dl_in",            r"\b[A-Z]{2}[0-9]{2}[\s]?[0-9]{11}\b",                      0.68),
    ("Driver License Number", "dl_us",            r"\b[A-Z][0-9]{7}\b",                                        0.52),
    ("Voter ID",              "voter_id",         r"\b[A-Z]{3}[0-9]{7}\b",                                     0.70),
    # ── Financial ─────────────────────────────────────────────────────────────
    ("IBAN",                  "iban",             r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})\b",    0.82),
    ("IFSC Code",             "ifsc",             r"\b[A-Z]{4}0[A-Z0-9]{6}\b",                                 0.88),
    ("Bank Account Number",   "bank_acc",         r"\b\d{9,18}\b",                                             0.45),
    # ── Network ───────────────────────────────────────────────────────────────
    ("MAC Address",           "mac_colon",        r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b",                0.90),
    ("MAC Address",           "mac_hyphen",       r"\b(?:[0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2}\b",                0.90),
    # ── Crypto ────────────────────────────────────────────────────────────────
    ("Crypto Wallet Address", "btc_legacy",       r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",                     0.72),
    ("Crypto Wallet Address", "eth",              r"\b0x[a-fA-F0-9]{40}\b",                                    0.88),
    # ── Dates ─────────────────────────────────────────────────────────────────
    ("Date of Birth",         "dob_iso",          r"\b(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}\b",                    0.68),
    ("Date of Birth",         "dob_dmy",          r"\b\d{2}[-/]\d{2}[-/](?:19|20)\d{2}\b",                    0.65),
    # ── UK specifics ──────────────────────────────────────────────────────────
    ("UK NHS Number",         "nhs",              r"\b\d{3}[\s\-]?\d{3}[\s\-]?\d{4}\b",                       0.70),
    ("National Insurance Number","uk_ni",
        r"\b[A-CEGHJ-PR-TW-Z]{1}[A-CEGHJ-NPR-TW-Z]{1}\d{6}[A-D]{1}\b",                                      0.88),
    # ── US specifics ──────────────────────────────────────────────────────────
    ("Tax Identification Number","us_itin",       r"\b9\d{2}[\s\-]?[78]\d[\s\-]?\d{4}\b",                     0.82),
    # ── Enterprise policy identifiers ─────────────────────────────────────────
    ("Employee Identifier",   "employee_id",      r"\b(?:EMP|EMPL|STAFF)[-_ ]?[A-Z0-9]{3,20}\b",              0.82),
    ("Customer Identifier",   "customer_id",      r"\b(?:CUST|CUSTOMER|CLIENT)[-_ ]?[A-Z0-9]{3,24}\b",        0.82),
    ("User Identifier",       "user_id",          r"\b(?:USR|USER)[-_ ]?[A-Z0-9]{3,24}\b",                    0.80),
    ("Contract Number",       "contract_number",
        r"\b(?:CTR|CONTRACT|AGR)[-/ _]?[A-Z0-9][A-Z0-9/_-]{4,30}\b",                                         0.82),
    ("Device Identifier",     "device_serial",
        r"\b(?:SN|SERIAL|DEV|DEVICE)[-_ ]?[A-Z0-9]{5,30}\b",                                                  0.82),
]

# Compile once
COMPILED_REGEX: List[Tuple[str, str, re.Pattern, float]] = [
    (entity, name, re.compile(pattern), score)
    for entity, name, pattern, score in _REGEX_SPECS
]


def regex_scan_value(value: str) -> list:
    """Return list of Detection dataclass instances from regex scanning one value."""
    from pii_detector.core.models import Detection

    detections = []
    for entity, name, pattern, score in COMPILED_REGEX:
        for match in pattern.finditer(value):
            span = match.group(0)
            vp = validator_for(entity, span)
            detections.append(Detection(
                value=value,
                entity=entity,
                source="regex",
                recognizer=name,
                score=score,
                start=match.start(),
                end=match.end(),
                pattern=name,
                validator_pass=vp,
            ))
    return detections


def pattern_reliability(name: str) -> float:
    return PATTERN_RELIABILITY.get(name, 0.50)


def pattern_class(name: str) -> str:
    if name in STRICT_PATTERN_NAMES:
        return "STRICT"
    if name in MODERATE_PATTERN_NAMES:
        return "MODERATE"
    if name in BROAD_PATTERN_NAMES:
        return "BROAD"
    return "UNKNOWN"
