"""
enterprise_pii_column_detector.py

Column-level hybrid PII detection:
1) Native Presidio/spaCy candidate detection.
2) Independent custom-regex candidate detection + deterministic validators.
3) Deterministic stratified evidence sampling (max 15 samples/column).
4) GLiNER zero-shot NER over contextual evidence samples.
5) Entity-specific conflict resolution and explainable policy-aware aggregation.

Usage:
    python enterprise_pii_column_detector.py --input data.csv --output results.csv
"""

# Noise suppression must precede third-party imports.
import os
import warnings
import logging

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("enterprise_pii")

for _name in (
    "huggingface_hub", "httpx", "httpcore", "gliner", "transformers",
    "presidio-analyzer", "presidio_analyzer", "spacy",
):
    logging.getLogger(_name).setLevel(logging.ERROR)

import argparse
import json
import hashlib
import ipaddress
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
from gliner import GLiNER
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GLINER_MODEL_NAME = "urchade/gliner_multi_pii-v1"
PRESIDIO_NLP_MODEL = "en_core_web_lg"
GLINER_THRESHOLD = 0.45
MAX_GLINER_SAMPLES = 15
SPARSE_PREVALENCE_MAX = 0.10
POLICY_TREAT_COARSE_LOCATION_AS_PII = False
POLICY_TREAT_ORGANIZATION_AS_PII = False
MIN_DOB_AGE = 0
MAX_DOB_AGE = 120
SCRIPT_VERSION = "13.0.0"
OUTPUT_SCHEMA_VERSION = "9.0"

# Privacy policy layer. Detection and policy classification are deliberately separate.
# Change these values to match organizational policy.
POLICY_TREAT_IDENTIFIERS_AS_PII = {
    "Employee Identifier": True,
    "Customer Identifier": True,
    "User Identifier": True,
    "Device Identifier": True,
    "Contract Number": True,
}

# Regex reliability classes: broad patterns generate candidates but cannot independently
# establish high-confidence PII without semantics, validation, or semantic NER support.
STRICT_PATTERN_NAMES = {
    "pan_strict", "email", "ssn_dashes", "ipv4", "ipv6_full", "cc_groups",
    "iban", "ifsc", "mac_colon", "mac_hyphen", "eth", "voter_id",
    "employee_id", "customer_id", "user_id", "contract_number", "device_serial",
}
MODERATE_PATTERN_NAMES = {
    "aadhaar", "phone_intl", "phone_in_mobile", "phone_us", "passport_in",
    "dl_in", "btc_legacy", "dob_iso", "dob_dmy",
}
BROAD_PATTERN_NAMES = {
    "ssn_compact", "passport_generic", "dl_us", "bank_acc",
}

PATTERN_RELIABILITY = {
    "pan_strict": 0.99, "email": 0.99, "ssn_dashes": 0.98,
    "ipv4": 0.98, "ipv6_full": 0.98, "cc_groups": 0.96,
    "iban": 0.99, "ifsc": 0.98, "mac_colon": 0.99, "mac_hyphen": 0.99,
    "eth": 0.99, "voter_id": 0.97,
    "employee_id": 0.93, "customer_id": 0.93, "user_id": 0.92,
    "contract_number": 0.92, "device_serial": 0.90,
    "aadhaar": 0.92, "phone_intl": 0.88, "phone_in_mobile": 0.90,
    "phone_us": 0.88, "passport_in": 0.90, "dl_in": 0.78,
    "btc_legacy": 0.96, "dob_iso": 0.92, "dob_dmy": 0.88,
    "ssn_compact": 0.55, "passport_generic": 0.42,
    "dl_us": 0.38, "bank_acc": 0.28,
}

PRESIDIO_ENTITY_RELIABILITY = {
    "Email Address": 0.99, "IP Address": 0.98, "Credit Card Number": 0.96,
    "IBAN": 0.99, "MAC Address": 0.99, "Crypto Wallet Address": 0.98,
    "Phone Number": 0.82, "Person Name": 0.78, "Location": 0.76,
    "Date Time": 0.82, "Passport Number": 0.72,
    "Driver License Number": 0.45, "Bank Account Number": 0.38,
    "UK NHS Number": 0.55, "URL": 0.90,
}

COLLISION_GROUPS = (
    {"Phone Number", "Bank Account Number", "Driver License Number", "UK NHS Number"},
    {"Aadhaar Number", "Bank Account Number", "Driver License Number"},
    {"Passport Number", "Driver License Number", "Device Identifier"},
    {"Credit Card Number", "Bank Account Number", "Driver License Number"},
    {"IP Address", "Date Time"},
    {"PAN Number", "Passport Number", "Driver License Number"},
)

ENTITY_PARENT = {"Address": "Location"}

GLINER_LABELS = [
    "person name", "email address", "phone number", "home address",
    "street address", "location", "date of birth", "age", "aadhaar number",
    "PAN number", "passport number", "driver license number", "voter ID",
    "bank account number", "credit card number", "IBAN", "IFSC code",
    "tax identification number", "social security number",
    "national insurance number", "IP address", "MAC address",
    "vehicle registration number", "crypto wallet address",
    "employee identifier", "customer identifier", "user identifier",
    "device identifier", "contract number", "organization name",
]

ENTITY_LABEL_MAP = {
    "IN_PAN": "PAN Number",
    "IN_AADHAAR": "Aadhaar Number",
    "EMAIL_ADDRESS": "Email Address",
    "US_SSN": "Social Security Number",
    "US_SSN_CUSTOM": "Social Security Number",
    "IP_ADDRESS": "IP Address",
    "CREDIT_CARD": "Credit Card Number",
    "CREDIT_CARD_CUSTOM": "Credit Card Number",
    "PHONE_NUMBER": "Phone Number",
    "PHONE_NUMBER_CUSTOM": "Phone Number",
    "PASSPORT_NUMBER": "Passport Number",
    "IBAN": "IBAN",
    "MAC_ADDRESS": "MAC Address",
    "CRYPTO_WALLET": "Crypto Wallet Address",
    "IN_VOTER_ID": "Voter ID",
    "DRIVER_LICENSE": "Driver License Number",
    "BANK_ACCOUNT": "Bank Account Number",
    "DATE_OF_BIRTH": "Date of Birth",
    "DATE_TIME": "Date Time",
    "PERSON": "Person Name",
    "LOCATION": "Location",
    "NRP": "Nationality",
    "ORGANIZATION": "Organization",
    "URL": "URL",
    "MEDICAL_LICENSE": "Medical License",
    "UK_NHS": "UK NHS Number",
    "US_DRIVER_LICENSE": "Driver License Number",
    "US_PASSPORT": "Passport Number",
    "US_BANK_NUMBER": "Bank Account Number",
    "US_ITIN": "Tax Identification Number",
}

CANONICAL = {
    "person name": "Person Name",
    "email address": "Email Address",
    "phone number": "Phone Number",
    "home address": "Address",
    "street address": "Address",
    "location": "Location",
    "date of birth": "Date of Birth",
    "age": "Age",
    "aadhaar number": "Aadhaar Number",
    "pan number": "PAN Number",
    "passport number": "Passport Number",
    "driver license number": "Driver License Number",
    "voter id": "Voter ID",
    "bank account number": "Bank Account Number",
    "credit card number": "Credit Card Number",
    "iban": "IBAN",
    "ifsc code": "IFSC Code",
    "tax identification number": "Tax Identification Number",
    "social security number": "Social Security Number",
    "national insurance number": "National Insurance Number",
    "ip address": "IP Address",
    "mac address": "MAC Address",
    "vehicle registration number": "Vehicle Registration Number",
    "crypto wallet address": "Crypto Wallet Address",
    "employee identifier": "Employee Identifier",
    "customer identifier": "Customer Identifier",
    "user identifier": "User Identifier",
    "device identifier": "Device Identifier",
    "contract number": "Contract Number",
    "organization name": "Organization",
}

GLINER_ENTITY_FAMILIES = {
    "Address": {"Address"},
    "Location": {"Location"},
    "Person Name": {"Person Name"},
    "Phone Number": {"Phone Number"},
    "Email Address": {"Email Address"},
    "Date of Birth": {"Date of Birth"},
    "Passport Number": {"Passport Number"},
    "Driver License Number": {"Driver License Number"},
    "Aadhaar Number": {"Aadhaar Number"},
    "PAN Number": {"PAN Number"},
    "Bank Account Number": {"Bank Account Number"},
    "Credit Card Number": {"Credit Card Number"},
    "IP Address": {"IP Address"},
    "MAC Address": {"MAC Address"},
    "IBAN": {"IBAN"},
    "IFSC Code": {"IFSC Code"},
    "Voter ID": {"Voter ID"},
    "Employee Identifier": {"Employee Identifier", "User Identifier"},
    "Customer Identifier": {"Customer Identifier", "User Identifier"},
    "User Identifier": {"User Identifier", "Customer Identifier", "Employee Identifier"},
    "Device Identifier": {"Device Identifier"},
    "Contract Number": {"Contract Number"},
    "Organization": {"Organization"},
}

# Policy: identifiers below are configurable enterprise identifiers. They are not
# automatically rejected; column semantics + evidence decide them.
POLICY_IDENTIFIER_ENTITIES = {
    "Employee Identifier", "Customer Identifier", "User Identifier",
    "Device Identifier", "Contract Number",
}

HIGH_RISK_STRUCTURAL = {
    "Aadhaar Number", "PAN Number", "Passport Number", "Driver License Number",
    "Voter ID", "Bank Account Number", "Credit Card Number", "IBAN",
    "IFSC Code", "Social Security Number", "IP Address", "MAC Address",
    "Crypto Wallet Address",
}

DIRECT_STRONG_ENTITIES = {
    "Email Address", "MAC Address", "IBAN", "Crypto Wallet Address",
}

NATURAL_LANGUAGE_ENTITIES = {"Person Name", "Address", "Location", "Organization"}
OPAQUE_STRUCTURED_ENTITIES = {
    "Aadhaar Number", "PAN Number", "Passport Number", "Driver License Number",
    "Voter ID", "Bank Account Number", "Credit Card Number", "IBAN",
    "IFSC Code", "Social Security Number", "IP Address", "MAC Address",
    "Crypto Wallet Address", "Employee Identifier", "Customer Identifier",
    "User Identifier", "Device Identifier", "Contract Number",
}

GENERIC_ID_TERMS = {
    "id", "identifier", "row number", "row num", "sequence", "seq",
    "index", "serial", "record number", "record id",
}

ENTITY_SEMANTIC_TERMS = {
    "Organization": {"organization", "organisation", "organization name", "organisation name",
                     "company", "company name", "employer", "business name", "vendor name",
                     "supplier name", "legal entity", "corporate name"},
    "Email Address": {"email", "email address", "mail", "e mail", "inbox"},
    "Phone Number": {"phone", "mobile", "mobile number", "contact no", "contact number",
                     "telephone", "tel", "whatsapp"},
    "Aadhaar Number": {"aadhaar", "aadhar", "aadhaar number", "aadhar number", "uidai"},
    "PAN Number": {"pan", "pan number", "permanent account number"},
    "Passport Number": {"passport", "passport number", "travel document"},
    "Driver License Number": {"driver license", "driving license", "driving licence",
                              "driver licence", "dl number"},
    "Voter ID": {"voter", "voter id", "epic", "election id"},
    "Bank Account Number": {"bank account", "account number", "bank account number",
                            "acct number", "account no"},
    "Credit Card Number": {"credit card", "card number", "debit card", "payment card"},
    "IBAN": {"iban", "international bank account"},
    "IFSC Code": {"ifsc", "ifsc code", "bank branch code"},
    "Social Security Number": {"ssn", "social security", "social security number"},
    "IP Address": {"ip", "ip address", "ipv4", "ipv6", "client ip", "server ip"},
    "MAC Address": {"mac", "mac address", "hardware address"},
    "Crypto Wallet Address": {"wallet", "crypto wallet", "bitcoin address",
                              "ethereum address", "blockchain address"},
    "Date of Birth": {"dob", "date of birth", "birth date", "birthday", "born"},
    "Person Name": {"name", "full name", "person name", "first name", "last name",
                    "surname", "given name", "customer name", "employee name"},
    "Address": {"address", "home address", "street address", "residential address"},
    "Location": {"location", "city", "state", "district", "country", "place"},
    "Employee Identifier": {"employee id", "employee number", "emp id", "staff id"},
    "Customer Identifier": {"customer id", "customer number", "client id"},
    "User Identifier": {"user id", "username", "login id", "account id"},
    "Device Identifier": {
        "device id", "device identifier", "device serial", "device serial number",
        "serial number", "serial no", "hardware id", "imei", "meid",
        "asset tag", "equipment id",
    },
    "Contract Number": {"contract number", "contract id", "agreement number"},
}

NEGATIVE_ENTITY_TERMS = {
    "Date of Birth": {"created date", "updated date", "order date", "invoice date",
                      "transaction date", "event date", "timestamp", "date"},
    "Phone Number": {
        "row number", "sequence", "index", "order number", "order id",
        "transaction id", "invoice id", "reference id", "record id",
        "product id", "customer id", "employee id", "internal id",
        "tracking number", "shipment id", "ticket number", "case number",
        "account reference", "registration number", "numeric sku",
    },
    "Bank Account Number": {"row number", "sequence", "index", "order id",
                            "product id", "transaction id", "invoice number"},
    "Passport Number": {"product code", "sku", "order id", "internal id"},
    "Driver License Number": {"product code", "sku", "order id", "internal id"},
    "PAN Number": {
        "product code", "sku", "coupon code", "coupon", "promo code",
        "promotion code", "voucher code", "token", "reference code",
        "campaign code", "internal code",
    },
    "IP Address": {
        "version", "release", "software version", "firmware version",
        "build version", "build number", "release version",
    },
}


# Explicit mutually contradictory semantic families used in cross-entity resolution.
SEMANTIC_CONTRADICTIONS = {
    "Passport Number": {"Driver License Number", "Phone Number", "Bank Account Number"},
    "Driver License Number": {"Passport Number", "Phone Number", "Bank Account Number"},
    "Phone Number": {"Passport Number", "Driver License Number", "Bank Account Number"},
    "Bank Account Number": {"Phone Number", "Passport Number", "Driver License Number"},
    "Date of Birth": {"IP Address", "Phone Number"},
    "IP Address": {"Date of Birth"},
    "Person Name": {"Location"},
}

ENTITY_PATTERN_SPECIFICITY = {
    "Email Address": 1.00,
    "PAN Number": 0.98,
    "IBAN": 0.98,
    "IFSC Code": 0.98,
    "MAC Address": 0.98,
    "IP Address": 0.96,
    "Credit Card Number": 0.95,
    "Aadhaar Number": 0.92,
    "Crypto Wallet Address": 0.92,
    "Voter ID": 0.88,
    "Social Security Number": 0.86,
    "Phone Number": 0.72,
    "Passport Number": 0.65,
    "Driver License Number": 0.62,
    "Bank Account Number": 0.45,
    "Date of Birth": 0.70,
    "Person Name": 0.75,
    "Address": 0.72,
    "Location": 0.62,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    value: str
    entity: str
    source: str
    recognizer: str
    score: float
    start: int
    end: int
    pattern: Optional[str] = None
    validator_pass: Optional[bool] = None


@dataclass
class ValueEvidence:
    value: str
    presidio: List[Detection] = field(default_factory=list)
    regex: List[Detection] = field(default_factory=list)

    @property
    def sources(self) -> Set[str]:
        out = set()
        if self.presidio:
            out.add("presidio")
        if self.regex:
            out.add("regex")
        return out

    @property
    def entities(self) -> Set[str]:
        return {d.entity for d in self.presidio + self.regex}

    @property
    def max_score(self) -> float:
        ds = self.presidio + self.regex
        return max((d.score for d in ds), default=0.0)


@dataclass
class GlinerEvidence:
    value: str
    value_predictions: List[Tuple[str, float]]
    context_predictions: List[Tuple[str, float]]
    predictions: List[Tuple[str, float]]
    best_entity: Optional[str]
    best_score: float
    inference_ok: bool = True


# ---------------------------------------------------------------------------
# Utility and validators
# ---------------------------------------------------------------------------

NULL_LIKE_VALUES = {
    "", "na", "n/a", "none", "null", "nil", "unknown", "not provided",
    "not available", "-", "--", "nan", "<na>",
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\u00a0", " ")
    value = value.replace("\u2010", "-").replace("\u2011", "-")
    value = value.replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", value).strip()


def is_null_like(value: str) -> bool:
    return normalize_text(value).lower() in NULL_LIKE_VALUES


def normalize_column(name: str) -> str:
    name = normalize_text(name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    name = re.sub(r"[_\-.]+", " ", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def compact_alnum(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(value))


def digits_only(value: str) -> str:
    return re.sub(r"\D", "", str(value))


def luhn_valid(value: str) -> bool:
    digits = digits_only(value)
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        n = int(ch)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


_VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9], [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6], [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8], [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2], [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4], [9,8,7,6,5,4,3,2,1,0],
]
_VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9], [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2], [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0], [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5], [7,0,4,6,9,1,3,2,5,8],
]


def verhoeff_valid(value: str) -> bool:
    digits = digits_only(value)
    if len(digits) != 12 or digits[0] not in "23456789":
        return False
    c = 0
    for i, item in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(item)]]
    return c == 0


def iban_valid(value: str) -> bool:
    iban = re.sub(r"\s+", "", value).upper()
    if not 15 <= len(iban) <= 34 or not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", iban):
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    remainder = 0
    for ch in numeric:
        remainder = (remainder * 10 + int(ch)) % 97
    return remainder == 1


def ip_valid(value: str) -> bool:
    try:
        ipaddress.ip_address(value.strip())
        return True
    except ValueError:
        return False


def parse_date(value: str) -> Optional[date]:
    text = normalize_text(value)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
                "%m-%d-%Y", "%m/%d/%Y"):
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


def validator_for(entity: str, value: str) -> Optional[bool]:
    if entity == "Credit Card Number":
        return luhn_valid(value)
    if entity == "Aadhaar Number":
        return verhoeff_valid(value)
    if entity == "IBAN":
        return iban_valid(value)
    if entity == "IP Address":
        return ip_valid(value)
    if entity == "Date of Birth":
        return plausible_dob(value)
    if entity == "PAN Number":
        return bool(re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", value.strip().upper()))
    if entity == "IFSC Code":
        return bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", value.strip().upper()))
    if entity == "MAC Address":
        return bool(re.fullmatch(r"(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}", value.strip()))
    if entity == "Email Address":
        return bool(re.fullmatch(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", value.strip()))
    if entity == "Phone Number":
        raw = value.strip()
        digits = digits_only(raw)
        # E.164-compatible international structure, Indian mobile structure,
        # or common 10-digit national numbering structure. This is structural
        # validation only; semantics still participate in adjudication.
        if raw.startswith("+"):
            return bool(re.fullmatch(r"\+[1-9]\d{6,14}", raw))
        if len(digits) == 10:
            return digits[0] != "0"
        if len(digits) == 12 and digits.startswith("91"):
            return digits[2] in "6789"
        return False
    return None


VALIDATOR_STRENGTH = {
    "Credit Card Number": ("CHECKSUM", 1.00),
    "Aadhaar Number": ("CHECKSUM", 1.00),
    "IBAN": ("CHECKSUM", 1.00),
    "PAN Number": ("STRICT_FORMAT", 0.65),
    "IFSC Code": ("STRICT_FORMAT", 0.75),
    "MAC Address": ("STRICT_FORMAT", 0.80),
    "Email Address": ("STRICT_FORMAT", 0.85),
    "IP Address": ("STANDARD_PARSE", 0.70),
    "Date of Birth": ("PLAUSIBILITY", 0.70),
    "Phone Number": ("BASIC_STRUCTURE", 0.45),
}


def validator_strength(entity: str) -> Tuple[str, float]:
    return VALIDATOR_STRENGTH.get(entity, ("NONE", 0.0))


def value_signature(value: str) -> str:
    """Structural signature used for diversity and near-duplicate suppression."""
    s = str(value).strip()
    mapped = []
    for ch in s:
        if ch.isdigit():
            mapped.append("9")
        elif ch.isupper():
            mapped.append("A")
        elif ch.islower():
            mapped.append("a")
        else:
            mapped.append(ch)
    signature = "".join(mapped)
    signature = re.sub(r"9{3,}", lambda m: f"9{{{len(m.group())}}}", signature)
    signature = re.sub(r"A{3,}", lambda m: f"A{{{len(m.group())}}}", signature)
    signature = re.sub(r"a{3,}", lambda m: f"a{{{len(m.group())}}}", signature)
    return signature[:100]


def _phrase_in_normalized_text(phrase: str, normalized: str) -> bool:
    """Token-boundary phrase matching; avoids substring collisions such as pan in company."""
    phrase = normalize_column(phrase)
    return bool(re.search(rf"(?:^|\s){re.escape(phrase)}(?:$|\s)", normalized))


def semantic_scores(column: str) -> Tuple[Dict[str, float], Set[str], str]:
    """
    Column-name semantic evidence with explicit ambiguity handling.
    Schema semantics are evidence, never proof.
    """
    norm = normalize_column(column)
    tokens = set(norm.split())
    scores: Dict[str, float] = {}

    for entity, terms in ENTITY_SEMANTIC_TERMS.items():
        score = 0.0
        for term in terms:
            term_norm = normalize_column(term)
            term_tokens = set(term_norm.split())
            if norm == term_norm:
                score = max(score, 1.0)
            elif _phrase_in_normalized_text(term_norm, norm):
                score = max(score, 0.90)
            elif term_tokens and term_tokens.issubset(tokens):
                score = max(score, 0.72)
        if score:
            if any(x in tokens for x in {"or", "maybe", "possible", "mixed", "alternate"}):
                score *= 0.88
            scores[entity] = round(score, 4)

    negatives: Set[str] = set()
    for entity, terms in NEGATIVE_ENTITY_TERMS.items():
        if any(_phrase_in_normalized_text(term, norm) for term in terms):
            negatives.add(entity)

    generic = norm in GENERIC_ID_TERMS
    meaning = "generic identifier or sequence field" if generic else (
        max(scores, key=scores.get) if scores else "unknown field semantics"
    )
    return scores, negatives, meaning


# ---------------------------------------------------------------------------
# Independent custom regex detector
# ---------------------------------------------------------------------------

REGEX_SPECS = [
    ("PAN Number", "pan_strict", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", 0.85),
    ("Aadhaar Number", "aadhaar", r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b", 0.80),
    ("Email Address", "email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", 0.95),
    ("Social Security Number", "ssn_dashes",
     r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b", 0.90),
    ("Social Security Number", "ssn_compact",
     r"\b(?!000|666|9\d{2})(?!000000000)\d{9}\b", 0.60),
    ("IP Address", "ipv4",
     r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b", 0.85),
    ("IP Address", "ipv6_full", r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b", 0.85),
    ("Credit Card Number", "cc_groups",
     r"\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{3,4}\b", 0.82),
    ("Phone Number", "phone_intl", r"\+[1-9]\d{6,14}\b", 0.78),
    ("Phone Number", "phone_in_mobile", r"\b[6-9]\d{9}\b", 0.75),
    ("Phone Number", "phone_us", r"\b(?:\+1[\s\-]?)?\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}\b", 0.72),
    ("Passport Number", "passport_in", r"\b[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9]\b", 0.78),
    ("Passport Number", "passport_generic", r"\b[A-Z]{1,2}\d{6,9}\b", 0.58),
    ("IBAN", "iban", r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})\b", 0.82),
    ("IFSC Code", "ifsc", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", 0.88),
    ("MAC Address", "mac_colon", r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b", 0.90),
    ("MAC Address", "mac_hyphen", r"\b(?:[0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2}\b", 0.90),
    ("Crypto Wallet Address", "btc_legacy", r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b", 0.72),
    ("Crypto Wallet Address", "eth", r"\b0x[a-fA-F0-9]{40}\b", 0.88),
    ("Voter ID", "voter_id", r"\b[A-Z]{3}[0-9]{7}\b", 0.70),
    ("Driver License Number", "dl_in", r"\b[A-Z]{2}[0-9]{2}[\s]?[0-9]{11}\b", 0.68),
    ("Driver License Number", "dl_us", r"\b[A-Z][0-9]{7}\b", 0.55),
    ("Bank Account Number", "bank_acc", r"\b\d{9,18}\b", 0.45),
    ("Date of Birth", "dob_iso", r"\b(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}\b", 0.68),
    ("Date of Birth", "dob_dmy", r"\b\d{2}[-/]\d{2}[-/](?:19|20)\d{2}\b", 0.65),

    # Enterprise policy identifiers: intentionally narrow prefixes/structures.
    ("Employee Identifier", "employee_id", r"\b(?:EMP|EMPL|STAFF)[-_ ]?[A-Z0-9]{3,20}\b", 0.82),
    ("Customer Identifier", "customer_id", r"\b(?:CUST|CUSTOMER|CLIENT)[-_ ]?[A-Z0-9]{3,24}\b", 0.82),
    ("User Identifier", "user_id", r"\b(?:USR|USER)[-_ ]?[A-Z0-9]{3,24}\b", 0.80),
    ("Contract Number", "contract_number", r"\b(?:CTR|CONTRACT|AGR)[-/ _]?[A-Z0-9][A-Z0-9/_-]{4,30}\b", 0.82),
    ("Device Identifier", "device_serial", r"\b(?:SN|SERIAL|DEV|DEVICE)[-_ ]?[A-Z0-9]{5,30}\b", 0.82),
]

COMPILED_REGEX = [
    (entity, name, re.compile(pattern), score)
    for entity, name, pattern, score in REGEX_SPECS
]


def regex_scan_value(value: str) -> List[Detection]:
    detections = []
    for entity, name, pattern, score in COMPILED_REGEX:
        for match in pattern.finditer(value):
            vp = validator_for(entity, match.group(0))
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


# ---------------------------------------------------------------------------
# Native Presidio/spaCy engine (custom regex is deliberately NOT injected)
# ---------------------------------------------------------------------------

def build_presidio_engine() -> AnalyzerEngine:
    logger.info("Initialising native Presidio + spaCy model '%s' ...", PRESIDIO_NLP_MODEL)
    t0 = time.perf_counter()

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": PRESIDIO_NLP_MODEL}],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {
                "PER": "PERSON",
                "PERSON": "PERSON",
                "LOC": "LOCATION",
                "GPE": "LOCATION",
                "ORG": "ORGANIZATION",
            },
            "low_confidence_score_multiplier": 0.4,
            "low_confidence_entities": [],
            "labels_to_ignore": [
                "CARDINAL", "DATE", "MONEY", "ORDINAL", "PERCENT",
                "QUANTITY", "TIME", "WORK_OF_ART", "EVENT", "FAC", "PRODUCT",
                "LANGUAGE",
            ],
        },
    })
    analyzer = AnalyzerEngine(
        nlp_engine=provider.create_engine(),
        supported_languages=["en"],
    )
    logger.info("Presidio ready (%.2f s)", time.perf_counter() - t0)
    return analyzer


def presidio_scan_value(analyzer: AnalyzerEngine, value: str) -> List[Detection]:
    try:
        results = analyzer.analyze(text=value, language="en")
    except Exception as exc:
        logger.debug("Presidio failed on %r: %s", value[:60], exc)
        return []

    out = []
    for result in results:
        entity = ENTITY_LABEL_MAP.get(result.entity_type, result.entity_type.replace("_", " ").title())
        recognizer = getattr(result, "recognition_metadata", None) or {}
        recognizer_name = recognizer.get("recognizer_name", "presidio_native")
        span_text = value[result.start:result.end]
        out.append(Detection(
            value=value,
            entity=entity,
            source="presidio",
            recognizer=str(recognizer_name),
            score=float(result.score),
            start=int(result.start),
            end=int(result.end),
            validator_pass=validator_for(entity, span_text),
        ))
    return out


_STAGE1_CACHE_MAX = 50000
_PRESIDIO_CACHE: "OrderedDict[str, List[Detection]]" = OrderedDict()
_REGEX_CACHE: "OrderedDict[str, List[Detection]]" = OrderedDict()


def _cache_get(cache: OrderedDict, key: str):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key: str, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _STAGE1_CACHE_MAX:
        cache.popitem(last=False)


def run_stage1(
    analyzer: AnalyzerEngine,
    unique_values: Sequence[str],
) -> Dict[str, ValueEvidence]:
    evidence = {}
    for value in unique_values:
        if len(value) > 12000:
            value = value[:6000] + "\n...[TRUNCATED]...\n" + value[-6000:]
        p = _cache_get(_PRESIDIO_CACHE, value)
        if p is None:
            p = presidio_scan_value(analyzer, value)
            _cache_put(_PRESIDIO_CACHE, value, p)

        r = _cache_get(_REGEX_CACHE, value)
        if r is None:
            r = regex_scan_value(value)
            _cache_put(_REGEX_CACHE, value, r)

        if p or r:
            evidence[value] = ValueEvidence(
                value=value,
                presidio=list(p),
                regex=list(r),
            )
    return evidence


# ---------------------------------------------------------------------------
# Stratified evidence sampling
# ---------------------------------------------------------------------------

def dominant_entity_for_value(ev: ValueEvidence) -> Optional[str]:
    ds = ev.presidio + ev.regex
    if not ds:
        return None
    weighted = defaultdict(float)
    for d in ds:
        reliability = detection_reliability(d)
        validation_bonus = 0.18 if d.validator_pass is True else (-0.20 if d.validator_pass is False else 0.0)
        weighted[d.entity] += max(0.0, d.score * reliability + validation_bonus)
    return max(weighted, key=weighted.get) if weighted else None


def sample_priority(ev: ValueEvidence) -> float:
    ds = ev.presidio + ev.regex
    if not ds:
        return 0.0
    score = max(d.score * detection_reliability(d) for d in ds)
    pe = {d.entity for d in ev.presidio}
    re_ = {d.entity for d in ev.regex}
    if pe and re_:
        score += 0.55
    if pe & re_:
        score += 0.35
    elif pe and re_:
        score += 0.30
    if any(d.validator_pass is True for d in ds):
        score += 0.30
    if any(d.validator_pass is False for d in ds):
        score += 0.22
    if 0.35 <= ev.max_score <= 0.72:
        score += 0.15
    return score


def stratified_sample(
    evidence: Dict[str, ValueEvidence],
    limit: int = MAX_GLINER_SAMPLES,
) -> List[ValueEvidence]:
    """
    Deterministic stratified sampling:
    1) reserve coverage for credible entity families;
    2) reserve source/agreement strata;
    3) fill by structural diversity and ambiguity.
    """
    candidates = list(evidence.values())
    if len(candidates) <= limit:
        return sorted(candidates, key=lambda e: (-sample_priority(e), e.value))

    selected: List[ValueEvidence] = []
    seen_values: Set[str] = set()
    seen_signatures: Counter = Counter()

    def add(ev: ValueEvidence, signature_cap: int = 2) -> bool:
        if ev.value in seen_values:
            return False
        sig = value_signature(ev.value)
        if seen_signatures[sig] >= signature_cap:
            return False
        selected.append(ev)
        seen_values.add(ev.value)
        seen_signatures[sig] += 1
        return True

    ranked = sorted(candidates, key=lambda e: (-sample_priority(e), e.value))

    # Credible entity-family reservation using per-detection reliability.
    entity_groups: Dict[str, List[ValueEvidence]] = defaultdict(list)
    entity_strength: Dict[str, float] = defaultdict(float)
    for ev in ranked:
        per_entity = defaultdict(float)
        for d in ev.presidio + ev.regex:
            per_entity[d.entity] = max(
                per_entity[d.entity],
                d.score * detection_reliability(d)
            )
        for entity, strength in per_entity.items():
            if strength >= 0.28:
                entity_groups[entity].append(ev)
                entity_strength[entity] = max(entity_strength[entity], strength)

    for entity in sorted(entity_groups, key=lambda e: (-entity_strength[e], e)):
        if len(selected) >= limit:
            break
        for ev in entity_groups[entity]:
            if add(ev):
                break

    # Source/agreement strata.
    strata = {"agree": [], "disagree": [], "presidio_only": [], "regex_only": []}
    for ev in ranked:
        pe = {d.entity for d in ev.presidio}
        re_ = {d.entity for d in ev.regex}
        if pe and re_:
            strata["agree" if pe & re_ else "disagree"].append(ev)
        elif pe:
            strata["presidio_only"].append(ev)
        elif re_:
            strata["regex_only"].append(ev)

    for key in ("agree", "disagree", "presidio_only", "regex_only"):
        if len(selected) >= limit:
            break
        for ev in strata[key]:
            if add(ev):
                break

    for ev in ranked:
        if len(selected) >= limit:
            break
        add(ev, signature_cap=2)

    for ev in ranked:
        if len(selected) >= limit:
            break
        add(ev, signature_cap=10**9)

    return selected[:limit]


# ---------------------------------------------------------------------------
# GLiNER semantic evidence
# ---------------------------------------------------------------------------

def load_gliner() -> GLiNER:
    preferred_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading GLiNER '%s' on %s ...", GLINER_MODEL_NAME, preferred_device)
    t0 = time.perf_counter()
    model = GLiNER.from_pretrained(GLINER_MODEL_NAME)

    if preferred_device == "cuda":
        try:
            model = model.to("cuda")
        except Exception as exc:
            logger.warning("GLiNER CUDA placement failed; falling back to CPU: %s", exc)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            model = model.to("cpu")
    else:
        try:
            model = model.to("cpu")
        except Exception:
            logger.debug("Explicit GLiNER CPU placement unavailable; using model default.")

    model.eval()
    logger.info("GLiNER ready (%.2f s)", time.perf_counter() - t0)
    return model


def format_detector_evidence(ds: Sequence[Detection]) -> str:
    if not ds:
        return "none"
    parts = []
    for d in sorted(ds, key=lambda x: -x.score)[:3]:
        val = ""
        if d.validator_pass is True:
            val = ", validator=pass"
        elif d.validator_pass is False:
            val = ", validator=fail"
        parts.append(f"{d.entity} ({d.score:.2f}{val})")
    return "; ".join(parts)


def gliner_analyze_sample(
    model: GLiNER,
    original_col: str,
    norm_col: str,
    semantic_meaning: str,
    ev: ValueEvidence,
) -> GlinerEvidence:
    """
    GLiNER is used only as zero-shot NER. The value-only view is independent.
    The context view adds schema context but never detector predictions.
    """
    context_label = norm_col or normalize_column(original_col)
    if semantic_meaning and semantic_meaning != "unknown field semantics":
        context_prefix = f"Field '{context_label}' ({semantic_meaning}) value: "
    else:
        context_prefix = f"Field '{context_label}' value: "

    views = [
        ("value", "", ev.value),
        ("context", context_prefix, context_prefix + ev.value),
    ]
    by_view: Dict[str, Dict[str, float]] = {"value": {}, "context": {}}
    inference_ok = True

    for view_name, prefix, view_text in views:
        value_start = len(prefix)
        value_end = len(view_text)
        try:
            entities = model.predict_entities(
                view_text, GLINER_LABELS, threshold=GLINER_THRESHOLD
            )
        except RuntimeError as exc:
            # CUDA OOM/failure: retry this inference on CPU once.
            if "cuda" in str(exc).lower() or "out of memory" in str(exc).lower():
                logger.warning("GLiNER CUDA inference failed; retrying model on CPU.")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    model.to("cpu")
                    entities = model.predict_entities(
                        view_text, GLINER_LABELS, threshold=GLINER_THRESHOLD
                    )
                except Exception as retry_exc:
                    logger.warning("GLiNER CPU retry failed: %s", retry_exc)
                    inference_ok = False
                    continue
            else:
                logger.warning("GLiNER inference failed: %s", exc)
                inference_ok = False
                continue
        except Exception as exc:
            logger.warning("GLiNER inference failed: %s", exc)
            inference_ok = False
            continue

        value_lower = ev.value.casefold()
        for item in entities:
            raw_label = normalize_text(str(item.get("label", ""))).strip().casefold()
            label = CANONICAL.get(raw_label, str(item.get("label", "")).strip().title())
            score = float(item.get("score", 0.0))
            ent_start, ent_end = item.get("start"), item.get("end")
            entity_text = normalize_text(str(item.get("text", ""))).casefold()
            overlaps = (
                ent_start is not None and ent_end is not None
                and int(ent_start) < value_end and int(ent_end) > value_start
            )
            text_match = bool(entity_text) and entity_text in value_lower
            if overlaps or text_match:
                by_view[view_name][label] = max(
                    by_view[view_name].get(label, 0.0), score
                )

    value_predictions = sorted(by_view["value"].items(), key=lambda x: (-x[1], x[0]))
    context_predictions = sorted(by_view["context"].items(), key=lambda x: (-x[1], x[0]))

    labels = set(by_view["value"]) | set(by_view["context"])
    combined = {}
    for label in labels:
        vs = by_view["value"].get(label, 0.0)
        cs = by_view["context"].get(label, 0.0)
        # Value extraction is independent semantic evidence. Context extraction
        # is schema-conditioned and is deliberately capped to prevent label leakage.
        combined[label] = (
            min(1.0, 0.88 * vs + 0.12 * min(cs, 0.85))
            if vs > 0 else
            0.12 * min(cs, 0.85)
        )

    predictions = sorted(combined.items(), key=lambda x: (-x[1], x[0]))
    best_entity = predictions[0][0] if predictions else None
    best_score = predictions[0][1] if predictions else 0.0
    return GlinerEvidence(
        ev.value, value_predictions, context_predictions,
        predictions, best_entity, best_score, inference_ok
    )


def run_gliner_stage(
    model: GLiNER,
    column: str,
    samples: Sequence[ValueEvidence],
) -> List[GlinerEvidence]:
    norm = normalize_column(column)
    _, _, meaning = semantic_scores(column)
    return [
        gliner_analyze_sample(model, column, norm, meaning, ev)
        for ev in samples
    ]


# ---------------------------------------------------------------------------
# Column evidence aggregation
# ---------------------------------------------------------------------------


def detection_reliability(d: Detection) -> float:
    if d.source == "regex":
        return PATTERN_RELIABILITY.get(d.pattern or "", 0.50)
    return PRESIDIO_ENTITY_RELIABILITY.get(d.entity, 0.70)


def weighted_entity_support(
    evidence: Dict[str, ValueEvidence],
    total_unique: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    p_sum = defaultdict(float)
    r_sum = defaultdict(float)
    u_sum = defaultdict(float)
    for ev in evidence.values():
        p_best, r_best = {}, {}
        for d in ev.presidio:
            p_best[d.entity] = max(p_best.get(d.entity, 0.0), detection_reliability(d))
        for d in ev.regex:
            r_best[d.entity] = max(r_best.get(d.entity, 0.0), detection_reliability(d))
        for entity, value in p_best.items():
            p_sum[entity] += value
        for entity, value in r_best.items():
            r_sum[entity] += value
        for entity in set(p_best) | set(r_best):
            u_sum[entity] += max(p_best.get(entity, 0.0), r_best.get(entity, 0.0))
    denom = max(1, total_unique)
    return (
        {k: v / denom for k, v in p_sum.items()},
        {k: v / denom for k, v in r_sum.items()},
        {k: v / denom for k, v in u_sum.items()},
    )


def collision_penalty_for_entity(
    entity: str,
    semantic_map: Dict[str, float],
    validator: Optional[float],
) -> float:
    own_sem = semantic_map.get(entity, 0.0)
    strongest_other = 0.0
    for group in COLLISION_GROUPS:
        if entity in group:
            for other in group - {entity}:
                strongest_other = max(strongest_other, semantic_map.get(other, 0.0))
    gap = max(0.0, strongest_other - own_sem)
    protection = 0.55 if validator is not None and validator >= 0.95 else 1.0
    return min(0.32, 0.30 * gap * protection)


def ontology_reduce_entities(entities: Sequence[str], profiles: Dict[str, dict]) -> List[str]:
    ordered, seen = [], set()
    for entity in entities:
        if entity not in seen:
            ordered.append(entity)
            seen.add(entity)
    for child, parent in ENTITY_PARENT.items():
        if child in seen and parent in seen:
            cp, pp = profiles.get(child, {}), profiles.get(parent, {})
            if (
                cp.get("accepted")
                or (
                    cp.get("semantic", 0.0) >= 0.70
                    and cp.get("related_support", 0.0) >= 0.20
                )
            ):
                ordered = [x for x in ordered if x != parent]
                seen.discard(parent)
    return ordered


def entity_support(
    evidence: Dict[str, ValueEvidence],
    total_unique: int,
) -> Tuple[Counter, Counter, Counter]:
    p = Counter()
    r = Counter()
    union = Counter()
    for ev in evidence.values():
        pe = {d.entity for d in ev.presidio}
        re_ = {d.entity for d in ev.regex}
        for entity in pe:
            p[entity] += 1
        for entity in re_:
            r[entity] += 1
        for entity in pe | re_:
            union[entity] += 1
    return p, r, union


def validator_rate_for_entity(
    evidence: Dict[str, ValueEvidence],
    entity: str,
) -> Optional[float]:
    """Return per-value validator pass rate; None means no validator applies."""
    per_value = []
    for ev in evidence.values():
        outcomes = [
            d.validator_pass
            for d in ev.regex + ev.presidio
            if d.entity == entity and d.validator_pass is not None
        ]
        if outcomes:
            per_value.append(any(outcomes))
    return sum(per_value) / len(per_value) if per_value else None

def detector_agreement_for_entity(
    evidence: Dict[str, ValueEvidence],
    entity: str,
) -> float:
    relevant_weight = 0.0
    agreement_weight = 0.0
    for ev in evidence.values():
        p = max(
            (detection_reliability(d) for d in ev.presidio if d.entity == entity),
            default=0.0,
        )
        r = max(
            (detection_reliability(d) for d in ev.regex if d.entity == entity),
            default=0.0,
        )
        if p > 0 or r > 0:
            relevant_weight += max(p, r)
            if p > 0 and r > 0:
                agreement_weight += min(p, r)
    return agreement_weight / relevant_weight if relevant_weight else 0.0


def gliner_metrics(
    gliner_results: Sequence[GlinerEvidence],
    entity: str,
    sampled_evidence: Optional[Sequence[ValueEvidence]] = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    Return value-confirmation, value-support, context-support, consistency,
    mean value score, and inference success rate. Context-only extraction is
    schema evidence, not entity confirmation.
    """
    if not gliner_results:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    family = GLINER_ENTITY_FAMILIES.get(entity, {entity})
    relevant = list(range(len(gliner_results)))
    if sampled_evidence and len(sampled_evidence) == len(gliner_results):
        idxs = [i for i, ev in enumerate(sampled_evidence) if entity in ev.entities]
        if idxs:
            relevant = idxs

    value_hits = context_hits = successful = 0
    value_scores: List[float] = []
    best_value_predictions: List[str] = []

    for i in relevant:
        g = gliner_results[i]
        if not g.inference_ok:
            continue
        successful += 1
        v = [(e, s) for e, s in g.value_predictions if e in family]
        x = [(e, s) for e, s in g.context_predictions if e in family]
        if v:
            value_hits += 1
            value_scores.append(max(s for _, s in v))
        if x:
            context_hits += 1
        if g.value_predictions:
            best_value_predictions.append(g.value_predictions[0][0])

    denom = max(1, successful)
    value_confirm = value_hits / denom
    context_confirm = context_hits / denom
    consistency = (
        sum(pred in family for pred in best_value_predictions) / len(best_value_predictions)
        if best_value_predictions else 0.0
    )
    mean_score = sum(value_scores) / len(value_scores) if value_scores else 0.0
    success_rate = successful / max(1, len(relevant))
    confirmation = value_confirm
    return confirmation, value_confirm, context_confirm, consistency, mean_score, success_rate


def candidate_entities(
    evidence: Dict[str, ValueEvidence],
    gliner_results: Sequence[GlinerEvidence],
    semantic: Dict[str, float],
) -> Set[str]:
    out = set(semantic)
    for ev in evidence.values():
        out.update(ev.entities)
    for g in gliner_results:
        out.update(entity for entity, _ in g.predictions)
        out.update(entity for entity, _ in g.value_predictions)
        out.update(entity for entity, _ in g.context_predictions)
    return out


def entity_threshold(entity: str) -> float:
    # Evidence-score thresholds after entity-specific weighting.
    return {
        "Person Name": 0.56,
        "Address": 0.58,
        "Location": 0.58,
        "Date of Birth": 0.56,
        "Aadhaar Number": 0.62,
        "Credit Card Number": 0.62,
        "Bank Account Number": 0.68,
        "Passport Number": 0.58,
        "Driver License Number": 0.60,
        "PAN Number": 0.58,
        "Phone Number": 0.58,
    }.get(entity, 0.56)


def _profile_score(
    entity: str,
    p_support: float,
    r_support: float,
    union_support: float,
    agreement: float,
    g_confirm: float,
    g_consistency: float,
    g_mean: float,
    semantic: float,
    validator: Optional[float],
    negative: float,
) -> float:
    """Bounded evidence fusion; inputs are reliability-weighted supports."""
    p = min(1.0, p_support / 0.50)
    r = min(1.0, r_support / 0.50)
    u = min(1.0, union_support / 0.50)

    if entity in NATURAL_LANGUAGE_ENTITIES:
        score = 0.46*p + 0.24*semantic + 0.18*g_confirm + 0.06*g_consistency + 0.06*u
    elif entity == "Date of Birth":
        score = 0.24*r + 0.10*p + 0.34*semantic + 0.10*g_confirm + 0.08*u
        if validator is not None:
            score += 0.14*validator
    elif entity in {"Aadhaar Number", "Credit Card Number", "IBAN"}:
        score = 0.30*r + 0.12*p + 0.16*semantic + 0.08*agreement + 0.08*g_confirm + 0.08*u
        if validator is not None:
            score += 0.24*validator - 0.45*(1.0-validator)
    elif entity in {"PAN Number", "IFSC Code", "MAC Address", "Email Address",
                    "IP Address", "Crypto Wallet Address"}:
        score = 0.30*r + 0.16*p + 0.18*semantic + 0.08*agreement + 0.08*g_confirm + 0.08*u
        if validator is not None:
            score += 0.12*validator - 0.24*(1.0-validator)
    elif entity in {"Bank Account Number", "Passport Number", "Driver License Number",
                    "Phone Number", "Social Security Number", "Voter ID"}:
        score = 0.26*r + 0.16*p + 0.25*semantic + 0.08*agreement + 0.10*g_confirm + 0.05*u
        if validator is not None:
            score += 0.10*validator - 0.18*(1.0-validator)
    elif entity in POLICY_IDENTIFIER_ENTITIES:
        score = 0.24*r + 0.10*p + 0.38*semantic + 0.12*g_confirm + 0.10*u
    else:
        score = 0.22*p + 0.22*r + 0.24*semantic + 0.08*agreement + 0.14*g_confirm + 0.10*u

    score -= 0.24*negative
    return max(0.0, min(1.0, score))


def regex_specificity_for_entity(
    evidence: Dict[str, ValueEvidence],
    entity: str,
) -> float:
    specs = []
    for ev in evidence.values():
        for d in ev.regex:
            if d.entity != entity:
                continue
            if d.pattern in STRICT_PATTERN_NAMES:
                specs.append(1.0)
            elif d.pattern in MODERATE_PATTERN_NAMES:
                specs.append(0.72)
            else:
                specs.append(0.38)
    return sum(specs) / len(specs) if specs else 0.0


def semantic_contradiction_penalty(
    entity: str,
    semantic_scores_map: Dict[str, float],
) -> float:
    """Penalty when another mutually-exclusive entity has stronger column semantics."""
    penalty = 0.0
    for semantic_entity, strength in semantic_scores_map.items():
        contradictions = SEMANTIC_CONTRADICTIONS.get(semantic_entity, set())
        if entity in contradictions and strength >= 0.55:
            penalty = max(penalty, 0.30 * strength)
    return penalty


def count_support_for_entity(
    evidence: Dict[str, ValueEvidence],
    entity: str,
) -> int:
    return sum(entity in ev.entities for ev in evidence.values())


def structural_consistency_for_entity(
    evidence: Dict[str, ValueEvidence],
    entity: str,
) -> float:
    """Fraction of entity candidate values sharing the dominant structural signature."""
    signatures = []
    for value, ev in evidence.items():
        if entity in ev.entities:
            signatures.append(value_signature(value))
    if not signatures:
        return 0.0
    counts = Counter(signatures)
    return counts.most_common(1)[0][1] / len(signatures)


def strict_or_moderate_regex_support(
    evidence: Dict[str, ValueEvidence],
    entity: str,
    total_unique: int,
) -> float:
    values = set()
    for value, ev in evidence.items():
        for d in ev.regex:
            if (
                d.entity == entity
                and d.pattern in (STRICT_PATTERN_NAMES | MODERATE_PATTERN_NAMES)
            ):
                values.add(value)
    return len(values) / max(1, total_unique)


SPARSE_HIGH_SPECIFICITY_ENTITIES = {
    "Email Address",
    "Credit Card Number",
    "Aadhaar Number",
    "PAN Number",
    "IBAN",
    "IFSC Code",
    "MAC Address",
    "IP Address",
    "Crypto Wallet Address",
}



def stage1_candidate_distribution(
    evidence: Dict[str, ValueEvidence],
    total_unique: int,
) -> str:
    wp, wr, wu = weighted_entity_support(evidence, total_unique)
    entities = set(wp) | set(wr) | set(wu)
    ranked = []
    for entity in entities:
        weighted = 0.40 * wp.get(entity, 0.0) + 0.40 * wr.get(entity, 0.0) + 0.20 * wu.get(entity, 0.0)
        ranked.append((entity, weighted, wp.get(entity, 0.0), wr.get(entity, 0.0), wu.get(entity, 0.0)))
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return " | ".join(
        f"{e}:w={w:.2f},p={p:.2f},r={r:.2f},u={u:.2f}"
        for e, w, p, r, u in ranked
    )


def secondary_entity_acceptance(profile: dict, total_unique: int) -> bool:
    if profile["hard_reject"] is not None:
        return False
    entity = profile["entity"]
    count = profile["support_count"]
    support = profile["union_support"]
    validator = profile["validator"]
    specificity = profile["regex_specificity"]

    if profile["accepted"]:
        return True
    if entity in {"Email Address", "Credit Card Number", "Aadhaar Number", "IBAN", "MAC Address"}:
        return count >= 2 and validator is not None and validator >= 0.95 and specificity >= 0.85
    if entity == "Phone Number":
        return (
            (count >= max(3, int(0.05 * total_unique)) and support >= 0.05 and profile["semantic"] >= 0.55)
            or
            (count >= max(5, int(0.10 * total_unique)) and profile["agreement"] >= 0.50
             and (validator or 0.0) >= 0.90 and profile["negative"] == 0.0)
        )
    if entity == "Person Name":
        return (
            count >= max(5, int(0.10 * total_unique))
            and profile["p_support"] >= 0.15
            and profile["row_support"] >= 0.10
            and (profile["semantic"] >= 0.55 or profile["g_value"] >= 0.45)
            and profile["negative"] == 0.0
        )
    if entity in {"Address", "Location"}:
        return (
            count >= max(3, int(0.05 * total_unique))
            and profile["p_support"] >= 0.10
            and (profile["g_value"] >= 0.25 or profile["semantic"] >= 0.70)
        )
    return False


def sample_coverage(
    samples: Sequence[ValueEvidence],
    evidence: Dict[str, ValueEvidence],
) -> Tuple[int, int, float]:
    total_unique = max(1, len(evidence))
    _, _, wu = weighted_entity_support(evidence, total_unique)
    credible = {e for e, s in wu.items() if s >= 0.08}
    sampled = set()
    for ev in samples:
        sampled.update(ev.entities)
    covered = len(credible & sampled)
    total = len(credible)
    return covered, total, (covered / total if total else 1.0)


def sparse_high_specificity_accept(
    entity: str,
    support_count: int,
    row_support: float,
    validator: Optional[float],
    regex_specificity: float,
    value_gliner_confirm: float,
    semantic: float,
    negative: float,
) -> bool:
    """Conservative sparse PII rescue for high-specificity entities only."""
    if entity not in SPARSE_HIGH_SPECIFICITY_ENTITIES:
        return False
    if not (0.0 < row_support <= SPARSE_PREVALENCE_MAX):
        return False
    if support_count < 1 or validator is None or validator < 0.95:
        return False
    if regex_specificity < 0.85 or negative > 0:
        return False

    kind, strength = validator_strength(entity)
    if kind == "CHECKSUM":
        return support_count >= 1 and (semantic >= 0.55 or value_gliner_confirm >= 0.50 or support_count >= 2)
    if entity in {"Email Address", "MAC Address", "IBAN", "IFSC Code", "Crypto Wallet Address"}:
        return support_count >= 1 and (value_gliner_confirm >= 0.50 or support_count >= 2)
    # PAN and IP are collision-prone despite valid structure.
    return semantic >= 0.55 or value_gliner_confirm >= 0.75



def aggregate_column(
    column: str,
    evidence: Dict[str, ValueEvidence],
    gliner_results: Sequence[GlinerEvidence],
    total_unique: int,
    sampled_evidence: Optional[Sequence[ValueEvidence]] = None,
    row_values: Optional[Sequence[str]] = None,
) -> dict:
    sem, negative_entities, meaning = semantic_scores(column)
    p_counts, r_counts, u_counts = entity_support(evidence, total_unique)
    wp_support, wr_support, wu_support = weighted_entity_support(evidence, total_unique)
    candidates = candidate_entities(evidence, gliner_results, sem)
    if not candidates:
        return clean_result(column, "No supported PII evidence.")

    norm = normalize_column(column)
    generic_id = norm in GENERIC_ID_TERMS
    scored = []

    for entity in sorted(candidates):
        raw_p = p_counts[entity] / total_unique
        raw_r = r_counts[entity] / total_unique
        raw_u = u_counts[entity] / total_unique
        p_support = wp_support.get(entity, 0.0)
        r_support = wr_support.get(entity, 0.0)
        union_support = wu_support.get(entity, 0.0)

        related_support = 0.0
        if entity == "Address":
            related_support = wu_support.get("Location", 0.0)
            # spaCy/Presidio commonly emits LOCATION for complete addresses.
            # Transfer partial evidence and let schema/value evidence resolve subtype.
            if related_support > 0:
                union_support = max(union_support, 0.72 * related_support)
                p_support = max(p_support, 0.72 * wp_support.get("Location", 0.0))
                raw_u = max(raw_u, 0.72 * (u_counts["Location"] / total_unique))
                raw_p = max(raw_p, 0.72 * (p_counts["Location"] / total_unique))
        elif entity == "Location":
            related_support = wu_support.get("Address", 0.0)

        agreement = detector_agreement_for_entity(evidence, entity)
        g_combined, g_value, g_context, g_consistency, g_mean, g_success = gliner_metrics(
            gliner_results, entity, sampled_evidence
        )
        validator = validator_rate_for_entity(evidence, entity)
        semantic = sem.get(entity, 0.0)
        negative = 1.0 if entity in negative_entities else 0.0
        regex_specificity = regex_specificity_for_entity(evidence, entity)
        contradiction_penalty = semantic_contradiction_penalty(entity, sem)
        collision_penalty = collision_penalty_for_entity(entity, sem, validator)
        support_count = count_support_for_entity(evidence, entity)
        structural_consistency = structural_consistency_for_entity(evidence, entity)
        specific_regex_support = strict_or_moderate_regex_support(evidence, entity, total_unique)

        if row_values:
            candidate_values = {v for v, ev in evidence.items() if entity in ev.entities}
            if entity == "Address":
                candidate_values |= {
                    v for v, ev in evidence.items()
                    if "Location" in ev.entities and sem.get("Address", 0.0) >= 0.55
                }
            row_support = sum(v in candidate_values for v in row_values) / max(1, len(row_values))
        else:
            row_support = raw_u

        if entity in NATURAL_LANGUAGE_ENTITIES:
            effective_gliner = min(1.0, 0.90*g_value + 0.10*min(g_context, 0.70))
        elif entity in OPAQUE_STRUCTURED_ENTITIES:
            effective_gliner = min(0.22, 0.90*g_value + 0.04*min(g_context, 0.70))
        else:
            effective_gliner = min(0.45, 0.88*g_value + 0.07*min(g_context, 0.70))

        score = _profile_score(
            entity, p_support, r_support, union_support, agreement,
            effective_gliner, g_consistency, g_mean, semantic, validator, negative,
        )

        if entity in {"Address", "Location"} and related_support > 0:
            score += 0.08 * min(1.0, related_support / 0.40)

        score += 0.04 * min(1.0, row_support / 0.50)
        score -= contradiction_penalty
        score -= collision_penalty

        if entity == "Person Name":
            org_support = wu_support.get("Organization", 0.0)
            if org_support >= 0.20 and semantic < 0.60:
                score -= min(0.20, 0.24 * org_support)

        validator_kind, _ = validator_strength(entity)
        if validator is not None:
            if validator_kind == "BASIC_STRUCTURE":
                score -= 0.05 * validator
            elif validator_kind == "STRICT_FORMAT" and semantic < 0.55 and g_value < 0.50:
                score -= 0.04 * validator

        if raw_r > 0 and regex_specificity < 0.50 and semantic < 0.55 and effective_gliner < 0.35:
            score *= 0.58

        # Sparse genuine PII must not be erased by column prevalence. A single
        # structurally valid phone can make the column PII-bearing when schema or
        # value-level semantic evidence corroborates it.
        sparse_phone = (
            entity == "Phone Number"
            and support_count >= 1
            and validator is not None and validator >= 0.95
            and regex_specificity >= 0.70
            and negative == 0.0
            and (semantic >= 0.55 or g_value >= 0.60 or agreement >= 0.50)
        )
        if support_count == 1 and total_unique >= 20 and semantic < 0.55 and effective_gliner < 0.45 and not sparse_phone:
            score *= 0.62

        hard_reject = None

        if entity == "Aadhaar Number":
            if validator is not None and validator < 0.80:
                hard_reject = "Aadhaar candidates fail Verhoeff validation."
            elif semantic < 0.55 and (validator or 0.0) < 0.95:
                score *= 0.65

        elif entity == "PAN Number":
            if entity in negative_entities and semantic < 0.55 and g_value < 0.50:
                hard_reject = "PAN-shaped values contradicted by code/token semantics."
            elif validator is not None and validator < 0.95:
                hard_reject = "PAN candidates fail strict format validation."

        elif entity == "Credit Card Number":
            if validator is None or validator < 0.85:
                hard_reject = "Credit-card candidates lack sufficient Luhn-valid support."

        elif entity == "Date of Birth":
            if semantic < 0.55:
                hard_reject = "Valid date structure lacks birth-related column semantics."
            elif validator is not None and validator < 0.70:
                hard_reject = "Birth-date candidates fail date plausibility validation."
            elif (
                semantic >= 0.70 and raw_u >= 0.20
                and validator is not None and validator >= 0.90
                and row_support >= 0.20
            ):
                score += 0.06 * min(1.0, raw_u) * min(1.0, row_support)

        elif entity == "Bank Account Number":
            if semantic < 0.55 and g_value < 0.50:
                hard_reject = "Long numeric pattern lacks bank-account semantics."

        elif entity == "Passport Number":
            passport_schema = semantic >= 0.70
            dl_schema = sem.get("Driver License Number", 0.0) >= 0.70
            if dl_schema and not passport_schema:
                score *= 0.42
            elif passport_schema and raw_u >= 0.40 and (
                specific_regex_support >= 0.30 or structural_consistency >= 0.60
            ):
                score += 0.08 * min(1.0, raw_u) * max(
                    specific_regex_support, structural_consistency
                )
            elif semantic < 0.50 and g_value < 0.45 and agreement < 0.30:
                score *= 0.52

        elif entity == "Driver License Number":
            passport_schema = sem.get("Passport Number", 0.0) >= 0.60
            if passport_schema:
                score *= 0.38
            elif semantic < 0.50 and g_value < 0.45 and agreement < 0.30:
                score *= 0.52

        elif entity == "Phone Number":
            if entity in negative_entities and semantic < 0.55 and g_value < 0.50:
                hard_reject = "Phone-shaped values contradicted by identifier/sequence semantics."
            elif semantic < 0.50 and g_value < 0.35 and agreement < 0.25:
                score *= 0.68

        elif entity == "IP Address":
            if entity in negative_entities and semantic < 0.55 and g_value < 0.50:
                hard_reject = "IP-shaped values contradicted by version/build semantics."
            elif validator is not None and validator < 0.90:
                hard_reject = "IP candidates fail deterministic parsing."

        sparse_accept = sparse_high_specificity_accept(
            entity, support_count, row_support, validator, regex_specificity,
            g_value, semantic, negative
        )
        if (sparse_accept or sparse_phone) and hard_reject is None:
            score = max(score, entity_threshold(entity) + 0.05)

        # Strong entity-specific evidence rescue: avoids threshold cliffs for
        # passport/DOB while still requiring schema + structure/plausibility.
        if entity == "Passport Number" and hard_reject is None:
            if semantic >= 0.70 and raw_u >= 0.20 and (structural_consistency >= 0.60 or specific_regex_support >= 0.20):
                score = max(score, threshold if 'threshold' in locals() else entity_threshold(entity))
        if entity == "Date of Birth" and hard_reject is None:
            if semantic >= 0.70 and (validator or 0.0) >= 0.90 and raw_u >= 0.10:
                score = max(score, entity_threshold(entity) + 0.02)

        if generic_id and entity in {
            "Phone Number", "Bank Account Number", "Aadhaar Number",
            "Credit Card Number", "Date of Birth",
        }:
            score *= 0.50

        if entity == "Address":
            if semantic >= 0.75:
                score += 0.10
            if semantic >= 0.75 and related_support >= 0.20:
                score += 0.10 * min(1.0, related_support / 0.60)
                score += 0.05 * min(1.0, row_support / 0.50)
        if entity == "Location" and sem.get("Address", 0.0) >= 0.75:
            score -= 0.20
        if entity == "Organization":
            if semantic >= 0.70:
                score += 0.10
            if raw_p >= 0.50 and g_value >= 0.30:
                score += 0.10 * min(1.0, raw_p)

        coarse_location = (
            entity == "Location"
            and any(_phrase_in_normalized_text(t, norm) for t in ("city", "state", "country", "district"))
            and not any(_phrase_in_normalized_text(t, norm) for t in ("address", "street", "home address", "residential address"))
        )
        policy_excluded = entity == "Organization" and not POLICY_TREAT_ORGANIZATION_AS_PII
        if coarse_location and not POLICY_TREAT_COARSE_LOCATION_AS_PII:
            policy_excluded = True

        threshold = entity_threshold(entity)
        policy_positive = False
        if entity in POLICY_TREAT_IDENTIFIERS_AS_PII:
            policy_accepts = POLICY_TREAT_IDENTIFIERS_AS_PII[entity]
            if not policy_accepts:
                policy_excluded = True
            else:
                policy_positive = (
                    (semantic >= 0.75 and raw_r >= 0.20 and regex_specificity >= 0.85)
                    or (raw_r >= 0.50 and regex_specificity >= 0.85)
                    or (semantic >= 0.80 and g_value >= 0.55)
                )
                if policy_positive and hard_reject is None:
                    score = max(score, threshold + 0.05)

        score = max(0.0, min(1.0, score))
        corroborated_org = (
            entity == "Organization"
            and raw_p >= 0.50
            and row_support >= 0.40
            and (g_value >= 0.30 or semantic >= 0.70)
        )
        detected = hard_reject is None and (
            score >= threshold or policy_positive or corroborated_org
        )
        policy_accepted = detected and not policy_excluded

        scored.append({
            "entity": entity, "score": score, "threshold": threshold,
            "detected": detected, "accepted": policy_accepted,
            "policy_excluded": policy_excluded, "policy_positive": policy_positive,
            "hard_reject": hard_reject,
            "p_support": raw_p, "r_support": raw_r, "union_support": raw_u,
            "agreement": agreement, "g_confirm": g_combined,
            "g_effective": effective_gliner, "g_value": g_value,
            "g_context": g_context, "g_success": g_success,
            "g_consistency": g_consistency, "validator": validator,
            "validator_kind": validator_kind, "semantic": semantic,
            "negative": negative, "meaning": meaning,
            "regex_specificity": regex_specificity,
            "contradiction_penalty": contradiction_penalty,
            "collision_penalty": collision_penalty,
            "weighted_p_support": p_support,
            "weighted_r_support": r_support,
            "weighted_union_support": union_support,
            "support_count": support_count, "row_support": row_support,
            "structural_consistency": structural_consistency,
            "specific_regex_support": specific_regex_support,
            "related_support": related_support,
        })

    scored.sort(key=lambda x: (-x["score"], x["entity"]))
    detected_profiles = [x for x in scored if x["detected"]]
    accepted_profiles = [x for x in scored if x["accepted"]]

    # Detection truth and policy action are separate. A policy-excluded high scorer
    # must never hide another accepted PII entity in the same column.
    adjudicated = detected_profiles[0] if detected_profiles else scored[0]
    primary = accepted_profiles[0] if accepted_profiles else adjudicated

    conflict = (
        len(accepted_profiles) > 1
        and accepted_profiles[1]["score"] >= accepted_profiles[0]["score"] - 0.05
        and accepted_profiles[1]["entity"] not in {
            ENTITY_PARENT.get(accepted_profiles[0]["entity"], "")
        }
    )

    if u_counts.get(primary["entity"], 0) > 0:
        stage1_entity = primary["entity"]
    else:
        stage1_entity = max(
            wu_support, key=wu_support.get
        ) if wu_support else (u_counts.most_common(1)[0][0] if u_counts else None)

    validator_output = (
        round(primary["validator"], 4) if primary["validator"] is not None else "N/A"
    )
    runner_up = next(
        (
            x for x in scored
            if x["entity"] != primary["entity"]
            and (
                x["union_support"] >= 0.03
                or x["support_count"] >= 2
                or x["g_value"] >= 0.35
                or (x["entity"] == "Address" and x.get("related_support", 0.0) >= 0.20)
            )
        ),
        None,
    )
    decision_margin = primary["score"] - primary["threshold"]
    runner_up_margin = max(0.0, primary["score"] - runner_up["score"]) if runner_up else 1.0

    entity_detected = bool(detected_profiles)
    policy_pii = bool(accepted_profiles)

    materially_supported = [
        x["entity"] for x in scored
        if (
            x["accepted"]
            or secondary_entity_acceptance(x, total_unique)
            or (
                x["entity"] == "Address"
                and x.get("related_support", 0.0) >= 0.20
                and x["semantic"] >= 0.75
                and x["g_context"] >= 0.45
            )
        )
        and (
            x["union_support"] >= 0.05
            or x["support_count"] >= 2
            or x.get("related_support", 0.0) >= 0.20
        )
    ]
    profile_map = {x["entity"]: x for x in scored}
    materially_supported = ontology_reduce_entities(materially_supported, profile_map)

    if policy_pii:
        parts = [
            f"{primary['entity']} accepted",
            f"support={primary['union_support']:.2f}",
            f"weighted_support={primary['weighted_union_support']:.2f}",
            f"semantic={primary['semantic']:.2f}",
            f"GLiNER_value={primary['g_value']:.2f}",
            f"GLiNER_context={primary['g_context']:.2f}",
            f"row_support={primary['row_support']:.2f}",
        ]
        if primary["validator"] is not None:
            parts.append(f"validator={primary['validator']:.2f}({primary['validator_kind']})")
        if conflict:
            parts.append("multiple accepted entity families")
        if len(materially_supported) > 1:
            parts.append("mixed PII: " + ", ".join(materially_supported[:6]))
        reason = "; ".join(parts) + "."
    elif entity_detected:
        excluded = [x["entity"] for x in detected_profiles if x["policy_excluded"]]
        reason = (
            f"{adjudicated['entity']} detected from evidence but excluded by configured privacy policy."
            if adjudicated["policy_excluded"] else
            f"{adjudicated['entity']} detected but no entity passed the protection policy."
        )
        if excluded:
            reason = reason.rstrip(".") + "; excluded entities: " + ", ".join(excluded[:6]) + "."
    else:
        reason = primary["hard_reject"] or (
            f"Rejected {primary['entity']}: evidence score {primary['score']:.2f} "
            f"below threshold {primary['threshold']:.2f}; support={primary['union_support']:.2f}, "
            f"weighted_support={primary['weighted_union_support']:.2f}, "
            f"semantic={primary['semantic']:.2f}, GLiNER_value={primary['g_value']:.2f}, "
            f"row_support={primary['row_support']:.2f}, "
            f"contradiction_penalty={primary['contradiction_penalty']:.2f}, "
            f"collision_penalty={primary['collision_penalty']:.2f}."
        )

    return {
        "Column_Name": column,
        "PII_Detected": policy_pii,
        "Entity_Detected": entity_detected,
        "Policy_PII": policy_pii,
        "Policy_Action": "PROTECT" if policy_pii else ("DETECTED_EXCLUDED" if entity_detected else "NONE"),
        "Final_Entity_Type": " | ".join(x["entity"] for x in accepted_profiles) if policy_pii else None,
        "Adjudicated_Entity": adjudicated["entity"] if entity_detected else None,
        "Primary_Entity": primary["entity"] if (policy_pii or entity_detected) else None,
        "Stage1_Detected_Entity": stage1_entity,
        "Stage1_Primary_Candidate": stage1_entity,
        "Stage1_Candidate_Entities": stage1_candidate_distribution(evidence, total_unique),
        "Detected_Entity_Set": " | ".join(materially_supported) if materially_supported else None,
        "Presidio_Support": round(primary["p_support"], 4),
        "Regex_Support": round(primary["r_support"], 4),
        "Detector_Agreement": round(primary["agreement"], 4),
        "Weighted_Presidio_Support": round(primary["weighted_p_support"], 4),
        "Weighted_Regex_Support": round(primary["weighted_r_support"], 4),
        "Weighted_Union_Support": round(primary["weighted_union_support"], 4),
        "Collision_Penalty": round(primary["collision_penalty"], 4),
        "GLiNER_Confirmation_Ratio": round(primary["g_value"], 4),
        "GLiNER_Combined_Support": round(
            min(1.0, 0.90 * primary["g_value"] + 0.10 * min(primary["g_context"], 0.70)), 4
        ),
        "GLiNER_Effective_Evidence": round(primary["g_effective"], 4),
        "GLiNER_Value_Confirmation": round(primary["g_value"], 4),
        "GLiNER_Context_Confirmation": round(primary["g_context"], 4),
        "GLiNER_Inference_Success_Rate": round(primary["g_success"], 4),
        "Validator_Pass_Rate": validator_output,
        "Validator_Type": primary["validator_kind"],
        "Confidence_Score": round(primary["score"], 4),
        "Confidence_Is_Calibrated": False,
        "Evidence_Score": round(primary["score"], 4),
        "Decision_Margin": round(decision_margin, 4),
        "Runner_Up_Entity": runner_up["entity"] if runner_up else None,
        "Runner_Up_Score": round(runner_up["score"], 4) if runner_up else None,
        "Runner_Up_Margin": round(runner_up_margin, 4),
        "Decision_Reason": reason,
    }

def clean_result(column: str, reason: str) -> dict:
    return {
        "Column_Name": column,
        "PII_Detected": False,
        "Entity_Detected": False,
        "Policy_PII": False,
        "Policy_Action": "NONE",
        "Final_Entity_Type": None,
        "Adjudicated_Entity": None,
        "Primary_Entity": None,
        "Stage1_Detected_Entity": None,
        "Stage1_Primary_Candidate": None,
        "Stage1_Candidate_Entities": None,
        "Detected_Entity_Set": None,
        "Presidio_Support": 0.0,
        "Regex_Support": 0.0,
        "Detector_Agreement": 0.0,
        "Weighted_Presidio_Support": 0.0,
        "Weighted_Regex_Support": 0.0,
        "Weighted_Union_Support": 0.0,
        "Collision_Penalty": 0.0,
        "GLiNER_Confirmation_Ratio": 0.0,
        "GLiNER_Combined_Support": 0.0,
        "GLiNER_Effective_Evidence": 0.0,
        "GLiNER_Value_Confirmation": 0.0,
        "GLiNER_Context_Confirmation": 0.0,
        "GLiNER_Inference_Success_Rate": 0.0,
        "Validator_Pass_Rate": "N/A",
        "Validator_Type": "NONE",
        "Confidence_Score": 0.0,
        "Confidence_Is_Calibrated": False,
        "Evidence_Score": 0.0,
        "Decision_Margin": 0.0,
        "Sampling_Entity_Coverage": 0.0,
        "Sampling_Entities_Covered": 0,
        "Sampling_Entities_Total": 0,
        "Runner_Up_Entity": None,
        "Runner_Up_Score": None,
        "Runner_Up_Margin": 0.0,
        "Decision_Reason": reason,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_column(
    column: str,
    df: pd.DataFrame,
    analyzer: AnalyzerEngine,
    gliner_model: GLiNER,
    max_samples: int,
) -> dict:
    logger.info("--- Column: %s", column)

    raw = (
        df[column]
        .dropna()
        .astype(str)
        .map(normalize_text)
    )
    raw = raw[~raw.map(is_null_like)]
    unique_values = raw.unique().tolist()
    total_unique = len(unique_values)

    if total_unique == 0:
        logger.info("    Empty column.")
        return clean_result(column, "Column has no non-empty values.")

    logger.info("    Stage 1: scanning %d unique values.", total_unique)
    evidence = run_stage1(analyzer, unique_values)

    if not evidence:
        logger.info("    Stage 1: no candidate detections; GLiNER bypassed.")
        return clean_result(column, "No Presidio or regex candidate evidence.")

    p_values = sum(bool(ev.presidio) for ev in evidence.values())
    r_values = sum(bool(ev.regex) for ev in evidence.values())
    logger.info(
        "    Stage 1: candidates=%d/%d, Presidio-values=%d, Regex-values=%d.",
        len(evidence), total_unique, p_values, r_values,
    )

    samples = stratified_sample(evidence, limit=max_samples)
    logger.info(
        "    Stage 2: GLiNER semantic evidence on %d representative samples (cap=%d).",
        len(samples), max_samples,
    )
    covered_entities, candidate_entity_count, coverage_ratio = sample_coverage(samples, evidence)
    logger.info(
        "    Stage 2 coverage: entity_families=%d/%d (%.2f).",
        covered_entities, candidate_entity_count, coverage_ratio,
    )
    gliner_results = run_gliner_stage(gliner_model, column, samples)

    row_values = raw.astype(str).tolist()
    result = aggregate_column(
        column,
        evidence,
        gliner_results,
        total_unique,
        sampled_evidence=samples,
        row_values=row_values,
    )
    result["Sampling_Entity_Coverage"] = round(coverage_ratio, 4)
    result["Sampling_Entities_Covered"] = covered_entities
    result["Sampling_Entities_Total"] = candidate_entity_count

    logger.info(
        "    Verdict: PII=%s Final=%s Evidence=%.4f",
        result["PII_Detected"],
        result["Final_Entity_Type"],
        result["Confidence_Score"],
    )
    return result


OUTPUT_COLUMNS = [
    "Column_Name",
    "PII_Detected",
    "Policy_Action",
    "Final_Entity_Type",
    "Primary_Entity",
    "Detected_Entity_Set",
    "Evidence_Score",
    "Decision_Margin",
    "Presidio_Support",
    "Regex_Support",
    "Detector_Agreement",
    "GLiNER_Value_Confirmation",
    "GLiNER_Context_Confirmation",
    "Validator_Pass_Rate",
    "Validator_Type",
    "Runner_Up_Entity",
    "Runner_Up_Score",
    "Decision_Reason",
]


def apply_policy_config(path: Optional[str]) -> None:
    """Apply a small, explicit JSON policy overlay without changing detector code."""
    global POLICY_TREAT_COARSE_LOCATION_AS_PII, POLICY_TREAT_ORGANIZATION_AS_PII
    if not path:
        return
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    identifiers = cfg.get("identifier_policy", {})
    for entity, value in identifiers.items():
        if entity not in POLICY_TREAT_IDENTIFIERS_AS_PII:
            raise ValueError(f"Unknown identifier policy entity: {entity}")
        if not isinstance(value, bool):
            raise ValueError(f"Policy value for {entity} must be boolean.")
        POLICY_TREAT_IDENTIFIERS_AS_PII[entity] = value

    if "treat_coarse_location_as_pii" in cfg:
        value = cfg["treat_coarse_location_as_pii"]
        if not isinstance(value, bool):
            raise ValueError("treat_coarse_location_as_pii must be boolean.")
        POLICY_TREAT_COARSE_LOCATION_AS_PII = value

    if "treat_organization_as_pii" in cfg:
        value = cfg["treat_organization_as_pii"]
        if not isinstance(value, bool):
            raise ValueError("treat_organization_as_pii must be boolean.")
        POLICY_TREAT_ORGANIZATION_AS_PII = value



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enterprise column-level PII detector: Presidio + Regex + Validators + GLiNER."
    )
    parser.add_argument("--input", required=True, metavar="CSV", help="Input CSV path.")
    parser.add_argument(
        "--output", default="pii_scan_results.csv", metavar="CSV",
        help="Output report path (default: pii_scan_results.csv).",
    )
    parser.add_argument(
        "--columns", nargs="*", default=None, metavar="COL",
        help="Optional subset of columns to scan.",
    )
    parser.add_argument(
        "--max-gliner-samples", type=int, default=MAX_GLINER_SAMPLES,
        help="Maximum representative GLiNER samples per candidate column (default: 15).",
    )
    parser.add_argument(
        "--policy-config", default=None, metavar="JSON",
        help="Optional JSON privacy-policy overlay.",
    )
    parser.add_argument(
        "--encoding", default="utf-8", help="Input CSV encoding (default: utf-8).",
    )
    parser.add_argument(
        "--delimiter", default=",", help="Input CSV delimiter (default: comma).",
    )
    parser.add_argument(
        "--region", default="IN", help="Data jurisdiction metadata (default: IN).",
    )
    parser.add_argument(
        "--language", default="en", help="Primary language metadata (default: en).",
    )
    args = parser.parse_args()

    if not 1 <= args.max_gliner_samples <= 15:
        raise ValueError("--max-gliner-samples must be between 1 and 15.")

    started = time.perf_counter()
    logger.info("=" * 78)
    logger.info("ENTERPRISE COLUMN-LEVEL PII DETECTION PIPELINE")
    logger.info("=" * 78)

    apply_policy_config(args.policy_config)

    # Preserve identifiers exactly. Automatic numeric inference can destroy leading
    # zeros or convert long identifiers to floating/scientific notation.
    df = pd.read_csv(
        args.input,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding=args.encoding,
        sep=args.delimiter,
    )
    columns = args.columns or list(df.columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in input CSV: {missing}")

    logger.info("Rows=%d | Columns=%d | Region=%s | Language=%s", len(df), len(columns), args.region, args.language)

    # Load each heavy component exactly once.
    analyzer = build_presidio_engine()
    gliner_model = load_gliner()

    rows = [
        process_column(
            column=column,
            df=df,
            analyzer=analyzer,
            gliner_model=gliner_model,
            max_samples=args.max_gliner_samples,
        )
        for column in columns
    ]

    report = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    report.insert(0, "Output_Schema_Version", OUTPUT_SCHEMA_VERSION)
    report.insert(1, "Detector_Version", SCRIPT_VERSION)
    report.to_csv(args.output, index=False)

    positives = report.loc[report["PII_Detected"], "Column_Name"].tolist()
    logger.info("=" * 78)
    logger.info("Columns scanned: %d", len(report))
    logger.info("PII-positive: %d -> %s", len(positives), positives)
    logger.info("Output: %s", args.output)
    logger.info("Elapsed: %.2f s", time.perf_counter() - started)
    logger.info("=" * 78)

    print("\n" + report.to_string(index=False))


if __name__ == "__main__":
    main()
