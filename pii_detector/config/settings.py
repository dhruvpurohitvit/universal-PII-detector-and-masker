"""
settings.py — Central configuration for the Enterprise PII Detector.
All thresholds, model names, policy switches, and entity mappings live here.
To adjust behaviour, change values in this file — never in detector logic.
"""

# ─── Model Identifiers ────────────────────────────────────────────────────────
GLINER_MODEL_NAME      = "urchade/gliner_multi_pii-v1"
PRESIDIO_NLP_MODEL     = "en_core_web_lg"
GLINER_THRESHOLD       = 0.40
MAX_GLINER_SAMPLES     = 20          # samples per column sent to GLiNER
STAGE1_CACHE_MAX       = 80_000      # LRU cap for presidio/regex caches

# ─── Script Metadata ──────────────────────────────────────────────────────────
SCRIPT_VERSION        = "16.0.0"
OUTPUT_SCHEMA_VERSION = "12.0"

# ─── Fuzzy column-name matching (handles typos like 'emial', 'phne_no') ────────
# difflib.SequenceMatcher cutoff; 0.72 catches 1-2 char typos cleanly.
COLUMN_FUZZY_MATCH_CUTOFF = 0.72

# ─── Prevalence Guard ─────────────────────────────────────────────────────────
SPARSE_PREVALENCE_MAX = 0.10   # sparse PII path triggers below this row-fraction

# ─── Privacy Policy Layer (organisation-tunable) ──────────────────────────────
POLICY_TREAT_COARSE_LOCATION_AS_PII  = False
POLICY_TREAT_ORGANIZATION_AS_PII     = False
POLICY_TREAT_IDENTIFIERS_AS_PII: dict = {
    "Employee Identifier": True,
    "Customer Identifier": True,
    "User Identifier":     True,
    "Device Identifier":   True,
    "Contract Number":     True,
}

MIN_DOB_AGE = 0
MAX_DOB_AGE = 120

# ─── Regex Reliability Classes ────────────────────────────────────────────────
STRICT_PATTERN_NAMES = {
    "pan_strict", "email", "ssn_dashes", "ipv4", "ipv6_full",
    "cc_groups", "iban", "ifsc", "mac_colon", "mac_hyphen", "eth",
    "voter_id", "employee_id", "customer_id", "user_id",
    "contract_number", "device_serial",
}
MODERATE_PATTERN_NAMES = {
    "aadhaar", "phone_intl", "phone_in_mobile", "phone_us",
    "passport_in", "dl_in", "btc_legacy", "dob_iso", "dob_dmy",
    "nhs", "uk_ni", "us_itin",
}
BROAD_PATTERN_NAMES = {
    "ssn_compact", "passport_generic", "dl_us", "bank_acc",
}

PATTERN_RELIABILITY: dict = {
    "pan_strict": 0.99, "email": 0.99, "ssn_dashes": 0.98,
    "ipv4": 0.98,       "ipv6_full": 0.98, "cc_groups": 0.96,
    "iban": 0.99,       "ifsc": 0.98, "mac_colon": 0.99,
    "mac_hyphen": 0.99, "eth": 0.99,  "voter_id": 0.97,
    "employee_id": 0.93,"customer_id": 0.93,"user_id": 0.92,
    "contract_number": 0.92,"device_serial": 0.90,
    "aadhaar": 0.92,    "phone_intl": 0.88,"phone_in_mobile": 0.90,
    "phone_us": 0.88,   "passport_in": 0.90,"dl_in": 0.78,
    "btc_legacy": 0.96, "dob_iso": 0.92,"dob_dmy": 0.88,
    "ssn_compact": 0.55,"passport_generic": 0.42,
    "dl_us": 0.38,      "bank_acc": 0.28,
    "nhs": 0.85,        "uk_ni": 0.90,  "us_itin": 0.92,
}

PRESIDIO_ENTITY_RELIABILITY: dict = {
    "Email Address": 0.99, "IP Address": 0.98, "Credit Card Number": 0.96,
    "IBAN": 0.99, "MAC Address": 0.99, "Crypto Wallet Address": 0.98,
    "Phone Number": 0.82,  "Person Name": 0.78, "Location": 0.76,
    "Date Time": 0.82,     "Passport Number": 0.72,
    "Driver License Number": 0.45, "Bank Account Number": 0.38,
    "UK NHS Number": 0.85, "URL": 0.90, "Social Security Number": 0.95,
    "Tax Identification Number": 0.92, "National Insurance Number": 0.90,
}

# ─── Entity Ontology ──────────────────────────────────────────────────────────
ENTITY_PARENT: dict = {"Address": "Location"}

COLLISION_GROUPS: tuple = (
    {"Phone Number", "Bank Account Number", "Driver License Number", "UK NHS Number"},
    {"Aadhaar Number", "Bank Account Number", "Driver License Number"},
    {"Passport Number", "Driver License Number", "Device Identifier"},
    {"Credit Card Number", "Bank Account Number", "Driver License Number"},
    {"IP Address", "Date Time"},
    {"PAN Number", "Passport Number", "Driver License Number"},
)

HIGH_RISK_STRUCTURAL = {
    "Aadhaar Number", "PAN Number", "Passport Number", "Driver License Number",
    "Voter ID", "Bank Account Number", "Credit Card Number", "IBAN",
    "IFSC Code", "Social Security Number", "IP Address", "MAC Address",
    "Crypto Wallet Address", "UK NHS Number", "Tax Identification Number",
    "National Insurance Number",
}

DIRECT_STRONG_ENTITIES = {
    "Email Address", "MAC Address", "IBAN", "Crypto Wallet Address",
}

NATURAL_LANGUAGE_ENTITIES      = {"Person Name", "Address", "Location", "Organization"}
POLICY_IDENTIFIER_ENTITIES     = {
    "Employee Identifier", "Customer Identifier", "User Identifier",
    "Device Identifier", "Contract Number",
}
OPAQUE_STRUCTURED_ENTITIES     = {
    "Aadhaar Number", "PAN Number", "Passport Number", "Driver License Number",
    "Voter ID", "Bank Account Number", "Credit Card Number", "IBAN",
    "IFSC Code", "Social Security Number", "IP Address", "MAC Address",
    "Crypto Wallet Address", "Employee Identifier", "Customer Identifier",
    "User Identifier", "Device Identifier", "Contract Number",
    "UK NHS Number", "Tax Identification Number", "National Insurance Number",
}

SPARSE_HIGH_SPECIFICITY_ENTITIES = {
    "Email Address", "Credit Card Number", "Aadhaar Number",
    "PAN Number", "IBAN", "IFSC Code", "MAC Address", "IP Address",
    "Crypto Wallet Address", "UK NHS Number",
}

# ─── Generic Column Names (suppress high-prevalence triggers) ─────────────────
GENERIC_ID_TERMS = {
    "id", "identifier", "row number", "row num", "sequence", "seq",
    "index", "serial", "record number", "record id",
}

# ─── Semantic Contradictions ──────────────────────────────────────────────────
SEMANTIC_CONTRADICTIONS: dict = {
    "Passport Number":    {"Driver License Number", "Phone Number", "Bank Account Number"},
    "Driver License Number": {"Passport Number", "Phone Number", "Bank Account Number"},
    "Phone Number":       {"Passport Number", "Driver License Number", "Bank Account Number"},
    "Bank Account Number":{"Phone Number", "Passport Number", "Driver License Number"},
    "Date of Birth":      {"IP Address", "Phone Number"},
    "IP Address":         {"Date of Birth"},
    "Person Name":        {"Location"},
}

# ─── Entity Threshold Map ─────────────────────────────────────────────────────
ENTITY_THRESHOLDS: dict = {
    "Person Name":         0.56,
    "Address":             0.58,
    "Location":            0.58,
    "Date of Birth":       0.56,
    "Aadhaar Number":      0.62,
    "Credit Card Number":  0.62,
    "Bank Account Number": 0.68,
    "Passport Number":     0.58,
    "Driver License Number": 0.60,
    "PAN Number":          0.58,
    "Phone Number":        0.58,
}
DEFAULT_ENTITY_THRESHOLD = 0.56

# ─── Canonical Label Map (GLiNER label → entity name) ────────────────────────
CANONICAL: dict = {
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
    "nhs number": "UK NHS Number",
    "national health service number": "UK NHS Number",
    "national insurance": "National Insurance Number",
    "itin": "Tax Identification Number",
}

ENTITY_LABEL_MAP: dict = {
    "IN_PAN":               "PAN Number",
    "IN_AADHAAR":           "Aadhaar Number",
    "EMAIL_ADDRESS":        "Email Address",
    "US_SSN":               "Social Security Number",
    "US_SSN_CUSTOM":        "Social Security Number",
    "IP_ADDRESS":           "IP Address",
    "CREDIT_CARD":          "Credit Card Number",
    "CREDIT_CARD_CUSTOM":   "Credit Card Number",
    "PHONE_NUMBER":         "Phone Number",
    "PHONE_NUMBER_CUSTOM":  "Phone Number",
    "PASSPORT_NUMBER":      "Passport Number",
    "IBAN":                 "IBAN",
    "MAC_ADDRESS":          "MAC Address",
    "CRYPTO_WALLET":        "Crypto Wallet Address",
    "IN_VOTER_ID":          "Voter ID",
    "DRIVER_LICENSE":       "Driver License Number",
    "BANK_ACCOUNT":         "Bank Account Number",
    "DATE_OF_BIRTH":        "Date of Birth",
    "DATE_TIME":            "Date Time",
    "PERSON":               "Person Name",
    "LOCATION":             "Location",
    "NRP":                  "Nationality",
    "ORGANIZATION":         "Organization",
    "URL":                  "URL",
    "MEDICAL_LICENSE":      "Medical License",
    "UK_NHS":               "UK NHS Number",
    "US_DRIVER_LICENSE":    "Driver License Number",
    "US_PASSPORT":          "Passport Number",
    "US_BANK_NUMBER":       "Bank Account Number",
    "US_ITIN":              "Tax Identification Number",
}

GLINER_LABELS = [
    "person name", "email address", "phone number", "home address",
    "street address", "location", "date of birth", "age", "aadhaar number",
    "PAN number", "passport number", "driver license number", "voter ID",
    "bank account number", "credit card number", "IBAN", "IFSC code",
    "tax identification number", "social security number",
    "national insurance number", "nhs number", "IP address", "MAC address",
    "vehicle registration number", "crypto wallet address",
    "employee identifier", "customer identifier", "user identifier",
    "device identifier", "contract number", "organization name",
]

GLINER_ENTITY_FAMILIES: dict = {
    "Address":              {"Address"},
    "Location":             {"Location"},
    "Person Name":          {"Person Name"},
    "Phone Number":         {"Phone Number"},
    "Email Address":        {"Email Address"},
    "Date of Birth":        {"Date of Birth"},
    "Passport Number":      {"Passport Number"},
    "Driver License Number":{"Driver License Number"},
    "Aadhaar Number":       {"Aadhaar Number"},
    "PAN Number":           {"PAN Number"},
    "Bank Account Number":  {"Bank Account Number"},
    "Credit Card Number":   {"Credit Card Number"},
    "IP Address":           {"IP Address"},
    "MAC Address":          {"MAC Address"},
    "IBAN":                 {"IBAN"},
    "IFSC Code":            {"IFSC Code"},
    "Voter ID":             {"Voter ID"},
    "UK NHS Number":        {"UK NHS Number"},
    "Social Security Number":{"Social Security Number"},
    "Tax Identification Number":{"Tax Identification Number"},
    "National Insurance Number":{"National Insurance Number"},
    "Employee Identifier":  {"Employee Identifier", "User Identifier"},
    "Customer Identifier":  {"Customer Identifier", "User Identifier"},
    "User Identifier":      {"User Identifier", "Customer Identifier", "Employee Identifier"},
    "Device Identifier":    {"Device Identifier"},
    "Contract Number":      {"Contract Number"},
    "Organization":         {"Organization"},
}

# ─── Semantic Terms (column name hinting) ─────────────────────────────────────
ENTITY_SEMANTIC_TERMS: dict = {
    "Organization": {
        "organization","organisation","organization name","organisation name",
        "company","company name","employer","business name","vendor name",
        "supplier name","legal entity","corporate name",
    },
    "Email Address":        {"email","email address","mail","e mail","inbox"},
    "Phone Number": {
        "phone","mobile","mobile number","contact no","contact number",
        "telephone","tel","whatsapp",
    },
    "Aadhaar Number":       {"aadhaar","aadhar","aadhaar number","aadhar number","uidai"},
    "PAN Number":           {"pan","pan number","permanent account number"},
    "Passport Number":      {"passport","passport number","travel document"},
    "Driver License Number":{
        "driver license","driving license","driving licence","driver licence","dl number",
    },
    "Voter ID":             {"voter","voter id","epic","election id"},
    "Bank Account Number":  {"bank account","account number","bank account number","acct number","account no"},
    "Credit Card Number":   {"credit card","card number","debit card","payment card"},
    "IBAN":                 {"iban","international bank account"},
    "IFSC Code":            {"ifsc","ifsc code","bank branch code"},
    "Social Security Number":{"ssn","social security","social security number"},
    "IP Address":           {"ip","ip address","ipv4","ipv6","client ip","server ip"},
    "MAC Address":          {"mac","mac address","hardware address"},
    "Crypto Wallet Address":{
        "wallet","crypto wallet","bitcoin address","ethereum address","blockchain address",
    },
    "Date of Birth":        {"dob","date of birth","birth date","birthday","born"},
    "Person Name": {
        "name","full name","person name","first name","last name","surname",
        "given name","customer name","employee name",
    },
    "Address":              {"address","home address","street address","residential address"},
    "Location":             {"location","city","state","district","country","place"},
    "Employee Identifier":  {"employee id","employee number","emp id","staff id"},
    "Customer Identifier":  {"customer id","customer number","client id"},
    "User Identifier":      {"user id","username","login id","account id"},
    "Device Identifier": {
        "device id","device identifier","device serial","device serial number",
        "serial number","serial no","hardware id","imei","meid","asset tag","equipment id",
    },
    "Contract Number":      {"contract number","contract id","agreement number"},
    "UK NHS Number":        {"nhs","nhs number","national health service"},
    "Tax Identification Number": {"itin","tin","tax id","tax identification"},
    "National Insurance Number": {"ni number","national insurance","nino"},
}

NEGATIVE_ENTITY_TERMS: dict = {
    "Date of Birth": {
        "created date","updated date","order date","invoice date",
        "transaction date","event date","timestamp","date",
    },
    "Phone Number": {
        "row number","sequence","index","order number","order id",
        "transaction id","invoice id","reference id","record id",
        "product id","customer id","employee id","internal id",
        "tracking number","shipment id","ticket number","case number",
        "account reference","registration number","numeric sku",
    },
    "Bank Account Number": {
        "row number","sequence","index","order id","product id",
        "transaction id","invoice number",
    },
    "Passport Number":     {"product code","sku","order id","internal id"},
    "Driver License Number":{"product code","sku","order id","internal id"},
    "PAN Number": {
        "product code","sku","coupon code","coupon","promo code",
        "promotion code","voucher code","token","reference code",
        "campaign code","internal code",
    },
    "IP Address": {
        "version","release","software version","firmware version",
        "build version","build number","release version",
    },
}

# ─── Validator Strength ────────────────────────────────────────────────────────
VALIDATOR_STRENGTH: dict = {
    "Credit Card Number":       ("CHECKSUM",        1.00),
    "Aadhaar Number":           ("CHECKSUM",        1.00),
    "IBAN":                     ("CHECKSUM",        1.00),
    "UK NHS Number":            ("CHECKSUM",        1.00),
    "PAN Number":               ("STRICT_FORMAT",   0.65),
    "IFSC Code":                ("STRICT_FORMAT",   0.75),
    "MAC Address":              ("STRICT_FORMAT",   0.80),
    "Email Address":            ("STRICT_FORMAT",   0.85),
    "IP Address":               ("STANDARD_PARSE",  0.70),
    "Date of Birth":            ("PLAUSIBILITY",    0.70),
    "Phone Number":             ("BASIC_STRUCTURE", 0.45),
    "Social Security Number":   ("STRICT_FORMAT",   0.80),
    "National Insurance Number":("STRICT_FORMAT",   0.80),
    "Tax Identification Number":("STRICT_FORMAT",   0.75),
}

NULL_LIKE_VALUES = {
    "", "na", "n/a", "none", "null", "nil", "unknown", "not provided",
    "not available", "-", "--", "nan", "<na>",
}
