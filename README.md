#  Enterprise PII Detector

> **Industry-grade, zero-shot, multi-engine Personally Identifiable Information (PII) detection and masking engine for any tabular CSV dataset.**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![AES-256-GCM](https://img.shields.io/badge/Encryption-AES--256--GCM-critical)](pii_detector/masking/aes_masker.py)
[![GLiNER](https://img.shields.io/badge/NLP-GLiNER-purple)](https://github.com/urchade/GLiNER)
[![Presidio](https://img.shields.io/badge/NLP-Presidio-blue)](https://microsoft.github.io/presidio/)
dockerized : https://hub.docker.com/r/dhruvpurohitx/universal-pii-detector

---

## Table of Contents

1. [What Is This?](#-what-is-this)
2. [Why Is This Novel?](#-why-is-this-novel)
3. [User Workflow](#-user-workflow)
4. [How the Engine Works (Step-by-Step)](#-how-the-engine-works-step-by-step)
5. [Backend Detection Pipeline (Flowchart)](#-backend-detection-pipeline-flowchart)
6. [Project Structure](#-project-structure)
7. [What Each File Does](#-what-each-file-does)
8. [AES-256-GCM Masking — How It Works](#-aes-256-gcm-masking--how-it-works)
9. [Output Files](#-output-files)
10. [Installation](#-installation)
11. [Running the App](#-running-the-app)
12. [Configuration](#-configuration)
13. [Supported PII Types](#-supported-pii-types)

---

## What Is This?

Most PII detectors require you to tell them **which column contains sensitive data**. This tool does the opposite — it figures that out entirely on its own.

You give it any CSV file. It scans every column, using three AI engines working together, and tells you:
- **Which columns contain PII** and exactly what type (email, phone, Aadhaar, credit card, etc.)
- **How confident it is** (a score from 0–100%)
- **Whether a formal validator passed** (e.g. Luhn checksum for credit cards)
- **How long each engine took**
- **A masked version of the file** where all PII cells are AES-256-GCM encrypted

It works **even when the column name is wrong, has a typo, or is completely unrelated to the data inside it.** The value inside the cell is always scanned first.

---

## Why Is This Novel?

Most commercial and open-source PII tools have one or more of these weaknesses:

| Limitation | This Tool |
|---|---|
| Rely on column names to decide what to scan | (X) We scan the values first; column names only add a confidence boost |
| Can't handle typos in column names | (✔) Fuzzy column name matching via `difflib` (edit-distance ≤ 2) |
| Single engine = single point of failure | (✔) Three independent engines vote and their evidence is fused |
| No checksums → many false positives | (✔) Luhn (CC), Verhoeff (Aadhaar), MOD-97 (IBAN), NHS checksum, etc. |
| Binary yes/no output | (✔) Probabilistic weighted evidence score with decision margin |
| Slow / no batch processing | (✔) LRU cache for repeated values; stratified GLiNER sampling |
| No encryption / masking | (✔) AES-256-GCM with PBKDF2-SHA256 key derivation (OWASP 2024) |
| No explainability | (✔) Per-column reason string explaining every decision |

The core innovation is the **three-engine evidence fusion layer** (`aggregator.py`), which combines NLP confidence, regex specificity, GLiNER zero-shot inference, checksum validation, and column semantic scores into a single calibrated confidence number — with entity-specific thresholds and collision-group penalty logic.

---

##  User Workflow

### Using the Streamlit Web UI

```
1. Run:   python -m streamlit run app.py
2. Open:  http://localhost:8501
3. Upload your CSV file (drag and drop)
4. Click " Start PII Scan"
5. Watch the live progress log as each column is scanned
6. Navigate the five result tabs:
   -  Summary   → Visual grid of all columns with PII/Safe status
   -  Detailed  → Full table with scores, validators, and reasons
   -  Timing    → Stacked bar chart of per-module runtimes
   -  Mask PII  → Encrypt PII columns + download masked CSV + key file
   -  Unmask    → Upload masked CSV + key file + password to restore
```

### Using the Command Line (CLI)

```bash
# Basic scan → outputs 4 files in outputs/ folder
python main.py --input your_data.csv --output-dir outputs/

# With AES masking of all detected PII columns
python main.py --input your_data.csv --output-dir outputs/ --mask-password "YourPassword123"

# Adjust GLiNER accuracy (default: 20 samples per column)
python main.py --input your_data.csv --output-dir outputs/ --max-samples 40
```

---

##  How the Engine Works (Step-by-Step)

Here is exactly what happens when you scan one column of your CSV file:

### Step 1 — Normalise & Deduplicate
Every value in the column is Unicode-normalised, whitespace-stripped, and deduplicated. Only unique non-null values are sent to the detection engines. This means scanning a 1,000,000-row file with only 500 unique emails costs the same as scanning 500 rows.

### Step 2 — Stage 1: Presidio NLP Scan 
Microsoft Presidio runs spaCy's large English NER model (`en_core_web_lg`) on every unique value. It recognises entities like `PERSON`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `CREDIT_CARD`, etc. Results are **cached** in an LRU cache (up to 80,000 entries) so repeated values across columns cost nothing extra.

### Step 3 — Stage 1: Regex Pattern Scan 
Forty-plus hand-crafted regular expressions run on every unique value in parallel with Presidio. Patterns cover:
- Structured IDs: Aadhaar, PAN, SSN, IBAN, IFSC, NHS Number, NI Number, ITIN
- Financial: Credit card (BIN-aware), Bank account numbers
- Network: IPv4, IPv6, MAC addresses (colon and hyphen)
- Crypto: Bitcoin legacy, Ethereum (0x...)
- Documents: Indian/generic passports, driving licences, Voter ID
- Identifiers: Employee ID, Customer ID, User ID, Device serial, Contract numbers
- Dates: ISO 8601, DD/MM/YYYY, MM/DD/YYYY

Each pattern has a **reliability weight** (0.28–0.99) and a **specificity class** (STRICT / MODERATE / BROAD) that feeds the fusion layer.

### Step 4 — Deterministic Validation 
For every candidate entity, a hard validator runs on the matched value:
- **Credit Card** → Luhn algorithm
- **Aadhaar** → Verhoeff checksum (12-digit, first digit 2–9)
- **IBAN** → ISO 7064 MOD-97-10
- **NHS Number** → weighted digit checksum
- **SSN** → area-group exclusion rules (000, 666, 9xx blocked)
- **PAN** → strict `XXXXX0000X` format
- **Email** → RFC-5322 regex
- **Phone** → E.164 + Indian mobile (6xxx–9xxx) + US format
- **Date of Birth** → plausibility check (age 0–120)

Validator pass/fail is fed into the fusion score — a column where every credit card number passes Luhn gets a major confidence boost; one where none pass is penalised.

### Step 5 — GLiNER Zero-Shot AI 
GLiNER (`urchade/gliner_multi_pii-v1`) is a transformer-based named entity recogniser that works without any fine-tuning on domain-specific labels. It runs in **two views**:
- **Value view**: just the raw cell value
- **Context view**: `"Field 'email_contact' (email address) value: john@corp.com"`

The context view lets GLiNER use column-name semantics even when the pattern didn't match. This is what catches PII in columns named `col_7` or `misc_data`.

GLiNER does **not** run on every value — it runs on a **stratified sample** of up to 20 values per column, chosen to maximise entity-family coverage. This keeps inference time practical while maintaining accuracy.

### Step 6 — Evidence Fusion & Scoring 
The aggregator (`aggregator.py`) fuses all evidence from all three engines into a single **Evidence Score** per entity per column:

```
Evidence Score = weighted combination of:
  + Presidio reliability-weighted support fraction
  + Regex reliability-weighted support fraction  
  + Unique-value union support fraction
  + Detector agreement (both engines agree?)
  + GLiNER value-level confirmation
  + GLiNER context-level confirmation
  + Semantic score (column name meaning)
  + Validator pass rate
  - Contradiction penalty (another entity has stronger column semantics)
  - Collision penalty (structurally similar entities compete)
  - Negative semantic penalty (column name says this is NOT PII)
```

Entity-specific weights and formulas are used — e.g. `Person Name` is mostly NLP-driven, while `Credit Card` is mostly regex + checksum driven.

### Step 7 — Policy Layer & Decision 
The final decision is made using entity-specific thresholds (0.56–0.68) and a configurable privacy policy:
- `PROTECT` → PII confirmed, column should be masked
- `DETECTED_EXCLUDED` → PII found, but policy says it's not reportable (e.g. Organization name)
- `NONE` → No sufficient PII evidence

---

##  Backend Detection Pipeline (Flowchart)

```
CSV File
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  For each column:                                   │
│                                                     │
│  1. Normalize & Deduplicate values                  │
│       │                                             │
│       ▼                                             │
│  2. ┌──────────────┐  ┌──────────────────────────┐ │
│     │ Presidio NLP │  │  Regex Pattern Engine    │ │
│     │ (spaCy lg)   │  │  (40+ compiled patterns) │ │
│     └──────┬───────┘  └───────────┬──────────────┘ │
│            │                      │                 │
│            ▼                      ▼                 │
│     Named Entity Tags      Pattern Matches          │
│            │                      │                 │
│            └──────────┬───────────┘                 │
│                       ▼                             │
│            3. Deterministic Validators              │
│               (Luhn / Verhoeff / MOD-97 / etc.)     │
│                       │                             │
│                       ▼                             │
│            4. Stratified Sampling                   │
│               (max 20 diverse values)               │
│                       │                             │
│                       ▼                             │
│            5. GLiNER Zero-Shot NER                  │
│               (value view + context view)           │
│                       │                             │
│                       ▼                             │
│            6. Evidence Fusion (aggregator.py)       │
│               Weighted score per candidate entity   │
│                       │                             │
│                       ▼                             │
│            7. Policy Layer → PROTECT / EXCLUDED / NONE
│                       │                             │
└───────────────────────┼─────────────────────────────┘
                        ▼
              28-column result per column
                        │
          ┌─────────────┼──────────────────┐
          ▼             ▼                  ▼
  summary_report   detailed_report   timing_report
      .csv              .csv              .txt
                        │
                        ▼ (if password set)
                  masked_data.csv
                  manifest.json
```

---

##  Project Structure

```
pii/
│
├── app.py                          ← Streamlit web UI (5 tabs)
├── main.py                         ← CLI entry point
├── requirements.txt                ← All dependencies
├── README.md                       ← This file
├── .gitignore                      ← Git exclusion rules
│
├── pii_detector/
│   │
│   ├── config/
│   │   └── settings.py             ← ALL tunable parameters (thresholds, models, policies)
│   │
│   ├── core/
│   │   ├── engines.py              ← Loads Presidio + GLiNER models
│   │   ├── models.py               ← Data classes: Detection, ValueEvidence, TimingInfo
│   │   ├── patterns.py             ← 40+ compiled regex patterns with reliability weights
│   │   ├── text_utils.py           ← Normalisation + fuzzy column name matching
│   │   └── validators.py           ← Luhn, Verhoeff, IBAN, NHS, SSN, PAN, phone, DOB
│   │
│   ├── masking/
│   │   ├── __init__.py
│   │   └── aes_masker.py           ← AES-256-GCM encrypt/decrypt with PBKDF2 key
│   │
│   └── pipeline/
│       ├── aggregator.py           ← Evidence fusion brain (1000 lines)
│       └── run_report.py           ← Generates all 4 output files
│
└── outputs/                        ← Auto-created on each CLI run
    ├── summary_report.csv
    ├── detailed_report.csv
    ├── timing_report.txt
    └── masked_data.csv
```

---

##  What Each File Does

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI. Loads engines, runs scan, renders 5 tabs, handles upload/download |
| `main.py` | CLI. Parses args, orchestrates detection loop, calls run_report, prints timing table |
| `settings.py` | Single source of truth for all knobs. Change thresholds here, never in logic files |
| `engines.py` | Initialises and caches Presidio AnalyzerEngine and GLiNER model |
| `models.py` | Pure data classes — `Detection`, `ValueEvidence`, `GlinerEvidence`, `TimingInfo` |
| `patterns.py` | 40+ regex patterns with entity name, pattern name, raw regex, and base score |
| `text_utils.py` | `normalize_column()`, `value_signature()`, `semantic_scores()` with fuzzy matching |
| `validators.py` | One function per entity type — returns `True/False/None` |
| `aes_masker.py` | `mask_dataframe()` + `unmask_dataframe()` using AES-256-GCM + PBKDF2-SHA256 |
| `aggregator.py` | The brain. Fuses evidence from all 3 engines into a per-entity confidence score |
| `run_report.py` | Writes `summary_report.csv`, `detailed_report.csv`, `timing_report.txt`, `masked_data.csv` |

---

##  AES-256-GCM Masking — How It Works

### Encryption

```
Password + Column Salt
        │
        ▼ (PBKDF2-HMAC-SHA256, 260,000 rounds)
  32-byte AES-256 Key     ← derived ONCE per column (cached)
        │
        ├─ Cell 1: random 12-byte nonce → AESGCM.encrypt(value) → base64 blob
        ├─ Cell 2: random 12-byte nonce → AESGCM.encrypt(value) → base64 blob
        └─ Cell N: random 12-byte nonce → AESGCM.encrypt(value) → base64 blob

Stored in cell: "AES256GCM::<base64url(nonce + ciphertext + GCM-tag)>"
```

### Why This Design?

- **One PBKDF2 per column** (not per cell): 260,000-round key derivation is computationally expensive by design (it makes brute-force attacks slow). Running it once per column instead of once per cell makes bulk masking ~1000x faster while keeping the same security.
- **Unique nonce per cell**: Even if two cells have the same value (e.g. two rows with the same email), they encrypt to completely different ciphertext — an attacker can't learn which cells are duplicates.
- **GCM authentication tag**: If any byte in the stored ciphertext is changed (by accident or deliberately), decryption throws an error immediately. Silent data corruption is impossible.
- **Manifest file**: The per-column salts are stored in `manifest.json` (not in the CSV). This means the masked CSV is useless without the manifest — adding an extra layer of security.

### Decryption

Upload both `masked_data.csv` and `manifest.json` in the Unmask tab, enter your password, and the original data is perfectly restored.

---

## Output Files

| File | Description |
|------|-------------|
| `summary_report.csv` | One row per column. Columns: `Column_Name`, `PII_Status`, `Entity_Type`, `Policy_Action` |
| `detailed_report.csv` | One row per column. 28+ technical columns including all scores, timing, validators, reason |
| `timing_report.txt` | ASCII box-art showing per-column and per-module runtimes |
| `masked_data.csv` | Original CSV with PII columns replaced by `AES256GCM::...` ciphertext |
| `manifest.json` | Decryption key metadata — required alongside password to unmask |

### Sample `summary_report.csv`

```
Column_Name,PII_Status,Entity_Type,Policy_Action
full_name,PII DETECTED,Person Name,PROTECT
emial,PII DETECTED,Email Address,PROTECT
ph_no,PII DETECTED,Phone Number,PROTECT
order_id,SAFE,—,NONE
```

Note how `emial` (typo) and `ph_no` (abbreviated) are correctly identified — because the **values** were scanned, not just the column names.

---

##  Installation

### Requirements
- Python 3.9 or higher
- Windows / Linux / macOS
- ~3 GB disk space for models (GLiNER + spaCy)

### Steps

```bash
# 1. Clone / download the project
cd "path/to/pii"

# 2. Create a virtual environment
python -m venv gliner_env

# 3. Activate it
# Windows:
gliner_env\Scripts\Activate.ps1
# Linux / macOS:
source gliner_env/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Download the spaCy NLP model (required for Presidio)
python -m spacy download en_core_web_lg
```

---

##  Running the App

### Web UI (Recommended)
```bash
python -m streamlit run app.py
# Opens at http://localhost:8501
```

### CLI
```bash
# Standard scan (4 output files created in outputs/)
python main.py --input your_data.csv --output-dir outputs/

# With AES-256 masking
python main.py --input your_data.csv --output-dir outputs/ --mask-password "StrongP@ss!"

# More GLiNER samples for harder datasets
python main.py --input your_data.csv --output-dir outputs/ --max-samples 40
```

---

##  Configuration

Edit `pii_detector/config/settings.py` to tune the engine. Key settings:

| Setting | Default | Effect |
|---------|---------|--------|
| `GLINER_THRESHOLD` | `0.40` | Lower = catch more PII (but more false positives) |
| `MAX_GLINER_SAMPLES` | `20` | Higher = more accurate (but slower) |
| `COLUMN_FUZZY_MATCH_CUTOFF` | `0.72` | Lower = more tolerant of column name typos |
| `POLICY_TREAT_COARSE_LOCATION_AS_PII` | `False` | Set `True` to flag city/country columns as PII |
| `POLICY_TREAT_ORGANIZATION_AS_PII` | `False` | Set `True` to flag company name columns |
| `ENTITY_THRESHOLDS` | per-entity map | Evidence score required to classify a column as PII |
| `MIN_DOB_AGE` / `MAX_DOB_AGE` | `0` / `120` | Valid age range for date-of-birth plausibility check |

---

##  Supported PII Types

| Category | Entities |
|----------|----------|
| **Identity** | Person Name, Age |
| **Contact** | Email Address, Phone Number |
| **Location** | Address (full), Location (city/state), Country |
| **Financial** | Credit Card Number (Luhn), Bank Account Number, IBAN (MOD-97), IFSC Code |
| **Government ID (India)** | Aadhaar (Verhoeff), PAN Number, Voter ID, Passport Number, Driver License |
| **Government ID (UK)** | NHS Number, National Insurance Number |
| **Government ID (US)** | Social Security Number (SSN), ITIN, Driver License |
| **Government ID (Generic)** | Passport Number |
| **Network** | IP Address (IPv4/IPv6), MAC Address |
| **Crypto** | Bitcoin Wallet, Ethereum Wallet |
| **Healthcare** | Date of Birth (plausibility-validated) |
| **Enterprise** | Employee ID, Customer ID, User ID, Device Serial, Contract Number |
| **Organisation** | Organisation Name *(policy-configurable)* |
