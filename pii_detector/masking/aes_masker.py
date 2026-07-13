"""
aes_masker.py — AES-256-GCM authenticated encryption for PII column masking.
=============================================================================

DESIGN GOALS:
  • Simple UX: one password to mask, same password to unmask. No manifest file.
  • Security: AES-256-GCM with authenticated tags (tamper-proof).
  • Speed: one PBKDF2 key derivation per column (not per cell) via column-keyed salt.
  • Portability: masked CSV is self-contained — works without any extra files.

HOW IT WORKS:
  Key = PBKDF2-HMAC-SHA256(password, SHA-256(column_name), 260_000 rounds)
  → Same password + same column name = same key, always.
  → No salt file, no manifest, no extra downloads.
  → Each CELL still gets a unique 12-byte random nonce (semantic security).
  → The 16-byte GCM tag is embedded in the blob (tamper detection).

Wire format per cell (base64url-safe, prefixed):
    "AES::<base64url( nonce[12] | ciphertext | tag[16] )>"

Decryption:
    Strip prefix → base64url-decode → split nonce / ciphertext+tag → AESGCM.decrypt
    The column name (from the CSV header) provides the deterministic key.

Public API:
    mask_dataframe(df, pii_columns, password)  →  masked_df
    unmask_dataframe(masked_df, password)      →  original_df
    preview_masked(series)                     →  preview_series  [UI]
    is_masked_series(series)                   →  bool
    is_masked_dataframe(df)                    →  List[str]  (masked column names)
"""

from __future__ import annotations

import base64
import hashlib
import os
from functools import lru_cache
from typing import List

import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────
NONCE_LEN  = 12    # GCM standard nonce
KEY_LEN    = 32    # AES-256
ITERATIONS = 260_000  # OWASP 2024 PBKDF2-SHA256 minimum

_PREFIX       = "AES::"
_MASKED_LABEL = " [MASKED]"


# ─── Key Derivation ───────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _column_key(password: str, column_name: str) -> bytes:
    """
    Derive a 32-byte AES key from (password, column_name).
    Using the column name as the salt means:
      - No separate manifest file is needed.
      - Same password + same column name = reproducible key.
      - Different columns get different keys even with the same password.
    The SHA-256 hash ensures the salt is always exactly 32 bytes regardless
    of column name length.
    """
    salt = hashlib.sha256(column_name.encode("utf-8")).digest()
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        ITERATIONS,
        dklen=KEY_LEN,
    )


# ─── Cell-level encrypt / decrypt ────────────────────────────────────────────

def _encrypt(plaintext: str, key: bytes) -> str:
    """Encrypt one string → prefixed base64url blob."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = os.urandom(NONCE_LEN)
    ct    = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return _PREFIX + base64.urlsafe_b64encode(nonce + ct).decode("ascii")


def _decrypt(encoded: str, key: bytes) -> str:
    """Decrypt a prefixed base64url blob → original string."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    raw   = base64.urlsafe_b64decode(encoded[len(_PREFIX):])
    nonce = raw[:NONCE_LEN]
    ct    = raw[NONCE_LEN:]
    try:
        return AESGCM(key).decrypt(nonce, ct, None).decode("utf-8")
    except Exception:
        raise ValueError(
            "Decryption failed — wrong password, wrong column name, or tampered data."
        )


# ─── Public DataFrame API ─────────────────────────────────────────────────────

def mask_dataframe(
    df: pd.DataFrame,
    pii_columns: List[str],
    password: str,
) -> pd.DataFrame:
    """
    Return a copy of df with every PII column AES-256-GCM encrypted.
    Column headers are renamed to  '<colname> [MASKED]' so they are easy to spot.
    Non-PII columns pass through completely unchanged.
    Null / empty cells are left as-is even inside PII columns.
    """
    masked = df.copy()
    for col in pii_columns:
        if col not in masked.columns:
            continue
        key = _column_key(password, col)      # one derivation per column
        masked[col] = masked[col].map(
            lambda v, k=key: (
                v if (pd.isna(v) or str(v).strip() == "")
                else _encrypt(str(v), k)
            )
        )
        masked.rename(columns={col: col + _MASKED_LABEL}, inplace=True)
    return masked


def unmask_dataframe(masked_df: pd.DataFrame, password: str) -> pd.DataFrame:
    """
    Detect all masked columns (by header suffix or cell prefix) in masked_df,
    decrypt them, and return a clean DataFrame with original column names restored.
    Raises ValueError if the password is wrong for any column.
    """
    df = masked_df.copy()

    for col in list(df.columns):
        # Determine original column name (strip [MASKED] suffix if present)
        original_col = col.replace(_MASKED_LABEL, "").strip()

        # Check whether this column actually contains masked values
        sample = df[col].dropna().astype(str)
        if not sample.str.startswith(_PREFIX).any():
            # Nothing to decrypt in this column
            if col != original_col:
                df.rename(columns={col: original_col}, inplace=True)
            continue

        key = _column_key(password, original_col)
        try:
            df[col] = df[col].map(
                lambda v, k=key: (
                    v if (pd.isna(v) or not str(v).startswith(_PREFIX))
                    else _decrypt(str(v), k)
                )
            )
        except ValueError:
            raise ValueError(
                f"Wrong password — could not decrypt column '{original_col}'."
            )

        if col != original_col:
            df.rename(columns={col: original_col}, inplace=True)

    return df


# ─── UI helpers ───────────────────────────────────────────────────────────────

def preview_masked(series: pd.Series, show_chars: int = 2) -> pd.Series:
    """
    Return a human-readable redacted preview for display:
        'alice@corp.com'  →  'al●●●●●●●●●●●'
    """
    def _redact(v):
        s = str(v)
        if pd.isna(v) or s.strip() == "":
            return v
        vis = s[:show_chars]
        return vis + "●" * max(6, len(s) - show_chars)
    return series.map(_redact)


def is_masked_series(series: pd.Series) -> bool:
    """True if the series contains at least one encrypted cell."""
    return series.dropna().astype(str).str.startswith(_PREFIX).any()


def is_masked_dataframe(df: pd.DataFrame) -> List[str]:
    """Return list of column names that appear to contain masked data."""
    return [c for c in df.columns if is_masked_series(df[c])]
