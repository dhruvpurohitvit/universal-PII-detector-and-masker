"""
text_utils.py — Text normalization and signature utilities.
"""
import re
import unicodedata

from pii_detector.config.settings import NULL_LIKE_VALUES, GENERIC_ID_TERMS, ENTITY_SEMANTIC_TERMS, NEGATIVE_ENTITY_TERMS


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
    phrase = normalize_column(phrase)
    return bool(re.search(rf"(?:^|\s){re.escape(phrase)}(?:$|\s)", normalized))

def semantic_scores(column: str) -> tuple:
    from difflib import SequenceMatcher
    from pii_detector.config.settings import COLUMN_FUZZY_MATCH_CUTOFF

    norm   = normalize_column(column)
    tokens = set(norm.split())
    scores: dict = {}

    for entity, terms in ENTITY_SEMANTIC_TERMS.items():
        score = 0.0
        for term in terms:
            term_norm   = normalize_column(term)
            term_tokens = set(term_norm.split())

            # Exact / phrase / subset match  (unchanged, high-confidence)
            if norm == term_norm:
                score = max(score, 1.0)
            elif _phrase_in_normalized_text(term_norm, norm):
                score = max(score, 0.90)
            elif term_tokens and term_tokens.issubset(tokens):
                score = max(score, 0.72)
            else:
                # ── Fuzzy fallback (handles typos) ────────────────────────
                # Compare whole string first
                ratio = SequenceMatcher(None, norm, term_norm).ratio()
                if ratio >= COLUMN_FUZZY_MATCH_CUTOFF:
                    score = max(score, 0.60 * ratio)
                else:
                    # Token-by-token fuzzy: match any single token from column
                    # against any single token from the term
                    for col_tok in tokens:
                        for term_tok in term_tokens:
                            if len(col_tok) < 3 or len(term_tok) < 3:
                                continue  # skip very short tokens — too noisy
                            tok_ratio = SequenceMatcher(None, col_tok, term_tok).ratio()
                            if tok_ratio >= COLUMN_FUZZY_MATCH_CUTOFF:
                                score = max(score, 0.55 * tok_ratio)

        if score:
            if any(x in tokens for x in {"or", "maybe", "possible", "mixed", "alternate"}):
                score *= 0.88
            scores[entity] = round(score, 4)

    negatives: set = set()
    for entity, terms in NEGATIVE_ENTITY_TERMS.items():
        if any(_phrase_in_normalized_text(term, norm) for term in terms):
            negatives.add(entity)

    generic = norm in GENERIC_ID_TERMS
    meaning = "generic identifier or sequence field" if generic else (
        max(scores, key=scores.get) if scores else "unknown field semantics"
    )
    return scores, negatives, meaning

