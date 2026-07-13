"""
aggregator.py — Column-level evidence aggregation and final PII decision.
This module contains the full scoring, thresholding, and policy layer.
"""
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pii_detector.config.settings import (
    PATTERN_RELIABILITY, PRESIDIO_ENTITY_RELIABILITY, COLLISION_GROUPS,
    ENTITY_PARENT, GLINER_ENTITY_FAMILIES, NATURAL_LANGUAGE_ENTITIES,
    OPAQUE_STRUCTURED_ENTITIES, POLICY_IDENTIFIER_ENTITIES,
    POLICY_TREAT_IDENTIFIERS_AS_PII, POLICY_TREAT_ORGANIZATION_AS_PII,
    POLICY_TREAT_COARSE_LOCATION_AS_PII, SPARSE_PREVALENCE_MAX,
    SPARSE_HIGH_SPECIFICITY_ENTITIES, GENERIC_ID_TERMS,
    SEMANTIC_CONTRADICTIONS, STRICT_PATTERN_NAMES, MODERATE_PATTERN_NAMES,
    OUTPUT_SCHEMA_VERSION, SCRIPT_VERSION,
)
from pii_detector.core.models import Detection, ValueEvidence, GlinerEvidence
from pii_detector.core.text_utils import (
    normalize_column, value_signature, semantic_scores, _phrase_in_normalized_text
)
from pii_detector.core.validators import validator_strength

def detection_reliability(d: Detection) -> float:
    if d.source == "regex":
        return PATTERN_RELIABILITY.get(d.pattern or "", 0.50)
    return PRESIDIO_ENTITY_RELIABILITY.get(d.entity, 0.70)


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
    return score


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


