"""
main.py — Enterprise PII Detector CLI
======================================
Usage:
    python main.py --input data.csv --output-dir outputs/
    python main.py --input data.csv --output-dir outputs/ --mask-password "s3cr3t"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from collections import Counter, OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd
from gliner import GLiNER
from presidio_analyzer import AnalyzerEngine

# ── suppress noisy startup logs ───────────────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pii_detector")

# ── internal imports ──────────────────────────────────────────────────────────
from pii_detector.config.settings import (
    CANONICAL, COLLISION_GROUPS, DEFAULT_ENTITY_THRESHOLD,
    ENTITY_PARENT, ENTITY_THRESHOLDS, GLINER_ENTITY_FAMILIES,
    GLINER_LABELS, GLINER_THRESHOLD, MAX_GLINER_SAMPLES,
    NATURAL_LANGUAGE_ENTITIES, OPAQUE_STRUCTURED_ENTITIES,
    OUTPUT_SCHEMA_VERSION, POLICY_IDENTIFIER_ENTITIES,
    POLICY_TREAT_COARSE_LOCATION_AS_PII, POLICY_TREAT_IDENTIFIERS_AS_PII,
    POLICY_TREAT_ORGANIZATION_AS_PII, SCRIPT_VERSION,
    SPARSE_HIGH_SPECIFICITY_ENTITIES, SPARSE_PREVALENCE_MAX,
    STAGE1_CACHE_MAX,
)
from pii_detector.core.engines import (
    build_presidio_engine, load_gliner, presidio_scan_value,
)
from pii_detector.core.models import Detection, GlinerEvidence, TimingInfo, ValueEvidence
from pii_detector.core.patterns import regex_scan_value
from pii_detector.core.text_utils import (
    is_null_like, normalize_column, normalize_text, semantic_scores,
    value_signature,
)
from pii_detector.pipeline.aggregator import (
    aggregate_column, clean_result, detection_reliability, sample_coverage,
    sample_priority,
)
from pii_detector.pipeline.run_report import generate_all_reports

# ─── LRU caches for Stage-1 ──────────────────────────────────────────────────
_PRESIDIO_CACHE: OrderedDict[str, List[Detection]] = OrderedDict()
_REGEX_CACHE:    OrderedDict[str, List[Detection]] = OrderedDict()


def _cache_get(cache: OrderedDict, key: str):
    v = cache.get(key)
    if v is not None:
        cache.move_to_end(key)
    return v


def _cache_put(cache: OrderedDict, key: str, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > STAGE1_CACHE_MAX:
        cache.popitem(last=False)


# ─── Stage 1: Presidio + Regex ───────────────────────────────────────────────

def run_stage1(
    analyzer: AnalyzerEngine,
    unique_values: Sequence[str],
    timing: TimingInfo,
) -> Dict[str, ValueEvidence]:
    evidence: Dict[str, ValueEvidence] = {}
    for value in unique_values:
        # Hard-cap to avoid OOM on pathologically long free-text fields
        if len(value) > 12_000:
            value = value[:6_000] + "\n…[TRUNCATED]…\n" + value[-6_000:]

        t0 = time.perf_counter()
        p = _cache_get(_PRESIDIO_CACHE, value)
        if p is None:
            p = presidio_scan_value(analyzer, value)
            _cache_put(_PRESIDIO_CACHE, value, p)
        timing.stage1_presidio += time.perf_counter() - t0

        t1 = time.perf_counter()
        r = _cache_get(_REGEX_CACHE, value)
        if r is None:
            r = regex_scan_value(value)
            _cache_put(_REGEX_CACHE, value, r)
        timing.stage1_regex += time.perf_counter() - t1

        if p or r:
            evidence[value] = ValueEvidence(value=value, presidio=list(p), regex=list(r))
    return evidence


# ─── Stage 2: GLiNER sampling ────────────────────────────────────────────────

def stratified_sample(
    evidence: Dict[str, ValueEvidence],
    limit: int = MAX_GLINER_SAMPLES,
) -> List[ValueEvidence]:
    candidates = list(evidence.values())
    if len(candidates) <= limit:
        return sorted(candidates, key=lambda e: (-sample_priority(e), e.value))

    selected: List[ValueEvidence] = []
    seen_values: Set[str] = set()
    seen_sigs: Counter = Counter()

    def _add(ev: ValueEvidence, sig_cap: int = 2) -> bool:
        if ev.value in seen_values:
            return False
        sig = value_signature(ev.value)
        if seen_sigs[sig] >= sig_cap:
            return False
        selected.append(ev)
        seen_values.add(ev.value)
        seen_sigs[sig] += 1
        return True

    ranked = sorted(candidates, key=lambda e: (-sample_priority(e), e.value))

    # ── pass 1: at least one sample per entity family ─────────────────────
    entity_groups: Dict[str, List[ValueEvidence]] = defaultdict(list)
    entity_strength: Dict[str, float] = defaultdict(float)
    for ev in ranked:
        per = defaultdict(float)
        for d in ev.presidio + ev.regex:
            per[d.entity] = max(per[d.entity], d.score * detection_reliability(d))
        for entity, strength in per.items():
            if strength >= 0.28:
                entity_groups[entity].append(ev)
                entity_strength[entity] = max(entity_strength[entity], strength)

    for entity in sorted(entity_groups, key=lambda e: (-entity_strength[e], e)):
        if len(selected) >= limit:
            break
        for ev in entity_groups[entity]:
            if _add(ev):
                break

    # ── pass 2: stratify by agreement ────────────────────────────────────
    strata: Dict[str, List[ValueEvidence]] = {
        "agree": [], "disagree": [], "presidio_only": [], "regex_only": [],
    }
    for ev in ranked:
        pe  = {d.entity for d in ev.presidio}
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
            if _add(ev):
                break

    # ── pass 3: fill remainder ───────────────────────────────────────────
    for ev in ranked:
        if len(selected) >= limit:
            break
        _add(ev, sig_cap=10 ** 9)

    return selected[:limit]


def gliner_analyze_sample(
    model: GLiNER,
    original_col: str,
    norm_col: str,
    semantic_meaning: str,
    ev: ValueEvidence,
) -> GlinerEvidence:
    context_label = norm_col or normalize_column(original_col)
    context_prefix = (
        f"Field '{context_label}' ({semantic_meaning}) value: "
        if semantic_meaning and semantic_meaning != "unknown field semantics"
        else f"Field '{context_label}' value: "
    )

    views = [
        ("value",   "",             ev.value),
        ("context", context_prefix, context_prefix + ev.value),
    ]
    by_view: Dict[str, Dict[str, float]] = {"value": {}, "context": {}}
    inference_ok = True

    for view_name, prefix, view_text in views:
        value_end = len(view_text)
        value_start = len(prefix)
        try:
            entities = model.predict_entities(view_text, GLINER_LABELS, threshold=GLINER_THRESHOLD)
        except Exception as exc:
            logger.warning("GLiNER inference failed: %s", exc)
            inference_ok = False
            continue

        value_lower = ev.value.casefold()
        for item in entities:
            raw_label   = normalize_text(str(item.get("label", ""))).strip().casefold()
            label       = CANONICAL.get(raw_label, str(item.get("label", "")).strip().title())
            score       = float(item.get("score", 0.0))
            ent_start   = item.get("start")
            ent_end     = item.get("end")
            entity_text = normalize_text(str(item.get("text", ""))).casefold()
            overlaps    = (
                ent_start is not None and ent_end is not None
                and int(ent_start) < value_end and int(ent_end) > value_start
            )
            text_match = bool(entity_text) and entity_text in value_lower
            if overlaps or text_match:
                by_view[view_name][label] = max(by_view[view_name].get(label, 0.0), score)

    value_predictions   = sorted(by_view["value"].items(),   key=lambda x: (-x[1], x[0]))
    context_predictions = sorted(by_view["context"].items(), key=lambda x: (-x[1], x[0]))

    labels   = set(by_view["value"]) | set(by_view["context"])
    combined = {}
    for label in labels:
        vs = by_view["value"].get(label, 0.0)
        cs = by_view["context"].get(label, 0.0)
        combined[label] = (
            min(1.0, 0.88 * vs + 0.12 * min(cs, 0.85)) if vs > 0
            else 0.12 * min(cs, 0.85)
        )

    predictions = sorted(combined.items(), key=lambda x: (-x[1], x[0]))
    best_entity = predictions[0][0] if predictions else None
    best_score  = predictions[0][1] if predictions else 0.0
    return GlinerEvidence(
        ev.value, value_predictions, context_predictions,
        predictions, best_entity, best_score, inference_ok,
    )


def run_gliner_stage(
    model: GLiNER,
    column: str,
    samples: List[ValueEvidence],
) -> List[GlinerEvidence]:
    _, _, meaning = semantic_scores(column)
    norm = normalize_column(column)
    return [
        gliner_analyze_sample(model, column, norm, meaning, ev)
        for ev in samples
    ]


# ─── Column orchestration ─────────────────────────────────────────────────────

def process_column(
    column: str,
    df: pd.DataFrame,
    analyzer: AnalyzerEngine,
    gliner_model: GLiNER,
    max_samples: int,
) -> dict:
    logger.info("Column ▶ %s", column)
    timing = TimingInfo()

    raw = df[column].dropna().astype(str).map(normalize_text)
    raw = raw[~raw.map(is_null_like)]
    unique_values = raw.unique().tolist()
    total_unique  = len(unique_values)

    if total_unique == 0:
        logger.info("  ↳ empty — skipped.")
        res = clean_result(column, "Column has no non-empty values.")
        return _inject_metadata(res, timing)

    logger.info("  Stage 1 → %d unique values", total_unique)
    evidence = run_stage1(analyzer, unique_values, timing)

    if not evidence:
        logger.info("  ↳ No stage-1 hits — GLiNER bypassed.")
        res = clean_result(column, "No Presidio or regex candidate evidence; semantic fallback not confirmed.")
        return _inject_metadata(res, timing)

    samples = stratified_sample(evidence, limit=max_samples)
    logger.info("  Stage 2 → GLiNER on %d / %d samples", len(samples), len(evidence))

    t2 = time.perf_counter()
    gliner_results = run_gliner_stage(gliner_model, column, samples)
    timing.stage2_gliner = time.perf_counter() - t2

    t3 = time.perf_counter()
    covered, n_entities, coverage = sample_coverage(samples, evidence)
    logger.info("  Coverage → %d/%d entity families (%.0f%%)", covered, n_entities, coverage * 100)

    row_values = raw.astype(str).tolist()
    result = aggregate_column(
        column, evidence, gliner_results, total_unique,
        sampled_evidence=samples, row_values=row_values,
    )
    result["Sampling_Entity_Coverage"]  = round(coverage, 4)
    result["Sampling_Entities_Covered"] = covered
    result["Sampling_Entities_Total"]   = n_entities
    timing.aggregation = time.perf_counter() - t3

    logger.info(
        "  ↳ %s  entity=%s  score=%.3f",
        result.get("Policy_Action", "?"),
        result.get("Final_Entity_Type") or "—",
        result.get("Evidence_Score", 0.0),
    )
    return _inject_metadata(result, timing)


def _inject_metadata(result: dict, timing: TimingInfo) -> dict:
    """Prepend schema version and append timing columns."""
    meta = {
        "Output_Schema_Version": OUTPUT_SCHEMA_VERSION,
        "Detector_Version":      SCRIPT_VERSION,
    }
    meta.update(result)
    meta["Time_Presidio_sec"]    = round(timing.stage1_presidio,  3)
    meta["Time_Regex_sec"]       = round(timing.stage1_regex,     3)
    meta["Time_GLiNER_sec"]      = round(timing.stage2_gliner,    3)
    meta["Time_Aggregation_sec"] = round(timing.aggregation,      3)
    meta["Time_Total_sec"]       = round(timing.total,            3)
    return meta


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _print_timing_summary(results: List[dict]) -> None:
    """Print a compact terminal timing table after all columns are processed."""
    W = 70
    sep = "─" * W
    print(f"\n{'═' * W}")
    print(f"  PII DETECTOR  v{SCRIPT_VERSION}  — RUN COMPLETE")
    print(f"{'═' * W}")
    total_presidio = sum(r.get("Time_Presidio_sec",   0) for r in results)
    total_regex    = sum(r.get("Time_Regex_sec",       0) for r in results)
    total_gliner   = sum(r.get("Time_GLiNER_sec",      0) for r in results)
    total_agg      = sum(r.get("Time_Aggregation_sec", 0) for r in results)
    grand_total    = total_presidio + total_regex + total_gliner + total_agg

    pii_cols  = sum(1 for r in results if r.get("Policy_Action") == "PROTECT")
    excl_cols = sum(1 for r in results if r.get("Policy_Action") == "DETECTED_EXCLUDED")
    safe_cols = len(results) - pii_cols - excl_cols

    print(f"  Columns scanned  : {len(results)}")
    print(f"  🔴 PII detected  : {pii_cols}")
    print(f"  🟡 Excl by policy: {excl_cols}")
    print(f"  🟢 Safe          : {safe_cols}")
    print(sep)
    print(f"  {'Module':<30}  {'Time':>8}")
    print(sep)
    print(f"  {'🔍 Presidio NLP scan':<30}  {total_presidio:>7.2f}s")
    print(f"  {'🔎 Regex pattern scan':<30}  {total_regex:>7.2f}s")
    print(f"  {'🤖 GLiNER AI inference':<30}  {total_gliner:>7.2f}s")
    print(f"  {'⚙  Aggregation':<30}  {total_agg:>7.2f}s")
    print(sep)
    print(f"  {'✅ TOTAL':<30}  {grand_total:>7.2f}s")
    print(f"{'═' * W}\n")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Enterprise PII Detector — detect, classify, and mask PII in CSV files.",
    )
    parser.add_argument("--input",         required=True,       help="Path to input CSV file.")
    parser.add_argument("--output-dir",    default="outputs",   help="Directory for all output files.")
    parser.add_argument("--mask-password", default=None,        help="Password for AES-256-GCM masking of PII columns.")
    parser.add_argument("--max-samples",   type=int, default=MAX_GLINER_SAMPLES,
                        help=f"GLiNER samples per column (default {MAX_GLINER_SAMPLES}).")
    args = parser.parse_args(argv)

    # ── Load input ──────────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        sys.exit(f"[ERROR] Input file not found: {args.input}")
    df = pd.read_csv(args.input, dtype=str).fillna("")
    logger.info("Loaded %d rows × %d columns from '%s'", len(df), len(df.columns), args.input)

    # ── Initialise engines ──────────────────────────────────────────────────
    analyzer     = build_presidio_engine()
    gliner_model = load_gliner()

    # ── Detect ──────────────────────────────────────────────────────────────
    results: List[dict] = []
    t_start = time.perf_counter()
    for col in df.columns:
        results.append(
            process_column(col, df, analyzer, gliner_model, max_samples=args.max_samples)
        )
    t_elapsed = time.perf_counter() - t_start

    # ── Terminal summary ────────────────────────────────────────────────────
    _print_timing_summary(results)

    # ── Write all output files ──────────────────────────────────────────────
    paths = generate_all_reports(
        results=results,
        original_df=df,
        output_dir=args.output_dir,
        mask_password=args.mask_password,
    )

    print("Output files written:")
    for name, path in paths.items():
        print(f"  📄 {name:<20} → {path}")

    if args.mask_password:
        print(f"\n  🔒 PII columns masked with AES-256-GCM.")
        print(f"  🔑 Keep your password safe — you need it to unmask!")


if __name__ == "__main__":
    main()
