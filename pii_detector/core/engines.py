"""
engines.py — Central initialization for ML models (Presidio and GLiNER).
"""
import logging
import time

import torch
from gliner import GLiNER
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from pii_detector.config.settings import GLINER_MODEL_NAME, PRESIDIO_NLP_MODEL, ENTITY_LABEL_MAP
from pii_detector.core.validators import validator_for
from pii_detector.core.models import Detection

logger = logging.getLogger("pii_detector")

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


def presidio_scan_value(analyzer: AnalyzerEngine, value: str) -> list:
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
