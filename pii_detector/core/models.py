"""
models.py — Shared dataclass definitions for the PII detection pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


@dataclass
class Detection:
    value:          str
    entity:         str
    source:         str           # "presidio" | "regex"
    recognizer:     str
    score:          float
    start:          int
    end:            int
    pattern:        Optional[str]  = None
    validator_pass: Optional[bool] = None


@dataclass
class ValueEvidence:
    value:    str
    presidio: List[Detection] = field(default_factory=list)
    regex:    List[Detection] = field(default_factory=list)

    @property
    def sources(self) -> Set[str]:
        out: Set[str] = set()
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
    value:               str
    value_predictions:   List[Tuple[str, float]]
    context_predictions: List[Tuple[str, float]]
    predictions:         List[Tuple[str, float]]
    best_entity:         Optional[str]
    best_score:          float
    inference_ok:        bool = True


@dataclass
class TimingInfo:
    """Per-column timing of each pipeline stage (seconds)."""
    stage1_presidio: float = 0.0
    stage1_regex:    float = 0.0
    stage2_gliner:   float = 0.0
    aggregation:     float = 0.0

    @property
    def total(self) -> float:
        return self.stage1_presidio + self.stage1_regex + self.stage2_gliner + self.aggregation
