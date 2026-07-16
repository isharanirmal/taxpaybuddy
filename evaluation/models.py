"""
Data model for one evaluation result.

Each EvaluationResult represents one question evaluated through
the complete TaxPayBuddy RAG pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import ClassVar, List


@dataclass
class EvaluationResult:

    # -----------------------------
    # Question Information
    # -----------------------------
    question: str
    expected_agent: str
    predicted_agent: str

    # -----------------------------
    # Retrieval Metrics
    # -----------------------------
    precision_at_1: float
    precision_at_3: float

    recall_at_1: float
    recall_at_3: float

    tp: int
    fp: int
    fn: int

    f1_score: float

    # -----------------------------
    # Answer Quality
    # -----------------------------
    cosine_accuracy: float

    keyword_score: float

    matched_keywords: List[str]

    faithfulness: float

    # -----------------------------
    # Performance
    # -----------------------------
    latency_seconds: float

    generated_answer: str

    error: str = ""

    # -----------------------------
    # CSV Header
    # -----------------------------
    CSV_FIELDNAMES: ClassVar[List[str]] = [

        "question",

        "expected_agent",
        "predicted_agent",

        "precision_at_1",
        "precision_at_3",

        "recall_at_1",
        "recall_at_3",

        "tp",
        "fp",
        "fn",

        "f1_score",

        "cosine_accuracy",

        "keyword_score",

        "matched_keywords",

        "faithfulness",

        "latency_seconds",

        "generated_answer",

        "error",
    ]

    def to_row(self) -> dict:
        """
        Converts the dataclass into a CSV row.
        """
        return asdict(self)