"""
Pure scoring functions used by the TaxPayBuddy evaluation framework.

This module contains only local scoring functions. No Gemini API calls
are made here. Faithfulness evaluation is handled separately by
llm_judge.py.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from math import sqrt
from typing import List, Sequence

from src.framework.core.data_models import DocumentChunk


# ============================================================
# Retrieval Metrics
# ============================================================

def precision_at_k(
    chunks: Sequence[DocumentChunk],
    keywords: Sequence[str],
    k: int,
) -> float:
    """
    Precision@K

    Precision = Relevant Retrieved / Retrieved
    """

    top_k = list(chunks[:k])

    if not top_k or not keywords:
        return 0.0

    relevant = sum(
        1
        for chunk in top_k
        if any(
            keyword.lower() in chunk.text.lower()
            for keyword in keywords
        )
    )

    return relevant / len(top_k)


def recall_at_k(
    chunks: Sequence[DocumentChunk],
    keywords: Sequence[str],
    k: int,
) -> float:
    """
    Recall@K

    Recall = Relevant Retrieved / Total Relevant
    """

    if not keywords:
        return 0.0

    text = " ".join(
        chunk.text.lower()
        for chunk in chunks[:k]
    )

    found = sum(
        1
        for keyword in keywords
        if keyword.lower() in text
    )

    return found / len(keywords)


def confusion_counts(
    chunks: Sequence[DocumentChunk],
    keywords: Sequence[str],
    k: int,
) -> dict:
    """
    Computes TP, FP and FN using keyword-based relevance.

    A retrieved chunk is considered relevant if it
    contains at least one ground-truth keyword.
    """

    top_k = list(chunks[:k])

    tp = 0
    fp = 0

    for chunk in top_k:

        if any(
            keyword.lower() in chunk.text.lower()
            for keyword in keywords
        ):
            tp += 1

        else:
            fp += 1

    fn = max(0, len(keywords) - tp)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def f1_score(
    precision: float,
    recall: float,
) -> float:
    """
    Harmonic mean of Precision and Recall.
    """

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
    ) / (
        precision + recall
    )


# ============================================================
# Answer Quality Metrics
# ============================================================

def keyword_coverage_score(
    answer: str,
    keywords: Sequence[str],
) -> float:
    """
    Fraction of expected keywords appearing
    in the generated answer.
    """

    if not answer or not keywords:
        return 0.0

    answer = answer.lower()

    found = sum(
        1
        for keyword in keywords
        if keyword.lower() in answer
    )

    return found / len(keywords)


def matched_keywords(
    answer: str,
    keywords: Sequence[str],
) -> List[str]:

    answer = answer.lower()

    return [
        keyword
        for keyword in keywords
        if keyword.lower() in answer
    ]


@lru_cache(maxsize=1)
def _embedding_model():
    """
    Loads SentenceTransformer once.
    """

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
def _word_count_cosine(
    text_a: str,
    text_b: str,
) -> float:
    """
    Dependency-free fallback cosine similarity using
    simple word-frequency vectors.
    """

    a_counts = Counter(text_a.lower().split())
    b_counts = Counter(text_b.lower().split())

    shared_words = set(a_counts) & set(b_counts)

    dot_product = sum(
        a_counts[word] * b_counts[word]
        for word in shared_words
    )

    norm_a = sqrt(
        sum(value * value for value in a_counts.values())
    )

    norm_b = sqrt(
        sum(value * value for value in b_counts.values())
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def cosine_similarity_score(
    text_a: str,
    text_b: str,
) -> float:
    """
    Semantic similarity between the generated answer
    and the reference answer.

    Uses SentenceTransformer locally when available.
    Falls back to word-count cosine similarity.
    """

    if not text_a or not text_b:
        return 0.0

    try:

        model = _embedding_model()

        embeddings = model.encode(
            [text_a, text_b]
        )

        vector_a = embeddings[0]
        vector_b = embeddings[1]

        norm_a = sqrt(
            sum(value * value for value in vector_a)
        )

        norm_b = sqrt(
            sum(value * value for value in vector_b)
        )

        if norm_a == 0 or norm_b == 0:
            return 0.0

        dot_product = sum(
            x * y
            for x, y in zip(vector_a, vector_b)
        )

        return float(
            dot_product / (norm_a * norm_b)
        )

    except Exception:

        return _word_count_cosine(
            text_a,
            text_b,
        )


def agent_routing_confusion_matrix(rows, labels):
    """
    Per-agent TP / FP / FN / TN for the routing decision,
    computed one-vs-rest across every evaluated question.
    """

    matrix = {}

    for label in labels:

        tp = fp = fn = tn = 0

        for row in rows:

            expected = row["expected_agent"]
            predicted = row["predicted_agent"]

            if expected == label and predicted == label:
                tp += 1
            elif expected != label and predicted == label:
                fp += 1
            elif expected == label and predicted != label:
                fn += 1
            else:
                tn += 1

        matrix[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    return matrix