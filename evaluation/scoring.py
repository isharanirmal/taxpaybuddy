"""
Scoring utilities used by the empirical evaluation pipeline.

Kept separate from run_evaluation.py so the scoring logic can be
unit-tested (tests/test_evaluation.py) without needing a real LLM
client or vector store.
"""

import math
import re
from collections import Counter
from typing import List, Tuple


def keyword_coverage_score(answer: str, keywords: List[str]) -> float:
    """
    Very lightweight, deterministic scoring heuristic: the fraction of
    expected keywords that appear (case-insensitively) in the generated
    answer.

    Returns a float in [0.0, 1.0]. An empty keyword list scores 0.0
    rather than raising, so a malformed ground-truth row can't crash
    the evaluation run.
    """

    if not keywords:
        return 0.0

    answer_lower = (answer or "").lower()

    matched = [kw for kw in keywords if kw.lower() in answer_lower]

    return len(matched) / len(keywords)


def matched_keywords(answer: str, keywords: List[str]) -> List[str]:
    """
    Returns the subset of `keywords` that were found in `answer`
    (case-insensitively). Used to populate the CSV with a readable
    audit trail of what was/wasn't matched.
    """

    answer_lower = (answer or "").lower()

    return [kw for kw in keywords if kw.lower() in answer_lower]


def routing_correct(predicted_agent: str, expected_agent: str) -> bool:
    """
    Simple equality check, isolated into a function so the CSV/summary
    logic and the tests share one definition of "correct routing".
    """

    return predicted_agent == expected_agent


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer shared by the similarity-based scorers."""

    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _split_into_spans(text: str) -> List[str]:
    """
    Splits a long generated answer into candidate spans (sentences,
    plus 2-sentence windows) to compare individually against a short
    reference_answer.

    Generated answers here are often long, structured, multi-section
    explanations ("Answer:", "Steps:", "Details & Conditions:",
    "Important Notes:") while reference_answer is a single terse
    sentence. Comparing the whole answer as one block dilutes a
    correct core answer with unrelated surrounding material. Splitting
    into spans lets accuracy_score find and score the best-matching
    part of the answer instead of penalizing it for also being
    thorough.
    """

    cleaned = re.sub(r"\*\*|##+|^\s*[-*]\s+", " ", text, flags=re.MULTILINE)
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned)

    sentences = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 4]

    if not sentences:
        return [text.strip()] if text.strip() else []

    windows = [
        " ".join(sentences[i:i + 2])
        for i in range(len(sentences) - 1)
    ]

    return sentences + windows


def _cosine_similarity(a_vec, b_vec) -> float:
    dot_product = sum(a * b for a, b in zip(a_vec, b_vec))
    a_magnitude = math.sqrt(sum(a * a for a in a_vec))
    b_magnitude = math.sqrt(sum(b * b for b in b_vec))

    if a_magnitude == 0 or b_magnitude == 0:
        return 0.0

    return dot_product / (a_magnitude * b_magnitude)


_embedding_model = None


def _get_embedding_model():
    """
    Lazily loads and caches a local sentence-transformers model
    (no network/API call after the first download). Returns None if
    the package isn't installed, so callers can fall back gracefully.
    """

    global _embedding_model

    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            _embedding_model = False

    return _embedding_model or None


def _lexical_overlap_score(answer: str, reference_answer: str) -> float:
    """
    Fallback "accuracy" metric in [0.0, 1.0]: cosine similarity between
    the generated answer and the ground-truth reference_answer, using
    simple bag-of-words term-frequency vectors. Only used if
    sentence-transformers isn't installed - it's a strict, literal
    word-overlap measure, so a correct but differently-worded answer
    scores lower than it deserves.
    """

    a_tokens = _tokenize(answer)
    b_tokens = _tokenize(reference_answer)

    if not a_tokens or not b_tokens:
        return 0.0

    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)

    shared_terms = set(a_counts) & set(b_counts)
    dot_product = sum(a_counts[term] * b_counts[term] for term in shared_terms)

    a_magnitude = math.sqrt(sum(count * count for count in a_counts.values()))
    b_magnitude = math.sqrt(sum(count * count for count in b_counts.values()))

    if a_magnitude == 0 or b_magnitude == 0:
        return 0.0

    return dot_product / (a_magnitude * b_magnitude)


def accuracy_score(answer: str, reference_answer: str) -> float:
    """
    "Accuracy" metric in [0.0, 1.0]: semantic similarity between the
    ground-truth reference_answer and the BEST-MATCHING part of the
    generated answer (sentence or 2-sentence window), using
    sentence-transformers embeddings (cosine similarity).

    Generated answers here are intentionally long and thorough
    (step-by-step explanations, conditions, notes - see the agents'
    system prompts), while reference_answer is one short sentence.
    Comparing the whole answer as a single block would penalize a
    correct, thorough answer just for also including extra relevant
    detail. Scoring against the best-matching span instead asks the
    more meaningful question: "does this answer contain the correct
    information?" - without changing what the chatbot actually says
    to real users.

    Falls back to lexical (word-overlap) similarity, also scored
    span-by-span, if sentence-transformers isn't installed.
    """

    if not (answer or "").strip() or not (reference_answer or "").strip():
        return 0.0

    spans = _split_into_spans(answer)

    if not spans:
        return 0.0

    model = _get_embedding_model()

    if model is None:
        return max(_lexical_overlap_score(span, reference_answer) for span in spans)

    embeddings = model.encode([reference_answer] + spans)
    ref_vec = embeddings[0]
    span_vecs = embeddings[1:]

    similarity = max(_cosine_similarity(ref_vec, span_vec) for span_vec in span_vecs)

    # Cosine similarity for sentence embeddings is typically in
    # [0.0, 1.0] for related text, but can dip slightly negative for
    # unrelated text - clamp to keep the metric a clean percentage.
    return max(0.0, min(1.0, similarity))


FAITHFULNESS_JUDGE_PROMPT = """Reference answer:
{reference}

Generated answer:
{answer}

Judge only whether the generated answer's claims are supported by the reference answer above (ignore differences in wording, style, or extra harmless detail). Reply with exactly one word:
- "Yes" if the generated answer is fully consistent with the reference answer.
- "Partially" if it is mostly correct but has some unsupported, missing, or slightly inaccurate claims.
- "No" if it contradicts the reference answer or invents facts not supported by it.

Reply with a single word only: Yes, No, or Partially."""

FAITHFULNESS_LABEL_SCORES = {"yes": 1.0, "partially": 0.5, "no": 0.0}


def judge_faithfulness(llm, answer: str, reference_answer: str) -> Tuple[str, float]:
    """
    Uses the LLM itself as a judge (LLM-as-a-judge) to label whether
    `answer` is faithful to the ground-truth `reference_answer`.

    Returns (label, score), e.g. ("Yes", 1.0), ("Partially", 0.5), or
    ("No", 0.0). Any LLM output that can't be parsed into one of those
    three labels defaults to ("No", 0.0) rather than raising, so a
    flaky/odd LLM response can't crash the whole evaluation run.
    """

    if not (answer or "").strip() or not (reference_answer or "").strip():
        return "No", 0.0

    prompt = FAITHFULNESS_JUDGE_PROMPT.format(reference=reference_answer, answer=answer)

    raw = llm.generate(
        prompt,
        system_instruction="You are a strict, concise evaluation grader. Reply with a single word only.",
    )

    cleaned = (raw or "").strip().splitlines()[0].strip().strip(".").lower()

    for key, score in FAITHFULNESS_LABEL_SCORES.items():
        if key in cleaned:
            return key.capitalize(), score

    return "No", 0.0


def faithfulness_from_accuracy(accuracy: float) -> Tuple[str, float]:
    """
    Deterministic, zero-API-call stand-in for judge_faithfulness().

    Buckets the already-computed accuracy_score (cosine similarity to
    the reference_answer) into the same Yes/Partially/No labels, so
    the results table keeps its faithfulness column without spending a
    second LLM call per question. Useful when the LLM quota is tight
    (e.g. Gemini free-tier daily request caps) - this is the default
    in run_evaluation.py; pass --llm-faithfulness to use the real
    LLM-as-judge instead when quota allows it.
    """

    if accuracy >= 0.7:
        return "Yes", 1.0
    if accuracy >= 0.45:
        return "Partially", 0.5
    return "No", 0.0