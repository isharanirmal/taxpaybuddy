"""
Evaluator classes: turn raw pipeline output (a RAGResponse + timing)
into scored fields for an EvaluationResult. Each class owns exactly one
concern, so run_evaluation.py stays a thin orchestrator instead of a
wall of scoring logic.
"""
from __future__ import annotations

from typing import Sequence

from src.framework.core.data_models import RAGResponse

from evaluation import metrics
from evaluation.llm_judge import LLMJudge


class RoutingEvaluator:
    """Was the query sent to the specialist agent the ground truth expects?"""

    @staticmethod
    def is_correct(expected_agent: str, predicted_agent: str) -> bool:
        return expected_agent == predicted_agent


class RetrievalEvaluator:

    def evaluate(
        self,
        response: RAGResponse,
        keywords: Sequence[str],
    ) -> dict:

        chunks = response.retrieved_chunks

        precision_1 = metrics.precision_at_k(chunks, keywords, 1)
        precision_3 = metrics.precision_at_k(chunks, keywords, 3)

        recall_1 = metrics.recall_at_k(chunks, keywords, 1)
        recall_3 = metrics.recall_at_k(chunks, keywords, 3)

        counts = metrics.confusion_counts(
            chunks,
            keywords,
            3,
        )

        return {
            "precision_at_1": precision_1,
            "precision_at_3": precision_3,

            "recall_at_1": recall_1,
            "recall_at_3": recall_3,

            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],

            "f1_score": metrics.f1_score(
                precision_3,
                recall_3,
            ),
        }


class AnswerEvaluator:
    """
    Keyword coverage + semantic (cosine) similarity are both local and
    free. Faithfulness is the ONE Gemini call per question, delegated
    to LLMJudge.
    """

    def __init__(self, judge: LLMJudge):
        self.judge = judge

    def evaluate(self, response: RAGResponse, keywords: Sequence[str], reference_answer: str) -> dict:
        answer = response.answer

        keyword_score = metrics.keyword_coverage_score(answer, keywords)
        cosine_accuracy = metrics.cosine_similarity_score(answer, reference_answer)

        faithfulness = self.judge.score_faithfulness(
            question=response.question,
            answer=answer,
            context_chunks=response.retrieved_chunks,
        )

        return {
    "keyword_score": keyword_score,
    "matched_keywords": metrics.matched_keywords(answer,keywords),
    "cosine_accuracy": cosine_accuracy,
    "faithfulness": faithfulness,
    }
