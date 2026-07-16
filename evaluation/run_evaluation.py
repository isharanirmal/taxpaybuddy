"""
Main evaluation program for TaxPayBuddy.

Runs the complete RAG pipeline against the ground truth
dataset and records retrieval, answer quality and
performance metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import time

from pathlib import Path
from typing import List, Optional

from src.agents.router_agent.router_main import RouterAgent
from src.framework.database.chroma_store import ChromaStore
from src.framework.llm.gemini_client import GeminiClient

from evaluation.models import EvaluationResult
from evaluation.evaluators import (
    RetrievalEvaluator,
    AnswerEvaluator,
)
from evaluation.llm_judge import LLMJudge
from evaluation.metrics import agent_routing_confusion_matrix


DEFAULT_GROUND_TRUTH_PATH = (
    Path(__file__).parent / "ground_truth.json"
)

DEFAULT_OUTPUT_PATH = (
    Path(__file__).parent / "results.csv"
)

CSV_FIELDNAMES = EvaluationResult.CSV_FIELDNAMES


def load_ground_truth(
    path: Path = DEFAULT_GROUND_TRUTH_PATH,
) -> List[dict]:
    """
    Load evaluation dataset.
    """

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _predicted_agent(
    router: RouterAgent,
) -> str:
    """
    Returns the routing label selected by RouterAgent.
    """

    return getattr(
        router,
        "last_routed_label",
        None,
    ) or "unknown"


def run_single(
    router: RouterAgent,
    item: dict,
    answer_evaluator: AnswerEvaluator,
    retrieval_evaluator: RetrievalEvaluator,
) -> dict:
    """
    Runs one evaluation question through the
    TaxPayBuddy pipeline.
    """

    question = item["question"]

    expected_agent = item["expected_agent"]

    keywords = item.get(
        "keywords",
        [],
    )

    reference_answer = item.get(
        "reference_answer",
        "",
    )

    result_kwargs = {

        "question": question,

        "expected_agent": expected_agent,

        "predicted_agent": "",

        "precision_at_1": 0.0,
        "precision_at_3": 0.0,

        "recall_at_1": 0.0,
        "recall_at_3": 0.0,

        "tp": 0,
        "fp": 0,
        "fn": 0,

        "f1_score": 0.0,

        "cosine_accuracy": 0.0,

        "keyword_score": 0.0,

        "matched_keywords": [],

        "faithfulness": 0.0,

        "latency_seconds": 0.0,

        "generated_answer": "",

        "error": "",
    }

    start = time.perf_counter()

    try:

        response = router.route_and_execute(
            question
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        retrieval_scores = (
            retrieval_evaluator.evaluate(
                response,
                keywords,
            )
        )

        answer_scores = (
            answer_evaluator.evaluate(
                response,
                keywords,
                reference_answer,
            )
        )

        result_kwargs.update(

            predicted_agent=_predicted_agent(
                router,
            ),

            latency_seconds=elapsed,

            generated_answer=response.answer,

            **retrieval_scores,

            **answer_scores,
        )

    except Exception as exc:

        result_kwargs["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

        result_kwargs[
            "latency_seconds"
        ] = (
            time.perf_counter()
            - start
        )

    return EvaluationResult(
        **result_kwargs
    ).to_row()
def write_results_csv(
    rows: List[dict],
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """
    Writes all evaluation results to a CSV file.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDNAMES,
        )

        writer.writeheader()

        writer.writerows(rows)


def summarize(
    rows: List[dict],
) -> dict:
    """
    Calculates overall evaluation statistics.
    """

    total_questions = len(rows)

    successful_rows = [
        row
        for row in rows
        if not row.get("error")
    ]

    errors = total_questions - len(successful_rows)

    def average(metric: str) -> float:

        values = [
            row[metric]
            for row in successful_rows
        ]

        if not values:
            return 0.0

        return sum(values) / len(values)

    routing_correct = sum(
        1
        for row in successful_rows
        if row["predicted_agent"] == row["expected_agent"]
    )

    agent_labels = sorted(
        {row["expected_agent"] for row in rows}
    )

    routing_confusion_matrix = agent_routing_confusion_matrix(
        successful_rows,
        agent_labels,
    )

    return {

        "total_questions": total_questions,

        "successful_runs": len(successful_rows),

        "errors": errors,

        "routing_accuracy":
            routing_correct / len(successful_rows)
            if successful_rows
            else 0.0,

        "routing_confusion_matrix": routing_confusion_matrix,

        "avg_precision_at_1":
            average("precision_at_1"),

        "avg_precision_at_3":
            average("precision_at_3"),

        "avg_recall_at_1":
            average("recall_at_1"),

        "avg_recall_at_3":
            average("recall_at_3"),

        "avg_f1_score":
            average("f1_score"),

        "avg_cosine_accuracy":
            average("cosine_accuracy"),

        "avg_keyword_score":
            average("keyword_score"),

        "avg_faithfulness":
            average("faithfulness"),

        "avg_latency_seconds":
            average("latency_seconds"),
    }
def run_evaluation(
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    limit: Optional[int] = None,
    router: Optional[RouterAgent] = None,
    judge_llm=None,
) -> dict:
    """
    Executes the complete evaluation pipeline.
    """

    items = load_ground_truth(
        ground_truth_path
    )

    if limit is not None:
        items = items[:limit]

    if router is None:

        llm = GeminiClient()

        vector_store = ChromaStore()

        router = RouterAgent(
            llm=llm,
            vector_store=vector_store,
        )

    judge = LLMJudge(
        judge_llm or router.llm_client
    )

    retrieval_evaluator = RetrievalEvaluator()

    answer_evaluator = AnswerEvaluator(
        judge
    )

    rows = []

    print("=" * 70)
    print("TaxPayBuddy Evaluation")
    print("=" * 70)

    for item in items:

        row = run_single(
            router,
            item,
            answer_evaluator,
            retrieval_evaluator,
        )

        rows.append(row)

        print(
            f"[{item.get('id', '-')}] "
            f"Agent={row['predicted_agent']} | "
            f"Precision@3={row['precision_at_3']:.2f} | "
            f"Recall@3={row['recall_at_3']:.2f} | "
            f"F1={row['f1_score']:.2f} | "
            f"Faithfulness={row['faithfulness']:.2f}"
            + (
                f" | ERROR: {row['error']}"
                if row["error"]
                else ""
            )
        )

    write_results_csv(
        rows,
        output_path,
    )

    return summarize(rows)


def main():

    parser = argparse.ArgumentParser(
        description="TaxPayBuddy Evaluation"
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    summary = run_evaluation(
        ground_truth_path=args.ground_truth,
        output_path=args.output,
        limit=args.limit,
    )

    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)

    for key, value in summary.items():

        if key == "routing_confusion_matrix":
            continue

        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    print("\n" + "-" * 70)
    print("Routing Confusion Matrix (per agent, one-vs-rest)")
    print("-" * 70)
    print(f"{'Agent':<32}{'TP':>6}{'FP':>6}{'FN':>6}{'TN':>6}")

    for agent, counts in summary["routing_confusion_matrix"].items():
        print(
            f"{agent:<32}"
            f"{counts['tp']:>6}"
            f"{counts['fp']:>6}"
            f"{counts['fn']:>6}"
            f"{counts['tn']:>6}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()