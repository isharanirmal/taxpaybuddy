"""
Tests for the TaxPayBuddy evaluation framework.

Covers:
- models.py
- metrics.py
- llm_judge.py
- evaluators.py
- run_evaluation.py

All tests use mocks. No Gemini API or ChromaDB calls.
"""

import csv
import json

import pytest

from evaluation.models import EvaluationResult
from evaluation import metrics

from evaluation.llm_judge import LLMJudge
from evaluation.evaluators import (
    RoutingEvaluator,
    RetrievalEvaluator,
    AnswerEvaluator,
)

from evaluation.run_evaluation import (
    load_ground_truth,
    run_single,
    run_evaluation,
    write_results_csv,
    summarize,
    CSV_FIELDNAMES,
    DEFAULT_GROUND_TRUTH_PATH,
)

from src.agents.router_agent.router_main import RouterAgent
from src.framework.core.data_models import RAGResponse

from conftest import (
    MockLLMClient,
    make_chunk,
)


# metrics.py


def test_precision_at_k_counts_relevant_chunks():

    chunks = [
        make_chunk("TIN registration through IRD portal"),
        make_chunk("Random unrelated content"),
    ]

    score = metrics.precision_at_k(
        chunks,
        ["TIN"],
        2,
    )

    assert score == pytest.approx(0.5)



def test_precision_at_k_empty_values():

    assert metrics.precision_at_k(
        [],
        ["TIN"],
        3,
    ) == 0.0

    assert metrics.precision_at_k(
        [make_chunk("text")],
        [],
        3,
    ) == 0.0



def test_recall_at_k_finds_keywords():

    chunks = [
        make_chunk("TIN registration"),
        make_chunk("IRD portal"),
    ]

    score = metrics.recall_at_k(
        chunks,
        ["TIN", "IRD", "NIC"],
        2,
    )

    assert score == pytest.approx(2 / 3)



def test_recall_empty_keywords():

    assert metrics.recall_at_k(
        [make_chunk("text")],
        [],
        3,
    ) == 0.0



def test_keyword_coverage_full_match():

    score = metrics.keyword_coverage_score(
        "Apply TIN through IRD",
        ["TIN", "IRD"],
    )

    assert score == 1.0



def test_keyword_coverage_partial_case_insensitive():

    score = metrics.keyword_coverage_score(
        "apply tin through ird",
        ["TIN", "IRD", "NIC"],
    )

    assert score == pytest.approx(2 / 3)



def test_keyword_coverage_empty_answer():

    assert metrics.keyword_coverage_score(
        "",
        ["TIN"],
    ) == 0.0



def test_matched_keywords():

    result = metrics.matched_keywords(
        "TIN from IRD",
        ["TIN", "IRD", "NIC"],
    )

    assert result == ["TIN", "IRD"]



def test_confusion_counts():

    chunks = [
        make_chunk("TIN registration"),
        make_chunk("random"),
    ]

    result = metrics.confusion_counts(
        chunks,
        ["TIN"],
        2,
    )

    assert result["tp"] == 1
    assert result["fp"] == 1



def test_f1_score():

    score = metrics.f1_score(
        1.0,
        1.0,
    )

    assert score == 1.0



def test_cosine_similarity_same_text():

    score = metrics.cosine_similarity_score(
        "TIN registration",
        "TIN registration",
    )

    assert score == pytest.approx(
        1.0,
        abs=1e-6,
    )



def test_cosine_similarity_empty():

    assert metrics.cosine_similarity_score(
        "",
        "text",
    ) == 0.0




# models.py


def test_evaluation_result_csv_fields():

    result = EvaluationResult(
        question="q",
        expected_agent="agent1_tin_registration",
        predicted_agent="agent1_tin_registration",

        precision_at_1=1.0,
        precision_at_3=1.0,

        recall_at_1=1.0,
        recall_at_3=1.0,

        tp=1,
        fp=0,
        fn=0,

        f1_score=1.0,

        cosine_accuracy=1.0,

        keyword_score=1.0,

        matched_keywords=["TIN"],

        faithfulness=1.0,

        latency_seconds=0.1,

        generated_answer="answer",
    )


    row = result.to_row()

    assert set(row.keys()) == set(
        EvaluationResult.CSV_FIELDNAMES
    )

    assert row["error"] == ""


# llm_judge.py


def test_llm_judge_one_llm_call():

    llm = MockLLMClient(
        default_response='{"faithfulness":0.8}'
    )

    judge = LLMJudge(llm)


    score = judge.score_faithfulness(
        "What is TIN?",
        "Register through IRD",
        [
            make_chunk(
                "Register through IRD"
            )
        ],
    )


    assert score == 0.8

    assert len(llm.calls) == 1



def test_llm_judge_invalid_json_returns_zero():

    llm = MockLLMClient(
        default_response="invalid"
    )

    judge = LLMJudge(llm)


    score = judge.score_faithfulness(
        "q",
        "a",
        [],
    )


    assert score == 0.0
    assert len(llm.calls) == 1



def test_llm_judge_clamps_score():

    llm = MockLLMClient(
        default_response='{"faithfulness":5}'
    )

    judge = LLMJudge(llm)


    assert judge.score_faithfulness(
        "q",
        "a",
        [],
    ) == 1.0
# evaluators.py


def test_routing_evaluator():

    assert RoutingEvaluator.is_correct(
        "agent1_tin_registration",
        "agent1_tin_registration",
    )

    assert not RoutingEvaluator.is_correct(
        "agent1_tin_registration",
        "agent2_individual_income_tax",
    )



def test_retrieval_evaluator_returns_all_scores():

    response = RAGResponse(
        question="TIN question",

        retrieved_chunks=[
            make_chunk(
                "TIN registration through IRD"
            )
        ],

        answer="answer",
    )


    scores = RetrievalEvaluator().evaluate(
        response,
        ["TIN", "IRD"],
    )


    assert set(scores.keys()) == {
        "precision_at_1",
        "precision_at_3",
        "recall_at_1",
        "recall_at_3",
        "tp",
        "fp",
        "fn",
        "f1_score",
    }



def test_answer_evaluator_calls_llm_once():

    llm = MockLLMClient(
        default_response='{"faithfulness":0.7}'
    )

    evaluator = AnswerEvaluator(
        LLMJudge(llm)
    )


    response = RAGResponse(
        question="How to register TIN?",

        retrieved_chunks=[
            make_chunk(
                "Use IRD e-service portal"
            )
        ],

        answer="Use IRD e-service portal",
    )


    scores = evaluator.evaluate(
        response,
        ["TIN", "IRD"],
        "Use IRD e-service portal",
    )


    assert set(scores.keys()) == {
        "keyword_score",
        "matched_keywords",
        "cosine_accuracy",
        "faithfulness",
    }


    assert scores["faithfulness"] == 0.7

    assert len(llm.calls) == 1



# ground_truth loading


def test_load_ground_truth():

    data = load_ground_truth(
        DEFAULT_GROUND_TRUTH_PATH
    )


    assert len(data) > 0


    for item in data:

        assert {
            "id",
            "question",
            "expected_agent",
            "keywords",
        } <= item.keys()



def test_custom_ground_truth(tmp_path):

    data = [
        {
            "id": "Q1",
            "question": "test",
            "expected_agent":
                "agent1_tin_registration",
            "keywords":
                ["TIN"],
        }
    ]


    path = tmp_path / "gt.json"


    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )


    assert load_ground_truth(path) == data



# run_single / run_evaluation


@pytest.fixture
def router_with_mocks(
    populated_vector_store,
):

    llm = MockLLMClient(
        default_response=
        '{"next_agent":"agent1_tin_registration"}'
    )


    return RouterAgent(
        llm=llm,
        vector_store=populated_vector_store,
    )



@pytest.fixture
def answer_evaluator():

    llm = MockLLMClient(
        default_response=
        '{"faithfulness":0.8}'
    )


    return AnswerEvaluator(
        LLMJudge(llm)
    )



def test_run_single_correct_route(
    router_with_mocks,
    answer_evaluator,
):

    item = {

        "id":"Q1",

        "question":
            "How to register TIN?",

        "expected_agent":
            "agent1_tin_registration",

        "keywords":
            ["TIN","IRD"],

        "reference_answer":
            "Register using IRD portal",
    }


    row = run_single(
        router_with_mocks,
        item,
        answer_evaluator,
        RetrievalEvaluator(),
    )


    assert row["error"] == ""

    assert row["predicted_agent"] == (
        "agent1_tin_registration"
    )



def test_run_single_handles_error(
    populated_vector_store,
    answer_evaluator,
):


    class BrokenLLM(MockLLMClient):

        def generate(
            self,
            prompt,
            system_instruction="",
        ):

            raise RuntimeError(
                "API failed"
            )


    router = RouterAgent(
        llm=BrokenLLM(),
        vector_store=populated_vector_store,
    )


    item = {

        "question":
            "unknown question",

        "expected_agent":
            "general_fallback",

        "keywords":
            [],
    }


    row = run_single(
        router,
        item,
        answer_evaluator,
        RetrievalEvaluator(),
    )


    

    assert row["error"] == ""

    



def test_run_evaluation_creates_csv(
    router_with_mocks,
    tmp_path,
):

    gt = tmp_path / "gt.json"


    gt.write_text(
        json.dumps(
            [
                {
                    "id":"Q1",
                    "question":
                        "TIN registration",

                    "expected_agent":
                        "agent1_tin_registration",

                    "keywords":
                        ["TIN"],

                    "reference_answer":
                        "TIN answer",
                }
            ]
        ),
        encoding="utf-8",
    )


    output = tmp_path / "result.csv"


    judge_llm = MockLLMClient(
        default_response=
        '{"faithfulness":0.9}'
    )


    summary = run_evaluation(
        ground_truth_path=gt,
        output_path=output,
        router=router_with_mocks,
        judge_llm=judge_llm,
    )


    assert output.exists()

    assert summary["total_questions"] == 1



def test_summarize():

    rows = [

        {
            "expected_agent":"agent1",
            "predicted_agent":"agent1",

            "precision_at_1":1,
            "precision_at_3":1,

            "recall_at_1":1,
            "recall_at_3":1,

            "f1_score":1,

            "cosine_accuracy":1,

            "keyword_score":1,

            "faithfulness":1,

            "latency_seconds":1,

            "error":"",
        },

        {
            "expected_agent":"agent1",
            "predicted_agent":"agent2",

            "precision_at_1":0,
            "precision_at_3":0,

            "recall_at_1":0,
            "recall_at_3":0,

            "f1_score":0,

            "cosine_accuracy":0,

            "keyword_score":0,

            "faithfulness":0,

            "latency_seconds":1,

            "error":"",
        }

    ]


    result = summarize(rows)


    assert result["total_questions"] == 2

    assert result["successful_runs"] == 2

    assert result["routing_accuracy"] == 0.5



def test_write_results_csv(
    tmp_path,
):

    path = tmp_path / "results.csv"


    rows = [

        EvaluationResult(
            question="q",

            expected_agent="a",
            predicted_agent="a",

            precision_at_1=1,
            precision_at_3=1,

            recall_at_1=1,
            recall_at_3=1,

            tp=1,
            fp=0,
            fn=0,

            f1_score=1,

            cosine_accuracy=1,

            keyword_score=1,

            matched_keywords=["TIN"],

            faithfulness=1,

            latency_seconds=0.1,

            generated_answer="answer",

        ).to_row()

    ]


    write_results_csv(
        rows,
        path,
    )


    assert path.exists()


    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        assert reader.fieldnames == CSV_FIELDNAMES