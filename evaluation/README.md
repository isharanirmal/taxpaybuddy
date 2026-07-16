# TaxPayBuddy - Empirical Evaluation

Runs the 10 questions in `ground_truth.json` through the real
RouterAgent -> specialist agent -> RAG pipeline (real Gemini + real
ChromaDB), scores each answer, and writes `results.csv`. That CSV is
the raw data behind Report Section IV (Empirical Evaluation).

## Requirements

- `GEMINI_API_KEY` set in `.env` (already present in this project)
- A populated `data/chroma_db` (run each agent's
  `build_knowledge_base.py` first if it's empty)
- `pip install -r requirements.txt`

## Running

From the project root (`TaxPayBuddy/`):

```bash
python -m evaluation.run_evaluation
```

Optional flags:

```bash
python -m evaluation.run_evaluation --limit 3                      # quick smoke run, first 3 questions only
python -m evaluation.run_evaluation --output evaluation/results_v2.csv
python -m evaluation.run_evaluation --ground-truth evaluation/my_questions.json
```

## Output

`evaluation/results.csv` with one row per question:

| Column | Meaning |
|---|---|
| `expected_agent` / `predicted_agent` | Which specialist agent *should* have answered vs. which one the router actually picked |
| `routing_correct` | Whether the router's dispatch matched `expected_agent` |
| `keyword_score` | Fraction of the question's expected `keywords` found in the generated answer (0.0-1.0) |
| `matched_keywords` | Which specific keywords were found |
| `num_chunks_retrieved` | How many chunks the RAG retriever returned |
| `latency_seconds` | Wall-clock time for routing + retrieval + generation |
| `answer` | The full generated answer |
| `error` | Populated instead of a score if that question's run raised an exception |

A summary (routing accuracy, average keyword score, average latency)
is printed to the console after the run and is a good source for
Report Section IV's headline numbers.

## Note on `ground_truth.json`

The `reference_answer` / `keywords` fields were authored as a
lightweight, reproducible scoring rubric for this evaluation harness,
not copied from an official IRD publication. Before quoting specific
rates or figures in the report, cross-check against the source PDFs in
`data/raw_pdfs/`.
