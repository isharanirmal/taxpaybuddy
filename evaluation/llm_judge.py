"""
LLM-as-judge faithfulness scoring.

HARD CONSTRAINT: exactly ONE call to the LLM client per question. The
judge packs everything it needs to score into a single prompt and asks
for a single JSON object back, instead of making separate calls for
separate sub-scores. This keeps the evaluation run's own API usage
predictable and avoids tripping Gemini's free-tier rate limit (see the
429 retry/backoff logic in GeminiClient) on top of whatever calls the
router/agent already made to answer the question.
"""
from __future__ import annotations

import json
from typing import Sequence

from src.framework.interfaces.interfaces import ILLMClient
from src.framework.core.data_models import DocumentChunk


JUDGE_SYSTEM_PROMPT = """
You are a strict faithfulness judge for a Sri Lankan tax Q&A system.

You will be given a QUESTION, the retrieved CONTEXT passages, and the
ANSWER the system produced.

Score how faithful the ANSWER is to the CONTEXT, from 0.0 to 1.0:
- 1.0 = every claim in the answer is directly supported by the context
- 0.5 = partially supported, or adds plausible but unverifiable detail
- 0.0 = contradicts the context, or is unrelated to it

Return ONLY this JSON object and nothing else -- no markdown, no
explanation:
{"faithfulness": <float between 0.0 and 1.0>}
"""


class LLMJudge:
    """Wraps a single ILLMClient call to score faithfulness."""

    def __init__(self, llm: ILLMClient):
        self.llm = llm

    def score_faithfulness(
        self,
        question: str,
        answer: str,
        context_chunks: Sequence[DocumentChunk],
    ) -> float:
        """
        Makes exactly ONE self.llm.generate() call. Returns 0.0 (rather
        than raising) if the call fails or the response can't be
        parsed, so a flaky judge never crashes the whole eval run.
        """
        context = "\n\n".join(chunk.text for chunk in context_chunks) or "(no context retrieved)"
        prompt = f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"

        try:
            raw = self.llm.generate(prompt=prompt, system_instruction=JUDGE_SYSTEM_PROMPT)

            cleaned = raw.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

            data = json.loads(cleaned)
            score = float(data.get("faithfulness", 0.0))
            return max(0.0, min(1.0, score))

        except Exception:
            return 0.0
